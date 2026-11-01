"""Independent four-system confirmation, with no old-development evaluation path."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import traceback

import numpy as np

import step28_continual_comparison as m
import step28_continual_comparison_run as run
import step28_continual_five_evaluate as prior

data, generator, validator = m.data, prior.generator, prior.validator
NAMESPACE = 'continual_comparison_confirmation_20261024'
CONFIRM_SEED, BOOTSTRAP_SEED = 2026102401, 2026102402
CELLS = ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
FIELDS = ('O', 'F', 'A_acq', 'N', 'Z', 'F_first', 'final_mean')
PRIMARY = (('H_pos_tuned', 'L2R_text_fixed'), ('H_pos_tuned', 'ER_GCR_group_fixed'))
SECONDARY = (('H_pos_tuned', 'S_tuned'), ('S_tuned', 'L2R_text_fixed'), ('S_tuned', 'ER_GCR_group_fixed'))


def trajectory_sources(allowed: dict, identity: str, completion: Path) -> list:
    recovery = allowed.get('recovery')
    if not recovery or identity not in recovery['reused']:
        return allowed['sources']
    data.verify(completion, recovery['reused'][identity])
    pinned = recovery['original_gate']
    return data.read_json(data.verify(Path(pinned['path']), pinned))['sources']


def full_gate(campaign: Path, allowed: dict) -> list[dict]:
    """Check all 108 actual endpoints; H/S are pinned existing runs, never substitutes."""
    old_gate_path = Path(allowed['reference_gate']['path'])
    old_gate = data.read_json(data.verify(old_gate_path, allowed['reference_gate']))
    if old_gate['status'] != 'APPROVED_IMPLEMENTATION_AND_RESOURCES':
        raise ValueError('Pinned H/S source gate missing')
    reference = Path(allowed['reference_campaign'])
    records, partition, archive = [], None, None
    for spec in m.specifications():
        new = spec.name in m.NEW_METHODS
        root = (campaign if new else reference)/spec.run_id/'run'
        row = data.read_json(root/'completion.json')
        expected_spec = spec.identity() if new else m.five.Spec(spec.name, spec.order, spec.seed).identity()
        expected_status = run.COMPLETE if new else prior.run.COMPLETE
        if (row['status'] != expected_status or row['spec'] != expected_spec or row['updates'] != 864
                or row['source_files'] != (trajectory_sources(allowed, spec.run_id, root/'completion.json') if new else old_gate['sources'])
                or set(row['points']) != {'1', '2', '3'}
                or row['access'] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
            raise ValueError('Incomplete/foreign comparison run; missing H/S never triggers retraining')
        shared = {k: row['partition'][k] for k in ('fit', 'calibration')}
        if partition is None:
            partition, archive = shared, row['pretrained_archive']
        if shared != partition or row['pretrained_archive'] != archive:
            raise ValueError('Training/calibration partition or pretrained source differs')
        for number in (1, 2, 3):
            point = row['points'][str(number)]
            expected_ids = [r['group_uid'] for r in shared['calibration'] if r['domain'] == spec.order[number-1]]
            method = m.inference_method(spec)
            if (point['stage'] != number or point['adam_step'] != 288*number
                    or point['calibration_ids'] != expected_ids or len(expected_ids) != 12
                    or not point['full_restore_verified'] or not point['inference_restore_verified']
                    or any(point.get(k) != v for k, v in method.metadata().items())):
                raise ValueError('Endpoint identity/restoration/current calibration differs')
            data.verify(root/point['model']['path'], point['model'])
            fitted = data.read_json(data.verify(root/point['mapping']['path'], point['mapping']))
            mapping = dict(zip(('a', 'b'), m.parent.calibration.parameters(fitted)))
            if (fitted['status'] != 'PASS_CALIBRATION_FIT' or mapping != point['maps']['stage']
                    or point['maps']['first'] != row['points']['1']['maps']['stage']):
                raise ValueError('Frozen calibration mapping differs')
        records.append(dict(spec=spec.identity(), root=str(root), record=row,
                            completion=data.record(root/'completion.json', root)))
    return records


def generate(campaign: Path, job: Path, records: list, budget) -> None:
    old_policy, catalog = data.read_json(generator.POLICY), data.read_json(generator.CATALOG)
    generator.check_contract(old_policy, catalog)
    if CONFIRM_SEED in (old_policy['root_seed'], prior.CONFIRM_SEED) or NAMESPACE in (*generator.SPLITS, prior.NAMESPACE):
        raise ValueError('Confirmation random stream/namespace reused')
    policy = {**old_policy, 'root_seed': CONFIRM_SEED}
    checker = validator.TextChecker(catalog)
    public_path, owner_path, pair_path = (job/name for name in ('groups.jsonl', 'owners.jsonl', 'pairs.jsonl'))
    prior.access(campaign, 'comparison_generation', ('new_generated_owners_and_pairs',))
    old_ids = {r['group_uid'] for item in records for role in item['record']['partition'].values() for r in role}
    groups, sellers, items_seen, controllers = set(), set(), set(), set()
    with public_path.open('x', encoding='utf-8') as public, owner_path.open('x', encoding='utf-8') as owners, pair_path.open('x', encoding='utf-8') as pairs:
        for domain in 'ABC':
            for index in range(40):
                budget.check()
                meta, items, identities = generator.make_group(policy, catalog, domain, NAMESPACE, index)
                pair_rows = list(generator.pair_rows(identities))
                checked = validator.check_group(items, identities, pair_rows, checker)
                expected_group = generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'group')
                expected_accounts = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'account', i) for i in range(28)}
                expected_controllers = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'controller', i) for i in range(12)}
                actual_controllers = {r['controller_uid'] for r in identities}
                if (meta['group_uid'] != expected_group or expected_group in groups | old_ids
                        or checked['accounts'] != expected_accounts or actual_controllers != expected_controllers
                        or sellers & expected_accounts or controllers & actual_controllers
                        or items_seen & checked['item_ids'] or meta['split'] != NAMESPACE):
                    raise ValueError('New controller/account/group random namespaces overlap')
                groups.add(expected_group)
                sellers |= expected_accounts
                controllers |= actual_controllers
                items_seen |= checked['item_ids']
                public.write(generator.json_line(dict(metadata=meta, items=items)))
                owners.write(generator.json_line(identities))
                pairs.write(generator.json_line(pair_rows))
    if (len(groups), len(sellers), len(controllers)) != (120, 3360, 1440):
        raise ValueError('Incomplete independent confirmation generation')
    files = {p.name: data.record(p, job) for p in (public_path, owner_path, pair_path)}
    owner_path.chmod(0)
    data.write_json(job/'result.json', dict(status='GENERATED_VALIDATED_OWNERS_SEALED', root_seed=CONFIRM_SEED,
        namespace=NAMESPACE, groups=120, files=files, sources=run.sources(),
        isolation='Pinned generator: new root and namespace enter all controller/account IDs and preference/style RNG; verified new actual IDs and old group exclusion. No old owners read or reconstructed.'))


def confirmation_public(job: Path) -> tuple:
    manifest = data.read_json(job/'result.json')
    if (manifest['namespace'] != NAMESPACE or manifest['root_seed'] != CONFIRM_SEED
            or manifest['sources'] != run.sources()):
        raise ValueError('Independent confirmation source differs')
    return prior.confirmation_public(job)


def calibrated_roles(raw: np.ndarray, maps: dict) -> tuple:
    if raw.shape != (40, 378) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Complete FP32 confirmation cell required')
    roles, evidence = {'raw': raw}, {}
    for role, key in (('stage-cal', 'stage'), ('first-cal', 'first')):
        roles[role] = m.parent.calibration.transform(raw, maps[key])
        evidence[role] = m.parent.calibration.preserve_order(raw, roles[role])
    return roles, evidence


def confirmation(campaign: Path, job: Path, records: list, budget, allowed: dict) -> None:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(allowed['limits']['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
    groups, domains = confirmation_public(campaign/'generation')
    by_domain = {d: [g for g, gd in zip(groups, domains, strict=True) if gd == d] for d in 'ABC'}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0)
    prior.access(campaign, 'comparison_blind_inference', ('new_public_text', '108_existing_inference_states'))
    for item in records:
        spec = m.Spec.from_dict(item['spec'])
        method, c, root = m.inference_method(spec), m.config(spec), Path(item['root'])
        model, optimizer = method.fresh(c)
        del optimizer
        result['runs'][spec.run_id] = {}
        for number in (1, 2, 3):
            budget.check()
            point = item['record']['points'][str(number)]
            rec = point['model']
            actual = method.restore(data.verify(root/rec['path'], rec), rec, model, None)
            if actual['stage'] != number or actual['adam_step'] != 288*number or actual['maps'] != point['maps']:
                raise ValueError('Actual loaded model/calibration identity differs')
            for _, position in [cell for cell in CELLS if cell[0] == number]:
                domain, name = spec.order[position-1], f'{number}_{position}'
                with m.diag.neutral(model):
                    raw = method.score(model, by_domain[domain], c, budget.check)
                roles, preserved = calibrated_roles(raw, point['maps'])
                result['runs'][spec.run_id][name] = dict(domain=domain, ids=[g.uid for g in by_domain[domain]],
                    preserved=preserved, scores={role: run.save_array(job/f'{spec.run_id}_{name}_{role}.npy', values, job)
                                                for role, values in roles.items()})
                result['forwards'] += 40
        del model
        gc.collect()
        torch.cuda.empty_cache()
        data.write_json(job/'blind_progress.json', result)
    validate_cells(result)
    # Validate every saved blind array before the one joint label opening.
    for spec in m.specifications():
        for point in result['runs'][spec.run_id].values():
            raw = np.load(data.verify(job/point['scores']['raw']['path'], point['scores']['raw']), allow_pickle=False)
            for role in ('stage-cal', 'first-cal'):
                rec = point['scores'][role]
                m.parent.calibration.preserve_order(raw, np.load(data.verify(job/rec['path'], rec), allow_pickle=False))
    data.write_json(job/'all_blind_complete.json', {**result, 'sources': run.sources()})
    prior.access(campaign, 'comparison_confirmation_joint', ('new_pairs',))
    generated = data.read_json(campaign/'generation/result.json')
    truth = {}
    with data.verify(campaign/'generation/pairs.jsonl', generated['files']['pairs.jsonl']).open(encoding='utf-8') as stream:
        for group, line in zip(groups, stream, strict=True):
            rows = [dict(row, label=str(row['label'])) for row in json.loads(line)]
            labels = data.align_labels(group, rows)
            if sum(labels) != 20:
                raise ValueError('Confirmation positive count differs')
            truth[group.uid] = labels
    for spec in m.specifications():
        for cell, point in result['runs'][spec.run_id].items():
            labels = np.asarray([truth[uid] for uid in point['ids']], np.uint8)
            point['metrics'] = {}
            for role, rec in point['scores'].items():
                budget.check()
                values = np.load(data.verify(job/rec['path'], rec), allow_pickle=False)
                point['metrics'][role] = prior.metric_set(job/f'metrics_{spec.run_id}_{cell}_{role}', labels, values)
    data.write_json(job/'result.json', {**result, 'status': 'ALL_COMPARISON_METRICS_COMPLETE', 'sources': run.sources()})


def validate_cells(result: dict) -> None:
    if (result['forwards'] != 8640 or len(result['groups']) != 120 or len(set(result['groups'])) != 120
            or result['domains'] != [d for d in 'ABC' for _ in range(40)]
            or set(result['runs']) != {s.run_id for s in m.specifications()}):
        raise ValueError('All 36 runs / 216 cells / 8640 group forwards must complete')
    for spec in m.specifications():
        if set(result['runs'][spec.run_id]) != {f'{i}_{j}' for i, j in CELLS}:
            raise ValueError('Missing/unplanned comparison cell')
        for stage, position in CELLS:
            cell = result['runs'][spec.run_id][f'{stage}_{position}']
            domain = spec.order[position-1]
            ids = [uid for uid, d in zip(result['groups'], result['domains'], strict=True) if d == domain]
            if cell['domain'] != domain or cell['ids'] != ids or set(cell['scores']) != set(m.parent.ROLES):
                raise ValueError('Actual-domain paired group order differs')


def contributions(spec: m.Spec, cells: dict, field: str) -> np.ndarray:
    terms = ((3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3)) if field == 'final_mean' else prior.field_terms(field)
    out = np.zeros((3, 40, 22), np.float64)
    for stage, position, coefficient in terms:
        out['ABC'.index(spec.order[position-1])] += coefficient*cells[f'{stage}_{position}']
    return out


def interval(values: np.ndarray, draws: np.ndarray, quantiles=(.025, .975), check=lambda: None) -> dict:
    estimate = values.mean(1).sum(0)
    samples = np.empty((len(draws), 22), np.float64)
    for start in range(0, len(draws), 100):
        check()
        block = draws[start:start+100]
        samples[start:start+100] = sum(values[d][block[:, d]].mean(1) for d in range(3))
    return {name: dict(mean=None if np.isnan(values[:, :, j]).any() else float(estimate[j]),
                      conditional_interval=None if np.isnan(values[:, :, j]).any() else np.quantile(samples[:, j], quantiles).tolist(),
                      quantiles=list(quantiles)) for j, name in enumerate(m.parent.metrics.COLUMNS)}


def primary_decision(comparisons: dict) -> dict:
    checks = {}
    for a, b in PRIMARY:
        values = comparisons[a+'-'+b]['raw']
        old, forgetting, new, last = [values[field]['map'] for field in ('O', 'F', 'N', 'Z')]
        checks[a+'-'+b] = dict(old_retention=old['conditional_interval'][0] > 0,
            forgetting=forgetting['conditional_interval'][1] < 0,
            acquisition_point=new['mean'] >= 0, final_domain_point=last['mean'] >= 0)
    return dict(passed=all(v for row in checks.values() for v in row.values()), checks=checks)


def analyze(campaign: Path, job: Path, budget) -> None:
    saved = data.read_json(campaign/'confirmation/result.json')
    if saved['status'] != 'ALL_COMPARISON_METRICS_COMPLETE' or saved['sources'] != run.sources():
        raise ValueError('Complete same-source comparison metrics required')
    validate_cells(saved)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(40, size=(5000, 3, 40))
    by_run, summaries, cell_points = {}, {}, {}
    for spec in m.specifications():
        by_run[spec.run_id] = {}
        for role in m.parent.ROLES:
            cells = {}
            for cell, point in saved['runs'][spec.run_id].items():
                rec = point['metrics'][role]['matrix']
                record = data.read_json(data.verify(campaign/'confirmation'/rec['path'], rec))
                values = np.asarray(record['values'], np.float64)
                if record['columns'] != list(m.parent.metrics.COLUMNS) or values.shape != (40, 22):
                    raise ValueError('Saved 22-metric schema differs')
                cells[cell] = values
                cell_points[f'{spec.run_id}_{role}_{cell}'] = dict(domain=point['domain'],
                    values=prior.nullable(values.mean(0)), counts=record['counts'])
            by_run[spec.run_id][role] = {field: contributions(spec, cells, field) for field in FIELDS}
            summaries[spec.run_id+'_'+role] = {field: prior.nullable(v.mean(1).sum(0))
                                             for field, v in by_run[spec.run_id][role].items()}
    averaged = {name: {role: {field: np.mean([by_run[s.run_id][role][field] for s in m.specifications() if s.name == name], axis=0)
                              for field in FIELDS} for role in m.parent.ROLES} for name in m.METHODS}
    comparisons = {}
    for a, b in (*PRIMARY, *SECONDARY):
        comparisons[a+'-'+b] = {role: {field: interval(averaged[a][role][field]-averaged[b][role][field], draws,
                (.0125, .9875) if (a, b) in PRIMARY and field in ('O', 'F') else (.025, .975),
                budget.check) for field in FIELDS}
                               for role in m.parent.ROLES}
    for roles in averaged.values():
        for fields in roles.values():
            left, right = fields['F'], fields['A_acq']-fields['O']
            finite = np.isfinite(left) & np.isfinite(right)
            np.testing.assert_allclose(left[finite], right[finite], atol=1e-10, rtol=1e-10)
    data.write_json(job/'result.json', dict(status='FIXED_COMPARISON_ANALYSIS_COMPLETE',
        primary=primary_decision(comparisons), comparisons=comparisons,
        columns=list(m.parent.metrics.COLUMNS), per_run_endpoint_points=summaries,
        per_run_actual_domain_cell_points=cell_points,
        absolute_endpoint_points={name: {role: {field: prior.nullable(v.mean(1).sum(0)) for field, v in fields.items()}
                                         for role, fields in roles.items()} for name, roles in averaged.items()},
        bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=5000, actual_domains='ABC', groups_per_domain=40,
                       shared_across_all_cells_methods_orders_seeds=True, training_runs_fixed_equal_average=9),
        limits='Fixed task adaptations, H/S each had 12 HPO trials and new methods zero; unequal architectures/losses/exposure/candidate access. Conditional group uncertainty, not retraining/general-market uncertainty. N/Z are aggregate point guards only.',
        sign_convention='Literal R formulas; Brier/log_loss are lower-better, so positive acquisition-minus-final loss is improvement. Only raw MAP defines success. stage-cal probability metrics primary, first-cal diagnostic.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'confirmation', 'analyze'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    allowed = run.gate(args.gate, args.action)
    if args.action in ('generate', 'confirmation') and allowed.get('new_confirmation_data') is not True:
        raise ValueError('New confirmation access not authorized')
    records = full_gate(args.campaign, allowed)
    job = args.campaign/({'generate': 'generation', 'confirmation': 'confirmation', 'analyze': 'analysis'}[args.action])
    job.mkdir()  # A failed attempt cannot silently retry under the same action.
    budget = run.Budget(args.campaign, args.deadline, allowed['limits'])
    try:
        if args.action == 'generate':
            generate(args.campaign, job, records, budget)
        elif args.action == 'confirmation':
            confirmation(args.campaign, job, records, budget, allowed)
        else:
            analyze(args.campaign, job, budget)
        data.write_json(job/'resource_receipt.json', budget.summary())
    except BaseException as error:
        data.write_json(job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), status='STOP_NO_RETRY'))
        raise
