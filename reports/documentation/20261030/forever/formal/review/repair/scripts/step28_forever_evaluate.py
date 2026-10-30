"""Independent 120-group confirmation of frozen H and FOREVER endpoints."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import traceback

import numpy as np

import step28_forever_formal as formal
import step28_continual_comparison_evaluate as shared
import step28_continual_five_evaluate as prior
from step28_bge_continual_run import save_array

queue, m, data = formal.queue, formal.m, formal.data
NAMESPACE = 'forever_confirmation_20261030'
CONFIRM_SEED, BOOTSTRAP_SEED = 2026103001, 2026103002
NAMES = ('H_pos_tuned', m.NAME)
CELLS, FIELDS = shared.CELLS, shared.FIELDS


def run_id(name, order, seed):
    return f'{order}_s{seed}_{name}'


def complete_training(campaign, allowed):
    batch = data.read_json(campaign/'training_complete.json')
    if (batch['status'] != 'ALL_TRAINING_COMPLETE_EVALUATION_CLOSED'
            or len(batch['completed']) != 9 or not all(r['success'] for r in batch['completed'])):
        raise ValueError('All nine FOREVER trajectories must complete before confirmation')
    formal.verify_reference(allowed)
    references = {}
    for item in allowed['h_reference_runs']:
        path = queue.verify(item['completion'])
        row = data.read_json(path)
        references[(row['spec']['order'], row['spec']['seed'])] = (path.parent, row)
    records = []
    for spec in formal.specs():
        root = campaign/spec.run_id/'run'
        row = formal.validate_completion(spec, root, allowed)
        old_root, old = references[(spec.order, spec.seed)]
        # Old H also recorded development IDs; this train-only run deliberately
        # has no development partition. Compare the two learning roles only.
        if (any(row['partition'][role] != old['partition'][role] for role in ('fit', 'calibration'))
                or row['pretrained_archive'] != old['pretrained_archive']):
            raise ValueError('H/FOREVER actual fit/calibration partition or pretrained archive differs')
        for name, path, item in ((m.NAME, root, row), ('H_pos_tuned', old_root, old)):
            records.append(dict(name=name, order=spec.order, seed=spec.seed, root=str(path), record=item))
    return records


def generate(campaign, job, records, budget, allowed):
    generator, validator = shared.generator, shared.validator
    policy, catalog = data.read_json(generator.POLICY), data.read_json(generator.CATALOG)
    generator.check_contract(policy, catalog)
    if CONFIRM_SEED in (policy['root_seed'], prior.CONFIRM_SEED, shared.CONFIRM_SEED):
        raise ValueError('New confirmation seed collides with a previous batch')
    policy = {**policy, 'root_seed': CONFIRM_SEED}
    checker = validator.TextChecker(catalog)
    old = {r['group_uid'] for item in records for rows in item['record']['partition'].values() for r in rows}
    seen, accounts, controllers, items_seen = set(), set(), set(), set()
    paths = [job/name for name in ('groups.jsonl', 'owners.jsonl', 'pairs.jsonl')]
    prior.access(campaign, 'forever_generation', ('new_generated_owners_and_pairs',))
    with paths[0].open('x') as public, paths[1].open('x') as owners, paths[2].open('x') as pairs:
        for domain in 'ABC':
            for index in range(40):
                budget.check()
                meta, items, identities = generator.make_group(policy, catalog, domain, NAMESPACE, index)
                pair_rows = list(generator.pair_rows(identities))
                checked = validator.check_group(items, identities, pair_rows, checker)
                expected = generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'group')
                ids = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'account', i) for i in range(28)}
                control = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, 'controller', i) for i in range(12)}
                if (meta['group_uid'] != expected or expected in old | seen or checked['accounts'] != ids
                        or {r['controller_uid'] for r in identities} != control or accounts & ids
                        or controllers & control or items_seen & checked['item_ids']):
                    raise ValueError('New generated identities or namespace overlap')
                seen.add(expected)
                accounts |= ids
                controllers |= control
                items_seen |= checked['item_ids']
                public.write(generator.json_line(dict(metadata=meta, items=items)))
                owners.write(generator.json_line(identities))
                pairs.write(generator.json_line(pair_rows))
    if (len(seen), len(accounts), len(controllers)) != (120, 3360, 1440):
        raise ValueError('New confirmation is incomplete')
    files = {p.name: data.record(p, job) for p in paths}
    paths[1].chmod(0)
    data.write_json(job/'result.json', dict(status='GENERATED_VALIDATED_OWNERS_SEALED',
        root_seed=CONFIRM_SEED, namespace=NAMESPACE, groups=120, files=files, sources=allowed['sources']))


def validate_cells(result):
    expected = {run_id(name, s.order, s.seed) for name in NAMES for s in formal.specs()}
    if (set(result['runs']) != expected or result['forwards'] != 4320
            or len(result['groups']) != 120 or len(set(result['groups'])) != 120
            or result['domains'] != [d for d in 'ABC' for _ in range(40)]):
        raise ValueError('All 18 trajectories/108 cells/4320 forwards are required')
    for name in NAMES:
        for spec in formal.specs():
            cells = result['runs'][run_id(name, spec.order, spec.seed)]
            if set(cells) != {f'{i}_{j}' for i, j in CELLS}:
                raise ValueError('Missing or extra confirmation cell')
            for stage, position in CELLS:
                cell = cells[f'{stage}_{position}']
                domain = spec.order[position-1]
                ids = [uid for uid, d in zip(result['groups'], result['domains'], strict=True) if d == domain]
                if cell['domain'] != domain or cell['ids'] != ids or set(cell['scores']) != set(m.parent.ROLES):
                    raise ValueError('Wrong domain/group alignment')


def confirmation(campaign, job, records, budget, allowed):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(allowed['limits']['allocator_bytes'] /
        torch.cuda.get_device_properties(0).total_memory)
    generated = data.read_json(campaign/'generation/result.json')
    if (generated['namespace'] != NAMESPACE or generated['root_seed'] != CONFIRM_SEED
            or generated['sources'] != allowed['sources']):
        raise ValueError('New confirmation source differs')
    groups, domains = prior.confirmation_public(campaign/'generation')
    by_domain = {d: [g for g, actual in zip(groups, domains, strict=True) if actual == d] for d in 'ABC'}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0)
    prior.access(campaign, 'forever_blind_inference', ('new_public_text', '54_existing_inference_states'))
    for item in records:
        spec, root = m.Spec(item['order'], item['seed']), Path(item['root'])
        identity = run_id(item['name'], spec.order, spec.seed)
        is_h = item['name'] == 'H_pos_tuned'
        method = m.five.Method(m.five.Spec('H_pos_tuned', spec.order, spec.seed)) if is_h else None
        c = m.five.config(method.spec) if is_h else m.config(spec)
        model, optimizer = method.fresh(c) if is_h else m.fresh(spec, c)
        del optimizer
        result['runs'][identity] = {}
        for stage in (1, 2, 3):
            budget.check()
            point = item['record']['points'][str(stage)]
            path = data.verify(root/point['model']['path'], point['model'])
            if is_h:
                actual = method.restore(path, point['model'], model, None)
            else:
                actual = m.core.restore_state(path, model, None, point['state_sha256'])
            if actual['stage'] != stage or actual['maps'] != point['maps']:
                raise ValueError('Actual inference model/calibration differs')
            for _, position in (cell for cell in CELLS if cell[0] == stage):
                domain, key = spec.order[position-1], f'{stage}_{position}'
                with m.diag.neutral(model):
                    raw = (method.score(model, by_domain[domain], c, budget.check) if is_h
                           else m.record.score(model, by_domain[domain], c, budget.check))
                roles, preserved = shared.calibrated_roles(raw, point['maps'])
                result['runs'][identity][key] = dict(domain=domain, ids=[g.uid for g in by_domain[domain]],
                    preserved=preserved, scores={role: save_array(job/f'{identity}_{key}_{role}.npy', values, job)
                                                for role, values in roles.items()})
                result['forwards'] += 40
        del model
        gc.collect()
        torch.cuda.empty_cache()
        data.write_json(job/'blind_progress.json', result)
    validate_cells(result)
    for cells in result['runs'].values():
        for cell in cells.values():
            raw = np.load(data.verify(job/cell['scores']['raw']['path'], cell['scores']['raw']), allow_pickle=False)
            for role in ('stage-cal', 'first-cal'):
                rec = cell['scores'][role]
                m.parent.calibration.preserve_order(raw, np.load(data.verify(job/rec['path'], rec), allow_pickle=False))
    data.write_json(job/'all_blind_complete.json', {**result, 'sources': allowed['sources']})
    prior.access(campaign, 'forever_confirmation_joint', ('new_pairs',))
    truth = {}
    with data.verify(campaign/'generation/pairs.jsonl', generated['files']['pairs.jsonl']).open() as stream:
        for group, line in zip(groups, stream, strict=True):
            labels = data.align_labels(group, [dict(r, label=str(r['label'])) for r in json.loads(line)])
            if sum(labels) != 20:
                raise ValueError('Fixed confirmation label structure differs')
            truth[group.uid] = labels
    for identity, cells in result['runs'].items():
        for key, cell in cells.items():
            labels = np.asarray([truth[uid] for uid in cell['ids']], np.uint8)
            cell['metrics'] = {}
            for role, rec in cell['scores'].items():
                budget.check()
                values = np.load(data.verify(job/rec['path'], rec), allow_pickle=False)
                cell['metrics'][role] = prior.metric_set(job/f'metrics_{identity}_{key}_{role}', labels, values)
    data.write_json(job/'result.json', {**result, 'status': 'ALL_FOREVER_METRICS_COMPLETE', 'sources': allowed['sources']})


def analyze(campaign, job, budget, allowed):
    saved = data.read_json(campaign/'confirmation/result.json')
    if saved['status'] != 'ALL_FOREVER_METRICS_COMPLETE' or saved['sources'] != allowed['sources']:
        raise ValueError('Same-source complete confirmation required')
    validate_cells(saved)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(40, size=(5000, 3, 40))
    by_run, cells_saved = {}, {}
    for name in NAMES:
        for spec in formal.specs():
            identity = run_id(name, spec.order, spec.seed)
            by_run[identity] = {}
            for role in m.parent.ROLES:
                cells = {}
                for key, point in saved['runs'][identity].items():
                    rec = point['metrics'][role]['matrix']
                    row = data.read_json(data.verify(campaign/'confirmation'/rec['path'], rec))
                    values = np.asarray(row['values'], np.float64)
                    if row['columns'] != list(m.parent.metrics.COLUMNS) or values.shape != (40, 22):
                        raise ValueError('22 saved metrics and 40 groups required')
                    cells[key] = values
                    cells_saved[f'{identity}_{role}_{key}'] = dict(domain=point['domain'],
                        values=prior.nullable(values.mean(0)), counts=row['counts'])
                by_run[identity][role] = {field: shared.contributions(spec, cells, field) for field in FIELDS}
    averaged = {name: {role: {field: np.mean([by_run[run_id(name, s.order, s.seed)][role][field]
        for s in formal.specs()], axis=0) for field in FIELDS} for role in m.parent.ROLES} for name in NAMES}
    difference = {role: {field: shared.interval(averaged[NAMES[0]][role][field] - averaged[NAMES[1]][role][field],
        draws, (.025, .975), budget.check) for field in FIELDS} for role in m.parent.ROLES}
    for roles in averaged.values():
        for fields in roles.values():
            left, right = fields['F'], fields['A_acq'] - fields['O']
            finite = np.isfinite(left) & np.isfinite(right)
            np.testing.assert_allclose(left[finite], right[finite], rtol=1e-10, atol=1e-10)
    raw = difference['raw']
    checks = dict(old=raw['O']['map']['conditional_interval'][0] > 0,
        forgetting=raw['F']['map']['conditional_interval'][1] < 0,
        new=raw['N']['map']['mean'] >= 0, last=raw['Z']['map']['mean'] >= 0)
    data.write_json(job/'result.json', dict(status='FOREVER_FIXED_COMPARISON_COMPLETE', comparison='H-FOREVER',
        checks=checks, passed=all(checks.values()), difference=difference, per_run_actual_domain_cells=cells_saved,
        absolute={name: {role: {field: prior.nullable(v.mean(1).sum(0)) for field, v in fields.items()}
                            for role, fields in roles.items()} for name, roles in averaged.items()},
        bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=5000, groups_per_domain=40, paired_fixed_nine_paths=True),
        limits='Conditional group uncertainty; unequal history state, compute and HPO. Old H-S failure unchanged.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'confirmation', 'analyze'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    allowed = queue.gate(args.gate, 'FOREVER', args.action)
    if allowed.get('new_confirmation_data') is not True:
        raise ValueError('New 120-group access has not been enabled')
    campaign = Path(allowed['campaign'])
    records = complete_training(campaign, allowed)
    job = campaign/{'generate': 'generation', 'confirmation': 'confirmation', 'analyze': 'analysis'}[args.action]
    if job.exists() and any(p.name not in ('console.txt', 'process.json') for p in job.iterdir()):
        raise ValueError('A previous action cannot silently restart')
    job.mkdir(exist_ok=True)
    budget = formal.Budget(campaign, args.deadline, allowed['limits'])
    try:
        if args.action == 'generate':
            generate(campaign, job, records, budget, allowed)
        elif args.action == 'confirmation':
            confirmation(campaign, job, records, budget, allowed)
        else:
            analyze(campaign, job, budget, allowed)
        data.write_json(job/'resource_receipt.json', budget.summary())
    except BaseException as error:
        data.write_json(job/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
