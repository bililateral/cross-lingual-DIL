"""Expression-shift pilot: single-domain, joint and sequential full-text training.

Linux execution requires the user's resumption of the separately reported stage.
Only new train binary supervision is parsed there; valid is scored blindly.
Windows evaluation consumes valid once after the complete result gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np

import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics
import step28_continual_population_run as persistence

POLICY = data.ROOT / 'schema/step28_continual_expression_run_policy.json'
COMPLETE = 'COMPLETE_EXPRESSION_PILOT_2160_VALID_BLIND'
ORDERS = ('ABC', 'BCA', 'CAB')
POINTS = ('initial',) + tuple(f'{o}_{s}' for o in ORDERS for s in ('shared', 'stage2', 'stage3')) + ('joint',)


def sources() -> list[dict]:
    paths = [Path(__file__), POLICY, Path(core.__file__), Path(data.__file__),
             Path(metrics.__file__), Path(persistence.__file__),
             data.ROOT / 'tests/test_step28_continual_expression_run_contracts.py',
             data.ROOT / 'scripts/run_step28_expression_linux_20260910.sh']
    return [data.record(p, data.ROOT) for p in paths]


def contract() -> dict:
    c = data.read_json(POLICY)
    e = c['evaluation']
    if (c['physical_updates'] != 2160 or c['orders'] != list(ORDERS)
            or c['epochs_per_stage'] != 3 or c['initialization_seed'] != 20260909
            or c['groups_per_domain'] != {'train': 60, 'development': 20, 'heldout': 40}
            or e['heldout_access'] or e['owners_access'] or e['split'] != 'development'
            or e['same_order_required'] != 2 or e['new_gain_min'] != .02
            or e['forgetting_min'] != .03 or e['single_gain_min'] != .02
            or e['joint_single_floor'] != -.02 or e['bootstrap_replicates'] != 5000):
        raise ValueError('Pilot differs from confirmed design')
    return c


def public_inputs(c: dict) -> tuple[dict, list[dict], dict]:
    root = data.ROOT / c['data_root']
    for name, key in (('manifest.json', 'data_manifest_sha256'), ('validation.json', 'data_validation_sha256')):
        if data.sha256(root / name) != c[key]:
            raise ValueError('Pinned expression dataset differs')
    dm = data.read_json(root / 'manifest.json')
    if (dm['study'] != 'seller_alias_continual_expression_shift' or dm['root_seed'] != 20260910
            or data.read_json(root / 'validation.json')['status'] != 'PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION'):
        raise ValueError('Expression generation qualification missing')
    checked = {}

    def path(name: str) -> Path:
        checked[name] = dm['files'][name]
        return data.verify(root / name, checked[name])

    with path('groups.csv').open(encoding='utf-8', newline='') as stream:
        meta = sorted((r for r in csv.DictReader(stream) if r['split'] in ('train', 'development')),
                      key=lambda r: (r['domain'], int(r['group_index'])))
    groups, sellers_seen, items_seen = {}, set(), set()
    for split, count in (('train', 60), ('development', 20)):
        rows = [r for r in meta if r['split'] == split]
        if len({r['group_uid'] for r in rows}) != 3 * count or any(
                sorted(int(r['group_index']) for r in rows if r['domain'] == d) != list(range(count)) for d in 'ABC'):
            raise ValueError('Public domain partition differs')
        collected = defaultdict(lambda: defaultdict(list))
        with path(f'{split}/items.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                if set(r) != {'group_uid', 'seller_uid', 'item_uid', 'title', 'description'} or r['item_uid'] in items_seen:
                    raise ValueError('Public item schema/identity differs')
                items_seen.add(r['item_uid'])
                collected[r['group_uid']][r['seller_uid']].append((r['item_uid'], r['title'], r['description']))
        if set(collected) != {r['group_uid'] for r in rows}:
            raise ValueError('Public group coverage differs')
        groups[split] = []
        for r in rows:
            sellers = collected[r['group_uid']]
            ids = tuple(sorted(sellers))
            if sellers_seen.intersection(ids):
                raise ValueError('Seller reused across groups/splits')
            sellers_seen.update(ids)
            g = data.Group(r['group_uid'], ids, tuple(tuple(sorted(sellers[s])) for s in ids))
            g.validate()
            if int(r['accounts']) != 28 or int(r['items']) != sum(map(len, g.items)):
                raise ValueError('Group record counts differ')
            groups[split].append(g)
    return groups, meta, checked


def attach_labels(groups: list[data.Group], c: dict, split: str) -> list[data.Group]:
    if split not in ('train', 'development'):
        raise ValueError('Forbidden supervision split')
    root = data.ROOT / c['data_root']
    dm = data.read_json(root / 'manifest.json')
    name = f'{split}/supervision/pairs.csv'
    rows = defaultdict(list)
    with data.verify(root / name, dm['files'][name]).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['group_uid', 'seller_uid_left', 'seller_uid_right', 'label']:
            raise ValueError('Unexpected binary supervision schema')
        for r in reader:
            rows[r['group_uid']].append(r)
    if set(rows) != {g.uid for g in groups}:
        raise ValueError('Supervision groups differ; do not retry')
    result = []
    for g in groups:
        labels = data.align_labels(g, rows[g.uid])
        if sum(labels) != 20:
            raise ValueError('Positive pair count differs')
        result.append(data.Group(g.uid, g.sellers, g.items, labels))
    return result


def segment(order: str, stage: int) -> tuple[str, str, int]:
    if order == 'JOINT' and stage == 1:
        return 'joint', 'ABC', 0
    if order not in ORDERS or stage not in (1, 2, 3):
        raise ValueError('Unknown training segment')
    return f'{order}_{"shared" if stage == 1 else "stage" + str(stage)}', order[stage - 1], (stage - 1) * 180


def segment_schedule(groups: list[data.Group], meta: list[dict], c: dict, order: str, stage: int) -> tuple[list, int]:
    _, domains, _ = segment(order, stage)
    selected = {r['group_uid'] for r in meta if r['split'] == 'train' and r['domain'] in domains}
    current = [g for g in groups if g.uid in selected]
    if len(current) != 60 * len(domains) or len({g.uid for g in current}) != len(current):
        raise ValueError('Current-domain supply differs')
    seed = data.seed_for(c['initialization_seed'], order, stage, 'current')
    return data.schedule(current, 3, seed), seed


def train_segment(model: Any, optimizer: Any, groups: list, meta: list, c: dict,
                  order: str, stage: int, budget: persistence.Budget) -> dict:
    started = time.monotonic()
    name, _, prior_steps = segment(order, stage)
    rows, seed = segment_schedule(groups, meta, c, order, stage)
    losses, norms, checks = [], [], {}
    for i, g in enumerate(rows):
        if i == 0:
            before = {n: core.state_digest(m.state_dict()) for n, m in (('encoder', model.encoder), ('head', model.head))}
        r = core.update(model, optimizer, g, None, c, data.seed_for(seed, i, 'dropout_current'),
                        data.seed_for(seed, i, 'dropout_replay'), budget.check)
        losses.append(r['current_bce'])
        norms.append(r['gradient_norm_before_clip'])
        if i == 0:
            for n, m in (('encoder', model.encoder), ('head', model.head)):
                ns = [float(p.grad.detach().float().norm()) for p in m.parameters() if p.grad is not None]
                if not ns or not np.isfinite(ns).all() or max(ns) <= 0 or before[n] == core.state_digest(m.state_dict()):
                    raise ValueError(f'Actual first update failed {n}')
                checks[n] = {'finite_nonzero_gradient': True, 'parameters_changed': True}
            if not optimizer.state or any(float(v['step']) != prior_steps + 1 for v in optimizer.state.values()):
                raise ValueError('Adam did not inherit the correct update history')
        if (i + 1) % 30 == 0:
            print(data.json_bytes({'event': 'updates', 'segment': name, 'completed': i + 1,
                                  'total': len(rows), **budget.state()}).decode(), flush=True)
    ids = [g.uid for g in rows]
    if not optimizer.state or any(float(v['step']) != prior_steps + len(rows) for v in optimizer.state.values()):
        raise ValueError('Adam final update count differs')
    return {'updates': len(rows), 'training_seconds': time.monotonic() - started,
            'group_ids': ids, 'order_sha256': hashlib.sha256(data.json_bytes(ids)).hexdigest(),
            'dropout_stream': seed, 'pair_presentations': len(rows) * 378,
            'first_adam_step': prior_steps + 1, 'last_adam_step': prior_steps + len(rows),
            'first_update_modules': checks, 'bce_by_epoch': np.asarray(losses).reshape(3, -1).mean(1).tolist(),
            'maximum_gradient_norm_before_clip': max(norms)}


def run(out: Path) -> dict:
    import torch
    c = contract()
    runtime = c['runtime']
    if platform.system() != 'Linux' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires resumed Linux stage and one visible GPU')
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < runtime['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient eligible idle GPU; wait')
    available = next(int(r.split()[1]) * 1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))
    if available < runtime['minimum_free_host_bytes'] or shutil.disk_usage(data.ROOT).free < runtime['maximum_output_bytes']:
        raise RuntimeError('Insufficient shared host/disk resources; wait')
    if out.exists() or not out.is_relative_to((data.ROOT / 'reports').resolve()):
        raise ValueError('Use a new reports directory')
    out.mkdir(parents=True)
    for name in ('work', 'models', 'scores'):
        (out / name).mkdir()
    budget = persistence.Budget(out, c)
    snapshot = sources()
    r = {'status': 'RUNNING', 'config': c, 'source_files': snapshot, 'points': {}, 'training': {}, 'models': {},
         'starts': {}, 'label_parses': {'train': 0, 'development': 0, 'heldout': 0, 'owners': 0},
         'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}}
    data.write_json(out / 'startup.json', r)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        pretrained = core.model_files(c)
        if any(pretrained[k] != c['model'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
            raise ValueError('Pretrained model differs')
        r['pretrained'] = pretrained
        groups, meta, checked = public_inputs(c)
        dm = data.read_json(data.ROOT / c['data_root'] / 'manifest.json')
        checked['train/supervision/pairs.csv'] = dm['files']['train/supervision/pairs.csv']
        r['inputs'] = {'files': checked, 'metadata': meta, 'valid_group_ids': [g.uid for g in groups['development']]}
        data.write_json(out / 'access.json', {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0})
        r['label_parses']['train'] = 1
        groups['train'] = attach_labels(groups['train'], c, 'train')
        model = core.load_model(c)
        evaluation = {'development': groups['development']}
        initial = persistence.point(model, None, 'initial', evaluation, c, out, budget)
        r['points']['initial'] = initial
        initial_path = out / initial['checkpoint']['path']
        for order in ORDERS:
            core.restore_state(initial_path, model, None, initial['checkpoint']['state_sha256'])
            r['starts'][order] = initial['checkpoint']['state_sha256']
            optimizer = core.make_optimizer(model, c)
            for stage in (1, 2, 3):
                name, _, _ = segment(order, stage)
                r['training'][name] = train_segment(model, optimizer, groups['train'], meta, c, order, stage, budget)
                p = persistence.point(model, optimizer, name, evaluation, c, out, budget)
                r['points'][name] = p
                if stage in (1, 3):
                    r['models'][name] = persistence.retain_inference(model, name, p, c, out, budget)
                if stage == 1:
                    # Explicit new optimizer restoration, not only inference-weight reuse.
                    del optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    optimizer = core.make_optimizer(model, c)
                    core.restore_state(out / p['checkpoint']['path'], model, optimizer, p['checkpoint']['state_sha256'])
                    r['starts'][order + '_continuation'] = p['checkpoint']['state_sha256']
                persistence.remove_work_file(out / p['checkpoint']['path'], out)
                data.write_json(out / 'progress.json', r)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        core.restore_state(initial_path, model, None, initial['checkpoint']['state_sha256'])
        r['starts']['JOINT'] = initial['checkpoint']['state_sha256']
        optimizer = core.make_optimizer(model, c)
        r['training']['joint'] = train_segment(model, optimizer, groups['train'], meta, c, 'JOINT', 1, budget)
        p = persistence.point(model, optimizer, 'joint', evaluation, c, out, budget)
        r['points']['joint'] = p
        r['models']['joint'] = persistence.retain_inference(model, 'joint', p, c, out, budget)
        persistence.remove_work_file(out / p['checkpoint']['path'], out)
        persistence.remove_work_file(initial_path, out)
        r['physical_updates'] = sum(x['updates'] for x in r['training'].values())
        if r['physical_updates'] != 2160 or sources() != snapshot:
            raise ValueError('Update count or source mismatch')
        r['formal_training_seconds'] = sum(x['training_seconds'] for x in r['training'].values())
        r['inference_seconds_including_replay'] = sum(x['timing']['inference_seconds_including_replay'] for x in r['points'].values())
        r['budget'] = budget.state()
        r['status'] = COMPLETE
        data.write_json(out / 'manifest.json', r)
        verify_models(out)
        return r
    except Exception as e:
        data.write_json(out / 'failure.json', {'status': 'FAILED_DO_NOT_EVALUATE_OR_AUTO_RESTART',
                        'error_type': type(e).__name__, 'error': str(e), 'label_parses': r['label_parses']})
        raise


def verify_models(out: Path) -> dict:
    r = data.read_json(out / 'manifest.json')
    if r['status'] != COMPLETE or len(r['models']) != 7:
        raise ValueError('Incomplete retained models')
    files = [data.record(data.verify(out / m['path'], m), out) for m in r['models'].values()]
    result = {'manifest_sha256': data.sha256(out / 'manifest.json'), 'files': files}
    data.write_json(out / 'model_verification.json', result)
    return result


def validated_scores(out: Path, c: dict, groups: dict, meta: list, checked: dict) -> tuple[dict, dict]:
    if (out / 'failure.json').exists():
        raise ValueError('Failed run cannot be evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != COMPLETE or r['config'] != c or r['source_files'] != sources()
            or r['label_parses'] != {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0}
            or r['physical_updates'] != 2160 or set(r['points']) != set(POINTS)
            or set(r['training']) != set(POINTS) - {'initial'} or set(r['models']) != set(c['retained_points'])
            or r['budget']['elapsed_seconds'] > c['runtime']['maximum_gpu_stage_seconds']
            or r['budget']['peak_observed_bytes'] > c['runtime']['maximum_output_bytes']):
        raise ValueError('Incomplete pilot result contract')
    dm = data.read_json(data.ROOT / c['data_root'] / 'manifest.json')
    expected_files = {**checked, 'train/supervision/pairs.csv': dm['files']['train/supervision/pairs.csv']}
    if r['inputs'] != {'files': expected_files, 'metadata': meta, 'valid_group_ids': [g.uid for g in groups['development']]}:
        raise ValueError('Input identities differ')
    if any(r['pretrained'][k] != c['model'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
        raise ValueError('Pretrained source differs')
    expected_starts = dict.fromkeys((*ORDERS, 'JOINT'), r['points']['initial']['checkpoint']['state_sha256'])
    expected_starts.update({o + '_continuation': r['points'][o + '_shared']['checkpoint']['state_sha256'] for o in ORDERS})
    if r['starts'] != expected_starts:
        raise ValueError('Initial/shared full-state mapping differs')
    for order, stage in [(o, s) for o in ORDERS for s in (1, 2, 3)] + [('JOINT', 1)]:
        name, _, prior_steps = segment(order, stage)
        rows, seed = segment_schedule(groups['train'], meta, c, order, stage)
        log, ids = r['training'][name], [g.uid for g in rows]
        if (log['group_ids'] != ids or log['updates'] != len(rows) or log['dropout_stream'] != seed
                or log['order_sha256'] != hashlib.sha256(data.json_bytes(ids)).hexdigest()
                or log['pair_presentations'] != len(rows) * 378
                or log['first_adam_step'] != prior_steps + 1 or log['last_adam_step'] != prior_steps + len(rows)
                or log['first_update_modules'] != {k: {'finite_nonzero_gradient': True, 'parameters_changed': True} for k in ('encoder', 'head')}
                or len(log['bce_by_epoch']) != 3 or not np.isfinite(log['bce_by_epoch']).all()):
            raise ValueError('Training schedule or actual update evidence differs')
    receipt = data.read_json(out / 'model_verification.json')
    expected_models = [{k: m[k] for k in ('path', 'bytes', 'sha256')} for m in r['models'].values()]
    if receipt['manifest_sha256'] != data.sha256(out / 'manifest.json') or receipt['files'] != expected_models:
        raise ValueError('Linux retained-model verification missing')
    for name, m in r['models'].items():
        if m['path'] != f'models/{name}.pt' or not m['actual_loaded_model_equals_replayed_state']:
            raise ValueError('Retained model mapping/reload differs')
    scores = {}
    for name, p in r['points'].items():
        if not p['full_model_and_adam_reloaded'] or set(p['scores']) != {'development'}:
            raise ValueError('Full-state/score replay missing')
        file = p['scores']['development']
        if file['path'] != f'scores/{name}_development.npy':
            raise ValueError('Score mapping differs')
        a = np.load(data.verify(out / file['path'], file), allow_pickle=False)
        if a.shape != (60, 378) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Incomplete/nonfinite score array')
        scores[name] = a
    return scores, r


def paired_summary(delta: np.ndarray, draws: np.ndarray) -> dict:
    if delta.shape != (3, 20) or not np.isfinite(delta).all():
        raise ValueError('Expected paired domain/group deltas')
    sampled = np.stack([delta[d][draws[:, d]] for d in range(3)], axis=1).mean((1, 2))
    return {'mean': float(delta.mean()), 'by_domain_or_first_domain': delta.mean(1).tolist(),
            'conditional_95pct_interval': np.quantile(sampled, [.025, .975]).tolist()}


def contrasts(arrays: dict, column: int) -> dict[str, np.ndarray]:
    initial = arrays['initial'][:, column].reshape(3, 20)
    singles = np.stack([arrays[o + '_shared'][i*20:(i+1)*20, column] for i, o in enumerate(ORDERS)])
    final = np.stack([arrays[o + '_stage3'][i*20:(i+1)*20, column] for i, o in enumerate(ORDERS)])
    # Every target domain appears twice as a new domain; combine its same groups BEFORE bootstrap.
    new_by_domain = defaultdict(list)
    for o in ORDERS:
        for stage in (2, 3):
            before = o + ('_shared' if stage == 2 else '_stage2')
            d = 'ABC'.index(o[stage - 1])
            new_by_domain[d].append(arrays[o + f'_stage{stage}'][d*20:(d+1)*20, column]
                                    - arrays[before][d*20:(d+1)*20, column])
    return {'single_gain': singles - initial,
            'joint_minus_single': arrays['joint'][:, column].reshape(3, 20) - singles,
            'first_domain_forgetting': singles - final,
            'new_domain_gain': np.stack([np.mean(new_by_domain[d], axis=0) for d in range(3)])}


def qualification(arrays: dict, comparison: dict, e: dict) -> dict:
    ap = metrics.COLUMNS.index('average_precision')
    c = contrasts(arrays, ap)
    matched = []
    order_details = {}
    for i, o in enumerate(ORDERS):
        gains = []
        for stage in (2, 3):
            before = o + ('_shared' if stage == 2 else '_stage2')
            d = 'ABC'.index(o[stage - 1])
            gains.append(float((arrays[o + f'_stage{stage}'] - arrays[before])[d*20:(d+1)*20, ap].mean()))
        f = float(c['first_domain_forgetting'][i].mean())
        ok = f > 0 and float(np.mean(gains)) >= e['new_gain_min']
        if ok:
            matched.append(o)
        order_details[o] = {'first_domain_forgetting': f, 'new_gain_stage2_stage3': gains,
                            'mean_new_gain': float(np.mean(gains)), 'joint_condition': bool(ok)}
    gates = {'single_domain_gain': bool(np.all(c['single_gain'].mean(1) >= e['single_gain_min'])),
             'joint_compatibility': bool(np.all(c['joint_minus_single'].mean(1) >= e['joint_single_floor'])),
             'mean_forgetting': bool(c['first_domain_forgetting'].mean() >= e['forgetting_min']),
             'positive_forgetting_interval': comparison['first_domain_forgetting']['conditional_95pct_interval'][0] > 0,
             'mean_new_gain': bool(c['new_domain_gain'].mean() >= e['new_gain_min']),
             'same_order_joint_condition': len(matched) >= e['same_order_required']}
    return {'gates': gates, 'all_numeric_gates_pass': all(gates.values()), 'orders': order_details,
            'matched_orders': matched, 'interpretation': 'Numeric qualification is not a novelty or publication claim. Inspect absolute performance, MAP and each transfer; it does not require every transfer to gain.'}


def evaluate(out: Path, destination: Path) -> dict:
    c = contract()
    if platform.system() != 'Windows' or destination.exists():
        raise ValueError('Requires Windows and new evaluation directory')
    groups, meta, checked = public_inputs(c)
    scores, run_record = validated_scores(out, c, groups, meta, checked)
    destination.mkdir(parents=True)
    data.write_json(destination / 'access.json', {'status': 'COMPLETE_GATE_PASSED_VALID_PARSE_STARTING',
                    'development_parse_attempts': 1, 'train': 0, 'heldout': 0, 'owners': 0})
    labelled = attach_labels(groups['development'], c, 'development')
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    arrays, points = {}, {}
    for name, s in scores.items():
        rows = [metrics.classification(y, x) for y, x in zip(truth, s)]
        matrix = np.column_stack((np.asarray([[r[k] for k in metrics.CLASS_KEYS] for r in rows]), metrics.retrieval(truth, s, 28)))
        arrays[name] = matrix
        file = destination / (name + '_metrics.npy')
        np.save(file, matrix, allow_pickle=False)
        points[name] = {'file': data.record(file, destination), 'confusion': [r['confusion'] for r in rows],
                       'by_domain': {d: dict(zip(metrics.COLUMNS, matrix[i*20:(i+1)*20].mean(0).tolist())) for i, d in enumerate('ABC')}}
    e = c['evaluation']
    draws = np.random.default_rng(e['bootstrap_seed']).integers(0, 20, size=(e['bootstrap_replicates'], 3, 20))
    comparisons = {name: {key: paired_summary(delta, draws) for key, delta in contrasts(arrays, i).items()}
                   for i, name in enumerate(metrics.COLUMNS)}
    result = {'status': 'EXPRESSION_PILOT_VALID_EVALUATED_INTERPRETATION_REQUIRED',
              'score_manifest_sha256': data.sha256(out / 'manifest.json'), 'points': points,
              'columns': list(metrics.COLUMNS), 'comparisons': comparisons,
              'qualification': qualification(arrays, comparisons['average_precision'], e),
              'training_bce': {k: v['bce_by_epoch'] for k, v in run_record['training'].items()},
              'label_parses': {'development': 1, 'train': 0, 'heldout': 0, 'owners': 0},
              'bootstrap': {'replicates': e['bootstrap_replicates'], 'seed': e['bootstrap_seed'],
                            'unit': 'paired independent full group within domain; repeated paths averaged first'}}
    data.write_json(destination / 'evaluation.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate', 'verify-models'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    if args.action == 'evaluate' and args.evaluation is None:
        parser.error('evaluate requires --evaluation')
    if args.action == 'train':
        result = run(args.out.resolve())
    elif args.action == 'verify-models':
        result = verify_models(args.out.resolve())
    else:
        result = evaluate(args.out.resolve(), args.evaluation.resolve())
    print(data.json_bytes({'status': result.get('status', 'MODEL_HASHES_VERIFIED'), 'out': str(args.out)}).decode())


if __name__ == '__main__':
    main()
