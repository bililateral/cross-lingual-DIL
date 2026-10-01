"""Fixed three/six-epoch joint diagnosis; no mitigation training or endpoint search.

Linux train access and Windows complete-gated valid access are separate once-only
stages. Launch only after the user resumes the separately reported Linux stage.
Original code/configuration and complete Adam state are pinned for exact replay.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np

import step28_continual_expression_run as pilot
from step28_continual_expression_run import core, data, metrics, persistence

POLICY = data.ROOT / 'schema/step28_continual_joint_policy.json'
COMPLETE = 'COMPLETE_JOINT_1080_REPLAYED_TRAIN_RISK_VALID_BLIND'
NAMES = ('joint', 'joint_six', 'ABC_shared', 'BCA_shared', 'CAB_shared')
TRAIN_COLUMNS = ('static_bce_with_logits', 'average_precision')


def contract() -> dict:
    c = data.read_json(POLICY)
    if (c['study'] != 'seller_alias_joint_extension_diagnosis'
            or c['epochs'] != [3, 6] or c['physical_updates'] != 1080
            or c['label_parses'] != {'train': 1, 'development': 1, 'heldout': 0, 'owners': 0}
            or c['runtime'] != {'maximum_gpu_stage_seconds': 10800,
                'maximum_output_bytes': 12 * 1024**3, 'minimum_free_gpu_bytes': 24 * 1024**3,
                'minimum_free_host_bytes': 16 * 1024**3, 'torch_cpu_threads': 1}):
        raise ValueError('Joint diagnosis differs from approved contract')
    return c


def sources() -> list[dict]:
    extra = [Path(__file__), POLICY,
             data.ROOT / 'tests/test_step28_continual_joint_contracts.py',
             data.ROOT / 'scripts/run_step28_joint_linux_20260910.sh']
    return pilot.sources() + [data.record(p, data.ROOT) for p in extra]


def origin(c: dict, groups: dict, meta: list, checked: dict) -> tuple[dict, dict, dict]:
    root = data.ROOT / c['origin_run']
    if data.sha256(root / 'manifest.json') != c['origin_manifest_sha256']:
        raise ValueError('Original run manifest differs')
    old = data.read_json(root / 'manifest.json')
    old_c = pilot.contract()
    if old['config'] != old_c:
        raise ValueError('Original training configuration differs')
    scores, verified = pilot.validated_scores(root, old_c, groups, meta, checked)
    path = data.ROOT / c['origin_evaluation']
    if data.sha256(path) != c['origin_evaluation_sha256']:
        raise ValueError('Original valid evaluation differs')
    e = data.read_json(path)
    if e['score_manifest_sha256'] != c['origin_manifest_sha256'] or e['columns'] != list(metrics.COLUMNS):
        raise ValueError('Original evaluation mapping differs')
    return verified, scores, e


def joint_schedule(groups: list, meta: list, old_c: dict) -> tuple[list, int]:
    original, seed = pilot.segment_schedule(groups, meta, old_c, 'JOINT', 1)
    rows = data.schedule(groups, 6, seed)
    if len(rows) != 1080 or [g.uid for g in rows[:540]] != [g.uid for g in original]:
        raise ValueError('Six-epoch schedule does not preserve original prefix')
    return rows, seed


def train_groups(name: str, groups: list, meta: list) -> list:
    if name in ('joint', 'joint_six'):
        return groups
    if name not in NAMES:
        raise ValueError('Unknown fixed endpoint')
    selected = {r['group_uid'] for r in meta if r['split'] == 'train' and r['domain'] == name[0]}
    result = [g for g in groups if g.uid in selected]
    if len(result) != 60:
        raise ValueError('Single-domain training-risk group mapping differs')
    return result


def training_record(ids: list[str], seed: int, start: int, losses: list,
                    norms: list, checks: dict, elapsed: float) -> dict:
    return {'updates': len(ids), 'group_ids': ids,
            'order_sha256': hashlib.sha256(data.json_bytes(ids)).hexdigest(),
            'dropout_stream': seed, 'dropout_index_start': start,
            'pair_presentations': len(ids) * 378, 'first_adam_step': start + 1,
            'last_adam_step': start + len(ids), 'first_update_modules': checks,
            'bce_by_epoch': np.asarray(losses).reshape(3, 180).mean(1).tolist(),
            'maximum_gradient_norm_before_clip': max(norms), 'training_seconds': elapsed}


def train_window(model: Any, optimizer: Any, rows: list, seed: int, start: int,
                 old_c: dict, budget: persistence.Budget) -> dict:
    if start not in (0, 540) or len(rows) != 1080:
        raise ValueError('Only fixed three/six epoch windows are allowed')
    begun = time.monotonic()
    losses, norms, checks = [], [], {}
    for index in range(start, start + 540):
        if index == start:
            before = {n: core.state_digest(m.state_dict()) for n, m in
                      (('encoder', model.encoder), ('head', model.head))}
        result = core.update(model, optimizer, rows[index], None, old_c,
                             data.seed_for(seed, index, 'dropout_current'),
                             data.seed_for(seed, index, 'dropout_replay'), budget.check)
        losses.append(result['current_bce'])
        norms.append(result['gradient_norm_before_clip'])
        if index == start:
            for n, module in (('encoder', model.encoder), ('head', model.head)):
                grads = [float(p.grad.detach().float().norm()) for p in module.parameters() if p.grad is not None]
                if (not grads or not np.isfinite(grads).all() or max(grads) <= 0
                        or before[n] == core.state_digest(module.state_dict())):
                    raise ValueError('Actual encoder/head update failed')
                checks[n] = {'finite_nonzero_gradient': True, 'parameters_changed': True}
            if not optimizer.state or any(float(v['step']) != start + 1 for v in optimizer.state.values()):
                raise ValueError('Adam history reset or incorrectly inherited')
        if (index + 1) % 30 == 0:
            print(data.json_bytes({'event': 'joint_updates', 'completed': index + 1,
                                  'total': 1080, **budget.state()}).decode(), flush=True)
    if not optimizer.state or any(float(v['step']) != start + 540 for v in optimizer.state.values()):
        raise ValueError('Adam final update count differs')
    return training_record([g.uid for g in rows[start:start + 540]], seed, start,
                           losses, norms, checks, time.monotonic() - begun)


def check_original_point(name: str, p: dict, old: dict, values: np.ndarray,
                         old_values: np.ndarray) -> None:
    expected = old['points'][name]
    # Singles have no optimizer; their full digest is the retained inference digest.
    state = (expected['checkpoint']['state_sha256'] if name == 'joint'
             else old['models'][name]['state_sha256'])
    if (p['checkpoint']['state_sha256'] != state
            or p['model_state_sha256'] != expected['model_state_sha256']
            or not np.array_equal(values, old_values)):
        raise ValueError('Original model/Adam/full valid scores did not replay exactly')


def static_train_metrics(groups: list, scores: np.ndarray) -> np.ndarray:
    if scores.shape != (len(groups), 378) or not np.isfinite(scores).all():
        raise ValueError('Training risk scores incomplete')
    rows = []
    for g, x in zip(groups, scores):
        if g.labels is None:
            raise ValueError('Static risk requires this stage authorized train supervision')
        y, x = np.asarray(g.labels), x.astype(np.float64)
        # Stable exact BCE-with-logits, not clipped probability log-loss.
        rows.append([float(np.mean(np.logaddexp(0., x) - y * x)),
                     metrics.curve_metrics(y, x)['average_precision']])
    return np.asarray(rows, dtype=np.float64)


def save_train_metrics(name: str, groups: list, p: dict, out: Path) -> dict:
    scores = np.load(data.verify(out / p['scores']['train']['path'], p['scores']['train']), allow_pickle=False)
    values = static_train_metrics(groups, scores)
    path = out / 'scores' / f'{name}_train_metrics.npy'
    np.save(path, values, allow_pickle=False)
    return {'columns': list(TRAIN_COLUMNS), 'group_ids': [g.uid for g in groups],
            'file': data.record(path, out)}


def run(out: Path) -> dict:
    import torch
    c, old_c = contract(), pilot.contract()
    runtime = c['runtime']
    if (platform.system() != 'Linux' or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Requires resumed Linux stage and one eligible visible GPU')
    available = next(int(r.split()[1]) * 1024 for r in Path('/proc/meminfo').read_text().splitlines()
                     if r.startswith('MemAvailable:'))
    if (torch.cuda.mem_get_info()[0] < runtime['minimum_free_gpu_bytes']
            or available < runtime['minimum_free_host_bytes']
            or shutil.disk_usage(data.ROOT).free < runtime['maximum_output_bytes']):
        raise RuntimeError('Insufficient shared resources; wait without modifying other processes')
    if out.exists() or not out.is_relative_to((data.ROOT / 'reports').resolve()):
        raise ValueError('Use a new project reports directory')
    out.mkdir(parents=True)
    for directory in ('work', 'scores', 'models'):
        (out / directory).mkdir()
    budget = persistence.Budget(out, c)
    snapshot = sources()
    r = {'status': 'RUNNING', 'config': c, 'source_files': snapshot, 'points': {},
         'training': {}, 'models': {}, 'train_metrics': {}, 'original_replays': {},
         'label_parses': {'train': 0, 'development': 0, 'heldout': 0, 'owners': 0},
         'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}}
    data.write_json(out / 'startup.json', r)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        groups, meta, checked = pilot.public_inputs(old_c)
        old, old_scores, _ = origin(c, groups, meta, checked)
        old_root = data.ROOT / c['origin_run']
        model_paths = {name: data.verify(old_root / old['models'][name]['path'], old['models'][name])
                       for name in NAMES if name != 'joint_six'}
        pretrained = core.model_files(old_c)
        if any(pretrained[k] != old_c['model'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
            raise ValueError('Original pretrained source differs')
        r['pretrained'] = pretrained
        r['inputs'] = old['inputs']
        r['original_model_files'] = {name: data.record(path, data.ROOT) for name, path in model_paths.items()}
        schedule, seed = joint_schedule(groups['train'], meta, old_c)
        if [g.uid for g in schedule[:540]] != old['training']['joint']['group_ids']:
            raise ValueError('Original recorded joint schedule differs')
        data.write_json(out / 'access.json', {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0})
        r['label_parses']['train'] = 1
        groups['train'] = pilot.attach_labels(groups['train'], old_c, 'train')
        schedule, seed = joint_schedule(groups['train'], meta, old_c)
        model = core.load_model(old_c)
        initial = core.state_digest({'model': model.state_dict(), 'optimizer': None,
                                    'metadata': {'point': 'initial', 'config': old_c}})
        if initial != old['points']['initial']['checkpoint']['state_sha256']:
            raise ValueError('Original initial model state differs')
        r['initial_state_sha256'] = initial
        optimizer = core.make_optimizer(model, old_c)
        for name, start in (('joint', 0), ('joint_six', 540)):
            r['training'][name] = train_window(model, optimizer, schedule, seed, start, old_c, budget)
            p = persistence.point(model, optimizer, name,
                {'train': groups['train'], 'development': groups['development']}, old_c, out, budget)
            r['points'][name] = p
            if name == 'joint':
                scores = np.load(out / p['scores']['development']['path'], allow_pickle=False)
                check_original_point(name, p, old, scores, old_scores[name])
                r['original_replays'][name] = True
            else:
                r['models'][name] = persistence.retain_inference(model, name, p, old_c, out, budget)
            r['train_metrics'][name] = save_train_metrics(name, groups['train'], p, out)
            persistence.remove_work_file(out / p['checkpoint']['path'], out)
            data.write_json(out / 'progress.json', r)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        for name in NAMES[2:]:
            metadata = core.restore_state(model_paths[name], model, None, old['models'][name]['state_sha256'])
            if metadata != {'point': name, 'config': old_c}:
                raise ValueError('Original single-domain model metadata differs')
            subset = train_groups(name, groups['train'], meta)
            p = persistence.point(model, None, name,
                {'train': subset, 'development': groups['development']}, old_c, out, budget)
            r['points'][name] = p
            values = np.load(out / p['scores']['development']['path'], allow_pickle=False)
            check_original_point(name, p, old, values, old_scores[name])
            r['original_replays'][name] = True
            r['train_metrics'][name] = save_train_metrics(name, subset, p, out)
            persistence.remove_work_file(out / p['checkpoint']['path'], out)
            data.write_json(out / 'progress.json', r)
        if sources() != snapshot:
            raise ValueError('Sources changed during training')
        r['physical_updates'] = sum(x['updates'] for x in r['training'].values())
        r['formal_training_seconds'] = sum(x['training_seconds'] for x in r['training'].values())
        r['inference_seconds_including_replay'] = sum(p['timing']['inference_seconds_including_replay'] for p in r['points'].values())
        r['scored_model_groups'] = 840
        r['group_forwards_with_replay'] = 1680
        r['budget'] = budget.state()
        r['status'] = COMPLETE
        data.write_json(out / 'manifest.json', r)
        verify_models(out)
        return r
    except Exception as error:
        data.write_json(out / 'failure.json', {'status': 'FAILED_NO_EVALUATION_OR_AUTO_RESTART',
            'error_type': type(error).__name__, 'error': str(error), 'label_parses': r['label_parses']})
        raise


def verify_models(out: Path) -> dict:
    r = data.read_json(out / 'manifest.json')
    if r['status'] != COMPLETE or set(r['models']) != {'joint_six'}:
        raise ValueError('Incomplete new joint model')
    m = r['models']['joint_six']
    receipt = {'manifest_sha256': data.sha256(out / 'manifest.json'),
               'files': [data.record(data.verify(out / m['path'], m), out)]}
    data.write_json(out / 'model_verification.json', receipt)
    return receipt


def validated_scores(out: Path, c: dict, groups: dict, meta: list, checked: dict) -> tuple[dict, dict, dict]:
    old, old_scores, old_evaluation = origin(c, groups, meta, checked)
    if (out / 'failure.json').exists():
        raise ValueError('Failed run cannot be evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != COMPLETE or r['config'] != c or r['source_files'] != sources()
            or r['physical_updates'] != 1080 or set(r['points']) != set(NAMES)
            or set(r['training']) != {'joint', 'joint_six'} or set(r['train_metrics']) != set(NAMES)
            or set(r['models']) != {'joint_six'}
            or r['label_parses'] != {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0}
            or r['original_replays'] != dict.fromkeys((n for n in NAMES if n != 'joint_six'), True)
            or r['initial_state_sha256'] != old['points']['initial']['checkpoint']['state_sha256']
            or r['inputs'] != old['inputs'] or r['pretrained'] != old['pretrained']
            or r['scored_model_groups'] != 840 or r['group_forwards_with_replay'] != 1680
            or not 0 <= r['budget']['elapsed_seconds'] <= c['runtime']['maximum_gpu_stage_seconds']
            or not 0 <= r['budget']['peak_observed_bytes'] <= c['runtime']['maximum_output_bytes']):
        raise ValueError('Incomplete new joint execution contract')
    expected_models = {name: {**{k: m[k] for k in ('bytes', 'sha256')},
        'path': (Path(c['origin_run']) / m['path']).as_posix()}
        for name, m in old['models'].items() if name in NAMES}
    if r['original_model_files'] != expected_models:
        raise ValueError('Original inference source mapping differs')
    schedule, seed = joint_schedule(groups['train'], meta, old['config'])
    for name, start in (('joint', 0), ('joint_six', 540)):
        log = r['training'][name]
        ids = [g.uid for g in schedule[start:start + 540]]
        if (log['updates'] != 540 or log['group_ids'] != ids or log['dropout_stream'] != seed
                or log['dropout_index_start'] != start or log['first_adam_step'] != start + 1
                or log['last_adam_step'] != start + 540 or log['pair_presentations'] != 540 * 378
                or log['order_sha256'] != hashlib.sha256(data.json_bytes(ids)).hexdigest()
                or log['first_update_modules'] != {k: {'finite_nonzero_gradient': True, 'parameters_changed': True} for k in ('encoder', 'head')}
                or len(log['bce_by_epoch']) != 3 or not np.isfinite(log['bce_by_epoch']).all()
                or not np.isfinite(log['training_seconds']) or log['training_seconds'] < 0):
            raise ValueError('Joint training schedule/update evidence differs')
    m = r['models']['joint_six']
    receipt = data.read_json(out / 'model_verification.json')
    if (m['path'] != 'models/joint_six.pt' or not m['actual_loaded_model_equals_replayed_state']
            or receipt != {'manifest_sha256': data.sha256(out / 'manifest.json'),
                           'files': [{k: m[k] for k in ('path', 'bytes', 'sha256')}]}):
        raise ValueError('New inference model reload/hash receipt missing')
    scores, risks = {}, {}
    for name in NAMES:
        p = r['points'][name]
        subset = train_groups(name, groups['train'], meta)
        if not p['full_model_and_adam_reloaded'] or set(p['scores']) != {'train', 'development'}:
            raise ValueError('Full checkpoint/score replay missing')
        for split, count in (('train', len(subset)), ('development', 60)):
            f = p['scores'][split]
            if f['path'] != f'scores/{name}_{split}.npy':
                raise ValueError('Score filename/point mismatch')
            a = np.load(data.verify(out / f['path'], f), allow_pickle=False)
            if a.shape != (count, 378) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Incomplete score array')
            if split == 'development':
                scores[name] = a
        if name != 'joint_six':
            check_original_point(name, p, old, scores[name], old_scores[name])
        t = r['train_metrics'][name]
        if (t['columns'] != list(TRAIN_COLUMNS) or t['group_ids'] != [g.uid for g in subset]
                or t['file']['path'] != f'scores/{name}_train_metrics.npy'):
            raise ValueError('Static training-risk identities differ')
        a = np.load(data.verify(out / t['file']['path'], t['file']), allow_pickle=False)
        if (a.shape != (len(subset), 2) or a.dtype != np.float64 or not np.isfinite(a).all()
                or np.any(a < 0) or np.any(a[:, 1] > 1)):
            raise ValueError('Static training-risk values invalid')
        risks[name] = a
    # Pinned saved metrics, no second read of original valid supervision.
    parent = (data.ROOT / c['origin_evaluation']).parent
    for name in NAMES:
        if name != 'joint_six':
            f = old_evaluation['points'][name]['file']
            data.verify(parent / f['path'], f)
    return scores, risks, r


def summarize(matrix: np.ndarray, count: int) -> dict:
    if matrix.shape[0] != 3 * count:
        raise ValueError('Domain ordering differs')
    return {d: matrix[i * count:(i + 1) * count].mean(0).tolist() for i, d in enumerate('ABC')}


def compare(arrays: dict, draws: np.ndarray) -> dict:
    result = {}
    for i, column in enumerate(metrics.COLUMNS):
        single = np.stack([arrays[name][j * 20:(j + 1) * 20, i] for j, name in enumerate(NAMES[2:])])
        three = arrays['joint'][:, i].reshape(3, 20)
        six = arrays['joint_six'][:, i].reshape(3, 20)
        result[column] = {key: pilot.paired_summary(delta, draws) for key, delta in
            {'six_minus_three': six - three, 'three_minus_single': three - single,
             'six_minus_single': six - single}.items()}
    return result


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != 'Windows' or destination.exists():
        raise ValueError('Requires Windows and a new evaluation directory')
    c, old_c = contract(), pilot.contract()
    groups, meta, checked = pilot.public_inputs(old_c)
    scores, risks, run_record = validated_scores(out, c, groups, meta, checked)
    destination.mkdir(parents=True)
    data.write_json(destination / 'access.json', {'status': 'COMPLETE_GATE_PASSED_VALID_PARSE_STARTING',
                    'development_parse_attempts': 1, 'train': 0, 'heldout': 0, 'owners': 0})
    labelled = pilot.attach_labels(groups['development'], old_c, 'development')
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    arrays, points = {}, {}
    original = data.read_json(data.ROOT / c['origin_evaluation'])
    for name in NAMES:
        rows = [metrics.classification(y, x) for y, x in zip(truth, scores[name])]
        matrix = np.column_stack((np.asarray([[r[k] for k in metrics.CLASS_KEYS] for r in rows]),
                                  metrics.retrieval(truth, scores[name], 28)))
        if name != 'joint_six':
            f = original['points'][name]['file']
            previous = np.load((data.ROOT / c['origin_evaluation']).parent / f['path'], allow_pickle=False)
            if not np.array_equal(matrix, previous):
                raise ValueError('Replayed original valid metrics differ; no retry')
        arrays[name] = matrix
        file = destination / f'{name}_metrics.npy'
        np.save(file, matrix, allow_pickle=False)
        points[name] = {'file': data.record(file, destination), 'confusion': [r['confusion'] for r in rows],
                       'by_domain': {d: dict(zip(metrics.COLUMNS, values)) for d, values in summarize(matrix, 20).items()}}
    e = old_c['evaluation']
    draws = np.random.default_rng(e['bootstrap_seed']).integers(0, 20, size=(e['bootstrap_replicates'], 3, 20))
    comparisons = compare(arrays, draws)
    train_summary = {n: (summarize(v, 60) if n in NAMES[:2] else {n[0]: v.mean(0).tolist()}) for n, v in risks.items()}
    result = {'status': 'JOINT_DIAGNOSIS_VALID_EVALUATED_INTERPRETATION_REQUIRED',
        'score_manifest_sha256': data.sha256(out / 'manifest.json'), 'config': c,
        'columns': list(metrics.COLUMNS), 'points': points, 'comparisons': comparisons,
        'joint_compatibility': {label: bool(np.all(np.asarray(comparisons['average_precision'][key]['by_domain_or_first_domain']) >= e['joint_single_floor']))
            for label, key in (('three_epochs', 'three_minus_single'), ('six_epochs', 'six_minus_single'))},
        'train_columns': list(TRAIN_COLUMNS), 'static_train_by_domain': train_summary,
        'online_training_bce_by_epoch': {k: v['bce_by_epoch'] for k, v in run_record['training'].items()},
        'label_parses': {'development': 1, 'train': 0, 'heldout': 0, 'owners': 0},
        'bootstrap': {'replicates': 5000, 'seed': 20260910, 'confidence_level': .95,
                      'unit': 'same independent full groups paired within domain; fixed training path'},
        'interpretation': 'No retrospective change to pilot qualification; no method benefit, unique cause or impossibility claim.'}
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
    print(data.json_bytes({'status': result.get('status', 'MODELS_VERIFIED'), 'out': str(args.out)}).decode())


if __name__ == '__main__':
    main()
