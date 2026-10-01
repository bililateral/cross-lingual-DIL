"""Approved fixed-model replay diagnostic; no updates or Linux label parsing."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable

import numpy as np

import step28_continual_replay as replay
from step28_continual_replay import core, data, metrics, pilot

POLICY = data.ROOT / 'schema/step28_continual_replay_probe_policy.json'
COMPLETE = 'COMPLETE_NINE_FIXED_MODELS_1080_FORWARDS_TRAIN_BLIND'
ROLES = ('shared', 'sequential', 'er')
NAMES = tuple(f'{o}_{r}' for o in pilot.ORDERS for r in ROLES)


def contract() -> dict:
    c = data.read_json(POLICY)
    if (c['study'] != 'seller_alias_replay_fixed_model_diagnostic'
            or c['orders'] != list(pilot.ORDERS) or c['roles'] != list(ROLES)
            or [c[k] for k in ('first_domain_train_groups', 'ever_replayed_groups',
                              'never_replayed_groups', 'group_forwards', 'optimizer_updates')]
            != [60, 6, 54, 1080, 0]
            or c['label_parses'] != {'linux': 0, 'windows_train': 1, 'development': 0,
                                      'heldout': 0, 'owners': 0}
            or c['runtime'] != {'maximum_gpu_stage_seconds': 3600,
                                'maximum_output_bytes': 256 * 1024**2, 'torch_cpu_threads': 1}):
        raise ValueError('Diagnostic differs from the approved scope')
    return c


def sources() -> list[dict]:
    return replay.sources() + [data.record(p, data.ROOT) for p in (
        Path(__file__), POLICY,
        data.ROOT / 'tests/test_step28_continual_replay_probe_contracts.py',
        data.ROOT / 'scripts/run_step28_replay_probe_linux_20260915.sh')]


def partition(ids: list[str], initially_kept: list[str], finally_kept: list[str]) -> dict:
    """An evicted old group was still replayed; never classify it as unexposed."""
    universe, ever, final = set(ids), set(initially_kept), set(finally_kept)
    if len(ids) != 60 or len(universe) != 60 or len(initially_kept) != 6 or len(ever) != 6 or not ever <= universe:
        raise ValueError('Expected 60 unique first-domain groups and six initially kept groups')
    if not (final & universe) <= ever:
        raise ValueError('A first-domain group cannot enter later from the new domain')
    return {'ever_replayed': [i for i, uid in enumerate(ids) if uid in ever],
            'never_replayed': [i for i, uid in enumerate(ids) if uid not in ever],
            'retained_until_end': [i for i, uid in enumerate(ids) if uid in final],
            'evicted_after_replay': [i for i, uid in enumerate(ids) if uid in ever - final]}


def context() -> dict:
    c, pc, rc = contract(), pilot.contract(), replay.contract()
    root = data.ROOT / c['replay_run']
    if data.sha256(root / 'manifest.json') != c['replay_manifest_sha256']:
        raise ValueError('Pinned replay run differs')
    groups, meta, checked = pilot.public_inputs(pc)  # Public text/IDs only.
    _, old_arrays, run = replay.validated_scores(root, rc, groups, meta, checked)
    old_root = data.ROOT / rc['origin_run']
    old = data.read_json(old_root / 'manifest.json')
    ep = data.ROOT / c['replay_evaluation']
    if data.sha256(ep) != c['replay_evaluation_sha256']:
        raise ValueError('Pinned replay evaluation differs')
    evaluation = data.read_json(ep)
    if (evaluation['score_manifest_sha256'] != c['replay_manifest_sha256']
            or evaluation['columns'] != list(metrics.COLUMNS)):
        raise ValueError('Saved valid metrics mapped to the wrong run')
    registry, selected, partitions, valid_metrics = {}, {}, {}, {}
    for order in pilot.ORDERS:
        selected[order] = replay.current_groups(groups['train'], meta, order[0])
        ids = [g.uid for g in selected[order]]
        partitions[order] = partition(ids, run['memory'][order + '_shared']['retained_groups'],
                                      run['memory'][order + '_er_stage2']['retained_groups'])
        draws = Counter(uid for stage in (2, 3)
                        for uid in run['training'][f'{order}_er_stage{stage}']['replay_group_ids'])
        partitions[order]['replay_counts'] = [draws[uid] for uid in ids]
        for role, name in (('shared', order + '_shared'), ('sequential', order + '_stage3'),
                           ('er', order + '_er_stage3')):
            src, folder = (run, root) if role == 'er' else (old, old_root)
            model, point = src['models'][name], src['points'][name]
            key = order + '_' + role
            registry[key] = {'model': {**model, 'path': (folder / model['path']).relative_to(data.ROOT).as_posix()},
                             'point_name': name, 'model_state_sha256': point['model_state_sha256'],
                             'valid': {**point['scores']['development'],
                                       'path': (folder / point['scores']['development']['path']).relative_to(data.ROOT).as_posix()}}
            if role == 'er':
                f = evaluation['points'][name]['file']
                a = np.load(data.verify(ep.parent / f['path'], f), allow_pickle=False)
            else:
                a = old_arrays[name]
            if a.shape != (60, 22) or a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError('Saved valid metrics incomplete')
            valid_metrics[key] = a
    return {'config': c, 'model_config': pc, 'groups': groups, 'selected': selected,
            'registry': registry, 'partitions': partitions, 'valid_metrics': valid_metrics,
            'inputs': run['inputs']}


def score_one(model: Any, spec: dict, train: list, valid: list,
              config: dict, check: Callable[[], None]) -> tuple[np.ndarray, np.ndarray]:
    path = data.verify(data.ROOT / spec['model']['path'], spec['model'])
    metadata = core.restore_state(path, model, None, spec['model']['state_sha256'])
    if (metadata != {'point': spec['point_name'], 'config': config}
            or core.state_digest(model.state_dict()) != spec['model_state_sha256']):
        raise ValueError('Actual loaded inference model differs')
    original = np.load(data.verify(data.ROOT / spec['valid']['path'], spec['valid']), allow_pickle=False)
    values = core.score(model, valid, config, check)
    if not np.array_equal(values, original):
        raise ValueError('Original full valid scores not exactly replayed')
    train_values = core.score(model, train, config, check)
    if core.state_digest(model.state_dict()) != spec['model_state_sha256']:
        raise ValueError('Inference modified the model')
    return train_values, values


def run(out: Path) -> dict:
    import torch
    c = contract()
    if (platform.system() != 'Linux' or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Requires resumed Linux stage and one eligible idle GPU')
    available = next(int(row.split()[1]) * 1024 for row in
                     Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    if (torch.cuda.mem_get_info()[0] < 12 * 1024**3 or available < 16 * 1024**3
            or shutil.disk_usage(data.ROOT).free < c['runtime']['maximum_output_bytes']):
        raise RuntimeError('Insufficient shared resources; wait')
    if out.exists() or not out.is_relative_to((data.ROOT / 'reports').resolve()):
        raise ValueError('Use a new project reports directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    snapshot = sources()

    def check() -> None:
        if time.monotonic() - started >= 3600:
            raise RuntimeError('Diagnostic time limit reached')
        # Includes Bash logs, unlike the historical run-only accounting.
        if sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) > 256 * 1024**2:
            raise RuntimeError('Diagnostic job output limit reached')

    r = {'status': 'RUNNING', 'config': c, 'source_files': snapshot, 'points': {},
         'optimizer_updates': 0, 'label_parses': 0,
         'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}}
    data.write_json(out / 'startup.json', r)
    try:
        ctx = context()
        r['inputs'], r['partitions'] = ctx['inputs'], ctx['partitions']
        r['train_ids'] = {o: [g.uid for g in gs] for o, gs in ctx['selected'].items()}
        r['valid_ids'] = [g.uid for g in ctx['groups']['development']]
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        pretrained = core.model_files(ctx['model_config'])
        if any(pretrained[k] != ctx['model_config']['model'][k]
               for k in ('file_count', 'total_size_bytes', 'content_sha256')):
            raise ValueError('Original tokenizer/LaBSE files differ')
        model = core.load_model(ctx['model_config'])
        for order in pilot.ORDERS:
            for role in ROLES:
                name = order + '_' + role
                before = time.monotonic()
                a, b = score_one(model, ctx['registry'][name], ctx['selected'][order],
                                 ctx['groups']['development'], ctx['model_config'], check)
                files = {}
                for split, values in (('train', a), ('development', b)):
                    target = out / f'{name}_{split}.npy'
                    np.save(target, values, allow_pickle=False)
                    files[split] = data.record(target, out)
                r['points'][name] = {'source': ctx['registry'][name], 'scores': files,
                                     'original_valid_exact': True, 'model_unchanged': True,
                                     'seconds': time.monotonic() - before}
                data.write_json(out / 'progress.json', r)
                check()
                print(data.json_bytes({'point': name, 'seconds': time.monotonic() - started}).decode(), flush=True)
        if sources() != snapshot:
            raise ValueError('Sources changed')
        r.update(status=COMPLETE, group_forwards=1080, total_seconds=time.monotonic() - started)
        data.write_json(out / 'manifest.json', r)
        check()
        return r
    except Exception as error:
        data.write_json(out / 'failure.json', {'status': 'FAILED_NO_LABELS_OR_AUTO_RESTART',
                        'error': str(error), 'label_parses': 0, 'optimizer_updates': 0})
        raise


def validated_scores(out: Path, ctx: dict) -> tuple[dict, dict]:
    if (out / 'failure.json').exists():
        raise ValueError('Failed diagnostic cannot be evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != COMPLETE or r['config'] != ctx['config'] or r['source_files'] != sources()
            or set(r['points']) != set(NAMES) or r['group_forwards'] != 1080
            or r['optimizer_updates'] != 0 or r['label_parses'] != 0
            or r['inputs'] != ctx['inputs'] or r['partitions'] != ctx['partitions']
            or r['train_ids'] != {o: [g.uid for g in gs] for o, gs in ctx['selected'].items()}
            or r['valid_ids'] != [g.uid for g in ctx['groups']['development']]
            or not 0 <= r['total_seconds'] <= 3600):
        raise ValueError('Incomplete/mismatched diagnostic')
    arrays = {}
    for name in NAMES:
        p = r['points'][name]
        if (p['source'] != ctx['registry'][name] or not p['original_valid_exact']
                or not p['model_unchanged'] or set(p['scores']) != {'train', 'development'}):
            raise ValueError('Missing model/score replay mapping')
        for split in ('train', 'development'):
            f = p['scores'][split]
            if f['path'] != f'{name}_{split}.npy':
                raise ValueError('Score path differs')
            a = np.load(data.verify(out / f['path'], f), allow_pickle=False)
            if a.shape != (60, 378) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Incomplete scores')
            if split == 'train':
                arrays[name] = a
            else:
                original = ctx['registry'][name]['valid']
                b = np.load(data.verify(data.ROOT / original['path'], original), allow_pickle=False)
                if not np.array_equal(a, b):
                    raise ValueError('Original valid scores differ')
    return arrays, r


def summarize(arrays: dict, masks: dict, valid: dict) -> dict:
    result = {}
    columns = (*metrics.COLUMNS, 'binary_cross_entropy')
    for order in pilot.ORDERS:
        subsets = {}
        for subset in ('ever_replayed', 'never_replayed'):
            indices = masks[order][subset]
            rows = {role: arrays[order + '_' + role][indices] for role in ROLES}
            absolute = {role: dict(zip(columns, a.mean(0).tolist())) for role, a in rows.items()}
            changes = {role + '_minus_shared': dict(zip(columns, (rows[role] - rows['shared']).mean(0).tolist()))
                       for role in ('sequential', 'er')}
            subsets[subset] = {'groups': len(indices), 'absolute': absolute, 'changes': changes,
                              'er_minus_seq': dict(zip(columns, (rows['er'] - rows['sequential']).mean(0).tolist()))}
        domain = 'ABC'.index(order[0])
        result[order] = {'train': subsets,
            'valid_first_domain_saved': {role: dict(zip(metrics.COLUMNS,
                valid[order + '_' + role][domain * 20:(domain + 1) * 20].mean(0).tolist())) for role in ROLES}}
    means = {subset: {metric: float(np.mean([result[o]['train'][subset]['er_minus_seq'][metric]
                  for o in pilot.ORDERS])) for metric in columns}
             for subset in ('ever_replayed', 'never_replayed')}
    return {'by_order': result, 'order_equal_er_minus_seq': means,
            'ever_minus_never_contrast': {k: means['ever_replayed'][k] - means['never_replayed'][k] for k in columns},
            'interpretation': 'Descriptive fixed-model training-set comparison, no success gate or unique causal attribution.'}


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != 'Windows' or destination.exists():
        raise ValueError('Requires Windows and a new evaluation directory')
    ctx = context()
    scores, run_record = validated_scores(out, ctx)  # All nine before any new labels.
    destination.mkdir(parents=True)
    data.write_json(destination / 'access.json', {'train_parse_attempts': 1, 'development': 0,
                                               'heldout': 0, 'owners': 0})
    labelled = pilot.attach_labels(ctx['groups']['train'], ctx['model_config'], 'train')
    labels = {g.uid: g.labels for g in labelled}
    arrays, points = {}, {}
    for order in pilot.ORDERS:
        truth = np.asarray([labels[g.uid] for g in ctx['selected'][order]], dtype=np.uint8)
        for role in ROLES:
            name = order + '_' + role
            x = scores[name]
            rows = [metrics.classification(y, v) for y, v in zip(truth, x)]
            # Stable, unclipped BCE at fixed models; distinct from clipped log_loss.
            v = x.astype(np.float64)
            bce = (np.logaddexp(0., v) - truth * v).mean(1)
            a = np.column_stack(([[r[k] for k in metrics.CLASS_KEYS] for r in rows],
                                 metrics.retrieval(truth, x, 28), bce))
            arrays[name] = a
            path = destination / f'{name}_metrics.npy'
            np.save(path, a, allow_pickle=False)
            points[name] = {'file': data.record(path, destination), 'confusion': [r['confusion'] for r in rows]}
    r = {'status': 'FIXED_MODEL_TRAIN_DIAGNOSTIC_EVALUATED_INTERPRETATION_REQUIRED',
         'config': ctx['config'], 'manifest_sha256': data.sha256(out / 'manifest.json'),
         'columns': [*metrics.COLUMNS, 'binary_cross_entropy'], 'points': points,
         'partitions': ctx['partitions'], 'train_ids': run_record['train_ids'],
         'comparison': summarize(arrays, ctx['partitions'], ctx['valid_metrics']),
         'label_parses': {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0}}
    data.write_json(destination / 'evaluation.json', r)
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('score', 'evaluate'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    if args.action == 'evaluate' and args.evaluation is None:
        parser.error('evaluate requires --evaluation')
    r = run(args.out.resolve()) if args.action == 'score' else evaluate(args.out.resolve(), args.evaluation.resolve())
    print(data.json_bytes({'status': r['status']}).decode())


if __name__ == '__main__':
    main()
