"""Two approved weight sensitivities; one shared supervision parse per stage.

Frozen predecessors remain unchanged. Arm subdirectories have independent
models, optimizers and teacher memories; only authorized train supervision is
parsed centrally once. No automatic Linux launch or partial evaluation.
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

import step28_continual_distillation as distill
from step28_continual_distillation import baseline, pilot, core, data, metrics, persistence

POLICY = data.ROOT / 'schema/step28_continual_sensitivity_policy.json'
COMPLETE = 'COMPLETE_WEIGHT_SENSITIVITY_3240_BOTH_ARMS_VALID_BLIND'
ARM_COMPLETE = 'COMPLETE_WEIGHT_SENSITIVITY_ARM_1620_VALID_BLIND'
ARMS = ('weaker_replay', 'weaker_distillation')
WEIGHTS = {'weaker_replay': {'alpha': .5, 'beta': .5},
           'weaker_distillation': {'alpha': 1., 'beta': .25}}
NAMES = distill.NAMES
ReplayState = distill.ReplayState
current_groups = distill.current_groups
check_shared = distill.check_shared
populate_targets = distill.populate_targets
save_memory = distill.save_memory
expected_history = distill.expected_history
origin = distill.origin


def contract() -> dict:
    c = data.read_json(POLICY)
    expected = distill.contract()
    expected.update(study='seller_alias_expression_weight_sensitivity', physical_updates=3240,
        configurations=WEIGHTS,
        distillation_reference={
            'run': 'reports/seller_alias_continual/20260915/distillation_execution/20260915_164430/job/run',
            'manifest_sha256': '61b1e972e282508386bacb60d05d8c2d13ed24594b08e4c1b1c1c2bc2f0a9721',
            'evaluation': 'reports/seller_alias_continual/20260915/distillation_evaluation/20260915_181330/evaluation.json',
            'evaluation_sha256': '2590dd75f6a343e4ed68b61bb0f75097f583b960e11c850f5b3e26a841fc3741'},
        boundary=c.get('boundary'))
    expected['runtime'].update(maximum_gpu_stage_seconds=18000, maximum_output_bytes=24 * 1024**3)
    del expected['memory']['replay_coefficient']
    del expected['distillation']['coefficient']
    if c != expected:
        raise ValueError('Sensitivity differs from the two adopted configurations')
    return c


def sources() -> list[dict]:
    return distill.sources() + [data.record(p, data.ROOT) for p in (
        Path(__file__), POLICY, data.ROOT / 'tests/test_step28_continual_sensitivity_contracts.py',
        data.ROOT / 'scripts/run_step28_sensitivity_linux_20260916.sh')]


def reference_distillation(c: dict, groups: dict, meta: list, checked: dict) -> dict:
    ref = c['distillation_reference']
    root, path = data.ROOT / ref['run'], data.ROOT / ref['evaluation']
    if (data.sha256(root / 'manifest.json') != ref['manifest_sha256']
            or data.sha256(path) != ref['evaluation_sha256']):
        raise ValueError('Pinned original distillation evidence differs')
    _, original, run_record = distill.validated_scores(root, distill.contract(), groups, meta, checked)
    e = data.read_json(path)
    if (e['score_manifest_sha256'] != ref['manifest_sha256'] or e['config'] != run_record['config']
            or e['columns'] != list(metrics.COLUMNS) or set(e['points']) != set(NAMES)):
        raise ValueError('Original distillation evaluation mapping differs')
    result = {'initial': original['initial']}
    for name, point in e['points'].items():
        a = np.load(data.verify(path.parent / point['file']['path'], point['file']), allow_pickle=False)
        if a.shape != (60, len(metrics.COLUMNS)) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('Original distillation matrix differs')
        if name.endswith('_shared') and not np.array_equal(a, original[name]):
            raise ValueError('Original distillation shared metrics differ')
        result[name.replace('_distill_stage', '_stage')] = a
    return result


def historical_loss(predicted: Any, truth: Any, target: Any, alpha: float, beta: float) -> tuple:
    import torch
    if (predicted.shape != truth.shape or target.shape != predicted.shape
            or predicted.ndim != 1 or target.requires_grad
            or not np.isfinite(alpha) or alpha < 0 or not np.isfinite(beta) or beta < 0
            or not torch.isfinite(predicted).all() or not torch.isfinite(target).all()):
        raise ValueError('Nonfinite, attached or misaligned historical objective')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(predicted, truth, reduction='mean')
    mse = (predicted - target).square().mean()
    loss = alpha * bce + beta * mse
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite historical loss')
    return loss, bce, mse


def observed_update(model: Any, optimizer: Any, current: data.Group, replay: data.Group,
                    config: dict, seed: int, replay_seed: int, check: Any,
                    targets: np.ndarray, *, observe: bool = True, alpha: float, beta: float) -> dict:
    """One current BCE backward, one shared BCE+MSE backward, one clip/update."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    branches, losses = {}, {}
    before = {n: core.state_digest(m.state_dict()) for n, m in (('encoder', model.encoder), ('head', model.head))} if observe else {}
    mse_value = None
    for label, group, stream in (('current', current, seed), ('replay', replay, replay_seed)):
        squared = {'encoder': 0., 'head': 0.}
        handles = []

        def observe_gradient(gradient: Any, name: str) -> None:
            squared[name] += float(gradient.detach().float().square().sum())

        try:
            for name, module in (('encoder', model.encoder), ('head', model.head)) if observe else ():
                for p in module.parameters():
                    if p.requires_grad:
                        handles.append(p.register_hook(lambda g, n=name: observe_gradient(g, n)))
            if label == 'current':
                losses[label] = core.backward_group(model, group, config, stream, check)
            else:
                if group.labels is None:
                    raise ValueError('Historical supervision missing')
                torch.manual_seed(stream)
                predicted = core.logits(model, group, config, check)
                truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
                target = torch.tensor(targets, dtype=torch.float32, device=predicted.device)
                loss, bce, mse = historical_loss(predicted, truth, target, alpha, beta)
                loss.backward()
                losses[label], mse_value = float(bce.detach()), float(mse.detach())
                del predicted, truth, target, loss, bce, mse
        finally:
            for handle in handles:
                handle.remove()
        if observe and any(not np.isfinite(v) or v <= 0 for v in squared.values()):
            raise ValueError('A current/replay encoder/head branch has no finite nonzero gradient')
        # This is the sum over gradient-hook events, not the final accumulated norm.
        branches[label] = squared
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_norm'], error_if_nonfinite=True)
    check()
    optimizer.step()
    if observe and any(before[n] == core.state_digest(m.state_dict()) for n, m in (('encoder', model.encoder), ('head', model.head))):
        raise ValueError('ER update did not change both encoder and head')
    return {'current_bce': losses['current'], 'replay_bce': losses['replay'], 'replay_mse': mse_value,
        'gradient_norm_before_clip': float(norm), 'branches_squared_hook_norm': branches,
        'encoder_and_head_changed': True}


def train_replay(model: Any, optimizer: Any, current: list, state: ReplayState,
                 old_c: dict, budget: persistence.Budget, weights: dict) -> dict:
    # Only the current domain and own bounded state enter this training function.
    if len(current) != 60 or state.index != 0 or {g.uid for g in current} & {g.uid for g in state.memory.groups}:
        raise ValueError('Current/history supply boundary differs')
    started = time.monotonic()
    seed = data.seed_for(20260909, state.order, state.stage, 'current')
    rows = data.schedule(current, 3, seed)
    memory_before = state.memory.to_bytes()
    losses, histories, norms, ids, mses, branch = [], [], [], [], [], None
    targets_before = data.json_bytes(state.targets)
    prior = 180 * (state.stage - 1)
    maximum = len(state.to_bytes())
    for i, group in enumerate(rows):
        replay = state.draw()
        maximum = max(maximum, len(state.to_bytes()))
        update = observed_update
        log = update(model, optimizer, group, replay, old_c,
            data.seed_for(seed, i, 'dropout_current'), data.seed_for(seed, i, 'dropout_replay'), budget.check,
            state.target(replay), observe=(i == 0), **weights)
        losses.append(log['current_bce'])
        histories.append(log['replay_bce'])
        mses.append(log['replay_mse'])
        norms.append(log['gradient_norm_before_clip'])
        ids.append(replay.uid)
        if i == 0:
            branch = {k: log[k] for k in ('branches_squared_hook_norm', 'encoder_and_head_changed')}
        if i in (0, 179) and (not optimizer.state or any(float(v['step']) != prior + i + 1 for v in optimizer.state.values())):
            raise ValueError('ER Adam continuity differs')
        if (i + 1) % 30 == 0:
            print(data.json_bytes({'event': 'replay_updates', 'order': state.order,
                'stage': state.stage, 'completed': i + 1, **budget.state()}).decode(), flush=True)
    if state.memory.to_bytes() != memory_before or data.json_bytes(state.targets) != targets_before:
        raise ValueError('Reservoir changed during training stage')
    return {'updates': 180, 'group_ids': [g.uid for g in rows], 'dropout_stream': seed,
        'replay_group_ids': ids, 'first_adam_step': prior + 1, 'last_adam_step': prior + 180,
        'current_pair_presentations': 68040, 'replay_pair_presentations': 68040,
        'current_bce_by_epoch': np.asarray(losses).reshape(3, 60).mean(1).tolist(),
        'replay_bce_by_epoch': np.asarray(histories).reshape(3, 60).mean(1).tolist(),
        'replay_mse_by_epoch': np.asarray(mses).reshape(3, 60).mean(1).tolist(),
        'replay_coefficient': weights['alpha'], 'distillation_coefficient': weights['beta'],
        'targets_unchanged': True,
        'maximum_gradient_norm_before_clip': max(norms), 'first_update': branch,
        'maximum_history_bytes': maximum, 'final_index': state.index,
        'final_state_sha256': hashlib.sha256(state.to_bytes()).hexdigest(),
        'reservoir_unchanged': True, 'training_seconds': time.monotonic() - started}


def run_arm(out: Path, arm: str, c: dict, old_c: dict, groups: dict,
            meta: list, old: dict, old_scores: dict, pretrained: dict,
            budget: persistence.Budget, snapshot: list, torch_tests: dict) -> dict:
    import torch
    if arm not in ARMS or out.exists():
        raise ValueError('Unknown/reused sensitivity arm')
    out.mkdir()
    for directory in ('work', 'scores', 'models', 'memory'):
        (out / directory).mkdir()
    r = {'status': 'RUNNING', 'config': c, 'arm': arm, 'weights': WEIGHTS[arm],
         'source_files': snapshot, 'points': {}, 'training': {}, 'models': {}, 'memory': {}, 'starts': {},
         'inputs': old['inputs'], 'pretrained': pretrained, 'torch_contracts': torch_tests,
         'label_parses': {'train': 0, 'development': 0, 'heldout': 0, 'owners': 0},
         'supervision_source': 'parent_stage_once_only_train_parse'}
    for order in pilot.ORDERS:
        model = core.load_model(old_c)
        initial = core.state_digest({'model': model.state_dict(), 'optimizer': None,
            'metadata': {'point': 'initial', 'config': old_c}})
        if initial != old['points']['initial']['checkpoint']['state_sha256']:
            raise ValueError('Original initialization differs')
        r['starts'][order] = initial
        optimizer = core.make_optimizer(model, old_c)
        memory = data.Memory(data.seed_for(20260909, order, 'memory'), 6, 1048576)
        state = None
        for stage in (1, 2, 3):
            current = current_groups(groups['train'], meta, order[stage - 1])
            name = f'{order}_shared' if stage == 1 else f'{order}_distill_stage{stage}'
            if stage == 1:
                r['training'][name] = pilot.train_segment(model, optimizer, current, meta, old_c, order, stage, budget)
            else:
                r['training'][name] = train_replay(model, optimizer, current, state, old_c, budget, WEIGHTS[arm])
                memory = state.memory
            p = persistence.point(model, optimizer, name, {'development': groups['development']}, old_c, out, budget)
            budget.check(1)
            r['points'][name] = p
            if stage == 1:
                check_shared(name, p, old, np.load(out / p['scores']['development']['path'], allow_pickle=False), old_scores[name])
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
                optimizer = core.make_optimizer(model, old_c)
                metadata = core.restore_state(out / p['checkpoint']['path'], model, optimizer, p['checkpoint']['state_sha256'])
                if metadata != {'point': name, 'config': old_c}:
                    raise ValueError('Shared continuation metadata differs')
                r['starts'][order + '_continuation'] = p['checkpoint']['state_sha256']
            if stage < 3:
                state = ReplayState(memory, order, stage + 1, state.targets if state else None)
                target_receipt = populate_targets(model, state, current, name, old_c, budget)
                state, receipt = save_memory(state, current, name, out, budget)
                receipt['teacher'] = target_receipt
                memory = state.memory
                r['memory'][name] = receipt
            else:
                r['models'][name] = persistence.retain_inference(model, name, p, old_c, out, budget)
                budget.check(1)  # Observe checkpoint/inference coexistence before deletion.
            persistence.remove_work_file(out / p['checkpoint']['path'], out)
            data.write_json(out / 'progress.json', r)
        del model, optimizer, memory, state, current
        gc.collect()
        torch.cuda.empty_cache()

    r.update(status=ARM_COMPLETE, physical_updates=sum(t['updates'] for t in r['training'].values()),
        formal_training_seconds=sum(t['training_seconds'] for t in r['training'].values()),
        inference_seconds_including_replay=sum(p['timing']['inference_seconds_including_replay'] for p in r['points'].values()),
        group_forwards_with_replay=1080,
        teacher_group_forwards=sum(m['teacher']['group_forwards'] for m in r['memory'].values()),
        teacher_seconds=sum(m['teacher']['seconds'] for m in r['memory'].values()), budget=budget.state())
    if r['physical_updates'] != 1620 or r['teacher_group_forwards'] != 25:
        raise ValueError('Arm physical/teacher count differs')
    data.write_json(out / 'manifest.json', r)
    verify_arm_models(out)
    return r


def verify_arm_models(out: Path) -> dict:
    r = data.read_json(out / 'manifest.json')
    if r['status'] != ARM_COMPLETE or set(r['models']) != {f'{o}_distill_stage3' for o in pilot.ORDERS}:
        raise ValueError('Incomplete sensitivity model set')
    receipt = {'manifest_sha256': data.sha256(out / 'manifest.json'),
        'files': [data.record(data.verify(out / m['path'], m), out) for _, m in sorted(r['models'].items())]}
    data.write_json(out / 'model_verification.json', receipt)
    return receipt


def run(out: Path) -> dict:
    import torch
    import sys
    import unittest
    c, old_c = contract(), pilot.contract()
    rt = c['runtime']
    if (platform.system() != 'Linux' or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Requires resumed Linux stage and one eligible visible GPU')
    available = next(int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    if (torch.cuda.mem_get_info()[0] < rt['minimum_free_gpu_bytes']
            or available < rt['minimum_free_host_bytes']
            or shutil.disk_usage(data.ROOT).free < rt['maximum_output_bytes']):
        raise RuntimeError('Insufficient shared resources; wait')
    if out.exists() or not out.is_relative_to((data.ROOT / 'reports').resolve()):
        raise ValueError('Use a new project reports directory')
    out.mkdir(parents=True)
    budget = persistence.Budget(out.parent, c)  # Includes BOTH arms and launcher logs.
    snapshot = sources()
    r = {'status': 'RUNNING', 'config': c, 'source_files': snapshot, 'arms': {},
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
        sys.path.insert(0, str(data.ROOT / 'tests'))
        suite = unittest.defaultTestLoader.loadTestsFromName('test_step28_continual_sensitivity_contracts.TorchContracts')
        test = unittest.TextTestRunner(verbosity=2).run(suite)
        if not test.wasSuccessful() or test.skipped or test.testsRun != 2:
            raise RuntimeError('Changed weighted loss/Adam CPU checks failed')
        r['torch_contracts'] = {'passed': test.testsRun, 'skipped': 0}
        groups, meta, checked = pilot.public_inputs(old_c)
        old, old_scores, _ = origin(c, groups, meta, checked)
        reference_distillation(c, groups, meta, checked)
        distill.reference_baseline(c, groups, meta, checked)
        pretrained = core.model_files(old_c)
        if any(pretrained[k] != old_c['model'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
            raise ValueError('Pretrained bytes differ')
        data.write_json(out / 'access.json', {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0})
        r['label_parses']['train'] = 1
        groups['train'] = pilot.attach_labels(groups['train'], old_c, 'train')
        for arm in ARMS:
            print(data.json_bytes({'event': 'arm_start', 'arm': arm, 'weights': WEIGHTS[arm]}).decode(), flush=True)
            result = run_arm(out / arm, arm, c, old_c, groups, meta, old, old_scores,
                             pretrained, budget, snapshot, r['torch_contracts'])
            r['arms'][arm] = {'manifest': data.record(out / arm / 'manifest.json', out),
                             'physical_updates': result['physical_updates'],
                             'formal_training_seconds': result['formal_training_seconds']}
            data.write_json(out / 'progress.json', r)
        if sources() != snapshot:
            raise ValueError('Sources changed during execution')
        r.update(status=COMPLETE, physical_updates=sum(v['physical_updates'] for v in r['arms'].values()),
                 formal_training_seconds=sum(v['formal_training_seconds'] for v in r['arms'].values()),
                 budget=budget.state())
        if r['physical_updates'] != 3240:
            raise ValueError('Total physical updates differ')
        data.write_json(out / 'manifest.json', r)
        return r
    except Exception as error:
        data.write_json(out / 'failure.json', {'status': 'FAILED_NO_PARTIAL_EVALUATION_OR_AUTO_RESTART',
            'error_type': type(error).__name__, 'error': str(error), 'label_parses': r['label_parses']})
        raise


def validated_scores(out: Path, c: dict, groups: dict, meta: list, checked: dict, arm: str) -> tuple[dict, dict, dict]:
    old, old_scores, old_arrays = origin(c, groups, meta, checked)
    if (out / 'failure.json').exists():
        raise ValueError('Failed run cannot be evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != ARM_COMPLETE or r.get('arm') != arm or r.get('weights') != WEIGHTS[arm] or r['config'] != c or r['source_files'] != sources()
            or r['physical_updates'] != 1620 or set(r['points']) != set(NAMES)
            or set(r['training']) != set(NAMES) or r['group_forwards_with_replay'] != 1080
            or set(r['models']) != {f'{o}_distill_stage3' for o in pilot.ORDERS}
            or set(r['memory']) != {f'{o}_{s}' for o in pilot.ORDERS for s in ('shared', 'distill_stage2')}
            or r['label_parses'] != {'train': 0, 'development': 0, 'heldout': 0, 'owners': 0}
            or r['torch_contracts'] != {'passed': 2, 'skipped': 0}
            or r.get('supervision_source') != 'parent_stage_once_only_train_parse'
            or r['inputs'] != old['inputs'] or r['pretrained'] != old['pretrained']
            or not 0 <= r['budget']['elapsed_seconds'] <= c['runtime']['maximum_gpu_stage_seconds']
            or not 0 <= r['budget']['peak_observed_bytes'] <= c['runtime']['maximum_output_bytes']):
        raise ValueError('Incomplete replay execution contract')
    starts = dict.fromkeys(pilot.ORDERS, old['points']['initial']['checkpoint']['state_sha256'])
    starts.update({o + '_continuation': old['points'][o + '_shared']['checkpoint']['state_sha256'] for o in pilot.ORDERS})
    if r['starts'] != starts:
        raise ValueError('Shared full Adam continuation differs')
    receipt = data.read_json(out / 'model_verification.json')
    if receipt != {'manifest_sha256': data.sha256(out / 'manifest.json'),
            'files': [{k: m[k] for k in ('path', 'bytes', 'sha256')} for _, m in sorted(r['models'].items())]}:
        raise ValueError('Retained model hash receipt missing')
    for name, m in r['models'].items():
        if m['path'] != f'models/{name}.pt' or not m['actual_loaded_model_equals_replayed_state']:
            raise ValueError('Retained inference mapping/reload differs')
    scores = {}
    teacher_count = 0
    public_groups = {g.uid: g for g in groups['train']}
    for order in pilot.ORDERS:
        history = expected_history(meta, order)
        previous_targets = {}
        for stage in (1, 2, 3):
            name = f'{order}_shared' if stage == 1 else f'{order}_distill_stage{stage}'
            t = r['training'][name]
            rows, seed = pilot.segment_schedule(groups['train'], meta, old['config'], order, stage)
            if (t['updates'] != 180 or t['group_ids'] != [g.uid for g in rows] or t['dropout_stream'] != seed
                    or t['first_adam_step'] != (stage - 1) * 180 + 1 or t['last_adam_step'] != stage * 180
                    or not np.isfinite(t['training_seconds']) or t['training_seconds'] < 0):
                raise ValueError('Current schedule/Adam count differs')
            if stage == 1:
                # Ignore timings only; all recorded losses/digests/update evidence must agree.
                for key in old['training'][name]:
                    if key != 'training_seconds' and t[key] != old['training'][name][key]:
                        raise ValueError('Shared training replay differs')
            else:
                branch = t['first_update']
                if (t['replay_group_ids'] != history[stage - 1]['replay_group_ids']
                        or not t['reservoir_unchanged'] or t['final_index'] != 180
                        or t['current_pair_presentations'] != 68040 or t['replay_pair_presentations'] != 68040
                        or not 0 < t['maximum_history_bytes'] <= 1048576
                        or not branch['encoder_and_head_changed']
                        or set(branch['branches_squared_hook_norm']) != {'current', 'replay'}):
                    raise ValueError('Replay supply/loss/bounded-state evidence differs')
                for norms in branch['branches_squared_hook_norm'].values():
                    if set(norms) != {'encoder', 'head'} or any(not np.isfinite(v) or v <= 0 for v in norms.values()):
                        raise ValueError('Separate branch gradient evidence missing')
                if (t.get('replay_coefficient') != WEIGHTS[arm]['alpha']
                        or t['distillation_coefficient'] != WEIGHTS[arm]['beta'] or not t['targets_unchanged']):
                    raise ValueError('Historical target/loss contract differs')
                for key in ('current_bce_by_epoch', 'replay_bce_by_epoch', 'replay_mse_by_epoch'):
                    if len(t[key]) != 3 or not np.isfinite(t[key]).all() or min(t[key]) < 0:
                        raise ValueError('Recorded BCE differs')
            p = r['points'][name]
            if (not p['full_model_and_adam_reloaded'] or set(p['scores']) != {'development'}
                    or p['checkpoint']['path'] != f'work/{name}.pt'):
                raise ValueError('Full-state score replay missing')
            f = p['scores']['development']
            if f['path'] != f'scores/{name}_development.npy':
                raise ValueError('Score path differs')
            a = np.load(data.verify(out / f['path'], f), allow_pickle=False)
            if a.shape != (60, 378) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Incomplete blind scores')
            scores[name] = a
            if stage == 1:
                check_shared(name, p, old, a, old_scores[name])
            if stage < 3:
                m, expected = r['memory'][name], history[stage]
                if (any(m[k] != v for k, v in expected.items() if k != 'replay_group_ids')
                        or m['next_draw'] != expected['replay_group_ids'][0]
                        or m['file']['path'] != f'memory/{name}.json' or not m['disk_roundtrip_exact']
                        or m['file']['bytes'] != m['full_state_bytes'] or not 0 < m['full_state_bytes'] <= 1048576):
                    raise ValueError('Memory persistence/membership receipt differs')
                data.verify(out / m['file']['path'], m['file'])  # Hash bytes only; never deserialize train labels.
                teacher = m['teacher']
                expected_new = [uid for uid in expected['retained_groups'] if uid in expected['added_groups']]
                targets = teacher['targets']
                if (teacher['new_groups'] != expected_new or teacher['group_forwards'] != len(expected_new)
                        or not teacher['model_state_unchanged']
                        or not np.isfinite(teacher['seconds']) or teacher['seconds'] < 0
                        or not m['full_state_bytes'] <= teacher['maximum_insertion_history_bytes'] <= 1048576
                        or set(targets) != set(expected['retained_groups'])):
                    raise ValueError('Teacher generation/byte accounting differs')
                for uid, target in targets.items():
                    if uid in expected_new:
                        if (target['point'] != name or target['model_state_sha256'] != p['model_state_sha256']
                                or target['pair_order_sha256'] != hashlib.sha256(data.json_bytes(public_groups[uid].sellers)).hexdigest()
                                or len(target['sha256']) != 64):
                            raise ValueError('Teacher source or pair mapping differs')
                    elif target != previous_targets[uid]:
                        raise ValueError('Old teacher target refreshed')
                previous_targets = targets
                teacher_count += teacher['group_forwards']
    if r['teacher_group_forwards'] != teacher_count or teacher_count != 25:
        raise ValueError('Teacher scoring total differs')
    return scores, old_arrays, r


def validated_stage(out: Path, c: dict, groups: dict, meta: list, checked: dict) -> tuple:
    if (out / 'failure.json').exists():
        raise ValueError('Failed stage cannot be partially evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != COMPLETE or r['config'] != c or r['source_files'] != sources()
            or set(r['arms']) != set(ARMS) or r['physical_updates'] != 3240
            or r['label_parses'] != {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0}
            or r['torch_contracts'] != {'passed': 2, 'skipped': 0}
            or not 0 <= r['budget']['elapsed_seconds'] <= c['runtime']['maximum_gpu_stage_seconds']
            or not 0 <= r['budget']['peak_observed_bytes'] <= c['runtime']['maximum_output_bytes']):
        raise ValueError('Incomplete two-arm sensitivity stage')
    if data.read_json(out / 'access.json') != {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0}:
        raise ValueError('Stage supervision record differs')
    scores, records = {}, {}
    for arm in ARMS:
        f = r['arms'][arm]['manifest']
        if f['path'] != f'{arm}/manifest.json':
            raise ValueError('Arm manifest mapping differs')
        data.verify(out / f['path'], f)
        scores[arm], original, records[arm] = validated_scores(out / arm, c, groups, meta, checked, arm)
        if (r['arms'][arm]['physical_updates'] != records[arm]['physical_updates']
                or r['arms'][arm]['formal_training_seconds'] != records[arm]['formal_training_seconds']):
            raise ValueError('Arm update/time receipt differs')
    if (sum(v['physical_updates'] for v in records.values()) != r['physical_updates']
            or sum(v['formal_training_seconds'] for v in records.values()) != r['formal_training_seconds']):
        raise ValueError('Stage update/training-time total differs')
    return scores, original, r


def compare(arrays: dict, reference: dict, draws: np.ndarray, c: dict, reference_role: str) -> dict:
    if reference_role not in ('sequential', 'random_er', 'original_distillation'):
        raise ValueError('Unknown sensitivity comparison role')
    remapped = {name.replace('_distill_stage', '_er_stage'): a for name, a in arrays.items()}
    result = baseline.compare(remapped, reference, draws, c)
    names = {'G_er': 'G_candidate', 'F_er': 'F_candidate', 'second_er_change': 'second_candidate_change',
             'G_er_transitions': 'G_candidate_transitions', 'er': 'candidate',
             'G_seq': 'G_reference', 'F_seq': 'F_reference', 'second_seq_change': 'second_reference_change',
             'G_seq_transitions': 'G_reference_transitions', 'sequential': 'reference',
             'final_er_minus_seq': 'final_candidate_minus_reference'}

    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {names.get(k, k): rename(v) for k, v in value.items()}
        return value

    result = rename(result)
    result['reference_role'] = reference_role
    if reference_role != 'sequential':
        del result['gate']  # The approved three gates are relative to SEQ only.
    return result


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != 'Windows' or destination.exists():
        raise ValueError('Requires Windows and a new evaluation directory')
    c, old_c = contract(), pilot.contract()
    groups, meta, checked = pilot.public_inputs(old_c)
    scores, original, run_record = validated_stage(out, c, groups, meta, checked)
    references = {'sequential': original,
                  'random_er': distill.reference_baseline(c, groups, meta, checked),
                  'original_distillation': reference_distillation(c, groups, meta, checked)}
    destination.mkdir(parents=True)
    data.write_json(destination / 'access.json', {'status': 'BOTH_ARMS_COMPLETE_GATE_PASSED_VALID_PARSE_STARTING',
        'development_parse_attempts': 1, 'train': 0, 'heldout': 0, 'owners': 0})
    labelled = pilot.attach_labels(groups['development'], old_c, 'development')
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    e = c['evaluation']
    draws = np.random.default_rng(e['bootstrap_seed']).integers(0, 20, size=(e['bootstrap_replicates'], 3, 20))
    result = {'status': 'BOTH_SENSITIVITIES_VALID_EVALUATED_INTERPRETATION_REQUIRED',
        'config': c, 'score_manifest_sha256': data.sha256(out / 'manifest.json'),
        'columns': list(metrics.COLUMNS), 'arms': {},
        'label_parses': {'train': 0, 'development': 1, 'heldout': 0, 'owners': 0},
        'formal_training_seconds': run_record['formal_training_seconds'],
        'bootstrap': {'replicates': 5000, 'seed': 20260910, 'confidence_level': .95,
            'unit': '20 groups per underlying domain; same draws across both arms/references/times/orders'},
        'interpretation': 'Finite two-point sensitivity on reused valid; conditional unadjusted intervals. No optimum, unique cause, automatic winner or further search.'}
    for arm in ARMS:
        arrays, points = {}, {}
        (destination / arm).mkdir()
        for name in NAMES:
            rows = [metrics.classification(y, x) for y, x in zip(truth, scores[arm][name])]
            matrix = np.column_stack((np.asarray([[row[k] for k in metrics.CLASS_KEYS] for row in rows]),
                metrics.retrieval(truth, scores[arm][name], 28)))
            if name.endswith('_shared') and not np.array_equal(matrix, original[name]):
                raise ValueError('Shared metrics differ; no retry')
            arrays[name] = matrix
            path = destination / arm / f'{name}_metrics.npy'
            np.save(path, matrix, allow_pickle=False)
            points[name] = {'file': data.record(path, destination), 'confusion': [row['confusion'] for row in rows],
                'by_domain': {d: dict(zip(metrics.COLUMNS, matrix[i*20:(i+1)*20].mean(0).tolist())) for i, d in enumerate('ABC')}}
        result['arms'][arm] = {'weights': WEIGHTS[arm], 'points': points,
            'comparisons': {role: compare(arrays, reference, draws, c, role) for role, reference in references.items()}}
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
    elif args.action == 'evaluate':
        result = evaluate(args.out.resolve(), args.evaluation.resolve())
    else:
        for arm in ARMS:
            verify_arm_models(args.out.resolve() / arm)
        result = {'status': 'BOTH_ARM_MODELS_VERIFIED'}
    print(data.json_bytes({'status': result['status'], 'out': str(args.out)}).decode())


if __name__ == '__main__':
    main()


