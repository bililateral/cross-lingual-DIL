"""Boundary-logit distillation with the frozen random full-group replay schedule.

No new Linux connection/launch before separate report and user resumption.
Windows verifies memory bytes but never decodes cached training supervision.
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Any

import numpy as np

import step28_continual_replay as baseline
import step28_continual_expression_run as pilot
from step28_continual_expression_run import core, data, metrics, persistence

POLICY = data.ROOT / 'schema/step28_continual_distillation_policy.json'
COMPLETE = 'COMPLETE_BOUNDARY_LOGIT_DISTILLATION_1620_RELOADED_VALID_BLIND'
NAMES = tuple(f'{o}_{s}' for o in pilot.ORDERS for s in ('shared', 'distill_stage2', 'distill_stage3'))


def contract() -> dict:
    c = data.read_json(POLICY)
    original = baseline.contract()
    for key in ('origin_run', 'origin_manifest_sha256', 'origin_evaluation',
                'origin_evaluation_sha256', 'physical_updates', 'memory', 'evaluation',
                'label_parses', 'runtime'):
        if c[key] != original[key]:
            raise ValueError('Distillation changed the paired replay contract')
    if (c['study'] != 'seller_alias_expression_boundary_logit_distillation'
            or c['distillation'] != {'coefficient': 0.5, 'target_dtype': '<f4',
                'target_timing': 'arrival_stage_end_after_reservoir_selection',
                'refresh_survivors': False, 'history_forward': 'shared_bce_mse_train_mode',
                'maximum_target_group_forwards': 36}):
        raise ValueError('Distillation differs from the adopted design')
    if (c['baseline_run'] != 'reports/seller_alias_continual/20260915/replay_execution/20260915_112424/job/run'
            or c['baseline_run_sha256'] != '470b723c2e79e790a6984087ebefb971807e489a0ca723b2dc1e2dcf4e1cdad6'
            or c['baseline_evaluation'] != 'reports/seller_alias_continual/20260915/replay_evaluation/20260915_125920/evaluation.json'
            or c['baseline_evaluation_sha256'] != 'd88a3f9bb00513eb83ffe98dc98468d0bdd62eceb5c55795c53fa07c3241b9e4'):
        raise ValueError('Random ER reference differs')
    return c


def sources() -> list[dict]:
    return baseline.sources() + [data.record(p, data.ROOT) for p in (
        Path(__file__), POLICY, data.ROOT / 'tests/test_step28_continual_distillation_contracts.py',
        data.ROOT / 'tests/test_step28_continual_population_contracts.py',
        data.ROOT / 'scripts/run_step28_distillation_linux_20260915.sh')]


def origin(c: dict, groups: dict, meta: list, checked: dict) -> tuple[dict, dict, dict]:
    root = data.ROOT / c['origin_run']
    if data.sha256(root / 'manifest.json') != c['origin_manifest_sha256']:
        raise ValueError('Original run differs')
    old_c = pilot.contract()
    scores, old = pilot.validated_scores(root, old_c, groups, meta, checked)
    path = data.ROOT / c['origin_evaluation']
    if data.sha256(path) != c['origin_evaluation_sha256']:
        raise ValueError('Original evaluation differs')
    e = data.read_json(path)
    if e['score_manifest_sha256'] != c['origin_manifest_sha256'] or e['columns'] != list(metrics.COLUMNS):
        raise ValueError('Original evaluation mapping differs')
    arrays = {}
    for name in pilot.POINTS:
        f = e['points'][name]['file']
        a = np.load(data.verify(path.parent / f['path'], f), allow_pickle=False)
        if a.shape != (60, len(metrics.COLUMNS)) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('Original metric matrix differs')
        arrays[name] = a
    return old, scores, arrays


def current_groups(groups: list, meta: list, domain: str) -> list:
    ids = {r['group_uid'] for r in meta if r['split'] == 'train' and r['domain'] == domain}
    result = sorted((g for g in groups if g.uid in ids), key=lambda g: g.uid)
    if len(result) != 60 or len({g.uid for g in result}) != 60:
        raise ValueError('Current domain supply differs')
    return result


def reference_baseline(c: dict, groups: dict, meta: list, checked: dict) -> dict:
    """Read saved ER evidence, never its supervision or model payloads."""
    root, path = data.ROOT / c['baseline_run'], data.ROOT / c['baseline_evaluation']
    if (data.sha256(root / 'manifest.json') != c['baseline_run_sha256']
            or data.sha256(path) != c['baseline_evaluation_sha256']):
        raise ValueError('Pinned random ER evidence differs')
    _, original, run_record = baseline.validated_scores(root, baseline.contract(), groups, meta, checked)
    evaluation = data.read_json(path)
    if (evaluation['score_manifest_sha256'] != c['baseline_run_sha256']
            or evaluation['columns'] != list(metrics.COLUMNS)
            or set(evaluation['points']) != set(baseline.NAMES)
            or evaluation['config'] != run_record['config']):
        raise ValueError('Random ER evaluation mapping differs')
    result = {'initial': original['initial']}
    for name, point in evaluation['points'].items():
        a = np.load(data.verify(path.parent / point['file']['path'], point['file']), allow_pickle=False)
        if a.shape != (60, len(metrics.COLUMNS)) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('Random ER saved metric matrix differs')
        if name.endswith('_shared') and not np.array_equal(a, original[name]):
            raise ValueError('Random ER shared metrics differ')
        result[name.replace('_er_stage', '_stage')] = a
    return result


class ReplayState(baseline.ReplayState):
    """Original reservoir/draw streams plus bounded immutable float32 targets."""

    def __init__(self, memory: data.Memory, order: str, stage: int, targets: dict | None = None):
        super().__init__(memory, order, stage)
        self.targets = {} if targets is None else targets

    def target(self, group: data.Group) -> np.ndarray:
        row = self.targets[group.uid]
        if row['pair_order_sha256'] != hashlib.sha256(data.json_bytes(group.sellers)).hexdigest():
            raise ValueError('Teacher/account pair order differs')
        raw = base64.b64decode(row['float32_base64'], validate=True)
        if len(raw) != 378 * 4 or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Teacher target size/hash differs')
        values = np.frombuffer(raw, dtype='<f4').copy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite teacher target')
        allowed = {self.order + '_shared'}
        if self.stage == 3:
            allowed.add(self.order + '_distill_stage2')
        if row['point'] not in allowed or len(row['model_state_sha256']) != 64:
            raise ValueError('Teacher provenance differs')
        return values

    def validate_targets(self) -> None:
        if set(self.targets) != {g.uid for g in self.memory.groups}:
            raise ValueError('Missing or evicted teacher target')
        for group in self.memory.groups:
            self.target(group)

    def to_bytes(self) -> bytes:
        payload = data.json_bytes({'replay': json.loads(super().to_bytes()), 'targets': self.targets})
        if len(payload) > self.memory.maximum_bytes:
            raise ValueError('Full history including teacher targets exceeds byte cap')
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> ReplayState:
        obj = json.loads(payload)
        old = baseline.ReplayState.from_bytes(data.json_bytes(obj['replay']))
        state = cls(old.memory, old.order, old.stage, obj['targets'])
        state.index = old.index
        state.rng.setstate(old.rng.getstate())
        state.validate_targets()
        if state.to_bytes() != payload:
            raise ValueError('Full distillation state does not roundtrip')
        return state

    def draw(self) -> data.Group:
        self.validate_targets()
        return super().draw()


def populate_targets(model: Any, state: ReplayState, current: list, name: str,
                     config: dict, budget: persistence.Budget) -> dict:
    """Use only just-arrived groups and surviving own targets; no archive lookups."""
    import torch
    if state.index != 0 or state.memory.seen != 60 * (state.stage - 2):
        raise ValueError('Target insertion stage differs')
    initial_targets = {uid: hashlib.sha256(data.json_bytes(row)).hexdigest()
                       for uid, row in state.targets.items()}
    maximum = len(state.to_bytes())
    for group in current:
        state.memory.add_stage([group])
        retained = {g.uid for g in state.memory.groups}
        state.targets = {uid: row for uid, row in state.targets.items() if uid in retained}
        maximum = max(maximum, len(state.to_bytes()))
    current_ids = {g.uid for g in current}
    new = [g for g in state.memory.groups if g.uid in current_ids]
    expected = {g.uid for g in state.memory.groups} - set(state.targets)
    if expected != {g.uid for g in new}:
        raise ValueError('Missing prior target or repeated arriving group')
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    modes = [(module, module.training) for module in model.modules()]
    before = core.state_digest(model.state_dict())
    started = time.monotonic()
    try:
        with torch.random.fork_rng(devices=devices):
            values = core.score(model, new, config, budget.check) if new else np.empty((0, 378), np.float32)
    finally:
        for module, training in modes:
            module.training = training
    if before != core.state_digest(model.state_dict()):
        raise ValueError('Teacher scoring changed the model')
    for group, value in zip(new, values):
        raw = np.asarray(value, dtype='<f4').tobytes()
        state.targets[group.uid] = {'float32_base64': base64.b64encode(raw).decode('ascii'),
            'sha256': hashlib.sha256(raw).hexdigest(), 'point': name,
            'model_state_sha256': before,
            'pair_order_sha256': hashlib.sha256(data.json_bytes(group.sellers)).hexdigest()}
        maximum = max(maximum, len(state.to_bytes()))
    state.validate_targets()
    if any(hashlib.sha256(data.json_bytes(state.targets[uid])).hexdigest() != digest
           for uid, digest in initial_targets.items() if uid in state.targets):
        raise ValueError('Surviving teacher target was refreshed')
    return {'new_groups': [g.uid for g in new], 'group_forwards': len(new),
        'seconds': time.monotonic() - started, 'model_state_unchanged': True,
        'maximum_insertion_history_bytes': maximum,
        'targets': {uid: {k: v for k, v in row.items() if k != 'float32_base64'}
                    for uid, row in state.targets.items()}}


def save_memory(state: ReplayState, current: list, name: str, out: Path,
                budget: persistence.Budget) -> tuple[ReplayState, dict]:
    payload = state.to_bytes()
    path = out / 'memory' / f'{name}.json'
    budget.check(len(payload))
    path.write_bytes(payload)
    budget.check(1)
    restored = ReplayState.from_bytes(path.read_bytes())
    # Full payload equality verifies exact text, account/pair alignment and labels.
    if restored.to_bytes() != payload:
        raise ValueError('Persisted memory differs')
    probe = random.Random()
    probe.setstate(restored.rng.getstate())
    return restored, {'file': data.record(path, out), 'seen': restored.memory.seen,
        'retained_groups': [g.uid for g in restored.memory.groups],
        'added_groups': [g.uid for g in current], 'order': state.order, 'stage': state.stage,
        'index': 0, 'next_draw': probe.choice(restored.memory.groups).uid,
        'disk_roundtrip_exact': True, 'full_state_bytes': len(payload)}


def historical_loss(predicted: Any, truth: Any, target: Any, coefficient: float) -> tuple:
    import torch
    if (predicted.shape != truth.shape or target.shape != predicted.shape
            or predicted.ndim != 1 or target.requires_grad
            or not np.isfinite(coefficient) or coefficient < 0
            or not torch.isfinite(predicted).all() or not torch.isfinite(target).all()):
        raise ValueError('Nonfinite, attached or misaligned historical objective')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(predicted, truth, reduction='mean')
    mse = (predicted - target).square().mean()
    loss = bce if coefficient == 0 else bce + coefficient * mse
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite historical loss')
    return loss, bce, mse


def observed_update(model: Any, optimizer: Any, current: data.Group, replay: data.Group,
                    config: dict, seed: int, replay_seed: int, check: Any,
                    targets: np.ndarray, *, observe: bool = True, coefficient: float = .5) -> dict:
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
                loss, bce, mse = historical_loss(predicted, truth, target, coefficient)
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
                 old_c: dict, budget: persistence.Budget) -> dict:
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
            state.target(replay), observe=(i == 0))
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
        'distillation_coefficient': 0.5, 'targets_unchanged': True,
        'maximum_gradient_norm_before_clip': max(norms), 'first_update': branch,
        'maximum_history_bytes': maximum, 'final_index': state.index,
        'final_state_sha256': hashlib.sha256(state.to_bytes()).hexdigest(),
        'reservoir_unchanged': True, 'training_seconds': time.monotonic() - started}


def check_shared(name: str, p: dict, old: dict, values: np.ndarray, old_values: np.ndarray) -> None:
    expected = old['points'][name]
    if (p['checkpoint']['state_sha256'] != expected['checkpoint']['state_sha256']
            or p['model_state_sha256'] != expected['model_state_sha256']
            or not np.array_equal(values, old_values)):
        raise ValueError('Original shared model/full Adam/scores not reproduced exactly')


def run(out: Path) -> dict:
    import torch
    c, old_c = contract(), pilot.contract()
    runtime = c['runtime']
    if (platform.system() != 'Linux' or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Requires resumed Linux stage and one eligible visible GPU')
    available = next(int(r.split()[1]) * 1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))
    if (torch.cuda.mem_get_info()[0] < runtime['minimum_free_gpu_bytes']
            or available < runtime['minimum_free_host_bytes']
            or shutil.disk_usage(data.ROOT).free < runtime['maximum_output_bytes']):
        raise RuntimeError('Insufficient shared resources; wait')
    if out.exists() or not out.is_relative_to((data.ROOT / 'reports').resolve()):
        raise ValueError('Use a new project reports directory')
    out.mkdir(parents=True)
    for directory in ('work', 'scores', 'models', 'memory'):
        (out / directory).mkdir()
    budget = persistence.Budget(out, c)
    snapshot = sources()
    r = {'status': 'RUNNING', 'config': c, 'source_files': snapshot, 'points': {}, 'training': {},
        'models': {}, 'memory': {}, 'starts': {},
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
        # Tiny CPU fixture, no formal labels/model loading/LaBSE updates.
        import sys
        import unittest
        sys.path.insert(0, str(data.ROOT / 'tests'))
        suite = unittest.defaultTestLoader.loadTestsFromName('test_step28_continual_distillation_contracts.TorchContracts')
        test = unittest.TextTestRunner(verbosity=2).run(suite)
        if not test.wasSuccessful() or test.skipped or test.testsRun != 3:
            raise RuntimeError('Actual first-update observation/reference check failed')
        r['torch_contracts'] = {'passed': test.testsRun, 'skipped': 0}
        groups, meta, checked = pilot.public_inputs(old_c)
        old, old_scores, _ = origin(c, groups, meta, checked)
        pretrained = core.model_files(old_c)
        if any(pretrained[k] != old_c['model'][k] for k in ('file_count', 'total_size_bytes', 'content_sha256')):
            raise ValueError('Pretrained model bytes differ')
        reference_baseline(c, groups, meta, checked)
        r['pretrained'], r['inputs'] = pretrained, old['inputs']
        data.write_json(out / 'access.json', {'train_parse_attempts': 1, 'development': 0, 'heldout': 0, 'owners': 0})
        r['label_parses']['train'] = 1
        groups['train'] = pilot.attach_labels(groups['train'], old_c, 'train')
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
                    r['training'][name] = train_replay(model, optimizer, current, state, old_c, budget)
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
        if sources() != snapshot:
            raise ValueError('Sources changed during execution')
        r['physical_updates'] = sum(t['updates'] for t in r['training'].values())
        if r['physical_updates'] != 1620:
            raise ValueError('Physical update total differs')
        r['formal_training_seconds'] = sum(t['training_seconds'] for t in r['training'].values())
        r['inference_seconds_including_replay'] = sum(p['timing']['inference_seconds_including_replay'] for p in r['points'].values())
        r['group_forwards_with_replay'] = 1080
        r['teacher_group_forwards'] = sum(m['teacher']['group_forwards'] for m in r['memory'].values())
        r['teacher_seconds'] = sum(m['teacher']['seconds'] for m in r['memory'].values())
        if not 0 <= r['teacher_group_forwards'] <= 36:
            raise ValueError('Teacher forward budget differs')
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
    if r['status'] != COMPLETE or set(r['models']) != {f'{o}_distill_stage3' for o in pilot.ORDERS}:
        raise ValueError('Incomplete retained model set')
    receipt = {'manifest_sha256': data.sha256(out / 'manifest.json'),
        'files': [data.record(data.verify(out / m['path'], m), out) for _, m in sorted(r['models'].items())]}
    data.write_json(out / 'model_verification.json', receipt)
    return receipt


def expected_history(meta: list, order: str) -> dict:
    """Independent ID-only Algorithm R; no cache JSON or training labels read."""
    rng = random.Random(data.seed_for(20260909, order, 'memory'))
    kept, seen, result = [], 0, {}
    for stage in (1, 2):
        added = sorted(r['group_uid'] for r in meta if r['split'] == 'train' and r['domain'] == order[stage - 1])
        if len(added) != 60:
            raise ValueError('Memory domain size differs')
        for uid in added:
            seen += 1
            if len(kept) < 6:
                kept.append(uid)
            else:
                j = rng.randrange(seen)
                if j < 6:
                    kept[j] = uid
        draw = random.Random(data.seed_for(20260909, order, stage + 1, 'replay_draw'))
        result[stage] = {'seen': seen, 'retained_groups': kept.copy(), 'added_groups': added,
            'order': order, 'stage': stage + 1, 'index': 0,
            'replay_group_ids': [draw.choice(kept) for _ in range(180)]}
    return result


def validated_scores(out: Path, c: dict, groups: dict, meta: list, checked: dict) -> tuple[dict, dict, dict]:
    old, old_scores, old_arrays = origin(c, groups, meta, checked)
    if (out / 'failure.json').exists():
        raise ValueError('Failed run cannot be evaluated')
    r = data.read_json(out / 'manifest.json')
    if (r['status'] != COMPLETE or r['config'] != c or r['source_files'] != sources()
            or r['physical_updates'] != 1620 or set(r['points']) != set(NAMES)
            or set(r['training']) != set(NAMES) or r['group_forwards_with_replay'] != 1080
            or set(r['models']) != {f'{o}_distill_stage3' for o in pilot.ORDERS}
            or set(r['memory']) != {f'{o}_{s}' for o in pilot.ORDERS for s in ('shared', 'distill_stage2')}
            or r['label_parses'] != {'train': 1, 'development': 0, 'heldout': 0, 'owners': 0}
            or r['torch_contracts'] != {'passed': 3, 'skipped': 0}
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
                if t['distillation_coefficient'] != .5 or not t['targets_unchanged']:
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
    if r['teacher_group_forwards'] != teacher_count or not 0 <= teacher_count <= 36:
        raise ValueError('Teacher scoring total differs')
    return scores, old_arrays, r


def compare(arrays: dict, old: dict, draws: np.ndarray, c: dict,
            reference_role: str = 'sequential') -> dict:
    remapped = {name.replace('_distill_stage', '_er_stage'): value for name, value in arrays.items()}
    result = baseline.compare(remapped, old, draws, c)
    # Rename role keys only; do not leave the new arm labelled as the old random ER.
    names = {'G_er': 'G_distill', 'F_er': 'F_distill', 'second_er_change': 'second_distill_change',
        'final_er_minus_seq': 'final_distill_minus_seq', 'G_er_transitions': 'G_distill_transitions',
        'er': 'distillation'}
    if reference_role == 'random_er':
        names.update({'G_seq': 'G_random_er', 'F_seq': 'F_random_er',
            'second_seq_change': 'second_random_er_change', 'G_seq_transitions': 'G_random_er_transitions',
            'final_er_minus_seq': 'final_distill_minus_random_er', 'sequential': 'random_er'})
    elif reference_role != 'sequential':
        raise ValueError('Unknown comparison reference')
    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {names.get(key, key): rename(child) for key, child in value.items()}
        return value
    return rename(result)


def evaluate(out: Path, destination: Path) -> dict:
    if platform.system() != 'Windows' or destination.exists():
        raise ValueError('Requires Windows and a new evaluation directory')
    c, old_c = contract(), pilot.contract()
    groups, meta, checked = pilot.public_inputs(old_c)
    scores, old_arrays, run_record = validated_scores(out, c, groups, meta, checked)
    reference = reference_baseline(c, groups, meta, checked)
    destination.mkdir(parents=True)
    data.write_json(destination / 'access.json', {'status': 'COMPLETE_GATE_PASSED_VALID_PARSE_STARTING',
        'development_parse_attempts': 1, 'train': 0, 'heldout': 0, 'owners': 0})
    labelled = pilot.attach_labels(groups['development'], old_c, 'development')
    truth = np.asarray([g.labels for g in labelled], dtype=np.uint8)
    arrays, points = {}, {}
    for name in NAMES:
        rows = [metrics.classification(y, x) for y, x in zip(truth, scores[name])]
        matrix = np.column_stack((np.asarray([[r[k] for k in metrics.CLASS_KEYS] for r in rows]),
            metrics.retrieval(truth, scores[name], 28)))
        if name.endswith('_shared') and not np.array_equal(matrix, old_arrays[name]):
            raise ValueError('Original shared valid metrics differ; no retry')
        arrays[name] = matrix
        path = destination / f'{name}_metrics.npy'
        np.save(path, matrix, allow_pickle=False)
        points[name] = {'file': data.record(path, destination), 'confusion': [r['confusion'] for r in rows],
            'by_domain': {d: dict(zip(metrics.COLUMNS, matrix[i * 20:(i + 1) * 20].mean(0).tolist())) for i, d in enumerate('ABC')}}
    e = c['evaluation']
    draws = np.random.default_rng(e['bootstrap_seed']).integers(0, 20, size=(e['bootstrap_replicates'], 3, 20))
    result = {'status': 'DISTILLATION_VALID_EVALUATED_INTERPRETATION_REQUIRED', 'config': c,
        'score_manifest_sha256': data.sha256(out / 'manifest.json'), 'columns': list(metrics.COLUMNS),
        'points': points, 'comparison': compare(arrays, old_arrays, draws, c),
        'versus_random_er': compare(arrays, reference, draws, c, 'random_er')['metrics'],
        'label_parses': {'train': 0, 'development': 1, 'heldout': 0, 'owners': 0},
        'formal_training_seconds': run_record['formal_training_seconds'],
        'bootstrap': {'replicates': 5000, 'seed': 20260910, 'confidence_level': .95,
            'unit': 'domain-specific groups, same draws across methods/times/orders; conditional on fixed trained paths'},
        'interpretation': 'Reused valid development screen; unadjusted secondary intervals. Boundary-logit distillation baseline adaptation only; no proprietary method or real-market claim.'}
    result['incremental_retention_passed'] = result['versus_random_er']['average_precision']['contrasts']['H']['conditional_95pct_interval'][0] > 0
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
