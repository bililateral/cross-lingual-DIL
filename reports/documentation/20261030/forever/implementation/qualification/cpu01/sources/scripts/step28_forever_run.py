"""FOREVER stage entry and atomic safe points; formal dispatch is not enabled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid
from typing import Callable

import numpy as np

import step28_forever as m
from step28_continual_comparison_run import training_inputs


def canonical(value):
    return json.loads(m.data.json_bytes(value))


class Supply:
    """Dispatcher wrapper with explicit acquired-vs-committed progress."""

    def __init__(self, supply, spec: m.Spec):
        self.supply, self.spec = supply, spec
        self.acquired = None

    def current(self, stage: int) -> tuple:
        if self.acquired is not None:
            raise ValueError('Current stage already dispatched')
        result = self.supply.current(self.spec.run_id, self.spec.order, stage)
        self.acquired = stage
        return result

    def state(self) -> dict:
        return dict(next_stage=self.supply.next_stage.get(self.spec.run_id, 1),
                    acquired=self.acquired, order=self.spec.order,
                    partition_sha256=hashlib.sha256(m.data.json_bytes(self.supply.partition)).hexdigest())

    def restore(self, saved: dict, learner: m.Learner) -> tuple | None:
        if saved['order'] != self.spec.order or saved['partition_sha256'] != self.state()['partition_sha256']:
            raise ValueError('Restored supply identity differs')
        if saved['acquired'] is not None:
            if saved['acquired'] != learner.stage or saved['next_stage'] != learner.stage+1 or learner.phase != 'current':
                raise ValueError('Acquired-stage cursor differs')
            # Reacquire ONLY this current domain; never replay an earlier Supply request.
            self.supply.next_stage[self.spec.run_id] = learner.stage
            self.supply.orders[self.spec.run_id] = self.spec.order
            result = self.current(learner.stage)
        else:
            if saved['next_stage'] != learner.stage or learner.phase not in ('await_current', 'complete'):
                # The completed third domain has next_stage=4, while its learner stage stays 3.
                if not (learner.phase == 'complete' and saved['next_stage'] == 4 and learner.stage == 3):
                    raise ValueError('Committed-stage cursor differs')
            self.supply.next_stage[self.spec.run_id] = saved['next_stage']
            self.supply.orders[self.spec.run_id] = self.spec.order
            result = None
        self.acquired = saved['acquired']
        return result


class Store:
    """Atomic latest publication. A failed transaction leaves its old latest usable."""

    def __init__(self, root: Path, identity: dict, check: Callable = lambda reserve=0: None):
        self.root, self.identity, self.check = root, canonical(identity), check
        for name in ('work', 'anchors', 'memory', 'models', 'maps'):
            (root/name).mkdir(parents=True, exist_ok=True)
        self.anchor_object = None
        self.anchor_record = None
        self.points = {}

    def record(self, path: Path) -> dict:
        return m.data.record(path, self.root)

    def verify(self, row: dict) -> Path:
        path = self.root/row['path']
        if not path.resolve().is_relative_to(self.root.resolve()) or self.record(path) != row:
            raise ValueError('Checkpoint dependency identity differs')
        return path

    def commit(self, learner: m.Learner, supply: Supply, hook: Callable = lambda phase: None) -> dict:
        import torch
        if learner.phase not in ('current', 'await_current', 'complete'):
            raise ValueError('Unsafe partial replay/stage checkpoint')
        started = time.monotonic()
        self.check(m.record.base.persistence.checkpoint_reserve(learner.model, learner.optimizer))
        learner.optimizer.zero_grad(set_to_none=True)
        generation = uuid.uuid4().hex
        old = m.data.read_json(self.root/'latest.json') if (self.root/'latest.json').exists() else None
        if learner.anchor is not self.anchor_object:
            self.anchor_record = None
            if learner.anchor is not None:
                path = self.root/'anchors'/f'{generation}.pt'
                torch.save(learner.anchor, path)
                self.anchor_record = self.record(path)
            self.anchor_object = learner.anchor
        state = learner.small()
        state.update(rng=m.diag.rng_state(), modes=[v.training for v in learner.model.modules()],
                     supply=supply.state(), parameters=m.mapping(learner.model, learner.optimizer))
        payload = learner.old.to_bytes()
        if len(payload)+len(m.data.json_bytes(state)) > 1048576:
            raise ValueError('Complete samples plus small scheduling/RNG state exceed 1 MiB')
        path = self.root/'memory'/f'{generation}.bin'
        path.write_bytes(payload)
        meta = canonical(dict(identity=self.identity, state=state, memory=self.record(path),
                              anchor=self.anchor_record, points=self.points))
        full = self.root/'work'/f'{generation}.pt'
        saved = m.core.save_state(full, learner.model, learner.optimizer, meta)
        row = dict(format='forever_safe_point_v1', identity=self.identity, generation=generation,
                   full=dict(file=self.record(full), state_sha256=saved['state_sha256']), metadata=meta)
        for record in (meta['memory'], meta['anchor'], row['full']['file']):
            if record:
                self.verify(record)
        temporary = self.root/f'latest.{generation}.json'
        m.data.write_json(temporary, row)
        hook('before_publish')
        temporary.replace(self.root/'latest.json')
        hook('after_publish')
        # Only obsolete dependency files referenced by the preceding latest are retired.
        if old:
            old_rows = [old['full']['file'], old['metadata']['memory'], old['metadata']['anchor']]
            keep = {r['path'] for r in (row['full']['file'], meta['memory'], meta['anchor']) if r}
            for old_row in old_rows:
                if old_row and old_row['path'] not in keep:
                    self.verify(old_row).unlink()
        return dict(generation=generation, full_bytes=full.stat().st_size,
                    elapsed_seconds=time.monotonic()-started, history_bytes=len(payload)+len(m.data.json_bytes(state)))

    def restore(self, learner: m.Learner, supply: Supply) -> tuple | None:
        import torch
        started = time.monotonic()
        row = m.data.read_json(self.root/'latest.json')
        if row['format'] != 'forever_safe_point_v1' or row['identity'] != self.identity:
            raise ValueError('Restoration run/config/source identity differs')
        meta = row['metadata']
        if meta['state']['parameters'] != m.mapping(learner.model, learner.optimizer):
            raise ValueError('Restored trainable parameter coverage differs')
        full = self.verify(row['full']['file'])
        if m.core.restore_state(full, learner.model, learner.optimizer, row['full']['state_sha256']) != meta:
            raise ValueError('Restored full metadata differs')
        learner.old = m.data.Memory.from_bytes(self.verify(meta['memory']).read_bytes())
        learner.anchor = (torch.load(self.verify(meta['anchor']), map_location='cpu', weights_only=True)
                          if meta['anchor'] else None)
        state = meta['state']
        if state['spec'] != learner.spec.identity():
            raise ValueError('Restored method identity differs')
        learner.stage, learner.phase = state['stage'], state['phase']
        learner.clock = m.Clock(**state['clock'])
        learner.counts, learner.maps = state['counts'], state['maps']
        if len(state['modes']) != len(list(learner.model.modules())):
            raise ValueError('Module mode coverage differs')
        for module, mode in zip(learner.model.modules(), state['modes'], strict=True):
            module.training = mode
        m.diag.restore_rng(state['rng'])
        if m.parent.adam_step(learner.optimizer) != sum(learner.counts.values()):
            raise ValueError('Restored Adam counters differ')
        self.anchor_record, self.anchor_object = meta['anchor'], learner.anchor
        self.points = meta['points']
        result = supply.restore(state['supply'], learner)
        self.last_restore_seconds = time.monotonic()-started
        return result


def endpoint(learner: m.Learner, calibration: list, store: Store) -> dict:
    """Finalized stage calibration and inference state, published with the same latest."""
    if learner.phase != 'endpoint' or len(calibration) != 12:
        raise ValueError('Endpoint precedes consolidation or calibration coverage differs')
    number = learner.stage
    generation = uuid.uuid4().hex
    with m.diag.neutral(learner.model):
        scores = m.record.score(learner.model, calibration, learner.c, store.check)
        fitted = m.parent.calibration.fit(scores, np.asarray([g.labels for g in calibration], np.uint8),
                                          role='calibration', check=store.check)
    if fitted['status'] != 'PASS_CALIBRATION_FIT':
        raise ValueError('Current calibration failed')
    current = dict(zip(('a', 'b'), m.parent.calibration.parameters(fitted)))
    learner.maps.setdefault('first', current)
    learner.maps[str(number)] = current
    metadata = canonical(dict(identity=store.identity, spec=learner.spec.identity(), stage=number,
                              maps=dict(first=learner.maps['first'], stage=current), counts=learner.counts))
    model_path = store.root/'models'/f'stage{number}_{generation}.pt'
    saved = m.core.save_state(model_path, learner.model, None, metadata)
    # Reuse the existing inference restoration and score invariance requirement.
    rng, modes = m.diag.rng_state(), [x.training for x in learner.model.modules()]
    m.core.restore_state(model_path, learner.model, None, saved['state_sha256'])
    with m.diag.neutral(learner.model):
        if not np.array_equal(scores, m.record.score(learner.model, calibration, learner.c, store.check)):
            raise ValueError('Inference restoration changed current calibration scores')
    for module, mode in zip(learner.model.modules(), modes, strict=True):
        module.training = mode
    m.diag.restore_rng(rng)
    map_path = store.root/'maps'/f'stage{number}_{generation}.json'
    m.data.write_json(map_path, fitted)
    return dict(stage=number, model=store.record(model_path), state_sha256=saved['state_sha256'],
                calibration=store.record(map_path), maps=metadata['maps'], counts=learner.counts.copy(),
                calibration_ids=[g.uid for g in calibration], inference_restore_verified=True)


def stage(learner: m.Learner, supply: Supply, store: Store, current=None, calibration=None,
          before_step: Callable = lambda: None, log: Callable = lambda row: None,
          hook: Callable = lambda phase: None) -> dict:
    if learner.phase == 'complete':
        raise ValueError('Trajectory already complete')
    if current is None:
        current, calibration = supply.current(learner.stage)
        learner.phase = 'current'
    elif supply.acquired != learner.stage or learner.phase != 'current':
        raise ValueError('Restored current stage was not reacquired')
    sequence, _ = m.parent.schedule(current, learner.c, learner.spec.order, learner.stage)
    while learner.clock.current < 288:
        learner.advance(sequence[learner.clock.current], store.check, before_step, log,
                        lambda: store.commit(learner, supply))
    learner.consolidate(store.check, before_step, log)
    point = endpoint(learner, calibration, store)
    learner.retain_and_switch(current)
    supply.acquired = None
    store.points[str(point['stage'])] = point
    store.commit(learner, supply, hook)
    return point


if __name__ == '__main__':
    raise SystemExit('Formal FOREVER dispatch is disabled. Use the bounded qualification entry only.')
