"""FOREVER's fixed group adaptation. No archive access or execution on import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable

import numpy as np

import step28_continual_five as five

record, parent, data, core, diag = five.record, five.parent, five.data, five.core, five.diag
NAME = 'FOREVER_group_fixed'
THRESHOLDS = (1, 2, 4, 7, 15, 30, 60, 90, 120)


@dataclass(frozen=True)
class Spec:
    order: str
    seed: int

    def __post_init__(self):
        if self.order not in five.ORDERS or self.seed not in five.SEEDS:
            raise ValueError('Outside the nine fixed FOREVER trajectories')

    @property
    def run_id(self) -> str:
        return f'{self.order}_s{self.seed}_{NAME}'

    def identity(self) -> dict:
        return dict(name=NAME, **asdict(self), initialization=2026101900+self.seed,
                    schedule=2026102900+self.seed, memory=2026103900+self.seed, partition=20260918)


def config(spec: Spec) -> dict:
    c = record.config()
    c.update(initialization_seed=spec.identity()['initialization'],
             schedule_seed=spec.identity()['schedule'], forever_spec=spec.identity())
    return c


def fresh(spec: Spec, c: dict, device: str = 'cuda:0') -> tuple:
    random.seed(spec.identity()['initialization'])
    np.random.seed(spec.identity()['initialization'])
    model = record.load_model(c, device)
    five.checkpoint_encoder(model.encoder)
    optimizer = core.make_optimizer(model, c)
    parameters(model, optimizer)
    return model, optimizer


def parameters(model: Any, optimizer: Any | None = None) -> dict:
    import torch
    result = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not result or any(p.dtype != torch.float32 for p in result.values()):
        raise ValueError('FOREVER requires nonempty FP32 trainable encoder and head')
    if not any(n.startswith('encoder.') for n in result) or not any(n.startswith('head.') for n in result):
        raise ValueError('Both encoder and head must be trainable')
    if optimizer is not None:
        actual = [id(p) for g in optimizer.param_groups for p in g['params']]
        if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in result.values()}:
            raise ValueError('Trainable P and optimizer membership differ')
    return result


def mapping(model: Any, optimizer: Any) -> list[dict]:
    return [dict(name=n, shape=list(p.shape), dtype=str(p.dtype), numel=p.numel(), bytes=p.numel()*p.element_size())
            for n, p in parameters(model, optimizer).items()]


def snapshot(model: Any) -> dict:
    return {n: p.detach().cpu().clone() for n, p in parameters(model).items()}


def delta(model: Any, before: dict, check: Callable = lambda: None) -> dict:
    """FP32 differences / FP64 sum, equivalent to one concatenated global L2."""
    import torch
    values = parameters(model)
    if values.keys() != before.keys():
        raise ValueError('Incomplete displacement snapshot')
    sums = dict(encoder=0., head=0.)
    with torch.no_grad():
        for name, p in values.items():
            check()
            if p.shape != before[name].shape or before[name].dtype != torch.float32:
                raise ValueError('Displacement tensor identity differs')
            value = p.detach().cpu().reshape(-1)
            saved = before[name].reshape(-1)
            for start in range(0, value.numel(), 1048576):
                part = (value[start:start+1048576]-saved[start:start+1048576]).double().square().sum().item()
                sums['encoder' if name.startswith('encoder.') else 'head'] += part
    if not all(math.isfinite(x) for x in sums.values()):
        raise ValueError('Nonfinite actual parameter displacement')
    return dict(total=math.sqrt(sum(sums.values())), **{k: math.sqrt(v) for k, v in sums.items()})


def regularize(model: Any, anchor: dict, beta: float, check: Callable = lambda: None) -> dict:
    """Add the exact sum-L2 gradient; no detached surrogate objective."""
    import torch
    values = parameters(model)
    if values.keys() != anchor.keys() or not .0005 <= beta <= .003:
        raise ValueError('Anchor coverage or effective beta differs')
    sums = {k: dict(task_sq=0., regularizer_sq=0., distance_sq=0.) for k in ('encoder', 'head')}
    with torch.no_grad():
        for name, p in values.items():
            check()
            a = anchor[name]
            if a.device.type != 'cpu' or a.dtype != torch.float32 or a.shape != p.shape:
                raise ValueError('Anchor must be a matching immutable FP32 CPU tensor')
            if p.grad is None:
                raise ValueError(f'Missing/nonfinite expected task gradient: {name}')
            s = sums['encoder' if name.startswith('encoder.') else 'head']
            if not p.is_contiguous() or not p.grad.is_contiguous():
                raise ValueError('Chunked in-place gradient requires contiguous parameters and gradients')
            values_flat, anchor_flat, gradient_flat = p.detach().view(-1), a.view(-1), p.grad.view(-1)
            for start in range(0, p.numel(), 1048576):
                check()
                gradient = gradient_flat[start:start+1048576]
                if not torch.isfinite(gradient).all():
                    raise ValueError(f'Nonfinite expected task gradient: {name}')
                difference = values_flat[start:start+1048576]-anchor_flat[start:start+1048576].to(p.device)
                s['distance_sq'] += difference.double().square().sum().item()
                s['task_sq'] += gradient.double().square().sum().item()
                difference.mul_(2*beta)
                s['regularizer_sq'] += difference.double().square().sum().item()
                gradient.add_(difference)
    distance = sum(s['distance_sq'] for s in sums.values())
    if not math.isfinite(distance):
        raise ValueError('Nonfinite anchor penalty')
    return dict(beta=beta, distance_sq=distance, penalty=beta*distance,
                norms={k: dict(task=math.sqrt(s['task_sq']), regularizer=math.sqrt(s['regularizer_sq']))
                       for k, s in sums.items()})


@dataclass
class Clock:
    current: int = 0
    tau: float = 0.
    tau_day: float | None = None
    mu0: float | None = None
    mu: float | None = None
    next_threshold: int = 0

    def observe(self, movement: float) -> None:
        if not math.isfinite(movement) or movement < 0 or self.current >= 288:
            raise ValueError('Invalid current update displacement/cursor')
        self.current += 1
        self.tau += movement
        if self.current == 24:
            if self.tau <= 0:
                raise ValueError('Calibration has no actual parameter movement')
            self.tau_day = max(self.tau, 1e-8)
            self.mu0 = self.mu = self.tau/24
        elif self.current > 24:
            self.mu = .95*self.mu+.05*movement

    def due(self, has_history: bool) -> bool:
        return (has_history and self.tau_day is not None and self.next_threshold < len(THRESHOLDS)
                and self.tau >= self.tau_day*THRESHOLDS[self.next_threshold])

    def beta(self) -> float:
        if self.mu0 is None or self.mu is None:
            raise ValueError('Clock is not calibrated')
        return .001*min(3., max(.5, self.mu/(self.mu0+1e-12)))


def memory(spec: Spec):
    return data.Memory(data.seed_for(spec.identity()['memory'], spec.order, 'retention'))


def replay_order(spec: Spec, stage: int, event: str, epoch: int, old: Any) -> list:
    values = sorted(old.groups, key=lambda g: g.uid)
    random.Random(data.seed_for(spec.identity()['memory'], NAME, spec.order, spec.seed,
                                stage, event, epoch, 'replay_order')).shuffle(values)
    return values


def step(model: Any, optimizer: Any, group: Any, c: dict, spec: Spec, stage: int,
         current_step: int, role: str, key: tuple = (), anchor: dict | None = None,
         beta: float | None = None, check: Callable = lambda: None,
         before_step: Callable = lambda: None) -> dict:
    import torch
    if role not in ('current', 'replay', 'fixture') or stage not in (1, 2, 3):
        raise ValueError('Invalid update role')
    if role == 'replay' and (anchor is None or beta is None):
        raise ValueError('Replay requires the previous-stage anchor and fixed event beta')
    if role != 'replay' and (anchor is not None or beta is not None):
        raise ValueError('Current/fixture objective must not use historical regularization')
    values = parameters(model, optimizer)
    before = snapshot(model) if role == 'current' else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer.param_groups[0]['lr'] = parent.stage_lr(current_step) if role == 'current' else 1e-5
    optimizer.param_groups[1]['lr'] = .001
    seed = data.seed_for(spec.identity()['initialization'], NAME, spec.order, stage,
                         role, current_step, *key, group.uid, 'dropout')
    logs = record.backward_group(model, group, c, arm='R0', dropout_seed=seed,
                                  regroup_seed=0, reference=None, check=check)
    missing = [n for n, p in values.items() if p.grad is None]
    if missing:
        raise ValueError(f'Expected trainable parameters have no task gradient: {missing}')
    reg = regularize(model, anchor, beta, check) if role == 'replay' else None
    norm = torch.nn.utils.clip_grad_norm_(values.values(), 1., error_if_nonfinite=True)
    check()
    before_step()  # Physical attempted calls are charged outside recoverable state.
    optimizer.step()
    movement = delta(model, before, check) if before is not None else None
    return dict(role=role, objective=logs, regularizer=reg, displacement=movement,
                gradient_norm=float(norm), encoder_lr=optimizer.param_groups[0]['lr'],
                head_lr=optimizer.param_groups[1]['lr'], adam_step=parent.adam_step(optimizer))


class Learner:
    """Only current groups and surviving history; no Supply/archive reference."""

    def __init__(self, spec: Spec, model: Any, optimizer: Any, c: dict):
        self.spec, self.model, self.optimizer, self.c = spec, model, optimizer, c
        self.old, self.anchor = memory(spec), None
        self.clock = Clock()
        self.stage = 1
        self.phase = 'current'
        self.counts = dict(current=0, replay=0, fixture=0)
        self.maps: dict = {}

    def small(self) -> dict:
        return dict(spec=self.spec.identity(), stage=self.stage, phase=self.phase,
                    clock=asdict(self.clock), counts=self.counts.copy(), maps=self.maps)

    def historical_bytes(self) -> int:
        # Includes reservoir RNG, maps, clock and small metadata; full P is an explicit exception.
        size = len(self.old.to_bytes())+len(data.json_bytes(self.small()))
        if size > 1048576:
            raise ValueError('Sample history plus scheduling state exceeds 1 MiB')
        return size

    def update(self, group: Any, role: str, key: tuple = (), beta: float | None = None,
               check: Callable = lambda: None, before_step: Callable = lambda: None) -> dict:
        number = self.clock.current+1 if role == 'current' else max(1, self.clock.current)
        row = step(self.model, self.optimizer, group, self.c, self.spec, self.stage,
                   number, role, key, self.anchor if role == 'replay' else None,
                   beta, check, before_step)
        self.counts[role] += 1
        if role == 'current':
            self.clock.observe(row['displacement']['total'])
        if parent.adam_step(self.optimizer) != sum(self.counts.values()):
            raise ValueError('Real Adam step and current/replay/fixture counts differ')
        return row

    def event(self, event: str, epochs: int, check: Callable = lambda: None,
              before_step: Callable = lambda: None, log: Callable = lambda row: None) -> None:
        beta = self.clock.beta()  # Recomputed separately at the end, fixed through this pass.
        self.phase = event
        for epoch in range(epochs):
            for index, group in enumerate(replay_order(self.spec, self.stage, event, epoch, self.old)):
                log(dict(group=group.uid, event=event, epoch=epoch, index=index,
                         **self.update(group, 'replay', (event, epoch, index), beta, check, before_step)))
        self.phase = 'current'

    def advance(self, group: Any, check: Callable = lambda: None,
                before_step: Callable = lambda: None, log: Callable = lambda row: None,
                commit: Callable = lambda: None) -> None:
        if self.phase != 'current':
            raise ValueError('Resume only a committed safe point')
        log(dict(group=group.uid, **self.update(group, 'current', check=check, before_step=before_step)))
        event_saved = False
        while self.clock.due(bool(self.old.groups)):
            self.event(f'event{self.clock.next_threshold}', 2, check, before_step, log)
            self.clock.next_threshold += 1
            if self.clock.current < 288:
                commit()
                event_saved = True
        if self.clock.current < 288 and self.clock.current % 48 == 0 and not event_saved:
            commit()

    def consolidate(self, check: Callable = lambda: None, before_step: Callable = lambda: None,
                    log: Callable = lambda row: None) -> None:
        if self.clock.current != 288 or self.clock.due(bool(self.old.groups)) or self.phase != 'current':
            raise ValueError('All current updates and due events must precede consolidation')
        if self.old.groups:
            self.event('end', 1, check, before_step, log)
        self.phase = 'endpoint'

    def retain_and_switch(self, current: list) -> None:
        if self.phase != 'endpoint' or len(current) != 48 or self.old.seen != 48*(self.stage-1):
            raise ValueError('Invalid single stage retention boundary')
        if self.stage < 3:
            self.old.add_stage(sorted(current, key=lambda g: g.uid))
            self.anchor = snapshot(self.model)
            self.stage += 1
            self.clock = Clock()
            self.phase = 'await_current'
        else:
            self.phase = 'complete'
        self.historical_bytes()
