"""ER+GCR over complete account groups; positive OMP and absolute replay weights."""
from __future__ import annotations

import json
import random

import numpy as np

import step28_record_replay as record

data = record.data


def nonnegative_omp(dictionary: np.ndarray, target: np.ndarray, ids: list[str]) -> tuple:
    """At most six positive-correlation expansions with augmented ridge NNLS."""
    from scipy.optimize import nnls
    d, b = np.asarray(dictionary, np.float64), np.asarray(target, np.float64)
    if (d.ndim != 2 or b.shape != (d.shape[0],) or d.shape[1] != len(ids)
            or ids != sorted(set(ids)) or not np.isfinite(d).all() or not np.isfinite(b).all()):
        raise ValueError('Invalid GCR dictionary/target/order')
    support, residual, weights, trace = [], b.copy(), np.zeros(0), []
    objective = float(b @ b)
    initial_correlations = d.T @ b
    if not np.isfinite(objective) or not np.isfinite(initial_correlations).all():
        raise ValueError('Nonfinite GCR objective/correlations')
    threshold = 1e-10*max(float(np.max(np.abs(initial_correlations), initial=0.)), 1e-30)
    reason = 'six_expansions'
    for _ in range(min(6, len(ids))):
        candidates = [j for j in range(len(ids)) if j not in support]
        correlations = d.T @ residual
        if not np.isfinite(correlations).all():
            raise ValueError('Nonfinite GCR residual correlations')
        chosen = min(candidates, key=lambda j: (-correlations[j], ids[j]))
        if correlations[chosen] <= threshold:
            reason = 'correlation_tolerance'
            break
        support.append(chosen)
        ds = d[:, support]
        augmented = np.vstack((ds, np.sqrt(.5)*np.eye(len(support))))
        weights, rnorm = nnls(augmented, np.r_[b, np.zeros(len(support))], maxiter=1000)
        normal, rhs = ds.T @ ds+.5*np.eye(len(support)), ds.T @ b
        h = normal @ weights-rhs
        tolerance = 1e-8*max(np.linalg.norm(rhs, np.inf), np.linalg.norm(normal @ weights, np.inf), 1e-30)
        violation = np.where(weights > 0, np.abs(h), np.maximum(0., -h))
        if (not np.isfinite(weights).all() or not np.isfinite(h).all() or not np.isfinite(rnorm)
                or np.any(weights < 0) or np.max(violation, initial=0.) > tolerance):
            raise ValueError('Nonnegative GCR refit failed fixed KKT conditions: '+json.dumps(dict(
                support=[ids[j] for j in support], weights=weights.tolist(), h=h.tolist(),
                residual_form_h=(ds.T @ (ds @ weights-b)+.5*weights).tolist(),
                kkt_max=float(np.max(violation, initial=0.)), tolerance=float(tolerance))))
        new_objective = float(rnorm*rnorm)
        if new_objective-objective > 1e-10*max(objective, 1e-30):
            raise ValueError('GCR objective increased beyond fixed numerical tolerance')
        trace.append(dict(selected=ids[chosen], correlation=float(correlations[chosen]),
                          objective=new_objective, change=new_objective-objective,
                          kkt_max=float(np.max(violation, initial=0.)), kkt_tolerance=float(tolerance)))
        objective, residual = new_objective, b-ds @ weights
    result = {ids[j]: float(w) for j, w in zip(support, weights, strict=True) if w > 0}
    return result, dict(stop_reason=reason, target_norm=float(np.linalg.norm(b)),
                        objective=objective, threshold=threshold, expansions=trace,
                        positive_count=len(result))


def supervised(table, group, _seed=None, _arm=None, _reference=None):
    import torch
    predicted = record.account_logits(table, record.assignment(group))
    if group.labels is None:
        raise ValueError('GCR requires authorized training supervision')
    truth = torch.tensor(group.labels, device=table.device, dtype=table.dtype)
    losses = record.parent.ranking.objectives(predicted, truth, .5)
    return losses['total'], {k: float(v.detach()) for k, v in losses.items()}


def backward(model, group, c: dict, seed: int, coefficient: float, check=lambda: None) -> dict:
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Only positive stored absolute weights enter replay')
    def objective(table, g, *args):
        loss, log = supervised(table, g)
        return coefficient*loss, {**log, 'coefficient': coefficient, 'weighted': float(coefficient*loss.detach())}
    return record.backward_group(model, group, c, 'R0', seed, 0, None, check, objective_fn=objective)


def head_gradient(model, group, c: dict, check=lambda: None) -> np.ndarray:
    """Exact chunked VJP of beta*L_G through every real head parameter."""
    import torch
    model.eval()
    parameters = list(model.head.parameters())
    with torch.no_grad():
        z = record.record_vectors(model, group, c, check)
        values = record.pair_table(model, z, check)
    proxy = values.detach().requires_grad_(True)
    loss, _ = supervised(proxy, group)
    derivative, = torch.autograd.grad(.5*loss, proxy)
    accumulators = [torch.zeros_like(p) for p in parameters]
    left, right = torch.triu_indices(len(z), len(z), 1, device=z.device)
    for start in range(0, len(left), 512):
        check()
        u, v = z[left[start:start+512]], z[right[start:start+512]]
        score = model.head(torch.cat(((u-v).abs(), u*v), dim=1)).flatten()
        gradients = torch.autograd.grad(score, parameters, derivative[start:start+512])
        for total, gradient in zip(accumulators, gradients, strict=True):
            total.add_(gradient)
    result = torch.cat([v.reshape(-1) for v in accumulators]).double().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite complete-head gradient')
    return result


class Memory:
    def __init__(self, spec: dict):
        self.spec, self.stage, self.seen = spec, 0, 0
        self.groups: dict[str, object] = {}
        self.weights: dict[str, float] = {}
        self.draw_rng = random.Random(data.seed_for(spec['roots']['memory'], spec['order'], 'gcr_draw'))

    def to_bytes(self) -> bytes:
        if (set(self.groups) != set(self.weights) or len(self.groups) > 6
                or any(not np.isfinite(w) or w <= 0 for w in self.weights.values())):
            raise ValueError('Invalid positive coreset history')
        for g in self.groups.values():
            g.validate()
            if g.labels is None:
                raise ValueError('Historical group lost supervision')
        payload = data.json_bytes(dict(spec=self.spec, stage=self.stage, seen=self.seen,
            groups={k: v.payload() for k, v in sorted(self.groups.items())}, weights=self.weights,
            draw_rng=self.draw_rng.getstate()))
        if len(payload) > 1048576:
            raise ValueError('Complete GCR history exceeds 1 MiB')
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes, spec: dict) -> Memory:
        row = json.loads(payload)
        if row['spec'] != spec:
            raise ValueError('Foreign GCR memory')
        result = cls(spec)
        result.stage, result.seen = row['stage'], row['seen']
        result.groups = {uid: data.Group.restore(g) for uid, g in row['groups'].items()}
        result.weights = row['weights']
        result.draw_rng.setstate(data.tuple_state(row['draw_rng']))
        if result.to_bytes() != payload:
            raise ValueError('GCR history roundtrip differs')
        return result

    def summary(self) -> dict:
        return dict(bytes=len(self.to_bytes()), groups=sorted(self.groups), weights=self.weights.copy(),
                    stage=self.stage, seen=self.seen, slots=28*len(self.groups))


class Stage:
    def __init__(self, spec: dict, stage: int, current: list, memory: Memory, _model):
        if stage != memory.stage+1 or len(current) != 48 or memory.seen != 48*(stage-1):
            raise ValueError('GCR stage/current counter scope differs')
        self.spec, self.stage, self.memory = spec, stage, memory
        self.current = {g.uid: g for g in current}
        self.candidates, self.entered = [], set()
        self.rng = random.Random(data.seed_for(spec['roots']['memory'], spec['order'], stage, 'gcr_reservoir'))
        memory.draw_rng.seed(data.seed_for(spec['roots']['memory'], spec['order'], stage, 'gcr_draw'))

    def draw(self) -> tuple:
        memory, current = self.memory, self.candidates
        if not current and not memory.groups:
            return None, None, None
        if not memory.groups:
            use_current = True
        elif not current:
            use_current = False
        else:
            m = len(self.entered)
            use_current = memory.draw_rng.random() < m/(memory.seen+m)
        if use_current:
            return self.current[memory.draw_rng.choice(current)], 1., 'current_pool'
        uid = memory.draw_rng.choice(sorted(memory.groups))
        return memory.groups[uid], memory.weights[uid], 'history_pool'

    def backward(self, model, group, c: dict, step: int, check=lambda: None) -> dict:
        replay, weight, pool = self.draw()
        model.train()
        seed = lambda role: data.seed_for(self.spec['roots']['schedule'], self.spec['order'],
                                         self.stage, step, role, 'gcr_dropout')
        result = dict(current=backward(model, group, c, seed('current'), 1., check))
        if replay is not None:
            result['replay'] = {**backward(model, replay, c, seed('replay'), .5*weight, check),
                                'uid': replay.uid, 'pool': pool, 'absolute_weight': weight}
        return result

    def after_update(self, group) -> None:
        if group.uid not in self.current or self.current[group.uid] != group:
            raise ValueError('GCR admission outside actual current fit')
        if group.uid in self.entered:
            return
        self.entered.add(group.uid)
        if len(self.candidates) < 6:
            self.candidates.append(group.uid)
        else:
            j = self.rng.randrange(len(self.entered))
            if j < 6:
                self.candidates[j] = group.uid

    def retain(self, model, c: dict, check=lambda: None) -> dict:
        if self.entered != set(self.current):
            raise ValueError('Incomplete unique current candidate stream')
        if self.stage == 3:
            return {'skipped_final_future_maintenance': True}
        groups = {**self.memory.groups, **{uid: self.current[uid] for uid in self.candidates}}
        ids = sorted(groups)
        dictionary = np.column_stack([head_gradient(model, groups[uid], c, check) for uid in ids])
        previous = np.array([self.memory.weights.get(uid, 1.) for uid in ids], np.float64)
        weights, evidence = nonnegative_omp(dictionary, dictionary @ previous, ids)
        self.memory.groups = {uid: groups[uid] for uid in weights}
        self.memory.weights = weights
        self.memory.stage, self.memory.seen = self.stage, self.memory.seen+len(self.entered)
        self.memory.to_bytes()
        return {**evidence, 'candidate_ids': ids, 'previous_weights': previous.tolist(),
                'gradient_parameters': dictionary.shape[0], 'selection_bytes': dictionary.nbytes}

    def transient(self) -> dict:
        return dict(stage=self.stage, current_ids=sorted(self.current), candidates=self.candidates,
                    entered=sorted(self.entered), rng=self.rng.getstate(), draw_rng=self.memory.draw_rng.getstate())

    def restore_transient(self, row: dict) -> None:
        if (row['stage'] != self.stage or row['current_ids'] != sorted(self.current)
                or not set(row['candidates']) <= set(row['entered']) <= set(self.current)
                or len(row['candidates']) > 6):
            raise ValueError('GCR transient current source differs')
        self.candidates, self.entered = row['candidates'], set(row['entered'])
        self.rng.setstate(data.tuple_state(row['rng']))
        self.memory.draw_rng.setstate(data.tuple_state(row['draw_rng']))
