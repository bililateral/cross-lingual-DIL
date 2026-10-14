"""Complete S/G/F scoring and envelope VJP; no data-opening entry here."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch

import step28_record_replay as record

data = record.data
ARMS = ("S", "G", "F")
RNG_ORDER = "ABC"
SOLVER = dict(version="log_sinkhorn_round_v1", tau=1., maximum_iterations=4096,
              marginal_tolerance=1e-10, feasible_tolerance=1e-12,
              gap_tolerance=5e-9, roundoff_factor=256.)


def identity(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown pooling arm")
    return {"arm": arm, "tau": 1., "solver": SOLVER.copy(),
            "record_order": "canonical_offdiagonal_upper_v1", "output_dtype": "float32",
            "numerics": {"solver": "float64", "encoder_forward": "bfloat16",
                         "head_parameters_gradients_adam": "float32", "tf32": False,
                         "deterministic_algorithms": True}}


def bind(model, metadata: dict) -> str:
    pooling = metadata.get("pooling", {})
    arm = pooling.get("arm")
    if pooling != identity(arm):
        raise ValueError("Missing or incompatible restored pooling metadata")
    model.pooling_identity = pooling
    return arm


@torch.no_grad()
def solve(blocks: torch.Tensor, stats: dict | None = None, *, maximum_iterations: int = 4096):
    try:
        return _solve(blocks, stats, maximum_iterations=maximum_iterations)
    except ValueError:
        if stats is not None:
            stats["failures"] = stats.get("failures", 0)+1
        raise


def _solve(blocks: torch.Tensor, stats: dict | None, *, maximum_iterations: int):
    """Batched, unpadded n*m blocks. Precision limits are never adapted to data."""
    if (blocks.ndim != 3 or not 2 <= blocks.shape[1] <= 8 or not 2 <= blocks.shape[2] <= 8
            or not 1 <= maximum_iterations <= SOLVER["maximum_iterations"]):
        raise ValueError("Unsupported transport shape or iteration budget")
    k = blocks.double()
    if not torch.isfinite(k).all():
        raise ValueError("Nonfinite transport input")
    batch, n, m = k.shape
    mu = k.mean((1, 2))
    w = k-k.mean(2, keepdim=True)-k.mean(1, keepdim=True)+mu[:, None, None]
    kernel = w-math.log(n*m)  # tau = 1, uniform U.
    a, b = k.new_zeros((batch, n)), k.new_zeros((batch, m))
    done = torch.zeros(batch, dtype=torch.bool, device=k.device)
    values, plans = mu.clone(), k.new_zeros(k.shape)
    maxima = dict(iterations=0, marginal_error=0., feasible_error=0., gap_checked=0.)
    for iteration in range(1, maximum_iterations+1):
        idx = torch.where(~done)[0]
        kw, aw, bw = kernel[idx], a[idx], b[idx]
        aw = -math.log(n)-torch.logsumexp(kw+bw[:, None, :], 2)
        bw = -math.log(m)-torch.logsumexp(kw+aw[:, :, None], 1)
        shift = aw.mean(1, keepdim=True)
        aw, bw = aw-shift, bw+shift
        a[idx], b[idx] = aw, bw
        p = (kw+aw[:, :, None]+bw[:, None, :]).exp()
        eps = SOLVER["roundoff_factor"]*torch.finfo(torch.float64).eps*(
            1+w[idx].abs().amax((1, 2))+aw.abs().amax(1)+bw.abs().amax(1))
        rowerr = (p.sum(2)-1/n).abs().amax(1)
        colerr = (p.sum(1)-1/m).abs().amax(1)
        q = p * torch.clamp((1/n)/p.sum(2), max=1.)[:, :, None]
        q = q * torch.clamp((1/m)/q.sum(1), max=1.)[:, None, :]
        er, ec = 1/n-q.sum(2), 1/m-q.sum(1)
        if (er < -eps[:, None]).any() or (ec < -eps[:, None]).any():
            raise ValueError("Negative transport rounding mass")
        er, ec = er.clamp_min(0), ec.clamp_min(0)
        mass = er.sum(1)
        q = q+er[:, :, None]*ec[:, None, :]/torch.where(mass > 0, mass, 1.)[:, None, None]
        primal = (q*w[idx]-torch.special.xlogy(q, q*(n*m))).sum((1, 2))
        dual = p.sum((1, 2))-1-aw.mean(1)-bw.mean(1)
        gap = dual-primal
        if not all(torch.isfinite(x).all() for x in (p, q, primal, dual, eps)) or (q < 0).any():
            raise ValueError("Nonfinite or negative transport result")
        if (gap < -eps).any():
            raise ValueError("Transport dual bound below primal beyond roundoff")
        checked = gap.clamp_min(0)+eps
        feasible = torch.maximum((q.sum(2)-1/n).abs().amax(1), (q.sum(1)-1/m).abs().amax(1))
        marginal = torch.maximum(rowerr, colerr)
        accept = ((marginal <= SOLVER["marginal_tolerance"])
                  & (feasible <= SOLVER["feasible_tolerance"])
                  & (checked <= SOLVER["gap_tolerance"]))
        if accept.any():
            finished = idx[accept]
            plans[finished], values[finished] = q[accept], mu[finished]+primal[accept]
            done[finished] = True
            maxima["iterations"] = iteration
            for name, tensor in (("marginal_error", marginal), ("feasible_error", feasible), ("gap_checked", checked)):
                maxima[name] = max(maxima[name], float(tensor[accept].max()))
        if done.all():
            if stats is not None:
                stats["blocks"] = stats.get("blocks", 0)+batch
                stats["failures"] = stats.get("failures", 0)
                for key, value in maxima.items():
                    stats[key] = max(stats.get(key, 0), value)
            return values, plans, maxima
    raise ValueError(f"Transport did not converge in {maximum_iterations} full iterations; "
                     f"remaining={int((~done).sum())}, marginal={float(marginal.max())}, gap={float(checked.max())}")


class _Envelope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, stats):
        values, plan, _ = solve(blocks, stats)
        ctx.save_for_backward(plan.to(blocks.dtype))
        return values.to(blocks.dtype)

    @staticmethod
    def backward(ctx, upstream):
        plan, = ctx.saved_tensors
        return upstream[:, None, None]*plan, None


def block_scores(blocks, arm: str, stats: dict | None = None):
    if arm == "F":
        return _Envelope.apply(blocks, stats)
    if arm == "G":
        return (torch.logsumexp(blocks.double().flatten(1), 1)-math.log(blocks.shape[1]*blocks.shape[2])).to(blocks.dtype)
    if arm == "S":
        return blocks.mean((1, 2))
    raise ValueError("Unknown scoring arm")


def account_logits(table, slots, arm: str, stats: dict | None = None):
    if arm == "S":
        return record.account_logits(table, slots)
    identity(arm)
    n = len(slots)
    if table.shape != (n*(n-1)//2,) or not np.array_equal(np.unique(slots), np.arange(28)):
        raise ValueError("Complete table and all canonical account slots required")
    members = [np.flatnonzero(slots == i) for i in range(28)]
    if any(not 2 <= len(x) <= 8 for x in members):
        raise ValueError("Account records outside contract")
    i, j = torch.triu_indices(n, n, 1, device=table.device)
    matrix = table.new_zeros((n, n))
    matrix[i, j], matrix[j, i] = table, table
    buckets = {}
    for index, (a, b) in enumerate(zip(*np.triu_indices(28, 1))):
        buckets.setdefault((len(members[a]), len(members[b])), []).append((index, members[a], members[b]))
    scores = table.new_zeros(378)
    for entries in buckets.values():
        ids = torch.tensor([e[0] for e in entries], device=table.device)
        rows = torch.as_tensor(np.stack([e[1] for e in entries]), device=table.device)
        cols = torch.as_tensor(np.stack([e[2] for e in entries]), device=table.device)
        blocks = matrix[rows[:, :, None], cols[:, None, :]]
        scores = scores.index_copy(0, ids, block_scores(blocks, arm, stats))
    return scores


def objective(table, group, regroup_seed, arm, reference=None, stats=None):
    if arm == "S":
        return record.objective(table, group, regroup_seed, arm, reference)
    if group.labels is None:
        raise ValueError("Authorized train labels required")
    slots = [record.assignment(group), record.assignment(group, regroup_seed)]
    predicted = [account_logits(table, a, arm, stats) for a in slots]
    truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
    losses = [record.parent.ranking.objectives(v, truth, .5)["total"] for v in predicted]
    supervised = (losses[0]+losses[1])/2
    mse = table.new_zeros(())
    if reference is not None:
        target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
        mse = ((predicted[0]-account_logits(target, slots[0], arm, stats))**2).mean()
    result = supervised if reference is None else .1*supervised+.5*mse
    return result, {"supervised": float(supervised.detach()), "mse0": float(mse.detach()),
                    "mse1": None, "distillation": float(mse.detach()), "weighted": float(result.detach())}


def update(model, optimizer, current, history, reference, c, arm, stage, step, check=lambda: None, observe=None):
    stats = c.setdefault("pooling_stats", {})
    result = record.update(model, optimizer, current, history, reference, c, arm, RNG_ORDER,
        stage, step, check, observe=observe,
        objective_fn=lambda t, g, seed, a, ref: objective(t, g, seed, a, ref, stats))
    for role, g in (("current", current), ("history", history)):
        if g is not None:
            key = data.seed_for(20260918, RNG_ORDER, stage, step, role, g.uid, "regroup")
            result[role]["assignment_sha256"] = hashlib.sha256(record.assignment(g, key).tobytes()).hexdigest()
    result["pooling"] = identity(arm)
    result["solver_summary"] = stats.copy()
    return result


def score(model: Any, groups: list, c: dict, arm: str, check=lambda: None):
    if not groups:
        return np.empty((0, 378), np.float32)
    if arm == "S":
        return record.score(model, groups, c, check)
    model.eval()
    with torch.inference_mode():
        rows = [account_logits(record.pair_table(model, record.record_vectors(model, g, c, check), check),
                              record.assignment(g), arm, c.setdefault("pooling_stats", {})).float().cpu().numpy()
                for g in groups]
    result = np.stack(rows).astype(np.float32)
    if result.shape != (len(groups), 378) or not np.isfinite(result).all():
        raise ValueError("Incomplete/nonfinite blind predictions")
    return result


def restored_score(model, groups, c, check=lambda: None):
    arm = bind(model, {"pooling": getattr(model, "pooling_identity", None)})
    return score(model, groups, c, arm, check)


def capture(model, optimizer, current, memory, valid, c, arm, stage, epoch, root, point, check=lambda: None):
    import step28_replay_diagnostics as diag
    return diag.capture(model, optimizer, current, memory, valid, c, arm, RNG_ORDER, stage, epoch, root,
        point, check, scope="AB", score_fn=lambda m, gs, cfg, a, ck: score(m, gs, cfg, a, ck),
        pooling_fn=lambda table, slots: account_logits(table, slots, arm, c.setdefault("pooling_stats", {})))
