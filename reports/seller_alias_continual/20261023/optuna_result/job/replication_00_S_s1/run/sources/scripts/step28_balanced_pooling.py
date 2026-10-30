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
SOLVER = dict(version="log_sinkhorn_terminal_q_v1", tau=1., maximum_iterations=4096,
              marginal_tolerance=1e-10, feasible_tolerance=1e-12,
              gap_tolerance=5e-9, roundoff_factor=256.)


class SolverFailure(ValueError):
    """Small numeric failure receipt; never retain a table or teacher payload."""
    def __init__(self, message, indices, n, m, iteration, **residuals):
        super().__init__(message)
        self.details = {"reason": message, "failed_blocks": []}
        for i, index in enumerate(indices.tolist()):
            row = {"batch_index": index, "n": n, "m": m, "iteration": iteration,
                   "marginal_error": None, "feasible_error": None, "gap_checked": None}
            for key, values in residuals.items():
                value = float(values[i])
                row[key] = value if math.isfinite(value) else None
            self.details["failed_blocks"].append(row)


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
    except SolverFailure as exc:
        if stats is not None:
            stats["failures"] = stats.get("failures", 0)+1
        exc.details["failure_count"] = stats["failures"] if stats is not None else 1
        raise


def _solve(blocks: torch.Tensor, stats: dict | None, *, maximum_iterations: int):
    """Batched, unpadded n*m blocks. Precision limits are never adapted to data."""
    if (blocks.ndim != 3 or not 2 <= blocks.shape[1] <= 8 or not 2 <= blocks.shape[2] <= 8
            or not 1 <= maximum_iterations <= SOLVER["maximum_iterations"]):
        raise ValueError("Unsupported transport shape or iteration budget")
    k = blocks.double()
    batch, n, m = k.shape
    if not torch.isfinite(k).all():
        raise SolverFailure("Nonfinite transport input", torch.where(~torch.isfinite(k).all((1, 2)))[0], n, m, 0)
    mu = k.mean((1, 2))
    w = k-k.mean(2, keepdim=True)-k.mean(1, keepdim=True)+mu[:, None, None]
    kernel = w-math.log(n*m)  # tau = 1, uniform U.
    a, b = k.new_zeros((batch, n)), k.new_zeros((batch, m))
    done = torch.zeros(batch, dtype=torch.bool, device=k.device)
    values, plans = mu.clone(), k.new_zeros(k.shape)
    maxima = dict(iterations=0, marginal_error=0., feasible_error=0., gap_checked=0.,
                  ordinary_marginal_error=None, terminal_acceptances=0, terminal_min_iteration=None)
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
        bad = (er < -eps[:, None]).any(1) | (ec < -eps[:, None]).any(1)
        if bad.any():
            raise SolverFailure("Negative transport rounding mass", idx[bad], n, m, iteration,
                                marginal_error=torch.maximum(rowerr, colerr)[bad])
        er, ec = er.clamp_min(0), ec.clamp_min(0)
        mass = er.sum(1)
        q = q+er[:, :, None]*ec[:, None, :]/torch.where(mass > 0, mass, 1.)[:, None, None]
        primal = (q*w[idx]-torch.special.xlogy(q, q*(n*m))).sum((1, 2))
        dual = p.sum((1, 2))-1-aw.mean(1)-bw.mean(1)
        gap = dual-primal
        bad = (~torch.isfinite(p).all((1, 2)) | ~torch.isfinite(q).all((1, 2))
               | ~torch.isfinite(primal) | ~torch.isfinite(dual) | ~torch.isfinite(eps) | (q < 0).any((1, 2)))
        if bad.any():
            raise SolverFailure("Nonfinite or negative transport result", idx[bad], n, m, iteration,
                                marginal_error=torch.maximum(rowerr, colerr)[bad])
        bad = gap < -eps
        if bad.any():
            raise SolverFailure("Transport dual bound below primal beyond roundoff", idx[bad], n, m, iteration,
                                marginal_error=torch.maximum(rowerr, colerr)[bad], gap_checked=gap[bad])
        checked = gap.clamp_min(0)+eps
        feasible = torch.maximum((q.sum(2)-1/n).abs().amax(1), (q.sum(1)-1/m).abs().amax(1))
        marginal = torch.maximum(rowerr, colerr)
        qualified = (feasible <= SOLVER["feasible_tolerance"]) & (checked <= SOLVER["gap_tolerance"])
        ordinary = qualified & (marginal <= SOLVER["marginal_tolerance"])
        terminal = qualified & ~ordinary & (iteration == SOLVER["maximum_iterations"] == 4096)
        accept = ordinary | terminal
        if accept.any():
            finished = idx[accept]
            plans[finished], values[finished] = q[accept], mu[finished]+primal[accept]
            done[finished] = True
            maxima["iterations"] = iteration
            for name, tensor in (("marginal_error", marginal), ("feasible_error", feasible), ("gap_checked", checked)):
                maxima[name] = max(maxima[name], float(tensor[accept].max()))
            if ordinary.any():
                maxima["ordinary_marginal_error"] = max(maxima["ordinary_marginal_error"] or 0., float(marginal[ordinary].max()))
            if terminal.any():
                maxima["terminal_acceptances"] += int(terminal.sum())
                maxima["terminal_min_iteration"] = min(maxima["terminal_min_iteration"] or iteration, iteration)
        if done.all():
            if stats is not None:
                stats["blocks"] = stats.get("blocks", 0)+batch
                stats["failures"] = stats.get("failures", 0)
                for key in ("iterations", "marginal_error", "feasible_error", "gap_checked"):
                    value = maxima[key]
                    stats[key] = max(stats.get(key, 0), value)
                stats["terminal_acceptances"] = stats.get("terminal_acceptances", 0)+maxima["terminal_acceptances"]
                for key, op in (("ordinary_marginal_error", max), ("terminal_min_iteration", min)):
                    values_to_merge = [v for v in (stats.get(key), maxima[key]) if v is not None]
                    stats[key] = op(values_to_merge) if values_to_merge else None
            return values, plans, maxima
    failed = ~accept
    raise SolverFailure(f"Transport did not converge in {maximum_iterations} full iterations", idx[failed], n, m,
                        maximum_iterations, marginal_error=marginal[failed], feasible_error=feasible[failed], gap_checked=checked[failed])


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


def account_logits(table, slots, arm: str, stats: dict | None = None, context=None):
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
        try:
            scores = scores.index_copy(0, ids, block_scores(blocks, arm, stats))
        except SolverFailure as exc:
            pairs = list(zip(*np.triu_indices(28, 1)))
            exc.details.update(context or {}, arm=arm)
            for failed in exc.details["failed_blocks"]:
                index = entries[failed["batch_index"]][0]
                failed.update(account_pair_index=index, account_slots=[int(x) for x in pairs[index]])
            raise
    return scores


def objective(table, group, regroup_seed, arm, reference=None, stats=None, context=None):
    if arm == "S":
        return record.objective(table, group, regroup_seed, arm, reference)
    if group.labels is None:
        raise ValueError("Authorized train labels required")
    slots = [record.assignment(group), record.assignment(group, regroup_seed)]
    context = {**(context or {}), "group_uid": group.uid}
    predicted = [account_logits(table, a, arm, stats, {**context, "view": "A"+str(i)}) for i, a in enumerate(slots)]
    truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
    losses = [record.parent.ranking.objectives(v, truth, .5)["total"] for v in predicted]
    supervised = (losses[0]+losses[1])/2
    mse = table.new_zeros(())
    if reference is not None:
        target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
        mse = ((predicted[0]-account_logits(target, slots[0], arm, stats, {**context, "view": "teacher_A0"}))**2).mean()
    result = supervised if reference is None else .1*supervised+.5*mse
    return result, {"supervised": float(supervised.detach()), "mse0": float(mse.detach()),
                    "mse1": None, "distillation": float(mse.detach()), "weighted": float(result.detach())}


def update(model, optimizer, current, history, reference, c, arm, stage, step, check=lambda: None, observe=None):
    stats = c.setdefault("pooling_stats", {})
    result = record.update(model, optimizer, current, history, reference, c, arm, RNG_ORDER,
        stage, step, check, observe=observe,
        objective_fn=lambda t, g, seed, a, ref: objective(t, g, seed, a, ref, stats,
            {"stage": stage, "step": step, "role": "current" if ref is None else "history"}))
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
                              record.assignment(g), arm, c.setdefault("pooling_stats", {}),
                              {**c.get("pooling_context", {}), "group_uid": g.uid, "view": "A0"}).float().cpu().numpy()
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
    cfg = {**c, "pooling_stats": c.setdefault("pooling_stats", {}),
           "pooling_context": {"stage": stage, "step": epoch*48, "role": "diagnostic_prediction"}}
    return diag.capture(model, optimizer, current, memory, valid, cfg, arm, RNG_ORDER, stage, epoch, root,
        point, check, scope="AB", score_fn=lambda m, gs, cfg, a, ck: score(m, gs, cfg, a, ck),
        pooling_fn=lambda table, slots, g, view: account_logits(table, slots, arm, c["pooling_stats"],
            {"stage": stage, "step": epoch*48, "role": "history_diagnostic", "group_uid": g.uid, "view": view}))
