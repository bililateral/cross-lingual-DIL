"""Single fixed half-mean/half-LogAvgExp intervention; no data entry."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch

import step28_record_replay as record

data = record.data
ARMS = ("G_half",)
RNG_ORDER = "ABC"


def identity(arm: str = "G_half") -> dict:
    if arm != "G_half":
        raise ValueError("Only the fixed G_half candidate is authorized")
    return {"arm": arm, "mean_weight": .5, "logavgexp_weight": .5, "tau": 1.,
            "record_order": "canonical_offdiagonal_upper_v1", "output_dtype": "float32",
            "numerics": {"pooling": "float64_then_one_float32_cast", "encoder_forward": "bfloat16",
                         "head_parameters_gradients_adam": "float32", "tf32": False,
                         "deterministic_algorithms": True}}


def bind(model, metadata: dict) -> str:
    if metadata.get("pooling") != identity():
        raise ValueError("Missing or incompatible G_half metadata")
    model.pooling_identity = identity()
    return "G_half"


def block_scores(blocks, arm: str = "G_half", stats: dict | None = None):
    identity(arm)
    if (blocks.ndim != 3 or not 2 <= blocks.shape[1] <= 8 or not 2 <= blocks.shape[2] <= 8
            or blocks.dtype not in (torch.float32, torch.float64) or not torch.isfinite(blocks).all()):
        raise ValueError("Finite complete 2..8 record blocks required")
    values = blocks.double().flatten(1)
    result = .5*values.mean(1)+.5*(torch.logsumexp(values, 1)-math.log(values.shape[1]))
    result = result.to(blocks.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite G_half output")
    return result


def account_logits(table, slots, arm: str, stats: dict | None = None, context=None):
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


def objective(table, group, regroup_seed, arm, reference=None, stats=None, context=None):
    identity(arm)
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
    return result


def score(model: Any, groups: list, c: dict, arm: str, check=lambda: None):
    if not groups:
        return np.empty((0, 378), np.float32)
    identity(arm)
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
