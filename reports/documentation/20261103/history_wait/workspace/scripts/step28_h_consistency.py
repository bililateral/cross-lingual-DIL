"""Fixed H scores with live natural/regrouped query JS; no data entry."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

import step28_g_half as h

record, data = h.record, h.data
ARMS = ("H_cons",)
RNG_ORDER = "ABC"


def identity(arm: str = "H_cons") -> dict:
    if arm != "H_cons":
        raise ValueError("Only H_cons is authorized")
    return h.identity()


def training_identity() -> dict:
    return dict(arm="H_cons", consistency="symmetric_query_JS", queries=28, candidates=27,
        candidate_order="canonical_index_excluding_self", lambda_current=1., lambda_history=.1,
        temperature=1., reduction="sum_candidates_mean_28_queries", log_base="e",
        students="both_live", mixture="live_arithmetic_half", precision="actual_H_FP32_to_FP64_one_FP32_loss_cast",
        current_supervision=1., history_supervision=.1, teacher_A0=.5, rng_order="ABC")


def metadata(arm: str = "H_cons") -> dict:
    return dict(pooling=identity(arm), training=training_identity())


def bind(model, meta: dict) -> str:
    if any(meta.get(k) != v for k, v in metadata().items()):
        raise ValueError("Missing or incompatible H_cons scoring/training metadata")
    model.pooling_identity, model.training_identity = identity(), training_identity()
    return "H_cons"


def account_logits(table, slots, arm="H_cons", stats=None, context=None):
    identity(arm)
    return h.account_logits(table, slots, "G_half", stats, context)


def query_logits(scores):
    if scores.shape != (378,) or scores.dtype not in (torch.float32, torch.float64) or not torch.isfinite(scores).all():
        raise ValueError("Finite canonical 378 H scores required")
    i, j = torch.triu_indices(28, 28, 1, device=scores.device)
    indices = torch.zeros((28, 28), dtype=torch.long, device=scores.device)
    indices[i, j] = indices[j, i] = torch.arange(378, device=scores.device)
    return scores[indices[~torch.eye(28, dtype=torch.bool, device=scores.device)].reshape(28, 27)]


def consistency(z0, z1):
    """Both student branches and mixture stay live; inputs already exclude self."""
    if (z0.shape != (28, 27) or z1.shape != z0.shape or z0.dtype != z1.dtype
            or z0.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(z0).all() or not torch.isfinite(z1).all()):
        raise ValueError("Two finite aligned 28x27 logit tables required")
    l0, l1 = torch.log_softmax(z0.double(), 1), torch.log_softmax(z1.double(), 1)
    lm = torch.logaddexp(l0, l1)-math.log(2.)
    result = (.5*(l0.exp()*(l0-lm)+l1.exp()*(l1-lm)).sum(1).mean()).to(z0.dtype)
    if not torch.isfinite(result):
        raise ValueError("Nonfinite query consistency")
    return result


def objective(table, group, regroup_seed, arm, reference=None, stats=None, context=None, *, qualification_without_c=False):
    identity(arm)
    if group.labels is None:
        raise ValueError("Authorized train labels required")
    slots = (record.assignment(group), record.assignment(group, regroup_seed))
    predicted = [account_logits(table, a, arm, stats, context) for a in slots]
    truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
    supervised = sum(record.parent.ranking.objectives(v, truth, .5)["total"] for v in predicted)/2
    coupling = consistency(*(query_logits(v) for v in predicted))
    mse = table.new_zeros(())
    if reference is not None:
        target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
        mse = (predicted[0]-account_logits(target, slots[0], arm, stats, context)).square().mean()
    weight = 1. if reference is None else .1
    c_weight = 0. if qualification_without_c else weight
    supervised_contribution, consistency_contribution = weight*supervised, c_weight*coupling
    teacher_contribution = .5*mse
    result = supervised_contribution+consistency_contribution+teacher_contribution
    return result, dict(supervised=float(supervised.detach()), consistency=float(coupling.detach()),
        supervised_contribution=float(supervised_contribution.detach()),
        consistency_contribution=float(consistency_contribution.detach()),
        teacher_contribution=float(teacher_contribution.detach()), mse0=float(mse.detach()),
        mse1=None, distillation=float(mse.detach()), weighted=float(result.detach()),
        qualification_without_c=qualification_without_c)


def update(model, optimizer, current, history, reference, c, arm, stage, step, check=lambda: None,
           observe=None, *, qualification_without_c=False, table_observer=None):
    bind(model, dict(pooling=getattr(model, "pooling_identity", None), training=getattr(model, "training_identity", None)))
    if (qualification_without_c or table_observer is not None) and c.get("handwritten_qualification") is not True:
        raise ValueError("Formal entry cannot disable C or inspect live qualification tables")
    def target(table, group, seed, candidate, ref):
        value, log = objective(table, group, seed, candidate, ref, c.setdefault("pooling_stats", {}),
                               qualification_without_c=qualification_without_c)
        if table_observer is not None:
            table_observer(table.detach(), group, seed)
        return value, log
    result = record.update(model, optimizer, current, history, reference, c, arm, RNG_ORDER,
                           stage, step, check, observe=observe, objective_fn=target)
    for role, group in (("current", current), ("history", history)):
        if group is not None:
            seed = data.seed_for(20260918, RNG_ORDER, stage, step, role, group.uid, "regroup")
            result[role]["assignment_sha256"] = hashlib.sha256(record.assignment(group, seed).tobytes()).hexdigest()
    result.update(metadata(arm))
    return result


def score(model, groups, c, arm, check=lambda: None):
    identity(arm)
    bind(model, dict(pooling=getattr(model, "pooling_identity", None), training=getattr(model, "training_identity", None)))
    return h.score(model, groups, c, "G_half", check)


def restored_score(model, groups, c, check=lambda: None):
    return score(model, groups, c, "H_cons", check)


def capture(model, optimizer, current, memory, valid, c, arm, stage, epoch, root, point, check=lambda: None):
    import step28_replay_diagnostics as diag
    result = diag.capture(model, optimizer, current, memory, valid, c, arm, RNG_ORDER, stage, epoch,
        root, point, check, scope="AB", score_fn=score,
        pooling_fn=lambda table, slots, group, view: account_logits(table, slots, arm))
    path = root/result["path"]
    value = data.read_json(path)
    value.update(metadata(arm))
    data.write_json(path, value)
    return data.record(path, root)
