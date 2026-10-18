"""Fixed S_pos/H_pos objectives on the existing record VJP; no data entry."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

import step28_g_half as h

record, data = h.record, h.data
ARMS = ("S_pos", "H_pos")
RNG_ORDER = "ABC"


def identity(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Only fixed S_pos/H_pos candidates are supported")
    if arm == "H_pos":
        return h.identity()
    return dict(arm="S", pooling="complete_record_mean", record_order="canonical_offdiagonal_upper_v1",
                output_dtype="float32", numerics="original_FP32_matrix_mean")


def training_identity(arm: str) -> dict:
    identity(arm)
    return dict(arm=arm, rank="mean_positive_logsumexp_zero_and_all_negative_differences",
                temperature=1., queries=28, candidates=27, positive_counts=[1, 2],
                reduction="mean_positive_then_mean_query", precision="actual_FP32_logits_FP32_rank",
                current_supervision=1., history_supervision=.1, teacher_A0=.5, hard_weight=.5,
                optimizer="AdamW", rng_order=RNG_ORDER)


def metadata(arm: str) -> dict:
    return dict(pooling=identity(arm), training=training_identity(arm))


def bind(model: Any, meta: dict) -> str:
    arm = meta.get("training", {}).get("arm")
    if any(meta.get(k) != v for k, v in metadata(arm).items()):
        raise ValueError("Candidate scoring/training identity mismatch")
    model.pooling_identity, model.training_identity = identity(arm), training_identity(arm)
    return arm


def bound_arm(model: Any) -> str:
    return bind(model, dict(pooling=getattr(model, "pooling_identity", None),
                            training=getattr(model, "training_identity", None)))


def account_logits(table: Any, slots: np.ndarray, arm: str) -> Any:
    identity(arm)
    return (record.account_logits(table, slots) if arm == "S_pos"
            else h.account_logits(table, slots, "G_half"))


def objectives(predicted: Any, truth: Any, *, old_rank: bool = False) -> dict:
    """Replace rank, retaining the established label validation, BCE and hard."""
    inherited = record.parent.ranking.objectives(predicted, truth, .5)
    scores, positive, _ = record.parent.ranking.hard_candidates(predicted, truth)
    diagonal = torch.eye(28, dtype=torch.bool, device=predicted.device)
    negative = ~positive & ~diagonal
    terms = []
    for q in range(28):
        pos, neg = scores[q, positive[q]], scores[q, negative[q]]
        differences = neg[None, :] - pos[:, None]
        terms.append(torch.logsumexp(torch.cat((differences.new_zeros((len(pos), 1)), differences), 1), 1).mean())
    rank_pos = torch.stack(terms).mean()
    if not torch.isfinite(rank_pos):
        raise ValueError("Nonfinite positive ranking loss")
    rank_used = inherited["rank"] if old_rank else rank_pos
    # Do not add rank_pos to inherited total: that would retain the old rank.
    return dict(bce=inherited["bce"], rank_pos=rank_pos, hard=inherited["hard"],
                rank_used=rank_used, total=inherited["bce"]+rank_used+.5*inherited["hard"])


def objective(table: Any, group: Any, regroup_seed: int, arm: str,
              reference: np.ndarray | None = None, *, old_rank: bool = False) -> tuple[Any, dict]:
    identity(arm)
    if group.labels is None:
        raise ValueError("Authorized training labels required")
    slots = (record.assignment(group), record.assignment(group, regroup_seed))
    predictions = [account_logits(table, a, arm) for a in slots]
    truth = torch.as_tensor(group.labels, dtype=table.dtype, device=table.device)
    pieces = [objectives(p, truth, old_rank=old_rank) for p in predictions]
    supervised = (pieces[0]["total"]+pieces[1]["total"])/2
    mse = table.new_zeros(())
    if reference is not None:
        target = torch.as_tensor(reference, dtype=table.dtype, device=table.device)
        mse = (predictions[0]-account_logits(target, slots[0], arm)).square().mean()
    weight = 1. if reference is None else .1
    supervised_contribution, teacher_contribution = weight*supervised, .5*mse
    total = supervised_contribution+teacher_contribution
    if not torch.isfinite(total):
        raise ValueError("Nonfinite complete objective")
    return total, dict(supervised=float(supervised.detach()), mse0=float(mse.detach()), mse1=None,
        distillation=float(mse.detach()), supervised_contribution=float(supervised_contribution.detach()),
        teacher_contribution=float(teacher_contribution.detach()), weighted=float(total.detach()),
        qualification_old_rank=old_rank,
        views=[{k: float(v.detach()) for k, v in p.items()} for p in pieces])


def update(model: Any, optimizer: Any, current: Any, history: Any, reference: Any, c: dict,
           arm: str, stage: int, step: int, check=lambda: None, observe=None,
           *, qualification_old_rank: bool = False, table_observer=None) -> dict:
    if bound_arm(model) != arm:
        raise ValueError("Bound candidate differs from requested update")
    if (qualification_old_rank or table_observer is not None) and c.get("handwritten_qualification") is not True:
        raise ValueError("Only handwritten qualification can switch rank or observe tables")
    if (history is None) != (reference is None):
        raise ValueError("Historical group and its own birth table must be paired")
    def target(table, group, seed, candidate, ref):
        value, log = objective(table, group, seed, candidate, ref, old_rank=qualification_old_rank)
        if table_observer is not None:
            table_observer(table.detach(), group, seed, ref)
        return value, log
    result = record.update(model, optimizer, current, history, reference, c, arm, RNG_ORDER,
                           stage, step, check, observe=observe, objective_fn=target)
    for role, group in (("current", current), ("history", history)):
        if group is not None:
            seed = data.seed_for(20260918, RNG_ORDER, stage, step, role, group.uid, "regroup")
            result[role]["assignment_sha256"] = hashlib.sha256(record.assignment(group, seed).tobytes()).hexdigest()
    result.update(metadata(arm))
    return result


def score(model: Any, groups: list, c: dict, arm: str, check=lambda: None) -> np.ndarray:
    if bound_arm(model) != arm:
        raise ValueError("Scoring candidate differs from restored training identity")
    return (record.score(model, groups, c, check) if arm == "S_pos"
            else h.score(model, groups, c, "G_half", check))


def restored_score(model: Any, groups: list, c: dict, check=lambda: None) -> np.ndarray:
    return score(model, groups, c, bound_arm(model), check)


def capture(model: Any, optimizer: Any, current: list, memory: Any, valid: list, c: dict,
            arm: str, stage: int, epoch: int, root, point: str, check=lambda: None) -> dict:
    import step28_replay_diagnostics as diag
    result = diag.capture(model, optimizer, current, memory, valid, c, arm, RNG_ORDER, stage, epoch,
        root, point, check, scope="AB", score_fn=score,
        pooling_fn=lambda table, slots, group, view: account_logits(table, slots, arm))
    path = root/result["path"]
    row = data.read_json(path)
    row.update(metadata(arm))
    data.write_json(path, row)
    return data.record(path, root)
