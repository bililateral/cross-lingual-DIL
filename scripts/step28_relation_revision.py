"""Task-aligned revision retaining cumulative relation-objective memory.

No formal-data CLI. The closed pilot keeps its original module and source gate.
"""
from __future__ import annotations

from typing import Any, Callable

import step28_relation_memory as original

HISTORY_RANK_WEIGHT = .1


def current_objective(scores: Any, labels: Any) -> dict:
    import torch
    truth = torch.as_tensor(labels, dtype=scores.dtype, device=scores.device)
    return original.parent.ranking.objectives(scores, truth, .5)


def historical_ranking(scores: Any, labels: Any) -> Any:
    """Full-query logistic ranking; no fixed upper target for correct margins."""
    import torch
    if scores.shape != (378,) or not torch.isfinite(scores).all():
        raise ValueError("Expected complete finite historical scores")
    pairs = original.query_indices(labels, scores.device)
    return torch.stack([
        torch.nn.functional.softplus(scores[n][None, :] - scores[p, None]).mean()
        for p, n in pairs
    ]).mean()


def history_objective(y: Any, weight: Any, x: Any,
                      memory: original.Memory, labels: Any) -> dict:
    compressed = original.historical_loss(
        y, weight, x, memory.h, memory.b, memory.constant, memory.count)
    # Use the same live scores and their encoder/head/weight graph.
    rank = historical_ranking(y.T @ weight, labels)
    return {"compressed": compressed, "rank": rank,
            "total": compressed + HISTORY_RANK_WEIGHT * rank}


def optimization_step(model: Any, optimizer: Any, current: Any, history: Any,
                      x: Any, memory: original.Memory, c: dict, encoder_lr: float,
                      *, check: Callable[[], None] = lambda: None) -> dict:
    import torch
    previous = original.parent.adam_step(optimizer)
    if (len(optimizer.param_groups) != 2 or (history is None) != (x is None)
            or (history is not None and (current.uid == history.uid or memory.count <= 0))):
        raise ValueError("Invalid current/history kernel inputs")
    for group, rate in zip(optimizer.param_groups, (encoder_lr, .001), strict=True):
        group["lr"] = rate
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z = original.relation_features(model, current, c, check)
    terms = current_objective(z.T @ model.weight, current.labels)
    terms["total"].backward()
    result = {"current_" + key: float(value.detach()) for key, value in terms.items()}
    del z, terms
    result.update(history_compressed=0., history_rank=0., history_total=0., history_uid=None)
    if history is not None:
        z = original.relation_features(model, history, c, check)
        terms = history_objective(z, model.weight, x, memory, history.labels)
        terms["total"].backward()
        result.update({"history_" + key: float(value.detach()) for key, value in terms.items()})
        result["history_uid"] = history.uid
        del z, terms
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    check()
    optimizer.step()
    check()
    if original.parent.adam_step(optimizer) != previous + 1:
        raise ValueError("Expected exactly one AdamW update")
    return {**result, "total": result["current_total"] + result["history_total"],
            "gradient_norm": float(norm), "adam_step": previous + 1,
            "encoder_lr": encoder_lr, "head_lr": .001,
            "history_rank_weight": HISTORY_RANK_WEIGHT}


def update(model: Any, optimizer: Any, current: Any, memory: original.Memory, c: dict,
           stage: int, step: int, *, check: Callable[[], None] = lambda: None) -> dict:
    if (stage not in (1, 2, 3) or not 1 <= step <= 288
            or original.parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1
            or memory.stage != stage - 1
            or (stage > 1 and (memory.draw_stage != stage or memory.draw_count != step - 1))
            or any(current.uid == g.uid for g in memory.reservoir.groups)):
        raise ValueError("Update phase or current/history separation differs")
    history, x = memory.draw() if stage > 1 else (None, None)
    result = optimization_step(model, optimizer, current, history, x, memory, c,
                               original.parent.stage_lr(step), check=check)
    return {**result, "stage": stage, "step": step}


def train_stage(model: Any, optimizer: Any, current: list, memory: original.Memory,
                c: dict, stage: int, check: Callable[[], None] = lambda: None) -> list[dict]:
    # Reuse the old scheduling definition only; it does not authorize a new pilot.
    sequence, _ = original.parent.schedule(current, original.parent.contract(), memory.order, stage)
    if stage > 1:
        memory.begin_stage(stage)
    records = [update(model, optimizer, g, memory, c, stage, i + 1, check=check)
               for i, g in enumerate(sequence)]
    if stage < 3:
        memory.consolidate(current, stage, lambda g: original.reference(model, g, c, check))
    return records
