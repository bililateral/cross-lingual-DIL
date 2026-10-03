"""Group-stratified one-sided task-risk retention; no archive or evaluation loader.

The stage API consumes only current fit groups and a self-contained live memory.
Formal experiment scheduling/authorization is deliberately outside this module.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

import step28_bge_continual as parent

base, data, core = parent.base, parent.data, parent.core
CHANNELS = ("rank", "positive", "negative")


def query_risks(predicted: Any, truth: Any) -> tuple[Any, Any]:
    """Return live [28,3] losses and positive counts, using all 27 candidates."""
    import torch

    truth = truth.to(device=predicted.device, dtype=predicted.dtype)
    base.objectives(predicted, truth, 1)  # Includes clique/degree/finite checks.
    left, right = torch.triu_indices(28, 28, 1, device=predicted.device)
    scores = predicted.new_zeros((28, 28))
    scores[left, right], scores[right, left] = predicted, predicted
    positive = torch.zeros((28, 28), dtype=torch.bool, device=predicted.device)
    positive[left, right], positive[right, left] = truth.bool(), truth.bool()
    diagonal = torch.eye(28, dtype=torch.bool, device=predicted.device)
    negative = ~positive & ~diagonal
    counts = positive.sum(1)
    rank = (torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), dim=1)
            - scores.masked_fill(~positive, 0).sum(1) / counts)
    pos = torch.nn.functional.softplus(-scores).masked_fill(~positive, 0).sum(1) / counts
    neg = torch.nn.functional.softplus(scores).masked_fill(~negative, 0).sum(1) / negative.sum(1)
    return torch.stack((rank, pos, neg), dim=1), counts


def validate_reference(reference: dict) -> None:
    if set(reference) != {"1", "2"}:
        raise ValueError("Need separate one/two-positive reference strata")
    for degree, count in (("1", 16), ("2", 12)):
        values = np.asarray(reference[degree], dtype=np.float64)
        if (values.shape != (count, 3) or not np.isfinite(values).all()
                or (values < 0).any() or (np.diff(values, axis=0) < 0).any()):
            raise ValueError("Reference must contain finite sorted nonnegative task risks")


def make_reference(predicted: np.ndarray, labels: tuple[int, ...]) -> dict:
    import torch

    if predicted.shape != (378,) or predicted.dtype != np.float32:
        raise ValueError("Origin-stage reference requires float32 complete logits")
    with torch.no_grad():
        risks, counts = query_risks(torch.from_numpy(predicted.copy()), torch.tensor(labels))
        result = {str(degree): torch.sort(risks[counts == degree], dim=0, stable=True).values.tolist()
                  for degree in (1, 2)}
    validate_reference(result)
    return result


def distribution_penalty(risks: Any, counts: Any, reference: dict) -> Any:
    """Per-channel penalties; query-uniform weights 16/28 and 12/28.

    Channels have separate marginal orderings. This does not preserve their joint
    dependence or each individual's loss. Sorted VALUES remain on the live graph.
    """
    import torch

    validate_reference(reference)
    if (risks.shape != (28, 3) or counts.shape != (28,) or not torch.isfinite(risks).all()
            or int((counts == 1).sum()) != 16 or int((counts == 2).sum()) != 12):
        raise ValueError("Risk/count shape or stratum cardinality differs")
    total = risks.new_zeros(3)
    for degree in (1, 2):
        ordered = torch.sort(risks[counts == degree], dim=0, stable=True).values
        target = torch.tensor(reference[str(degree)], dtype=risks.dtype, device=risks.device)
        total = total + (ordered - target).clamp_min(0).square().sum(0) / 28
    return total


class RiskMemory:
    """Reuse Algorithm R/draw RNG; charge all references inside parent auxiliary."""

    def __init__(self, order: str, seed: int, first_map: dict):
        self.cache = parent.Memory(order, seed, False, first_map)
        self.cache.auxiliary["risk_replay"] = {"schema": 1, "groups": {}}

    def to_bytes(self) -> bytes:
        spec = self.cache.auxiliary["risk_replay"]
        if spec["schema"] != 1 or self.cache.with_logits:
            raise ValueError("Wrong risk memory schema")
        groups = {g.uid for g in self.cache.reservoir.groups}
        if set(spec["groups"]) != groups:
            raise ValueError("Only live groups may own risk references")
        for item in spec["groups"].values():
            if item["stage"] not in (1, 2):
                raise ValueError("Invalid reference origin")
            validate_reference(item["values"])
        return self.cache.to_bytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> RiskMemory:
        cache = parent.Memory.from_bytes(payload)
        result = cls(cache.order, cache.seed, cache.first_map)
        result.cache = cache
        if result.to_bytes() != payload:
            raise ValueError("Risk memory restore differs")
        return result

    def retain(self, current: list, stage: int, model: Any, optimizer: Any, c: dict,
               check: Callable[[], None] = lambda: None) -> None:
        """Only newcomers scored, at stage end in eval mode; survivors never refresh."""
        if parent.adam_step(optimizer) != stage * 288:
            raise ValueError("References require a completed originating stage")
        candidate = parent.Memory.from_bytes(self.cache.to_bytes())
        old = candidate.auxiliary.pop("risk_replay")["groups"]
        candidate.retain(current, stage, None)
        live = candidate.reservoir.groups
        retained = {g.uid: old[g.uid] for g in live if g.uid in old}
        newcomers = [g for g in live if g.uid not in old]
        was_training = model.training
        try:
            if newcomers:
                values = parent.ranking.score(model, newcomers, c, check)
                for group, row in zip(newcomers, values[::-1], strict=True):
                    retained[group.uid] = {"stage": stage, "values": make_reference(row, group.labels)}
        finally:
            model.train(was_training)
        candidate.auxiliary["risk_replay"] = {"schema": 1, "groups": retained}
        candidate.to_bytes()
        self.cache = candidate
        self.to_bytes()

    def begin_stage(self, stage: int) -> None:
        self.to_bytes()
        self.cache.begin_stage(stage)

    def draw(self) -> tuple[Any, dict]:
        group, _ = self.cache.draw()
        item = self.cache.auxiliary["risk_replay"]["groups"][group.uid]
        if item["stage"] >= self.cache.draw_stage:
            raise ValueError("Future/current risk reference entered history")
        return group, copy.deepcopy(item["values"])


def update(model: Any, optimizer: Any, current: Any, history: Any, reference: dict,
           c: dict, stage: int, step: int, current_seed: int, history_seed: int,
           *, history_weight: float = .1, retention_weight: float = .5,
           observe: bool = False,
           check: Callable[[], None] = lambda: None) -> dict:
    """Two forwards/backwards, one common clip and Adam step; no extra teacher."""
    import torch

    if (stage not in (2, 3) or current.uid == history.uid
            or current.labels is None or history.labels is None
            or len(optimizer.param_groups) != 2
            or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1
            or not np.isfinite([history_weight, retention_weight]).all()
            or history_weight < 0 or retention_weight < 0):
        raise ValueError("Invalid supervised continuation or coefficients")
    validate_reference(reference)
    rates = (parent.stage_lr(step), .001)
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    model.train()
    result = {}
    before = {name: core.state_digest(module.state_dict()) for name, module in
              (("encoder", model.encoder), ("head", model.head))} if observe else {}
    for role, group, seed, weight in (("current", current, current_seed, 1.),
                                     ("history", history, history_seed, history_weight)):
        torch.manual_seed(seed)
        scores = base.logits(model, group, c, "split_rank", check)
        truth = torch.tensor(group.labels, dtype=scores.dtype, device=scores.device)
        terms = parent.ranking.objectives(scores, truth, .5)
        objective = weight * terms["total"]
        if role == "history":
            risks, counts = query_risks(scores, truth)
            penalties = distribution_penalty(risks, counts, reference)
            # Three equal channels, independent of the history-supervision weight.
            objective = objective + retention_weight * penalties.mean()
            result.update({"retention_" + name: float(value.detach())
                           for name, value in zip(CHANNELS, penalties, strict=True)})
            result["retention"] = float(penalties.mean().detach())
            if observe and step == 1:
                probe = model.head[0].weight
                result["head_gradient_diagnostics"] = {
                    name: float(torch.autograd.grad(term, probe, retain_graph=True)[0].float().norm())
                    for name, term in [("weighted_history", history_weight * terms["total"]),
                                       *[(name, retention_weight * penalties[i] / 3)
                                         for i, name in enumerate(CHANNELS)]]}
        objective.backward()
        result.update({role + "_" + name: float(value.detach()) for name, value in terms.items()})
        # Release current graph before the historical forward.
        del scores, truth, terms, objective
    result["total"] = (result["current_total"] + history_weight * result["history_total"]
                       + retention_weight * result["retention"])
    evidence = {}
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
                raise ValueError("No finite task gradient in " + name)
            evidence[name] = {"finite_nonzero_combined_gradient": True}
    result["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), c["optimizer"]["clip_norm"], error_if_nonfinite=True))
    check()
    optimizer.step()
    if parent.adam_step(optimizer) != (stage - 1) * 288 + step:
        raise ValueError("Exactly one continuous Adam step required")
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            changed = before[name] != core.state_digest(module.state_dict())
            if changed != (name == "head" or rates[0] > 0):
                raise ValueError("Unexpected parameter update: " + name)
            evidence[name]["parameters_changed"] = changed
    result["modules"] = evidence
    result["weighted_history_total"] = history_weight * result["history_total"]
    result.update(stage=stage, step=step, adam_step=parent.adam_step(optimizer),
                  history_weight=history_weight, retention_weight=retention_weight,
                  encoder_lr=rates[0], head_lr=rates[1])
    return result


def train_stage(model: Any, optimizer: Any, current: list, memory: RiskMemory, c: dict,
                order: str, stage: int, *, history_weight: float = .1,
                retention_weight: float = .5, check: Callable[[], None] = lambda: None) -> dict:
    """Complete 288-update learner entry; fit groups only, no evaluator/archive."""
    policy = parent.contract()
    if memory.cache.order != order or memory.cache.seed != policy["memory_seed"]:
        raise ValueError("Memory order/seed differs from the paired schedule")
    if {g.uid for g in current} & {g.uid for g in memory.cache.reservoir.groups}:
        raise ValueError("Current fit groups overlap history")
    sequence, stream = parent.schedule(current, policy, order, stage)
    memory.begin_stage(stage)
    records, history_ids = [], []
    for index, group in enumerate(sequence):
        history, reference = memory.draw()
        history_ids.append(history.uid)
        records.append(update(model, optimizer, group, history, reference, c, stage, index + 1,
                              data.seed_for(stream, index, "dropout"),
                              data.seed_for(policy["memory_seed"], order, stage, index, "history_dropout"),
                              history_weight=history_weight, retention_weight=retention_weight, check=check))
    return {"current_ids": [g.uid for g in sequence], "history_ids": history_ids,
            "updates": records, "physical_updates": len(records)}
