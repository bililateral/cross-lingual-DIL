"""Confirmed ER strength intervention; parent scientific sources remain unchanged."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_bge_continual as parent
import step28_bge_continual_run as prior

base, data, core, metrics = parent.base, parent.data, parent.core, parent.metrics
POLICY = data.ROOT / "schema/step28_er_weight_policy.json"
POLICY_SHA256 = "9eddf92646892d5890491e068ecbd4834463d0751de530a4701b7de8a98e4396"
ARMS = {"half": .5, "quarter": .25}
ORDERS, ROLES = parent.ORDERS, parent.ROLES
STEP_COLUMNS = ("current_bce", "current_rank", "current_hard", "current_total",
                "history_bce", "history_rank", "history_hard", "history_total",
                "weighted_history_total", "total", "encoder_lr", "head_lr",
                "gradient_norm", "history_weight")


def contract() -> dict:
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("ER weight policy changed")
    p = data.read_json(POLICY)
    old = parent.contract()
    if (p["arms"] != ARMS or p["orders"] != list(ORDERS) or p["stages"] != [2, 3]
            or p["physical_updates"] != 3456 or p["gradient_group_presentations"] != 6912
            or p["parent_policy_sha256"] != parent.POLICY_SHA256
            or old["memory"]["capacity_groups"] != 6
            or p["supervision"]["test_access"] or p["supervision"]["owners_access"]):
        raise ValueError("Confirmed intervention or boundaries differ")
    return p


def config(p: dict) -> dict:
    result = parent.config(parent.contract())
    result["runtime"] = copy.deepcopy(p["runtime"])
    return result


def sources() -> list[dict]:
    additional = ["docs/SELLER_ALIAS_ER_WEIGHT.zh.md", "schema/step28_er_weight_policy.json",
                  "scripts/step28_er_weight.py", "scripts/step28_er_weight_run.py",
                  "scripts/step28_er_weight_evaluate.py", "scripts/step28_er_weight_check.py",
                  "tests/test_step28_er_weight_contracts.py",
                  "scripts/run_step28_er_weight_linux_20261001.sh",
                  "scripts/run_step28_er_check_linux_20261001.sh",
                  "reports/documentation/20261001/er_weight/decision.json"]
    rows = prior.sources() + [data.record(data.ROOT / name, data.ROOT) for name in additional]
    return sorted(rows, key=lambda row: row["path"])


def point_name(order: str, arm: str, stage: int) -> str:
    if order not in ORDERS or arm not in ARMS or stage not in (2, 3):
        raise ValueError("Unknown new ER endpoint")
    return f"{order}_{arm}_stage{stage}"


def expected_points() -> list[str]:
    return [point_name(order, arm, stage) for order in ORDERS for arm in ARMS for stage in (2, 3)]


def baseline(p: dict, job: Path | None = None) -> dict:
    """Verify pinned small records; native state and cache are checked when restored."""
    job = job if job is not None else data.ROOT / p["baseline"]["linux_job"]
    records = {name: data.read_json(data.verify(job / name, rec))
               for name, rec in p["baseline"]["records"].items()}
    manifest = records["run/manifest.json"]
    collected = records["evaluation/collected.json"]
    if (manifest["status"] != prior.COMPLETE or manifest["source_files"] != prior.sources()
            or collected["source_files"] != prior.sources()
            or collected["status"] != "ALL_PILOT_MATRICES_SAVED_BEFORE_COMPARISONS"
            or records["evaluation/evaluation.json"]["status"] != "COMPLETE_BGE_CONTINUAL_DIAGNOSTIC"
            or collected["metric_columns"] != list(metrics.COLUMNS)):
        raise ValueError("Original completed evidence or frozen scientific sources differ")
    partition = records["run/partition.json"]
    if (collected["group_ids"] != [row["group_uid"] for row in partition["development"]]
            or collected["domains"] != [row["domain"] for row in partition["development"]]):
        raise ValueError("Original valid identities do not align")
    for key, value in p["baseline"]["environment"].items():
        if records["execution.json"][key] != value:
            raise ValueError("Original environment differs from pinned baseline")
    return {"job": job, "manifest": manifest, "partition": partition,
            "collected": collected, "execution": records["execution.json"]}


class ContinuationSupply(parent.Supply):
    def resume(self, path: str, order: str, shared: dict) -> None:
        """Register a hash-verified shared checkpoint without requesting old fit groups."""
        if (path in self.next_stage or order not in ORDERS or shared["order"] != order
                or shared["stage"] != 1 or shared["completed_updates"] != 288
                or not shared.get("full_model_adam_and_rng_restore_verified")
                or not shared.get("model_state_sha256")):
            raise ValueError("Continuation requires an intact completed shared first stage")
        self.next_stage[path], self.orders[path] = 2, order


def update(model: Any, optimizer: Any, current: Any, history: Any, c: dict,
           weight: float, stage: int, step: int, current_seed: int, history_seed: int,
           *, observe: bool = False, check: Callable[[], None] = lambda: None) -> dict:
    """Only lambda changes: two supervised forwards, weighted history, one clip/update."""
    import torch

    if (type(weight) not in (int, float) or weight not in (1., .5, .25)
            or stage not in (2, 3) or current is None or history is None
            or current.labels is None or history.labels is None
            or history.uid == current.uid or len(optimizer.param_groups) != 2
            or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1):
        raise ValueError("Unexpected ER continuation or supervision")
    rates = (parent.stage_lr(step), .001)
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {name: core.state_digest(module.state_dict()) for name, module in
              (("encoder", model.encoder), ("head", model.head))} if observe else {}
    result = {name: 0. for name in STEP_COLUMNS}
    for role, group, seed, coefficient in (("current", current, current_seed, 1.),
                                           ("history", history, history_seed, weight)):
        torch.manual_seed(seed)
        predicted = base.logits(model, group, c, "split_rank", check)
        truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
        terms = parent.ranking.objectives(predicted, truth, .5)
        # Keep the original current backward operation exactly; lambda applies to ALL old terms.
        objective = terms["total"] if role == "current" else coefficient * terms["total"]
        objective.backward()
        result.update({role + "_" + key: float(value.detach()) for key, value in terms.items()})
        del predicted, truth, terms, objective
    result["weighted_history_total"] = weight * result["history_total"]
    result["total"] = result["current_total"] + result["weighted_history_total"]
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
        raise ValueError("Adam did not make exactly one update")
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            changed = before[name] != core.state_digest(module.state_dict())
            if changed != (name == "head" or rates[0] > 0):
                raise ValueError("Unexpected parameter update: " + name)
            evidence[name]["parameters_changed"] = changed
    result.update(history_weight=weight, encoder_lr=rates[0], head_lr=rates[1],
                  modules=evidence, stage=stage, step=step, adam_step=parent.adam_step(optimizer))
    return result
