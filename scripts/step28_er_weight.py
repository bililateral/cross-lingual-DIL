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
LOW_POLICY = data.ROOT / "schema/step28_er_low_policy.json"
LOW_POLICY_SHA256 = "201c7a766965df127803be89a9f3b7b1a5246053d95b2bcafa4dcc08cfa08c3d"
LOGIT_POLICY = data.ROOT / "schema/step28_logit_weight_policy.json"
LOGIT_POLICY_SHA256 = "2aaf41724055ed3dff93ad65a8be2cc4e83c8f35d4515dace34bcdc44b081deb"
LOGIT_LOW_POLICY = data.ROOT / "schema/step28_logit_low_policy.json"
LOGIT_LOW_POLICY_SHA256 = "8df56ee17a0159e6d3e56e86dadf8dab26290c6b34bc8dec04e926826af61da0"
ARMS = {"half": .5, "quarter": .25}
ORDERS, ROLES = parent.ORDERS, parent.ROLES
STEP_COLUMNS = ("current_bce", "current_rank", "current_hard", "current_total",
                "history_bce", "history_rank", "history_hard", "history_total",
                "weighted_history_total", "total", "encoder_lr", "head_lr",
                "gradient_norm", "history_weight")


def contract(study: str = "weight") -> dict:
    if study == "risk":
        from step28_risk_study import contract as risk_contract
        return risk_contract()
    if study not in ("weight", "low", "logit", "logit_low"):
        raise ValueError("Unknown confirmed ER study")
    path, digest, arms = {
        "weight": (POLICY, POLICY_SHA256, ARMS),
        "low": (LOW_POLICY, LOW_POLICY_SHA256, {"tenth": .1}),
        "logit": (LOGIT_POLICY, LOGIT_POLICY_SHA256, {"logit_quarter": .25}),
        "logit_low": (LOGIT_LOW_POLICY, LOGIT_LOW_POLICY_SHA256, {"logit_tenth": .1}),
    }[study]
    updates = len(arms) * 3 * 2 * 288
    if data.sha256(path) != digest:
        raise ValueError("ER weight policy changed")
    p = data.read_json(path)
    old = parent.contract()
    if (p["arms"] != arms or p["orders"] != list(ORDERS) or p["stages"] != [2, 3]
            or p["physical_updates"] != updates or p["gradient_group_presentations"] != 2 * updates
            or p["parent_policy_sha256"] != parent.POLICY_SHA256
            or old["memory"]["capacity_groups"] != 6
            or p["supervision"]["test_access"] or p["supervision"]["owners_access"]):
        raise ValueError("Confirmed intervention or boundaries differ")
    return p


def policy_sha256(p: dict) -> str:
    if is_risk(p):
        from step28_risk_study import POLICY_SHA256 as risk_digest
        return risk_digest
    return {"seller_alias_er_low_weight": LOW_POLICY_SHA256,
            "seller_alias_er_history_weight": POLICY_SHA256,
            "seller_alias_logit_weight": LOGIT_POLICY_SHA256,
            "seller_alias_logit_low": LOGIT_LOW_POLICY_SHA256}[p["study"]]


def with_logits(p: dict) -> bool:
    return p.get("memory_arm", "er") == "logit"


def is_risk(p: dict) -> bool:
    return p["study"] == "seller_alias_risk_replay"


def evidence_tag(p: dict) -> str:
    if is_risk(p):
        return "RISK_REPLAY"
    return "LOGIT_WEIGHT" if with_logits(p) else "ER_WEIGHT"


def step_columns(p: dict) -> tuple[str, ...]:
    if is_risk(p):
        return STEP_COLUMNS + ("retention_rank", "retention_positive", "retention_negative", "retention", "retention_weight")
    return STEP_COLUMNS + (("logit_mse", "logit_weight") if with_logits(p) else ())


def all_weights(p: dict) -> dict[str, float]:
    return p.get("comparison_methods", {"seq": 0., "er": 1., **p.get("reference_arms", {}), **p["arms"]})


def config(p: dict) -> dict:
    result = parent.config(parent.contract())
    result["runtime"] = copy.deepcopy(p["runtime"])
    return result


def sources(p: dict | None = None) -> list[dict]:
    if p is not None and is_risk(p):
        from step28_risk_study import sources as risk_sources
        return risk_sources()
    additional = ["docs/SELLER_ALIAS_ER_WEIGHT.zh.md", "schema/step28_er_weight_policy.json",
                  "scripts/step28_er_weight.py", "scripts/step28_er_weight_run.py",
                  "scripts/step28_er_weight_evaluate.py", "scripts/step28_er_weight_check.py",
                  "tests/test_step28_er_weight_contracts.py",
                  "scripts/run_step28_er_weight_linux_20261001.sh",
                  "scripts/run_step28_er_check_linux_20261001.sh",
                  "reports/documentation/20261001/er_weight/decision.json"]
    if p is not None and p["study"] in ("seller_alias_er_low_weight", "seller_alias_logit_weight", "seller_alias_logit_low"):
        additional += ["docs/SELLER_ALIAS_ER_LOW.zh.md", "schema/step28_er_low_policy.json",
                       "reports/documentation/20261002/er_low/decision.json"]
    if p is not None and with_logits(p):
        additional += ["docs/SELLER_ALIAS_LOGIT_WEIGHT.zh.md", "schema/step28_logit_weight_policy.json",
                       "reports/documentation/20261002/logit_weight/decision.json"]
    if p is not None and p["study"] == "seller_alias_logit_low":
        additional += ["docs/SELLER_ALIAS_LOGIT_LOW.zh.md", "schema/step28_logit_low_policy.json",
                       "reports/documentation/20261003/logit_low/decision.json"]
    rows = prior.sources() + [data.record(data.ROOT / name, data.ROOT) for name in additional]
    return sorted(rows, key=lambda row: row["path"])


def point_name(order: str, arm: str, stage: int, p: dict | None = None) -> str:
    if order not in ORDERS or arm not in (ARMS if p is None else p["arms"]) or stage not in (2, 3):
        raise ValueError("Unknown new ER endpoint")
    return f"{order}_{arm}_stage{stage}"


def expected_points(p: dict | None = None) -> list[str]:
    return [point_name(order, arm, stage, p) for order in ORDERS
            for arm in (ARMS if p is None else p["arms"]) for stage in (2, 3)]


def baseline(p: dict, job: Path | None = None, weight_job: Path | None = None,
             continuation_jobs: dict[str, Path] | None = None) -> dict:
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
    result = {"job": job, "manifest": manifest, "partition": partition,
              "collected": collected, "execution": records["execution.json"]}
    if "weight_reference" in p:
        spec = p["weight_reference"]
        weight_job = weight_job if weight_job is not None else data.ROOT / spec["linux_job"]
        saved = {name: data.read_json(data.verify(weight_job / name, rec))
                 for name, rec in spec["records"].items()}
        wm, wc = saved["run/manifest.json"], saved["evaluation/collected.json"]
        if (wm["status"] != "COMPLETE_3456_ER_WEIGHT_UPDATES_VALID_BLIND"
                or wm["policy_sha256"] != POLICY_SHA256 or wc["policy_sha256"] != POLICY_SHA256
                or wm["physical_updates"] != 3456 or wm["gradient_group_presentations"] != 6912
                or wm["source_files"] != wc["source_files"]
                or wm["baseline_records"] != p["baseline"]["records"]
                or wc["status"] != "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS"
                or set(wc["points"]) != set(expected_points())
                or any(wc[key] != collected[key] for key in ("group_ids", "domains", "metric_columns"))
                or any(saved["execution.json"][key] != value for key, value in p["baseline"]["environment"].items())):
            raise ValueError("Frozen half/quarter reference is not aligned with the shared pilot")
        for order in ORDERS:
            rec = manifest["points"][order + "_shared"]
            shared = data.read_json(data.verify(job / "run" / rec["path"], rec))
            for arm in p["reference_arms"]:
                start = wm["restored_starts"][order + "_" + arm]
                if (start["full_checkpoint"] != shared["full_checkpoint"]
                        or start["model_state_sha256"] != shared["model_state_sha256"]
                        or start["first_map"] != shared["first_map_parameters"]
                        or start["memory_source"] != manifest["memories"][order + "_er_after1"]["file"]
                        or start["adam_step"] != 288 or not start["first_scores_replayed_exactly"]):
                    raise ValueError("A historical reference used a different first-stage state")
        result["weight_reference"] = {"job": weight_job, "manifest": wm, "collected": wc}
    for arm, spec in p.get("continuation_references", {}).items():
        ref_job = (continuation_jobs or {}).get(arm, data.ROOT / spec["linux_job"])
        saved = {name: data.read_json(data.verify(ref_job / name, rec))
                 for name, rec in spec["records"].items()}
        rm, rc = saved["run/manifest.json"], saved["evaluation/collected.json"]
        names = {f"{order}_{arm}_stage{stage}" for order in ORDERS for stage in (2, 3)}
        tag = spec["evidence_tag"]
        if (rm["status"] != f"COMPLETE_1728_{tag}_UPDATES_VALID_BLIND"
                or rc["status"] != f"ALL_18_{tag}_MATRICES_SAVED_BEFORE_COMPARISONS"
                or saved["evaluation/evaluation.json"]["status"] != f"COMPLETE_{tag}_DEVELOPMENT_COMPARISON"
                or any(saved[name]["policy_sha256"] != spec["policy_sha256"] for name in
                       ("run/manifest.json", "evaluation/collected.json", "evaluation/evaluation.json"))
                or rm["physical_updates"] != 1728 or rm["gradient_group_presentations"] != 3456
                or set(rm["points"]) != names or set(rc["points"]) != names
                or rm["baseline_records"] != p["baseline"]["records"]
                or saved["run/partition.json"] != partition
                or any(saved[name]["source_files"] != rm["source_files"] for name in
                       ("execution.json", "evaluation/collected.json", "evaluation/evaluation.json"))
                or any(rc[key] != collected[key] for key in ("group_ids", "domains", "metric_columns"))
                or any(saved["execution.json"][key] != value for key, value in p["baseline"]["environment"].items())):
            raise ValueError("Frozen continuation reference is incomplete or unpaired: " + arm)
        for order in ORDERS:
            rec = manifest["points"][order + "_shared"]
            shared = data.read_json(data.verify(job / "run" / rec["path"], rec))
            start = rm["restored_starts"][order + "_" + arm]
            if (start["full_checkpoint"] != shared["full_checkpoint"]
                    or start["model_state_sha256"] != shared["model_state_sha256"]
                    or start["first_map"] != shared["first_map_parameters"]
                    or start["memory_source"] != manifest["memories"][order + "_" + spec["memory_arm"] + "_after1"]["file"]
                    or start["adam_step"] != 288 or not start["first_scores_replayed_exactly"]):
                raise ValueError("Continuation reference used a different shared start: " + arm)
        result.setdefault("continuation_references", {})[arm] = {
            "job": ref_job, "manifest": rm, "collected": rc}
    return result


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
           *, reference: np.ndarray | None = None, logit_weight: float = 0.,
           observe: bool = False, check: Callable[[], None] = lambda: None) -> dict:
    """Two forwards; independently weighted history supervision/MSE, one clip/update."""
    import torch

    if (type(weight) not in (int, float) or weight not in (1., .5, .25, .1)
            or stage not in (2, 3) or current is None or history is None
            or current.labels is None or history.labels is None
            or history.uid == current.uid or len(optimizer.param_groups) != 2
            or parent.adam_step(optimizer) != (stage - 1) * 288 + step - 1):
        raise ValueError("Unexpected ER continuation or supervision")
    if (logit_weight not in (0., .5) or (reference is not None) != (logit_weight == .5)
            or (reference is not None and (not isinstance(reference, np.ndarray)
                or reference.shape != (378,) or reference.dtype != np.float32 or not np.isfinite(reference).all()))):
        raise ValueError("Historical raw-logit reference or independent MSE weight differs")
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
        if role == "history" and reference is not None:
            target = torch.tensor(reference, dtype=predicted.dtype, device=predicted.device)
            penalty = (predicted - target).square().mean()
            objective = objective + logit_weight * penalty
            result.update(logit_mse=float(penalty.detach()), logit_weight=logit_weight)
        objective.backward()
        result.update({role + "_" + key: float(value.detach()) for key, value in terms.items()})
        del predicted, truth, terms, objective
    result["weighted_history_total"] = weight * result["history_total"]
    result["total"] = result["current_total"] + result["weighted_history_total"]
    if reference is not None:
        result["total"] += logit_weight * result["logit_mse"]
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
