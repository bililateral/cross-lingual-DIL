"""D, encoder learning-rate schedule, and known hard-candidate ranking.

Model inputs and the D head are unchanged. No formal data loader or test entry.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_chinese_base as base

data, core, metrics = base.data, base.core, base.metrics
POLICY = data.ROOT / "schema/step28_alias_ranking_policy.json"
POLICY_SHA256 = "b5b7e4fc628c389d9cf91664dcff64c28b5fc8e8295bc3d8a82a6e5cb20f9b01"
SEEDS = ("s0", "s1", "s2")
ARMS = ("d", "schedule", "hard")
RUNS = tuple(f"{seed}_{arm}" for seed in SEEDS for arm in ARMS)
LOSS_NAMES = ("bce", "rank", "hard", "total")
GUARDS = {"average_precision": 1, "roc_auc": 1, "brier": -1, "log_loss": -1}


def contract() -> dict:
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("Ranking policy differs from the frozen contract")
    p = data.read_json(POLICY)
    if (p["arms"] != list(ARMS) or p["runs"] != list(RUNS)
            or p["physical_updates"] != 7776 or p["updates_per_run"] != 864
            or p["epochs"] != 6 or p["score_epochs"] != [3, 6] or p["primary_epoch"] != 6
            or p["primary_comparison"] != ["hard", "d"] or p["fixed_test_seed"] != "s0"
            or p["hard_ranking"]["count"] != 5 or p["hard_ranking"]["weight"] != .5
            or p["evaluation"]["guard_signs"] != GUARDS
            or p["evaluation"]["recall_at_5_strict_improvement_mean_and_s0"] is not True
            or p["supervision"]["test_access"] is not False or p["supervision"]["owners_access"] is not False):
        raise ValueError("Confirmed comparison or supervision scope differs")
    base.contract()
    if (p["base_policy_sha256"] != base.POLICY_SHA256
            or data.sha256(Path(base.__file__)) != p["base_runner_sha256"]):
        raise ValueError("Frozen D implementation differs")
    return p


def reference_config(policy: dict, seed: str) -> dict:
    if seed not in SEEDS:
        raise ValueError("Unknown paired seed")
    c = copy.deepcopy(base.contract())
    c.update(policy["seeds"][seed])
    c["runtime"] = copy.deepcopy(policy["runtime"])
    return c


def split_run(run_id: str) -> tuple[str, str]:
    if run_id not in RUNS:
        raise ValueError("Unknown run identity")
    seed, arm = run_id.split("_")
    return seed, arm


def load_model(policy: dict, seed: str, device: str = "cuda:0") -> Any:
    return base.load_model(reference_config(policy, seed), "split_rank", device)


def encoder_lr(policy: dict, arm: str, step: int) -> float:
    """One-based actual optimizer step; the final scheduled encoder rate is zero."""
    if arm not in ARMS or type(step) is not int or not 1 <= step <= policy["updates_per_run"]:
        raise ValueError("Unknown arm or update outside the fixed budget")
    if arm == "d":
        return 2e-5
    spec = policy["encoder_schedule"]
    warmup, total = spec["warmup_updates"], policy["updates_per_run"]
    if warmup != math.ceil(spec["warmup_fraction"] * total) or not 0 < warmup < total:
        raise ValueError("Invalid warmup schedule")
    factor = step / warmup if step <= warmup else (total - step) / (total - warmup)
    return float(spec["peak_lr"] * factor)


def set_learning_rate(optimizer: Any, policy: dict, arm: str, step: int) -> tuple[float, float]:
    if len(optimizer.param_groups) != 2:
        raise ValueError("Expected encoder and head optimizer groups")
    rates = (encoder_lr(policy, arm, step), policy["encoder_schedule"]["head_lr"])
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate
    return rates


def hard_candidates(predicted: Any, truth: Any, count: int = 5) -> tuple[Any, Any, Any]:
    """Known-negative indices only, deterministic exact-tie order, live score values."""
    import torch

    if predicted.shape != (378,) or truth.shape != (378,) or count != 5:
        raise ValueError("Expected 28-account scores and five hard negatives")
    # Independent shape/label/connectivity validation remains in base.objectives.
    if not torch.isfinite(predicted).all() or not torch.isin(truth, truth.new_tensor([0, 1])).all():
        raise ValueError("Nonfinite scores or nonbinary labels")
    left, right = torch.triu_indices(28, 28, 1, device=predicted.device)
    scores = predicted.new_zeros((28, 28))
    scores[left, right], scores[right, left] = predicted, predicted
    positive = torch.zeros((28, 28), dtype=torch.bool, device=predicted.device)
    positive[left, right], positive[right, left] = truth.bool(), truth.bool()
    negative = ~positive & ~torch.eye(28, dtype=torch.bool, device=predicted.device)
    if (positive.sum(1) < 1).any() or (negative.sum(1) < count).any():
        raise ValueError("Each query needs positive and known negative candidates")
    selected = torch.argsort(scores.detach().masked_fill(~negative, -torch.inf),
                             dim=1, descending=True, stable=True)[:, :count]
    if not negative.gather(1, selected).all():
        raise ValueError("Self or positive entered hard negatives")
    return scores, positive, selected


def objectives(predicted: Any, truth: Any, hard_weight: float) -> dict:
    import torch

    if hard_weight not in (0., .5):
        raise ValueError("Only disabled or approved fixed hard weight is allowed")
    result = base.objectives(predicted, truth, 1)
    if hard_weight == 0:
        return {**result, "hard": predicted.new_zeros(())}
    scores, positive, selected = hard_candidates(predicted, truth)
    terms = []
    for query in range(28):
        pos = scores[query, positive[query]]
        neg = scores[query, selected[query]]
        terms.append(torch.nn.functional.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean())
    hard = torch.stack(terms).mean()
    return {"bce": result["bce"], "rank": result["rank"], "hard": hard,
            "total": result["total"] + hard_weight * hard}


def score(model: Any, groups: list, config: dict, check: Callable[[], None] = lambda: None) -> np.ndarray:
    return base.score(model, groups, config, "split_rank", check)


def update(model: Any, optimizer: Any, group: Any, config: dict, policy: dict,
           arm: str, step: int, seed: int, *, observe: bool = False,
           check: Callable[[], None] = lambda: None) -> dict:
    import torch

    if group.labels is None:
        raise ValueError("Only supervised fitting groups enter updates")
    if step == 1 and optimizer.state:
        raise ValueError("New run optimizer already has history")
    if step > 1 and (not optimizer.state or any(float(v["step"]) != step - 1
                                               for v in optimizer.state.values())):
        raise ValueError("Optimizer history does not match the declared update")
    rates = set_learning_rate(optimizer, policy, arm, step)
    if arm != "hard":
        # Exact original D computation. No extra forward/random draws/zero-loss graph.
        result = base.update(model, optimizer, group, config, "split_rank", seed,
                             observe=observe and rates[0] > 0, check=check)
        return {**result, "hard": 0., "step": step, "encoder_lr": rates[0], "head_lr": rates[1]}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {name: core.state_digest(module.state_dict()) for name, module in
              (("encoder", model.encoder), ("head", model.head))} if observe else {}
    torch.manual_seed(seed)
    predicted = base.logits(model, group, config, "split_rank", check)
    truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
    terms = objectives(predicted, truth, policy["hard_ranking"]["weight"])
    terms["total"].backward()
    evidence = {}
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
                raise ValueError("Missing finite nonzero task gradient: " + name)
            evidence[name] = {"finite_nonzero_gradient": True}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"],
                                         error_if_nonfinite=True)
    check()
    optimizer.step()
    if observe:
        for name, module in (("encoder", model.encoder), ("head", model.head)):
            changed = before[name] != core.state_digest(module.state_dict())
            expected = name == "head" or rates[0] > 0
            if changed != expected:
                raise ValueError("Actual update differs from learning-rate scope: " + name)
            evidence[name]["parameters_changed"] = changed
    return {**{name: float(terms[name].detach()) for name in LOSS_NAMES}, "modules": evidence,
            "gradient_norm": float(norm), "step": step, "encoder_lr": rates[0], "head_lr": rates[1]}


def paired_summary(deltas: np.ndarray, domains: list[str], evaluation: dict) -> dict:
    """Three fixed paired seeds, then whole-group resampling separately within domains."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (3, 60, len(metrics.COLUMNS)) or len(domains) != 60 or not np.isfinite(values).all():
        raise ValueError("Expected complete paired seed-by-group matrices")
    rows = [np.flatnonzero(np.asarray(domains) == d) for d in "ABC"]
    if any(len(row) != 20 for row in rows):
        raise ValueError("Unbalanced domain partition")
    seed = evaluation["valid_bootstrap_seed"]
    draws = np.random.default_rng(seed).integers(0, 20, size=(evaluation["bootstrap_replicates"], 3, 20))
    average = values.mean(0)
    point = np.stack([average[row].mean(0) for row in rows]).mean(0)
    per_seed = np.stack([values[:, row].mean(1) for row in rows]).mean(0)
    result = {}
    for index, name in enumerate(metrics.COLUMNS):
        boot = np.stack([average[row, index][draws[:, d]].mean(1) for d, row in enumerate(rows)]).mean(0)
        result[name] = {"mean": float(point[index]), "per_seed": per_seed[:, index].tolist(),
                        "conditional_95pct_interval": np.quantile(boot, [.025, .975], method="linear").tolist(),
                        "by_domain": {d: float(average[row, index].mean()) for d, row in zip("ABC", rows, strict=True)}}
    return {"metrics": result, "replicates": evaluation["bootstrap_replicates"], "seed": seed,
            "scope": evaluation["scope"]}


def acceptance(summary: dict, evaluation: dict) -> dict:
    m = summary["metrics"]
    checks = {"map_minimum_observed_gain": m["map"]["mean"] >= evaluation["minimum_map_gain"],
              "map_interval_above_zero": m["map"]["conditional_95pct_interval"][0] > 0,
              "map_improves_each_seed": all(v > 0 for v in m["map"]["per_seed"]),
              "recall_at_5_mean_strictly_improves": m["recall_at_5"]["mean"] > 0,
              "recall_at_5_s0_strictly_improves": m["recall_at_5"]["per_seed"][0] > 0}
    for name, sign in GUARDS.items():
        checks[name + "_mean_non_degradation"] = sign * m[name]["mean"] >= 0
        checks[name + "_s0_non_degradation"] = sign * m[name]["per_seed"][0] >= 0
    return {"passed": all(checks.values()), "checks": checks,
            "failed": [name for name, ok in checks.items() if not ok],
            "next": "REVIEW_THEN_SEPARATELY_AUTHORIZED_TEST" if all(checks.values())
                    else "RETAIN_NEGATIVE_RESULT_AND_DEFER_CONTINUAL_DESIGN",
            "scope": "Observed-value guards and conditional uncertainty, not population noninferiority."}
