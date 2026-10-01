"""D versus learned item moments: model operations and prospective acceptance.

Formal execution is Linux-only. This module has no label loader or test entry.
"""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_chinese_base as base

data = base.data
core = base.core
metrics = base.metrics
POLICY = data.ROOT / "schema/step28_alias_pooling_policy.json"
POLICY_SHA256 = "e308ec646f47554525e0254ea2bb6c454d29db7cd1fb35788c28be29c2be1986"
SEED_IDS = ("s0", "s1", "s2")
NEW_RUNS = ("s0_weighted", "s1_d", "s1_weighted", "s2_d", "s2_weighted")
ALL_RUNS = ("s0_d", "s0_weighted", "s1_d", "s1_weighted", "s2_d", "s2_weighted")
GUARD_SIGNS = {"recall_at_5": 1, "average_precision": 1, "roc_auc": 1,
               "brier": -1, "log_loss": -1}


def contract() -> dict:
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("Pooling policy differs from the frozen contract")
    result = data.read_json(POLICY)
    if (result["new_runs"] != list(NEW_RUNS) or result["physical_updates"] != 4320
            or result["epochs"] != 6 or result["score_epochs"] != [3, 6]
            or result["primary_epoch"] != 6 or result["test_access_now"] is not False
            or result["owners_access"] is not False
            or result["evaluation"]["guard_signs"] != GUARD_SIGNS):
        raise ValueError("Approved experiment scope differs")
    base.contract()
    if (result["base_policy_sha256"] != base.POLICY_SHA256
            or data.sha256(Path(base.__file__)) != result["base_runner_sha256"]):
        raise ValueError("Incorrect frozen D policy")
    return result


def reference_config(policy: dict, seed_id: str) -> dict:
    """D operations take their original configuration; only approved seeds differ."""
    if seed_id not in SEED_IDS:
        raise ValueError("Unknown paired seed")
    result = copy.deepcopy(base.contract())
    seeds = policy["seeds"][seed_id]
    result["initialization_seed"] = seeds["initialization_seed"]
    result["schedule_seed"] = seeds["schedule_seed"]
    result["runtime"] = copy.deepcopy(policy["runtime"])
    return result


def make_aggregation(dimension: int, policy: dict, seed: int) -> Any:
    import torch

    p = policy["aggregation"]

    class ItemMoments(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(2 * dimension, p["hidden"])
            self.output = torch.nn.Linear(p["hidden"], 1, bias=False)
            torch.nn.init.xavier_uniform_(self.hidden.weight)
            torch.nn.init.zeros_(self.hidden.bias)
            torch.nn.init.zeros_(self.output.weight)

        def forward(self, title: Any, description: Any) -> Any:
            if title.shape != description.shape or title.ndim != 2 or not len(title):
                raise ValueError("Paired item channels differ")
            scores = self.output(torch.tanh(self.hidden(torch.cat((title, description), 1))))
            attention = torch.softmax(scores.flatten(), dim=0)
            return p["uniform_fraction"] / len(title) + (1 - p["uniform_fraction"]) * attention

    # Module construction and explicit initialization must not consume the
    # shared CPU or CUDA random streams. All new parameters are created on CPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(data.seed_for(seed, "item_aggregation"))
        result = ItemMoments()
    return result


def attach_aggregation(model: Any, policy: dict, seed: int, dimension: int = 1024) -> Any:
    if hasattr(model, "aggregation"):
        raise ValueError("Aggregation is already attached")
    model.add_module("aggregation", make_aggregation(dimension, policy, seed).to(
        next(model.parameters()).device))
    return model


def load_model(policy: dict, seed_id: str, weighted: bool, device: str = "cuda:0") -> Any:
    c = reference_config(policy, seed_id)
    model = base.load_model(c, "split_rank", device)
    if weighted:
        attach_aggregation(model, policy, c["initialization_seed"])
    return model


def common_digest(model: Any) -> str:
    return core.state_digest({name: value for name, value in model.state_dict().items()
                              if not name.startswith("aggregation.")})


def make_optimizer(model: Any, c: dict, policy: dict) -> Any:
    optimizer = core.make_optimizer(model, c)
    if hasattr(model, "aggregation"):
        optimizer.add_param_group({"params": model.aggregation.parameters(),
                                   "lr": policy["aggregation"]["learning_rate"],
                                   "weight_decay": policy["aggregation"]["weight_decay"]})
    supplied = [p for group in optimizer.param_groups for p in group["params"]]
    if (len({id(p) for p in supplied}) != len(supplied)
            or {id(p) for p in supplied} != {id(p) for p in model.parameters()}
            or not all(p.requires_grad for p in supplied)):
        raise ValueError("Optimizer must contain every parameter exactly once")
    return optimizer


def pool_channels(vectors: Any, counts: list[int], aggregation: Any | None) -> Any:
    import torch

    if aggregation is None:
        return base.pool_channels(vectors, counts, "separate_moments")
    if (not counts or any(n <= 0 for n in counts) or vectors.ndim != 2
            or len(vectors) != 2 * sum(counts) or not torch.isfinite(vectors).all()):
        raise ValueError("Incomplete or nonfinite paired item vectors")
    normalized = torch.nn.functional.normalize(vectors.float(), dim=1)
    title, description = normalized.split(sum(counts))
    accounts = []
    for left, right in zip(title.split(counts), description.split(counts), strict=True):
        weights = aggregation(left, right)
        parts = []
        for channel in (left, right):
            mean = (weights[:, None] * channel).sum(0)
            variance = (weights[:, None] * (channel - mean).square()).sum(0)
            parts.extend((mean, (variance + 1e-8).sqrt()))
        accounts.append(torch.cat(parts))
    return torch.nn.functional.normalize(torch.stack(accounts), dim=1)


def account_vectors(model: Any, group: Any, c: dict,
                    check: Callable[[], None] = lambda: None) -> Any:
    import torch

    if not hasattr(model, "aggregation"):
        return base.account_vectors(model, group, c, "split_rank", check)
    device = next(model.parameters()).device
    texts = base.record_texts(group, "separate_moments")
    size = c["input"]["microbatch"]
    blocks = []
    for start in range(0, len(texts), size):
        check()
        features = model.encoder.tokenizer(texts[start:start + size], padding=True,
                                            truncation=False, return_tensors="pt")
        if features["input_ids"].shape[1] > c["input"]["token_budget"]:
            raise ValueError("Whole item exceeds token limit; truncation is forbidden")
        features = {key: value.to(device) for key, value in features.items()}
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" and c["input"]["encoder_bf16"]
                   else contextlib.nullcontext())
        with context:
            blocks.append(model.encoder(features)["sentence_embedding"].float())
    return pool_channels(torch.cat(blocks), [len(items) for items in group.items], model.aggregation)


def logits(model: Any, group: Any, c: dict, check: Callable[[], None] = lambda: None) -> Any:
    return model.pair_logits(account_vectors(model, group, c, check))


def score(model: Any, groups: list, c: dict, check: Callable[[], None] = lambda: None) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        result = np.stack([logits(model, group, c, check).float().cpu().numpy() for group in groups])
    if result.shape != (len(groups), 378) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("Blind scores differ in dimensions, dtype or finiteness")
    return result


def update(model: Any, optimizer: Any, group: Any, c: dict, seed: int, *,
           observe: bool = False, check: Callable[[], None] = lambda: None) -> dict:
    import torch

    # The paired D training path calls the original operation directly.
    if not hasattr(model, "aggregation"):
        return base.update(model, optimizer, group, c, "split_rank", seed, observe=observe, check=check)
    if group.labels is None:
        raise ValueError("Training requires authorized fitting supervision")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    modules = {"encoder": model.encoder, "head": model.head, "aggregation": model.aggregation}
    before = {name: core.state_digest(module.state_dict()) for name, module in modules.items()} if observe else {}
    torch.manual_seed(seed)
    predicted = logits(model, group, c, check)
    truth = torch.tensor(group.labels, dtype=torch.float32, device=predicted.device)
    terms = base.objectives(predicted, truth, 1)
    terms["total"].backward()
    evidence = {}
    if observe:
        for name, module in modules.items():
            norms = [float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None]
            if not norms or not np.isfinite(norms).all() or max(norms) <= 0:
                raise ValueError("No finite nonzero task gradient in " + name)
            evidence[name] = {"finite_nonzero_gradient": True}
        evidence["aggregation"]["parameter_gradient_norms_before_clip"] = {
            name: float(p.grad.float().norm()) if p.grad is not None else None
            for name, p in model.aggregation.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["optimizer"]["clip_norm"], error_if_nonfinite=True)
    check()
    optimizer.step()
    if observe:
        for name, module in modules.items():
            changed = core.state_digest(module.state_dict()) != before[name]
            if not changed:
                raise ValueError("Module parameters did not change: " + name)
            evidence[name]["parameters_changed"] = True
    return {**{name: float(value.detach()) for name, value in terms.items()},
            "gradient_norm_before_clip": float(norm), "modules": evidence}


def paired_summary(deltas: np.ndarray, domains: list[str], evaluation: dict,
                   *, test: bool = False) -> dict:
    """Average paired seeds first, then resample the same whole groups by domain."""
    values = np.asarray(deltas, dtype=np.float64)
    expected_groups = 120 if test else 60
    expected_seeds = 1 if test else 3
    if (values.shape != (expected_seeds, expected_groups, len(metrics.COLUMNS))
            or len(domains) != expected_groups or not np.isfinite(values).all()):
        raise ValueError("Expected complete paired seed-by-group metric differences")
    rows = [np.flatnonzero(np.asarray(domains) == d) for d in "ABC"]
    n = expected_groups // 3
    if any(len(index) != n for index in rows):
        raise ValueError("Paired bootstrap domain partition differs")
    seed = evaluation["test_bootstrap_seed"] if test else evaluation["valid_bootstrap_seed"]
    draws = np.random.default_rng(seed).integers(0, n, size=(evaluation["bootstrap_replicates"], 3, n))
    mean_seeds = values.mean(0)
    point = np.stack([mean_seeds[index].mean(0) for index in rows]).mean(0)
    per_seed = np.stack([values[:, index].mean(1) for index in rows]).mean(0)
    result = {}
    for column_index, name in enumerate(metrics.COLUMNS):
        boot = np.stack([mean_seeds[index, column_index][draws[:, j]].mean(1)
                         for j, index in enumerate(rows)]).mean(0)
        interval = np.quantile(boot, [.025, .975], method="linear")
        result[name] = {"mean": float(point[column_index]),
                        "per_seed": per_seed[:, column_index].tolist(),
                        "conditional_95pct_interval": interval.tolist(),
                        "by_domain": {d: float(mean_seeds[index, column_index].mean())
                                      for d, index in zip("ABC", rows, strict=True)}}
    return {"metrics": result, "replicates": evaluation["bootstrap_replicates"], "seed": seed,
            "unit": "paired whole group within each domain; average fixed seeds before resampling",
            "scope": "Conditional on fixed models, seeds, generated data and thresholds; not training-seed population uncertainty."}


def acceptance(summary: dict, evaluation: dict, *, test: bool = False) -> dict:
    """Every prespecified condition must pass; no favorable metric substitution."""
    m = summary["metrics"]
    checks = {
        "map_minimum_observed_gain": m["map"]["mean"] >= evaluation["minimum_map_gain"],
        "map_interval_above_zero": m["map"]["conditional_95pct_interval"][0] > 0,
    }
    if not test:
        checks["map_improves_each_paired_seed"] = all(value > 0 for value in m["map"]["per_seed"])
    for metric, sign in GUARD_SIGNS.items():
        checks[metric + "_mean_non_degradation"] = sign * m[metric]["mean"] >= 0
        if not test:
            checks[metric + "_fixed_s0_non_degradation"] = sign * m[metric]["per_seed"][0] >= 0
    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "failed": [name for name, value in checks.items() if not value],
            "next": ("ELIGIBLE_FOR_RESULT_REVIEW_AND_SEPARATELY_AUTHORIZED_TEST" if passed and not test
                     else "ELIGIBLE_FOR_RESULT_REVIEW_BEFORE_CONTINUAL_DESIGN" if passed
                     else "RETAIN_NEGATIVE_RESULT_AND_DEFER_CONTINUAL_DESIGN"),
            "guard_scope": "Observed macro values, with paired uncertainty reported; no claim of strict population noninferiority or all operating points."}
