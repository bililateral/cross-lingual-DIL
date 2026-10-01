"""Independent saved-logit evaluation; development truth is read only after replay qualification."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_data as data

ARMS = ("frozen", "sequential", "er", "cumulative")
KS = (1, 3, 5, 10)
CURVE_KEYS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct")
CLASS_KEYS = CURVE_KEYS + ("brier", "log_loss", "precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc")
RETRIEVAL_KEYS = ("map", "mrr") + tuple(f"recall_at_{k}" for k in KS) + tuple(f"ndcg_at_{k}" for k in KS)


def curve_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or y.shape != scores.shape or not len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(scores).all() or not 0 < y.sum() < len(y)):
        raise ValueError("Curve inputs need aligned finite scores and both classes")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    end = np.r_[ordered[1:] != ordered[:-1], True]
    tp = np.cumsum(y[order], dtype=np.float64)[end]
    fp = np.cumsum(1 - y[order], dtype=np.float64)[end]
    recall, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    precision = tp / (tp + fp)
    dr = np.diff(np.r_[0., recall])
    return {"average_precision": float(dr @ precision),
            "trapezoidal_pr_auc": float(dr @ ((precision + np.r_[1., precision[:-1]]) / 2)),
            "roc_auc": float(np.diff(np.r_[0., fpr]) @ ((recall + np.r_[0., recall[:-1]]) / 2)),
            "recall_at_fpr_1pct": float(np.max(np.r_[0., recall[fpr <= .01]]))}


def confusion_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": precision, "recall": recall, "specificity": specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "balanced_accuracy": (recall + specificity) / 2,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.}


def classification(y: np.ndarray, scores: np.ndarray, threshold: float = 0.) -> dict[str, Any]:
    result: dict[str, Any] = curve_metrics(y, scores)
    scores = np.asarray(scores, dtype=np.float64)
    probability = np.exp(-np.logaddexp(0., -scores))
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    result["brier"] = float(np.mean((probability - y) ** 2))
    result["log_loss"] = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    predicted, positive = scores >= threshold, y == 1
    tp, fp = int(np.sum(predicted & positive)), int(np.sum(predicted & ~positive))
    fn, tn = int(np.sum(~predicted & positive)), int(np.sum(~predicted & ~positive))
    result.update(confusion_metrics(tp, fp, fn, tn))
    result["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


def retrieval(y: np.ndarray, scores: np.ndarray, sellers: int) -> np.ndarray:
    """Complete sorted K_n edges; score ties use ascending public seller order."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    pairs = sellers * (sellers - 1) // 2
    if y.shape != scores.shape or y.ndim != 2 or y.shape[1] != pairs or not np.isfinite(scores).all():
        raise ValueError("Retrieval rows differ from complete worlds")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Nonbinary relevance")
    left, right = np.triu_indices(sellers, 1)
    adjacency = np.full((len(y), sellers, sellers), -np.inf)
    relevant = np.zeros((len(y), sellers, sellers), dtype=np.uint8)
    adjacency[:, left, right] = adjacency[:, right, left] = scores
    relevant[:, left, right] = relevant[:, right, left] = y
    order = np.argsort(-adjacency, axis=2, kind="stable")[:, :, :sellers - 1]
    ranked = np.take_along_axis(relevant, order, axis=2)
    total = relevant.sum(axis=2)
    if not np.all((total >= 1) & (total <= 2)):
        raise ValueError("Each qualified query must have one or two relevant candidates")
    ranks = np.arange(1, sellers)
    ap = (np.cumsum(ranked, axis=2) * ranked / ranks).sum(axis=2) / total
    rr = 1. / (np.argmax(ranked, axis=2) + 1)
    values = [ap, rr]
    values.extend(ranked[:, :, :k].sum(axis=2) / total for k in KS)
    discounts = 1. / np.log2(ranks + 1)
    for k in KS:
        ideal = np.asarray([discounts[:min(k, int(n))].sum() for n in total.ravel()]).reshape(total.shape)
        values.append((ranked[:, :, :k] * discounts[:k]).sum(axis=2) / ideal)
    return np.stack([v.mean(axis=1) for v in values], axis=1)


def point_metrics(y: np.ndarray, scores: np.ndarray, no_strong: np.ndarray, sellers: int) -> tuple[dict, np.ndarray]:
    if y.shape != scores.shape or y.shape != no_strong.shape or y.ndim != 2 or no_strong.dtype != np.dtype(bool):
        raise ValueError("World-aligned metric inputs differ")
    classification_worlds, subset_worlds = [], []
    for labels, values, selected in zip(y, scores, no_strong, strict=True):
        classification_worlds.append(classification(labels, values))
        subset_worlds.append(classification(labels[selected], values[selected]))
    numeric = np.asarray([[r[k] for k in CLASS_KEYS] for r in classification_worlds])
    subset = np.asarray([[r[k] for k in CLASS_KEYS] for r in subset_worlds])
    ranking = retrieval(y, scores, sellers)
    columns = list(CLASS_KEYS + RETRIEVAL_KEYS) + ["no_strong_" + k for k in CLASS_KEYS]
    per_world = np.column_stack((numeric, ranking, subset))
    result = {"world_equal": dict(zip(columns, per_world.mean(axis=0).tolist(), strict=True)),
              "pooled": classification(y.ravel(), scores.ravel()),
              "no_strong_pooled": classification(y[no_strong], scores[no_strong]),
              "world_metric_columns": columns,
              "threshold": {"logit": 0., "probability": .5, "selection": "fixed before labels; no fitting"},
              "recall_at_fpr_note": "Empirical labeled-sample diagnostic, not a frozen deployable threshold",
              "per_world_confusion": [r["confusion"] for r in classification_worlds],
              "no_strong_per_world_confusion": [r["confusion"] for r in subset_worlds]}
    for key, records, sizes in (("all", classification_worlds, np.full(len(y), y.shape[1])),
                                ("no_strong", subset_worlds, no_strong.sum(axis=1))):
        counts = {k: float(np.mean([r["confusion"][k] / n for r, n in zip(records, sizes)]))
                  for k in ("tp", "fp", "fn", "tn")}
        result[key + "_world_equal_confusion"] = {
            "normalized_counts": counts, "metrics": confusion_metrics(**counts)}
    return result, per_world


def paired_interval(delta: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """Average fixed arrival orders FIRST; resample the same development worlds."""
    if delta.ndim != 2 or indices.ndim != 2 or indices.shape[1] != delta.shape[1]:
        raise ValueError("Paired interval dimensions differ")
    if (not np.issubdtype(indices.dtype, np.integer) or not np.isfinite(delta).all()
            or np.min(indices) < 0 or np.max(indices) >= delta.shape[1]):
        raise ValueError("Invalid paired interval values")
    values = delta.mean(axis=0)
    boot = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "per_order": delta.mean(axis=1).tolist(),
            "conditional_95pct_interval": np.quantile(boot, [.025, .975]).tolist(),
            "conditional_on": "Saved models and three fixed arrival orders; worlds are paired resampling units"}


def validate_run(run: Path, policy: dict) -> dict:
    manifest = data.read_json(run / "manifest.json")
    if (manifest["status"] != "ALL_FOUR_ARM_SCORES_SAVED_AND_REPLAYED_NO_DEVELOPMENT_LABELS"
            or manifest["code"] != data.code_records() or manifest["policy"] != policy
            or [r["seed"] for r in manifest["orders"]] != policy["order_seeds"]
            or manifest["label_reads"] != {"train_csv_offline_packaging": 1, "development": 0, "audit_a": 0, "audit_b": 0}):
        raise ValueError("Full prediction/replay qualification is missing or stale")
    for order in manifest["orders"]:
        base = run / f"order{order['seed']}"
        required = {"initial", "shared"} | {f"{arm}_stage{stage}" for arm in ARMS[1:] for stage in range(2, 6)}
        if set(order["points"]) != required:
            raise ValueError("Missing or extra checkpoints")
        arrival = data.read_json(data.ROOT / policy["arrival"]["path"])
        stages = next(o["train_world_ordinals_by_stage"] for o in arrival["orders"] if o["seed"] == order["seed"])
        accesses = []
        for stage in range(1, 6):
            worlds = stages[stage - 1]
            accesses.append({"stage": stage, "scope": "current", "world_ordinals": worlds, "rows": len(worlds) * 378})
            if stage > 1:
                worlds = [w for group in stages[:stage] for w in group]
                accesses.append({"stage": stage, "scope": "arrived_prefix", "world_ordinals": worlds, "rows": len(worlds) * 378})
        if order["stage_accesses"] != accesses or not 0 < order["first_memory_bytes"] <= policy["memory_bytes"]:
            raise ValueError("Stage access or first memory budget differs")
        for arm in ARMS:
            expected = ["initial", "shared"] + (["shared"] * 4 if arm == "frozen" else [f"{arm}_stage{s}" for s in range(2, 6)])
            if order["trajectory"][arm] != expected:
                raise ValueError("Model/stage mapping differs")
            if len(order["training"][arm]) != 5 or order["training"][arm][0] != order["training"]["frozen"][0]:
                raise ValueError("First-stage sharing or stage logs differ")
            for stage, log in enumerate(order["training"][arm], 1):
                rows = 37800 * stage if arm == "cumulative" else 37800
                updates = policy["epochs"] * math.ceil(rows / policy["batch_size"])
                expected_counts = (updates, rows * policy["epochs"], updates * policy["batch_size"] if arm == "er" and stage > 1 else 0)
                if arm == "frozen" and stage > 1:
                    expected_counts = (0, 0, 0)
                if tuple(log[k] for k in ("updates", "current_presentations", "replay_presentations")) != expected_counts:
                    raise ValueError("Recorded update or presentation budget differs")
                expected_new = 0 if arm == "frozen" and stage > 1 else 37800 * policy["epochs"]
                expected_prior = (stage - 1) * 37800 * policy["epochs"] if arm == "cumulative" else 0
                if (log["new_arrival_presentations"], log["prior_arrival_presentations"]) != (expected_new, expected_prior):
                    raise ValueError("New/prior-arrival cost accounting differs")
                if arm in ("sequential", "er") and log["current_order_sha256"] != order["training"]["sequential"][stage - 1]["current_order_sha256"]:
                    raise ValueError("Paired new-sample order differs")
        for stage in range(2, 6):
            memory = order["points"][f"er_stage{stage}"]["memory"]
            if (memory["seen_unique_rows"] != stage * 37800 or memory["records"] != 2284
                    or not 0 < memory["total_bytes"] == memory["array_bytes"] + memory["state_bytes"] <= policy["memory_bytes"]
                    or memory["disk_and_next_draw_replayed"] is not True):
                raise ValueError("Historical memory boundary differs")
        for point in order["points"].values():
            if point["disk_model_optimizer_preprocessing_scores_exact"] is not True:
                raise ValueError("Checkpoint was not replay-qualified")
            data.verify(point["checkpoint"], base)
            values = np.load(data.verify(point["scores"], base), allow_pickle=False)
            if values.shape != (189000,) or values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                raise ValueError("Saved complete score array differs")
    return manifest


def evaluate(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Do not overwrite evaluation evidence")
    policy = data.load_policy()
    data.verify_public(policy)
    manifest = validate_run(run, policy)  # All 42 arrays and checkpoints BEFORE truth.
    y = data.aligned_labels(data.verify(policy["development_labels"]),
                            data.ROOT / policy["public_files"]["development_rows"]["path"],
                            "development", 500, 28, 20).reshape(500, 378)
    identity = np.load(data.ROOT / policy["public_files"]["development_identity"]["path"], allow_pickle=False)
    mask = ~((identity[:, policy["no_strong_identity_columns"][0]] > 0)
             | (identity[:, policy["no_strong_identity_columns"][1]] > 0))
    mask = mask.reshape(500, 378)
    del identity
    output.mkdir()
    summaries, arrays, curves = {}, {}, {}
    for order in manifest["orders"]:
        seed = order["seed"]
        for name, point in order["points"].items():
            key = f"order{seed}_{name}"
            scores = np.load(run / f"order{seed}" / point["scores"]["path"], allow_pickle=False).reshape(500, 378)
            summary, per_world = point_metrics(y, scores, mask, 28)
            path = output / (key + ".npy")
            np.save(path, per_world, allow_pickle=False)
            summary["world_metrics"] = data.record(path, output)
            summaries[key], arrays[key] = summary, per_world
        curves[str(seed)] = {arm: [summaries[f"order{seed}_{name}"]["world_equal"] for name in order["trajectory"][arm]] for arm in ARMS}
    indices = np.random.default_rng(policy["bootstrap_seed"]).integers(0, 500, (policy["bootstrap_replicates"], 500), dtype=np.int32)
    columns = next(iter(summaries.values()))["world_metric_columns"]
    comparisons = {}
    for target, control in (("er", "sequential"), ("sequential", "frozen"), ("cumulative", "sequential"), ("er", "cumulative")):
        contrasts = []
        for order in manifest["orders"]:
            prefix = f"order{order['seed']}_"
            contrasts.append(arrays[prefix + order["trajectory"][target][-1]] - arrays[prefix + order["trajectory"][control][-1]])
        delta = np.stack(contrasts)
        comparisons[target + "_minus_" + control] = {
            metric: paired_interval(delta[:, :, columns.index(metric)], indices)
            for metric in ("average_precision", "no_strong_average_precision")}
    changes = {}
    for arm in ARMS:
        changes[arm] = {}
        for before, after in ((0, 1), (1, 5), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)):
            deltas = []
            for order in manifest["orders"]:
                prefix, trajectory = f"order{order['seed']}_", order["trajectory"][arm]
                deltas.append(arrays[prefix + trajectory[after]] - arrays[prefix + trajectory[before]])
            delta = np.stack(deltas)
            changes[arm][f"stage{after}_minus_stage{before}"] = {
                metric: paired_interval(delta[:, :, columns.index(metric)], indices)
                for metric in ("average_precision", "no_strong_average_precision")}
    result = {"status": "EXPLORATORY_CONTINUAL_DEVELOPMENT_EVALUATION_COMPLETE",
              "run_manifest": data.record(run / "manifest.json"), "code": data.code_records(),
              "unique_score_arrays": len(summaries), "points": summaries, "stage_trajectories": curves,
              "final_paired_comparisons": comparisons, "fixed_development_changes": changes,
              "bootstrap": {"seed": policy["bootstrap_seed"], "replicates": policy["bootstrap_replicates"],
                            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()},
              "label_reads": {"development": 1, "train": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
              "domain_BWT_or_forgetting": None,
              "limits": ["Fixed development population under random whole-world arrival, not distinct-domain forgetting",
                         "Already-consumed development split: exploratory, not independent confirmation",
                         "No synthetic result establishes real Chinese-market performance",
                         "All comparisons are predefined; no automatic favorable rerun or selector expansion"]}
    data.write_json(output / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.run, args.output)["status"])


if __name__ == "__main__":
    main()
