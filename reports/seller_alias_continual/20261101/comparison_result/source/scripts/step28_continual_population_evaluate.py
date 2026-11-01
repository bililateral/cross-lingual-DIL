"""Windows valid-only evaluation after complete blind output qualification.

Pure curve/classification/retrieval functions preserve the audited text evaluator
definitions. No old archive, consumed label parser or identity stratification is
imported. The Linux-only model receipt operation hashes outputs without labels.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import step28_continual_population_data as data
import step28_continual_population_run as runner

KS = (1, 3, 5, 10)
CURVE_KEYS = ("average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct")
CLASS_KEYS = CURVE_KEYS + ("brier", "log_loss", "precision", "recall", "f1",
                         "specificity", "balanced_accuracy", "mcc")
RETRIEVAL_KEYS = ("map", "mrr") + tuple(f"recall_at_{k}" for k in KS) + tuple(f"ndcg_at_{k}" for k in KS)
COLUMNS = CLASS_KEYS + RETRIEVAL_KEYS

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


def verify_models(run: Path, output: Path) -> dict:
    """Read-only Linux payload verification receipt for the Windows copy stage."""
    manifest = data.read_json(run / "manifest.json")
    if manifest["status"] != runner.COMPLETE or output.exists():
        raise ValueError("Requires a complete run and a new receipt file")
    found = []
    for order in manifest["orders"]:
        for record in order["models"].values():
            path = data.verify(run / record["path"], record)
            found.append(data.record(path, run))
    if len(found) != 12:
        raise ValueError("Missing final inference payload")
    result = {"status": "TWELVE_LINUX_MODEL_PAYLOADS_HASH_VERIFIED_NO_LABELS",
              "manifest_sha256": data.sha256(run / "manifest.json"),
              "files": sorted(found, key=lambda r: r["path"])}
    data.write_json(output, result)
    return result


def require_memory_points(points: dict) -> None:
    actual = {name for name, record in points.items() if "memory" in record}
    if actual != {"shared", "er_stage2", "er_stage3"}:
        raise ValueError("Missing/extra required historical memory evidence")


def validate_run(run: Path, config: dict, archive: data.Archive) -> tuple[dict, dict]:
    if (run / "failure.json").exists():
        raise ValueError("Failed run cannot be evaluated")
    manifest = data.read_json(run / "manifest.json")
    if (manifest["status"] != runner.COMPLETE or manifest["config"] != config
            or manifest["source_files"] != runner.sources()
            or manifest["label_reads"] != {"new_train_csv_offline_packaging": 1,
                                           "development": 0, "heldout": 0, "owners": 0, "old_assets": 0}
            or manifest["physical_updates"] != 5400
            or [o["order"] for o in manifest["orders"]] != config["orders"]
            or manifest["runtime_check"]["status"] != "PASS_REAL_LABSE_NEW_INPUT_HEAD_MEMORY_ADAM_RELOAD"
            or manifest["runtime_check"]["torch_contracts"] != {"passed": 3, "skipped": 0, "failed": 0}
            or manifest["runtime_check"]["formal_initialization_restored"] is not True):
        raise ValueError("Incomplete, stale, or unauthorized execution")
    if (manifest["budget"]["elapsed_seconds"] > config["runtime"]["maximum_gpu_stage_seconds"]
            or manifest["budget"]["peak_observed_bytes"] > config["runtime"]["maximum_output_bytes"]):
        raise ValueError("Unapproved resource expansion")
    for split in ("development", "heldout"):
        if manifest["evaluation_group_ids"][split] != [g.uid for g in archive.groups(split)]:
            raise ValueError("Blind score row identities differ")
    expected_inputs = {**archive.checked,
                       "train/supervision/pairs.csv": archive.manifest["files"]["train/supervision/pairs.csv"]}
    if manifest["verified_input_files"] != expected_inputs:
        raise ValueError("Training inputs/provenance differ")
    for key in ("file_count", "total_size_bytes", "content_sha256"):
        if manifest["pretrained_model"][key] != config["model"][key]:
            raise ValueError("Pretrained model source differs")
    receipt = data.read_json(run / "model_verification.json")
    expected_models = []
    scores: dict = {}
    for order in manifest["orders"]:
        order_id = order["order"]
        expected_points = {"initial", "shared"} | {f"{arm}_stage{s}" for arm in runner.ARMS for s in (2, 3)}
        if set(order["points"]) != expected_points or set(order["models"]) != {"frozen", *runner.ARMS}:
            raise ValueError("Missing/extra point or model")
        require_memory_points(order["points"])
        for arm, record in order["models"].items():
            if record["actual_loaded_model_equals_replayed_state"] is not True:
                raise ValueError("Inference payload not actually loaded and compared")
            if record["path"] != f"models/{order_id}_{arm}.pt":
                raise ValueError("Model point mapping differs")
            expected_models.append({k: record[k] for k in ("path", "bytes", "sha256")})
        if order["trajectory"]["frozen"] != ["initial", "shared", "shared", "shared"]:
            raise ValueError("Frozen trajectory differs")
        for arm in runner.ARMS:
            if order["trajectory"][arm] != ["initial", "shared", f"{arm}_stage2", f"{arm}_stage3"]:
                raise ValueError("Model trajectory differs")
        for arm, stage, log in [("shared", 1, order["training"]["shared"])] + [
                (arm, s, order["training"][arm][s - 2]) for arm in runner.ARMS for s in (2, 3)]:
            domains = order_id[:stage] if arm == "cumulative" else order_id[stage - 1]
            seed = data.seed_for(config["initialization_seed"], order_id, stage,
                                 "cumulative" if arm == "cumulative" else "current")
            rows = data.schedule(archive.groups("train", domains), 3, seed)
            if (log["current_group_ids"] != [g.uid for g in rows]
                    or log["current_dropout_stream"] != seed
                    or log["updates"] != len(rows) or log["current_pair_presentations"] != len(rows) * 378
                    or log["memory_frozen_during_stage"] is not True):
                raise ValueError("Training supply, schedule or update counts differ")
            history = log["replay_group_ids"]
            if arm == "er":
                previous = "shared" if stage == 2 else "er_stage2"
                available = order["points"][previous]["memory"]["retained_groups"]
                if len(history) != len(rows) or not set(history).issubset(available):
                    raise ValueError("Replay accessed an unretained/future group")
            elif history:
                raise ValueError("Unexpected historical replay")
            if log["replay_pair_presentations"] != len(history) * 378:
                raise ValueError("Replay computation accounting differs")
        scores[order_id] = {}
        for name, record in order["points"].items():
            if record["full_model_and_adam_reloaded"] is not True or set(record["scores"]) != {"development", "heldout"}:
                raise ValueError("Incomplete actual checkpoint replay")
            if "memory" in record:
                mem = record["memory"]
                expected_seen = 60 if name == "shared" else 60 * int(name[-1])
                if (mem["disk_roundtrip_exact"] is not True or mem["seen"] != expected_seen
                        or len(set(mem["retained_groups"])) != 6 or not 0 < mem["bytes"] <= 1048576):
                    raise ValueError("Historical memory evidence differs")
                # Hash only; do not parse archived train labels during valid evaluation.
                data.verify(run / mem["path"], mem)
            for split, expected in record["scores"].items():
                if expected["path"] != f"scores/{order_id}_{name}_{split}.npy":
                    raise ValueError("Prediction point mapping differs")
                path = data.verify(run / expected["path"], expected)
                values = np.load(path, allow_pickle=False)
                if (values.dtype != np.dtype("float32") or values.shape != (len(archive.groups(split)), 378)
                        or not np.isfinite(values).all()):
                    raise ValueError("Blind score dimensions or values differ")
                if split == "development":
                    scores[order_id][name] = values
    if (receipt["status"] != "TWELVE_LINUX_MODEL_PAYLOADS_HASH_VERIFIED_NO_LABELS"
            or receipt["manifest_sha256"] != data.sha256(run / "manifest.json")
            or receipt["files"] != sorted(expected_models, key=lambda r: r["path"])):
        raise ValueError("Linux model verification is missing or belongs to another run")
    return manifest, scores


def group_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    results = [classification(y, s) for y, s in zip(labels, scores, strict=True)]
    numeric = np.array([[row[k] for k in CLASS_KEYS] for row in results])
    return np.column_stack((numeric, retrieval(labels, scores, 28))), [r["confusion"] for r in results]


def paired_interval(delta: np.ndarray, domain_rows: list[np.ndarray], config: dict) -> dict:
    """Average fixed orders first; bootstrap independent groups within each domain."""
    if delta.ndim != 2 or not np.isfinite(delta).all():
        raise ValueError("Paired differences must be finite order-by-group rows")
    flat = np.concatenate(domain_rows)
    if len(flat) != delta.shape[1] or set(flat.tolist()) != set(range(delta.shape[1])):
        raise ValueError("Domain bootstrap partition differs")
    rng = np.random.default_rng(config["bootstrap_seed"])
    averaged = delta.mean(axis=0)
    replicates = []
    for rows in domain_rows:
        indices = rng.choice(rows, size=(config["bootstrap_replicates"], len(rows)), replace=True)
        replicates.append(averaged[indices].mean(axis=1))
    boot = np.stack(replicates).mean(axis=0)
    tail = (1 - config["confidence_level"]) / 2
    return {"mean": float(np.mean([averaged[rows].mean() for rows in domain_rows])),
            "per_order": [float(np.mean([values[rows].mean() for rows in domain_rows])) for values in delta],
            "conditional_95pct_interval": np.quantile(boot, [tail, 1 - tail]).tolist(),
            "conditional_on": "This generated dataset, initialization and three fixed orders; independent groups only"}


def old_domain_change(current: np.ndarray, when_learned: np.ndarray,
                      history: np.ndarray, rows: np.ndarray) -> dict:
    # R[t,j] averages independent groups FIRST, then takes a maximum over stages.
    # Averaging per-group maxima would overstate domain-level forgetting.
    current_mean = float(current[rows].mean())
    return {"current_minus_when_learned": current_mean - float(when_learned[rows].mean()),
            "best_previous_minus_current": float(history[:, rows].mean(axis=1).max()) - current_mean}


def evaluate(run: Path, output: Path) -> dict:
    config = data.policy()
    if config["evaluation"]["split"] != "development" or config["evaluation"]["heldout_labels_allowed"]:
        raise ValueError("Only the confirmed valid-first stage is authorized")
    if output.exists():
        raise FileExistsError("Do not repeat label evaluation into an existing result directory")
    archive = data.Archive(config, train_labels=False)
    manifest, scores = validate_run(run, config, archive)
    groups = archive.groups("development")
    # All output/reload/supply gates above precede the single valid binary parse.
    output.mkdir(parents=True)
    data.write_json(output / "access.json", {"phase": "COMPLETE_OUTPUT_CHECKS_PASSED_BEFORE_VALID_PARSE",
                    "manifest_sha256": data.sha256(run / "manifest.json"), "heldout_labels": 0,
                    "train_labels": 0, "owners": 0, "old_assets": 0})
    path = data.verify(archive.root / "development/supervision/pairs.csv",
                       archive.manifest["files"]["development/supervision/pairs.csv"])
    grouped = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["group_uid"]].append(row)
    if set(grouped) != {g.uid for g in groups}:
        raise ValueError("Valid binary groups differ")
    labels = np.array([data.align_labels(g, grouped[g.uid]) for g in groups], dtype=np.uint8)
    if not np.all(labels.sum(axis=1) == 20):
        raise ValueError("Valid positive counts differ")
    domain_by_id = {r["group_uid"]: r["domain"] for r in archive.metadata}
    domain_rows = [np.array([i for i, g in enumerate(groups) if domain_by_id[g.uid] == d]) for d in "ABC"]
    result: dict = {"status": "VALID_EVALUATION_COMPLETE_TEST_TRUTH_UNREAD", "columns": COLUMNS,
                    "run_manifest_sha256": data.sha256(run / "manifest.json"),
                    "label_reads": {"development_binary_csv": 1, "heldout": 0, "train": 0, "owners": 0, "old_assets": 0},
                    "config": config, "groups": [g.uid for g in groups], "orders": {},
                    "threshold": {"logit": 0, "probability": 0.5},
                    "ranking": "raw logit; query score ties use ascending opaque account ID",
                    "recall_at_fpr_note": "Labeled-sample diagnostic, not a deployable threshold"}
    metrics = {}
    for order in manifest["orders"]:
        order_id = order["order"]
        metrics[order_id] = {}
        order_result: dict = {"points": {}, "transitions": {}}
        for name, values in scores[order_id].items():
            matrix, confusion = group_metrics(labels, values)
            metrics[order_id][name] = matrix
            target = output / f"{order_id}_{name}_metrics.npy"
            np.save(target, matrix, allow_pickle=False)
            domain = {d: dict(zip(COLUMNS, matrix[rows].mean(axis=0).tolist())) for d, rows in zip("ABC", domain_rows)}
            order_result["points"][name] = {
                "domain_equal": dict(zip(COLUMNS, np.stack([matrix[r].mean(axis=0) for r in domain_rows]).mean(axis=0).tolist())),
                "by_domain": domain, "per_group_confusion": confusion, "per_group_metrics": data.record(target, output),
                "pooled_pair_metrics": classification(labels.ravel(), values.ravel())}
        for arm, trajectory in order["trajectory"].items():
            transitions = []
            for stage in (1, 2, 3):
                now, before = (metrics[order_id][trajectory[t]][:, 0] for t in (stage, stage - 1))
                new_rows = domain_rows["ABC".index(order_id[stage - 1])]
                old = {}
                for learned in range(1, stage):
                    domain = order_id[learned - 1]
                    rows = domain_rows["ABC".index(domain)]
                    at_learning = metrics[order_id][trajectory[learned]][:, 0]
                    previous = np.stack([metrics[order_id][trajectory[t]][:, 0] for t in range(learned, stage)])
                    old[domain] = old_domain_change(now, at_learning, previous, rows)
                transitions.append({"stage": stage, "new_domain": order_id[stage - 1],
                                    "new_domain_after_minus_before": float((now[new_rows] - before[new_rows]).mean()),
                                    "old_domains": old})
            order_result["transitions"][arm] = transitions
        result["orders"][order_id] = order_result
    delta = np.stack([metrics[o]["er_stage3"][:, 0] - metrics[o]["sequential_stage3"][:, 0] for o in config["orders"]])
    result["primary_er_minus_sequential_ap"] = paired_interval(delta, domain_rows, config["evaluation"])
    result["final_fixed_order_equal"] = {
        arm: dict(zip(COLUMNS, np.stack([
            metrics[o]["shared" if arm == "frozen" else f"{arm}_stage3"].mean(axis=0)
            for o in config["orders"]]).mean(axis=0).tolist()))
        for arm in ("frozen", *runner.ARMS)}
    data.write_json(output / "evaluation.json", result)
    return {"status": result["status"], "primary": result["primary_er_minus_sequential_ap"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate-valid", "verify-models"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_models(args.run, args.out) if args.mode == "verify-models" else evaluate(args.run, args.out)
    print(data.json_bytes(result).decode())


if __name__ == "__main__":
    main()
