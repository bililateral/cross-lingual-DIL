"""Audit and summarize saved Chinese-base results; no labels, text or weights.

Runs on Linux py310 with one CPU thread. Recalculates aggregates and paired
intervals from saved matrices/counts, without importing the training evaluator.
It cannot independently reconstruct label-dependent curves or retrieval metrics.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("mean_bce", "mean_rank", "split_bce", "split_rank")
COLUMNS = (
    "average_precision", "trapezoidal_pr_auc", "roc_auc", "recall_at_fpr_1pct",
    "brier", "log_loss", "precision", "recall", "f1", "specificity",
    "balanced_accuracy", "mcc", "map", "mrr", "recall_at_1", "recall_at_3",
    "recall_at_5", "recall_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class Checks:
    def __init__(self) -> None:
        self.numbers = 0
        self.maximum_error = 0.0
        self.blind_group_counts = 0

    def close(self, actual, expected) -> None:
        left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Nonfinite or differently shaped comparison")
        error = float(np.max(np.abs(left - right), initial=0.0))
        self.maximum_error = max(error, self.maximum_error)
        self.numbers += left.size
        if error > 1e-12:
            raise ValueError(f"Saved-result arithmetic differs: {error}")

    def blind(self, scores: np.ndarray, threshold: float, counts: np.ndarray) -> None:
        # Compare predicted-positive totals only; do not infer pair labels.
        predicted = np.count_nonzero(scores.astype(np.float64) >= threshold, axis=1)
        if not np.array_equal(predicted, counts[:, :2].sum(axis=1)):
            raise ValueError("Blind score positive totals differ from saved counts")
        self.blind_group_counts += len(predicted)


def rates(counts: np.ndarray) -> dict:
    tp, fp, fn, tn = [int(x) for x in counts.sum(axis=0)]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "fpr": fp / (fp + tn),
            "f1": 2 * tp / (2 * tp + fp + fn)}


def confusion_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([[r[k] for k in ("tp", "fp", "fn", "tn")] for r in rows], dtype=np.int64)


def validate_counts(counts: np.ndarray) -> None:
    if counts.ndim != 2 or counts.shape[1] != 4 or (counts < 0).any():
        raise ValueError("Confusion matrix shape or sign")
    if not (np.all(counts[:, 0] + counts[:, 2] == 20)
            and np.all(counts[:, 1] + counts[:, 3] == 358)):
        raise ValueError("Positive/negative group totals")


def check_fixed(matrix: np.ndarray, counts: np.ndarray, audit: Checks) -> None:
    validate_counts(counts)
    expected = []
    for tp, fp, fn, tn in counts:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        divisor = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        expected.append([precision, recall, 2 * tp / (2 * tp + fp + fn), specificity,
                         (recall + specificity) / 2, (tp * tn - fp * fn) / divisor if divisor else 0.0])
    audit.close(matrix[:, 6:12], expected)


def csv_file(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(run_dir: Path, evaluation: Path, output: Path, historical: Path) -> dict:
    if platform.system() != "Linux" or (output / "analysis.json").exists():
        raise ValueError("Linux and fresh analysis output required")
    audit = Checks()
    e = read(evaluation / "evaluation.json")
    c = read(evaluation / "collected.json")
    manifest = read(run_dir / "manifest.json")
    if tuple(e["columns"]) != COLUMNS or e["label_parses"] != {"train": 0, "development": 1, "heldout": 0, "owners": 0}:
        raise ValueError("Metric schema or evaluation access differs")
    if e["manifest_sha256"] != record(run_dir / "manifest.json")["sha256"]:
        raise ValueError("Result and training provenance differ")
    if c["status"] != "CHINESE_BASE_METRICS_COLLECTED_BEFORE_UNCERTAINTY":
        raise ValueError("Missing pre-uncertainty collection")
    domains = np.asarray(e["domains"])
    selections = {d: np.flatnonzero(domains == d) for d in "ABC"}
    if len(set(e["group_ids"])) != 60 or any(len(v) != 20 for v in selections.values()):
        raise ValueError("Group alignment differs")
    scopes = {**selections, "pooled": np.arange(60)}
    matrices, counts_by_arm, thresholds = {}, {}, {}
    metric_rows, auto_rows, fixed_rows, training_rows, comparison_rows = [], [], [], [], []
    sources = [record(evaluation / name) for name in ("evaluation.json", "collected.json")]
    sources.append(record(run_dir / "manifest.json"))

    def array(base: Path, row: dict) -> np.ndarray:
        path = base / row["path"]
        actual = record(path)
        if actual["bytes"] != row["bytes"] or actual["sha256"] != row["sha256"]:
            raise ValueError("Saved array hash differs: " + str(path))
        sources.append(actual)
        return np.load(path, allow_pickle=False)

    for arm in (*ARMS, "historical_labse"):
        if arm in ARMS:
            a = e["arms"][arm]
            points = a["points"]
            auto = a["automatic_classification"]
            am = read(run_dir / arm / "manifest.json")
            thresholds[arm] = auto["threshold"]
        else:
            a = e[arm]
            points = {"6": a}
            auto = a["automatic_classification"]
            thresholds[arm] = a["threshold"]
        counts_by_arm[arm] = np.asarray(a.get("counts_by_group", auto.get("counts_by_group")), dtype=np.int64)
        validate_counts(counts_by_arm[arm])
        saved_raw = c["arms"][arm]["automatic_classification"] if arm in ARMS else c[arm]
        if saved_raw["counts_by_group"] != counts_by_arm[arm].tolist():
            raise ValueError("Counts changed after collection")
        for epoch, p in points.items():
            matrix = array(evaluation, p["file"])
            if matrix.shape != (60, 22) or matrix.dtype != np.float64:
                raise ValueError("Metric matrix shape/dtype")
            zero = confusion_rows(p["counts_at_logit_zero"])
            check_fixed(matrix, zero, audit)
            if arm in ARMS:
                score = array(run_dir / arm, am["points"][epoch]["scores"]["development"])
            else:
                score_row = e["historical_reference"]["labse/scores/epoch6_development.npy"]
                score = array(ROOT, score_row)
            audit.blind(score, 0.0, zero)
            if epoch == "6":
                matrices[arm] = matrix
                audit.blind(score, thresholds[arm], counts_by_arm[arm])
            for scope, indices in scopes.items():
                mean = matrix[indices].mean(axis=0)
                if arm in ARMS:
                    saved = p["mean"] if scope == "pooled" else p["by_domain"][scope]
                    audit.close(mean, [saved[k] for k in COLUMNS])
                metric_rows.append({"arm": arm, "epoch": epoch, "scope": scope,
                                    **dict(zip(COLUMNS, mean.tolist()))})
                fixed = rates(zero[indices])
                saved = p["fixed_classification"]["pooled"] if scope == "pooled" else p["fixed_classification"]["by_domain"][scope]
                audit.close(list(fixed.values()), [saved[k] for k in fixed])
                fixed_rows.append({"arm": arm, "epoch": epoch, "scope": scope, "threshold": 0.0, **fixed})
            if arm in ARMS:
                tr = {"arm": arm, "epoch": int(epoch)}
                for role in ("fit", "calibration"):
                    saved = am["points"][epoch]["train_metrics"][role]
                    values = array(run_dir / arm, saved["file"])
                    count = confusion_rows(saved["counts"])
                    check_fixed(values, count, audit)
                    blind = array(run_dir / arm, am["points"][epoch]["scores"][role])
                    audit.blind(blind, 0.0, count)
                    for key in ("average_precision", "map", "mrr", "log_loss"):
                        tr[f"{role}_{key}"] = float(values[:, COLUMNS.index(key)].mean())
                for key in ("average_precision", "map", "mrr", "log_loss"):
                    tr[f"valid_{key}"] = float(matrix[:, COLUMNS.index(key)].mean())
                training_rows.append(tr)
        if arm in ARMS:
            cal = read(run_dir / arm / "calibration.json")
            cal_counts = np.asarray(cal["counts_by_group"], dtype=np.int64)
            validate_counts(cal_counts)
            cal_scores = array(run_dir / arm, am["points"]["6"]["scores"]["calibration"])
            audit.blind(cal_scores, cal["threshold"], cal_counts)
            bounds = cal["bounds"]
            audit.close(cal["threshold"], max(v["threshold"] for v in bounds.values()))
            for d in "ABC":
                audit.close(bounds[d]["threshold"], np.nextafter(np.float64(bounds[d]["next_negative_logit"]), np.inf))
                if cal["counts_by_domain"][d][1] > 4:
                    raise ValueError("Calibration FPR budget exceeded")

    # Use frequency-weighted dot products, independent of the evaluator's
    # advanced-index group aggregation. Preserve its approved random draws.
    draws = np.random.default_rng(20260918).integers(0, 20, size=(5000, 3, 20))
    frequencies = np.eye(20, dtype=np.int16)[draws].sum(axis=2)
    boot_counts = {arm: np.stack([frequencies[:, j] @ counts_by_arm[arm][selections[d]]
                                 for j, d in enumerate("ABC")], axis=1)
                   for arm in counts_by_arm}

    def boot_rate(values: np.ndarray, name: str) -> np.ndarray:
        numerator, other = {"fpr": (1, 3), "recall": (0, 2), "precision": (0, 1)}[name]
        denominator = values[:, numerator] + values[:, other]
        return np.divide(values[:, numerator], denominator,
                         out=np.zeros(len(values)), where=denominator != 0)

    for arm, counts in counts_by_arm.items():
        auto = e["arms"][arm]["automatic_classification"] if arm in ARMS else e[arm]["automatic_classification"]
        passed = []
        for scope, indices in scopes.items():
            point = rates(counts[indices])
            saved = auto["pooled"] if scope == "pooled" else auto["by_domain"][scope]
            audit.close(list(point.values()), [saved[k] for k in point])
            totals = boot_counts[arm].sum(axis=1) if scope == "pooled" else boot_counts[arm][:, "ABC".index(scope)]
            for name in ("fpr", "recall", "precision"):
                audit.close(np.quantile(boot_rate(totals, name), [.025, .975]), saved["conditional_95pct_intervals"][name])
            gate = point["fp"] <= 7 and point["tp"] >= 200
            if scope != "pooled":
                if gate != saved["passes_point_gates"]:
                    raise ValueError("Per-domain qualification differs")
                passed.append(gate)
            auto_rows.append({"arm": arm, "scope": scope, "threshold": thresholds[arm],
                              **point, "domain_gate": gate if scope != "pooled" else "not_applicable"})
        if all(passed) != auto["all_domains_pass"]:
            raise ValueError("Joint qualification differs")

    for candidate, baseline in e["config"]["comparisons"]:
        name = f"{candidate}_minus_{baseline}"
        delta = matrices[candidate] - matrices[baseline]
        boot = sum(frequencies[:, j] @ delta[selections[d]] for j, d in enumerate("ABC")) / 60
        for col, key in enumerate(COLUMNS):
            interval = np.quantile(boot[:, col], [.025, .975])
            mean = float(delta[:, col].mean())
            saved = e["comparisons"][name]["metrics"][key]
            audit.close([mean, *interval], [saved["mean"], *saved["conditional_95pct_interval"]])
            comparison_rows.append({"comparison": name, "metric": key, "difference": mean,
                                    "lower_95": interval[0], "upper_95": interval[1]})
        for scope, indices in scopes.items():
            candidate_totals = boot_counts[candidate].sum(1) if scope == "pooled" else boot_counts[candidate][:, "ABC".index(scope)]
            baseline_totals = boot_counts[baseline].sum(1) if scope == "pooled" else boot_counts[baseline][:, "ABC".index(scope)]
            for key in ("fpr", "recall", "precision"):
                point = rates(counts_by_arm[candidate][indices])[key] - rates(counts_by_arm[baseline][indices])[key]
                interval = np.quantile(boot_rate(candidate_totals, key) - boot_rate(baseline_totals, key), [.025, .975])
                saved = e["comparisons"][name]["automatic"][key][scope]
                audit.close([point, *interval], [saved["difference"], *saved["conditional_95pct_interval"]])

    old = read(historical / "evaluation.json")
    old_point = old["arms"]["labse"]["points"]["6"]
    old_matrix = array(historical, old_point["file"])
    delta = np.abs(old_matrix - matrices["historical_labse"])
    historical_comparison = {"values": int(delta.size), "different_values": int(np.count_nonzero(delta)),
                             "max_absolute_difference": float(delta.max())}
    audit.close(old_matrix, matrices["historical_labse"])
    if old["arms"]["labse"]["automatic_classification"]["counts_by_group"] != counts_by_arm["historical_labse"].tolist():
        raise ValueError("Historical calibrated counts changed")

    summary = {"status": "SAVED_RESULTS_AUDITED_NO_NEW_LABEL_ACCESS", "observed_at": datetime.now().astimezone().isoformat(),
               "training_status": manifest["status"], "physical_updates": manifest["physical_updates"],
               "formal_training_seconds": manifest["formal_training_seconds"],
               "primary_candidate": "split_rank", "primary_comparison": e["primary_comparison"],
               "all_models_fail_absolute_gate": all(not (e["arms"][a]["automatic_classification"]["all_domains_pass"]) for a in ARMS),
               "metrics": metric_rows, "automatic": auto_rows, "fixed_zero": fixed_rows,
               "training_progression": training_rows, "comparisons": e["comparisons"],
               "resources": {a: read(run_dir / a / "manifest.json")["resources"] for a in ARMS}}
    verification = {"status": "PASS_SAVED_RESULT_ARITHMETIC_AND_ALIGNMENT", "numeric_values_checked": int(audit.numbers),
                    "maximum_absolute_difference": audit.maximum_error, "blind_group_positive_counts_checked": audit.blind_group_counts,
                    "historical_labse_matrix": historical_comparison, "source": record(Path(__file__).resolve()),
                    "inputs": sources, "label_file_bytes_read": 0, "label_parses": 0, "model_loads": 0, "updates": 0,
                    "bootstrap": {"replicates": 5000, "seed": 20260918, "unit": "whole group within domain, paired across arms", "recomputation": "frequency dot products"},
                    "limits": "Rechecks saved matrices/counts and blind-positive totals. Does not newly reconstruct label-dependent curve/retrieval metrics, inspect raw text, establish multi-seed generalization, or read heldout/owners. Conditional intervals do not correct repeated valid development or multiple comparisons."}
    output.mkdir(parents=True, exist_ok=True)
    write(output / "analysis.json", summary)
    write(output / "verification.json", verification)
    for name, rows in (("metrics.csv", metric_rows), ("automatic.csv", auto_rows),
                       ("fixed_zero.csv", fixed_rows), ("training.csv", training_rows),
                       ("comparisons.csv", comparison_rows)):
        csv_file(output / name, rows)
    return {k: verification[k] for k in ("status", "numeric_values_checked", "maximum_absolute_difference", "blind_group_positive_counts_checked", "historical_labse_matrix")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run.resolve(), args.evaluation.resolve(), args.out.resolve(),
                         args.historical.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
