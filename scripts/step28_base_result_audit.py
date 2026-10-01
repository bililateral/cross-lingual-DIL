"""Independently audit saved base-model metrics and counts without label access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def array(root: Path, spec: dict) -> np.ndarray:
    path = root / spec["path"]
    assert path.stat().st_size == spec["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    return np.load(path, allow_pickle=False)


def audit(run: Path, evaluation: Path, destination: Path) -> dict:
    result = read(evaluation / "evaluation.json")
    partition = read(run / "partition.json")
    columns = result["columns"]
    domain_rows = [np.array([i for i, row in enumerate(partition["development"]) if row["domain"] == d]) for d in "ABC"]
    assert [len(rows) for rows in domain_rows] == [20, 20, 20]
    config = result["config"]["evaluation"]
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, 20, size=(config["bootstrap_replicates"], 3, 20))
    comparisons, maximum_error = 0, 0.

    def check(actual, expected) -> None:
        nonlocal comparisons, maximum_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a - b)))
        comparisons += a.size
        maximum_error = max(maximum_error, error)
        assert error <= 1e-12, (error, a, b)

    matrices, counts, boot_counts, summary = {}, {}, {}, {}
    blind_checks = 0
    for arm, records in result["arms"].items():
        manifest = read(run / arm / "manifest.json")
        for epoch, point in records["points"].items():
            matrix = array(evaluation, point["file"])
            score = array(run / arm, manifest["points"][epoch]["scores"]["development"])
            check(matrix.mean(0), [point["mean"][c] for c in columns])
            for d, rows in zip("ABC", domain_rows):
                check(matrix[rows].mean(0), [point["by_domain"][d][c] for c in columns])
            for i, row in enumerate(point["counts_at_logit_zero"]):
                tp, fp, fn, tn = [row[k] for k in ("tp", "fp", "fn", "tn")]
                assert tp + fn == 20 and fp + tn == 358
                assert tp + fp == int((score[i].astype(float) >= 0).sum())
                blind_checks += 1
                precision = tp / (tp + fp) if tp + fp else 0.
                recall, specificity = tp / 20, tn / 358
                denom = math.sqrt((tp + fp) * 20 * 358 * (tn + fn))
                expected = [precision, recall, 2 * tp / (2 * tp + fp + fn), specificity,
                            (recall + specificity) / 2, (tp * tn - fp * fn) / denom if denom else 0.]
                check(matrix[i, 6:12], expected)
            if epoch == "6":
                matrices[arm] = matrix
        auto = records["automatic_classification"]
        counts[arm] = np.asarray(auto["counts_by_group"], dtype=np.int64)
        assert np.all(counts[arm][:, [0, 2]].sum(1) == 20)
        assert np.all(counts[arm][:, [1, 3]].sum(1) == 358)
        score = array(run / arm, manifest["points"]["6"]["scores"]["development"])
        assert np.array_equal((score.astype(float) >= auto["threshold"]).sum(1), counts[arm][:, :2].sum(1))
        blind_checks += 60
        boot_counts[arm] = np.stack([counts[arm][rows][draws[:, i]].sum(1) for i, rows in enumerate(domain_rows)], axis=1)
        for j, scope in enumerate((*"ABC", "pooled")):
            point = auto["by_domain"][scope] if scope != "pooled" else auto["pooled"]
            totals = counts[arm][domain_rows[j]].sum(0) if scope != "pooled" else counts[arm].sum(0)
            samples = boot_counts[arm][:, j] if scope != "pooled" else boot_counts[arm].sum(1)
            check(totals, [point[k] for k in ("tp", "fp", "fn", "tn")])
            for key, num, other in (("fpr", 1, 3), ("recall", 0, 2), ("precision", 0, 1)):
                denominator = samples[:, num] + samples[:, other]
                values = np.divide(samples[:, num], denominator, out=np.zeros(len(samples)), where=denominator != 0)
                check(point[key], totals[num] / (totals[num] + totals[other]) if totals[num] + totals[other] else 0.)
                check(point["conditional_95pct_intervals"][key], np.percentile(values, [2.5, 97.5]))
            if scope != "pooled":
                assert point["passes_point_gates"] == bool(totals[1] <= 7 and totals[0] >= 200)
        assert auto["all_domains_pass"] == all(p["passes_point_gates"] for p in auto["by_domain"].values())
        summary[arm] = {"epoch6": records["points"]["6"]["mean"], "epoch3": records["points"]["3"]["mean"],
                        "automatic": auto, "fit_calibration": {}}
        for split in ("fit", "calibration"):
            m = array(run / arm, manifest["points"]["6"]["train_metrics"][split]["file"])
            summary[arm]["fit_calibration"][split] = dict(zip(columns, m.mean(0).tolist()))
    for arm, values in result["comparisons_to_labse"].items():
        delta = matrices[arm] - matrices["labse"]
        for i, column in enumerate(columns):
            sampled = sum(delta[rows, i][draws[:, j]].mean(1) for j, rows in enumerate(domain_rows)) / 3
            check(values[column]["mean"], delta[:, i].mean())
            check(values[column]["conditional_95pct_interval"], np.percentile(sampled, [2.5, 97.5]))
        for metric, num, other in (("fpr", 1, 3), ("recall", 0, 2), ("precision", 0, 1)):
            for j, scope in enumerate((*"ABC", "pooled")):
                ratios, points = [], []
                for model in (arm, "labse"):
                    x = boot_counts[model][:, j] if scope != "pooled" else boot_counts[model].sum(1)
                    totals = counts[model][domain_rows[j]].sum(0) if scope != "pooled" else counts[model].sum(0)
                    denom = x[:, num] + x[:, other]
                    ratios.append(np.divide(x[:, num], denom, out=np.zeros(len(x)), where=denom != 0))
                    points.append(totals[num] / (totals[num] + totals[other]) if totals[num] + totals[other] else 0.)
                saved = result["automatic_comparisons_to_labse"][arm][metric][scope]
                check(saved["difference"], points[0] - points[1])
                check(saved["conditional_95pct_interval"], np.percentile(ratios[0] - ratios[1], [2.5, 97.5]))
    record = {"status": "PASS_SAVED_METRICS_COUNTS_AND_PAIRED_INTERVALS", "numeric_values_compared": comparisons,
              "maximum_absolute_difference": maximum_error, "blind_prediction_count_checks": blind_checks,
              "formal_label_parses": 0, "model_loading": 0,
              "limitations": "AP/PR-AUC/ROC/retrieval/probability metrics are accepted from saved per-group matrices, not recomputed from labels. Uncertainty is conditional on fixed model, calibration and threshold.",
              "models": summary, "comparisons_to_labse": result["comparisons_to_labse"]}
    destination.mkdir(parents=True)
    (destination / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# 三模型完整指标（第6轮，60群等权）", "", "分类指标按 logit≥0；自动识别的校准阈值指标见主报告。AP与梯形PR-AUC分列。", "", "|指标|LaBSE|E5-large|BGE-M3|", "|---|---:|---:|---:|"]
    for column in columns:
        rows.append("|" + column + "|" + "|".join(f"{summary[arm]['epoch6'][column]:.9f}" for arm in ("labse", "multilingual_e5_large", "bge_m3")) + "|")
    (destination / "metrics.zh.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.evaluation, args.out)
    print(json.dumps({k: result[k] for k in ("status", "numeric_values_compared", "maximum_absolute_difference", "blind_prediction_count_checks")}))


if __name__ == "__main__":
    main()
