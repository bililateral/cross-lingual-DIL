#!/usr/bin/env python3
"""Freeze a metric-v2 Step7 source-fusion control without retraining or overwriting Step7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_SUMMARY = ROOT / "reports" / "step7_training_summary.json"
DEFAULT_LABELS = ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
DEFAULT_PREDICTIONS = (
    ROOT / "reports" / "step7_core_zero_shot_default_predictions.zh_target_strict_test.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "reports" / "step15_v6" / "baselines" / "step7_source_only_default_fusion_summary.json"
)
EXPERIMENT = "core_zero_shot_default"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", default=str(DEFAULT_SOURCE_SUMMARY))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source_summary_path = resolve(args.source_summary)
    labels_path = resolve(args.labels)
    predictions_path = resolve(args.predictions)
    output_path = resolve(args.output)
    for path in (source_summary_path, labels_path, predictions_path, Path(__file__).resolve()):
        if not path.exists():
            raise FileNotFoundError(path)

    source_summary = step7.load_json(source_summary_path)
    source_experiment = (source_summary.get("experiments") or {}).get(EXPERIMENT)
    if not source_experiment:
        raise ValueError(f"Step7 source summary is missing {EXPERIMENT}")
    threshold = source_experiment.get("selected_threshold")
    if threshold is None:
        raise ValueError(f"Step7 source experiment has no frozen source-valid threshold: {EXPERIMENT}")

    eligible_labels = {
        row["pair_uid"]: row
        for row in load_csv(labels_path)
        if row.get("split_name") == "test"
        and row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    }
    prediction_rows = load_csv(predictions_path)
    prediction_index = {row["pair_uid"]: row for row in prediction_rows}
    if len(prediction_index) != len(prediction_rows) or set(prediction_index) != set(eligible_labels):
        raise ValueError(
            "Step15-v6 Step7 control prediction coverage does not match the frozen Chinese test"
        )
    pair_order = sorted(eligible_labels)
    y_true = np.asarray(
        [1 if eligible_labels[pair_uid]["review_label"] == "positive" else 0 for pair_uid in pair_order],
        dtype=float,
    )
    scores = np.asarray(
        [float(prediction_index[pair_uid]["prob_positive"]) for pair_uid in pair_order],
        dtype=float,
    )
    for pair_uid, expected_y in zip(pair_order, y_true, strict=True):
        recorded = int(float(prediction_index[pair_uid].get("y_true", expected_y)))
        if recorded != int(expected_y):
            raise ValueError(f"Step7 prediction label disagrees with frozen test: {pair_uid}")
    metrics = step7.evaluate_probabilities(y_true, scores, float(threshold))

    frozen_experiment = dict(source_experiment)
    frozen_experiment["zh_zero_shot_test_metrics"] = metrics
    frozen_experiment["selected_threshold"] = float(threshold)
    frozen_experiment["metric_semantics_version"] = step7.METRIC_SEMANTICS_VERSION
    payload = {
        "step": "step15_v6_refresh_step7_control",
        "metric_semantics_version": step7.METRIC_SEMANTICS_VERSION,
        "pr_auc_definition": metrics.get("pr_auc_definition"),
        "role": "read_only_metric_refresh_no_retraining_no_canonical_overwrite",
        "source_summary": str(source_summary_path.relative_to(ROOT)),
        "source_summary_sha256": sha256(source_summary_path),
        "prediction_path": str(predictions_path.relative_to(ROOT)),
        "prediction_sha256": sha256(predictions_path),
        "labels_path": str(labels_path.relative_to(ROOT)),
        "labels_sha256": sha256(labels_path),
        "producer_script_sha256": sha256(Path(__file__).resolve()),
        "experiments": {EXPERIMENT: frozen_experiment},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(ROOT)),
                "experiment": EXPERIMENT,
                "zh_test_metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
