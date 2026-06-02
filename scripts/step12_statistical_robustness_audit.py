#!/usr/bin/env python3
"""Step 12 fixed-test statistical robustness audit.

This script keeps the current zh_target_strict test split fixed and estimates
metric uncertainty by resampling Step 5 split components, not individual edges.
It intentionally does not implement random K-fold CV because that would blur the
active frozen benchmark boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABELS = Path("reports/step5_zh_target_strict_frozen_silver_labels.csv")
DEFAULT_FEATURES = Path("reports/step7_pair_features.zh_target_strict.csv")
DEFAULT_OUT_JSON = Path("reports/step12_statistical_robustness_zh_test_20260513.json")
DEFAULT_OUT_METRICS = Path("reports/step12_statistical_robustness_model_metrics_20260513.csv")
DEFAULT_OUT_COMPARISONS = Path("reports/step12_statistical_robustness_paired_comparisons_20260513.csv")

DEFAULT_RESAMPLES = 5000
DEFAULT_SEED = 20260513
CONFIDENCE_LEVEL = 0.95


MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model_id": "raw_e5_cosine",
        "role": "raw_semantic_control",
        "kind": "feature",
        "path": DEFAULT_FEATURES,
        "score_column": "embedding_cosine_multilingual_e5_large",
    },
    {
        "model_id": "raw_labse_cosine",
        "role": "raw_semantic_control",
        "kind": "feature",
        "path": DEFAULT_FEATURES,
        "score_column": "embedding_cosine_labse",
    },
    {
        "model_id": "raw_bge_m3_cosine",
        "role": "raw_semantic_control",
        "kind": "feature",
        "path": DEFAULT_FEATURES,
        "score_column": "embedding_cosine_bge_m3",
    },
    {
        "model_id": "step7_core_zero_shot_default",
        "role": "step7_clean_fusion_control",
        "kind": "prediction",
        "path": Path("reports/step7_core_zero_shot_default_predictions.zh_target_strict_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step7_core_zero_shot_bge_m3",
        "role": "step7_clean_fusion_control",
        "kind": "prediction",
        "path": Path("reports/step7_core_zero_shot_bge_m3_predictions.zh_target_strict_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step7_core_zero_shot_default_no_structural",
        "role": "step7_clean_ablation_control",
        "kind": "prediction",
        "path": Path("reports/step7_core_zero_shot_default_no_structural_predictions.zh_target_strict_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step7_identifier_augmented_default",
        "role": "step7_operational_identifier_control",
        "kind": "prediction",
        "path": Path("reports/step7_identifier_augmented_default_predictions.zh_target_strict_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_e5_lr_l2_50pct_seed_20260320",
        "role": "step9_clean_current_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_e5_lr_l2_50pct_seed_20260321",
        "role": "step9_clean_current_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_e5_lr_l2_50pct_seed_20260322",
        "role": "step9_clean_current_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260320",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_50pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260321",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_50pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260322",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_50pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260320",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260321",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260322",
        "role": "step9_clean_minority_regularization_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
        "optional_until_generated": True,
    },
    {
        "model_id": "step9_bge_m3_residual_lr_100pct_seed_20260320",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_bge_m3_residual_lr_100pct_seed_20260321",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_bge_m3_residual_lr_100pct_seed_20260322",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_labse_lr_l2_100pct_seed_20260320",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_labse_lr_l2_100pct_seed_20260321",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_labse_lr_l2_100pct_seed_20260322",
        "role": "step9_clean_control_seed",
        "kind": "prediction",
        "path": Path("reports/step9_core_few_shot_labse_lr_l2_ratio_100pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_identifier_augmented_lr_l2_100pct_seed_20260320",
        "role": "step9_operational_identifier_seed",
        "kind": "prediction",
        "path": Path("reports/step9_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260320_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_identifier_augmented_lr_l2_100pct_seed_20260321",
        "role": "step9_operational_identifier_seed",
        "kind": "prediction",
        "path": Path("reports/step9_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260321_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
    {
        "model_id": "step9_identifier_augmented_lr_l2_100pct_seed_20260322",
        "role": "step9_operational_identifier_seed",
        "kind": "prediction",
        "path": Path("reports/step9_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_20260322_predictions.zh_test.csv"),
        "score_column": "prob_positive",
    },
]

ENSEMBLES: dict[str, dict[str, Any]] = {
    "step9_e5_lr_l2_50pct_seed_mean": {
        "role": "step9_clean_current_seed_mean",
        "members": [
            "step9_e5_lr_l2_50pct_seed_20260320",
            "step9_e5_lr_l2_50pct_seed_20260321",
            "step9_e5_lr_l2_50pct_seed_20260322",
        ],
    },
    "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean": {
        "role": "step9_clean_minority_regularization_seed_mean",
        "optional_until_generated": True,
        "members": [
            "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260320",
            "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260321",
            "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_20260322",
        ],
    },
    "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean": {
        "role": "step9_clean_minority_regularization_seed_mean",
        "optional_until_generated": True,
        "members": [
            "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260320",
            "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260321",
            "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_20260322",
        ],
    },
    "step9_bge_m3_residual_lr_100pct_seed_mean": {
        "role": "step9_clean_control_seed_mean",
        "members": [
            "step9_bge_m3_residual_lr_100pct_seed_20260320",
            "step9_bge_m3_residual_lr_100pct_seed_20260321",
            "step9_bge_m3_residual_lr_100pct_seed_20260322",
        ],
    },
    "step9_labse_lr_l2_100pct_seed_mean": {
        "role": "step9_clean_control_seed_mean",
        "members": [
            "step9_labse_lr_l2_100pct_seed_20260320",
            "step9_labse_lr_l2_100pct_seed_20260321",
            "step9_labse_lr_l2_100pct_seed_20260322",
        ],
    },
    "step9_identifier_augmented_lr_l2_100pct_seed_mean": {
        "role": "step9_operational_identifier_seed_mean",
        "members": [
            "step9_identifier_augmented_lr_l2_100pct_seed_20260320",
            "step9_identifier_augmented_lr_l2_100pct_seed_20260321",
            "step9_identifier_augmented_lr_l2_100pct_seed_20260322",
        ],
    },
}

PAIRED_COMPARISONS: list[tuple[str, str, str]] = [
    ("step9_e5_lr_l2_50pct_seed_mean", "raw_e5_cosine", "primary_clean_vs_raw_e5"),
    ("step9_e5_lr_l2_50pct_seed_20260320", "raw_e5_cosine", "primary_seed_vs_raw_e5"),
    ("step9_e5_lr_l2_50pct_seed_20260321", "raw_e5_cosine", "primary_seed_vs_raw_e5"),
    ("step9_e5_lr_l2_50pct_seed_20260322", "raw_e5_cosine", "primary_seed_vs_raw_e5"),
    ("step9_e5_lr_l2_50pct_seed_mean", "raw_labse_cosine", "primary_clean_vs_raw_labse"),
    ("step9_e5_lr_l2_50pct_seed_mean", "raw_bge_m3_cosine", "primary_clean_vs_raw_bge"),
    ("step9_e5_lr_l2_50pct_seed_mean", "step7_core_zero_shot_bge_m3", "primary_clean_vs_step7_bge_fusion"),
    ("step9_e5_lr_l2_50pct_seed_mean", "step7_core_zero_shot_default", "primary_clean_vs_step7_default_fusion"),
    (
        "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
        "raw_e5_cosine",
        "mixup_clean_vs_raw_e5",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
        "step9_e5_lr_l2_50pct_seed_mean",
        "mixup_clean_vs_non_mixup_e5_lr_l2",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
        "step7_core_zero_shot_default",
        "mixup_clean_vs_step7_default_fusion",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
        "raw_e5_cosine",
        "mixup_100pct_clean_vs_raw_e5",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
        "step9_e5_lr_l2_50pct_seed_mean",
        "mixup_100pct_clean_vs_non_mixup_e5_lr_l2_50pct",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
        "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
        "mixup_100pct_clean_vs_mixup_50pct",
    ),
    (
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
        "step7_core_zero_shot_default",
        "mixup_100pct_clean_vs_step7_default_fusion",
    ),
    ("step9_bge_m3_residual_lr_100pct_seed_mean", "raw_bge_m3_cosine", "bge_residual_vs_raw_bge"),
    ("step9_labse_lr_l2_100pct_seed_mean", "raw_labse_cosine", "labse_lr_vs_raw_labse"),
    ("step9_identifier_augmented_lr_l2_100pct_seed_mean", "raw_e5_cosine", "operational_identifier_vs_raw_e5"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_OUT_METRICS)
    parser.add_argument("--output-comparisons", type=Path, default=DEFAULT_OUT_COMPARISONS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(value: str, *, path: Path, column: str, pair_uid: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for {column} in {path} pair_uid={pair_uid}")
    return float(value)


def load_test_rows(labels_path: Path) -> list[dict[str, Any]]:
    rows = read_csv(labels_path)
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split_name") != "test":
            continue
        if row.get("review_label") not in {"positive", "negative"}:
            continue
        if row.get("usable_for_supervision") != "1":
            continue
        y_true = 1 if row["review_label"] == "positive" else 0
        group_id = row.get("split_component_id") or row.get("pair_uid")
        test_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "y_true": y_true,
                "group_id": group_id,
                "review_stratum": row.get("review_stratum", ""),
                "seller_uid_left": row.get("seller_uid_left", ""),
                "seller_uid_right": row.get("seller_uid_right", ""),
            }
        )
    if not test_rows:
        raise ValueError(f"No fixed zh_test supervision rows found in {labels_path}")
    labels = {row["y_true"] for row in test_rows}
    if labels != {0, 1}:
        raise ValueError("Fixed zh_test rows must contain both positive and negative labels")
    return test_rows


def load_score_map(spec: dict[str, Any]) -> dict[str, float]:
    path = Path(spec["path"])
    score_column = spec["score_column"]
    rows = read_csv(path)
    scores: dict[str, float] = {}
    for row in rows:
        pair_uid = row.get("pair_uid", "")
        if not pair_uid:
            continue
        scores[pair_uid] = to_float(row.get(score_column, ""), path=path, column=score_column, pair_uid=pair_uid)
    return scores


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = avg_rank
        i = j
    return ranks


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    if pos_count == 0 or neg_count == 0:
        return math.nan
    ranks = average_ranks(scores)
    pos_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)


def average_precision(y_true: list[int], scores: list[float]) -> float:
    pos_count = sum(y_true)
    if pos_count == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / pos_count


def metric_value(metric: str, y_true: list[int], scores: list[float]) -> float:
    if metric == "roc_auc":
        return roc_auc(y_true, scores)
    if metric == "average_precision":
        return average_precision(y_true, scores)
    raise ValueError(f"Unsupported metric: {metric}")


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[int(pos)]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def stddev(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return math.nan
    m = mean(clean)
    return math.sqrt(sum((value - m) ** 2 for value in clean) / (len(clean) - 1))


def build_bootstrap_indices(group_ids: list[str], resamples: int, seed: int) -> list[list[int]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        by_group[group_id].append(idx)
    groups = sorted(by_group)
    rng = random.Random(seed)
    samples: list[list[int]] = []
    for _ in range(resamples):
        sampled_indices: list[int] = []
        for _ in groups:
            group = rng.choice(groups)
            sampled_indices.extend(by_group[group])
        samples.append(sampled_indices)
    return samples


def subset(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[idx] for idx in indices]


def format_float(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def bootstrap_distribution(
    metric: str,
    y_true: list[int],
    scores: list[float],
    bootstrap_indices: list[list[int]],
) -> list[float]:
    values: list[float] = []
    for indices in bootstrap_indices:
        y_sample = subset(y_true, indices)
        if sum(y_sample) == 0 or sum(y_sample) == len(y_sample):
            continue
        values.append(metric_value(metric, y_sample, subset(scores, indices)))
    return values


def bootstrap_diff_distribution(
    metric: str,
    y_true: list[int],
    scores_a: list[float],
    scores_b: list[float],
    bootstrap_indices: list[list[int]],
) -> list[float]:
    values: list[float] = []
    for indices in bootstrap_indices:
        y_sample = subset(y_true, indices)
        if sum(y_sample) == 0 or sum(y_sample) == len(y_sample):
            continue
        a = metric_value(metric, y_sample, subset(scores_a, indices))
        b = metric_value(metric, y_sample, subset(scores_b, indices))
        if not math.isnan(a) and not math.isnan(b):
            values.append(a - b)
    return values


def bootstrap_sign_p(diff_values: list[float]) -> float:
    clean = [value for value in diff_values if not math.isnan(value)]
    if not clean:
        return math.nan
    le_zero = sum(1 for value in clean if value <= 0) / len(clean)
    ge_zero = sum(1 for value in clean if value >= 0) / len(clean)
    return min(1.0, 2.0 * min(le_zero, ge_zero))


def main() -> None:
    args = parse_args()
    test_rows = load_test_rows(args.labels)
    pair_uids = [row["pair_uid"] for row in test_rows]
    y_true = [int(row["y_true"]) for row in test_rows]
    group_ids = [row["group_id"] for row in test_rows]
    label_counts = Counter(y_true)
    group_counts = Counter(group_ids)

    score_maps: dict[str, dict[str, float]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    input_files = {args.labels, args.features}
    skipped_model_specs: list[dict[str, str]] = []
    for spec in MODEL_SPECS:
        spec = dict(spec)
        if spec["kind"] == "feature":
            spec["path"] = args.features
        if bool(spec.get("optional_until_generated", False)) and not Path(spec["path"]).exists():
            skipped_model_specs.append(
                {
                    "model_id": str(spec["model_id"]),
                    "reason": "optional_prediction_file_not_found",
                    "path": str(spec["path"]),
                }
            )
            continue
        score_map = load_score_map(spec)
        missing = [pair_uid for pair_uid in pair_uids if pair_uid not in score_map]
        if missing:
            raise ValueError(
                f"{spec['model_id']} is missing {len(missing)} fixed-test pair scores; "
                f"first missing pair_uid={missing[0]}"
            )
        model_id = spec["model_id"]
        score_maps[model_id] = score_map
        model_metadata[model_id] = {
            "role": spec["role"],
            "kind": spec["kind"],
            "path": str(spec["path"]),
            "score_column": spec["score_column"],
        }
        input_files.add(Path(spec["path"]))

    for model_id, cfg in ENSEMBLES.items():
        members = cfg["members"]
        missing_members = [member for member in members if member not in score_maps]
        if missing_members and bool(cfg.get("optional_until_generated", False)):
            skipped_model_specs.append(
                {
                    "model_id": model_id,
                    "reason": "optional_ensemble_members_not_found",
                    "path": " || ".join(missing_members),
                }
            )
            continue
        for member in members:
            if member not in score_maps:
                raise ValueError(f"Ensemble {model_id} references missing member {member}")
        score_maps[model_id] = {
            pair_uid: sum(score_maps[member][pair_uid] for member in members) / len(members)
            for pair_uid in pair_uids
        }
        model_metadata[model_id] = {
            "role": cfg["role"],
            "kind": "seed_mean",
            "path": "",
            "score_column": "mean(prob_positive)",
            "members": members,
        }

    bootstrap_indices = build_bootstrap_indices(group_ids, args.resamples, args.seed)
    lower_q = (1.0 - CONFIDENCE_LEVEL) / 2.0
    upper_q = 1.0 - lower_q

    metrics_rows: list[dict[str, Any]] = []
    metrics_json: dict[str, Any] = {}
    for model_id in sorted(score_maps):
        scores = [score_maps[model_id][pair_uid] for pair_uid in pair_uids]
        metrics_json[model_id] = {}
        for metric in ("roc_auc", "average_precision"):
            observed = metric_value(metric, y_true, scores)
            distribution = bootstrap_distribution(metric, y_true, scores, bootstrap_indices)
            row = {
                "model_id": model_id,
                "role": model_metadata[model_id]["role"],
                "metric": metric,
                "observed": format_float(observed),
                "bootstrap_mean": format_float(mean(distribution)),
                "bootstrap_std": format_float(stddev(distribution)),
                "ci_lower": format_float(percentile(distribution, lower_q)),
                "ci_upper": format_float(percentile(distribution, upper_q)),
                "valid_resamples": len(distribution),
                "n_rows": len(y_true),
                "n_positive": label_counts[1],
                "n_negative": label_counts[0],
                "n_groups": len(group_counts),
            }
            metrics_rows.append(row)
            metrics_json[model_id][metric] = {
                "observed": observed,
                "bootstrap_mean": mean(distribution),
                "bootstrap_std": stddev(distribution),
                "ci_lower": percentile(distribution, lower_q),
                "ci_upper": percentile(distribution, upper_q),
                "valid_resamples": len(distribution),
            }

    comparison_rows: list[dict[str, Any]] = []
    comparison_json: list[dict[str, Any]] = []
    skipped_paired_comparisons: list[dict[str, str]] = []
    for model_a, model_b, comparison_role in PAIRED_COMPARISONS:
        if model_a not in score_maps or model_b not in score_maps:
            skipped_paired_comparisons.append(
                {
                    "comparison_role": comparison_role,
                    "model_a": model_a,
                    "model_b": model_b,
                    "reason": "model_scores_not_available",
                }
            )
            continue
        scores_a = [score_maps[model_a][pair_uid] for pair_uid in pair_uids]
        scores_b = [score_maps[model_b][pair_uid] for pair_uid in pair_uids]
        for metric in ("roc_auc", "average_precision"):
            observed_a = metric_value(metric, y_true, scores_a)
            observed_b = metric_value(metric, y_true, scores_b)
            observed_diff = observed_a - observed_b
            diff_distribution = bootstrap_diff_distribution(metric, y_true, scores_a, scores_b, bootstrap_indices)
            ci_lower = percentile(diff_distribution, lower_q)
            ci_upper = percentile(diff_distribution, upper_q)
            p_value = bootstrap_sign_p(diff_distribution)
            row = {
                "comparison_role": comparison_role,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "observed_a": format_float(observed_a),
                "observed_b": format_float(observed_b),
                "observed_diff_a_minus_b": format_float(observed_diff),
                "diff_bootstrap_mean": format_float(mean(diff_distribution)),
                "diff_ci_lower": format_float(ci_lower),
                "diff_ci_upper": format_float(ci_upper),
                "bootstrap_sign_p_two_sided": format_float(p_value),
                "valid_resamples": len(diff_distribution),
                "supports_positive_difference": "yes" if ci_lower > 0 else "no",
            }
            comparison_rows.append(row)
            comparison_json.append(
                {
                    "comparison_role": comparison_role,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "observed_a": observed_a,
                    "observed_b": observed_b,
                    "observed_diff_a_minus_b": observed_diff,
                    "diff_bootstrap_mean": mean(diff_distribution),
                    "diff_ci_lower": ci_lower,
                    "diff_ci_upper": ci_upper,
                    "bootstrap_sign_p_two_sided": p_value,
                    "valid_resamples": len(diff_distribution),
                    "supports_positive_difference": ci_lower > 0,
                }
            )

    write_csv(
        args.output_metrics,
        metrics_rows,
        [
            "model_id",
            "role",
            "metric",
            "observed",
            "bootstrap_mean",
            "bootstrap_std",
            "ci_lower",
            "ci_upper",
            "valid_resamples",
            "n_rows",
            "n_positive",
            "n_negative",
            "n_groups",
        ],
    )
    write_csv(
        args.output_comparisons,
        comparison_rows,
        [
            "comparison_role",
            "model_a",
            "model_b",
            "metric",
            "observed_a",
            "observed_b",
            "observed_diff_a_minus_b",
            "diff_bootstrap_mean",
            "diff_ci_lower",
            "diff_ci_upper",
            "bootstrap_sign_p_two_sided",
            "valid_resamples",
            "supports_positive_difference",
        ],
    )

    summary = {
        "audit_version": "step12_statistical_robustness_zh_test_20260513",
        "scope": "zh_target_strict_fixed_test",
        "fixed_test_policy": {
            "split_name": "test",
            "do_not_mix_train_valid_test": True,
            "grouping_unit": "split_component_id",
        },
        "bootstrap": {
            "num_resamples_requested": args.resamples,
            "random_seed": args.seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "resample_unit": "split_component_id",
            "skip_resamples_without_both_classes": True,
        },
        "dataset": {
            "n_rows": len(y_true),
            "n_positive": label_counts[1],
            "n_negative": label_counts[0],
            "n_groups": len(group_counts),
            "largest_group_size": max(group_counts.values()),
            "top_group_sizes": group_counts.most_common(10),
        },
        "inputs": {
            str(path): sha256_file(path)
            for path in sorted(input_files, key=lambda p: str(p))
            if path and path.exists()
        },
        "model_metadata": model_metadata,
        "skipped_model_specs": skipped_model_specs,
        "model_metrics": metrics_json,
        "paired_comparisons": comparison_json,
        "skipped_paired_comparisons": skipped_paired_comparisons,
        "outputs": {
            "summary_json": str(args.output_json),
            "model_metrics_csv": str(args.output_metrics),
            "paired_comparisons_csv": str(args.output_comparisons),
        },
        "interpretation_rules": [
            "A positive observed difference is not enough for a strong claim if the grouped bootstrap CI crosses 0.",
            "Raw semantic controls remain required reporting baselines.",
            "Identifier-augmented rows are operational controls, not clean scientific mainline evidence.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_metrics}")
    print(f"Wrote {args.output_comparisons}")


if __name__ == "__main__":
    main()
