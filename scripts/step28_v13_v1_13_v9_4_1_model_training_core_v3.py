#!/usr/bin/env python3
"""Frozen numerical core for the V9.4.1 M0/M1/M2/M3 experiment.

This module contains deterministic, testable numerical routines only.  It has
no formal supervision reader and no command that can train on the V9.4.1
dataset.  Formal I/O remains fail-closed until a later one-time authorization
pins the exact commit and public-projection manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common_v1
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT / "schema/step28_v13_v1_13_v9_4_1_model_training_policy_v3.json"
)
POLICY_SIZE_BYTES = 7878
POLICY_SHA256 = "5ae871ec70b80429bf20e7d3c2bf90ed42e75e8285bebdbf2f7fdac62e6a58da"
POLICY_CANONICAL_SELF_HASH = (
    "5e26b6c5fd6fea068c5a766ddb94302248f4db4d3ec3ea8d180f60334fa8b4bb"
)

MODEL_IDS = (
    "c0",
    "m0",
    "m1_r01",
    "m1_r02",
    "m1_r03",
    "m1_r04",
    "m1_r05",
    "m2",
    "m3_base",
    "m3_joint",
)
M1_IDS = ("m1_r01", "m1_r02", "m1_r03", "m1_r04", "m1_r05")
M1_REPEAT_IDS = ("r01", "r02", "r03", "r04", "r05")
FORMAL_SPLITS = ("train", "development", "audit_a", "audit_b")
FORMAL_REPORT_SPLITS = ("development", "audit_a", "audit_b")
RETRIEVAL_K = (1, 3, 5, 10)
L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
M3_GRID = (
    (3, 10, 0.03, 200),
    (3, 25, 0.03, 200),
    (7, 10, 0.03, 200),
    (7, 25, 0.03, 200),
    (3, 10, 0.1, 75),
    (3, 25, 0.1, 75),
    (7, 10, 0.1, 75),
    (7, 25, 0.1, 75),
)
BOOTSTRAP_SHA256 = {
    "development": "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661",
    "audit_a": "617be9200ad55b45eda8b1800989d7e0b50579bb53ecee675713f8ba2cd4c3e4",
    "audit_b": "12565157b109301070a3648989e74a1faab05d015b5ac0dbcd772c38a5a91a87",
}


class ModelTrainingV3Error(ValueError):
    """Raised when a frozen V3 scientific contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_record(spec: Mapping[str, Any], *, label: str) -> Path:
    path = ROOT / str(spec["path"])
    if not path.is_file():
        raise ModelTrainingV3Error(f"Missing {label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise ModelTrainingV3Error(f"{label} size drift")
    if sha256_file(path) != spec["sha256"]:
        raise ModelTrainingV3Error(f"{label} SHA-256 drift")
    return path


def _validate_policy_semantics(policy: Mapping[str, Any]) -> None:
    if policy.get("version") != "step28-v13-v1.13-v9.4.1-model-training-v3":
        raise ModelTrainingV3Error("V3 policy version drift")
    if policy.get("status") != (
        "FROZEN_IMPLEMENTATION_POLICY_NO_PUBLIC_PROJECTION_EXECUTION_"
        "NO_SUPERVISION_NO_TRAINING_NO_AUDIT_TRUTH"
    ):
        raise ModelTrainingV3Error("V3 policy status drift")
    scope = policy["implementation_scope"]
    if tuple(scope["formal_report_models"]) != MODEL_IDS:
        raise ModelTrainingV3Error("Formal model registry drift")
    if tuple(scope["m1_repeat_ids"]) != M1_REPEAT_IDS:
        raise ModelTrainingV3Error("M1 repeat registry drift")
    if scope["five_m1_probability_averaging_allowed"] is not False:
        raise ModelTrainingV3Error("M1 probability averaging was enabled")
    expected_future_pins = (
        "schema/step28_v13_v1_13_v9_4_1_model_training_policy_v3.json",
        "scripts/step28_v13_v1_13_v9_4_1_model_training_core_v3.py",
        "scripts/step28_v13_v1_13_v9_4_1_confirmatory_evaluator_v3.py",
        "scripts/step28_v13_v1_13_v9_4_1_blind_stage_protocol_v3.py",
        "tests/test_step28_v13_v1_13_v9_4_1_model_training_v3_contracts.py",
    )
    if tuple(scope["future_exact_commit_authorization_must_pin"]) != expected_future_pins:
        raise ModelTrainingV3Error("Future exact-commit implementation pin registry drift")
    expected_stage_order = (
        "freeze_v3_contract_policy_implementation_tests",
        "current_regression_commit_push_independent_review",
        "one_time_public_projection_authorization_and_freeze",
        "one_time_train_development_truth_and_training_authorization",
        "freeze_train_models_development_thresholds_and_development_gate",
        "freeze_audit_a_blind_predictions_without_truth",
        "one_time_audit_a_truth_authorization_and_evaluation",
        "freeze_audit_b_blind_predictions_only_after_pinned_audit_a_pass",
        "one_time_audit_b_truth_authorization_and_evaluation",
    )
    if tuple(policy["stage_order"]) != expected_stage_order:
        raise ModelTrainingV3Error("Formal stage order drift")
    inputs = policy["formal_input_contract"]
    expected_inputs = {
        "worlds_per_split": 500,
        "sellers_per_world": 28,
        "pairs_per_world": 378,
        "positive_pairs_per_world": 20,
        "negative_pairs_per_world": 358,
        "rows_per_split": 189000,
        "base_feature_count": 24,
        "identity_feature_count": 33,
        "joint_feature_count": 57,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ModelTrainingV3Error(f"Formal input contract drift: {key}")
    if tuple(inputs["split_order"]) != FORMAL_SPLITS:
        raise ModelTrainingV3Error("Formal split order drift")
    training = policy["training_contract"]
    if tuple(float(value) for value in training["m1_m2_shared_l2_grid"]) != L2_GRID:
        raise ModelTrainingV3Error("M1/M2 L2 grid drift")
    predecessor_v1 = common_v1.load_policy()
    predecessor_grid = tuple(
        tuple(value) for value in predecessor_v1["m3"]["grid_order"]
    )
    if predecessor_grid != M3_GRID:
        raise ModelTrainingV3Error("Inherited M3 grid drift")
    metrics = policy["metric_contract"]
    if not metrics["ap_and_trapezoidal_pr_auc_are_distinct"]:
        raise ModelTrainingV3Error("AP and trapezoidal PR-AUC were conflated")
    required_retrieval = {
        "map",
        "mrr",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "ndcg_at_10",
    }
    if set(metrics["retrieval_metrics"]) != required_retrieval:
        raise ModelTrainingV3Error("Retrieval metric registry drift")
    bootstrap = policy["bootstrap_contract"]
    if (
        bootstrap["replicates"] != 9999
        or bootstrap["world_count"] != 500
        or bootstrap["draw_size"] != 500
        or bootstrap["dtype"] != "<i8"
        or bootstrap["order"] != "C"
        or bootstrap["quantile_method"] != "linear"
        or bootstrap["formal_evaluator_accepts_precomputed_metric_series"] is not False
        or bootstrap["formal_evaluator_rehashes_actual_index_matrix"] is not True
    ):
        raise ModelTrainingV3Error("Bootstrap contract drift")
    for split, expected in BOOTSTRAP_SHA256.items():
        if bootstrap["splits"][split]["index_bytes_sha256"] != expected:
            raise ModelTrainingV3Error(f"Bootstrap digest drift: {split}")
    authorization = policy["authorization_state"]
    if set(authorization.values()) != {False}:
        raise ModelTrainingV3Error("A formal V3 authorization was enabled prematurely")
    if (
        policy["m0_m1_m2_m3_training_authorized"] is not False
        or policy["audit_a_truth_authorized"] is not False
        or policy["audit_b_truth_authorized"] is not False
    ):
        raise ModelTrainingV3Error("Top-level formal authorization drift")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY.resolve():
        raise ModelTrainingV3Error("Only the frozen V3 policy path is valid")
    raw = path.read_bytes()
    if len(raw) != POLICY_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise ModelTrainingV3Error("Frozen V3 policy bytes drift")
    policy = json.loads(raw.decode("utf-8"))
    recorded = policy.get("canonical_self_hash")
    body = dict(policy)
    body.pop("canonical_self_hash", None)
    if recorded != POLICY_CANONICAL_SELF_HASH or canonical_sha256(body) != recorded:
        raise ModelTrainingV3Error("Frozen V3 policy canonical self-hash drift")
    for label, spec in policy["authority_registry"].items():
        authority_path = _verify_file_record(spec, label=label)
        if "canonical_self_hash" in spec:
            value = json.loads(authority_path.read_text(encoding="utf-8"))
            stored = value.get("canonical_self_hash")
            stripped = dict(value)
            stripped.pop("canonical_self_hash", None)
            if stored != spec["canonical_self_hash"] or canonical_sha256(stripped) != stored:
                raise ModelTrainingV3Error(f"{label} canonical self-hash drift")
    common_v2.load_policy()
    _validate_policy_semantics(policy)
    return policy


def require_formal_execution_authorization(policy: Mapping[str, Any]) -> None:
    """Fail before any formal projection, truth read, or model write."""

    if set(policy["authorization_state"].values()) == {False}:
        raise ModelTrainingV3Error(
            "V3 is implementation-only; no public projection, supervision, training, "
            "blind-audit prediction, or audit-truth execution is authorized"
        )
    raise ModelTrainingV3Error(
        "The frozen V3 implementation policy cannot itself grant a formal capability; "
        "a future exact-commit one-time authorization validator is required"
    )


def _float64_vector(values: Sequence[float] | np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype="<f8")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ModelTrainingV3Error(f"{label} must be a finite one-dimensional float64 vector")
    return np.ascontiguousarray(result, dtype="<f8")


def _binary_labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.isin(result, (0, 1)).all():
        raise ModelTrainingV3Error("Labels must be a one-dimensional binary vector")
    return np.ascontiguousarray(result, dtype=np.int8)


def _world_ordinals(values: Sequence[int] | np.ndarray, row_count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != row_count:
        raise ModelTrainingV3Error("World ordinals do not align with rows")
    if not np.issubdtype(result.dtype, np.integer) or np.any(result < 0):
        raise ModelTrainingV3Error("World ordinals must be nonnegative integers")
    return np.ascontiguousarray(result, dtype=np.int64)


def validate_formal_split_layout(
    labels: Sequence[int] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = _binary_labels(labels)
    worlds = _world_ordinals(world_ordinals, len(y))
    if len(y) != 189000 or set(np.unique(worlds).tolist()) != set(range(500)):
        raise ModelTrainingV3Error("Formal split row/world cardinality drift")
    row_counts = np.bincount(worlds, minlength=500)
    positive_counts = np.bincount(worlds, weights=y, minlength=500)
    if not np.all(row_counts == 378):
        raise ModelTrainingV3Error("Formal world does not contain 378 rows")
    if not np.all(positive_counts == 20):
        raise ModelTrainingV3Error("Formal world does not contain 20 positive rows")
    return y, worlds


def _grouped_curve_counts(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order].astype(np.float64, copy=False)
    row_weights = (
        np.ones(len(scores), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)[order]
    )
    if row_weights.shape != scores.shape or not np.isfinite(row_weights).all() or np.any(row_weights < 0):
        raise ModelTrainingV3Error("Metric weights are invalid")
    positive_total = float(np.sum(row_weights * ordered_labels))
    negative_total = float(np.sum(row_weights * (1.0 - ordered_labels)))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ModelTrainingV3Error("Score metric requires both classes with positive weight")
    group_end = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    tp = np.cumsum(row_weights * ordered_labels, dtype=np.float64)[group_end]
    fp = np.cumsum(row_weights * (1.0 - ordered_labels), dtype=np.float64)[group_end]
    return tp, fp, positive_total, negative_total


def score_curve_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y = _binary_labels(labels)
    p = _float64_vector(scores, label="Scores")
    if len(y) != len(p):
        raise ModelTrainingV3Error("Labels and scores do not align")
    metric_weights = None if weights is None else _float64_vector(weights, label="Weights")
    if metric_weights is not None and len(metric_weights) != len(y):
        raise ModelTrainingV3Error("Metric weights do not align")
    tp, fp, positive_total, negative_total = _grouped_curve_counts(
        y, p, metric_weights
    )
    recall = tp / positive_total
    precision = np.divide(
        tp,
        tp + fp,
        out=np.ones_like(tp),
        where=(tp + fp) != 0.0,
    )
    previous_recall = np.r_[0.0, recall[:-1]]
    previous_precision = np.r_[1.0, precision[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))
    trapezoidal_pr_auc = float(
        np.sum((recall - previous_recall) * (precision + previous_precision) / 2.0)
    )
    tpr = recall
    fpr = fp / negative_total
    previous_tpr = np.r_[0.0, tpr[:-1]]
    previous_fpr = np.r_[0.0, fpr[:-1]]
    roc_auc = float(np.sum((fpr - previous_fpr) * (tpr + previous_tpr) / 2.0))
    recall_at_fpr = float(np.max(np.r_[0.0, tpr[fpr <= 0.01]]))
    return {
        "average_precision": average_precision,
        "trapezoidal_pr_auc": trapezoidal_pr_auc,
        "roc_auc": roc_auc,
        "recall_at_fpr_1pct": recall_at_fpr,
    }


def probabilistic_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y = _binary_labels(labels).astype(np.float64)
    p = _float64_vector(probabilities, label="Probabilities")
    if len(y) != len(p) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ModelTrainingV3Error("Probabilities are invalid or misaligned")
    w = np.ones(len(y), dtype=np.float64) if weights is None else _float64_vector(weights, label="Weights")
    if len(w) != len(y) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
        raise ModelTrainingV3Error("Probability metric weights are invalid")
    denominator = float(np.sum(w))
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return {
        "brier": float(np.sum(w * np.square(p - y)) / denominator),
        "log_loss": float(
            -np.sum(w * (y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))
            / denominator
        ),
    }


def confusion_counts(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, int]:
    y = _binary_labels(labels)
    p = _float64_vector(probabilities, label="Probabilities")
    if len(y) != len(p) or math.isnan(float(threshold)):
        raise ModelTrainingV3Error("Threshold confusion inputs are invalid")
    predicted = p >= float(threshold)
    positive = y == 1
    return {
        "tp": int(np.sum(predicted & positive)),
        "fp": int(np.sum(predicted & ~positive)),
        "fn": int(np.sum(~predicted & positive)),
        "tn": int(np.sum(~predicted & ~positive)),
    }


def _threshold_metrics_from_confusion(confusion: Mapping[str, float]) -> dict[str, float]:
    tp = float(confusion["tp"])
    fp = float(confusion["fp"])
    fn = float(confusion["fn"])
    tn = float(confusion["tn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": mcc,
    }


def world_equal_confusion(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = _binary_labels(labels)
    p = _float64_vector(probabilities, label="Probabilities")
    worlds = _world_ordinals(world_ordinals, len(y))
    if len(p) != len(y):
        raise ModelTrainingV3Error("World-equal confusion inputs do not align")
    values = {key: [] for key in ("tp", "fp", "fn", "tn")}
    for world in np.unique(worlds):
        selector = worlds == world
        counts = confusion_counts(y[selector], p[selector], threshold)
        denominator = int(np.sum(selector))
        for key in values:
            values[key].append(counts[key] / denominator)
    return {key: float(np.mean(value)) for key, value in values.items()}


def threshold_report(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    raw = confusion_counts(labels, probabilities, threshold)
    world_equal = world_equal_confusion(labels, probabilities, world_ordinals, threshold)
    return {
        "threshold": float(threshold),
        "raw_confusion_matrix": raw,
        "world_equal_confusion_matrix": world_equal,
        "raw_rows": _threshold_metrics_from_confusion(raw),
        "world_equal_confusion": _threshold_metrics_from_confusion(world_equal),
    }


def select_development_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
) -> float:
    y = _binary_labels(labels)
    p = _float64_vector(probabilities, label="Probabilities")
    worlds = _world_ordinals(world_ordinals, len(y))
    if len(p) != len(y):
        raise ModelTrainingV3Error("Threshold selection inputs do not align")
    if len(y) == 0:
        raise ModelTrainingV3Error("Threshold selection requires at least one row")

    # Each row contributes 1 / rows_in_its_world to the world-equal
    # confusion matrix.  Sorting once and updating cumulative weighted counts
    # is exactly the same decision rule as rescanning every row for every
    # unique threshold, but changes the runtime from quadratic to O(n log n).
    _, inverse, rows_per_world = np.unique(
        worlds, return_inverse=True, return_counts=True
    )
    row_weights = 1.0 / rows_per_world[inverse].astype(np.float64)
    order = np.argsort(-p, kind="stable")
    ordered_scores = p[order]
    ordered_labels = y[order].astype(np.float64)
    ordered_weights = row_weights[order]
    cumulative_tp = np.cumsum(ordered_weights * ordered_labels, dtype=np.float64)
    cumulative_fp = np.cumsum(
        ordered_weights * (1.0 - ordered_labels), dtype=np.float64
    )
    group_end = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    positive_total = float(np.sum(row_weights * y))

    # +inf predicts no positives.  Visiting score groups in descending order
    # means an exact F1 tie automatically retains the higher threshold.
    best_threshold = np.inf
    best_f1 = 0.0
    for index in group_end:
        tp = float(cumulative_tp[index])
        fp = float(cumulative_fp[index])
        fn = positive_total - tp
        denominator = 2.0 * tp + fp + fn
        f1 = 2.0 * tp / denominator if denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(ordered_scores[index])
    return best_threshold


def complete_classification_report(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = _binary_labels(labels)
    p = _float64_vector(probabilities, label="Probabilities")
    worlds = _world_ordinals(world_ordinals, len(y))
    pooled = score_curve_metrics(y, p) | probabilistic_metrics(y, p)
    per_world = []
    for world in np.unique(worlds):
        selector = worlds == world
        per_world.append(
            score_curve_metrics(y[selector], p[selector])
            | probabilistic_metrics(y[selector], p[selector])
        )
    world_equal_sensitivity = {
        key: float(np.mean([record[key] for record in per_world]))
        for key in per_world[0]
    }
    return {
        "pooled": pooled,
        "world_equal_sensitivity": world_equal_sensitivity,
        "threshold": threshold_report(y, p, worlds, threshold),
        "raw_probability_sha256": hashlib.sha256(p.tobytes(order="C")).hexdigest(),
        "log_loss_clipped_probability_sha256": hashlib.sha256(
            np.ascontiguousarray(np.clip(p, 1e-15, 1.0 - 1e-15), dtype="<f8").tobytes(order="C")
        ).hexdigest(),
    }


def retrieval_report(
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
    seller_uid_left: Sequence[str],
    seller_uid_right: Sequence[str],
    relevance: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    p = _float64_vector(probabilities, label="Probabilities")
    rel = _binary_labels(relevance)
    worlds = _world_ordinals(world_ordinals, len(p))
    if not (len(rel) == len(p) == len(seller_uid_left) == len(seller_uid_right)):
        raise ModelTrainingV3Error("Retrieval inputs do not align")
    world_records: list[dict[str, float]] = []
    for world in np.unique(worlds):
        indices = np.flatnonzero(worlds == world)
        sellers = sorted(
            set(seller_uid_left[index] for index in indices)
            | set(seller_uid_right[index] for index in indices),
            key=lambda value: value.encode("utf-8"),
        )
        if len(sellers) != 28 or len(indices) != 378:
            raise ModelTrainingV3Error("Retrieval world is not a complete K28 graph")
        pair_lookup: dict[tuple[str, str], tuple[float, int]] = {}
        for index in indices:
            left = str(seller_uid_left[index])
            right = str(seller_uid_right[index])
            if left == right:
                raise ModelTrainingV3Error("Retrieval graph contains a self-pair")
            key = tuple(sorted((left, right), key=lambda value: value.encode("utf-8")))
            if key in pair_lookup:
                raise ModelTrainingV3Error("Retrieval graph contains a duplicate pair")
            pair_lookup[key] = (float(p[index]), int(rel[index]))
        if len(pair_lookup) != 378:
            raise ModelTrainingV3Error("Retrieval graph pair universe drift")
        query_metrics: list[dict[str, float]] = []
        for query in sellers:
            candidates = []
            for candidate in sellers:
                if candidate == query:
                    continue
                key = tuple(
                    sorted((query, candidate), key=lambda value: value.encode("utf-8"))
                )
                score, is_relevant = pair_lookup[key]
                candidates.append((score, candidate, is_relevant))
            candidates.sort(key=lambda row: (-row[0], row[1].encode("utf-8")))
            relevant_total = sum(row[2] for row in candidates)
            if relevant_total not in (1, 2):
                raise ModelTrainingV3Error("Each retrieval query must have one or two relevant candidates")
            relevant_seen = 0
            ap_sum = 0.0
            first_rank = None
            metrics = {}
            for rank, (_, _, is_relevant) in enumerate(candidates, start=1):
                if is_relevant:
                    relevant_seen += 1
                    ap_sum += relevant_seen / rank
                    if first_rank is None:
                        first_rank = rank
            metrics["map"] = ap_sum / relevant_total
            metrics["mrr"] = 1.0 / float(first_rank)
            for k in RETRIEVAL_K:
                top_relevant = sum(row[2] for row in candidates[:k])
                metrics[f"recall_at_{k}"] = top_relevant / relevant_total
                dcg = sum(
                    row[2] / math.log2(rank + 1.0)
                    for rank, row in enumerate(candidates[:k], start=1)
                )
                ideal = sum(
                    1.0 / math.log2(rank + 1.0)
                    for rank in range(1, min(k, relevant_total) + 1)
                )
                metrics[f"ndcg_at_{k}"] = dcg / ideal
            query_metrics.append(metrics)
        world_records.append(
            {
                key: float(np.mean([record[key] for record in query_metrics]))
                for key in query_metrics[0]
            }
        )
    aggregate = {
        key: float(np.mean([record[key] for record in world_records]))
        for key in world_records[0]
    }
    return {"aggregate": aggregate, "per_world": world_records}


def validate_bootstrap_indices(
    policy: Mapping[str, Any], split: str, indices: np.ndarray
) -> np.ndarray:
    if split not in FORMAL_REPORT_SPLITS:
        raise ModelTrainingV3Error("Bootstrap split is not formal")
    matrix = np.asarray(indices)
    contract = policy["bootstrap_contract"]
    expected_shape = (contract["replicates"], contract["draw_size"])
    if (
        matrix.shape != expected_shape
        or matrix.dtype.str != "<i8"
        or not matrix.flags.c_contiguous
        or np.any(matrix < 0)
        or np.any(matrix >= contract["world_count"])
    ):
        raise ModelTrainingV3Error("Actual bootstrap index matrix shape/dtype/range drift")
    digest = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    if digest != contract["splits"][split]["index_bytes_sha256"]:
        raise ModelTrainingV3Error("Actual bootstrap index matrix digest drift")
    return matrix


def build_bootstrap_indices(policy: Mapping[str, Any], split: str) -> np.ndarray:
    contract = policy["bootstrap_contract"]
    spec = contract["splits"][split]
    generator = np.random.Generator(np.random.PCG64(int(spec["seed"])))
    matrix = generator.integers(
        0,
        contract["world_count"],
        size=(contract["replicates"], contract["draw_size"]),
        endpoint=False,
        dtype=np.int64,
    )
    matrix = np.ascontiguousarray(matrix, dtype="<i8")
    return validate_bootstrap_indices(policy, split, matrix)


def _bootstrap_world_multiplicity(index_batch: np.ndarray, world_count: int) -> np.ndarray:
    result = np.zeros((len(index_batch), world_count), dtype=np.int16)
    for row, draw in enumerate(index_batch):
        result[row] = np.bincount(draw, minlength=world_count)
    return result


def bootstrap_pooled_score_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    world_ordinals: Sequence[int] | np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Recompute pooled metrics under whole-world multiplicity.

    Equal-score rows remain a single threshold group.  The implementation
    reuses the fixed score order but recomputes weighted cumulative counts for
    every draw; it never substitutes mean per-world AP for pooled AP.
    """

    y = _binary_labels(labels)
    p = _float64_vector(probabilities, label="Probabilities")
    worlds = _world_ordinals(world_ordinals, len(y))
    matrix = np.asarray(indices)
    if len(p) != len(y) or matrix.ndim != 2 or batch_size <= 0:
        raise ModelTrainingV3Error("Bootstrap pooled metric inputs are invalid")
    world_count = matrix.shape[1]
    if set(np.unique(worlds).tolist()) != set(range(world_count)):
        raise ModelTrainingV3Error("Bootstrap rows do not cover the expected world ordinals")
    order = np.argsort(-p, kind="stable")
    sorted_scores = p[order]
    sorted_labels = y[order].astype(np.float64)
    sorted_worlds = worlds[order]
    group_end = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    per_world_positive = np.bincount(worlds, weights=y, minlength=world_count)
    per_world_negative = np.bincount(worlds, weights=1 - y, minlength=world_count)
    per_world_rows = np.bincount(worlds, minlength=world_count)
    brier_row = np.square(p - y)
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    log_row = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    per_world_brier = np.bincount(worlds, weights=brier_row, minlength=world_count)
    per_world_log = np.bincount(worlds, weights=log_row, minlength=world_count)
    output = {
        name: np.empty(len(matrix), dtype="<f8")
        for name in (
            "average_precision",
            "trapezoidal_pr_auc",
            "roc_auc",
            "brier",
            "log_loss",
            "recall_at_fpr_1pct",
        )
    }
    for start in range(0, len(matrix), batch_size):
        stop = min(start + batch_size, len(matrix))
        multiplicity = _bootstrap_world_multiplicity(matrix[start:stop], world_count)
        positive_total = multiplicity @ per_world_positive
        negative_total = multiplicity @ per_world_negative
        row_total = multiplicity @ per_world_rows
        if np.any(positive_total <= 0) or np.any(negative_total <= 0):
            raise ModelTrainingV3Error("Bootstrap draw lost a class")
        row_weights = multiplicity[:, sorted_worlds]
        tp = np.cumsum(row_weights * sorted_labels, axis=1, dtype=np.float64)[:, group_end]
        fp = np.cumsum(row_weights * (1.0 - sorted_labels), axis=1, dtype=np.float64)[:, group_end]
        recall = tp / positive_total[:, None]
        denominator = tp + fp
        precision = np.divide(
            tp,
            denominator,
            out=np.ones_like(tp),
            where=denominator != 0.0,
        )
        previous_recall = np.concatenate(
            (np.zeros((len(multiplicity), 1)), recall[:, :-1]), axis=1
        )
        previous_precision = np.concatenate(
            (np.ones((len(multiplicity), 1)), precision[:, :-1]), axis=1
        )
        output["average_precision"][start:stop] = np.sum(
            (recall - previous_recall) * precision, axis=1
        )
        output["trapezoidal_pr_auc"][start:stop] = np.sum(
            (recall - previous_recall) * (precision + previous_precision) / 2.0,
            axis=1,
        )
        fpr = fp / negative_total[:, None]
        previous_tpr = previous_recall
        previous_fpr = np.concatenate(
            (np.zeros((len(multiplicity), 1)), fpr[:, :-1]), axis=1
        )
        output["roc_auc"][start:stop] = np.sum(
            (fpr - previous_fpr) * (recall + previous_tpr) / 2.0, axis=1
        )
        allowed = fpr <= 0.01
        output["recall_at_fpr_1pct"][start:stop] = np.max(
            np.where(allowed, recall, 0.0), axis=1
        )
        output["brier"][start:stop] = (multiplicity @ per_world_brier) / row_total
        output["log_loss"][start:stop] = (multiplicity @ per_world_log) / row_total
    return output


def residual_objective_and_gradient(
    beta: np.ndarray,
    p0: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    base = common_v2.validate_p0(p0)
    features = np.asarray(phi, dtype="<f8", order="C")
    y = _binary_labels(labels).astype(np.float64)
    coefficients = np.asarray(beta, dtype="<f8")
    if (
        features.ndim != 2
        or features.shape[0] != len(base)
        or coefficients.shape != (features.shape[1],)
        or len(y) != len(base)
        or not np.isfinite(features).all()
        or not np.isfinite(coefficients).all()
        or not math.isfinite(float(l2))
        or l2 <= 0.0
    ):
        raise ModelTrainingV3Error("Residual objective inputs are invalid")
    eta = common_v2.raw_logit(base) + features @ coefficients
    data_loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    probabilities = np.empty_like(eta)
    nonnegative = eta >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-eta[nonnegative]))
    exponential = np.exp(eta[~nonnegative])
    probabilities[~nonnegative] = exponential / (1.0 + exponential)
    gradient = features.T @ (probabilities - y) / len(y) + float(l2) * coefficients
    objective = data_loss + 0.5 * float(l2) * float(coefficients @ coefficients)
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise ModelTrainingV3Error("Residual objective became non-finite")
    return objective, np.ascontiguousarray(gradient, dtype="<f8")


def _fit_m2_with_fitted_transform_for_fixture(
    p0: np.ndarray,
    identity33: np.ndarray,
    labels: np.ndarray,
    world_uids: Sequence[str],
    l2: float,
) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - formal runtime gate
        raise ModelTrainingV3Error("SciPy is required for frozen residual fitting") from exc
    identity = np.asarray(identity33, dtype="<f8", order="C")
    if identity.shape != (len(p0), 33) or len(world_uids) != len(p0):
        raise ModelTrainingV3Error("Residual training rows are misaligned")
    scale, mu = common_v1.fit_identity_transform(identity, world_uids)
    phi, active = common_v1.apply_identity_transform(identity, scale, mu)

    def evaluate(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return residual_objective_and_gradient(beta, p0, phi, labels, l2)

    result = minimize(
        evaluate,
        np.zeros(33, dtype="<f8"),
        jac=True,
        method="L-BFGS-B",
        bounds=None,
        options={
            "gtol": 1e-8,
            "ftol": 0.0,
            "maxiter": 10000,
            "maxfun": 200000,
            "maxls": 50,
        },
    )
    beta = np.ascontiguousarray(result.x, dtype="<f8")
    objective, gradient = evaluate(beta)
    gradient_norm = float(np.max(np.abs(gradient)))
    if (
        not bool(result.success)
        or not np.isfinite(beta).all()
        or not math.isfinite(objective)
        or gradient_norm > 1e-7
    ):
        raise ModelTrainingV3Error(
            "Frozen residual optimizer did not satisfy success/gradient gates: "
            f"success={bool(result.success)} gradient_inf={gradient_norm}"
        )
    probabilities = common_v2.residual_probabilities(p0, phi, beta, active)
    return {
        "beta": beta,
        "scale": np.ascontiguousarray(scale, dtype="<f8"),
        "mu": np.ascontiguousarray(mu, dtype="<f8"),
        "objective": objective,
        "gradient_infinity_norm": gradient_norm,
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "probabilities": probabilities,
    }


def fit_residual_with_frozen_transform(
    p0: np.ndarray,
    model_identity33: np.ndarray,
    labels: np.ndarray,
    scale: np.ndarray,
    mu: np.ndarray,
    l2: float,
) -> dict[str, Any]:
    """Fit one M1/M2 beta with a transform learned from correct identity rows."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - formal runtime gate
        raise ModelTrainingV3Error("SciPy is required for frozen residual fitting") from exc
    identity = np.asarray(model_identity33, dtype="<f8", order="C")
    if identity.shape != (len(p0), 33):
        raise ModelTrainingV3Error("Residual identity matrix shape drift")
    phi, active = common_v1.apply_identity_transform(identity, scale, mu)

    def evaluate(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return residual_objective_and_gradient(beta, p0, phi, labels, l2)

    result = minimize(
        evaluate,
        np.zeros(33, dtype="<f8"),
        jac=True,
        method="L-BFGS-B",
        bounds=None,
        options={
            "gtol": 1e-8,
            "ftol": 0.0,
            "maxiter": 10000,
            "maxfun": 200000,
            "maxls": 50,
        },
    )
    beta = np.ascontiguousarray(result.x, dtype="<f8")
    objective, gradient = evaluate(beta)
    gradient_norm = float(np.max(np.abs(gradient)))
    if (
        not bool(result.success)
        or not np.isfinite(beta).all()
        or not math.isfinite(objective)
        or gradient_norm > 1e-7
    ):
        raise ModelTrainingV3Error(
            "Frozen residual optimizer did not satisfy success/gradient gates: "
            f"success={bool(result.success)} gradient_inf={gradient_norm}"
        )
    return {
        "beta": beta,
        "scale": np.ascontiguousarray(scale, dtype="<f8"),
        "mu": np.ascontiguousarray(mu, dtype="<f8"),
        "objective": objective,
        "gradient_infinity_norm": gradient_norm,
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "training_probabilities": common_v2.residual_probabilities(
            p0, phi, beta, active
        ),
    }


def _fold_array(world_uids: Sequence[str]) -> np.ndarray:
    assignments = common_v1.assign_world_folds(sorted(set(world_uids)))
    result = np.asarray([assignments[value] for value in world_uids], dtype=np.int8)
    if set(result.tolist()) != set(range(5)):
        raise ModelTrainingV3Error("Training rows do not cover the common five folds")
    return result


def _validate_formal_training_rows(
    labels: np.ndarray, world_uids: Sequence[str]
) -> None:
    if len(labels) != 189000 or len(world_uids) != len(labels):
        raise ModelTrainingV3Error("Formal training row count drift")
    unique_worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    if len(unique_worlds) != 500:
        raise ModelTrainingV3Error("Formal training world count drift")
    world_array = np.asarray(world_uids, dtype=object)
    for world_uid in unique_worlds:
        selector = world_array == world_uid
        if int(np.sum(selector)) != 378 or int(np.sum(labels[selector])) != 20:
            raise ModelTrainingV3Error(
                "Formal training world does not contain exactly 378 rows and 20 positives"
            )


def _validate_m1_source_indices(
    mappings: Mapping[str, np.ndarray],
    world_uids: Sequence[str],
    seller_uid_left: Sequence[str] | None = None,
    seller_uid_right: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    row_count = len(world_uids)
    expected = set(M1_IDS)
    if set(mappings) != expected:
        raise ModelTrainingV3Error("M1 mapping registry drift")
    worlds = np.asarray(world_uids, dtype=object)
    if (seller_uid_left is None) != (seller_uid_right is None):
        raise ModelTrainingV3Error("M1 endpoint inputs must be supplied together")
    if seller_uid_left is not None and (
        len(seller_uid_left) != row_count or len(seller_uid_right) != row_count
    ):
        raise ModelTrainingV3Error("M1 endpoint rows do not align")
    validated: dict[str, np.ndarray] = {}
    complete = np.arange(row_count, dtype=np.int64)
    for model_id in M1_IDS:
        indices = np.asarray(mappings[model_id])
        if (
            indices.shape != (row_count,)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= row_count)
        ):
            raise ModelTrainingV3Error(f"{model_id} mapping shape/range drift")
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        if np.any(indices == complete):
            raise ModelTrainingV3Error(f"{model_id} mapping contains a fixed row")
        if np.any(worlds[indices] != worlds):
            raise ModelTrainingV3Error(f"{model_id} mapping crosses worlds")
        for world_uid in sorted(set(world_uids), key=lambda value: value.encode("utf-8")):
            target = np.flatnonzero(worlds == world_uid)
            if set(indices[target].tolist()) != set(target.tolist()):
                raise ModelTrainingV3Error(f"{model_id} is not a within-world bijection")
            if seller_uid_left is not None and seller_uid_right is not None:
                pair_to_index = {}
                sellers = set()
                for row_index in target:
                    pair = common_v1.canonical_pair_endpoints(
                        str(seller_uid_left[row_index]), str(seller_uid_right[row_index])
                    )
                    if pair in pair_to_index:
                        raise ModelTrainingV3Error("M1 endpoint table contains duplicate pairs")
                    pair_to_index[pair] = int(row_index)
                    sellers.update(pair)
                if len(sellers) != 28 or len(pair_to_index) != 378:
                    raise ModelTrainingV3Error("M1 endpoint table is not a complete K28 graph")
                repeat_id = model_id.removeprefix("m1_")
                expected_mapping = common_v1.build_m1_mapping(
                    world_uid,
                    sorted(sellers, key=lambda value: value.encode("utf-8")),
                    repeat_id,
                )
                for target_pair, source_pair in expected_mapping.items():
                    target_index = pair_to_index[target_pair]
                    expected_source_index = pair_to_index[source_pair]
                    if int(indices[target_index]) != expected_source_index:
                        raise ModelTrainingV3Error(
                            f"{model_id} does not match the frozen one-factor mapping"
                        )
        validated[model_id] = indices
    digests = {
        hashlib.sha256(value.astype("<i8", copy=False).tobytes(order="C")).hexdigest()
        for value in validated.values()
    }
    if len(digests) != 5:
        raise ModelTrainingV3Error("Five M1 mappings are not distinct")
    return validated


def fit_m1_m2_family(
    p0: np.ndarray,
    identity33: np.ndarray,
    labels: np.ndarray,
    world_uids: Sequence[str],
    m1_source_indices: Mapping[str, np.ndarray],
    seller_uid_left: Sequence[str] | None = None,
    seller_uid_right: Sequence[str] | None = None,
    *,
    enforce_runtime: bool = True,
) -> dict[str, Any]:
    """Select the shared L2 and fit five M1 models plus M2.

    Every fold estimates one transform from the *correct*, unpermuted identity
    rows before any labels are used by the optimizer.  M1 uses mapped rows only
    for fitting; all held-out predictions use the correct observable rows.
    """

    if enforce_runtime:
        validate_supervised_runtime()
    base = common_v2.validate_p0(p0)
    identity = np.asarray(identity33, dtype="<f8", order="C")
    y = _binary_labels(labels)
    if identity.shape != (len(base), 33) or len(y) != len(base) or len(world_uids) != len(base):
        raise ModelTrainingV3Error("M1/M2 family inputs are misaligned")
    if enforce_runtime:
        _validate_formal_training_rows(y, world_uids)
        if seller_uid_left is None or seller_uid_right is None:
            raise ModelTrainingV3Error(
                "Formal M1/M2 fitting requires endpoint rows to replay exact M1 maps"
            )
    mappings = _validate_m1_source_indices(
        m1_source_indices, world_uids, seller_uid_left, seller_uid_right
    )
    folds = _fold_array(world_uids)
    family_ids = M1_IDS + ("m2",)
    oof_by_l2: dict[float, dict[str, np.ndarray]] = {}
    shared_loss_by_l2: dict[float, float] = {}
    for l2 in L2_GRID:
        oof = {model_id: np.empty(len(base), dtype="<f8") for model_id in family_ids}
        for fold in range(5):
            train = folds != fold
            validation = ~train
            train_worlds = [world_uids[index] for index in np.flatnonzero(train)]
            scale, mu = common_v1.fit_identity_transform(identity[train], train_worlds)
            m2_artifact = fit_residual_with_frozen_transform(
                base[train], identity[train], y[train], scale, mu, l2
            )
            oof["m2"][validation] = predict_residual_model(
                m2_artifact, base[validation], identity[validation]
            )
            for model_id in M1_IDS:
                mapped_train_identity = identity[mappings[model_id][train]]
                artifact = fit_residual_with_frozen_transform(
                    base[train], mapped_train_identity, y[train], scale, mu, l2
                )
                oof[model_id][validation] = predict_residual_model(
                    artifact, base[validation], identity[validation]
                )
        losses = {
            model_id: probabilistic_metrics(y, oof[model_id])["log_loss"]
            for model_id in family_ids
        }
        shared_loss_by_l2[l2] = 0.5 * losses["m2"] + 0.5 * float(
            np.mean([losses[model_id] for model_id in M1_IDS])
        )
        oof_by_l2[l2] = oof
    selected_l2 = common_v1.select_shared_l2(shared_loss_by_l2)
    scale, mu = common_v1.fit_identity_transform(identity, world_uids)
    artifacts = {
        "m2": fit_residual_with_frozen_transform(
            base, identity, y, scale, mu, selected_l2
        )
    }
    for model_id in M1_IDS:
        artifacts[model_id] = fit_residual_with_frozen_transform(
            base, identity[mappings[model_id]], y, scale, mu, selected_l2
        )
    return {
        "selected_l2": selected_l2,
        "shared_loss_by_l2": shared_loss_by_l2,
        "selected_oof_probabilities": oof_by_l2[selected_l2],
        "artifacts": artifacts,
        "folds": folds,
        "common_scale": scale,
        "common_mu": mu,
    }


def _lightgbm_classifier(grid_entry: Sequence[float | int]) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - formal runtime gate
        raise ModelTrainingV3Error("LightGBM 4.6.0 is required for M3") from exc
    leaves, child, rate, trees = grid_entry
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=int(leaves),
        min_child_samples=int(child),
        learning_rate=float(rate),
        n_estimators=int(trees),
        max_depth=-1,
        max_bin=63,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_split_gain=0.0,
        deterministic=True,
        force_col_wise=True,
        num_threads=1,
        random_state=20260810,
        bagging_seed=20260810,
        feature_fraction_seed=20260810,
        data_random_seed=20260810,
    )


def fit_m3_one(
    matrix: np.ndarray,
    labels: np.ndarray,
    world_uids: Sequence[str],
    *,
    enforce_runtime: bool = True,
) -> dict[str, Any]:
    if enforce_runtime:
        validate_supervised_runtime()
    values = np.asarray(matrix, dtype="<f8", order="C")
    y = _binary_labels(labels)
    if values.ndim != 2 or len(values) != len(y) or len(world_uids) != len(y) or np.isinf(values).any():
        raise ModelTrainingV3Error("M3 inputs are invalid or misaligned")
    if enforce_runtime:
        _validate_formal_training_rows(y, world_uids)
    folds = _fold_array(world_uids)
    oof_by_grid: list[np.ndarray] = []
    ap_by_grid: list[float] = []
    for grid_entry in M3_GRID:
        oof = np.empty(len(y), dtype="<f8")
        for fold in range(5):
            train = folds != fold
            validation = ~train
            medians = common_v1.finite_median_impute_fit(values[train])
            train_matrix = common_v1.impute_with_medians(values[train], medians)
            validation_matrix = common_v1.impute_with_medians(values[validation], medians)
            model = _lightgbm_classifier(grid_entry)
            model.fit(train_matrix, y[train])
            oof[validation] = np.asarray(
                model.predict_proba(validation_matrix)[:, 1], dtype="<f8"
            )
        if not np.isfinite(oof).all() or np.any(oof < 0.0) or np.any(oof > 1.0):
            raise ModelTrainingV3Error("M3 fold produced invalid probabilities")
        oof_by_grid.append(oof)
        ap_by_grid.append(score_curve_metrics(y, oof)["average_precision"])
    best_ap = max(ap_by_grid)
    selected_index = next(index for index, value in enumerate(ap_by_grid) if value == best_ap)
    medians = common_v1.finite_median_impute_fit(values)
    full_matrix = common_v1.impute_with_medians(values, medians)
    model = _lightgbm_classifier(M3_GRID[selected_index])
    model.fit(full_matrix, y)
    return {
        "selected_grid_index": selected_index,
        "selected_grid": M3_GRID[selected_index],
        "oof_average_precision_by_grid": ap_by_grid,
        "selected_oof_probabilities": oof_by_grid[selected_index],
        "medians": medians,
        "model": model,
        "folds": folds,
    }


def fit_m3_family(
    base24: np.ndarray,
    identity33: np.ndarray,
    labels: np.ndarray,
    world_uids: Sequence[str],
    *,
    enforce_runtime: bool = True,
) -> dict[str, Any]:
    base = np.asarray(base24, dtype="<f8", order="C")
    identity = np.asarray(identity33, dtype="<f8", order="C")
    if base.shape != (len(labels), 24) or identity.shape != (len(labels), 33):
        raise ModelTrainingV3Error("M3 base/joint feature shape drift")
    if not np.isfinite(identity).all():
        raise ModelTrainingV3Error("M3 identity33 contains non-finite values")
    joint = np.ascontiguousarray(np.column_stack((base, identity)), dtype="<f8")
    return {
        "m3_base": fit_m3_one(
            base, labels, world_uids, enforce_runtime=enforce_runtime
        ),
        "m3_joint": fit_m3_one(
            joint, labels, world_uids, enforce_runtime=enforce_runtime
        ),
    }


def predict_residual_model(
    artifact: Mapping[str, Any],
    p0: np.ndarray,
    identity33: np.ndarray,
) -> np.ndarray:
    phi, active = common_v1.apply_identity_transform(
        identity33,
        np.asarray(artifact["scale"], dtype="<f8"),
        np.asarray(artifact["mu"], dtype="<f8"),
    )
    return common_v2.residual_probabilities(
        p0, phi, np.asarray(artifact["beta"], dtype="<f8"), active
    )


def validate_supervised_runtime() -> dict[str, str]:
    return common_v1.validate_supervised_cpu_runtime(common_v1.load_policy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract", "formal-run"))
    args = parser.parse_args()
    policy = load_policy()
    if args.command == "formal-run":
        require_formal_execution_authorization(policy)
    print(
        json.dumps(
            {
                "status": "PASSED_V3_CONTRACT_VALIDATION_NO_FORMAL_EXECUTION",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_execution_performed": False,
                "supervision_or_audit_truth_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
