#!/usr/bin/env python3
"""Separate numerical evaluator for V9.4.1 development and blind audits.

The only evaluation API accepts model predictions, pair/world rows, labels,
retrieval relevance, thresholds, and the actual paired-world index matrix in a
single call.  It never accepts precomputed AP or bootstrap metric series.
Formal truth I/O is deliberately unavailable until a later split-specific,
exact-commit authorization wrapper is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


HIGHER_IS_BETTER = (
    "average_precision",
    "trapezoidal_pr_auc",
    "roc_auc",
    "recall_at_fpr_1pct",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
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
)
LOWER_IS_BETTER = ("brier", "log_loss")
COMPARISONS = (
    ("primary_m2_vs_mean_five_m1", "m2", "mean_five_individual_m1_metrics"),
    ("m2_vs_m0", "m2", "m0"),
    ("m3_joint_vs_m3_base", "m3_joint", "m3_base"),
    ("m2_vs_m3_joint", "m2", "m3_joint"),
    ("m1_equivalence_vs_m0", "mean_five_individual_m1_metrics", "m0"),
)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _world_multiplicity(indices: np.ndarray, world_count: int) -> np.ndarray:
    result = np.zeros((len(indices), world_count), dtype=np.int16)
    for row, draw in enumerate(indices):
        result[row] = np.bincount(draw, minlength=world_count)
    return result


def _per_world_score_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    worlds: np.ndarray,
) -> dict[str, np.ndarray]:
    records = []
    for world in range(int(np.max(worlds)) + 1):
        selector = worlds == world
        records.append(
            core.score_curve_metrics(labels[selector], probabilities[selector])
            | core.probabilistic_metrics(labels[selector], probabilities[selector])
        )
    return {
        key: np.asarray([record[key] for record in records], dtype="<f8")
        for key in records[0]
    }


def _per_world_confusion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    worlds: np.ndarray,
    threshold: float,
) -> np.ndarray:
    rows = []
    for world in range(int(np.max(worlds)) + 1):
        selector = worlds == world
        counts = core.confusion_counts(labels[selector], probabilities[selector], threshold)
        denominator = int(np.sum(selector))
        rows.append([counts[key] / denominator for key in ("tp", "fp", "fn", "tn")])
    return np.asarray(rows, dtype="<f8")


def _threshold_series(confusion: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, fn, tn = (confusion[:, index] for index in range(4))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0.0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0.0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) != 0.0)
    f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) != 0.0,
    )
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(tp),
        where=denominator != 0.0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": mcc,
    }


def _intervals(series: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {"q025": _quantile(values, 0.025), "q975": _quantile(values, 0.975)}
        for name, values in series.items()
    }


def _model_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    worlds: np.ndarray,
    seller_left: Sequence[str],
    seller_right: Sequence[str],
    relevance: np.ndarray,
    threshold: float,
    indices: np.ndarray,
    multiplicity: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    pooled = core.bootstrap_pooled_score_metrics(
        labels, probabilities, worlds, indices
    )
    per_world_score = _per_world_score_metrics(labels, probabilities, worlds)
    world_equal_series = {
        f"world_equal_{name}": (multiplicity @ values) / indices.shape[1]
        for name, values in per_world_score.items()
    }
    per_world_confusion = _per_world_confusion(labels, probabilities, worlds, threshold)
    confusion_series = (multiplicity @ per_world_confusion) / indices.shape[1]
    threshold_series = _threshold_series(confusion_series)
    retrieval = core.retrieval_report(
        probabilities, worlds, seller_left, seller_right, relevance
    )
    retrieval_keys = tuple(retrieval["aggregate"])
    per_world_retrieval = np.asarray(
        [[record[key] for key in retrieval_keys] for record in retrieval["per_world"]],
        dtype="<f8",
    )
    boot_retrieval = (multiplicity @ per_world_retrieval) / indices.shape[1]
    retrieval_series = {
        key: boot_retrieval[:, index] for index, key in enumerate(retrieval_keys)
    }
    series = pooled | threshold_series | retrieval_series | world_equal_series
    point = core.complete_classification_report(labels, probabilities, worlds, threshold)
    point["retrieval"] = retrieval["aggregate"]
    return series, point, retrieval


def _mean_m1_series(
    model_series: Mapping[str, Mapping[str, np.ndarray]], metric: str
) -> np.ndarray:
    return np.mean(
        np.vstack([model_series[model_id][metric] for model_id in core.M1_IDS]), axis=0
    )


def _mean_m1_point(model_points: Mapping[str, Mapping[str, Any]], metric: str) -> float:
    values = []
    for model_id in core.M1_IDS:
        point = model_points[model_id]
        if metric in point["pooled"]:
            values.append(point["pooled"][metric])
        elif metric in point["threshold"]["world_equal_confusion"]:
            values.append(point["threshold"]["world_equal_confusion"][metric])
        else:
            values.append(point["retrieval"][metric])
    return float(np.mean(values))


def _point_metric(point: Mapping[str, Any], metric: str) -> float:
    if metric in point["pooled"]:
        return float(point["pooled"][metric])
    if metric in point["threshold"]["world_equal_confusion"]:
        return float(point["threshold"]["world_equal_confusion"][metric])
    return float(point["retrieval"][metric])


def _comparison_result(
    model_series: Mapping[str, Mapping[str, np.ndarray]],
    model_points: Mapping[str, Mapping[str, Any]],
    target: str,
    control: str,
    metric: str,
) -> dict[str, float]:
    if target == "mean_five_individual_m1_metrics":
        target_point = _mean_m1_point(model_points, metric)
        target_series = _mean_m1_series(model_series, metric)
    else:
        target_point = _point_metric(model_points[target], metric)
        target_series = model_series[target][metric]
    if control == "mean_five_individual_m1_metrics":
        control_point = _mean_m1_point(model_points, metric)
        control_series = _mean_m1_series(model_series, metric)
    else:
        control_point = _point_metric(model_points[control], metric)
        control_series = model_series[control][metric]
    if metric in LOWER_IS_BETTER:
        point = control_point - target_point
        series = control_series - target_series
    else:
        point = target_point - control_point
        series = target_series - control_series
    return {
        "point": float(point),
        "q025": _quantile(series, 0.025),
        "q05": _quantile(series, 0.05),
        "q95": _quantile(series, 0.95),
        "q975": _quantile(series, 0.975),
    }


def _confirmatory_gate(
    split: str,
    model_series: Mapping[str, Mapping[str, np.ndarray]],
    model_points: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    m1_mean_series = _mean_m1_series(model_series, "average_precision")
    m1_mean_point = _mean_m1_point(model_points, "average_precision")
    m0_series = model_series["m0"]["average_precision"]
    m0_point = _point_metric(model_points["m0"], "average_precision")
    equivalence_series = m1_mean_series - m0_series
    equivalence = {
        "point": m1_mean_point - m0_point,
        "q05": _quantile(equivalence_series, 0.05),
        "q95": _quantile(equivalence_series, 0.95),
    }
    equivalence["passed"] = bool(
        equivalence["q05"] > -0.01 and equivalence["q95"] < 0.01
    )
    if split == "development":
        return {
            "status": (
                "PASSED_DEVELOPMENT_M1_M0_EQUIVALENCE_GATE"
                if equivalence["passed"]
                else "FAILED_DEVELOPMENT_M1_M0_EQUIVALENCE_GATE_NO_AUDIT_TRUTH_ALLOWED"
            ),
            "m1_m0_equivalence": equivalence,
            "future_audit_truth_authorization_may_be_requested": bool(
                equivalence["passed"]
            ),
            "audit_truth_opened_or_authorized_by_this_result": False,
        }
    m2_series = model_series["m2"]["average_precision"]
    m2_point = _point_metric(model_points["m2"], "average_precision")
    comparison_ids = (
        "m2_minus_mean_five_individual_m1",
        "m2_minus_m1_r01",
        "m2_minus_m1_r02",
        "m2_minus_m1_r03",
        "m2_minus_m1_r04",
        "m2_minus_m1_r05",
    )
    point_deltas = [m2_point - m1_mean_point] + [
        m2_point - _point_metric(model_points[model_id], "average_precision")
        for model_id in core.M1_IDS
    ]
    bootstrap_deltas = np.column_stack(
        [m2_series - m1_mean_series]
        + [m2_series - model_series[model_id]["average_precision"] for model_id in core.M1_IDS]
    )
    point_array = np.asarray(point_deltas, dtype="<f8")
    errors = point_array[None, :] - bootstrap_deltas
    critical_value = _quantile(np.max(errors, axis=1), 0.95)
    lower_bounds = point_array - critical_value
    primary_required = 0.03 if split == "audit_a" else 0.015
    primary_pass = bool(lower_bounds[0] > primary_required)
    individual_pass = bool(np.all(lower_bounds[1:] > 0.0))
    passed = bool(primary_pass and individual_pass and equivalence["passed"])
    return {
        "status": (
            f"PASSED_{split.upper()}_NUMERICAL_CONFIRMATORY_GATE"
            if passed
            else f"FAILED_{split.upper()}_NUMERICAL_CONFIRMATORY_GATE"
        ),
        "formal_conclusion_authorized": False,
        "comparison_order": list(comparison_ids),
        "point_deltas": point_array.tolist(),
        "simultaneous_critical_value": critical_value,
        "simultaneous_lower_bounds": lower_bounds.tolist(),
        "primary_required_strictly_greater_than": primary_required,
        "m1_m0_equivalence": equivalence,
        "numerical_gate_passed": passed,
        "audit_b_cannot_rescue_audit_a_failure": True,
    }


def evaluate_split_from_raw_inputs(
    *,
    policy: Mapping[str, Any],
    split: str,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
    world_ordinals: Sequence[int] | np.ndarray,
    seller_uid_left: Sequence[str],
    seller_uid_right: Sequence[str],
    labels: Sequence[int] | np.ndarray,
    retrieval_relevance: Sequence[int] | np.ndarray,
    actual_bootstrap_indices: np.ndarray,
    frozen_development_thresholds: Mapping[str, float] | None = None,
    enforce_formal_layout: bool = True,
) -> dict[str, Any]:
    """Compute all metrics and gates from raw, model-keyed inputs.

    There is intentionally no parameter for AP points, AP bootstrap series,
    comparison deltas, or a caller-supplied gate decision.
    """

    if split not in core.FORMAL_REPORT_SPLITS:
        raise core.ModelTrainingV3Error("Evaluator split is not formal")
    if set(predictions) != set(core.MODEL_IDS) or set(thresholds) != set(core.MODEL_IDS):
        raise core.ModelTrainingV3Error("Evaluator model registry drift")
    if enforce_formal_layout:
        frozen_policy = core.load_policy()
        if core.canonical_json_bytes(dict(policy)) != core.canonical_json_bytes(frozen_policy):
            raise core.ModelTrainingV3Error("Evaluator did not receive the frozen V3 policy")
    y = core._binary_labels(labels)
    relevance = core._binary_labels(retrieval_relevance)
    worlds = core._world_ordinals(world_ordinals, len(y))
    if not np.array_equal(y, relevance):
        raise core.ModelTrainingV3Error("Classification labels and retrieval relevance disagree")
    if enforce_formal_layout:
        core.validate_formal_split_layout(y, worlds)
        indices = core.validate_bootstrap_indices(policy, split, actual_bootstrap_indices)
        predecessor = core.common_v1.load_policy()
        if float(thresholds["m0"]) != float(predecessor["frozen_models"]["m0"]["threshold"]):
            raise core.ModelTrainingV3Error("Formal M0 threshold drift")
        if float(thresholds["c0"]) != float(predecessor["frozen_models"]["c0"]["threshold"]):
            raise core.ModelTrainingV3Error("Formal C0 threshold drift")
        if split == "development":
            if frozen_development_thresholds is not None:
                raise core.ModelTrainingV3Error(
                    "Development evaluation must derive and freeze its own thresholds"
                )
            for model_id in core.M1_IDS + ("m2", "m3_base", "m3_joint"):
                expected_threshold = core.select_development_threshold(
                    y, predictions[model_id], worlds
                )
                if float(thresholds[model_id]) != expected_threshold:
                    raise core.ModelTrainingV3Error(
                        f"Formal development threshold drift for {model_id}"
                    )
        else:
            if (
                frozen_development_thresholds is None
                or set(frozen_development_thresholds) != set(core.MODEL_IDS)
            ):
                raise core.ModelTrainingV3Error(
                    "Audit evaluation requires the complete frozen development threshold registry"
                )
            for model_id in core.MODEL_IDS:
                if float(thresholds[model_id]) != float(
                    frozen_development_thresholds[model_id]
                ):
                    raise core.ModelTrainingV3Error(
                        f"Audit threshold differs from frozen development threshold: {model_id}"
                    )
    else:
        indices = np.asarray(actual_bootstrap_indices)
        world_count = len(np.unique(worlds))
        if (
            indices.ndim != 2
            or indices.shape[1] != world_count
            or np.any(indices < 0)
            or np.any(indices >= world_count)
        ):
            raise core.ModelTrainingV3Error("Fixture bootstrap matrix drift")
    multiplicity = _world_multiplicity(indices, indices.shape[1])
    model_series: dict[str, dict[str, np.ndarray]] = {}
    model_points: dict[str, dict[str, Any]] = {}
    model_intervals: dict[str, dict[str, dict[str, float]]] = {}
    model_series_digests: dict[str, dict[str, str]] = {}
    probability_digests: dict[str, str] = {}
    for model_id in core.MODEL_IDS:
        probability = core._float64_vector(predictions[model_id], label=model_id)
        if len(probability) != len(y) or np.any(probability < 0.0) or np.any(probability > 1.0):
            raise core.ModelTrainingV3Error(f"{model_id} probability vector drift")
        threshold = float(thresholds[model_id])
        if math.isnan(threshold):
            raise core.ModelTrainingV3Error(f"{model_id} threshold is NaN")
        series, point, _ = _model_bootstrap_metrics(
            y,
            probability,
            worlds,
            seller_uid_left,
            seller_uid_right,
            relevance,
            threshold,
            indices,
            multiplicity,
        )
        model_series[model_id] = series
        model_points[model_id] = point
        model_intervals[model_id] = _intervals(series)
        model_series_digests[model_id] = {
            metric: hashlib.sha256(
                np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
            ).hexdigest()
            for metric, values in series.items()
        }
        probability_digests[model_id] = hashlib.sha256(
            probability.tobytes(order="C")
        ).hexdigest()
    comparison_metrics = tuple(HIGHER_IS_BETTER) + tuple(LOWER_IS_BETTER)
    comparisons = {
        comparison_id: {
            metric: _comparison_result(
                model_series, model_points, target, control, metric
            )
            for metric in comparison_metrics
        }
        for comparison_id, target, control in COMPARISONS
    }
    gate = _confirmatory_gate(split, model_series, model_points)
    return {
        "status": "NUMERICAL_EVALUATION_COMPLETE_NO_FORMAL_TRUTH_AUTHORITY",
        "split": split,
        "model_order": list(core.MODEL_IDS),
        "model_points": model_points,
        "model_bootstrap_intervals": model_intervals,
        "model_bootstrap_series_sha256": model_series_digests,
        "model_probability_sha256": probability_digests,
        "comparisons": comparisons,
        "gate": gate,
        "bootstrap_index_sha256": hashlib.sha256(
            np.ascontiguousarray(indices, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "precomputed_metric_series_accepted": False,
        "formal_conclusion_authorized": False,
    }


def require_split_specific_formal_authorization() -> None:
    raise core.ModelTrainingV3Error(
        "The numerical V3 evaluator has no formal truth capability.  A later "
        "split-specific exact-commit authorization wrapper must bind the blind "
        "prediction manifest and consume its receipt before opening truth."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract", "formal-evaluate"))
    args = parser.parse_args()
    policy = core.load_policy()
    if args.command == "formal-evaluate":
        require_split_specific_formal_authorization()
    print(
        json.dumps(
            {
                "status": "PASSED_SEPARATE_EVALUATOR_CONTRACT_NO_TRUTH_READ",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_evaluation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
