#!/usr/bin/env python3
"""Numerical training and evaluation helpers used by Step7-v3.1."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

import step7_v3_1_source_data as common


EPS = 1e-12
SHORTCUT_CONTROL_STRATUM_FIELD = "_shortcut_control_stratum"

def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        raise ValueError("Average precision requires at least one positive")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    cumulative_positive = 0
    cumulative_total = 0
    result = 0.0
    start = 0
    while start < len(y_sorted):
        stop = start + 1
        while stop < len(y_sorted) and s_sorted[stop] == s_sorted[start]:
            stop += 1
        group_positive = int(np.sum(y_sorted[start:stop] == 1))
        cumulative_positive += group_positive
        cumulative_total += stop - start
        if group_positive:
            result += (group_positive / positives) * (
                cumulative_positive / cumulative_total
            )
        start = stop
    return float(result)


def weighted_average_precision(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    """Tie-aware AP in which each component may contribute equal total mass."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or w.shape != y.shape:
        raise ValueError("Weighted average precision input shape mismatch")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("Weighted average precision requires finite positive weights")
    positive_mass = float(np.sum(w[y == 1]))
    if positive_mass <= 0.0:
        raise ValueError("Weighted average precision requires positive label mass")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    w_sorted = w[order]
    cumulative_positive_mass = 0.0
    cumulative_total_mass = 0.0
    result = 0.0
    start = 0
    while start < len(y_sorted):
        stop = start + 1
        while stop < len(y_sorted) and s_sorted[stop] == s_sorted[start]:
            stop += 1
        group_weights = w_sorted[start:stop]
        group_positive_mass = float(
            np.sum(group_weights[y_sorted[start:stop] == 1])
        )
        cumulative_positive_mass += group_positive_mass
        cumulative_total_mass += float(np.sum(group_weights))
        if group_positive_mass > 0.0:
            result += (group_positive_mass / positive_mass) * (
                cumulative_positive_mass / cumulative_total_mass
            )
        start = stop
    return float(result)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(scores, dtype=np.float64) >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "accuracy": float((tp + tn) / len(y)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def weighted_threshold_selection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(scores, dtype=np.float64) >= float(threshold)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape or w.shape != y.shape:
        raise ValueError("Weighted threshold input shape mismatch")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("Weighted threshold selection requires finite positive weights")
    tp = float(np.sum(w[(y == 1) & pred]))
    tn = float(np.sum(w[(y == 0) & ~pred]))
    fp = float(np.sum(w[(y == 0) & pred]))
    fn = float(np.sum(w[(y == 1) & ~pred]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "weighted_true_positive_mass": tp,
        "weighted_true_negative_mass": tn,
        "weighted_false_positive_mass": fp,
        "weighted_false_negative_mass": fn,
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "weighted_specificity": float(specificity),
        "weighted_balanced_accuracy": float((recall + specificity) / 2.0),
    }


def choose_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, dict]:
    selection_weights = (
        np.ones(len(labels), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    thresholds = [float(np.nextafter(unique[-1], math.inf))]
    thresholds.extend(float(value) for value in unique)
    best = None
    best_key = None
    for threshold in thresholds:
        result = weighted_threshold_selection_metrics(
            labels, scores, threshold, selection_weights
        )
        key = (
            result["weighted_balanced_accuracy"],
            result["weighted_f1"],
            result["weighted_precision"],
            result["weighted_recall"],
            threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = result
    assert best is not None
    best = {
        **best,
        "weight_total": float(np.sum(selection_weights)),
        "unweighted_metrics_at_selected_threshold": threshold_metrics(
            labels, scores, float(best["threshold"])
        ),
    }
    return float(best["threshold"]), best

def labelled_pair_ranking_metrics(
    rows: list[dict], labels: np.ndarray, scores: np.ndarray
) -> dict:
    queries: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for row, label, score in zip(rows, labels, scores, strict=True):
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        queries[left].append((right, int(label), float(score)))
        queries[right].append((left, int(label), float(score)))
    reciprocal_ranks = []
    hits_at_1 = []
    recalls = {1: [], 3: [], 5: []}
    candidate_counts = []
    positive_counts = []
    for query_uid, candidates in sorted(queries.items()):
        positive_count = sum(label == 1 for _candidate, label, _score in candidates)
        if positive_count == 0:
            continue
        ranked = sorted(candidates, key=lambda item: (-item[2], item[0]))
        first_positive_rank = next(
            rank for rank, (_candidate, label, _score) in enumerate(ranked, start=1) if label == 1
        )
        reciprocal_ranks.append(1.0 / first_positive_rank)
        hits_at_1.append(float(first_positive_rank == 1))
        for k in recalls:
            found = sum(label == 1 for _candidate, label, _score in ranked[:k])
            recalls[k].append(found / positive_count)
        candidate_counts.append(len(ranked))
        positive_counts.append(positive_count)
    if not reciprocal_ranks:
        return {
            "status": "not_estimable_no_query_with_positive_label",
            "eligible_query_count": 0,
        }
    return {
        "status": "diagnostic_labelled_pair_set_only",
        "eligible_query_count": len(reciprocal_ranks),
        "mean_candidate_count": float(np.mean(candidate_counts)),
        "mean_positive_count": float(np.mean(positive_counts)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hits_at_1": float(np.mean(hits_at_1)),
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_3": float(np.mean(recalls[3])),
        "recall_at_5": float(np.mean(recalls[5])),
    }


def full_metrics(
    rows: list[dict], labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    output = {
        "row_count": len(y),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "positive_prevalence": float(np.mean(y)),
        "roc_auc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    }
    output.update(threshold_metrics(y, p, threshold))
    output["labelled_candidate_ranking"] = labelled_pair_ranking_metrics(rows, y, p)
    return output


def component_weights(rows: list[dict], mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(rows), dtype=np.float64)
    if mode != "component_equal_normalized_to_row_count":
        raise ValueError(f"Unsupported Step7-v3 weighting mode: {mode}")
    counts = Counter(row["component_id"] for row in rows)
    weights = np.asarray([1.0 / counts[row["component_id"]] for row in rows], dtype=np.float64)
    return weights * (len(rows) / float(np.sum(weights)))

def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
    max_iter: int,
    tolerance: float,
    armijo_c1: float,
    minimum_line_search_step: float,
) -> dict:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(w) != len(y):
        raise ValueError("Step7-v3 logistic input shape mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("Step7-v3 logistic input contains non-finite values")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Step7-v3 logistic training requires both binary classes")
    if np.any(w <= 0.0):
        raise ValueError("Step7-v3 logistic weights must be positive")
    weight_total = float(np.sum(w))
    mean = np.sum(x * w[:, None], axis=0) / weight_total
    variance = np.sum(((x - mean) ** 2) * w[:, None], axis=0) / weight_total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    params = np.zeros(z.shape[1] + 1, dtype=np.float64)
    converged = False
    final_delta = math.inf
    final_normalized_gradient_inf_norm = math.inf
    final_objective = math.inf

    def state(current: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        logits = current[0] + z @ current[1:]
        probabilities = safe_sigmoid(logits)
        objective = float(
            np.sum(w * (np.logaddexp(0.0, logits) - y * logits))
            + 0.5 * float(l2_penalty) * np.dot(current[1:], current[1:])
        )
        residual = (probabilities - y) * w
        gradient = np.empty_like(current)
        gradient[0] = np.sum(residual)
        gradient[1:] = z.T @ residual + float(l2_penalty) * current[1:]
        curvature = probabilities * (1.0 - probabilities) * w
        return objective, gradient, curvature

    for iteration in range(1, int(max_iter) + 1):
        objective, gradient, curvature = state(params)
        normalized_gradient_inf_norm = float(
            np.max(np.abs(gradient)) / weight_total
        )
        if normalized_gradient_inf_norm <= float(tolerance):
            converged = True
            final_delta = 0.0
            final_normalized_gradient_inf_norm = normalized_gradient_inf_norm
            final_objective = objective
            break
        weighted_z = z * curvature[:, None]
        hessian = np.empty((len(params), len(params)), dtype=np.float64)
        hessian[0, 0] = np.sum(curvature)
        hessian[0, 1:] = np.sum(weighted_z, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = z.T @ weighted_z
        hessian[1:, 1:] += np.eye(z.shape[1]) * float(l2_penalty)
        try:
            newton_delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            newton_delta = np.linalg.pinv(hessian) @ gradient
        direction = -newton_delta
        directional_derivative = float(np.dot(gradient, direction))
        if not np.all(np.isfinite(direction)) or directional_derivative >= 0.0:
            direction = -gradient / max(float(np.linalg.norm(gradient)), 1.0)
            directional_derivative = float(np.dot(gradient, direction))
        step_size = 1.0
        accepted = False
        while step_size >= float(minimum_line_search_step):
            proposed = params + step_size * direction
            proposed_objective, _proposed_gradient, _proposed_curvature = state(
                proposed
            )
            if proposed_objective <= (
                objective + float(armijo_c1) * step_size * directional_derivative
            ):
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            raise ValueError(
                "Step7-v3 logistic Armijo line search failed before convergence: "
                f"l2={l2_penalty} gradient={normalized_gradient_inf_norm}"
            )
        applied_delta = step_size * direction
        params = proposed
        final_delta = float(np.linalg.norm(applied_delta))
        final_objective, final_gradient, _final_curvature = state(params)
        final_normalized_gradient_inf_norm = float(
            np.max(np.abs(final_gradient)) / weight_total
        )
        if final_normalized_gradient_inf_norm <= float(tolerance):
            converged = True
            break
    if not converged:
        raise ValueError(
            "Step7-v3 logistic solver did not converge: "
            f"l2={l2_penalty} delta={final_delta} "
            f"normalized_gradient={final_normalized_gradient_inf_norm}"
        )
    return {
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "intercept": float(params[0]),
        "coefficients": [float(value) for value in params[1:]],
        "l2_penalty": float(l2_penalty),
        "solver_iterations": iteration,
        "solver_final_delta_norm": final_delta,
        "solver_final_normalized_gradient_inf_norm": (
            final_normalized_gradient_inf_norm
        ),
        "solver_final_objective": final_objective,
        "solver_convergence_criterion": "normalized_gradient_inf_norm_at_most_tolerance",
        "solver_line_search": "armijo_backtracking",
        "solver_converged": converged,
        "sample_weight_total": weight_total,
    }

def apply_logistic(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if x.shape[1] != len(coefficients):
        raise ValueError("Step7-v3 logistic artifact feature dimension mismatch")
    return safe_sigmoid(
        float(artifact["intercept"]) + ((x - mean) / scale) @ coefficients
    )


def balanced_component_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["component_id"]].append(row)
    if len(grouped) < fold_count:
        raise ValueError("Step7-v3 has fewer training components than folds")
    totals = (
        len(rows) / fold_count,
        sum(row["review_label"] == "positive" for row in rows) / fold_count,
        sum(row["review_label"] == "negative" for row in rows) / fold_count,
    )
    records = []
    for component_id, component_rows in grouped.items():
        positives = sum(row["review_label"] == "positive" for row in component_rows)
        negatives = len(component_rows) - positives
        mass = (
            len(component_rows) / max(totals[0], 1.0)
            + positives / max(totals[1], 1.0)
            + negatives / max(totals[2], 1.0)
        )
        tie = hashlib.sha256(f"{seed}|{component_id}".encode("utf-8")).hexdigest()
        records.append((component_id, len(component_rows), positives, negatives, mass, tie))
    records.sort(key=lambda item: (-item[4], item[5], item[0]))
    if sum(record[2] > 0 for record in records) < fold_count:
        raise ValueError("Step7-v3 has too few positive components for grouped folds")
    if sum(record[3] > 0 for record in records) < fold_count:
        raise ValueError("Step7-v3 has too few negative components for grouped folds")

    assignments: dict[str, int] = {}
    fold_totals = [[0, 0, 0] for _ in range(fold_count)]
    remaining = list(records)

    def assign(record: tuple, fold: int) -> None:
        component_id, count, positives, negatives, _mass, _tie = record
        assignments[component_id] = fold
        fold_totals[fold][0] += count
        fold_totals[fold][1] += positives
        fold_totals[fold][2] += negatives
        remaining.remove(record)

    for fold in range(fold_count):
        candidate = next(record for record in remaining if record[2] > 0)
        assign(candidate, fold)
    for fold in range(fold_count):
        if fold_totals[fold][2] == 0:
            candidate = next(record for record in remaining if record[3] > 0)
            assign(candidate, fold)
    for record in list(remaining):
        best_fold = min(
            range(fold_count),
            key=lambda fold: (
                ((fold_totals[fold][0] + record[1]) / max(totals[0], 1.0)) ** 2
                + ((fold_totals[fold][1] + record[2]) / max(totals[1], 1.0)) ** 2
                + ((fold_totals[fold][2] + record[3]) / max(totals[2], 1.0)) ** 2,
                fold_totals[fold][0],
                fold,
            ),
        )
        assign(record, best_fold)
    for fold, (_count, positives, negatives) in enumerate(fold_totals):
        if positives == 0 or negatives == 0:
            raise ValueError(f"Step7-v3 grouped fold {fold} is single-class")
    return assignments


def component_fold_diagnostics(
    rows: list[dict], assignments: dict[str, int], fold_count: int
) -> list[dict]:
    component_sizes = Counter(row["component_id"] for row in rows)
    diagnostics = []
    for fold in range(fold_count):
        holdout = [row for row in rows if assignments[row["component_id"]] == fold]
        labels = np.asarray(
            [1 if row["review_label"] == "positive" else 0 for row in holdout],
            dtype=np.int8,
        )
        equal_weights = component_weights(
            holdout, "component_equal_normalized_to_row_count"
        )
        component_ids = {row["component_id"] for row in holdout}
        diagnostics.append(
            {
                "fold": fold,
                "row_count": len(holdout),
                "component_count": len(component_ids),
                "positive_count": int(np.sum(labels == 1)),
                "negative_count": int(np.sum(labels == 0)),
                "largest_component_row_count": max(
                    component_sizes[component_id] for component_id in component_ids
                ),
                "largest_component_row_fraction": float(
                    max(component_sizes[component_id] for component_id in component_ids)
                    / len(holdout)
                ),
                "component_equal_positive_weight_fraction": float(
                    np.sum(equal_weights[labels == 1]) / np.sum(equal_weights)
                ),
            }
        )
    return diagnostics

def tune_and_fit(
    train_rows: list[dict],
    train_matrix: np.ndarray,
    feature_names: list[str],
    policy: dict,
    weighting_mode: str,
) -> dict:
    cfg = policy["training"]
    labels = np.asarray(
        [1 if row["review_label"] == "positive" else 0 for row in train_rows],
        dtype=np.int8,
    )
    folds = balanced_component_folds(
        train_rows, int(cfg["fold_count"]), int(cfg["fold_seed"])
    )
    fold_diagnostics = component_fold_diagnostics(
        train_rows, folds, int(cfg["fold_count"])
    )
    oof_selection_weights = component_weights(train_rows, weighting_mode)
    grid_results = []
    best = None
    for l2_penalty in [float(value) for value in cfg["l2_grid"]]:
        oof = np.full(len(train_rows), np.nan, dtype=np.float64)
        fold_artifacts = []
        for fold in range(int(cfg["fold_count"])):
            fit_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(train_rows)
                    if folds[row["component_id"]] != fold
                ],
                dtype=int,
            )
            hold_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(train_rows)
                    if folds[row["component_id"]] == fold
                ],
                dtype=int,
            )
            fit_rows = [train_rows[index] for index in fit_indices]
            weights = component_weights(fit_rows, weighting_mode)
            artifact = fit_logistic(
                train_matrix[fit_indices],
                labels[fit_indices],
                weights,
                l2_penalty,
                int(cfg["max_iter"]),
                float(cfg["tolerance"]),
                float(cfg["armijo_c1"]),
                float(cfg["minimum_line_search_step"]),
            )
            oof[hold_indices] = apply_logistic(train_matrix[hold_indices], artifact)
            fold_artifacts.append(
                {
                    "fold": fold,
                    "fit_row_count": len(fit_indices),
                    "holdout_row_count": len(hold_indices),
                    "solver_iterations": artifact["solver_iterations"],
                }
            )
        if not np.all(np.isfinite(oof)):
            raise ValueError("Step7-v3 OOF predictions are incomplete")
        shortcut_conditioned = shortcut_conditioned_component_equal_average_precision(
            train_rows,
            labels,
            oof,
            policy,
            require_expected_strata=True,
        )
        weighted_ap = weighted_average_precision(labels, oof, oof_selection_weights)
        row_ap = average_precision(labels, oof)
        auc = roc_auc(labels, oof)
        result = {
            "l2_penalty": l2_penalty,
            "oof_shortcut_conditioned_macro_component_equal_average_precision": (
                shortcut_conditioned["macro_average_precision"]
            ),
            "oof_shortcut_conditioned_details": shortcut_conditioned,
            "oof_selection_weighted_average_precision": weighted_ap,
            "oof_row_average_precision": row_ap,
            "oof_roc_auc": auc,
            "folds": fold_artifacts,
            "oof_scores": oof,
        }
        grid_results.append(result)
        # Shortcut-conditioned AP selects regularization.  Global component-equal
        # AP is the first tie-breaker; exact ties prefer the stronger L2.
        key = (
            shortcut_conditioned["macro_average_precision"],
            weighted_ap,
            l2_penalty,
        )
        if best is None or key > best[0]:
            best = (key, result)
    assert best is not None
    selected = best[1]
    threshold, threshold_summary = choose_threshold(
        labels, selected["oof_scores"], oof_selection_weights
    )
    final_weights = component_weights(train_rows, weighting_mode)
    final_artifact = fit_logistic(
        train_matrix,
        labels,
        final_weights,
        float(selected["l2_penalty"]),
        int(cfg["max_iter"]),
        float(cfg["tolerance"]),
        float(cfg["armijo_c1"]),
        float(cfg["minimum_line_search_step"]),
    )
    final_artifact["feature_names"] = feature_names
    return {
        "weighting_mode": weighting_mode,
        "selected_l2_penalty": float(selected["l2_penalty"]),
        "l2_grid": [
            {
                key: value
                for key, value in result.items()
                if key != "oof_scores"
            }
            for result in grid_results
        ],
        "train_oof_scores": selected["oof_scores"],
        "train_oof_metrics": full_metrics(
            train_rows, labels, selected["oof_scores"], threshold
        ),
        "train_oof_selection_weighted_average_precision": float(
            selected["oof_selection_weighted_average_precision"]
        ),
        "train_oof_shortcut_conditioned_macro_component_equal_average_precision": float(
            selected[
                "oof_shortcut_conditioned_macro_component_equal_average_precision"
            ]
        ),
        "selected_threshold": threshold,
        "threshold_selection": threshold_summary,
        "final_train_artifact": final_artifact,
        "component_fold_assignment_sha256": common.canonical_hash(folds),
        "component_fold_diagnostics": fold_diagnostics,
    }

def verify_model_fingerprint(model_key: str, fingerprint: dict, cfg: dict) -> None:
    files = fingerprint.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Step7-v3 model fingerprint lacks per-file hashes: {model_key}")
    if common.canonical_hash(files) != fingerprint.get("content_sha256"):
        raise ValueError(f"Step7-v3 model per-file fingerprint is inconsistent: {model_key}")
    expected = {
        "content_sha256": cfg["expected_content_sha256"],
        "file_count": int(cfg["expected_file_count"]),
        "total_size_bytes": int(cfg["expected_total_size_bytes"]),
    }
    for field, value in expected.items():
        if fingerprint.get(field) != value:
            raise ValueError(
                f"Step7-v3 preregistered model fingerprint mismatch: {model_key}/{field}"
            )


def matrix_for_rows(
    rows: list[dict], features: dict[str, dict], feature_names: list[str]
) -> np.ndarray:
    matrix = np.asarray(
        [[features[row["pair_uid"]][name] for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(feature_names)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Step7-v3 candidate feature matrix is invalid")
    return matrix


def labels_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [1 if row["review_label"] == "positive" else 0 for row in rows], dtype=np.int8
    )


def shortcut_control_stratum(feature_row: dict) -> str:
    def normalized_binary(name: str) -> str:
        try:
            value = float(feature_row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Step7-v3 shortcut-control field is not numeric: {name}"
            ) from exc
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError(
                f"Step7-v3 shortcut-control field must be exactly zero or one: {name}"
            )
        return str(int(value))

    same_market = normalized_binary("same_market_bool")
    same_source = normalized_binary("same_source_dataset_bool")
    return f"same_market={same_market}|same_source={same_source}"


def attach_shortcut_control_strata(
    rows: list[dict], features: dict[str, dict]
) -> None:
    for row in rows:
        feature_row = features.get(row["pair_uid"])
        if feature_row is None:
            raise ValueError(
                f"Step7-v3 shortcut-control feature row is missing: {row['pair_uid']}"
            )
        row[SHORTCUT_CONTROL_STRATUM_FIELD] = shortcut_control_stratum(feature_row)

def evidence_slice_rates(rows: list[dict], scores: np.ndarray, threshold: float, policy: dict) -> dict:
    predicted = np.asarray(scores) >= threshold
    hard_types = set(policy["evaluation"]["hard_negative_evidence_types"])
    direct_types = set(policy["evaluation"]["direct_positive_evidence_types"])
    hard_indices = [
        index
        for index, row in enumerate(rows)
        if row["review_label"] == "negative" and row["evidence_type"] in hard_types
    ]
    direct_indices = [
        index
        for index, row in enumerate(rows)
        if row["review_label"] == "positive" and row["evidence_type"] in direct_types
    ]
    return {
        "hard_negative": {
            "row_count": len(hard_indices),
            "false_positive_rate": float(np.mean(predicted[hard_indices]))
            if hard_indices
            else None,
        },
        "direct_positive": {
            "row_count": len(direct_indices),
            "recall": float(np.mean(predicted[direct_indices])) if direct_indices else None,
        },
    }


def component_equal_average_precision(
    rows: list[dict], labels: np.ndarray, scores: np.ndarray
) -> float:
    return weighted_average_precision(
        labels,
        scores,
        component_weights(rows, "component_equal_normalized_to_row_count"),
    )

def shortcut_conditioned_component_equal_average_precision(
    rows: list[dict],
    labels: np.ndarray,
    scores: np.ndarray,
    policy: dict,
    *,
    require_expected_strata: bool,
) -> dict:
    """Macro-average component-equal AP inside market/source strata.

    Computing AP separately prevents a model from winning merely by assigning
    different baseline scores to strata with different positive prevalence.
    Single-class strata cannot define AP and are retained as explicit audit
    rows rather than silently mixed into the primary metric.
    """
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(y) or s.shape != y.shape:
        raise ValueError("Step7-v3 shortcut-conditioned metric shape mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        stratum = row.get(SHORTCUT_CONTROL_STRATUM_FIELD)
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("Step7-v3 row lacks shortcut-control stratum metadata")
        grouped[stratum].append(index)

    per_stratum = {}
    estimable = []
    for stratum, indices_list in sorted(grouped.items()):
        indices = np.asarray(indices_list, dtype=int)
        selected_labels = y[indices]
        selected_rows = [rows[index] for index in indices]
        label_counts = {
            "positive": int(np.sum(selected_labels == 1)),
            "negative": int(np.sum(selected_labels == 0)),
        }
        record = {
            "row_count": len(indices),
            "component_count": len({row["component_id"] for row in selected_rows}),
            "label_counts": label_counts,
            "estimable": len(np.unique(selected_labels)) == 2,
            "component_equal_average_precision": None,
        }
        if record["estimable"]:
            value = component_equal_average_precision(
                selected_rows, selected_labels, s[indices]
            )
            record["component_equal_average_precision"] = value
            estimable.append((stratum, value))
        per_stratum[stratum] = record

    cfg = policy["evaluation"]["shortcut_conditioned_primary_metric"]
    minimum = int(cfg["minimum_estimable_strata"])
    observed_estimable = [stratum for stratum, _value in estimable]
    if len(estimable) < minimum:
        raise ValueError(
            "Step7-v3 shortcut-conditioned AP has too few two-class strata: "
            f"observed={observed_estimable} minimum={minimum}"
        )
    if require_expected_strata:
        split_names = {row["split_name"] for row in rows}
        if len(split_names) != 1:
            raise ValueError("Step7-v3 conditioned metric rows mix split names")
        split_name = next(iter(split_names))
        expected_key = f"expected_{split_name}_estimable_strata"
        expected = cfg.get(expected_key)
        if expected is not None and observed_estimable != expected:
            raise ValueError(
                "Step7-v3 shortcut-conditioned estimable stratum drift: "
                f"split={split_name} expected={expected} observed={observed_estimable}"
            )
    return {
        "metric": "shortcut_conditioned_macro_component_equal_average_precision",
        "macro_average_precision": float(np.mean([value for _stratum, value in estimable])),
        "estimable_strata": observed_estimable,
        "excluded_single_class_strata": [
            stratum for stratum, record in per_stratum.items() if not record["estimable"]
        ],
        "per_stratum": per_stratum,
    }

def grouped_bootstrap_ap_delta(
    rows: list[dict], top_scores: np.ndarray, runner_scores: np.ndarray, policy: dict
) -> dict:
    """Component bootstrap for shortcut-conditioned component-equal AP.

    Every sampled component draw has total mass one.  If the same component is
    drawn multiple times, its mass is repeated; it is not collapsed back to a
    single original component.  This is the cluster-bootstrap analogue of the
    primary validation metric.
    """
    cfg = policy["evaluation"]["bootstrap"]
    labels = labels_array(rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["component_id"]].append(index)
    components = sorted(grouped)
    rng = np.random.default_rng(int(cfg["seed"]))
    deltas = []
    skipped = 0
    for _ in range(int(cfg["resamples"])):
        sampled_indices = []
        sampled_rows = []
        for draw_index, _component in enumerate(components):
            sampled = components[int(rng.integers(0, len(components)))]
            component_indices = grouped[sampled]
            sampled_indices.extend(component_indices)
            for index in component_indices:
                sampled_rows.append(
                    {
                        **rows[index],
                        # A repeated cluster draw must retain repeated mass.
                        "component_id": f"bootstrap_draw_{draw_index:06d}",
                    }
                )
        indices = np.asarray(sampled_indices, dtype=int)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        # Repeated component draws are represented by repeated rows.  Their
        # component-equal weights therefore repeat the sampled cluster mass.
        try:
            top_metric = shortcut_conditioned_component_equal_average_precision(
                sampled_rows,
                sampled_labels,
                top_scores[indices],
                policy,
                require_expected_strata=False,
            )["macro_average_precision"]
            runner_metric = shortcut_conditioned_component_equal_average_precision(
                sampled_rows,
                sampled_labels,
                runner_scores[indices],
                policy,
                require_expected_strata=False,
            )["macro_average_precision"]
        except ValueError:
            skipped += 1
            continue
        deltas.append(top_metric - runner_metric)
    if len(deltas) < max(100, int(int(cfg["resamples"]) * 0.90)):
        raise ValueError("Step7-v3 grouped bootstrap produced too few two-class resamples")
    values = np.asarray(deltas, dtype=np.float64)
    alpha = 1.0 - float(cfg["confidence"])
    return {
        "resamples_requested": int(cfg["resamples"]),
        "resamples_completed": len(deltas),
        "single_class_resamples_skipped": skipped,
        "component_count": len(components),
        "metric": "shortcut_conditioned_macro_component_equal_average_precision",
        "point_delta_average_precision": float(
            shortcut_conditioned_component_equal_average_precision(
                rows,
                labels,
                top_scores,
                policy,
                require_expected_strata=True,
            )["macro_average_precision"]
            - shortcut_conditioned_component_equal_average_precision(
                rows,
                labels,
                runner_scores,
                policy,
                require_expected_strata=True,
            )["macro_average_precision"]
        ),
        "mean_delta_average_precision": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values, alpha / 2.0)),
        "ci95_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "probability_delta_above_zero": float(np.mean(values > 0.0)),
    }


def without_score_arrays(training_result: dict) -> dict:
    return {
        key: value
        for key, value in training_result.items()
        if key != "train_oof_scores"
    }

def shortcut_feature_label_association_audit(
    rows: list[dict], features: dict[str, dict], policy: dict
) -> dict:
    """Report shortcut/label association without making it selectable."""
    labels = labels_array(rows)
    output = {}
    for feature_name in policy["pair_feature_roles"][
        "shortcut_audit_only_features"
    ]:
        values = np.asarray(
            [float(features[row["pair_uid"]][feature_name]) for row in rows],
            dtype=np.float64,
        )
        by_label = {}
        for label_name, label_value in (("negative", 0), ("positive", 1)):
            selected = values[labels == label_value]
            by_label[label_name] = {
                "row_count": len(selected),
                "mean": float(np.mean(selected)),
                "zero_count": int(np.sum(selected == 0.0)),
                "one_count": int(np.sum(selected == 1.0)),
            }
        auc = roc_auc(labels, values)
        output[feature_name] = {
            "by_label": by_label,
            "roc_auc_positive_direction": auc,
            "direction_free_separation_auc": max(auc, 1.0 - auc),
            "used_for_candidate_training": False,
            "used_for_encoder_selection": False,
            "used_for_m0_selection": False,
        }
    return {
        "status": "report_only_not_used_for_selection",
        "features": output,
    }


def rank_candidate_ids(
    candidate_ids: list[str],
    public_results: dict[str, dict],
    primary_field: str,
    global_component_equal_field: str,
    metrics_field: str,
) -> list[str]:
    if len(candidate_ids) < 2 or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Step7-v3 selection group must contain distinct candidates")
    return sorted(
        candidate_ids,
        key=lambda candidate_id: (
            -float(public_results[candidate_id][primary_field]),
            -float(public_results[candidate_id][global_component_equal_field]),
            -float(public_results[candidate_id][metrics_field]["average_precision"]),
            -float(public_results[candidate_id][metrics_field]["roc_auc"]),
            candidate_id,
        ),
    )

def assess_candidate_group(
    candidate_ids: list[str],
    public_results: dict[str, dict],
    internal_results: dict[str, dict],
    valid_rows: list[dict],
    policy: dict,
    *,
    training_weight_sensitivity_applicable: bool,
) -> dict:
    ranked = rank_candidate_ids(
        candidate_ids,
        public_results,
        "primary_valid_shortcut_conditioned_macro_component_equal_average_precision",
        "primary_valid_component_equal_average_precision",
        "primary_valid_metrics",
    )
    top_id, runner_id = ranked[:2]
    top_public = public_results[top_id]
    runner_public = public_results[runner_id]
    top_internal = internal_results[top_id]
    runner_internal = internal_results[runner_id]
    bootstrap = grouped_bootstrap_ap_delta(
        valid_rows,
        top_internal["primary_valid_scores"],
        runner_internal["primary_valid_scores"],
        policy,
    )
    primary_delta = float(
        top_public[
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision"
        ]
        - runner_public[
            "primary_valid_shortcut_conditioned_macro_component_equal_average_precision"
        ]
    )

    global_component_ranked = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            -float(
                public_results[candidate_id][
                    "primary_valid_component_equal_average_precision"
                ]
            ),
            -float(
                public_results[candidate_id]["primary_valid_metrics"][
                    "average_precision"
                ]
            ),
            candidate_id,
        ),
    )
    global_component_best_other = next(
        candidate_id for candidate_id in global_component_ranked if candidate_id != top_id
    )
    global_component_delta = float(
        top_public["primary_valid_component_equal_average_precision"]
        - public_results[global_component_best_other][
            "primary_valid_component_equal_average_precision"
        ]
    )

    row_ranked = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            -float(
                public_results[candidate_id]["primary_valid_metrics"][
                    "average_precision"
                ]
            ),
            -float(public_results[candidate_id]["primary_valid_metrics"]["roc_auc"]),
            candidate_id,
        ),
    )
    row_best_other = next(candidate_id for candidate_id in row_ranked if candidate_id != top_id)
    row_delta = float(
        top_public["primary_valid_metrics"]["average_precision"]
        - public_results[row_best_other]["primary_valid_metrics"]["average_precision"]
    )

    uniform_ranked = None
    uniform_delta = None
    if training_weight_sensitivity_applicable:
        uniform_ranked = rank_candidate_ids(
            candidate_ids,
            public_results,
            "uniform_weight_sensitivity_valid_shortcut_conditioned_macro_"
            "component_equal_average_precision",
            "uniform_weight_sensitivity_valid_component_equal_average_precision",
            "uniform_weight_sensitivity_valid_metrics",
        )
        uniform_best_other = next(
            candidate_id for candidate_id in uniform_ranked if candidate_id != top_id
        )
        uniform_delta = float(
            top_public[
                "uniform_weight_sensitivity_valid_shortcut_conditioned_macro_"
                "component_equal_average_precision"
            ]
            - public_results[uniform_best_other][
                "uniform_weight_sensitivity_valid_shortcut_conditioned_macro_"
                "component_equal_average_precision"
            ]
        )

    top_hard_fpr = top_public["valid_evidence_slices"]["hard_negative"][
        "false_positive_rate"
    ]
    runner_hard_fpr = runner_public["valid_evidence_slices"]["hard_negative"][
        "false_positive_rate"
    ]
    top_direct_recall = top_public["valid_evidence_slices"]["direct_positive"][
        "recall"
    ]
    runner_direct_recall = runner_public["valid_evidence_slices"]["direct_positive"][
        "recall"
    ]
    if None in (top_hard_fpr, runner_hard_fpr, top_direct_recall, runner_direct_recall):
        raise ValueError("Step7-v3 winner guard evidence slices are not estimable")

    rule = policy["selection_rule"]["unique_winner_requires_all"]
    checks = {
        "shortcut_conditioned_valid_ap_delta": {
            "observed": primary_delta,
            "required_minimum": float(
                rule[
                    "valid_shortcut_conditioned_macro_component_equal_average_"
                    "precision_delta_vs_runner_up_at_least"
                ]
            ),
            "pass": primary_delta
            >= float(
                rule[
                    "valid_shortcut_conditioned_macro_component_equal_average_"
                    "precision_delta_vs_runner_up_at_least"
                ]
            ),
        },
        "shortcut_conditioned_grouped_bootstrap_ci_lower": {
            "observed": bootstrap["ci95_lower"],
            "required_above": float(
                rule["grouped_bootstrap_ap_delta_ci95_lower_above"]
            ),
            "pass": bootstrap["ci95_lower"]
            > float(rule["grouped_bootstrap_ap_delta_ci95_lower_above"]),
        },
        "global_component_equal_ap_delta": {
            "observed": global_component_delta,
            "required_above": float(
                rule["global_component_equal_average_precision_delta_above"]
            ),
            "pass": global_component_delta
            > float(rule["global_component_equal_average_precision_delta_above"]),
        },
        "global_component_equal_ap_top_rank_consistent": {
            "observed_top_candidate": global_component_ranked[0],
            "required_top_candidate": top_id,
            "pass": global_component_ranked[0] == top_id,
        },
        "row_ap_delta": {
            "observed": row_delta,
            "required_above": float(rule["row_average_precision_delta_above"]),
            "pass": row_delta > float(rule["row_average_precision_delta_above"]),
        },
        "row_ap_top_rank_consistent": {
            "observed_top_candidate": row_ranked[0],
            "required_top_candidate": top_id,
            "pass": row_ranked[0] == top_id,
        },
        "hard_negative_fpr_increase": {
            "observed": float(top_hard_fpr - runner_hard_fpr),
            "allowed_maximum": float(rule["hard_negative_fpr_increase_at_most"]),
            "pass": top_hard_fpr - runner_hard_fpr
            <= float(rule["hard_negative_fpr_increase_at_most"]),
        },
        "direct_positive_recall_drop": {
            "observed": float(runner_direct_recall - top_direct_recall),
            "allowed_maximum": float(rule["direct_positive_recall_drop_at_most"]),
            "pass": runner_direct_recall - top_direct_recall
            <= float(rule["direct_positive_recall_drop_at_most"]),
        },
    }
    if training_weight_sensitivity_applicable:
        assert uniform_ranked is not None and uniform_delta is not None
        checks["uniform_training_shortcut_conditioned_ap_delta"] = {
            "observed": uniform_delta,
            "required_above": float(
                rule[
                    "uniform_weight_sensitivity_shortcut_conditioned_ap_delta_above"
                ]
            ),
            "pass": uniform_delta
            > float(
                rule[
                    "uniform_weight_sensitivity_shortcut_conditioned_ap_delta_above"
                ]
            ),
        }
        checks["uniform_training_top_rank_consistent"] = {
            "observed_top_candidate": uniform_ranked[0],
            "required_top_candidate": top_id,
            "pass": uniform_ranked[0] == top_id,
        }
    return {
        "candidate_ids": list(candidate_ids),
        "candidate_ranking": ranked,
        "global_component_equal_ap_sensitivity_ranking": global_component_ranked,
        "row_ap_sensitivity_ranking": row_ranked,
        "uniform_training_shortcut_conditioned_ap_ranking": uniform_ranked,
        "training_weight_sensitivity_applicable": (
            training_weight_sensitivity_applicable
        ),
        "top_candidate": top_id,
        "runner_up_candidate": runner_id,
        "top_vs_runner_up_shortcut_conditioned_ap_delta": primary_delta,
        "top_vs_runner_up_grouped_bootstrap": bootstrap,
        "unique_winner_checks": checks,
        "unique_winner": all(item["pass"] for item in checks.values()),
    }
