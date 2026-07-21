#!/usr/bin/env python3
"""Select Step7-v2 English M0 on validation, then run a delayed historical test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_v2_common as common


EPS = 1e-12


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
        raise ValueError(f"Unsupported Step7-v2 weighting mode: {mode}")
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
) -> dict:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(w) != len(y):
        raise ValueError("Step7-v2 logistic input shape mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("Step7-v2 logistic input contains non-finite values")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Step7-v2 logistic training requires both binary classes")
    if np.any(w <= 0.0):
        raise ValueError("Step7-v2 logistic weights must be positive")
    weight_total = float(np.sum(w))
    mean = np.sum(x * w[:, None], axis=0) / weight_total
    variance = np.sum(((x - mean) ** 2) * w[:, None], axis=0) / weight_total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    params = np.zeros(z.shape[1] + 1, dtype=np.float64)
    converged = False
    final_delta = math.inf
    for iteration in range(1, int(max_iter) + 1):
        probabilities = safe_sigmoid(params[0] + z @ params[1:])
        residual = (probabilities - y) * w
        gradient = np.empty_like(params)
        gradient[0] = np.sum(residual)
        gradient[1:] = z.T @ residual + float(l2_penalty) * params[1:]
        curvature = probabilities * (1.0 - probabilities) * w
        weighted_z = z * curvature[:, None]
        hessian = np.empty((len(params), len(params)), dtype=np.float64)
        hessian[0, 0] = np.sum(curvature)
        hessian[0, 1:] = np.sum(weighted_z, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = z.T @ weighted_z
        hessian[1:, 1:] += np.eye(z.shape[1]) * float(l2_penalty)
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        delta = np.clip(delta, -5.0, 5.0)
        params -= delta
        final_delta = float(np.linalg.norm(delta))
        if final_delta <= float(tolerance):
            converged = True
            break
    if not converged:
        raise ValueError(
            f"Step7-v2 logistic solver did not converge: l2={l2_penalty} delta={final_delta}"
        )
    return {
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "intercept": float(params[0]),
        "coefficients": [float(value) for value in params[1:]],
        "l2_penalty": float(l2_penalty),
        "solver_iterations": iteration,
        "solver_final_delta_norm": final_delta,
        "solver_converged": converged,
        "sample_weight_total": weight_total,
    }


def apply_logistic(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if x.shape[1] != len(coefficients):
        raise ValueError("Step7-v2 logistic artifact feature dimension mismatch")
    return safe_sigmoid(
        float(artifact["intercept"]) + ((x - mean) / scale) @ coefficients
    )


def balanced_component_folds(rows: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["component_id"]].append(row)
    if len(grouped) < fold_count:
        raise ValueError("Step7-v2 has fewer training components than folds")
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
        raise ValueError("Step7-v2 has too few positive components for grouped folds")
    if sum(record[3] > 0 for record in records) < fold_count:
        raise ValueError("Step7-v2 has too few negative components for grouped folds")

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
            raise ValueError(f"Step7-v2 grouped fold {fold} is single-class")
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
            raise ValueError("Step7-v2 OOF predictions are incomplete")
        weighted_ap = weighted_average_precision(labels, oof, oof_selection_weights)
        row_ap = average_precision(labels, oof)
        auc = roc_auc(labels, oof)
        result = {
            "l2_penalty": l2_penalty,
            "oof_selection_weighted_average_precision": weighted_ap,
            "oof_row_average_precision": row_ap,
            "oof_roc_auc": auc,
            "folds": fold_artifacts,
            "oof_scores": oof,
        }
        grid_results.append(result)
        # AP alone selects regularization; exact AP ties prefer the stronger L2.
        # AUC is reported but cannot silently change the preregistered choice.
        key = (weighted_ap, l2_penalty)
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
        "selected_threshold": threshold,
        "threshold_selection": threshold_summary,
        "final_train_artifact": final_artifact,
        "component_fold_assignment_sha256": common.canonical_hash(folds),
        "component_fold_diagnostics": fold_diagnostics,
    }


def candidate_specs(policy: dict) -> list[dict]:
    safe_features = list(policy["safe_pair_features"])
    reranker_feature = policy["shared_reranker"]["feature_name"]
    output = []
    for model_key, model_cfg in policy["embedding_models"].items():
        embedding_feature = model_cfg["feature_name"]
        for tier_name, template in policy["candidate_tiers"].items():
            feature_names = []
            for entry in template:
                if entry == "{embedding_feature}":
                    feature_names.append(embedding_feature)
                elif entry == "{safe_pair_features}":
                    feature_names.extend(safe_features)
                elif entry == "{shared_reranker_feature}":
                    feature_names.append(reranker_feature)
                else:
                    raise ValueError(f"Unknown Step7-v2 candidate feature placeholder: {entry}")
            if len(feature_names) != len(set(feature_names)):
                raise ValueError(f"Step7-v2 candidate has duplicate features: {model_key}/{tier_name}")
            output.append(
                {
                    "candidate_id": f"{model_key}__{tier_name}",
                    "model_key": model_key,
                    "tier": tier_name,
                    "feature_names": feature_names,
                }
            )
    if len(output) != 15:
        raise ValueError(f"Step7-v2 expected 15 candidates, observed {len(output)}")
    return output


def index_unique(rows: list[dict], name: str) -> dict[str, dict]:
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Step7-v2 duplicate pair_uid in {name}")
    return index


def verify_file_record(record: dict, role: str) -> Path:
    path = common.resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v2 {role} file is missing: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v2 {role} file size drift: {record['path']}")
    if common.sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v2 {role} file hash drift: {record['path']}")
    return path


def verify_policy_input_records(
    policy: dict, manifest: dict, input_names: tuple[str, ...], role: str
) -> None:
    records = manifest.get("input_manifest", {})
    if set(records) != set(input_names):
        raise ValueError(f"Step7-v2 {role} input-manifest universe mismatch")
    for input_name in input_names:
        record = records[input_name]
        spec = policy["inputs"][input_name]
        if record.get("path") != spec["path"] or record.get("sha256") != spec["sha256"]:
            raise ValueError(f"Step7-v2 {role} input pin mismatch: {input_name}")
        if int(record.get("size_bytes", 0)) <= 0:
            raise ValueError(f"Step7-v2 {role} input size is invalid: {input_name}")


def expected_gpu_output_paths(policy: dict) -> set[str]:
    outputs = policy["outputs"]
    def normalized(path_value: str) -> str:
        return str(common.resolve(path_value).relative_to(common.ROOT)).replace("\\", "/")

    paths = set()
    for model_key in policy["embedding_models"]:
        paths.update(
            {
                normalized(outputs["embedding_matrix_template"].format(model_key=model_key)),
                normalized(outputs["embedding_manifest_template"].format(model_key=model_key)),
                normalized(outputs["embedding_pair_scores_template"].format(model_key=model_key)),
            }
        )
    paths.update(
        {
            normalized(outputs["reranker_pair_scores"]),
            normalized(outputs["reranker_manifest"]),
        }
    )
    return paths


def verify_model_fingerprint(model_key: str, fingerprint: dict, cfg: dict) -> None:
    files = fingerprint.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Step7-v2 model fingerprint lacks per-file hashes: {model_key}")
    if common.canonical_hash(files) != fingerprint.get("content_sha256"):
        raise ValueError(f"Step7-v2 model per-file fingerprint is inconsistent: {model_key}")
    expected = {
        "content_sha256": cfg["expected_content_sha256"],
        "file_count": int(cfg["expected_file_count"]),
        "total_size_bytes": int(cfg["expected_total_size_bytes"]),
    }
    for field, value in expected.items():
        if fingerprint.get(field) != value:
            raise ValueError(
                f"Step7-v2 preregistered model fingerprint mismatch: {model_key}/{field}"
            )


def verify_runtime_provenance(policy: dict) -> dict:
    outputs = policy["outputs"]
    public_path = common.resolve(outputs["preparation_manifest"])
    development_path = common.resolve(outputs["development_labels_manifest"])
    for path in (public_path, development_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step7-v2 required Windows manifest is missing: {path}")
    public = common.load_json(public_path)
    development = common.load_json(development_path)
    if public.get("step") != "step7_v2_prepare_public_label_free_data":
        raise ValueError("Step7-v2 public preparation role mismatch")
    if public.get("version") != policy["version"]:
        raise ValueError("Step7-v2 public preparation version mismatch")
    public_policy_path = common.resolve(public.get("policy_path", ""))
    if not public_policy_path.is_file() or public.get("policy_sha256") != common.sha256_file(
        public_policy_path
    ):
        raise ValueError("Step7-v2 public preparation policy hash drift")
    if public.get("feature_generation_uses_review_label_values") is not False:
        raise ValueError("Step7-v2 public features are not label isolated")
    residue_scan = public.get("identity_residue_scan", {})
    if residue_scan.get("status") != "pass" or residue_scan.get(
        "total_residue_count"
    ) != 0:
        raise ValueError("Step7-v2 public corpus identity-residue audit did not pass")
    if residue_scan.get("claim_scope") != policy["clean_text_contract"].get(
        "identity_residue_claim_scope"
    ) or residue_scan.get("unknown_identifier_absence_proven") is not False:
        raise ValueError("Step7-v2 identity-residue claim boundary is overstated")
    common.validate_content_fidelity_manifest(policy, public)
    common.validate_global_identity_audit_manifest(policy, public)
    preparation_script = common.resolve("scripts/step7_v2_prepare_clean_data.py")
    common_script = common.resolve("scripts/step7_v2_common.py")
    step3_script = common.resolve("scripts/step3_build_seller_profiles.py")
    for manifest, role in ((public, "public"), (development, "development-label")):
        if manifest.get("generator_script_sha256") != common.sha256_file(
            preparation_script
        ):
            raise ValueError(f"Step7-v2 {role} preparation script drift")
        if manifest.get("common_script_sha256") != common.sha256_file(common_script):
            raise ValueError(f"Step7-v2 {role} common script drift")
        if manifest.get("redaction_dependency_script_sha256") != common.sha256_file(
            step3_script
        ):
            raise ValueError(f"Step7-v2 {role} Step3 redaction dependency drift")
    if development.get("splits_written") != ["train", "valid"]:
        raise ValueError("Step7-v2 development labels must contain train and valid only")
    if development.get("other_split_label_values_used_during_materialization") is not False:
        raise ValueError("Step7-v2 development materialization used another split's labels")
    if development.get("split_projection_applied_before_label_or_evidence_access") is not True:
        raise ValueError("Step7-v2 development label projection contract is missing")
    if development.get("version") != policy["version"]:
        raise ValueError("Step7-v2 development-label version mismatch")
    if development.get("policy_sha256") != public.get("policy_sha256"):
        raise ValueError("Step7-v2 development-label policy hash drift")
    if development.get("public_preparation_manifest_sha256") != common.sha256_file(public_path):
        raise ValueError("Step7-v2 development/public preparation provenance drift")
    if set(public.get("output_files", {})) != {
        "pair_manifest",
        "clean_corpus",
        "train_feature_reference",
        "safe_pair_features",
    }:
        raise ValueError("Step7-v2 public preparation output universe mismatch")
    if set(development.get("output_files", {})) != {
        "private_labels_train",
        "private_labels_valid",
    }:
        raise ValueError("Step7-v2 development-label output universe mismatch")
    for record in public["output_files"].values():
        verify_file_record(record, "public-preparation")
    for record in development["output_files"].values():
        verify_file_record(record, "development-label")
    verify_policy_input_records(
        policy,
        public,
        ("seller_profiles", "item_identity_signals", "component_assignments"),
        "public-preparation",
    )
    verify_policy_input_records(
        policy,
        development,
        ("frozen_labels", "evidence_labels"),
        "development-label",
    )

    sync_path = common.resolve(outputs["gpu_sync_manifest"])
    bundle_path = common.resolve(outputs["gpu_output_manifest"])
    if not sync_path.is_file() or not bundle_path.is_file():
        raise FileNotFoundError("Step7-v2 GPU provenance bundle is incomplete")
    sync = common.load_json(sync_path)
    bundle = common.load_json(bundle_path)
    if sync.get("step") != "step7_v2_label_free_windows_to_linux_gpu_sync":
        raise ValueError("Step7-v2 GPU sync role mismatch")
    if sync.get("label_files_included") is not False or sync.get(
        "raw_source_files_included"
    ) is not False:
        raise ValueError("Step7-v2 GPU sync was not label/raw-source isolated")
    if sync.get("policy_contract_sha256") != common.canonical_hash(policy):
        raise ValueError("Step7-v2 GPU sync policy contract mismatch")
    if sync.get("policy_sha256") != public.get("policy_sha256"):
        raise ValueError("Step7-v2 GPU sync policy file hash mismatch")
    if sync.get("public_preparation_manifest_sha256") != common.sha256_file(public_path):
        raise ValueError("Step7-v2 GPU sync/public preparation drift")
    sync_builder_path = common.resolve("scripts/step7_v2_build_sync_manifest.py")
    if sync.get("generator_script_path") != "scripts/step7_v2_build_sync_manifest.py":
        raise ValueError("Step7-v2 GPU sync builder path drift")
    if sync.get("generator_script_sha256") != common.sha256_file(sync_builder_path):
        raise ValueError("Step7-v2 GPU sync builder script drift")
    for record in sync.get("files", []):
        verify_file_record(record, "GPU-sync")
    required_sync_paths = {
        str(public_policy_path.relative_to(common.ROOT)).replace("\\", "/"),
        "scripts/step3_build_seller_profiles.py",
        "scripts/step7_v2_common.py",
        "scripts/step7_v2_build_sync_manifest.py",
        "scripts/run_step7_v2_clean_source_linux_20260721.sh",
        outputs["pair_manifest"],
        outputs["clean_corpus"],
        outputs["preparation_manifest"],
        "scripts/step7_v2_encode_clean_models.py",
    }
    observed_sync_paths = {record["path"] for record in sync.get("files", [])}
    if observed_sync_paths != required_sync_paths or len(sync.get("files", [])) != len(
        required_sync_paths
    ):
        raise ValueError("Step7-v2 GPU sync file universe mismatch")
    if sync.get("file_count") != len(required_sync_paths) or sync.get(
        "total_file_bytes"
    ) != sum(int(record["size_bytes"]) for record in sync["files"]):
        raise ValueError("Step7-v2 GPU sync file totals mismatch")
    forbidden = set(sync.get("forbidden_workspace_paths", []))
    if forbidden & observed_sync_paths:
        raise ValueError("Step7-v2 GPU sync includes a forbidden source/label path")
    for model_key, cfg in {
        **policy["embedding_models"],
        policy["shared_reranker"]["model_key"]: policy["shared_reranker"],
    }.items():
        registered = sync.get("model_directories", {}).get(model_key)
        if registered is None or registered.get("path") != cfg["local_path"]:
            raise ValueError(f"Step7-v2 GPU sync model registration missing: {model_key}")
        verify_model_fingerprint(model_key, registered, cfg)

    if bundle.get("step") != "step7_v2_label_free_gpu_output_bundle":
        raise ValueError("Step7-v2 GPU output bundle role mismatch")
    if bundle.get("gpu_sync_manifest_sha256") != common.sha256_file(sync_path):
        raise ValueError("Step7-v2 GPU output/sync provenance drift")
    if bundle.get("policy_contract_sha256") != common.canonical_hash(policy):
        raise ValueError("Step7-v2 GPU output policy contract mismatch")
    if bundle.get("public_preparation_manifest_sha256") != common.sha256_file(public_path):
        raise ValueError("Step7-v2 GPU output/public preparation drift")
    encoder_path = common.resolve("scripts/step7_v2_encode_clean_models.py")
    if bundle.get("generator_script_sha256") != common.sha256_file(encoder_path):
        raise ValueError("Step7-v2 GPU output generator script drift")
    if bundle.get("label_or_raw_source_files_present_in_gpu_workspace") is not False:
        raise ValueError("Step7-v2 GPU output reports label/raw-source access")
    bundle_records = bundle.get("files", [])
    expected_bundle_paths = expected_gpu_output_paths(policy)
    if {record["path"] for record in bundle_records} != expected_bundle_paths or len(
        bundle_records
    ) != len(expected_bundle_paths):
        raise ValueError("Step7-v2 GPU output bundle file universe mismatch")
    if bundle.get("file_count") != len(expected_bundle_paths) or bundle.get(
        "total_file_bytes"
    ) != sum(int(record["size_bytes"]) for record in bundle_records):
        raise ValueError("Step7-v2 GPU output bundle totals mismatch")
    for record in bundle_records:
        verify_file_record(record, "GPU-output")
    return {
        "public_preparation": public,
        "development_labels": development,
        "gpu_sync": sync,
        "gpu_output": bundle,
        "gpu_sync_sha256": common.sha256_file(sync_path),
        "public_preparation_sha256": common.sha256_file(public_path),
        "encoder_script_sha256": common.sha256_file(encoder_path),
    }


def require_manifest_fields(manifest: dict, expected: dict, role: str) -> None:
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Step7-v2 {role} manifest mismatch: {field}")


def validate_token_length_diagnostics(
    manifest: dict, expected_rows: int, expected_max_length: int, role: str
) -> None:
    diagnostics = manifest.get("token_length_diagnostics", {})
    if diagnostics.get("row_count") != expected_rows or diagnostics.get(
        "max_length_contract"
    ) != expected_max_length:
        raise ValueError(f"Step7-v2 {role} token-length audit boundary mismatch")
    truncated = diagnostics.get("truncated_row_count")
    fraction = diagnostics.get("truncated_row_fraction")
    if (
        not isinstance(truncated, int)
        or not 0 <= truncated <= expected_rows
        or not isinstance(fraction, (int, float))
        or not 0.0 <= float(fraction) <= 1.0
        or not math.isclose(
            float(fraction), truncated / expected_rows, abs_tol=1e-12
        )
    ):
        raise ValueError(f"Step7-v2 {role} token-length audit values are invalid")
    statistic_fields = (
        "token_length_min",
        "token_length_median",
        "token_length_p90",
        "token_length_p95",
        "token_length_max",
    )
    for field in statistic_fields:
        value = diagnostics.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Step7-v2 {role} token-length statistic is invalid: {field}")
    statistics = [float(diagnostics[field]) for field in statistic_fields]
    if statistics != sorted(statistics):
        raise ValueError(f"Step7-v2 {role} token-length statistics are not monotonic")
    for field in ("torch_version", "transformers_version"):
        if not str(manifest.get(field, "")).strip():
            raise ValueError(f"Step7-v2 {role} runtime version is missing: {field}")


def replay_embedding_pair_scores(
    policy: dict,
    model_key: str,
    matrix_path: Path,
    manifest: dict,
    pair_rows: list[dict],
    score_index: dict[str, dict],
    feature_name: str,
) -> dict:
    """Prove that every serialized cosine is derived from the pinned matrix."""
    try:
        matrix = np.load(matrix_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Step7-v2 cannot safely load embedding matrix for {model_key}"
        ) from exc
    expected_shape = tuple(manifest["shape"])
    if matrix.dtype != np.float32 or matrix.shape != expected_shape:
        raise ValueError(
            f"Step7-v2 embedding matrix dtype/shape drift for {model_key}: "
            f"dtype={matrix.dtype} shape={matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Step7-v2 embedding matrix is non-finite for {model_key}")
    norms = np.linalg.norm(matrix, axis=1)
    observed_norm_error = float(np.max(np.abs(norms - 1.0)))
    if not math.isfinite(observed_norm_error) or observed_norm_error > 1e-3:
        raise ValueError(f"Step7-v2 stored embedding matrix is not normalized: {model_key}")
    seller_uids = manifest["seller_uids"]
    seller_index = {seller_uid: index for index, seller_uid in enumerate(seller_uids)}
    if len(seller_index) != len(seller_uids):
        raise ValueError(f"Step7-v2 embedding seller UID list is not unique: {model_key}")
    tolerance = float(
        policy["evaluation"]["embedding_score_replay_absolute_tolerance"]
    )
    maximum_difference = 0.0
    for pair in pair_rows:
        try:
            left = matrix[seller_index[pair["seller_uid_left"]]]
            right = matrix[seller_index[pair["seller_uid_right"]]]
        except KeyError as exc:
            raise ValueError(
                f"Step7-v2 embedding matrix lacks pair endpoint for {model_key}"
            ) from exc
        replayed = float(np.dot(left, right))
        serialized = float(score_index[pair["pair_uid"]][feature_name])
        difference = abs(replayed - serialized)
        maximum_difference = max(maximum_difference, difference)
        if not math.isfinite(replayed) or difference > tolerance:
            raise ValueError(
                f"Step7-v2 embedding score does not replay from matrix for "
                f"{model_key}:{pair['pair_uid']}; difference={difference:.9g} "
                f"tolerance={tolerance:.9g}"
            )
    return {
        "status": "pass",
        "matrix_dtype": str(matrix.dtype),
        "matrix_shape": list(matrix.shape),
        "stored_matrix_maximum_unit_norm_error": observed_norm_error,
        "pair_count_replayed": len(pair_rows),
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": float(maximum_difference),
    }


def load_feature_bundle(policy: dict) -> tuple[list[dict], dict[str, dict], dict]:
    outputs = policy["outputs"]
    provenance = verify_runtime_provenance(policy)
    pair_path = common.resolve(outputs["pair_manifest"])
    corpus_path = common.resolve(outputs["clean_corpus"])
    safe_path = common.resolve(outputs["safe_pair_features"])
    pair_rows = common.load_csv(pair_path)
    corpus_rows = common.load_jsonl(corpus_path)
    common.validate_public_pair_rows(policy, pair_rows)
    common.validate_clean_corpus_rows(corpus_rows)
    seller_uids = [row["seller_uid"] for row in corpus_rows]
    expected_pair_uids = [row["pair_uid"] for row in pair_rows]
    if len(expected_pair_uids) != len(set(expected_pair_uids)):
        raise ValueError("Step7-v2 pair manifest has duplicate pair_uid values")
    safe_rows = common.load_csv(safe_path)
    common.validate_safe_pair_feature_rows(safe_rows)
    safe_index = index_unique(safe_rows, "safe pair features")
    if set(safe_index) != set(expected_pair_uids):
        raise ValueError("Step7-v2 safe feature pair universe mismatch")
    features: dict[str, dict] = {
        pair_uid: {name: float(safe_index[pair_uid][name]) for name in common.SAFE_FEATURE_NAMES}
        for pair_uid in expected_pair_uids
    }
    runtime_manifests = {}
    for model_key, cfg in policy["embedding_models"].items():
        score_path = common.resolve(
            outputs["embedding_pair_scores_template"].format(model_key=model_key)
        )
        manifest_path = common.resolve(
            outputs["embedding_manifest_template"].format(model_key=model_key)
        )
        if not score_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Step7-v2 GPU score missing for {model_key}; run the Linux encoding stage"
            )
        manifest = common.load_json(manifest_path)
        require_manifest_fields(
            manifest,
            {
                "step": "step7_v2_encode_clean_embedding",
                "version": policy["version"],
                "model_key": model_key,
                "repo_id": cfg["repo_id"],
                "local_path": cfg["local_path"],
                "feature_name": cfg["feature_name"],
                "pooling_contract": cfg["pooling_contract"],
                "text_prefix": cfg["text_prefix"],
                "max_length": int(cfg["max_length"]),
                "pair_count": len(pair_rows),
                "seller_uids": seller_uids,
                "pair_manifest_sha256": common.sha256_file(pair_path),
                "clean_corpus_sha256": common.sha256_file(corpus_path),
                "gpu_sync_manifest_sha256": provenance["gpu_sync_sha256"],
                "policy_contract_sha256": common.canonical_hash(policy),
                "public_preparation_manifest_sha256": provenance[
                    "public_preparation_sha256"
                ],
                "generator_script_sha256": provenance["encoder_script_sha256"],
                "feature_generation_reads_label_values": False,
                "label_or_raw_source_files_present_in_gpu_workspace": False,
                "device": "cuda",
            },
            f"embedding/{model_key}",
        )
        shape = manifest.get("shape", [])
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != len(seller_uids)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError(f"Step7-v2 embedding seller matrix shape drift: {model_key}")
        norm_error = manifest.get("maximum_unit_norm_error")
        if (
            not isinstance(norm_error, (int, float))
            or not math.isfinite(float(norm_error))
            or not 0.0 <= float(norm_error) <= 1e-3
        ):
            raise ValueError(f"Step7-v2 embedding norm audit failed: {model_key}")
        if not str(manifest.get("sentence_transformers_version", "")).strip():
            raise ValueError(
                f"Step7-v2 embedding runtime version is missing: {model_key}"
            )
        validate_token_length_diagnostics(
            manifest, len(seller_uids), int(cfg["max_length"]), f"embedding/{model_key}"
        )
        layout = manifest.get("layout_validation", {})
        if layout.get("pooling") != cfg["expected_pooling"] or layout.get(
            "has_dense_module"
        ) is not bool(cfg["expected_dense_module"]):
            raise ValueError(f"Step7-v2 model-native pooling/dense drift: {model_key}")
        verify_model_fingerprint(model_key, manifest.get("model_fingerprint", {}), cfg)
        registered = dict(provenance["gpu_sync"]["model_directories"][model_key])
        registered.pop("path", None)
        if manifest["model_fingerprint"] != registered:
            raise ValueError(f"Step7-v2 embedding/sync model fingerprint drift: {model_key}")
        if manifest.get("pair_scores_sha256") != common.sha256_file(score_path):
            raise ValueError(f"Step7-v2 embedding score hash drift for {model_key}")
        matrix_path = common.resolve(
            outputs["embedding_matrix_template"].format(model_key=model_key)
        )
        if manifest.get("embedding_matrix_sha256") != common.sha256_file(matrix_path):
            raise ValueError(f"Step7-v2 embedding matrix hash drift for {model_key}")
        score_rows = common.load_csv(score_path)
        if not score_rows or list(score_rows[0]) != ["pair_uid", cfg["feature_name"]]:
            raise ValueError(f"Step7-v2 embedding score schema drift for {model_key}")
        score_index = index_unique(score_rows, f"embedding scores {model_key}")
        if set(score_index) != set(expected_pair_uids):
            raise ValueError(f"Step7-v2 embedding score pair universe mismatch for {model_key}")
        feature_name = cfg["feature_name"]
        numeric_replay = replay_embedding_pair_scores(
            policy,
            model_key,
            matrix_path,
            manifest,
            pair_rows,
            score_index,
            feature_name,
        )
        for pair_uid in expected_pair_uids:
            value = float(score_index[pair_uid][feature_name])
            if not math.isfinite(value) or value < -1.000001 or value > 1.000001:
                raise ValueError(f"Step7-v2 non-finite score for {model_key}:{pair_uid}")
            features[pair_uid][feature_name] = value
        runtime_manifests[model_key] = {
            **manifest,
            "selection_loader_numeric_replay": numeric_replay,
        }

    reranker_path = common.resolve(outputs["reranker_pair_scores"])
    reranker_manifest_path = common.resolve(outputs["reranker_manifest"])
    if not reranker_path.is_file() or not reranker_manifest_path.is_file():
        raise FileNotFoundError("Step7-v2 shared reranker score is missing")
    reranker_manifest = common.load_json(reranker_manifest_path)
    reranker_cfg = policy["shared_reranker"]
    require_manifest_fields(
        reranker_manifest,
        {
            "step": "step7_v2_encode_clean_reranker",
            "version": policy["version"],
            "model_key": reranker_cfg["model_key"],
            "repo_id": reranker_cfg["repo_id"],
            "local_path": reranker_cfg["local_path"],
            "feature_name": reranker_cfg["feature_name"],
            "pair_symmetrization": reranker_cfg["pair_symmetrization"],
            "single_logit_transform": reranker_cfg["single_logit_transform"],
            "max_length": int(reranker_cfg["max_length"]),
            "pair_count": len(pair_rows),
            "pair_manifest_sha256": common.sha256_file(pair_path),
            "clean_corpus_sha256": common.sha256_file(corpus_path),
            "gpu_sync_manifest_sha256": provenance["gpu_sync_sha256"],
            "policy_contract_sha256": common.canonical_hash(policy),
            "public_preparation_manifest_sha256": provenance[
                "public_preparation_sha256"
            ],
            "generator_script_sha256": provenance["encoder_script_sha256"],
            "feature_generation_reads_label_values": False,
            "label_or_raw_source_files_present_in_gpu_workspace": False,
            "device": "cuda",
        },
        "reranker",
    )
    verify_model_fingerprint(
        reranker_cfg["model_key"], reranker_manifest.get("model_fingerprint", {}), reranker_cfg
    )
    expected_reranker_layout = common.validate_reranker_layout(
        reranker_cfg["model_key"], reranker_cfg
    )
    if reranker_manifest.get("layout_validation") != expected_reranker_layout:
        raise ValueError("Step7-v2 reranker architecture/logit contract drift")
    validate_token_length_diagnostics(
        reranker_manifest,
        len(pair_rows),
        int(reranker_cfg["max_length"]),
        "reranker",
    )
    direction_gap = reranker_manifest.get("forward_reverse_mean_absolute_gap")
    if (
        not isinstance(direction_gap, (int, float))
        or not math.isfinite(float(direction_gap))
        or not 0.0 <= float(direction_gap) <= 1.0
    ):
        raise ValueError("Step7-v2 reranker direction-gap audit is invalid")
    registered_reranker = dict(
        provenance["gpu_sync"]["model_directories"][reranker_cfg["model_key"]]
    )
    registered_reranker.pop("path", None)
    if reranker_manifest["model_fingerprint"] != registered_reranker:
        raise ValueError("Step7-v2 reranker/sync model fingerprint drift")
    if reranker_manifest.get("pair_scores_sha256") != common.sha256_file(reranker_path):
        raise ValueError("Step7-v2 reranker score hash drift")
    reranker_rows = common.load_csv(reranker_path)
    if not reranker_rows or list(reranker_rows[0]) != [
        "pair_uid",
        reranker_cfg["feature_name"],
    ]:
        raise ValueError("Step7-v2 reranker score schema drift")
    reranker_index = index_unique(reranker_rows, "reranker scores")
    if set(reranker_index) != set(expected_pair_uids):
        raise ValueError("Step7-v2 reranker score pair universe mismatch")
    reranker_feature = reranker_cfg["feature_name"]
    for pair_uid in expected_pair_uids:
        value = float(reranker_index[pair_uid][reranker_feature])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Step7-v2 non-finite reranker score for {pair_uid}")
        features[pair_uid][reranker_feature] = value
    runtime_manifests["shared_reranker"] = reranker_manifest
    return pair_rows, features, runtime_manifests


def load_split_rows(policy: dict, split: str, pair_rows: list[dict]) -> list[dict]:
    output_key = {
        "train": "train_labels",
        "valid": "valid_labels",
        "test": "historical_test_labels",
    }[split]
    label_path = common.resolve(policy["outputs"][output_key])
    labels = common.load_csv(label_path)
    pair_index = {row["pair_uid"]: row for row in pair_rows}
    joined = []
    for label in labels:
        pair = pair_index.get(label["pair_uid"])
        if pair is None or pair["split_name"] != split:
            raise ValueError(f"Step7-v2 private label/pair mismatch for {split}:{label['pair_uid']}")
        if label["component_id"] != pair["component_id"]:
            raise ValueError(f"Step7-v2 private label/component mismatch for {label['pair_uid']}")
        joined.append({**pair, **label})
    expected_total = int(policy["supervision_boundary"]["expected_counts"][split]["total"])
    if len(joined) != expected_total or len({row["pair_uid"] for row in joined}) != len(joined):
        raise ValueError(f"Step7-v2 private {split} label boundary drift")
    expected = policy["supervision_boundary"]["expected_counts"][split]
    observed = Counter(row["review_label"] for row in joined)
    if set(observed) != {"positive", "negative"} or observed != Counter(
        {"positive": int(expected["positive"]), "negative": int(expected["negative"])}
    ):
        raise ValueError(f"Step7-v2 private {split} label-count drift")
    expected_pair_uids = {
        row["pair_uid"] for row in pair_rows if row["split_name"] == split
    }
    if {row["pair_uid"] for row in joined} != expected_pair_uids:
        raise ValueError(f"Step7-v2 private {split} label universe mismatch")
    return joined


def matrix_for_rows(
    rows: list[dict], features: dict[str, dict], feature_names: list[str]
) -> np.ndarray:
    matrix = np.asarray(
        [[features[row["pair_uid"]][name] for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(feature_names)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Step7-v2 candidate feature matrix is invalid")
    return matrix


def labels_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [1 if row["review_label"] == "positive" else 0 for row in rows], dtype=np.int8
    )


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


def grouped_bootstrap_ap_delta(
    rows: list[dict], top_scores: np.ndarray, runner_scores: np.ndarray, policy: dict
) -> dict:
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
        for _component in components:
            sampled = components[int(rng.integers(0, len(components)))]
            sampled_indices.extend(grouped[sampled])
        indices = np.asarray(sampled_indices, dtype=int)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            skipped += 1
            continue
        deltas.append(
            average_precision(sampled_labels, top_scores[indices])
            - average_precision(sampled_labels, runner_scores[indices])
        )
    if len(deltas) < max(100, int(int(cfg["resamples"]) * 0.90)):
        raise ValueError("Step7-v2 grouped bootstrap produced too few two-class resamples")
    values = np.asarray(deltas, dtype=np.float64)
    alpha = 1.0 - float(cfg["confidence"])
    return {
        "resamples_requested": int(cfg["resamples"]),
        "resamples_completed": len(deltas),
        "single_class_resamples_skipped": skipped,
        "component_count": len(components),
        "point_delta_average_precision": float(
            average_precision(labels, top_scores) - average_precision(labels, runner_scores)
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


def runtime_input_fingerprints(policy: dict) -> dict[str, dict]:
    outputs = policy["outputs"]
    paths = {
        "policy": common.DEFAULT_POLICY,
        "selection_script": Path(__file__).resolve(),
        "common_script": Path(common.__file__).resolve(),
        "preparation_manifest": common.resolve(outputs["preparation_manifest"]),
        "pair_manifest": common.resolve(outputs["pair_manifest"]),
        "clean_corpus": common.resolve(outputs["clean_corpus"]),
        "safe_pair_features": common.resolve(outputs["safe_pair_features"]),
        "train_labels": common.resolve(outputs["train_labels"]),
        "valid_labels": common.resolve(outputs["valid_labels"]),
        "development_labels_manifest": common.resolve(outputs["development_labels_manifest"]),
        "gpu_sync_manifest": common.resolve(outputs["gpu_sync_manifest"]),
        "gpu_output_manifest": common.resolve(outputs["gpu_output_manifest"]),
        "reranker_scores": common.resolve(outputs["reranker_pair_scores"]),
        "reranker_manifest": common.resolve(outputs["reranker_manifest"]),
    }
    for model_key in policy["embedding_models"]:
        paths[f"embedding_scores:{model_key}"] = common.resolve(
            outputs["embedding_pair_scores_template"].format(model_key=model_key)
        )
        paths[f"embedding_manifest:{model_key}"] = common.resolve(
            outputs["embedding_manifest_template"].format(model_key=model_key)
        )
    records = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Step7-v2 runtime input missing: {key}={path}")
        records[key] = {
            "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": common.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def run_selection(policy: dict) -> tuple[dict, list[dict], list[dict], dict]:
    pair_rows, features, runtime_manifests = load_feature_bundle(policy)
    train_rows = load_split_rows(policy, "train", pair_rows)
    valid_rows = load_split_rows(policy, "valid", pair_rows)
    train_labels = labels_array(train_rows)
    valid_labels = labels_array(valid_rows)
    valid_component_equal_weights = component_weights(
        valid_rows, "component_equal_normalized_to_row_count"
    )
    candidates = candidate_specs(policy)
    primary_mode = policy["training"]["primary_sample_weight"]
    sensitivity_mode = policy["training"]["sensitivity_sample_weight"]
    internal_results = {}
    prediction_rows = []
    train_oof_prediction_rows = []
    public_results = {}
    for spec in candidates:
        train_matrix = matrix_for_rows(train_rows, features, spec["feature_names"])
        valid_matrix = matrix_for_rows(valid_rows, features, spec["feature_names"])
        primary = tune_and_fit(
            train_rows, train_matrix, spec["feature_names"], policy, primary_mode
        )
        sensitivity = tune_and_fit(
            train_rows, train_matrix, spec["feature_names"], policy, sensitivity_mode
        )
        primary_valid_scores = apply_logistic(
            valid_matrix, primary["final_train_artifact"]
        )
        sensitivity_valid_scores = apply_logistic(
            valid_matrix, sensitivity["final_train_artifact"]
        )
        primary_metrics = full_metrics(
            valid_rows,
            valid_labels,
            primary_valid_scores,
            primary["selected_threshold"],
        )
        sensitivity_metrics = full_metrics(
            valid_rows,
            valid_labels,
            sensitivity_valid_scores,
            sensitivity["selected_threshold"],
        )
        component_equal_valid_ap = weighted_average_precision(
            valid_labels, primary_valid_scores, valid_component_equal_weights
        )
        slices = evidence_slice_rates(
            valid_rows, primary_valid_scores, primary["selected_threshold"], policy
        )
        internal_results[spec["candidate_id"]] = {
            "spec": spec,
            "primary": primary,
            "sensitivity": sensitivity,
            "primary_valid_scores": primary_valid_scores,
            "sensitivity_valid_scores": sensitivity_valid_scores,
            "valid_evidence_slices": slices,
        }
        public_results[spec["candidate_id"]] = {
            **spec,
            "primary_training": without_score_arrays(primary),
            "primary_valid_metrics": primary_metrics,
            "primary_valid_component_equal_average_precision": component_equal_valid_ap,
            "uniform_weight_sensitivity_training": without_score_arrays(sensitivity),
            "uniform_weight_sensitivity_valid_metrics": sensitivity_metrics,
            "valid_evidence_slices": slices,
        }
        for row, probability in zip(
            train_rows, primary["train_oof_scores"], strict=True
        ):
            train_oof_prediction_rows.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "pair_uid": row["pair_uid"],
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "component_id": row["component_id"],
                    "oof_prob_positive": f"{float(probability):.12f}",
                    "threshold_from_train_oof": f"{float(primary['selected_threshold']):.12f}",
                    "predicted_positive": int(
                        probability >= primary["selected_threshold"]
                    ),
                }
            )
        for row, probability in zip(valid_rows, primary_valid_scores, strict=True):
            prediction_rows.append(
                {
                    "candidate_id": spec["candidate_id"],
                    "pair_uid": row["pair_uid"],
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "component_id": row["component_id"],
                    "prob_positive": f"{float(probability):.12f}",
                    "threshold_from_train_oof": f"{float(primary['selected_threshold']):.12f}",
                    "predicted_positive": int(probability >= primary["selected_threshold"]),
                }
            )

    ranked = sorted(
        candidates,
        key=lambda spec: (
            -public_results[spec["candidate_id"]]["primary_valid_metrics"][
                "average_precision"
            ],
            -public_results[spec["candidate_id"]]["primary_valid_metrics"]["roc_auc"],
            spec["candidate_id"],
        ),
    )
    top_id = ranked[0]["candidate_id"]
    runner_id = ranked[1]["candidate_id"]
    top_internal = internal_results[top_id]
    runner_internal = internal_results[runner_id]
    bootstrap = grouped_bootstrap_ap_delta(
        valid_rows,
        top_internal["primary_valid_scores"],
        runner_internal["primary_valid_scores"],
        policy,
    )
    top_public = public_results[top_id]
    runner_public = public_results[runner_id]
    primary_delta = (
        top_public["primary_valid_metrics"]["average_precision"]
        - runner_public["primary_valid_metrics"]["average_precision"]
    )
    uniform_ranked = sorted(
        candidates,
        key=lambda spec: (
            -public_results[spec["candidate_id"]][
                "uniform_weight_sensitivity_valid_metrics"
            ]["average_precision"],
            -public_results[spec["candidate_id"]][
                "uniform_weight_sensitivity_valid_metrics"
            ]["roc_auc"],
            spec["candidate_id"],
        ),
    )
    uniform_top_id = uniform_ranked[0]["candidate_id"]
    uniform_best_other_id = next(
        spec["candidate_id"]
        for spec in uniform_ranked
        if spec["candidate_id"] != top_id
    )
    sensitivity_delta = (
        top_public["uniform_weight_sensitivity_valid_metrics"]["average_precision"]
        - public_results[uniform_best_other_id][
            "uniform_weight_sensitivity_valid_metrics"
        ]["average_precision"]
    )
    component_equal_ranked = sorted(
        candidates,
        key=lambda spec: (
            -public_results[spec["candidate_id"]][
                "primary_valid_component_equal_average_precision"
            ],
            spec["candidate_id"],
        ),
    )
    component_equal_top_id = component_equal_ranked[0]["candidate_id"]
    component_equal_best_other_id = next(
        spec["candidate_id"]
        for spec in component_equal_ranked
        if spec["candidate_id"] != top_id
    )
    component_equal_delta = float(
        top_public["primary_valid_component_equal_average_precision"]
        - public_results[component_equal_best_other_id][
            "primary_valid_component_equal_average_precision"
        ]
    )
    top_hard_fpr = top_public["valid_evidence_slices"]["hard_negative"][
        "false_positive_rate"
    ]
    runner_hard_fpr = runner_public["valid_evidence_slices"]["hard_negative"][
        "false_positive_rate"
    ]
    top_direct_recall = top_public["valid_evidence_slices"]["direct_positive"]["recall"]
    runner_direct_recall = runner_public["valid_evidence_slices"]["direct_positive"]["recall"]
    if None in (top_hard_fpr, runner_hard_fpr, top_direct_recall, runner_direct_recall):
        raise ValueError("Step7-v2 winner guard evidence slices are not estimable")
    rule = policy["selection_rule"]["unique_winner_requires_all"]
    checks = {
        "valid_ap_delta": {
            "observed": primary_delta,
            "required_minimum": float(
                rule["valid_average_precision_delta_vs_runner_up_at_least"]
            ),
            "pass": primary_delta
            >= float(rule["valid_average_precision_delta_vs_runner_up_at_least"]),
        },
        "bootstrap_ci_lower": {
            "observed": bootstrap["ci95_lower"],
            "required_above": float(rule["grouped_bootstrap_ap_delta_ci95_lower_above"]),
            "pass": bootstrap["ci95_lower"]
            > float(rule["grouped_bootstrap_ap_delta_ci95_lower_above"]),
        },
        "uniform_weight_ap_delta": {
            "observed": sensitivity_delta,
            "required_above": float(rule["uniform_weight_sensitivity_ap_delta_above"]),
            "pass": sensitivity_delta
            > float(rule["uniform_weight_sensitivity_ap_delta_above"]),
        },
        "uniform_weight_top_rank_consistent": {
            "observed_top_candidate": uniform_top_id,
            "required_top_candidate": top_id,
            "pass": uniform_top_id == top_id,
        },
        "component_equal_validation_ap_delta": {
            "observed": component_equal_delta,
            "required_above": float(
                rule["component_equal_validation_ap_delta_above"]
            ),
            "pass": component_equal_delta
            > float(rule["component_equal_validation_ap_delta_above"]),
        },
        "component_equal_validation_top_rank_consistent": {
            "observed_top_candidate": component_equal_top_id,
            "required_top_candidate": top_id,
            "pass": component_equal_top_id == top_id,
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
    unique_winner = all(item["pass"] for item in checks.values())
    if unique_winner:
        carry_forward = [top_id]
        outcome = "unique_validation_winner"
    else:
        e5_candidate = next(
            spec["candidate_id"]
            for spec in ranked
            if spec["model_key"] == "multilingual_e5_large"
        )
        challenger = next(
            spec["candidate_id"]
            for spec in ranked
            if spec["model_key"] != "multilingual_e5_large"
        )
        carry_forward = [e5_candidate, challenger]
        outcome = "no_unique_winner_carry_top_e5_and_top_non_e5_candidates"

    identity_scores = np.asarray(
        [float(row["identity_rule_control_score"]) for row in valid_rows], dtype=np.float64
    )
    identity_control_metrics = full_metrics(
        valid_rows, valid_labels, identity_scores, threshold=0.5
    )
    summary = {
        "step": "step7_v2_select_source_model",
        "version": policy["version"],
        "scope": policy["result_scope"],
        "train_counts": dict(Counter(row["review_label"] for row in train_rows)),
        "valid_counts": dict(Counter(row["review_label"] for row in valid_rows)),
        "candidate_count": len(candidates),
        "candidate_ranking": [spec["candidate_id"] for spec in ranked],
        "uniform_weight_sensitivity_candidate_ranking": [
            spec["candidate_id"] for spec in uniform_ranked
        ],
        "component_equal_validation_ap_candidate_ranking": [
            spec["candidate_id"] for spec in component_equal_ranked
        ],
        "top_candidate": top_id,
        "runner_up_candidate": runner_id,
        "top_vs_runner_up_grouped_bootstrap": bootstrap,
        "unique_winner_checks": checks,
        "selection_outcome": outcome,
        "carry_forward_to_step28": carry_forward,
        "identity_rule_control": {
            "eligible_for_m0": False,
            "valid_metrics": identity_control_metrics,
        },
        "candidates": public_results,
        "runtime_model_manifests": runtime_manifests,
        "historical_test_label_values_parsed_during_selection": False,
        "historical_test_label_file_touched_during_selection": False,
        "test_metrics_used_for_selection": False,
        "prospective_claim_allowed": False,
    }
    freeze = {
        "step": "step7_v2_frozen_source_selection",
        "version": policy["version"],
        "policy_contract_sha256": common.canonical_hash(policy),
        "candidate_specs_sha256": common.canonical_hash(candidates),
        "selection_outcome": outcome,
        "carry_forward_to_step28": carry_forward,
        "top_candidate": top_id,
        "runner_up_candidate": runner_id,
        "runtime_inputs": runtime_input_fingerprints(policy),
        "historical_test_label_values_parsed_during_selection": False,
        "historical_test_label_file_hashed_during_selection": False,
        "historical_test_requires_explicit_acknowledgement": True,
    }
    return summary, prediction_rows, train_oof_prediction_rows, freeze


def verify_frozen_selection_consistency(
    policy: dict, freeze: dict, selection_summary: dict
) -> None:
    if freeze.get("step") != "step7_v2_frozen_source_selection":
        raise ValueError("Step7-v2 frozen selection role mismatch")
    if freeze.get("version") != policy["version"] or selection_summary.get(
        "version"
    ) != policy["version"]:
        raise ValueError("Step7-v2 frozen selection version mismatch")
    if selection_summary.get("step") != "step7_v2_select_source_model":
        raise ValueError("Step7-v2 selection summary role mismatch")
    if freeze.get("policy_contract_sha256") != common.canonical_hash(policy):
        raise ValueError("Step7-v2 frozen selection policy contract drift")
    if freeze.get("candidate_specs_sha256") != common.canonical_hash(
        candidate_specs(policy)
    ):
        raise ValueError("Step7-v2 frozen candidate contract drift")
    for field in (
        "selection_outcome",
        "carry_forward_to_step28",
        "top_candidate",
        "runner_up_candidate",
    ):
        if freeze.get(field) != selection_summary.get(field):
            raise ValueError(f"Step7-v2 frozen selection/summary mismatch: {field}")
    carried = freeze.get("carry_forward_to_step28")
    if not isinstance(carried, list) or not carried or len(carried) != len(set(carried)):
        raise ValueError("Step7-v2 frozen carry-forward candidate list is invalid")


def verify_frozen_inputs(freeze: dict) -> None:
    for key, record in freeze["runtime_inputs"].items():
        path = common.resolve(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Step7-v2 input is missing: {key}={path}")
        observed = common.sha256_file(path)
        if observed != record["sha256"]:
            raise ValueError(
                f"Frozen Step7-v2 input changed after selection: {key} "
                f"expected={record['sha256']} observed={observed}"
            )


def verify_historical_test_labels(policy: dict) -> dict:
    outputs = policy["outputs"]
    manifest_path = common.resolve(outputs["historical_test_labels_manifest"])
    label_path = common.resolve(outputs["historical_test_labels"])
    if not manifest_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("Step7-v2 historical-development labels are not materialized")
    manifest = common.load_json(manifest_path)
    if manifest.get("version") != policy["version"] or manifest.get("splits_written") != [
        "test"
    ]:
        raise ValueError("Step7-v2 historical-test label manifest mismatch")
    if manifest.get("other_split_label_values_used_during_materialization") is not False:
        raise ValueError("Step7-v2 historical-test materialization used another split")
    if manifest.get("split_projection_applied_before_label_or_evidence_access") is not True:
        raise ValueError("Step7-v2 historical-test split projection contract is missing")
    public_manifest = common.load_json(
        common.resolve(policy["outputs"]["preparation_manifest"])
    )
    if manifest.get("policy_sha256") != public_manifest.get("policy_sha256"):
        raise ValueError("Step7-v2 historical-test policy hash drift")
    if manifest.get("generator_script_sha256") != common.sha256_file(
        common.resolve("scripts/step7_v2_prepare_clean_data.py")
    ):
        raise ValueError("Step7-v2 historical-test preparation script drift")
    if manifest.get("common_script_sha256") != common.sha256_file(
        common.resolve("scripts/step7_v2_common.py")
    ):
        raise ValueError("Step7-v2 historical-test common script drift")
    if manifest.get("redaction_dependency_script_sha256") != common.sha256_file(
        common.resolve("scripts/step3_build_seller_profiles.py")
    ):
        raise ValueError("Step7-v2 historical-test Step3 dependency drift")
    public_path = common.resolve(outputs["preparation_manifest"])
    if manifest.get("public_preparation_manifest_sha256") != common.sha256_file(public_path):
        raise ValueError("Step7-v2 historical-test/public preparation drift")
    if set(manifest.get("output_files", {})) != {"private_labels_test"}:
        raise ValueError("Step7-v2 historical-test output universe mismatch")
    record = manifest["output_files"].get("private_labels_test")
    if record is None or common.resolve(record["path"]) != label_path:
        raise ValueError("Step7-v2 historical-test label path mismatch")
    verify_file_record(record, "historical-test-label")
    verify_policy_input_records(
        policy,
        manifest,
        ("frozen_labels", "evidence_labels"),
        "historical-test-label",
    )
    observed_inputs = common.validate_input_hashes(
        policy, ("frozen_labels", "evidence_labels")
    )
    if observed_inputs != manifest.get("input_manifest"):
        raise ValueError("Step7-v2 historical-test source inputs no longer replay")
    pair_path = common.resolve(outputs["pair_manifest"])
    pair_record = public_manifest.get("output_files", {}).get("pair_manifest")
    if pair_record is None or verify_file_record(
        pair_record, "historical-test-public-pair"
    ) != pair_path:
        raise ValueError("Step7-v2 historical-test public pair provenance mismatch")
    pair_rows = common.load_csv(pair_path)
    common.validate_public_pair_rows(policy, pair_rows)
    import step7_v2_prepare_clean_data as preparation

    replayed = preparation.prepare_private_labels(policy, pair_rows, ("test",))
    if replayed["input_manifest"] != observed_inputs:
        raise ValueError("Step7-v2 historical-test replay input manifest mismatch")
    if replayed["label_counts"] != manifest.get("label_counts"):
        raise ValueError("Step7-v2 historical-test replay label counts mismatch")
    replayed_bytes = common.render_csv(replayed["private"]["test"])
    if replayed_bytes != label_path.read_bytes():
        raise ValueError(
            "Step7-v2 historical-test labels do not byte-replay from pinned sources"
        )
    return {
        **manifest,
        "execution_time_source_inputs_rehashed": True,
        "execution_time_test_projection_byte_replayed": True,
        "execution_time_replayed_label_sha256": hashlib.sha256(
            replayed_bytes
        ).hexdigest(),
    }


def run_historical_test(policy: dict, freeze: dict, selection_summary: dict) -> tuple[dict, list[dict]]:
    verify_frozen_selection_consistency(policy, freeze, selection_summary)
    verify_frozen_inputs(freeze)
    historical_label_verification = verify_historical_test_labels(policy)
    pair_rows, features, _runtime_manifests = load_feature_bundle(policy)
    test_rows = load_split_rows(policy, "test", pair_rows)
    test_labels = labels_array(test_rows)
    candidate_by_id = {spec["candidate_id"]: spec for spec in candidate_specs(policy)}
    results = {}
    predictions = []
    for candidate_id in freeze["carry_forward_to_step28"]:
        spec = candidate_by_id.get(candidate_id)
        if spec is None:
            raise ValueError(f"Frozen Step7-v2 candidate no longer exists: {candidate_id}")
        public_candidate = selection_summary["candidates"].get(candidate_id)
        if public_candidate is None:
            raise ValueError(f"Frozen Step7-v2 candidate artifact missing: {candidate_id}")
        training = public_candidate["primary_training"]
        artifact = training["final_train_artifact"]
        if artifact["feature_names"] != spec["feature_names"]:
            raise ValueError(f"Frozen Step7-v2 feature contract drift: {candidate_id}")
        matrix = matrix_for_rows(test_rows, features, spec["feature_names"])
        scores = apply_logistic(matrix, artifact)
        threshold = float(training["selected_threshold"])
        results[candidate_id] = {
            "candidate_id": candidate_id,
            "feature_names": spec["feature_names"],
            "selected_l2_penalty_from_train_oof": training["selected_l2_penalty"],
            "threshold_from_train_oof": threshold,
            "historical_development_test_metrics": full_metrics(
                test_rows, test_labels, scores, threshold
            ),
            "historical_development_test_evidence_slices": evidence_slice_rates(
                test_rows, scores, threshold, policy
            ),
        }
        for row, probability in zip(test_rows, scores, strict=True):
            predictions.append(
                {
                    "candidate_id": candidate_id,
                    "pair_uid": row["pair_uid"],
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "component_id": row["component_id"],
                    "prob_positive": f"{float(probability):.12f}",
                    "threshold_from_train_oof": f"{threshold:.12f}",
                    "predicted_positive": int(probability >= threshold),
                }
            )
    identity_scores = np.asarray(
        [float(row["identity_rule_control_score"]) for row in test_rows], dtype=np.float64
    )
    summary = {
        "step": "step7_v2_historical_internal_development_test",
        "version": policy["version"],
        "test_role": policy["result_scope"]["english_test_role"],
        "prospective_claim_allowed": False,
        "selection_outcome": freeze["selection_outcome"],
        "evaluated_frozen_candidates_only": list(freeze["carry_forward_to_step28"]),
        "test_counts": dict(Counter(row["review_label"] for row in test_rows)),
        "candidates": results,
        "identity_rule_control": {
            "eligible_for_m0": False,
            "historical_development_test_metrics": full_metrics(
                test_rows, test_labels, identity_scores, threshold=0.5
            ),
        },
        "historical_test_metrics_computed_after_selection_freeze": True,
        "historical_test_source_inputs_rehashed_before_metrics": historical_label_verification[
            "execution_time_source_inputs_rehashed"
        ],
        "historical_test_projection_byte_replayed_before_metrics": historical_label_verification[
            "execution_time_test_projection_byte_replayed"
        ],
        "historical_test_labels_may_exist_before_selection": True,
        "warning": (
            "This English test was consumed by earlier project iterations. It is a delayed "
            "Step7-v2 development-metric check, not an untouched or cryptographically "
            "hidden prospective confirmation set."
        ),
    }
    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--stage", choices=("select", "test"), default="select")
    parser.add_argument(
        "--acknowledge-historical-test-is-development-only", action="store_true"
    )
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    specs = candidate_specs(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "stage": args.stage,
                    "candidate_count": len(specs),
                    "candidate_ids": [spec["candidate_id"] for spec in specs],
                    "train_rows": policy["supervision_boundary"]["expected_counts"]["train"],
                    "valid_rows": policy["supervision_boundary"]["expected_counts"]["valid"],
                    "test_rows": policy["supervision_boundary"]["expected_counts"]["test"],
                    "test_requires_explicit_acknowledgement": True,
                    "gpu_required_for_evaluation_after_scores_exist": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    outputs = policy["outputs"]
    if args.stage == "select":
        summary, predictions, train_oof_predictions, freeze = run_selection(policy)
        summary_path = common.resolve(outputs["selection_summary"])
        prediction_path = common.resolve(outputs["valid_predictions"])
        train_oof_prediction_path = common.resolve(outputs["train_oof_predictions"])
        freeze_path = common.resolve(outputs["frozen_selection_manifest"])
        common.write_json_immutable(summary_path, summary)
        common.write_csv_immutable(prediction_path, predictions)
        common.write_csv_immutable(train_oof_prediction_path, train_oof_predictions)
        freeze["selection_summary"] = {
            "path": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": common.sha256_file(summary_path),
            "size_bytes": summary_path.stat().st_size,
        }
        freeze["valid_predictions"] = {
            "path": str(prediction_path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": common.sha256_file(prediction_path),
            "size_bytes": prediction_path.stat().st_size,
        }
        freeze["train_oof_predictions"] = {
            "path": str(train_oof_prediction_path.relative_to(common.ROOT)).replace(
                "\\", "/"
            ),
            "sha256": common.sha256_file(train_oof_prediction_path),
            "size_bytes": train_oof_prediction_path.stat().st_size,
        }
        # A non-default policy path must be frozen exactly, rather than silently hashing
        # the repository default policy.
        freeze["runtime_inputs"]["policy"] = {
            "path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
            "sha256": common.sha256_file(policy_path),
            "size_bytes": policy_path.stat().st_size,
        }
        common.write_json_immutable(freeze_path, freeze)
        print(
            json.dumps(
                {
                    "status": "selection_frozen",
                    "selection_outcome": summary["selection_outcome"],
                    "top_candidate": summary["top_candidate"],
                    "carry_forward_to_step28": summary["carry_forward_to_step28"],
                    "historical_test_label_values_parsed": False,
                    "next_command": (
                        "python scripts/step7_v2_select_source_model.py --stage test "
                        "--acknowledge-historical-test-is-development-only"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.acknowledge_historical_test_is_development_only:
        raise SystemExit(
            "Opening Step7-v2 test requires --acknowledge-historical-test-is-development-only"
        )
    freeze_path = common.resolve(outputs["frozen_selection_manifest"])
    summary_path = common.resolve(outputs["selection_summary"])
    if not freeze_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Step7-v2 selection must be frozen before opening test")
    freeze = common.load_json(freeze_path)
    selection_summary = common.load_json(summary_path)
    if common.sha256_file(summary_path) != freeze["selection_summary"]["sha256"]:
        raise ValueError("Step7-v2 selection summary changed after freeze")
    for role in ("valid_predictions", "train_oof_predictions"):
        verify_file_record(freeze[role], f"frozen-selection/{role}")
    test_summary, predictions = run_historical_test(policy, freeze, selection_summary)
    test_summary_path = common.resolve(outputs["test_summary"])
    test_prediction_path = common.resolve(outputs["test_predictions"])
    common.write_json_immutable(test_summary_path, test_summary)
    common.write_csv_immutable(test_prediction_path, predictions)
    print(
        json.dumps(
            {
                "status": "historical_development_test_complete",
                "prospective_claim_allowed": False,
                "evaluated_candidates": test_summary["evaluated_frozen_candidates_only"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
