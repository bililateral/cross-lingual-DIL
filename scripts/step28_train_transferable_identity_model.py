#!/usr/bin/env python3
"""Fit and audit the Step28/v4 synthetic-history residual model on CPU."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import step28_common as base
import step28_history_common as history


POLICY_PATH = history.POLICY_PATH


def load_split(policy: dict, split: str) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row
        for row in base.load_csv(base.output_root(policy) / policy["outputs"]["model_inputs"])
        if row["synthetic_split"] == split
    ]
    names = policy["model"]["feature_names"]
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    labels = np.asarray([int(row["review_label"] == "positive") for row in rows], dtype=float)
    source = np.asarray([float(row["source_probability"]) for row in rows], dtype=float)
    if not rows or len(np.unique(labels)) != 2:
        raise ValueError(f"Step28/v4 split is empty or one-class: {split}")
    return rows, matrix, labels, source


def objective(source: np.ndarray, matrix: np.ndarray, labels: np.ndarray, beta: np.ndarray, l2: float) -> float:
    margin = base.logit(source) + matrix @ beta
    return float(np.sum(np.logaddexp(0.0, margin) - labels * margin) + 0.5 * l2 * np.dot(beta, beta))


def fit_residual(
    source: np.ndarray,
    raw_matrix: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    active_count: int,
    policy: dict,
    model_id: str,
) -> dict:
    matrix = np.asarray(raw_matrix, dtype=float) / scales
    if active_count < matrix.shape[1]:
        matrix = matrix.copy()
        matrix[:, active_count:] = 0.0
    beta = np.zeros(matrix.shape[1], dtype=float)
    l2 = float(policy["model"]["l2_penalty"])
    tolerance = float(policy["model"]["tolerance"])
    maximum_iterations = int(policy["model"]["maximum_iterations"])
    converged = False
    current_objective = objective(source, matrix, labels, beta, l2)
    gradient_max = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        probability = np.asarray(base.sigmoid(base.logit(source) + matrix @ beta), dtype=float)
        gradient = matrix.T @ (probability - labels) + l2 * beta
        if active_count < len(beta):
            gradient[active_count:] = 0.0
        gradient_max = float(np.max(np.abs(gradient[:active_count])))
        if gradient_max <= tolerance:
            converged = True
            break
        curvature = probability * (1.0 - probability)
        hessian = matrix.T @ (matrix * curvature[:, None]) + np.eye(matrix.shape[1]) * l2
        if active_count < len(beta):
            hessian[active_count:, :] = 0.0
            hessian[:, active_count:] = 0.0
            hessian[active_count:, active_count:] = np.eye(len(beta) - active_count)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        accepted = False
        for line_search in range(30):
            candidate = beta - step * (0.5 ** line_search)
            if active_count < len(beta):
                candidate[active_count:] = 0.0
            candidate_objective = objective(source, matrix, labels, candidate, l2)
            if candidate_objective <= current_objective + 1e-12:
                beta = candidate
                current_objective = candidate_objective
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"Step28/v4 line search failed: {model_id}")
    if not converged:
        raise RuntimeError(f"Step28/v4 solver did not converge: {model_id}:{gradient_max}")
    return {
        "model_id": model_id,
        "family": policy["model"]["family"],
        "feature_names": policy["model"]["feature_names"],
        "feature_scales": scales.tolist(),
        "coefficients": beta.tolist(),
        "active_feature_count": int(active_count),
        "source_logit_coefficient": 1.0,
        "intercept": 0.0,
        "l2_penalty": l2,
        "solver_iterations": iteration,
        "solver_converged": converged,
        "solver_gradient_max_abs": gradient_max,
        "solver_objective": current_objective,
        "chinese_real_identity_history_rows_used_for_fitting_or_selection": False,
        "frozen_source_carrier_domain": policy["generation"].get("source_carrier_domain", "zh_train_legacy"),
    }


def score(source: np.ndarray, matrix: np.ndarray, artifact: dict) -> np.ndarray:
    return history.predict_with_artifact(source, matrix, artifact)


def permutation_scores(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    correction = history.identity_correction(matrix, artifact)
    return np.asarray(base.sigmoid(correction), dtype=float)


def recipe_rate(rows: list[dict], labels: np.ndarray, scores: np.ndarray, threshold: float, recipes: set[str], positive: bool) -> float:
    indices = [
        index
        for index, row in enumerate(rows)
        if row["recipe_id"] in recipes and int(labels[index]) == int(positive)
    ]
    if not indices:
        raise ValueError(f"Step28/v4 recipe diagnostic has no rows: {sorted(recipes)}")
    predicted = scores[indices] >= threshold
    return float(np.mean(predicted))


def source_carrier_independence_preflight(
    policy: dict,
    splits: dict[str, tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]],
    generation_summary: dict,
) -> dict:
    """Stop before fitting if a label-blind policy contains carrier-label signal."""
    assignment = policy["generation"].get(
        "source_carrier_assignment", "legacy_label_correlated"
    )
    if assignment != "label_blind_exact_pairing":
        return {"required": False, "passed": True, "checks": {}}

    tolerance = float(
        policy["audit_gates"]["source_only_auc_max_abs_deviation_from_chance"]
    )
    split_diagnostics = {}
    for split, (rows, _matrix, labels, source) in splits.items():
        positive_indices = np.where(labels == 1.0)[0]
        negative_indices = np.where(labels == 0.0)[0]
        positive_scores = np.sort(source[positive_indices])
        negative_scores = np.sort(source[negative_indices])
        score_multiset_equal = (
            len(positive_scores) == len(negative_scores)
            and np.array_equal(positive_scores, negative_scores)
        )
        positive_carriers = Counter(
            rows[index]["source_carrier_pair_uid"] for index in positive_indices
        )
        negative_carriers = Counter(
            rows[index]["source_carrier_pair_uid"] for index in negative_indices
        )
        source_auc = base.roc_auc(labels, source)
        recorded = generation_summary["source_label_balance_by_split"][split]
        split_diagnostics[split] = {
            "source_carrier_uid_multiset_exactly_equal": (
                positive_carriers == negative_carriers
            ),
            "source_probability_multiset_exactly_equal": score_multiset_equal,
            "source_only_roc_auc": source_auc,
            "source_only_auc_abs_deviation_from_chance": abs(source_auc - 0.5),
            "generation_summary_agrees": (
                bool(recorded["source_carrier_uid_multiset_exactly_equal"])
                == (positive_carriers == negative_carriers)
                and bool(recorded["source_probability_multiset_exactly_equal"])
                == score_multiset_equal
                and abs(float(recorded["source_only_roc_auc"]) - source_auc) <= 1e-15
            ),
        }

    checks = {
        "source_carrier_label_input_key_absent": (
            "source_carrier_labels" not in policy["inputs"]
        ),
        "source_carrier_label_file_never_opened": (
            generation_summary["source_carrier_label_file_open_count"] == 0
        ),
        "source_carrier_label_column_absent": (
            generation_summary["source_carrier_label_column_count"] == 0
        ),
        "exact_carrier_multiset_balance_all_splits": all(
            item["source_carrier_uid_multiset_exactly_equal"]
            for item in split_diagnostics.values()
        ),
        "exact_source_probability_multiset_balance_all_splits": all(
            item["source_probability_multiset_exactly_equal"]
            for item in split_diagnostics.values()
        ),
        "source_only_auc_at_chance_all_splits": all(
            item["source_only_auc_abs_deviation_from_chance"] <= tolerance
            for item in split_diagnostics.values()
        ),
        "generation_summary_matches_model_inputs": all(
            item["generation_summary_agrees"]
            for item in split_diagnostics.values()
        ),
    }
    payload = {
        "required": True,
        "passed": all(checks.values()),
        "checks": checks,
        "splits": split_diagnostics,
        "fit_started_only_after_this_preflight": True,
    }
    if not payload["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "Step28 label-blind source-carrier preflight failed before fitting: "
            + ",".join(failed)
        )
    return payload


def _v7_mechanism_group(recipe: str) -> str:
    name = recipe.removesuffix("_holdout")
    if "source_only" in name:
        return "source_only"
    if "rotation" in name or "graph" in name:
        return "rotation_or_graph"
    if "support" in name:
        return "support_context"
    if "noisy" in name or "product" in name:
        return "risk_context"
    return "direct_reuse_or_collision"


def _v7_state_records(
    rows: list[dict], matrix: np.ndarray, labels: np.ndarray
) -> list[dict]:
    records: dict[str, dict] = {}
    for index, values in enumerate(matrix):
        state_hash = history.observable_state_hash(values)
        canonical = history.observable_state_values(values)
        current = records.setdefault(
            state_hash,
            {
                "state_hash": state_hash,
                "values": canonical,
                "positive_count": 0,
                "negative_count": 0,
                "recipes": set(),
                "row_indices": [],
            },
        )
        if current["values"] != canonical:
            raise RuntimeError("Step28/v7 observable-state hash collision")
        label_name = "positive" if labels[index] == 1.0 else "negative"
        current[f"{label_name}_count"] += 1
        current["recipes"].add(rows[index]["recipe_id"])
        current["row_indices"].append(index)
    return [records[key] for key in sorted(records)]


def _equal_observable_state_weights(matrix: np.ndarray) -> np.ndarray:
    """Give every observable state total weight one without consulting labels."""
    hashes = [history.observable_state_hash(values) for values in matrix]
    counts = Counter(hashes)
    return np.asarray([1.0 / counts[value] for value in hashes], dtype=float)


def _state_hashes(matrix: np.ndarray) -> list[str]:
    return [history.observable_state_hash(values) for values in matrix]


def _v7_support_map(
    splits: dict[str, tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]],
    selected_splits: tuple[str, ...],
) -> dict[str, dict]:
    support: dict[str, dict] = {}
    for split in selected_splits:
        rows, matrix, labels, _source = splits[split]
        for index, values in enumerate(matrix):
            state_hash = history.observable_state_hash(values)
            current = support.setdefault(
                state_hash,
                {
                    "positive_count": 0,
                    "negative_count": 0,
                    "splits": set(),
                    "recipes": set(),
                },
            )
            label_name = "positive" if labels[index] == 1.0 else "negative"
            current[f"{label_name}_count"] += 1
            current["splits"].add(split)
            current["recipes"].add(rows[index]["recipe_id"])
    output = {}
    for state_hash, current in sorted(support.items()):
        positive = int(current["positive_count"])
        negative = int(current["negative_count"])
        status = (
            "ambiguous"
            if positive and negative
            else "positive_only"
            if positive
            else "negative_only"
        )
        output[state_hash] = {
            "positive_count": positive,
            "negative_count": negative,
            "status": status,
            "splits": sorted(current["splits"]),
            "recipes": sorted(current["recipes"]),
        }
    return output


def _v7_block_permutation(
    labels: np.ndarray, groups: list[str], rng: np.random.Generator
) -> np.ndarray:
    permuted = np.asarray(labels, dtype=float).copy()
    for group in sorted(set(groups)):
        indices = np.asarray(
            [index for index, value in enumerate(groups) if value == group],
            dtype=int,
        )
        if len(indices) < 2 or len(np.unique(labels[indices])) < 2:
            raise RuntimeError(
                f"Step28/v7 permutation block lacks both labels: {group}"
            )
        permuted[indices] = rng.permutation(labels[indices])
    return permuted


def train_v7(
    policy: dict,
    splits: dict[str, tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]],
    generation_summary: dict,
    source_preflight: dict,
) -> tuple[dict, list[dict], dict]:
    """Fit one equally weighted row per identifiable observable history state."""
    state_weighted_audit = (
        policy["model"].get("audit_evaluation_unit")
        == "all_rows_equal_observable_state_weight"
    )
    train_rows, train_matrix, train_labels, _train_source = splits["synthetic_train"]
    train_states = _v7_state_records(train_rows, train_matrix, train_labels)
    train_ambiguous_state_hashes = {
        record["state_hash"]
        for record in train_states
        if bool(record["positive_count"]) and bool(record["negative_count"])
    }
    all_identifiable_train = [
        record
        for record in train_states
        if bool(record["positive_count"]) != bool(record["negative_count"])
    ]
    if not all_identifiable_train:
        raise RuntimeError("Step28/v7 has no identifiable training states")
    grouped_states: dict[str, dict[int, list[dict]]] = {}
    for record in all_identifiable_train:
        groups = sorted({_v7_mechanism_group(recipe) for recipe in record["recipes"]})
        if len(groups) != 1:
            raise RuntimeError(
                "Step28/v7 observable state spans multiple permutation blocks: "
                + ",".join(groups)
            )
        label = int(bool(record["positive_count"]))
        grouped_states.setdefault(groups[0], {0: [], 1: []})[label].append(record)
    identifiable_train = []
    for group, by_label in sorted(grouped_states.items()):
        for label in (0, 1):
            by_label[label].sort(key=lambda record: record["state_hash"])
        balanced_count = min(len(by_label[0]), len(by_label[1]))
        if balanced_count == 0:
            raise RuntimeError(
                f"Step28/v7 fit block lacks an identifiable class: {group}"
            )
        identifiable_train.extend(by_label[0][:balanced_count])
        identifiable_train.extend(by_label[1][:balanced_count])
    identifiable_train.sort(key=lambda record: record["state_hash"])
    fit_matrix = np.asarray([record["values"] for record in identifiable_train], dtype=float)
    fit_labels = np.asarray(
        [float(bool(record["positive_count"])) for record in identifiable_train],
        dtype=float,
    )
    fit_groups = [
        sorted({_v7_mechanism_group(recipe) for recipe in record["recipes"]})[0]
        for record in identifiable_train
    ]
    scales = np.sqrt(np.mean(np.square(fit_matrix), axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    neutral_source = np.full(len(fit_labels), 0.5, dtype=float)
    full_count = len(policy["model"]["feature_names"])
    direct_count = int(policy["model"]["direct_history_feature_count"])
    direct = fit_residual(
        neutral_source,
        fit_matrix,
        fit_labels,
        scales,
        direct_count,
        policy,
        "m1_unique_state_direct_history_only",
    )
    full = fit_residual(
        neutral_source,
        fit_matrix,
        fit_labels,
        scales,
        full_count,
        policy,
        "m2_unique_state_full_history",
    )
    for artifact in (direct, full):
        artifact["fit_row_unit"] = "one_equal_weight_row_per_unique_identifiable_observable_state"
        artifact["raw_synthetic_train_row_count"] = len(train_rows)
        artifact["unique_train_state_count"] = len(train_states)
        artifact["identifiable_train_state_count"] = len(identifiable_train)
        artifact["all_identifiable_train_state_count_before_block_balance"] = len(
            all_identifiable_train
        )
        artifact["ambiguous_train_state_count"] = len(train_states) - len(
            all_identifiable_train
        )
        artifact["identifiable_train_states_excluded_by_block_balance"] = len(
            all_identifiable_train
        ) - len(identifiable_train)
        artifact["source_probability_used_during_identity_fit"] = 0.5

    support_map = _v7_support_map(
        splits, ("synthetic_train", "synthetic_development")
    )
    full["observable_state_support"] = support_map
    direct["observable_state_support"] = support_map

    development = _v7_state_records(*splits["synthetic_development"][:3])
    development_rows, development_raw_matrix, development_raw_labels, _ = splits[
        "synthetic_development"
    ]
    if state_weighted_audit:
        # Development labels may select a threshold, but audit labels must never
        # select audit rows.  Exclude only states that the already-frozen
        # train+development support map marks observationally contradictory.
        threshold_indices = np.asarray(
            [
                index
                for index, state_hash in enumerate(_state_hashes(development_raw_matrix))
                if support_map[state_hash]["status"] != "ambiguous"
            ],
            dtype=int,
        )
        development_matrix = development_raw_matrix[threshold_indices]
        development_labels = development_raw_labels[threshold_indices]
        development_weights = _equal_observable_state_weights(development_matrix)
        development_scores = np.asarray(
            base.sigmoid(history.identity_correction(development_matrix, full)),
            dtype=float,
        )
        threshold, threshold_metrics = history.choose_threshold_weighted(
            development_labels, development_scores, development_weights
        )
        identifiable_development = []
    else:
        identifiable_development = [
            record
            for record in development
            if bool(record["positive_count"]) != bool(record["negative_count"])
            and record["state_hash"] not in train_ambiguous_state_hashes
        ]
        development_matrix = np.asarray(
            [record["values"] for record in identifiable_development], dtype=float
        )
        development_labels = np.asarray(
            [float(bool(record["positive_count"])) for record in identifiable_development],
            dtype=float,
        )
        development_scores = np.asarray(
            base.sigmoid(history.identity_correction(development_matrix, full)),
            dtype=float,
        )
        threshold, threshold_metrics = history.choose_threshold(
            development_labels, development_scores
        )

    rng = np.random.default_rng(int(policy["generation"]["seed"]) + 991)
    placebo_artifacts = []
    for repetition in range(int(policy["model"]["permutation_repetitions"])):
        permuted_labels = _v7_block_permutation(fit_labels, fit_groups, rng)
        placebo_artifacts.append(
            fit_residual(
                neutral_source,
                fit_matrix,
                permuted_labels,
                scales,
                full_count,
                policy,
                f"n0_block_permuted_unique_state_labels_{repetition:03d}",
            )
        )

    metrics_by_split: dict[str, dict] = {}
    prediction_rows: list[dict] = []
    state_records_by_split: dict[str, list[dict]] = {}
    identity_scores_by_split: dict[str, np.ndarray] = {}
    direct_scores_by_split: dict[str, np.ndarray] = {}
    for split, (rows, matrix, labels, source) in splits.items():
        state_records = _v7_state_records(rows, matrix, labels)
        identifiable = [
            record
            for record in state_records
            if bool(record["positive_count"]) != bool(record["negative_count"])
        ]
        state_matrix = np.asarray([record["values"] for record in identifiable], dtype=float)
        state_labels = np.asarray(
            [float(bool(record["positive_count"])) for record in identifiable],
            dtype=float,
        )
        full_identity = np.asarray(
            base.sigmoid(history.identity_correction(state_matrix, full)), dtype=float
        )
        direct_identity = np.asarray(
            base.sigmoid(history.identity_correction(state_matrix, direct)), dtype=float
        )
        state_records_by_split[split] = identifiable
        identity_scores_by_split[split] = full_identity
        direct_scores_by_split[split] = direct_identity
        row_full_identity = np.asarray(
            base.sigmoid(history.identity_correction(matrix, full)), dtype=float
        )
        row_direct_identity = np.asarray(
            base.sigmoid(history.identity_correction(matrix, direct)), dtype=float
        )
        row_combined = score(source, matrix, full)
        primary_metrics = {
            "unique_identifiable_states_label_oracle_diagnostic_only": {
                "s0_no_history": history.metrics(
                    state_labels, np.full(len(state_labels), 0.5), threshold
                ),
                "m1_direct_history": history.metrics(
                    state_labels, direct_identity, threshold
                ),
                "m2_full_history": history.metrics(
                    state_labels, full_identity, threshold
                ),
            }
        }
        if state_weighted_audit:
            state_weights = _equal_observable_state_weights(matrix)
            primary_metrics["all_rows_equal_observable_state_weight"] = {
                "s0_no_history": history.weighted_metrics(
                    labels, np.full(len(labels), 0.5), threshold, state_weights
                ),
                "m1_direct_history": history.weighted_metrics(
                    labels, row_direct_identity, threshold, state_weights
                ),
                "m2_full_history": history.weighted_metrics(
                    labels, row_full_identity, threshold, state_weights
                ),
            }
        metrics_by_split[split] = {
            **primary_metrics,
            "row_level_secondary": {
                "m2_identity_only": history.metrics(labels, row_full_identity, threshold),
                "m3_frozen_source_plus_history": history.metrics(
                    labels, row_combined, threshold
                ),
            },
            "raw_row_count": len(rows),
            "unique_state_count": len(state_records),
            "identifiable_state_count": len(identifiable),
            "ambiguous_state_count": len(state_records) - len(identifiable),
        }
        for index, row in enumerate(rows):
            state_hash = history.observable_state_hash(matrix[index])
            support = support_map.get(
                state_hash,
                {"status": "unseen", "positive_count": 0, "negative_count": 0},
            )
            correction = float(history.identity_correction(matrix[index : index + 1], full)[0])
            current_scores = {
                "s0_no_history": 0.5,
                "m1_direct_history": float(row_direct_identity[index]),
                "m2_full_history": float(row_full_identity[index]),
                "m3_frozen_source_plus_history": float(row_combined[index]),
            }
            for model_id, value in current_scores.items():
                prediction_rows.append(
                    {
                        "pair_uid": row["pair_uid"],
                        "synthetic_split": split,
                        "recipe_id": row["recipe_id"],
                        "review_label": row["review_label"],
                        "model_id": model_id,
                        "score": f"{value:.12f}",
                        "threshold": f"{threshold:.12f}",
                        "predicted_label": (
                            "positive" if value >= threshold else "negative"
                        ),
                        "observable_state_hash": state_hash,
                        "state_support_status": support["status"],
                        "identity_correction": f"{correction:.12f}",
                    }
                )

    audit_records = state_records_by_split["synthetic_audit"]
    if state_weighted_audit:
        audit_rows, audit_matrix, audit_labels, _audit_source = splits[
            "synthetic_audit"
        ]
        audit_weights = _equal_observable_state_weights(audit_matrix)
        audit_metrics = metrics_by_split["synthetic_audit"][
            "all_rows_equal_observable_state_weight"
        ]["m2_full_history"]
        audit_direct_metrics = metrics_by_split["synthetic_audit"][
            "all_rows_equal_observable_state_weight"
        ]["m1_direct_history"]
        placebo_aucs = np.asarray(
            [
                base.weighted_roc_auc(
                    audit_labels,
                    np.asarray(
                        base.sigmoid(
                            history.identity_correction(audit_matrix, artifact)
                        ),
                        dtype=float,
                    ),
                    audit_weights,
                )
                for artifact in placebo_artifacts
            ],
            dtype=float,
        )
    else:
        audit_labels = np.asarray(
            [float(bool(record["positive_count"])) for record in audit_records],
            dtype=float,
        )
        audit_metrics = metrics_by_split["synthetic_audit"][
            "unique_identifiable_states_label_oracle_diagnostic_only"
        ]["m2_full_history"]
        audit_direct_metrics = metrics_by_split["synthetic_audit"][
            "unique_identifiable_states_label_oracle_diagnostic_only"
        ]["m1_direct_history"]
        audit_matrix = np.asarray(
            [record["values"] for record in audit_records], dtype=float
        )
        placebo_aucs = np.asarray(
            [
                base.roc_auc(
                    audit_labels,
                    np.asarray(
                        base.sigmoid(
                            history.identity_correction(audit_matrix, artifact)
                        ),
                        dtype=float,
                    ),
                )
                for artifact in placebo_artifacts
            ],
            dtype=float,
        )
    permutation_p_value = float(
        (1 + int(np.sum(placebo_aucs >= audit_metrics["roc_auc"])))
        / (len(placebo_aucs) + 1)
    )

    audit_rows, audit_row_matrix, audit_row_labels, _audit_source = splits["synthetic_audit"]
    audit_row_scores = np.asarray(
        base.sigmoid(history.identity_correction(audit_row_matrix, full)), dtype=float
    )
    audit_row_corrections = history.identity_correction(audit_row_matrix, full)
    support_matrix = np.vstack(
        [splits["synthetic_train"][1], splits["synthetic_development"][1]]
    )
    feature_minimum = np.min(support_matrix, axis=0)
    feature_maximum = np.max(support_matrix, axis=0)
    support_corrections = history.identity_correction(support_matrix, full)
    correction_minimum = float(np.min(support_corrections))
    correction_maximum = float(np.max(support_corrections))
    recipe_diagnostics = {}
    for recipe in sorted({row["recipe_id"] for row in audit_rows}):
        indices = [
            index for index, row in enumerate(audit_rows) if row["recipe_id"] == recipe
        ]
        expected_positive = recipe.startswith("positive_")
        state_calls: dict[str, dict] = {}
        for index in indices:
            state_hash = history.observable_state_hash(audit_row_matrix[index])
            support = support_map.get(
                state_hash,
                {
                    "status": "unseen",
                    "positive_count": 0,
                    "negative_count": 0,
                    "splits": [],
                },
            )
            values = audit_row_matrix[index]
            out_of_support = bool(
                np.any(
                    (values < feature_minimum - 1e-12)
                    | (values > feature_maximum + 1e-12)
                )
            )
            bounded = np.clip(values, feature_minimum, feature_maximum)[None, :]
            bounded_correction = float(history.identity_correction(bounded, full)[0])
            bounded_correction = float(
                np.clip(bounded_correction, correction_minimum, correction_maximum)
            )
            current = {
                "model_positive": bool(audit_row_scores[index] >= threshold),
                "production_review_eligible": history.positive_review_eligible(
                    identity_correction=bounded_correction,
                    out_of_support=out_of_support,
                    support=support,
                    policy=policy,
                ),
                "support_status": support["status"],
                "out_of_support": out_of_support,
            }
            previous = state_calls.setdefault(state_hash, current)
            if previous != current:
                raise RuntimeError(
                    "Step28 equal observable state produced inconsistent recipe calls"
                )
        unique_calls = list(state_calls.values())
        status_counts = Counter(row["support_status"] for row in unique_calls)
        recipe_diagnostics[recipe] = {
            "raw_row_count": len(indices),
            "unique_observable_state_count": len(unique_calls),
            "expected_label": "positive" if expected_positive else "negative",
            "identity_model_positive_state_rate": float(
                np.mean([row["model_positive"] for row in unique_calls])
            ),
            "production_review_eligible_state_rate": float(
                np.mean([row["production_review_eligible"] for row in unique_calls])
            ),
            "out_of_support_state_rate": float(
                np.mean([row["out_of_support"] for row in unique_calls])
            ),
            "support_status_counts_by_unique_state": dict(status_counts),
        }

    recipe_rules = policy["audit_gates"]["audit_recipe_rules"]
    recipe_checks = {}
    for recipe, rule in recipe_rules.items():
        metric = rule["metric"]
        observed = recipe_diagnostics[recipe][metric]
        if rule["comparison"] == "minimum":
            recipe_checks[recipe] = observed >= float(rule["threshold"])
        elif rule["comparison"] == "maximum":
            recipe_checks[recipe] = observed <= float(rule["threshold"])
        else:
            raise ValueError(
                f"Step28 recipe rule comparison is invalid: {recipe}"
            )

    audit_row_direct_scores = np.asarray(
        base.sigmoid(history.identity_correction(audit_row_matrix, direct)), dtype=float
    )

    def cohort_metrics(recipes: set[str], *, unseen_hashes: set[str] | None = None) -> dict:
        indices = [
            index
            for index, row in enumerate(audit_rows)
            if row["recipe_id"] in recipes
            and (
                unseen_hashes is None
                or history.observable_state_hash(audit_row_matrix[index])
                not in unseen_hashes
            )
        ]
        if not indices:
            return {
                "raw_row_count": 0,
                "unique_observable_state_count": 0,
                "unique_states_by_recipe": {},
                "m1_direct_history": None,
                "m2_full_history": None,
            }
        matrix = audit_row_matrix[indices]
        labels = audit_row_labels[indices]
        weights = _equal_observable_state_weights(matrix)
        return {
            "raw_row_count": len(indices),
            "unique_observable_state_count": len(set(_state_hashes(matrix))),
            "unique_states_by_recipe": {
                recipe: len(
                    {
                        history.observable_state_hash(audit_row_matrix[index])
                        for index in indices
                        if audit_rows[index]["recipe_id"] == recipe
                    }
                )
                for recipe in sorted(recipes)
            },
            "m1_direct_history": history.weighted_metrics(
                labels, audit_row_direct_scores[indices], threshold, weights
            ),
            "m2_full_history": history.weighted_metrics(
                labels, audit_row_scores[indices], threshold, weights
            ),
        }

    recipe_groups = policy["audit_gates"]["audit_recipe_groups"]
    discrimination_recipes = set(recipe_groups["model_discrimination"])
    abstention_recipes = set(recipe_groups["guard_abstention"])
    all_audit_recipes = {row["recipe_id"] for row in audit_rows}
    discrimination_metrics = cohort_metrics(discrimination_recipes)
    current_support_hashes = set(support_map)
    unseen_current_discrimination_metrics = cohort_metrics(
        discrimination_recipes, unseen_hashes=current_support_hashes
    )

    prior_state_hashes: set[str] = set()
    for key in policy["generation"].get(
        "cross_version_reference_model_input_keys", []
    ):
        for row in base.load_csv(policy["inputs"][key]):
            values = np.asarray(
                [float(row[name]) for name in policy["model"]["feature_names"]],
                dtype=float,
            )
            prior_state_hashes.add(history.observable_state_hash(values))
    prior_and_current_hashes = prior_state_hashes | current_support_hashes
    unseen_all_prior_discrimination_metrics = cohort_metrics(
        discrimination_recipes, unseen_hashes=prior_and_current_hashes
    )

    gates = policy["audit_gates"]
    maximum_split_overlap = max(
        value
        for field in generation_summary["split_overlap_counts"].values()
        for value in field.values()
    )
    audit_state_count = int(
        generation_summary["feature_state_diagnostics"]["synthetic_audit"][
            "unique_feature_state_count"
        ]
    )
    train_audit_overlap = int(
        generation_summary["feature_state_overlap_counts"][
            "synthetic_train__synthetic_audit"
        ]
    )
    audit_overlap_fraction = train_audit_overlap / max(audit_state_count, 1)
    matrix_ranks = {
        split: int(details["feature_matrix_rank"])
        for split, details in generation_summary["feature_state_diagnostics"].items()
    }
    zero_history = np.max(np.abs(audit_row_matrix), axis=1) == 0.0
    zero_correction = float(
        np.max(np.abs(audit_row_corrections[zero_history]))
    )
    forbidden = policy["feature_boundary"]["forbidden_feature_name_fragments"]
    leak_names = [
        name
        for name in policy["model"]["feature_names"]
        if any(fragment in name.lower() for fragment in forbidden)
    ]
    all_feature_matrix = np.vstack([values[1] for values in splits.values()])
    dead_feature_names = [
        name
        for index, name in enumerate(policy["model"]["feature_names"])
        if np.max(np.abs(all_feature_matrix[:, index])) <= 1e-15
    ]
    discrimination_full = discrimination_metrics["m2_full_history"]
    unseen_current_full = unseen_current_discrimination_metrics["m2_full_history"]
    checks = {
        "source_carrier_independence_preflight": source_preflight["passed"],
        "parser_context_recovery": generation_summary["parser_recovery"]
        >= float(gates["parser_recovery_minimum"]),
        "split_overlap": maximum_split_overlap <= int(gates["split_overlap_maximum"]),
        "cross_version_synthetic_overlap": generation_summary[
            "maximum_cross_version_synthetic_overlap"
        ]
        <= int(gates["cross_version_synthetic_overlap_maximum"]),
        "no_duplicate_generated_ids": all(
            generation_summary[key] == 0
            for key in (
                "duplicate_world_uid_count",
                "duplicate_pair_uid_count",
                "duplicate_item_uid_count",
            )
        ),
        "feature_oracle_leak": len(leak_names)
        <= int(gates["feature_oracle_leak_count_maximum"]),
        "no_dead_model_features": not dead_feature_names,
        "train_unique_state_count": len(train_states)
        >= int(gates["train_unique_feature_state_minimum"]),
        "development_unique_state_count": len(development)
        >= int(gates["development_unique_feature_state_minimum"]),
        "audit_unique_state_count": audit_state_count
        >= int(gates["audit_unique_feature_state_minimum"]),
        "train_feature_rank": matrix_ranks["synthetic_train"]
        >= int(gates["train_feature_matrix_rank_minimum"]),
        "audit_feature_rank": matrix_ranks["synthetic_audit"]
        >= int(gates["audit_feature_matrix_rank_minimum"]),
        "audit_train_exact_state_overlap_fraction": audit_overlap_fraction
        <= float(gates["audit_train_exact_state_overlap_fraction_maximum"]),
        "audit_identity_roc_auc": audit_metrics["roc_auc"]
        >= float(gates["audit_roc_auc_minimum"]),
        "audit_identity_average_precision": audit_metrics["average_precision"]
        >= float(gates["audit_average_precision_minimum"]),
        "audit_ap_gain_over_direct": (
            audit_metrics["average_precision"] - audit_direct_metrics["average_precision"]
        )
        >= float(gates["audit_ap_gain_over_direct_model_minimum"]),
        "model_discrimination_recipe_partition": (
            not (discrimination_recipes & abstention_recipes)
            and discrimination_recipes | abstention_recipes == all_audit_recipes
            and all(
                (
                    rule["layer"] == "identity_model"
                    and recipe in discrimination_recipes
                    and rule["metric"] == "identity_model_positive_state_rate"
                )
                or (
                    rule["layer"] == "production_guard"
                    and recipe in abstention_recipes
                    and rule["metric"] == "production_review_eligible_state_rate"
                )
                for recipe, rule in recipe_rules.items()
            )
        ),
        "model_discrimination_cohort_auc": (
            discrimination_full is not None
            and discrimination_full["roc_auc"]
            >= float(gates["model_discrimination_cohort_roc_auc_minimum"])
        ),
        "model_discrimination_cohort_ap": (
            discrimination_full is not None
            and discrimination_full["average_precision"]
            >= float(gates["model_discrimination_cohort_average_precision_minimum"])
        ),
        "unseen_current_discrimination_state_count": (
            unseen_current_discrimination_metrics["unique_observable_state_count"]
            >= int(gates["unseen_current_discrimination_state_minimum"])
        ),
        "unseen_current_discrimination_auc": (
            unseen_current_full is not None
            and unseen_current_full["roc_auc"]
            >= float(gates["unseen_current_discrimination_roc_auc_minimum"])
        ),
        "unseen_current_discrimination_ap": (
            unseen_current_full is not None
            and unseen_current_full["average_precision"]
            >= float(gates["unseen_current_discrimination_average_precision_minimum"])
        ),
        "block_permutation_empirical_p_value": permutation_p_value
        <= float(gates["permutation_empirical_p_value_maximum"]),
        "block_permutation_auc_mean": abs(float(np.mean(placebo_aucs)) - 0.5)
        <= float(gates["permutation_auc_mean_max_abs_deviation_from_chance"]),
        "zero_history_identity_correction": zero_correction
        <= float(gates["zero_history_source_fallback_max_abs_error"]),
        "all_audit_recipes_explicitly_gated": set(recipe_diagnostics) == set(recipe_rules),
        "all_audit_recipe_rules_pass": all(recipe_checks.values()),
        "primary_audit_does_not_filter_on_audit_labels": state_weighted_audit,
    }
    audit = {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "metrics_by_split": metrics_by_split,
        "threshold_selected_on": (
            "all_synthetic_development_rows_with_equal_observable_state_weight_"
            "excluding_only_states_ambiguous_in_frozen_train_development_support"
            if state_weighted_audit
            else "legacy_unique_identifiable_synthetic_development_states"
        ),
        "threshold": threshold,
        "threshold_development_metrics": threshold_metrics,
        "observable_state_accounting": {
            "raw_train_rows": len(train_rows),
            "unique_train_states": len(train_states),
            "identifiable_train_states_used_for_fit": len(identifiable_train),
            "all_identifiable_train_states_before_block_balance": len(
                all_identifiable_train
            ),
            "ambiguous_train_states_excluded_from_fit": len(train_states)
            - len(all_identifiable_train),
            "identifiable_train_states_excluded_by_block_balance": len(
                all_identifiable_train
            ) - len(identifiable_train),
            "unique_development_states": len(development),
            "development_raw_rows_used_for_threshold": len(development_matrix),
            "development_unique_states_used_for_threshold": len(
                set(_state_hashes(development_matrix))
            ),
            "development_states_excluded_due_to_train_development_ambiguity": sum(
                support_map[record["state_hash"]]["status"] == "ambiguous"
                for record in development
            ),
            "unique_audit_states": audit_state_count,
            "audit_raw_rows_used_for_primary_metrics": len(audit_row_matrix),
            "audit_state_equal_weight_total": float(
                np.sum(_equal_observable_state_weights(audit_row_matrix))
            ),
            "audit_label_oracle_identifiable_state_diagnostic_count": len(audit_records),
            "audit_train_exact_state_overlap_count": train_audit_overlap,
            "audit_train_exact_state_overlap_fraction": audit_overlap_fraction,
            "feature_matrix_rank_by_split": matrix_ranks,
        },
        "audit_recipe_diagnostics": recipe_diagnostics,
        "audit_recipe_checks": recipe_checks,
        "model_discrimination_cohort": discrimination_metrics,
        "unseen_current_support_model_discrimination_cohort": (
            unseen_current_discrimination_metrics
        ),
        "unseen_all_prior_and_current_support_model_discrimination_cohort": (
            unseen_all_prior_discrimination_metrics
        ),
        "prior_observable_state_hash_count": len(prior_state_hashes),
        "block_permutation": {
            "fit_unit": "unique_identifiable_observable_state",
            "audit_unit": (
                "all_rows_equal_observable_state_weight_without_audit_label_filter"
                if state_weighted_audit
                else "legacy_unique_identifiable_observable_state"
            ),
            "blocks": sorted(set(fit_groups)),
            "repetition_count": len(placebo_aucs),
            "audit_auc_mean": float(np.mean(placebo_aucs)),
            "audit_auc_standard_deviation": float(np.std(placebo_aucs)),
            "audit_auc_minimum": float(np.min(placebo_aucs)),
            "audit_auc_maximum": float(np.max(placebo_aucs)),
            "observed_audit_auc": audit_metrics["roc_auc"],
            "empirical_one_sided_p_value": permutation_p_value,
        },
        "audit_diagnostics": {
            "source_carrier_independence_preflight": source_preflight,
            "audit_identity_auc": audit_metrics["roc_auc"],
            "audit_identity_average_precision": audit_metrics["average_precision"],
            "audit_ap_gain_over_direct": audit_metrics["average_precision"]
            - audit_direct_metrics["average_precision"],
            "maximum_split_overlap": maximum_split_overlap,
            "maximum_cross_version_synthetic_overlap": generation_summary[
                "maximum_cross_version_synthetic_overlap"
            ],
            "zero_history_identity_correction_max_abs": zero_correction,
            "feature_oracle_leak_names": leak_names,
            "dead_model_feature_names": dead_feature_names,
            "block_permutation_empirical_p_value": permutation_p_value,
        },
        "old_valid_test_open_count": 0,
        "chinese_real_rows_used_for_history_residual_fitting_selection_or_gating": 0,
        "claim_boundary": policy["scientific_position"]["allowed_claim"],
    }
    artifacts = {
        "decision": audit["decision"],
        "primary_model": full,
        "direct_history_control": direct,
        "permuted_label_control_summary": audit["block_permutation"],
        "frozen_threshold": threshold,
        "frozen_source_scorer": policy["frozen_source_scorer"],
        "model_score_is_real_probability": False,
        "prospective_real_review_required": True,
        "observable_state_support_rule": (
            "positive review calls require positive_only support in synthetic train+development; "
            "ambiguous, negative_only, and unseen states abstain"
        ),
    }
    return artifacts, prediction_rows, audit


def train(policy: dict) -> tuple[dict, list[dict], dict]:
    splits = {
        name: load_split(policy, name)
        for name in ("synthetic_train", "synthetic_development", "synthetic_audit")
    }
    generation_summary = base.load_json(
        base.output_root(policy) / policy["outputs"]["generation_summary"]
    )
    source_preflight = source_carrier_independence_preflight(
        policy, splits, generation_summary
    )
    if policy["model"].get("training_protocol_version") in {
        "v7_unique_states",
        "v12_all_audit_states_equal_weight",
    }:
        return train_v7(policy, splits, generation_summary, source_preflight)
    train_rows, train_matrix, train_labels, train_source = splits["synthetic_train"]
    scales = np.sqrt(np.mean(np.square(train_matrix), axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    full_count = len(policy["model"]["feature_names"])
    direct_count = int(policy["model"]["direct_history_feature_count"])
    direct = fit_residual(
        train_source, train_matrix, train_labels, scales, direct_count, policy,
        "m1_learned_direct_history_only",
    )
    full = fit_residual(
        train_source, train_matrix, train_labels, scales, full_count, policy,
        "m2_learned_full_history",
    )
    rng = np.random.default_rng(int(policy["generation"]["seed"]) + 991)
    placebo_artifacts = []
    for repetition in range(int(policy["model"].get("permutation_repetitions", 1))):
        permuted_labels = rng.permutation(train_labels)
        placebo_artifacts.append(fit_residual(
            np.full_like(train_source, 0.5), train_matrix, permuted_labels, scales,
            full_count, policy, f"n0_permuted_synthetic_labels_{repetition:02d}",
        ))

    train_full_scores = score(train_source, train_matrix, full)
    threshold, threshold_metrics = history.choose_threshold(train_labels, train_full_scores)
    model_scores: dict[str, dict[str, np.ndarray]] = {}
    metrics_by_split: dict[str, dict[str, dict]] = {}
    placebo_distribution_by_split: dict[str, dict] = {}
    prediction_rows: list[dict] = []
    for split, (rows, matrix, labels, source) in splits.items():
        placebo_repetition_scores = [
            permutation_scores(matrix, item) for item in placebo_artifacts
        ]
        current = {
            "s0_frozen_source": source,
            "m1_learned_direct_history_only": score(source, matrix, direct),
            "m2_learned_full_history": score(source, matrix, full),
            "n0_permuted_synthetic_labels": np.mean(
                np.vstack(placebo_repetition_scores),
                axis=0,
            ),
        }
        model_scores[split] = current
        metrics_by_split[split] = {
            model_id: history.metrics(labels, values, threshold)
            for model_id, values in current.items()
        }
        placebo_aucs = np.asarray([
            base.roc_auc(labels, values) for values in placebo_repetition_scores
        ], dtype=float)
        placebo_aps = np.asarray([
            base.average_precision(labels, values) for values in placebo_repetition_scores
        ], dtype=float)
        placebo_distribution_by_split[split] = {
            "repetition_count": len(placebo_repetition_scores),
            "roc_auc_mean": float(np.mean(placebo_aucs)),
            "roc_auc_standard_deviation": float(np.std(placebo_aucs)),
            "roc_auc_minimum": float(np.min(placebo_aucs)),
            "roc_auc_maximum": float(np.max(placebo_aucs)),
            "average_precision_mean": float(np.mean(placebo_aps)),
            "auc_of_mean_scores_not_used_for_gate": metrics_by_split[split][
                "n0_permuted_synthetic_labels"
            ]["roc_auc"],
        }
        for index, row in enumerate(rows):
            for model_id, values in current.items():
                prediction_rows.append({
                    "pair_uid": row["pair_uid"],
                    "synthetic_split": split,
                    "recipe_id": row["recipe_id"],
                    "review_label": row["review_label"],
                    "model_id": model_id,
                    "score": f"{float(values[index]):.12f}",
                    "threshold": f"{threshold:.12f}",
                    "predicted_label": "positive" if values[index] >= threshold else "negative",
                })

    audit_rows, audit_matrix, audit_labels, audit_source = splits["synthetic_audit"]
    audit_source_metrics = metrics_by_split["synthetic_audit"]["s0_frozen_source"]
    audit_direct_metrics = metrics_by_split["synthetic_audit"]["m1_learned_direct_history_only"]
    audit_full_metrics = metrics_by_split["synthetic_audit"]["m2_learned_full_history"]
    audit_placebo_metrics = placebo_distribution_by_split["synthetic_audit"]
    audit_full_scores = model_scores["synthetic_audit"]["m2_learned_full_history"]
    rotation_recall = recipe_rate(
        audit_rows, audit_labels, audit_full_scores, threshold,
        {"positive_cross_channel_rotation_holdout", "positive_noisy_rotation_holdout"},
        True,
    )
    collision_fpr = recipe_rate(
        audit_rows, audit_labels, audit_full_scores, threshold,
        {"negative_email_hub_holdout", "negative_product_telegram_holdout"},
        False,
    )
    zero_history = np.max(np.abs(audit_matrix), axis=1) == 0.0
    zero_error = float(np.max(np.abs(audit_full_scores[zero_history] - audit_source[zero_history])))
    maximum_overlap = max(
        value
        for field in generation_summary["split_overlap_counts"].values()
        for value in field.values()
    )
    forbidden = policy["feature_boundary"]["forbidden_feature_name_fragments"]
    leak_names = [
        name
        for name in policy["model"]["feature_names"]
        if any(fragment in name.lower() for fragment in forbidden)
    ]
    gates = policy["audit_gates"]
    checks = {
        "source_carrier_independence_preflight": source_preflight["passed"],
        "parser_recovery": generation_summary["parser_recovery"] >= float(gates["parser_recovery_minimum"]),
        "split_overlap": maximum_overlap <= int(gates["split_overlap_maximum"]),
        "feature_oracle_leak": len(leak_names) <= int(gates["feature_oracle_leak_count_maximum"]),
        "audit_roc_auc": audit_full_metrics["roc_auc"] >= float(gates["audit_roc_auc_minimum"]),
        "audit_average_precision": audit_full_metrics["average_precision"] >= float(gates["audit_average_precision_minimum"]),
        "audit_ap_gain_over_source": (
            audit_full_metrics["average_precision"] - audit_source_metrics["average_precision"]
            >= float(gates["audit_ap_gain_over_source_minimum"])
        ),
        "audit_ap_gain_over_direct_model": (
            audit_full_metrics["average_precision"] - audit_direct_metrics["average_precision"]
            >= float(gates["audit_ap_gain_over_direct_model_minimum"])
        ),
        "audit_positive_rotation_recall": rotation_recall >= float(gates["audit_positive_rotation_recall_minimum"]),
        "audit_public_or_product_collision_fpr": collision_fpr <= float(gates["audit_public_or_product_collision_fpr_maximum"]),
        "permutation_auc_lower": audit_placebo_metrics["roc_auc_mean"] >= float(gates["permutation_auc_minimum"]),
        "permutation_auc_upper": audit_placebo_metrics["roc_auc_mean"] <= float(gates["permutation_auc_maximum"]),
        "zero_history_source_fallback": zero_error <= float(gates["zero_history_source_fallback_max_abs_error"]),
    }
    audit = {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "metrics_by_split": metrics_by_split,
        "permuted_label_auc_distribution_by_split": placebo_distribution_by_split,
        "audit_diagnostics": {
            "source_carrier_independence_preflight": source_preflight,
            "audit_source_only_auc_abs_deviation_from_chance": abs(
                audit_source_metrics["roc_auc"] - 0.5
            ),
            "ap_gain_over_source": audit_full_metrics["average_precision"] - audit_source_metrics["average_precision"],
            "ap_gain_over_direct_model": audit_full_metrics["average_precision"] - audit_direct_metrics["average_precision"],
            "positive_rotation_recall": rotation_recall,
            "public_or_product_collision_fpr": collision_fpr,
            "zero_history_source_fallback_max_abs_error": zero_error,
            "maximum_split_overlap": maximum_overlap,
            "feature_oracle_leak_names": leak_names,
            "permuted_label_audit_auc_mean": audit_placebo_metrics["roc_auc_mean"],
            "permuted_label_audit_auc_standard_deviation": audit_placebo_metrics[
                "roc_auc_standard_deviation"
            ],
        },
        "threshold_selected_on": "synthetic_train",
        "threshold": threshold,
        "threshold_train_metrics": threshold_metrics,
        "old_valid_test_open_count": 0,
        "source_carrier_label_file_open_count": generation_summary[
            "source_carrier_label_file_open_count"
        ],
        "source_carrier_label_column_count": generation_summary[
            "source_carrier_label_column_count"
        ],
        "chinese_real_rows_used_for_history_residual_fitting_selection_or_gating": 0,
        "frozen_source_carrier_pair_count": int(sum(
            generation_summary["source_carrier_partition_counts"][split][label]
            for split in generation_summary["source_carrier_partition_counts"]
            for label in generation_summary["source_carrier_partition_counts"][split]
        )),
        "claim_boundary": policy["scientific_position"]["allowed_claim"],
    }
    artifacts = {
        "decision": audit["decision"],
        "primary_model": full,
        "direct_history_control": direct,
        "permuted_label_controls": placebo_artifacts,
        "frozen_threshold": threshold,
        "frozen_source_scorer": policy["frozen_source_scorer"],
        "model_score_is_real_probability": False,
        "prospective_real_review_required": True,
    }
    return artifacts, prediction_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy = history.load_policy(args.policy)
    base.validate_frozen_inputs(policy)
    artifacts, predictions, summary = train(policy)
    root = base.output_root(policy)
    outputs = policy["outputs"]
    base.write_json_immutable(root / outputs["model_artifacts"], artifacts)
    base.write_csv_immutable(
        root / outputs["synthetic_predictions"], predictions,
        [
            "pair_uid", "synthetic_split", "recipe_id", "review_label",
            "model_id", "score", "threshold", "predicted_label",
            "observable_state_hash", "state_support_status", "identity_correction",
        ],
    )
    base.write_json_immutable(root / outputs["training_summary"], summary)
    print(json.dumps({"status": "ok", "decision": summary["decision"], **summary["audit_diagnostics"]}, ensure_ascii=False))
    if summary["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
