#!/usr/bin/env python3
"""Fail-closed replay audit for current Step28 synthetic and application stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step3_build_seller_profiles as step3
import step28_common as base
import step28_history_common as history
import step28_score_real_identity_candidates as scorer
import step28_train_transferable_identity_model as trainer


def close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def independent_state_weights(matrix: np.ndarray) -> np.ndarray:
    hashes = [history.observable_state_hash(row) for row in matrix]
    counts = Counter(hashes)
    return np.asarray([1.0 / counts[value] for value in hashes], dtype=float)


def independent_weighted_auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    positive = float(np.sum(w[y == 1]))
    negative = float(np.sum(w[y == 0]))
    if positive <= 0.0 or negative <= 0.0:
        return float("nan")
    total = 0.0
    negative_below = 0.0
    order = np.argsort(s, kind="mergesort")
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group = order[index:end]
        positive_here = float(np.sum(w[group][y[group] == 1]))
        negative_here = float(np.sum(w[group][y[group] == 0]))
        total += positive_here * (negative_below + 0.5 * negative_here)
        negative_below += negative_here
        index = end
    return total / (positive * negative)


def independent_weighted_ap(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    positive = float(np.sum(w[y == 1]))
    if positive <= 0.0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    cumulative_positive = 0.0
    cumulative_total = 0.0
    value = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group = order[index:end]
        positive_here = float(np.sum(w[group][y[group] == 1]))
        cumulative_positive += positive_here
        cumulative_total += float(np.sum(w[group]))
        value += (cumulative_positive / cumulative_total) * (
            positive_here / positive
        )
        index = end
    return value


def independent_weighted_confusion(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= threshold
    w = np.asarray(weights, dtype=float)
    tp = float(np.sum(w[(y == 1) & predicted]))
    fp = float(np.sum(w[(y == 0) & predicted]))
    tn = float(np.sum(w[(y == 0) & ~predicted]))
    fn = float(np.sum(w[(y == 1) & ~predicted]))
    recall = tp / max(tp + fn, 1e-15)
    specificity = tn / max(tn + fp, 1e-15)
    precision = tp / max(tp + fp, 1e-15)
    return {
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_tn": tn,
        "weighted_fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-15),
    }


def independent_weighted_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 1e-12, 1.0 - 1e-12)
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    return {
        "raw_row_count": len(y),
        "state_equal_weight_total": total,
        "positive_weight": float(np.sum(w[y == 1.0])),
        "negative_weight": float(np.sum(w[y == 0.0])),
        "roc_auc": independent_weighted_auc(y, p, w),
        "average_precision": independent_weighted_ap(y, p, w),
        "logloss": float(
            np.sum(w * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
            / total
        ),
        "threshold": threshold,
        **independent_weighted_confusion(y, p, w, threshold),
    }


def independent_weighted_threshold(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[float, dict]:
    candidates = np.concatenate(
        ([0.0], np.unique(np.asarray(scores, dtype=float)), [1.0])
    )
    best = None
    best_threshold = 0.5
    for threshold in candidates:
        current = independent_weighted_confusion(
            labels, scores, weights, float(threshold)
        )
        key = (
            current["balanced_accuracy"],
            current["f1"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, current)
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best[1]


def state_records(rows: list[dict], names: list[str]) -> list[dict]:
    records: dict[str, dict] = {}
    for row in rows:
        values = np.asarray([float(row[name]) for name in names], dtype=float)
        state_hash = history.observable_state_hash(values)
        canonical = history.observable_state_values(values)
        current = records.setdefault(
            state_hash,
            {
                "values": canonical,
                "positive_count": 0,
                "negative_count": 0,
            },
        )
        if current["values"] != canonical:
            raise RuntimeError("Step28/v10 self-audit found a state-hash collision")
        current[f"{row['review_label']}_count"] += 1
    return [records[key] for key in sorted(records)]


def support_map(rows: list[dict], names: list[str]) -> dict[str, dict]:
    support: dict[str, dict] = {}
    for row in rows:
        if row["synthetic_split"] not in {
            "synthetic_train",
            "synthetic_development",
        }:
            continue
        values = np.asarray([float(row[name]) for name in names], dtype=float)
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
        current[f"{row['review_label']}_count"] += 1
        current["splits"].add(row["synthetic_split"])
        current["recipes"].add(row["recipe_id"])
    output = {}
    for state_hash, current in sorted(support.items()):
        positive = int(current["positive_count"])
        negative = int(current["negative_count"])
        output[state_hash] = {
            "positive_count": positive,
            "negative_count": negative,
            "status": (
                "ambiguous"
                if positive and negative
                else "positive_only"
                if positive
                else "negative_only"
            ),
            "splits": sorted(current["splits"]),
            "recipes": sorted(current["recipes"]),
        }
    return output


def reparse_items(items: list[dict]) -> list[dict]:
    fields = list(step3.ITEM_SIGNAL_FIELDS)
    parsed = []
    for item in items:
        meta = {
            key: item[key]
            for key in (
                "data_bucket",
                "source_dataset",
                "source_row_number",
                "seller_uid",
                "source_market_raw",
                "source_seller_raw",
                "source_seller_id_raw",
                "alias_normalized",
            )
        }
        rows = step3.extract_item_identity_signals(
            meta,
            title_raw=item["title_raw"],
            description_raw=item["description_raw"],
            structured_snapshot=item["structured_snapshot"],
        )
        for row in rows:
            parsed.append(
                {
                    "world_uid": item["world_uid"],
                    **{name: str(row.get(name, "")) for name in fields},
                }
            )
    return parsed


def policy_chain(path: Path) -> list[Path]:
    raw = base.load_json(path)
    parent = raw.get("_extends")
    paths = policy_chain(base.resolve(parent)) if parent else []
    return [*paths, path.resolve()]


def add_unique(paths: list[Path], *values: Path) -> None:
    known = {path.resolve() for path in paths}
    for value in values:
        resolved = value.resolve()
        if resolved not in known:
            paths.append(resolved)
            known.add(resolved)


def audit(training_policy_path: Path, application_policy_path: Path) -> tuple[dict, list[Path]]:
    training_policy = history.load_policy(training_policy_path)
    application_policy = history.load_policy(application_policy_path)
    frozen_training_inputs = base.validate_frozen_inputs(training_policy)
    frozen_application_inputs = base.validate_frozen_inputs(application_policy)
    training_root = base.output_root(training_policy)
    application_root = base.output_root(application_policy)
    train_outputs = training_policy["outputs"]
    app_outputs = application_policy["outputs"]

    truth = base.load_jsonl(training_root / train_outputs["world_truth"])
    items = base.load_jsonl(training_root / train_outputs["synthetic_items"])
    parsed = base.load_csv(training_root / train_outputs["parsed_occurrences"])
    model_rows = base.load_csv(training_root / train_outputs["model_inputs"])
    generation = base.load_json(training_root / train_outputs["generation_summary"])
    artifacts = base.load_json(training_root / train_outputs["model_artifacts"])
    predictions = base.load_csv(training_root / train_outputs["synthetic_predictions"])
    training = base.load_json(training_root / train_outputs["training_summary"])
    real_candidates = base.load_csv(application_policy["inputs"]["real_unlabeled_candidate_pool"])
    real_signals = base.load_csv(application_policy["inputs"]["real_item_identity_signals"])
    reviewed_registry_rows = base.load_csv(
        application_policy["inputs"]["known_reviewed_pair_uid_exclusions"]
    )
    reviewed = {row["pair_uid"] for row in reviewed_registry_rows}
    scores = base.load_csv(application_root / app_outputs["real_candidate_scores"])
    queue = base.load_csv(application_root / app_outputs["prospective_review_queue"])
    blind = base.load_csv(application_root / app_outputs["blind_evidence_packet"])
    adjudication = base.load_csv(
        application_root / app_outputs["blind_adjudication_template"]
    )
    real_summary = base.load_json(application_root / app_outputs["real_scoring_summary"])

    names = training_policy["model"]["feature_names"]
    model = artifacts["primary_model"]
    threshold = float(artifacts["frozen_threshold"])
    split_rows = {
        split: [row for row in model_rows if row["synthetic_split"] == split]
        for split in (
            "synthetic_train",
            "synthetic_development",
            "synthetic_audit",
        )
    }

    source_balance_recomputed = {}
    feature_state_recomputed = {}
    for split, rows in split_rows.items():
        positive = [row for row in rows if row["review_label"] == "positive"]
        negative = [row for row in rows if row["review_label"] == "negative"]
        positive_carriers = Counter(row["source_carrier_pair_uid"] for row in positive)
        negative_carriers = Counter(row["source_carrier_pair_uid"] for row in negative)
        positive_source = sorted(float(row["source_probability"]) for row in positive)
        negative_source = sorted(float(row["source_probability"]) for row in negative)
        labels = np.asarray(
            [float(row["review_label"] == "positive") for row in rows], dtype=float
        )
        source = np.asarray([float(row["source_probability"]) for row in rows])
        source_balance_recomputed[split] = {
            "carrier_equal": positive_carriers == negative_carriers,
            "source_equal": positive_source == negative_source,
            "source_auc": base.roc_auc(labels, source),
        }
        matrix = np.asarray(
            [[float(row[name]) for name in names] for row in rows], dtype=float
        )
        records = state_records(rows, names)
        feature_state_recomputed[split] = {
            "unique_feature_state_count": len(records),
            "feature_matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "cross_label_ambiguous_state_count": sum(
                bool(record["positive_count"]) and bool(record["negative_count"])
                for record in records
            ),
        }

    recomputed_support = support_map(model_rows, names)
    support_exact = recomputed_support == model.get("observable_state_support")
    development_rows = split_rows["synthetic_development"]
    development_all_matrix = np.asarray(
        [[float(row[name]) for name in names] for row in development_rows],
        dtype=float,
    )
    development_all_labels = np.asarray(
        [float(row["review_label"] == "positive") for row in development_rows]
    )
    development_indices = np.asarray(
        [
            index
            for index, values in enumerate(development_all_matrix)
            if recomputed_support[history.observable_state_hash(values)]["status"]
            != "ambiguous"
        ],
        dtype=int,
    )
    development_matrix = development_all_matrix[development_indices]
    development_labels = development_all_labels[development_indices]
    development_weights = independent_state_weights(development_matrix)
    development_scores = np.asarray(
        base.sigmoid(history.identity_correction(development_matrix, model)), dtype=float
    )
    recomputed_threshold, _ = independent_weighted_threshold(
        development_labels, development_scores, development_weights
    )
    audit_rows = split_rows["synthetic_audit"]
    audit_matrix = np.asarray(
        [[float(row[name]) for name in names] for row in audit_rows], dtype=float
    )
    audit_labels = np.asarray(
        [float(row["review_label"] == "positive") for row in audit_rows]
    )
    audit_weights = independent_state_weights(audit_matrix)
    audit_scores = np.asarray(
        base.sigmoid(history.identity_correction(audit_matrix, model)), dtype=float
    )
    direct_model = artifacts["direct_history_control"]
    audit_direct_scores = np.asarray(
        base.sigmoid(history.identity_correction(audit_matrix, direct_model)),
        dtype=float,
    )
    recomputed_audit_metrics = {
        "m1_direct_history": independent_weighted_metrics(
            audit_labels, audit_direct_scores, audit_weights, threshold
        ),
        "m2_full_history": independent_weighted_metrics(
            audit_labels, audit_scores, audit_weights, threshold
        ),
    }
    recorded_audit_metrics = training["metrics_by_split"]["synthetic_audit"][
        "all_rows_equal_observable_state_weight"
    ]
    replay_artifacts, replay_predictions, replay_training = trainer.train(
        training_policy
    )
    full_training_replay_exact = (
        replay_artifacts == artifacts
        and replay_predictions == predictions
        and replay_training == training
    )
    support_matrix = np.asarray(
        [
            [float(row[name]) for name in names]
            for row in model_rows
            if row["synthetic_split"]
            in {"synthetic_train", "synthetic_development"}
        ],
        dtype=float,
    )
    feature_minimum = np.min(support_matrix, axis=0)
    feature_maximum = np.max(support_matrix, axis=0)
    support_corrections = history.identity_correction(support_matrix, model)
    correction_minimum = float(np.min(support_corrections))
    correction_maximum = float(np.max(support_corrections))
    score_recomputation_errors = 0
    support_annotation_errors = 0
    production_eligibility_errors = 0
    rank_errors = 0
    for index, row in enumerate(scores, 1):
        values = np.asarray([float(row[name]) for name in names], dtype=float)
        state_hash = history.observable_state_hash(values)
        support = recomputed_support.get(
            state_hash,
            {
                "positive_count": 0,
                "negative_count": 0,
                "status": "unseen",
                "splits": [],
            },
        )
        out = np.where(
            (values < feature_minimum - 1e-12)
            | (values > feature_maximum + 1e-12)
        )[0]
        bounded = np.clip(values, feature_minimum, feature_maximum)[None, :]
        correction = float(history.identity_correction(bounded, model)[0])
        correction = float(np.clip(correction, correction_minimum, correction_maximum))
        production_eligible = history.positive_review_eligible(
            identity_correction=correction,
            out_of_support=bool(len(out)),
            support=support,
            policy=application_policy,
        )
        source = history.source_probability_from_cosine(float(row["source_cosine"]), training_policy)
        combined = float(base.sigmoid(base.logit(source) + correction))
        score_recomputation_errors += int(
            not close(source, float(row["frozen_source_score"]))
            or not close(correction, float(row["identity_logit_correction"]))
            or not close(combined, float(row["synthetic_scale_model_score"]))
            or len(out) != int(row["out_of_support_feature_count"])
        )
        support_annotation_errors += int(
            row["observable_state_hash"] != state_hash
            or row["synthetic_train_development_support_status"]
            != support["status"]
            or int(row["synthetic_train_development_positive_support_count"])
            != int(support["positive_count"])
            or int(row["synthetic_train_development_negative_support_count"])
            != int(support["negative_count"])
            or row["synthetic_support_splits"] != ";".join(support["splits"])
        )
        production_eligibility_errors += int(
            int(row["production_review_eligible"]) != int(production_eligible)
        )
        rank_errors += int(int(row["rank"]) != index)

    eligible = [
        row
        for row in scores
        if int(row["production_review_eligible"]) == 1
    ][: int(application_policy["real_scoring"]["review_queue_size"])]
    expected_queue_uids = [row["pair_uid"] for row in eligible]
    observed_queue_uids = [row["pair_uid"] for row in queue]

    score_path = application_root / app_outputs["real_candidate_scores"]
    queue_path = application_root / app_outputs["prospective_review_queue"]
    blind_path = application_root / app_outputs["blind_evidence_packet"]
    with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
        score_header = next(csv.reader(handle))
    with queue_path.open("r", encoding="utf-8-sig", newline="") as handle:
        queue_header = next(csv.reader(handle))
    with blind_path.open("r", encoding="utf-8-sig", newline="") as handle:
        blind_header = next(csv.reader(handle))
    expected_queue_header = [
        "queue_rank",
        "blind_id",
        *[name for name in score_header if name != "rank"],
        "review_status",
        "review_label",
        "review_notes",
    ]
    blind_columns = set(blind_header)
    blind_forbidden_fragments = {
        "score", "rank", "feature", "support", "correction", "probability", "pair_uid"
    }
    blind_json_errors = 0
    blind_occurrence_schema_errors = 0

    def audit_blind_json(value: object) -> None:
        nonlocal blind_occurrence_schema_errors
        if isinstance(value, list):
            for item in value:
                audit_blind_json(item)
        elif isinstance(value, dict):
            if "source_dataset" in value:
                blind_occurrence_schema_errors += int(
                    set(value) != set(scorer.BLIND_OCCURRENCE_FIELDS)
                )
            blind_occurrence_schema_errors += sum(
                key in {
                    "evidence_level",
                    "seller_facing_context",
                    "product_data_risk_context",
                    "direct_identity_eligible",
                    "support_only",
                    "context",
                }
                for key in value
            )
            for item in value.values():
                audit_blind_json(item)

    for row in blind:
        for key in ("shared_identity_evidence_json", "rotation_identity_evidence_json"):
            try:
                value = json.loads(row[key])
                blind_json_errors += int(not isinstance(value, list))
                audit_blind_json(value)
            except json.JSONDecodeError:
                blind_json_errors += 1

    reparsed = reparse_items(items)
    parser_reproduction_exact = reparsed == parsed
    world_split = {row["world_uid"]: row["synthetic_split"] for row in truth}
    parser_type_counts_by_split: dict[str, Counter] = defaultdict(Counter)
    parser_role_counts_by_split: dict[str, Counter] = defaultdict(Counter)
    for row in parsed:
        split = world_split[row["world_uid"]]
        parser_type_counts_by_split[split][row["contact_type"]] += 1
        role = (
            "risk"
            if history.is_risky(row)
            else "support"
            if history.is_support(row)
            else "direct"
            if history.is_direct(row)
            else "other"
        )
        parser_role_counts_by_split[split][role] += 1
    synthetic_values = {
        str(value).strip().lower()
        for row in truth
        for value in row.get("identifier_values", [])
        if str(value).strip()
    }
    real_values = {
        str(row.get("normalized_value", "")).strip().lower()
        for row in real_signals
        if str(row.get("normalized_value", "")).strip()
    }
    cross_version_overlap = {}
    prior_item_paths: list[Path] = []
    for key in training_policy["generation"][
        "cross_version_reference_world_truth_keys"
    ]:
        prior = base.load_jsonl(training_policy["inputs"][key])
        prior_world_path = base.resolve(training_policy["inputs"][key])
        prior_item_path = Path(
            str(prior_world_path).replace("_world_truth.jsonl", "_synthetic_items.jsonl")
        )
        if not prior_item_path.is_file():
            raise FileNotFoundError(
                f"Step28 prior synthetic item lineage is missing: {prior_item_path}"
            )
        prior_item_paths.append(prior_item_path)
        prior_items = base.load_jsonl(prior_item_path)
        cross_version_overlap[key] = {
            "world_uid": len(
                {row["world_uid"] for row in truth}
                & {row["world_uid"] for row in prior}
            ),
            "pair_uid": len(
                {row["pair_uid"] for row in truth}
                & {row["pair_uid"] for row in prior}
            ),
            "synthetic_seller_uid": len(
                set().union(*(set(row["synthetic_seller_uids"]) for row in truth))
                & set().union(*(set(row["synthetic_seller_uids"]) for row in prior))
            ),
            "controller_uid": len(
                set().union(*(set(row["controller_uids"]) for row in truth))
                & set().union(*(set(row["controller_uids"]) for row in prior))
            ),
            "item_uid": len(
                {row["item_uid"] for row in items}
                & {row["item_uid"] for row in prior_items}
            ),
            "identifier_value": len(
                set().union(*(set(row["identifier_values"]) for row in truth))
                & set().union(*(set(row["identifier_values"]) for row in prior))
            ),
        }

    reviewed_score_overlap = reviewed & {row["pair_uid"] for row in scores}
    reviewed_queue_overlap = reviewed & set(observed_queue_uids)
    candidate_boundary_errors = sum(
        row.get("review_status") != "pending"
        or bool(str(row.get("review_label", "")).strip())
        for row in real_candidates
    )
    recorded_feature_states_match = all(
        all(
            feature_state_recomputed[split][key]
            == generation["feature_state_diagnostics"][split][key]
            for key in feature_state_recomputed[split]
        )
        for split in feature_state_recomputed
    )
    metric_fields = (
        "raw_row_count", "state_equal_weight_total", "positive_weight",
        "negative_weight", "roc_auc", "average_precision", "logloss", "threshold",
        "weighted_tp", "weighted_fp", "weighted_tn", "weighted_fn",
        "recall", "specificity", "precision", "balanced_accuracy", "f1",
    )
    metrics_exact = all(
        close(recomputed_audit_metrics[model_id][key], recorded_audit_metrics[model_id][key])
        for model_id in ("m1_direct_history", "m2_full_history")
        for key in metric_fields
    )
    permutation = training["block_permutation"]

    checks = {
        "all_frozen_input_hashes_match": True,
        "synthetic_row_counts_recompute": len(truth) == len(model_rows) == generation["row_count"],
        "synthetic_item_and_parser_counts_recompute": (
            len(items) == generation["item_count"]
            and len(parsed) == generation["parsed_occurrence_count"]
        ),
        "production_parser_output_reproduces_exactly": parser_reproduction_exact,
        "all_identifier_types_are_parser_observed_in_every_split": all(
            set(parser_type_counts_by_split[split])
            == set(history.SUPPORTED_IDENTITY_TYPES)
            for split in split_rows
        ),
        "direct_risk_and_support_contexts_are_observed_in_every_split": all(
            {"direct", "risk", "support"}
            <= set(parser_role_counts_by_split[split])
            for split in split_rows
        ),
        "latent_controller_labels_exact": all(
            bool(row["same_latent_controller"])
            == (row["review_label"] == "positive")
            for row in truth
        ),
        "generated_ids_are_unique": (
            len({row["world_uid"] for row in truth}) == len(truth)
            and len({row["pair_uid"] for row in truth}) == len(truth)
            and len({row["item_uid"] for row in items}) == len(items)
        ),
        "cross_version_synthetic_ids_and_values_are_new": all(
            value == 0
            for fields in cross_version_overlap.values()
            for value in fields.values()
        ),
        "synthetic_identifiers_do_not_copy_real_values": not (synthetic_values & real_values),
        "source_carriers_are_label_blind_and_exactly_paired": all(
            details["carrier_equal"]
            and details["source_equal"]
            and details["source_auc"] == 0.5
            for details in source_balance_recomputed.values()
        ),
        "feature_state_counts_and_ranks_recompute": recorded_feature_states_match,
        "training_and_all_registered_gates_are_go": (
            training["decision"] == artifacts["decision"] == "GO"
            and all(training["checks"].values())
        ),
        "full_training_and_199_permutation_replay_exact": full_training_replay_exact,
        "training_policy_is_fully_expanded": (
            "_extends" not in base.load_json(training_policy_path)
        ),
        "model_feature_order_matches_policy": model["feature_names"] == names,
        "model_is_finite_intercept_free_and_fixed_source": (
            model["intercept"] == 0.0
            and model["source_logit_coefficient"] == 1.0
            and all(math.isfinite(float(value)) for value in model["coefficients"])
        ),
        "development_threshold_recomputes": close(recomputed_threshold, threshold),
        "audit_all_states_equal_weight_metrics_recompute": metrics_exact,
        "observable_state_support_recomputes": support_exact,
        "permutation_is_balanced_and_significant": (
            int(permutation["repetition_count"]) == 199
            and abs(float(permutation["audit_auc_mean"]) - 0.5) <= 0.10
            and float(permutation["empirical_one_sided_p_value"]) <= 0.05
            and permutation["audit_unit"]
            == "all_rows_equal_observable_state_weight_without_audit_label_filter"
        ),
        "real_candidate_input_cells_are_unlabeled": candidate_boundary_errors == 0,
        "reviewed_registry_is_uid_only_unique_and_counted": (
            len(reviewed_registry_rows) == len(reviewed) == 1259
            and all(set(row) == {"pair_uid"} for row in reviewed_registry_rows)
        ),
        "known_reviewed_pairs_are_absent_before_scoring": (
            not reviewed_score_overlap
            and real_summary["known_reviewed_pair_uid_remaining_in_universe_count"] == 0
        ),
        "known_reviewed_pairs_are_absent_from_queue": not reviewed_queue_overlap,
        "all_real_scores_recompute": score_recomputation_errors == 0,
        "all_real_support_annotations_recompute": support_annotation_errors == 0,
        "production_review_eligibility_recomputes": (
            production_eligibility_errors == 0
        ),
        "real_score_ranks_are_contiguous": rank_errors == 0,
        "real_score_count_matches_summary": len(scores) == real_summary["scored_candidate_count"],
        "queue_is_exact_eligible_top_n": observed_queue_uids == expected_queue_uids,
        "empty_internal_queue_keeps_full_stable_schema": (
            queue_header == expected_queue_header
        ),
        "queue_is_empty_valid_abstention": (
            not queue
            and real_summary["prospective_review_queue_count"] == 0
            and real_summary["review_queue_empty_is_valid_abstention"] is True
        ),
        "blind_packet_has_no_model_outputs": (
            blind_header == scorer.BLIND_PACKET_FIELDS
            and not any(
                fragment in column.lower()
                for column in blind_columns
                for fragment in blind_forbidden_fragments
            )
            and set(scorer.BLIND_OCCURRENCE_FIELDS)
            == {
                "source_dataset", "source_row_number", "source_market_raw",
                "source_field", "contact_type", "normalized_value", "raw_value",
                "title_snippet", "description_snippet",
            }
        ),
        "blind_packet_and_adjudication_match_queue": (
            {row["blind_id"] for row in blind}
            == {row["blind_id"] for row in adjudication}
            == {row["blind_id"] for row in queue}
            and blind_json_errors == 0
            and blind_occurrence_schema_errors == 0
        ),
        "real_rows_never_fit_or_gate_model": real_summary[
            "real_candidate_rows_used_for_model_fitting_selection_or_gating"
        ] == 0,
        "no_real_probability_or_performance_claim": (
            real_summary["model_score_is_real_probability"] is False
            and real_summary["real_performance_claim_allowed"] is False
        ),
    }
    decision = (
        "PASS_SYNTHETIC_REPLICATION_REAL_APPLICATION_ABSTENTION"
        if all(checks.values())
        else "FAIL"
    )
    payload = {
        "decision": decision,
        "checks": checks,
        "counts": {
            "synthetic_worlds": len(truth),
            "synthetic_items": len(items),
            "synthetic_parser_occurrences": len(parsed),
            "unique_feature_states_by_split": {
                split: values["unique_feature_state_count"]
                for split, values in feature_state_recomputed.items()
            },
            "known_reviewed_pair_uid_exclusions": len(reviewed),
            "real_candidates_after_exclusion": len(scores),
            "real_candidates_with_nonzero_identity_correction": sum(
                abs(float(row["identity_logit_correction"])) > 1e-15 for row in scores
            ),
            "real_candidates_with_positive_identity_correction": sum(
                float(row["identity_logit_correction"]) > 0.0 for row in scores
            ),
            "internal_review_queue": len(queue),
            "blind_evidence_packet": len(blind),
        },
        "recomputed_synthetic_audit_metrics": recomputed_audit_metrics,
        "recorded_block_permutation": permutation,
        "recomputed_cross_version_overlap": cross_version_overlap,
        "real_application": real_summary,
        "scientific_interpretation": {
            "supported": (
                "V12 is a corrected replication inside the predeclared synthetic generator family. "
                "Its primary audit retains every audit row, gives each observable state total "
                "weight one, and separately verifies model-discrimination and production-guard layers."
            ),
            "not_supported": (
                "It does not establish real same-identity accuracy, calibration, universal unseen-state "
                "generalization, or a new real identity link."
            ),
            "real_outcome": (
                "After excluding all historically reviewed pairs, the frozen model produced "
                "no positive identity correction and therefore correctly emitted no blind-review queue."
            ),
        },
        "failure_lineage": {
            "v5": "INVALID: English labels were coupled to synthetic labels during carrier selection.",
            "v6_v6_1": "WITHDRAWN: reused synthetic namespace, only 14 states, incomplete hard negatives, and all three queue rows were historically reviewed.",
            "v7": "NO_GO: block-imbalance permutation mean was 0.6439 and stable reuse was wrongly treated as identifiable.",
            "v8": "NO_GO: hard negative recipe gates still used raw binary calls instead of support-aware abstention.",
            "v9": "NO_GO: noisy/support-contaminated positive mechanisms were still wrongly required to be binary-positive.",
            "v10": "GO but superseded after the final audit required train-ambiguous states to be removed from development threshold selection.",
            "v11": "WITHDRAWN AS FINAL GO: audit labels removed 49 conflicting audit states, recipe gates mixed model success with guard abstention, and three dead features were retained.",
            "v12": "CURRENT SYNTHETIC REPLICATION: all audit states retained with equal state weight and gate layers separated."
        },
    }

    manifest_paths: list[Path] = []
    for path in [*policy_chain(training_policy_path), *policy_chain(application_policy_path)]:
        add_unique(manifest_paths, path)
    add_unique(
        manifest_paths,
        base.resolve("schema/step28_transferable_identity_history_v6_policy.json"),
        *prior_item_paths,
        *frozen_training_inputs,
        *frozen_application_inputs,
        training_root / train_outputs["world_truth"],
        training_root / train_outputs["synthetic_items"],
        training_root / train_outputs["parsed_occurrences"],
        training_root / train_outputs["model_inputs"],
        training_root / train_outputs["generation_summary"],
        training_root / train_outputs["model_artifacts"],
        training_root / train_outputs["synthetic_predictions"],
        training_root / train_outputs["training_summary"],
        application_root / app_outputs["real_candidate_scores"],
        application_root / app_outputs["prospective_review_queue"],
        application_root / app_outputs["blind_evidence_packet"],
        application_root / app_outputs["blind_adjudication_template"],
        application_root / app_outputs["real_scoring_summary"],
        base.resolve("scripts/step3_build_seller_profiles.py"),
        base.resolve("scripts/step28_common.py"),
        base.resolve("scripts/step28_history_common.py"),
        base.resolve("scripts/step28_generate_transferable_identity_histories.py"),
        base.resolve("scripts/step28_train_transferable_identity_model.py"),
        base.resolve("scripts/step28_score_real_identity_candidates.py"),
        base.resolve("scripts/step28_build_known_reviewed_pair_exclusion.py"),
        Path(__file__).resolve(),
        base.resolve("tests/test_step28_v4_transferable_identity_history.py"),
        base.resolve("tests/test_step28_v11_application_contracts.py"),
        base.resolve("tests/test_step28_v12_application_contracts.py"),
        base.resolve("docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_RESULT_20260720.zh.md"),
        base.resolve("docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_V6_REPAIR_RESULT_20260720.zh.md"),
        base.resolve("docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_V11_FINAL_AUDIT_20260720.zh.md"),
        base.resolve("docs/STEP28_TRANSFERABLE_IDENTITY_HISTORY_V12_CORRECTED_REPLICATION_20260720.zh.md"),
        base.resolve("docs/PROJECT_PROGRESS.md"),
        base.resolve("docs/AI_RESEARCH_HANDOFF_20260719.zh.md"),
    )
    missing = [str(path) for path in manifest_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Step28/v10 manifest is incomplete: " + ",".join(missing))
    return payload, manifest_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-policy", required=True)
    parser.add_argument("--application-policy", required=True)
    args = parser.parse_args()
    training_policy_path = base.resolve(args.training_policy)
    application_policy_path = base.resolve(args.application_policy)
    application_policy = history.load_policy(application_policy_path)
    payload, manifest_paths = audit(training_policy_path, application_policy_path)
    root = base.output_root(application_policy)
    outputs = application_policy["outputs"]
    self_audit_path = root / outputs["self_audit"]
    base.write_json_immutable(self_audit_path, payload)
    add_unique(manifest_paths, self_audit_path)
    manifest = {
        "step": 28,
        "run_id": application_policy["run_id"],
        "decision": payload["decision"],
        "complete": True,
        "artifact_count": len(manifest_paths),
        "artifacts": [base.file_record(path) for path in manifest_paths],
        "sync_manifest_self_excluded": True,
    }
    base.write_json_immutable(root / outputs["sync_manifest"], manifest)
    print(
        json.dumps(
            {"status": "ok", "decision": payload["decision"], **payload["counts"]},
            ensure_ascii=False,
        )
    )
    if payload["decision"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
