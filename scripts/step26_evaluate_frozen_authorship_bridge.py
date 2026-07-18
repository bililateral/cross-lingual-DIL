#!/usr/bin/env python3
"""Evaluate frozen Step24 source scorers on the exact corrected Step15-v8 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import step24_common as step24
import step26_common as common


DIRECT_TYPES = {
    "same_controller_direct_identifier",
    "same_controller_component_anchor",
}
SOFT_TYPES = {"same_controller_style_structural_soft"}
NEGATIVE_TYPES = {
    "ordinary_negative",
    "public_contact_or_url_noise",
    "semantic_topic_not_controller",
    "template_clone_not_controller",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("Step26 ranking metrics require a non-empty binary slice")
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "row_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc": float(auc(recall, precision)),
    }


def average_precision_metric(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def apply_frozen_artifact(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    standardization = artifact["standardization"]
    mean = np.asarray(standardization["mean"], dtype=float)
    scale = np.asarray(standardization["scale"], dtype=float)
    coefficients = np.asarray(artifact["parameter_coefficients"], dtype=float)
    if matrix.shape[1] != len(mean) or len(mean) != len(coefficients):
        raise ValueError("Step26 frozen artifact feature dimensions disagree")
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("Step26 frozen artifact has an invalid scale")
    standardized = (matrix - mean) / scale
    logits = float(artifact["parameter_intercept"]) + standardized @ coefficients
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def load_evaluation_feature_caches(policy: dict) -> dict[str, tuple[dict[str, int], np.ndarray]]:
    output_root = common.resolve(policy["outputs_root"])
    data_cfg = policy["evaluation_data"]
    e5_index, e5_matrix, e5_metadata = step24.load_normalized_cache(
        common.resolve(data_cfg["identifier_redacted_e5_metadata"]),
        common.resolve(data_cfg["identifier_redacted_e5_matrix"]),
    )
    if e5_metadata.get("identifier_redacted") is not True:
        raise ValueError("Step26 E5 cache is not identifier-redacted")
    caches = {"identifier_redacted_e5_cosine": (e5_index, e5_matrix)}
    for encoder_key, feature_name in (
        ("pcm_multilingual_authorship", "pcm_multilingual_authorship_cosine"),
        ("mstyledistance", "mstyledistance_cosine"),
    ):
        stem = output_root / "embeddings" / f"{encoder_key}.evaluation"
        index, matrix, metadata = step24.load_normalized_cache(
            Path(f"{stem}.json"), Path(f"{stem}.npy")
        )
        if metadata.get("identifier_redacted") is not True:
            raise ValueError(f"Step26 style cache is not identifier-redacted: {encoder_key}")
        if metadata.get("encoder_parameters_updated") is not False:
            raise ValueError(f"Step26 style encoder was updated: {encoder_key}")
        caches[feature_name] = (index, matrix)
    return caches


def pair_cosine(cache: tuple[dict[str, int], np.ndarray], left: str, right: str) -> float:
    index, matrix = cache
    if left not in index or right not in index:
        missing = left if left not in index else right
        raise ValueError(f"Step26 seller is absent from an embedding cache: {missing}")
    value = float(np.dot(matrix[index[left]], matrix[index[right]]))
    if not np.isfinite(value):
        raise ValueError("Step26 pair cosine is non-finite")
    return value


def slice_rows(rows: list[dict], positive_types: set[str]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if row["review_label"] == "negative" or row["evidence_type"] in positive_types
    ]


def tail_summary(rows: list[dict], scores: np.ndarray, evidence_type: str) -> dict:
    indexes = [
        index
        for index, row in enumerate(rows)
        if row["review_label"] == "negative" and row["evidence_type"] == evidence_type
    ]
    if not indexes:
        return {
            "count": 0,
            "mean_score": None,
            "q95_score": None,
            "maximum_score": None,
        }
    values = scores[np.asarray(indexes, dtype=int)]
    return {
        "count": len(indexes),
        "mean_score": float(np.mean(values)),
        "q95_score": float(np.quantile(values, 0.95)),
        "maximum_score": float(np.max(values)),
    }


def top_budget_intrusions(rows: list[dict], scores: np.ndarray) -> dict[str, int]:
    positive_count = sum(row["review_label"] == "positive" for row in rows)
    ranked = np.argsort(-scores, kind="stable")[:positive_count]
    counts = {evidence_type: 0 for evidence_type in NEGATIVE_TYPES}
    for index in ranked:
        row = rows[int(index)]
        if row["review_label"] == "negative" and row["evidence_type"] in counts:
            counts[row["evidence_type"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    policy_path, policy, _ = common.load_policy(args.policy)
    frozen = common.validate_frozen_sources(policy)
    if args.validate_config_only:
        rows_by_split = common.load_evaluation_rows(policy)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "split_counts": {key: len(value) for key, value in rows_by_split.items()},
                    "source_artifact_keys": policy["frozen_models"]["source_artifact_keys"],
                    "model_refit_performed": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    caches = load_evaluation_feature_caches(policy)
    source_artifacts = frozen["step24_artifacts"]["artifacts"]["source_only"]
    required = policy["frozen_models"]["source_artifact_keys"]
    if any(key not in source_artifacts for key in required):
        raise ValueError("Step26 source artifact is missing")
    feature_names = [
        "identifier_redacted_e5_cosine",
        "pcm_multilingual_authorship_cosine",
        "mstyledistance_cosine",
    ]
    output_root = common.resolve(policy["outputs_root"])
    pair_feature_rows = []
    prediction_rows = []
    model_metric_rows = []
    slice_metric_rows = []
    score_index: dict[tuple[str, str], np.ndarray] = {}
    model_order = [
        "raw_identifier_redacted_e5_cosine",
        "raw_pcm_multilingual_authorship_cosine",
        "raw_mstyledistance_cosine",
        "source_only_e5_lr_l2_control",
        "source_only_style_only_lr_l2_control",
        "source_only_semantic_style_lr_l2_primary",
        "step15_v8_b0",
        "step15_v8_clean",
        "step15_v8_contextual",
    ]
    # Complete every frozen Step24 score before joining any Chinese label/evidence field.
    blind_allowlists = common.load_blind_pair_allowlists(policy)
    blind_features = {}
    blind_source_scores = {}
    for split_name, pair_uids in blind_allowlists.items():
        feature_matrix = np.zeros((len(pair_uids), 3), dtype=float)
        for row_index, pair_uid in enumerate(pair_uids):
            left, right = common.pair_uid_sellers(pair_uid)
            feature_matrix[row_index] = [
                pair_cosine(caches[name], left, right) for name in feature_names
            ]
        blind_features[split_name] = feature_matrix
        source_scores = {
            model_order[0]: feature_matrix[:, 0],
            model_order[1]: feature_matrix[:, 1],
            model_order[2]: feature_matrix[:, 2],
        }
        for artifact_key, model_id in (
            ("e5_lr_l2_control", model_order[3]),
            ("style_only_lr_l2_control", model_order[4]),
            ("semantic_style_lr_l2_primary", model_order[5]),
        ):
            record = source_artifacts[artifact_key]
            columns = [feature_names.index(name) for name in record["feature_names"]]
            source_scores[model_id] = apply_frozen_artifact(
                feature_matrix[:, columns], record["logistic_artifact"]
            )
        blind_source_scores[split_name] = source_scores

    rows_by_split = common.load_evaluation_rows(policy)
    for split_name, rows in rows_by_split.items():
        pair_uids = [row["pair_uid"] for row in rows]
        if pair_uids != blind_allowlists[split_name]:
            raise ValueError(f"Step26 blind-score and labeled pair order differs: {split_name}")
        feature_matrix = blind_features[split_name]
        for row_index, row in enumerate(rows):
            left = row["seller_uid_left"]
            right = row["seller_uid_right"]
            values = feature_matrix[row_index].tolist()
            pair_feature_rows.append(
                {
                    "step26_split": split_name,
                    "pair_uid": row["pair_uid"],
                    "seller_uid_left": left,
                    "seller_uid_right": right,
                    "review_label": row["review_label"],
                    "evidence_type": row["evidence_type"],
                    "v7_component_id": row["v7_component_id"],
                    **{name: f"{value:.12f}" for name, value in zip(feature_names, values)},
                }
            )
        scores = dict(blind_source_scores[split_name])
        scores[model_order[6]] = np.asarray(
            [float(row["v8_b0_prob_positive"]) for row in rows], dtype=float
        )
        scores[model_order[7]] = np.asarray(
            [float(row["v8_clean_prob_positive"]) for row in rows], dtype=float
        )
        scores[model_order[8]] = np.asarray(
            [float(row["v8_contextual_prob_positive"]) for row in rows], dtype=float
        )
        labels = common.labels_array(rows)
        for model_id in model_order:
            model_scores = np.asarray(scores[model_id], dtype=float)
            if len(model_scores) != len(rows) or not np.all(np.isfinite(model_scores)):
                raise ValueError(f"Step26 invalid model scores: {split_name}:{model_id}")
            score_index[(split_name, model_id)] = model_scores
            metrics = ranking_metrics(labels, model_scores)
            model_metric_rows.append(
                {"step26_split": split_name, "model_id": model_id, **metrics}
            )
            direct_indexes = np.asarray(slice_rows(rows, DIRECT_TYPES), dtype=int)
            soft_indexes = np.asarray(slice_rows(rows, SOFT_TYPES), dtype=int)
            for slice_name, indexes in (
                ("direct_component_positive_plus_all_negatives", direct_indexes),
                ("soft_positive_plus_all_negatives", soft_indexes),
            ):
                slice_metrics = ranking_metrics(labels[indexes], model_scores[indexes])
                slice_metric_rows.append(
                    {
                        "step26_split": split_name,
                        "model_id": model_id,
                        "slice_name": slice_name,
                        **slice_metrics,
                        "mean_score": "",
                        "q95_score": "",
                        "maximum_score": "",
                        "top_positive_budget_intrusion_count": "",
                    }
                )
            intrusions = top_budget_intrusions(rows, model_scores)
            for evidence_type in sorted(NEGATIVE_TYPES):
                tail = tail_summary(rows, model_scores, evidence_type)
                slice_metric_rows.append(
                    {
                        "step26_split": split_name,
                        "model_id": model_id,
                        "slice_name": evidence_type,
                        "row_count": tail["count"],
                        "positive_count": 0,
                        "negative_count": tail["count"],
                        "roc_auc": "",
                        "average_precision": "",
                        "pr_auc": "",
                        "mean_score": tail["mean_score"],
                        "q95_score": tail["q95_score"],
                        "maximum_score": tail["maximum_score"],
                        "top_positive_budget_intrusion_count": intrusions[evidence_type],
                    }
                )
            for row, score in zip(rows, model_scores, strict=True):
                prediction_rows.append(
                    {
                        "step26_split": split_name,
                        "pair_uid": row["pair_uid"],
                        "review_label": row["review_label"],
                        "evidence_type": row["evidence_type"],
                        "v7_component_id": row["v7_component_id"],
                        "model_id": model_id,
                        "probability_or_similarity_score": f"{score:.12f}",
                        "threshold_status": "not_used_for_ranking_evaluation",
                    }
                )

    valid_name = "representative_valid"
    valid_rows = rows_by_split[valid_name]
    primary_id = policy["frozen_models"]["primary_bridge_model"]
    baseline_id = "step15_v8_clean"
    primary = score_index[(valid_name, primary_id)]
    baseline = score_index[(valid_name, baseline_id)]
    bootstrap_cfg = policy["evaluation"]
    bootstrap = common.grouped_bootstrap_delta(
        valid_rows,
        baseline,
        primary,
        average_precision_metric,
        int(bootstrap_cfg["component_grouped_bootstrap_resamples"]),
        int(bootstrap_cfg["component_grouped_bootstrap_seed"]),
    )
    bootstrap_rows = [
        {
            "step26_split": valid_name,
            "candidate_model_id": primary_id,
            "baseline_model_id": baseline_id,
            "metric": "average_precision",
            **bootstrap,
        }
    ]
    labels = common.labels_array(valid_rows)
    direct_indexes = np.asarray(slice_rows(valid_rows, DIRECT_TYPES), dtype=int)
    soft_indexes = np.asarray(slice_rows(valid_rows, SOFT_TYPES), dtype=int)
    direct_delta = float(
        average_precision_metric(labels[direct_indexes], primary[direct_indexes])
        - average_precision_metric(labels[direct_indexes], baseline[direct_indexes])
    )
    soft_delta = float(
        average_precision_metric(labels[soft_indexes], primary[soft_indexes])
        - average_precision_metric(labels[soft_indexes], baseline[soft_indexes])
    )
    primary_intrusions = top_budget_intrusions(valid_rows, primary)
    baseline_intrusions = top_budget_intrusions(valid_rows, baseline)
    gates_cfg = policy["promotion_gates"]
    gate_records = {
        "primary_valid_ap_gain": {
            "observed": bootstrap["point_delta"],
            "required_minimum": gates_cfg["primary_valid_ap_gain_over_v8_clean_minimum"],
            "passed": bootstrap["point_delta"]
            >= float(gates_cfg["primary_valid_ap_gain_over_v8_clean_minimum"]),
        },
        "paired_bootstrap_ci_lower": {
            "observed": bootstrap["ci95_lower"],
            "required_minimum": gates_cfg[
                "primary_valid_ap_delta_bootstrap_ci_lower_minimum"
            ],
            "passed": bootstrap["ci95_lower"]
            >= float(gates_cfg["primary_valid_ap_delta_bootstrap_ci_lower_minimum"]),
        },
        "direct_component_ap_delta": {
            "observed": direct_delta,
            "required_minimum": gates_cfg["direct_component_valid_ap_delta_minimum"],
            "passed": direct_delta
            >= float(gates_cfg["direct_component_valid_ap_delta_minimum"]),
        },
        "soft_positive_ap_delta": {
            "observed": soft_delta,
            "required_minimum": gates_cfg["soft_positive_valid_ap_delta_minimum"],
            "passed": soft_delta >= float(gates_cfg["soft_positive_valid_ap_delta_minimum"]),
        },
        "template_top_budget_intrusion_increase": {
            "observed": primary_intrusions["template_clone_not_controller"]
            - baseline_intrusions["template_clone_not_controller"],
            "required_maximum": gates_cfg[
                "template_top_budget_intrusion_increase_maximum"
            ],
        },
        "public_noise_top_budget_intrusion_increase": {
            "observed": primary_intrusions["public_contact_or_url_noise"]
            - baseline_intrusions["public_contact_or_url_noise"],
            "required_maximum": gates_cfg[
                "public_noise_top_budget_intrusion_increase_maximum"
            ],
        },
    }
    for name in (
        "template_top_budget_intrusion_increase",
        "public_noise_top_budget_intrusion_increase",
    ):
        gate_records[name]["passed"] = gate_records[name]["observed"] <= float(
            gate_records[name]["required_maximum"]
        )
    eligible = all(record["passed"] for record in gate_records.values())

    pair_manifest_rows = [
        {
            "step26_split": split_name,
            "pair_uid": row["pair_uid"],
            "seller_uid_left": row["seller_uid_left"],
            "seller_uid_right": row["seller_uid_right"],
            "review_label": row["review_label"],
            "evidence_type": row["evidence_type"],
            "v7_component_id": row["v7_component_id"],
            "selection_role": "internal_bridge_gate"
            if split_name == valid_name
            else "diagnostic_only_no_gate",
        }
        for split_name, rows in rows_by_split.items()
        for row in rows
    ]
    outputs = policy["outputs"]
    common.write_csv_immutable(output_root / outputs["evaluation_pair_manifest"], pair_manifest_rows)
    common.write_csv_immutable(output_root / outputs["pair_features"], pair_feature_rows)
    common.write_csv_immutable(output_root / outputs["predictions"], prediction_rows)
    common.write_csv_immutable(output_root / outputs["model_metrics"], model_metric_rows)
    common.write_csv_immutable(output_root / outputs["slice_metrics"], slice_metric_rows)
    common.write_csv_immutable(output_root / outputs["bootstrap_comparisons"], bootstrap_rows)
    summary = {
        "step": "step26_frozen_authorship_bridge",
        "version": policy["version"],
        "status": "pass",
        "scientific_question": policy["objective"],
        "split_counts": {
            split_name: {
                "rows": len(rows),
                "positive": int(common.labels_array(rows).sum()),
                "negative": int(len(rows) - common.labels_array(rows).sum()),
            }
            for split_name, rows in rows_by_split.items()
        },
        "primary_bridge_model": primary_id,
        "matched_gate_baseline": baseline_id,
        "model_refit_performed": False,
        "encoder_parameters_updated": False,
        "chinese_valid_test_used_for_model_or_threshold_selection": False,
        "internal_test_metrics_used_for_promotion": False,
        "representative_valid_gate": {
            "eligible_for_one_step26b_experiment": eligible,
            "gates": gate_records,
            "paired_bootstrap": bootstrap,
        },
        "publication_claim_allowed": False,
        "publication_claim_requires": "Step20 genuinely prospective holdout after configuration freeze",
        "source_hashes": frozen["hashes"],
        "output_paths": {
            key: str((output_root / value).relative_to(common.ROOT)).replace("\\", "/")
            for key, value in outputs.items()
            if key != "sync_manifest"
        },
        "policy_sha256": common.sha256(policy_path),
        "producer_sha256": common.sha256(Path(__file__).resolve()),
    }
    summary["summary_payload_sha256"] = common.canonical_hash(summary)
    common.write_json_immutable(output_root / outputs["evaluation_summary"], summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "eligible_for_step26b": eligible,
                "valid_primary_ap_delta": bootstrap["point_delta"],
                "valid_primary_ap_delta_ci95": [bootstrap["ci95_lower"], bootstrap["ci95_upper"]],
                "summary": summary["output_paths"]["evaluation_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
