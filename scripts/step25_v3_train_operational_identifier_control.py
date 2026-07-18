#!/usr/bin/env python3
"""Train the separate English-only Step25-v3 occurrence identifier control."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step24_evaluate_content_independent_authorship as step24_eval
import step25_common as step25_v1
import step25_v3_common as common


def aligned_scores(path: Path, rows: list[dict], score_name: str) -> np.ndarray:
    records = step24.load_csv(path)
    index = {row["pair_uid"]: row for row in records}
    if len(index) != len(records):
        raise ValueError(f"Step25-v3 operational input contains duplicate pairs: {path}")
    if set(index) != {row["pair_uid"] for row in rows}:
        raise ValueError(f"Step25-v3 operational input boundary differs: {path}")
    return np.asarray([float(index[row["pair_uid"]][score_name]) for row in rows])


def mean_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(np.mean(values[mask])) if np.any(mask) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    (
        policy_path,
        policy,
        _v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    ) = common.load_policy(args.policy)
    operational = policy["operational_identifier_control"]
    cfg = step25_v1_policy["occurrence_reliability"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "role": operational["role"],
                    "training_domain": "en_only",
                    "chinese_labels_used_for_training": False,
                    "clean_model_selection_use": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    common.require_parent_manifests(policy)
    root = common.resolve(policy["outputs_root"])
    evaluation_path = root / policy["outputs"]["evaluation_summary"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("valid_or_test_rows_read_or_scored") != 0:
        raise ValueError("Step25-v3 operational control received a contaminated clean scorer")
    rows_by_pool = common.load_rows(step25_v2_policy, step24_policy, step25_v1_policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    en_prediction_path = root / policy["outputs"]["predictions_en"]
    zh_prediction_path = root / policy["outputs"]["predictions_zh"]
    en_scores = aligned_scores(
        en_prediction_path,
        en_rows,
        "english_oof_C2_copy_aware_dual_channel_primary",
    )
    zh_scores = aligned_scores(
        zh_prediction_path,
        zh_rows,
        "source_only_C2_copy_aware_dual_channel_primary",
    )
    indexes = step25_v1.occurrence_indexes(step25_v1_policy, step24_policy, rows_by_pool)
    frequency_threshold = int(cfg["public_identifier_train_seller_frequency_threshold"])
    en_evidence = step25_v1.occurrence_for_rows(en_rows, indexes, frequency_threshold)
    zh_evidence = step25_v1.occurrence_for_rows(zh_rows, indexes, frequency_threshold)
    actionable_states = set(cfg["actionable_training_states"])
    train_indices = np.asarray(
        [
            index
            for index, evidence in enumerate(en_evidence)
            if evidence["evidence_state"] in actionable_states
        ],
        dtype=int,
    )
    en_labels = v7_common.labels_array(en_rows)
    positive_count = int(np.sum(en_labels[train_indices] == 1.0))
    negative_count = int(np.sum(en_labels[train_indices] == 0.0))
    if positive_count < int(cfg["minimum_actionable_positive_rows"]):
        raise ValueError("Step25-v3 operational expert lacks actionable English positives")
    if negative_count < int(cfg["minimum_actionable_negative_rows"]):
        raise ValueError("Step25-v3 operational expert lacks actionable English negatives")
    en_matrix = step25_v1.reliability_feature_matrix(en_evidence, cfg["feature_names"])
    zh_matrix = step25_v1.reliability_feature_matrix(zh_evidence, cfg["feature_names"])
    artifact = step25_v1.fit_offset_reliability_expert(
        en_matrix[train_indices],
        en_labels[train_indices],
        en_scores[train_indices],
        cfg,
    )
    if not artifact["solver_converged"]:
        raise ValueError("Step25-v3 operational occurrence expert did not converge")
    corrections = step25_v1.reliability_corrections(zh_matrix, artifact)
    fused, decisions = step25_v1.apply_direction_constrained_reliability(
        zh_scores, zh_evidence, corrections
    )
    zh_labels = v7_common.labels_array(zh_rows)
    clean_metrics = step24_eval.metrics(zh_labels, zh_scores)
    fused_metrics = step24_eval.metrics(zh_labels, fused)
    state_counts = Counter(item["evidence_state"] for item in zh_evidence)
    state_metrics = {}
    for state in sorted(state_counts):
        mask = np.asarray([item["evidence_state"] == state for item in zh_evidence])
        labels = zh_labels[mask]
        record = {
            "row_count": int(np.sum(mask)),
            "positive_count": int(np.sum(labels == 1.0)),
            "negative_count": int(np.sum(labels == 0.0)),
            "clean_mean_score": float(np.mean(zh_scores[mask])),
            "fused_mean_score": float(np.mean(fused[mask])),
            "mean_applied_logit_correction": float(
                np.mean(
                    [
                        decisions[index]["applied_logit_correction"]
                        for index in np.flatnonzero(mask)
                    ]
                )
            ),
        }
        if len(np.unique(labels)) == 2:
            record["clean_metrics"] = step24_eval.metrics(labels, zh_scores[mask])
            record["fused_metrics"] = step24_eval.metrics(labels, fused[mask])
        state_metrics[state] = record
    direct_mask = np.asarray(
        [
            row["review_label"] == "positive"
            and row["evidence_type"]
            in {"same_controller_direct_identifier", "same_controller_component_anchor"}
            for row in zh_rows
        ]
    )
    public_mask = np.asarray(
        [
            row["review_label"] == "negative"
            and row["evidence_type"] == "public_contact_or_url_noise"
            for row in zh_rows
        ]
    )
    predictions = []
    for index, row in enumerate(zh_rows):
        evidence = zh_evidence[index]
        decision = decisions[index]
        predictions.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step25_pool"],
                "boundary": policy["boundary"]["name"],
                "component_id": row["step24_component_id"],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                "evidence_state": evidence["evidence_state"],
                "shared_identifier_types": ";".join(evidence["shared_identifier_types"]),
                "shared_token_hashes": ";".join(evidence["shared_token_hashes"]),
                "clean_source_only_score": f"{float(zh_scores[index]):.12f}",
                "raw_logit_correction": f"{float(decision['raw_logit_correction']):.12f}",
                "applied_logit_correction": f"{float(decision['applied_logit_correction']):.12f}",
                "expert_action": decision["expert_action"],
                "operational_fused_score": f"{float(fused[index]):.12f}",
            }
        )
    prediction_path = root / policy["outputs"]["operational_predictions_zh"]
    step24.write_csv_immutable(prediction_path, predictions)
    summary = {
        "step": "step25_v3_operational_identifier_control",
        "version": policy["version"],
        "status": "pass",
        "boundary": policy["boundary"]["name"],
        "role": operational["role"],
        "training_domain": "en_only",
        "chinese_labels_used_for_training": False,
        "identifier_features_entered_clean_scorer": False,
        "clean_model_selection_use": False,
        "d1_replication_candidate_eligibility_changed": False,
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
        "english_actionable_training_rows": {
            "row_count": int(len(train_indices)),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "state_counts": dict(
                sorted(
                    Counter(
                        en_evidence[index]["evidence_state"] for index in train_indices
                    ).items()
                )
            ),
        },
        "chinese_state_counts": dict(sorted(state_counts.items())),
        "clean_metrics": clean_metrics,
        "operational_fused_metrics": fused_metrics,
        "delta_average_precision": float(
            fused_metrics["average_precision"] - clean_metrics["average_precision"]
        ),
        "state_metrics": state_metrics,
        "sensitivity": {
            "direct_component_positive_count": int(np.sum(direct_mask)),
            "direct_component_clean_mean": mean_or_none(zh_scores, direct_mask),
            "direct_component_fused_mean": mean_or_none(fused, direct_mask),
            "public_noise_negative_count": int(np.sum(public_mask)),
            "public_noise_clean_mean": mean_or_none(zh_scores, public_mask),
            "public_noise_fused_mean": mean_or_none(fused, public_mask),
        },
        "artifact": artifact,
        "state_actions": cfg["state_actions"],
        "input_provenance": {
            "evaluation_summary_sha256": step24.sha256_file(evaluation_path),
            "english_oof_predictions_sha256": step24.sha256_file(en_prediction_path),
            "chinese_source_predictions_sha256": step24.sha256_file(zh_prediction_path),
            "occurrence_inputs": {
                pool_name: {
                    "path": step24_policy["pools"][pool_name]["item_identity_signals"],
                    "sha256": step24.sha256_file(
                        common.resolve(
                            step24_policy["pools"][pool_name]["item_identity_signals"]
                        )
                    ),
                }
                for pool_name in rows_by_pool
            },
        },
        "outputs": {
            "predictions": str(prediction_path.relative_to(common.ROOT)).replace("\\", "/")
        },
        "policy_sha256": step24.sha256_file(policy_path),
        "predictions_sha256": step24.sha256_file(prediction_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = root / policy["outputs"]["operational_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "delta_average_precision": summary["delta_average_precision"],
                "publication_promotion_eligible": False,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
