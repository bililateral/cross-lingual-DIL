#!/usr/bin/env python3
"""Train the independent Step25 source-domain occurrence reliability expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step24_evaluate_content_independent_authorship as step24_eval
import step25_common as common


def aligned_scores(path: Path, rows: list[dict], score_name: str) -> np.ndarray:
    records = step24.load_csv(path)
    index = {row["pair_uid"]: row for row in records}
    if len(index) != len(records):
        raise ValueError(f"Step25 reliability prediction input has duplicate pairs: {path}")
    missing = [row["pair_uid"] for row in rows if row["pair_uid"] not in index]
    if missing:
        raise ValueError(f"Step25 reliability input misses a pair: {missing[0]}")
    return np.asarray([float(index[row["pair_uid"]][score_name]) for row in rows], dtype=float)


def mean_for_mask(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(np.mean(values[mask])) if np.any(mask) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    cfg = policy["occurrence_reliability"]
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "training_domain": cfg["training_domain"],
                    "role": cfg["role"],
                    "feature_names": cfg["feature_names"],
                    "review_label_used_as_feature": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    output_root = common.resolve(policy["outputs_root"])
    rows_by_pool = common.load_rows(policy, step24_policy)
    en_rows = rows_by_pool["en_content_train_pool"]
    zh_rows = rows_by_pool["zh_target_strict"]
    en_clean_oof = aligned_scores(
        output_root / policy["outputs"]["oof_predictions_en"],
        en_rows,
        "source_oof_decontaminated_style_lr_l2_primary",
    )
    zh_clean_source = aligned_scores(
        output_root / policy["outputs"]["oof_predictions_zh"],
        zh_rows,
        "source_only_decontaminated_style_lr_l2_primary",
    )
    indexes = common.occurrence_indexes(policy, step24_policy, rows_by_pool)
    threshold = int(cfg["public_identifier_train_seller_frequency_threshold"])
    en_evidence = common.occurrence_for_rows(en_rows, indexes, threshold)
    zh_evidence = common.occurrence_for_rows(zh_rows, indexes, threshold)
    actionable = set(cfg["actionable_training_states"])
    train_indices = np.asarray(
        [index for index, item in enumerate(en_evidence) if item["evidence_state"] in actionable],
        dtype=int,
    )
    en_labels = v7_common.labels_array(en_rows)
    positive_count = int(np.sum(en_labels[train_indices] == 1.0))
    negative_count = int(np.sum(en_labels[train_indices] == 0.0))
    if positive_count < int(cfg["minimum_actionable_positive_rows"]):
        raise ValueError(
            f"Step25 occurrence expert has too few actionable English positives: {positive_count}"
        )
    if negative_count < int(cfg["minimum_actionable_negative_rows"]):
        raise ValueError(
            f"Step25 occurrence expert has too few actionable English negatives: {negative_count}"
        )
    en_matrix = common.reliability_feature_matrix(en_evidence, cfg["feature_names"])
    zh_matrix = common.reliability_feature_matrix(zh_evidence, cfg["feature_names"])
    artifact = common.fit_offset_reliability_expert(
        en_matrix[train_indices],
        en_labels[train_indices],
        en_clean_oof[train_indices],
        cfg,
    )
    if not artifact["solver_converged"]:
        raise ValueError("Step25 occurrence reliability expert did not converge")
    raw_corrections = common.reliability_corrections(zh_matrix, artifact)
    fused, decisions = common.apply_direction_constrained_reliability(
        zh_clean_source, zh_evidence, raw_corrections
    )
    zh_labels = v7_common.labels_array(zh_rows)
    clean_metrics = step24_eval.metrics(zh_labels, zh_clean_source)
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
            "clean_mean_score": float(np.mean(zh_clean_source[mask])),
            "fused_mean_score": float(np.mean(fused[mask])),
            "mean_applied_logit_correction": float(
                np.mean([decisions[index]["applied_logit_correction"] for index in np.where(mask)[0]])
            ),
        }
        if len(np.unique(labels)) == 2:
            record["clean_metrics"] = step24_eval.metrics(labels, zh_clean_source[mask])
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
    template_mask = np.asarray(
        [
            row["review_label"] == "negative"
            and row["evidence_type"] == "template_clone_not_controller"
            for row in zh_rows
        ]
    )
    prediction_rows = []
    for index, row in enumerate(zh_rows):
        evidence = zh_evidence[index]
        decision = decisions[index]
        prediction_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "pool": row["step25_pool"],
                "boundary": "d0_current_canonical_train",
                "component_id": row["step25_component_id"],
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "silver_train_only": row.get("silver_train_only", ""),
                "evidence_state": evidence["evidence_state"],
                "shared_identifier_types": ";".join(evidence["shared_identifier_types"]),
                "shared_token_hashes": ";".join(evidence["shared_token_hashes"]),
                "clean_source_only_score": f"{float(zh_clean_source[index]):.12f}",
                "raw_logit_correction": f"{float(decision['raw_logit_correction']):.12f}",
                "applied_logit_correction": f"{float(decision['applied_logit_correction']):.12f}",
                "expert_action": decision["expert_action"],
                "reliability_fused_score": f"{float(fused[index]):.12f}",
            }
        )
    predictions_path = output_root / policy["outputs"]["reliability_predictions"]
    step24.write_csv_immutable(predictions_path, prediction_rows)
    evaluation_summary_path = output_root / policy["outputs"]["evaluation_summary"]
    en_prediction_input_path = output_root / policy["outputs"]["oof_predictions_en"]
    zh_prediction_input_path = output_root / policy["outputs"]["oof_predictions_zh"]
    occurrence_input_records = {
        pool_name: {
            "path": step24_policy["pools"][pool_name]["item_identity_signals"],
            "sha256": step24.sha256_file(
                common.resolve(step24_policy["pools"][pool_name]["item_identity_signals"])
            ),
        }
        for pool_name in rows_by_pool
    }
    summary = {
        "step": "step25_occurrence_reliability",
        "version": policy["version"],
        "status": "pass",
        "boundary": "d0_current_canonical_train",
        "role": cfg["role"],
        "training_domain": "en",
        "target_domain_labels_used_for_training": False,
        "clean_probability_training_source": cfg["clean_probability_source"],
        "review_label_evidence_type_or_split_membership_used_as_feature": False,
        "publication_promotion_eligible": False,
        "english_actionable_training_rows": {
            "row_count": int(len(train_indices)),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "state_counts": dict(
                sorted(Counter(en_evidence[index]["evidence_state"] for index in train_indices).items())
            ),
        },
        "chinese_state_counts": dict(sorted(state_counts.items())),
        "clean_metrics": clean_metrics,
        "reliability_fused_metrics": fused_metrics,
        "delta_average_precision": float(
            fused_metrics["average_precision"] - clean_metrics["average_precision"]
        ),
        "state_metrics": state_metrics,
        "sensitivity": {
            "direct_component_positive_count": int(np.sum(direct_mask)),
            "direct_component_clean_mean": mean_for_mask(zh_clean_source, direct_mask),
            "direct_component_fused_mean": mean_for_mask(fused, direct_mask),
            "public_noise_negative_count": int(np.sum(public_mask)),
            "public_noise_clean_mean": mean_for_mask(zh_clean_source, public_mask),
            "public_noise_fused_mean": mean_for_mask(fused, public_mask),
            "template_clone_negative_count": int(np.sum(template_mask)),
            "template_clone_clean_mean": mean_for_mask(zh_clean_source, template_mask),
            "template_clone_fused_mean": mean_for_mask(fused, template_mask),
        },
        "artifact": artifact,
        "state_actions": cfg["state_actions"],
        "input_provenance": {
            "evaluation_summary_sha256": step24.sha256_file(evaluation_summary_path),
            "english_oof_predictions_sha256": step24.sha256_file(
                en_prediction_input_path
            ),
            "chinese_source_predictions_sha256": step24.sha256_file(
                zh_prediction_input_path
            ),
            "occurrence_inputs": occurrence_input_records,
        },
        "outputs": {
            "predictions": str(predictions_path.relative_to(common.ROOT)).replace("\\", "/")
        },
        "policy_sha256": step24.sha256_file(policy_path),
        "predictions_sha256": step24.sha256_file(predictions_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    summary["summary_sha256"] = step24.canonical_hash(summary)
    summary_path = output_root / policy["outputs"]["reliability_summary"]
    step24.write_json_immutable(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "english_actionable_training_rows": summary[
                    "english_actionable_training_rows"
                ],
                "delta_average_precision": summary["delta_average_precision"],
                "publication_promotion_eligible": False,
                "summary": str(summary_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
