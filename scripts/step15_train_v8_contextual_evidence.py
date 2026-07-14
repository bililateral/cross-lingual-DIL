#!/usr/bin/env python3
"""Train the Step15-v8 occurrence-level, direction-constrained evidence expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7
import step15_v7_common as v7
import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


OUTPUT_FIELDS = [
    "pair_uid",
    "step15_pool",
    "split_name",
    "review_label",
    "evidence_type",
    "v7_component_id",
    "clean_prob_positive",
    "contextual_evidence_prob_positive",
    "clean_selected_threshold",
    "clean_predicted_label",
    "contextual_evidence_predicted_label_at_clean_threshold",
    "evidence_state",
    "expert_action",
    "raw_logit_correction",
    "applied_logit_correction",
    "shared_token_count",
    "verified_direct_token_count",
    "risky_only_token_count",
    "support_only_token_count",
    "mixed_context_token_count",
    "high_frequency_token_count",
]


def load_aligned_scores(path: Path, rows: list[dict]) -> np.ndarray:
    indexed = {row["pair_uid"]: row for row in common.load_csv(path)}
    if set(indexed) != {row["pair_uid"] for row in rows}:
        missing = sorted({row["pair_uid"] for row in rows} - set(indexed))
        extra = sorted(set(indexed) - {row["pair_uid"] for row in rows})
        raise ValueError(
            f"Step15-v8 prediction alignment failed: {path}; missing={missing[:1]} extra={extra[:1]}"
        )
    scores = []
    for row in rows:
        prediction = indexed[row["pair_uid"]]
        if prediction.get("review_label") != row["review_label"]:
            raise ValueError(f"Prediction label mismatch: {path}:{row['pair_uid']}")
        if prediction.get("v7_component_id") != row["v7_component_id"]:
            raise ValueError(f"Prediction component mismatch: {path}:{row['pair_uid']}")
        scores.append(float(prediction["prob_positive"]))
    return np.asarray(scores, dtype=float)


def occurrence_for_rows(
    rows: list[dict],
    indexes: dict[str, tuple[dict, Counter]],
    threshold: int,
) -> list[dict]:
    output = []
    for row in rows:
        by_seller, token_df = indexes[row["step15_pool"]]
        output.append(common.occurrence_evidence(row, by_seller, token_df, threshold))
    return output


def output_rows(
    rows: list[dict],
    clean: np.ndarray,
    fused: np.ndarray,
    evidence: list[dict],
    decisions: list[dict],
    threshold: float,
    split_name: str,
) -> list[dict]:
    result = []
    for row, clean_score, fused_score, item, decision in zip(
        rows, clean, fused, evidence, decisions, strict=True
    ):
        result.append(
            {
                "pair_uid": row["pair_uid"],
                "step15_pool": row["step15_pool"],
                "split_name": split_name,
                "review_label": row["review_label"],
                "evidence_type": row["evidence_type"],
                "v7_component_id": row["v7_component_id"],
                "clean_prob_positive": f"{float(clean_score):.12f}",
                "contextual_evidence_prob_positive": f"{float(fused_score):.12f}",
                "clean_selected_threshold": f"{float(threshold):.12f}",
                "clean_predicted_label": int(float(clean_score) >= threshold),
                "contextual_evidence_predicted_label_at_clean_threshold": int(
                    float(fused_score) >= threshold
                ),
                "evidence_state": item["evidence_state"],
                "expert_action": decision["expert_action"],
                "raw_logit_correction": f"{decision['raw_logit_correction']:.12f}",
                "applied_logit_correction": f"{decision['applied_logit_correction']:.12f}",
                "shared_token_count": item["shared_token_count"],
                "verified_direct_token_count": item["verified_direct_token_count"],
                "risky_only_token_count": item["risky_only_token_count"],
                "support_only_token_count": item["support_only_token_count"],
                "mixed_context_token_count": item["mixed_context_token_count"],
                "high_frequency_token_count": item["high_frequency_token_count"],
            }
        )
    return result


def slice_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    result = {}
    for evidence_type in sorted({row["evidence_type"] for row in rows}):
        mask = np.asarray([row["evidence_type"] == evidence_type for row in rows], dtype=bool)
        y_true = v7.labels_array([row for row in rows if row["evidence_type"] == evidence_type])
        result[evidence_type] = {
            "count": int(np.sum(mask)),
            "positive": int(np.sum(y_true)),
            "negative": int(len(y_true) - np.sum(y_true)),
            "false_positive_rate": common.false_positive_rate(y_true, scores[mask], threshold),
            "recall": common.recall_at_threshold(y_true, scores[mask], threshold),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    if args.validate_config_only:
        print(json.dumps(validation, indent=2))
        return
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    bridge_root = root / policy["bridge_audit"]["output_subdirectory"]
    bridge_summary_path = bridge_root / "step15_v8_bridge_audit_summary.json"
    if not bridge_summary_path.is_file():
        raise FileNotFoundError("Step15-v8 bridge audit must complete before evidence training")
    bridge = common.load_json(bridge_summary_path)
    if bridge["selection"]["representative_valid_metrics_used_for_selection"] is not False:
        raise ValueError("Bridge representation selection consumed representative validation")
    if bridge["selection"]["internal_test_metrics_used_for_selection"] is not False:
        raise ValueError("Bridge representation selection consumed the internal test")

    final_root = root / policy["occurrence_evidence_expert"]["output_subdirectory"]
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step15-v8 contextual evidence: {final_root} / {staging_root}"
        )
    rows_by_pool = common.load_joined_rows(policy, v7_policy, root)
    splits = common.split_rows(rows_by_pool)
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    test_rows = splits["internal_development_test"]
    train_sellers_by_pool = {
        pool_name: {
            str(row[key])
            for row in rows
            if row["v7_split_name"] == "train"
            for key in ("seller_uid_left", "seller_uid_right")
        }
        for pool_name, rows in rows_by_pool.items()
    }
    indexes = {
        pool_name: common.item_signal_index(
            common.resolve(policy["pools"][pool_name]["item_identity_signals"]),
            train_sellers_by_pool[pool_name],
        )
        for pool_name in policy["pools"]
    }
    frequency_threshold = int(
        policy["occurrence_evidence_expert"][
            "public_identifier_train_seller_frequency_threshold"
        ]
    )
    train_evidence = occurrence_for_rows(train_rows, indexes, frequency_threshold)
    valid_evidence = occurrence_for_rows(valid_rows, indexes, frequency_threshold)
    test_evidence = occurrence_for_rows(test_rows, indexes, frequency_threshold)
    train_x = common.evidence_feature_matrix(train_rows, train_evidence, policy)
    valid_x = common.evidence_feature_matrix(valid_rows, valid_evidence, policy)
    test_x = common.evidence_feature_matrix(test_rows, test_evidence, policy)
    actionable_states = {
        "verified_direct_both_sides",
        "risky_only_shared",
        "support_only_shared",
        "high_frequency_public",
    }
    actionable = np.asarray(
        [item["evidence_state"] in actionable_states for item in train_evidence], dtype=bool
    )
    if set(v7.labels_array([row for row, keep in zip(train_rows, actionable, strict=True) if keep])) != {
        0.0,
        1.0,
    }:
        raise ValueError("Occurrence evidence expert actionable train rows lack both labels")
    base_weights, weight_diagnostics = v7.factorized_evidence_weights(
        train_rows, v7_policy["factorized_evidence_weighting"]
    )
    seeds = [int(value) for value in policy["bridge_audit"]["seeds"]]
    selected_feature_set = bridge["selection"]["feature_representation"][
        "selected_feature_set_id"
    ]
    selected_family = bridge["selection"]["model_family"]["selected_model_family"]
    staging_root.mkdir(parents=True, exist_ok=False)
    seed_records = []
    clean_matrices = {"valid": [], "internal_development_test": []}
    fused_matrices = {"valid": [], "internal_development_test": []}

    for seed in seeds:
        oof_record = next(
            row
            for row in bridge["oof_records"]
            if row["feature_set_id"] == selected_feature_set
            and row["model_family"] == selected_family
            and int(row["seed"]) == seed
        )
        final_record = next(
            row
            for row in bridge["final_seed_records"]
            if row["output_id"] == "selected_clean" and int(row["seed"]) == seed
        )
        clean_train = load_aligned_scores(common.resolve(oof_record["prediction_path"]), train_rows)
        clean_valid = load_aligned_scores(
            common.resolve(final_record["valid_prediction_path"]), valid_rows
        )
        clean_test = load_aligned_scores(
            common.resolve(final_record["internal_test_prediction_path"]), test_rows
        )
        expert = common.fit_offset_logistic_expert(
            train_x[actionable],
            v7.labels_array(train_rows)[actionable],
            clean_train[actionable],
            base_weights[actionable],
            policy,
        )
        valid_correction = common.expert_logit_correction(valid_x, expert)
        test_correction = common.expert_logit_correction(test_x, expert)
        fused_valid, valid_decisions = common.apply_constrained_expert(
            clean_valid, valid_evidence, valid_correction
        )
        fused_test, test_decisions = common.apply_constrained_expert(
            clean_test, test_evidence, test_correction
        )
        prediction_rows_source = common.load_csv(
            common.resolve(final_record["valid_prediction_path"])
        )
        thresholds = {
            float(row["selected_threshold"]) for row in prediction_rows_source
        }
        if len(thresholds) != 1:
            raise ValueError("Clean bridge prediction has no unique valid threshold")
        threshold = next(iter(thresholds))
        reported_threshold = float(final_record["valid_metrics"]["threshold"])
        if abs(round(threshold, 6) - reported_threshold) > 1e-12:
            raise ValueError("Bridge summary and persisted valid threshold disagree")
        valid_path = staging_root / "predictions" / f"contextual_evidence__seed_{seed}.zh_valid.csv"
        test_path = staging_root / "predictions" / f"contextual_evidence__seed_{seed}.internal_dev_test.csv"
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.write_bytes(
            common.render_csv(
                output_rows(
                    valid_rows,
                    clean_valid,
                    fused_valid,
                    valid_evidence,
                    valid_decisions,
                    threshold,
                    "representative_valid",
                ),
                OUTPUT_FIELDS,
            )
        )
        test_path.write_bytes(
            common.render_csv(
                output_rows(
                    test_rows,
                    clean_test,
                    fused_test,
                    test_evidence,
                    test_decisions,
                    threshold,
                    "internal_development_test",
                ),
                OUTPUT_FIELDS,
            )
        )
        artifact_path = staging_root / "artifacts" / f"contextual_evidence__seed_{seed}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "artifact_type": "step15_v8_occurrence_context_offset_logistic",
                    "run_id": run_id,
                    "seed": seed,
                    "selected_clean_feature_set": selected_feature_set,
                    "selected_clean_model_family": selected_family,
                    "clean_train_probability_source": "component_grouped_oof",
                    "actionable_train_count": int(np.sum(actionable)),
                    "actionable_state_counts": dict(
                        sorted(
                            Counter(
                                item["evidence_state"]
                                for item, keep in zip(train_evidence, actionable, strict=True)
                                if keep
                            ).items()
                        )
                    ),
                    "factorized_weight_diagnostics": weight_diagnostics,
                    "expert": expert,
                    "state_actions": policy["occurrence_evidence_expert"]["state_actions"],
                    "representative_valid_used_for_model_fitting": False,
                    "internal_test_used_for_model_fitting_or_selection": False,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        clean_matrices["valid"].append(clean_valid)
        clean_matrices["internal_development_test"].append(clean_test)
        fused_matrices["valid"].append(fused_valid)
        fused_matrices["internal_development_test"].append(fused_test)
        seed_records.append(
            {
                "seed": seed,
                "clean_valid_metrics": step7.evaluate_probabilities(
                    v7.labels_array(valid_rows), clean_valid, threshold
                ),
                "fused_valid_metrics_at_clean_threshold": step7.evaluate_probabilities(
                    v7.labels_array(valid_rows), fused_valid, threshold
                ),
                "clean_internal_test_metrics": step7.evaluate_probabilities(
                    v7.labels_array(test_rows), clean_test, threshold
                ),
                "fused_internal_test_metrics_at_clean_threshold": step7.evaluate_probabilities(
                    v7.labels_array(test_rows), fused_test, threshold
                ),
                "valid_prediction_path": str(
                    (final_root / valid_path.relative_to(staging_root)).relative_to(ROOT)
                ).replace("\\", "/"),
                "internal_test_prediction_path": str(
                    (final_root / test_path.relative_to(staging_root)).relative_to(ROOT)
                ).replace("\\", "/"),
                "artifact_path": str(
                    (final_root / artifact_path.relative_to(staging_root)).relative_to(ROOT)
                ).replace("\\", "/"),
            }
        )

    clean_valid_mean = np.mean(np.vstack(clean_matrices["valid"]), axis=0)
    clean_test_mean = np.mean(np.vstack(clean_matrices["internal_development_test"]), axis=0)
    fused_valid_mean = np.mean(np.vstack(fused_matrices["valid"]), axis=0)
    fused_test_mean = np.mean(np.vstack(fused_matrices["internal_development_test"]), axis=0)
    clean_threshold = step7.choose_threshold(
        v7.labels_array(valid_rows),
        clean_valid_mean,
        policy["threshold_selection"]["metric"],
        policy,
    )
    mean_paths = {}
    neutral_decisions_valid = [
        {
            "expert_action": "seed_mean_aggregated",
            "raw_logit_correction": 0.0,
            "applied_logit_correction": float(
                np.log(np.clip(fused, 1e-6, 1 - 1e-6) / np.clip(1 - fused, 1e-6, 1))
                - np.log(np.clip(clean, 1e-6, 1 - 1e-6) / np.clip(1 - clean, 1e-6, 1))
            ),
        }
        for clean, fused in zip(clean_valid_mean, fused_valid_mean, strict=True)
    ]
    neutral_decisions_test = [
        {
            "expert_action": "seed_mean_aggregated",
            "raw_logit_correction": 0.0,
            "applied_logit_correction": float(
                np.log(np.clip(fused, 1e-6, 1 - 1e-6) / np.clip(1 - fused, 1e-6, 1))
                - np.log(np.clip(clean, 1e-6, 1 - 1e-6) / np.clip(1 - clean, 1e-6, 1))
            ),
        }
        for clean, fused in zip(clean_test_mean, fused_test_mean, strict=True)
    ]
    for split_name, rows, clean, fused, evidence, decisions in (
        (
            "zh_valid",
            valid_rows,
            clean_valid_mean,
            fused_valid_mean,
            valid_evidence,
            neutral_decisions_valid,
        ),
        (
            "internal_dev_test",
            test_rows,
            clean_test_mean,
            fused_test_mean,
            test_evidence,
            neutral_decisions_test,
        ),
    ):
        path = staging_root / "predictions" / f"contextual_evidence__seed_mean.{split_name}.csv"
        path.write_bytes(
            common.render_csv(
                output_rows(
                    rows,
                    clean,
                    fused,
                    evidence,
                    decisions,
                    clean_threshold,
                    "representative_valid"
                    if split_name == "zh_valid"
                    else "internal_development_test",
                ),
                OUTPUT_FIELDS,
            )
        )
        mean_paths[split_name] = str(
            (final_root / path.relative_to(staging_root)).relative_to(ROOT)
        ).replace("\\", "/")

    summary = {
        "step": "step15_train_v8_contextual_evidence",
        "version": policy["version"],
        "run_id": run_id,
        "selected_clean_feature_set": selected_feature_set,
        "selected_clean_model_family": selected_family,
        "seed_count": len(seeds),
        "clean_threshold_from_representative_valid": clean_threshold,
        "train_evidence_state_counts": dict(
            sorted(Counter(item["evidence_state"] for item in train_evidence).items())
        ),
        "valid_evidence_state_counts": dict(
            sorted(Counter(item["evidence_state"] for item in valid_evidence).items())
        ),
        "internal_test_evidence_state_counts": dict(
            sorted(Counter(item["evidence_state"] for item in test_evidence).items())
        ),
        "seed_records": seed_records,
        "seed_mean": {
            "clean_valid_metrics": step7.evaluate_probabilities(
                v7.labels_array(valid_rows), clean_valid_mean, clean_threshold
            ),
            "fused_valid_metrics_at_clean_threshold": step7.evaluate_probabilities(
                v7.labels_array(valid_rows), fused_valid_mean, clean_threshold
            ),
            "clean_internal_test_metrics": step7.evaluate_probabilities(
                v7.labels_array(test_rows), clean_test_mean, clean_threshold
            ),
            "fused_internal_test_metrics_at_clean_threshold": step7.evaluate_probabilities(
                v7.labels_array(test_rows), fused_test_mean, clean_threshold
            ),
            "clean_valid_slices": slice_metrics(valid_rows, clean_valid_mean, clean_threshold),
            "fused_valid_slices": slice_metrics(valid_rows, fused_valid_mean, clean_threshold),
            "clean_internal_test_slices": slice_metrics(test_rows, clean_test_mean, clean_threshold),
            "fused_internal_test_slices": slice_metrics(test_rows, fused_test_mean, clean_threshold),
            "prediction_paths": mean_paths,
        },
        "representative_valid_used_for_expert_model_fitting": False,
        "internal_test_used_for_model_fitting_selection_or_threshold": False,
        "mixed_context_hard_veto_applied": False,
        "ambiguous_hard_veto_applied": False,
        "policy_sha256": common.sha256(policy_path),
        "bridge_summary_sha256": common.sha256(bridge_summary_path),
    }
    summary["summary_sha256"] = common.canonical_hash(summary)
    summary_path = staging_root / "step15_v8_contextual_evidence_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    staging_root.replace(final_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "selected_clean_feature_set": selected_feature_set,
                "selected_clean_model_family": selected_family,
                "summary": str((final_root / summary_path.name).relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
