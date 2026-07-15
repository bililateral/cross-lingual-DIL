#!/usr/bin/env python3
"""Read-only Linux data/model preflight before any Step15-v8 GPU work."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import step15_v7_common as v7
import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def validate_v7_feature_availability(
    rows: list[dict], policy: dict, v7_policy: dict
) -> dict:
    """Validate the v7 feature contract without rejecting imputable cells."""
    b1_fields = common.feature_names(
        "B1_v7_20d_e5_cosine_only", policy, v7_policy
    )
    retrieval_fields = [
        "sparse_lexical_similarity_raw",
        "structural_support_score_raw",
    ]
    required_fields = sorted(set(b1_fields) | set(retrieval_fields))
    absent_fields = sorted(
        field for field in required_fields if any(field not in row for row in rows)
    )
    if absent_fields:
        raise ValueError(
            f"Step15-v8 required v7 columns are absent: {absent_fields}"
        )

    nonfinite_counts = {
        field: sum(
            not math.isfinite(common._float_or_nan(row.get(field)))
            for row in rows
        )
        for field in required_fields
    }
    entirely_missing_fields = sorted(
        field for field, count in nonfinite_counts.items() if count == len(rows)
    )
    if entirely_missing_fields:
        raise ValueError(
            "Step15-v8 v7 features are entirely missing on train: "
            f"{entirely_missing_fields}"
        )
    imputation_mode = v7_policy["inductive_features"]["missing_value_imputation"]
    if imputation_mode != "train_median_per_feature":
        fields_requiring_imputation = sorted(
            field for field, count in nonfinite_counts.items() if count
        )
        if fields_requiring_imputation:
            raise ValueError(
                "Step15-v8 v7 features contain non-finite cells without the "
                f"preregistered train-median imputation contract: {fields_requiring_imputation}"
            )

    # This is the same fold-train transform used by the bridge. It fails closed
    # when a configured B1 feature is entirely missing and records only train
    # medians, never representative-valid or internal-test statistics.
    _, transform = common.fit_feature_transform(
        rows,
        "B1_v7_20d_e5_cosine_only",
        policy,
        v7_policy,
        latent=None,
    )
    retrieval_domain_stats = {
        field: common._fit_domain_stats(rows, field) for field in retrieval_fields
    }
    return {
        "required_field_count": len(required_fields),
        "absent_fields": [],
        "nonfinite_cell_counts": {
            field: count for field, count in sorted(nonfinite_counts.items()) if count
        },
        "imputation_mode": imputation_mode,
        "b1_train_median_count": len(transform["median_imputation"]),
        "retrieval_domains": {
            field: sorted(stats) for field, stats in sorted(retrieval_domain_stats.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    policy_path, policy, v7_policy = common.load_policy(args.policy)
    validation = common.validate_policy_contract(policy, v7_policy)
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    if root.exists():
        raise FileExistsError(
            f"Step15-v8 run-id already exists; choose a new V8_RUN_ID instead of overwriting: {root}"
        )
    required_paths = [
        policy_path,
        common.resolve(policy["frozen_dependencies"]["v7_policy"]),
        common.resolve(policy["frozen_dependencies"]["v6_negative_freeze"]),
        common.resolve(policy["frozen_dependencies"]["representative_validation_assignments"]),
        common.resolve(policy["frozen_dependencies"]["representative_validation_manifest"]),
    ]
    for pool in policy["pools"].values():
        required_paths.extend(
            common.resolve(pool[key])
            for key in (
                "frozen_labels",
                "evidence_labels",
                "seller_profiles",
                "item_identity_signals",
                "step4_candidates",
                "v7_pair_features",
                "v7_clean_e5_metadata",
                "v7_clean_e5_matrix",
            )
        )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step15-v8 input: {path}")
    semantic_policy = common.load_json(
        common.resolve(policy["clean_semantics"]["semantic_model_policy"])
    )
    model_paths = []
    for model_key in policy["clean_semantics"]["embedding_model_keys"]:
        model_paths.append(common.resolve(semantic_policy["embedding_models"][model_key]["local_path"]))
    reranker_key = policy["clean_semantics"]["reranker_model_key"]
    model_paths.append(
        common.resolve(semantic_policy["reranker_models"][reranker_key]["local_path"])
    )
    for path in model_paths:
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing or empty local Step15-v8 model directory: {path}")

    rows_by_pool = v7.load_joined_rows(v7_policy)
    splits = common.split_rows(rows_by_pool)
    assignment_path = common.resolve(
        policy["frozen_dependencies"]["representative_validation_assignments"]
    )
    split_manifest = common.load_json(
        common.resolve(policy["frozen_dependencies"]["representative_validation_manifest"])
    )
    expected_manifest_hash = split_manifest.get("manifest_sha256")
    unsigned_manifest = dict(split_manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    current_hash_valid = expected_manifest_hash == common.canonical_hash(unsigned_manifest)
    legacy_unsigned_manifest = dict(unsigned_manifest)
    legacy_unsigned_manifest.pop("assignment_csv_sha256", None)
    legacy_hash_valid = (
        "manifest_hash_scope" not in split_manifest
        and expected_manifest_hash == common.canonical_hash(legacy_unsigned_manifest)
    )
    if not current_hash_valid and not legacy_hash_valid:
        raise ValueError("Representative validation manifest self-hash is invalid")
    if common.sha256(assignment_path) != split_manifest["assignment_csv_sha256"]:
        raise ValueError("Representative validation assignment hash differs from its manifest")
    if common.sha256(common.resolve(policy["frozen_dependencies"]["v7_policy"])) != split_manifest[
        "policy_sha256"
    ]:
        raise ValueError("Frozen v7 policy hash differs from the representative split manifest")
    for input_group in ("inputs", "effective_inputs"):
        for relative_path, expected_hash in split_manifest.get(input_group, {}).items():
            if common.sha256(common.resolve(relative_path)) != expected_hash:
                raise ValueError(
                    f"Representative split {input_group} hash changed: {relative_path}"
                )
    observed_split_counts = {name: len(rows) for name, rows in splits.items()}
    if observed_split_counts != split_manifest["row_counts"]:
        raise ValueError(
            f"Representative split counts changed: {observed_split_counts} != "
            f"{split_manifest['row_counts']}"
        )
    if len(splits["internal_development_test"]) != int(
        policy["evaluation"]["current_internal_test_row_count_expected"]
    ):
        raise ValueError("Step15-v8 internal-test boundary differs from preregistration")
    train_rows = splits["train"]
    fold_counts = []
    for seed in policy["bridge_audit"]["seeds"]:
        folds = common.seeded_component_group_folds(
            train_rows, int(policy["bridge_audit"]["group_folds"]), int(seed)
        )
        fold_counts.append([len(held) for _, held in folds])
    corpus_context = common.load_corpus_reference_context(policy, v7_policy)
    reference = common.fit_corpus_reference(train_rows, corpus_context)
    transformed = common.apply_corpus_reference(train_rows, reference, corpus_context)
    v7_feature_availability = validate_v7_feature_availability(
        transformed, policy, v7_policy
    )
    occurrence_required = {
        "seller_uid",
        "contact_type",
        "normalized_value",
        "seller_facing_context",
        "product_data_risk_context",
        "direct_identity_eligible",
        "support_only",
        "context",
    }
    occurrence_counts = {}
    for pool_name, pool in policy["pools"].items():
        rows = common.load_csv(common.resolve(pool["item_identity_signals"]))
        if rows and not occurrence_required.issubset(rows[0]):
            raise ValueError(
                f"Step15-v8 occurrence schema missing fields for {pool_name}: "
                f"{sorted(occurrence_required - set(rows[0]))}"
            )
        occurrence_counts[pool_name] = len(rows)
        pair_uids = {
            row["pair_uid"] for row in common.load_csv(common.resolve(pool["v7_pair_features"]))
        }
        step4_uids = {
            row["pair_uid"] for row in common.load_csv(common.resolve(pool["step4_candidates"]))
        }
        if pair_uids != step4_uids:
            raise ValueError(f"Step4/v7 pair universe differs for {pool_name}")
    readiness_valid_rows = (
        splits["valid"] + splits["evidence_expert_valid_controls"]
    )
    valid_counts = Counter(row["evidence_type"] for row in splits["valid"])
    control_valid_counts = Counter(
        row["evidence_type"]
        for row in splits["evidence_expert_valid_controls"]
    )
    zh_rows = rows_by_pool["zh_target_strict"]
    zh_train_sellers = {
        str(row[key])
        for row in zh_rows
        if row["v7_split_name"] == "train"
        for key in ("seller_uid_left", "seller_uid_right")
    }
    zh_occurrence_index, zh_token_df = common.item_signal_index(
        common.resolve(policy["pools"]["zh_target_strict"]["item_identity_signals"]),
        zh_train_sellers,
    )
    frequency_threshold = int(
        policy["occurrence_evidence_expert"][
            "public_identifier_train_seller_frequency_threshold"
        ]
    )
    valid_states = [
        common.occurrence_evidence(
            row, zh_occurrence_index, zh_token_df, frequency_threshold
        )["evidence_state"]
        for row in readiness_valid_rows
    ]
    slice_masks = common.validation_slice_masks(readiness_valid_rows, valid_states)
    readiness = {
        key: {
            "observed": int(sum(slice_masks[key])),
            "required": int(required),
            "met": int(sum(slice_masks[key])) >= int(required),
        }
        for key, required in policy["promotion_gates"][
            "minimum_valid_slice_counts"
        ].items()
    }
    unmet_readiness = {key: item for key, item in readiness.items() if not item["met"]}
    if unmet_readiness:
        raise ValueError(
            "Step15-v8 representative validation lacks occurrence-state-backed evidence. "
            "Run the Step16-v8 blind review/refreeze workflow; thresholds must not be lowered. "
            f"unmet={json.dumps(unmet_readiness, ensure_ascii=False, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                **validation,
                "run_id": run_id,
                "run_root_absent": True,
                "model_directories": [str(path.relative_to(ROOT)).replace("\\", "/") for path in model_paths],
                "split_counts": {
                    split: {
                        "total": len(rows),
                        "positive": sum(row["review_label"] == "positive" for row in rows),
                        "negative": sum(row["review_label"] == "negative" for row in rows),
                    }
                    for split, rows in splits.items()
                },
                "oof_held_fold_counts_by_seed": fold_counts,
                "v7_feature_availability": v7_feature_availability,
                "occurrence_counts": occurrence_counts,
                "representative_valid_evidence_counts": dict(sorted(valid_counts.items())),
                "evidence_expert_valid_control_counts": dict(
                    sorted(control_valid_counts.items())
                ),
                "readiness_valid_scope": (
                    "primary_representative_valid_plus_isolated_evidence_expert_"
                    "validation_controls"
                ),
                "evidence_expert_controls_used_for_primary_model_selection": False,
                "representative_valid_occurrence_state_readiness": readiness,
                "legacy_evidence_type_only_public_count_used_for_readiness": False,
                "representative_manifest_hash_mode": (
                    "all_fields_except_self_hash"
                    if current_hash_valid
                    else "legacy_pre_assignment_hash_with_independent_assignment_check"
                ),
                "all_checks_read_only": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
