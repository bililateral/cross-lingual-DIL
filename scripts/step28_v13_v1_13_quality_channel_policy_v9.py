#!/usr/bin/env python3
"""Validate the implementation-only v9 quality-channel machine policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import step28_v13_v1_13_quality_channel_views_v9 as channel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT / "schema" / "step28_v13_v1_13_quality_channel_sensitivity_policy_v9.json"
)
EXPECTED_POLICY_SELF_HASH = (
    "3147c01032aed44f6465ac9db654c944c538fb326d0ee0234773a36eae32c9f9"
)
EXPECTED_PIN_NAMES = {
    "scientific_contract",
    "quality_audit_c_amendment",
    "quality_audit_v8_scale_amendment",
    "v9_channel_contract",
    "channel_views_source",
    "channel_views_tests",
    "channel_materializer_source",
    "channel_materializer_tests",
    "text_probe_views_source",
    "text_probe_views_tests",
    "probe_preparer_source",
    "probe_preparer_tests",
    "probe_validator_source",
    "probe_validator_tests",
    "truth_capability_source",
    "truth_capability_tests",
    "structure_aggregator_source",
    "structure_aggregator_tests",
    "audit_runner_source",
    "audit_runner_tests",
}
EXPECTED_MATERIALIZED_SPLIT_FILES = [
    "observed/redacted_items.jsonl",
    "observed/redacted_items.code_masked.jsonl",
    "observed/redacted_items.code_neutralized.jsonl",
    "observed/model_seller_profiles.jsonl",
    "observed/model_seller_profiles.code_masked.jsonl",
    "observed/model_seller_profiles.code_neutralized.jsonl",
    "private/public_code_probe_input.jsonl",
    "private/text_probe_eligibility_input.jsonl",
    "private/channel_structure_audit.jsonl",
]


class QualityChannelPolicyError(ValueError):
    """Raised when the implementation-only channel policy drifts."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pin(pin: Mapping[str, Any]) -> None:
    if set(pin) != {"path", "size_bytes", "sha256"}:
        raise QualityChannelPolicyError("Pinned file schema drift")
    relative = Path(str(pin["path"]))
    if relative.is_absolute():
        raise QualityChannelPolicyError("Pinned path must be repository-relative")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise QualityChannelPolicyError("Pinned path escapes the repository") from exc
    if (
        not path.is_file()
        or isinstance(pin["size_bytes"], bool)
        or not isinstance(pin["size_bytes"], int)
        or path.stat().st_size != pin["size_bytes"]
        or _sha256_file(path) != pin["sha256"]
    ):
        raise QualityChannelPolicyError(f"Pinned file drift: {relative.as_posix()}")


def _feature_names_hash(names: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(names))).hexdigest()


EXPECTED_LOGISTIC = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": None,
    "max_iter": 10000,
    "n_jobs": None,
    "penalty": "l2",
    "random_state": 793820367,
    "solver": "lbfgs",
    "tol": 1e-10,
    "verbose": 0,
    "warm_start": False,
}

EXPECTED_TREE = {
    "class": "sklearn.ensemble.HistGradientBoostingClassifier",
    "categorical_features": "from_dtype",
    "class_weight": None,
    "early_stopping": False,
    "interaction_cst": None,
    "l2_regularization": 1.0,
    "learning_rate": 0.03,
    "loss": "log_loss",
    "max_bins": 255,
    "max_depth": 2,
    "max_features": 1.0,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "monotonic_cst": None,
    "n_iter_no_change": 10,
    "random_state": 793820367,
    "scoring": "loss",
    "tol": 1e-7,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}


def validate_policy(policy: Mapping[str, Any], *, check_pins: bool = True) -> None:
    expected_top = {
        "version",
        "status",
        "canonical_self_hash",
        "authorization",
        "pins",
        "design_scale",
        "model_views",
        "text_probe_family",
        "public_code_probe",
        "decoded_slot_probe",
        "probe_models",
        "quality_gates",
        "bootstrap",
        "read_order",
        "failure_rules",
        "authority_fields",
    }
    if set(policy) != expected_top:
        raise QualityChannelPolicyError("Quality-channel policy top-level schema drift")
    payload = dict(policy)
    observed_self_hash = payload.pop("canonical_self_hash")
    if (
        not isinstance(observed_self_hash, str)
        or len(observed_self_hash) != 64
        or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != observed_self_hash
    ):
        raise QualityChannelPolicyError("Quality-channel policy self hash drift")
    if observed_self_hash != EXPECTED_POLICY_SELF_HASH:
        raise QualityChannelPolicyError(
            "Quality-channel policy is not the frozen semantic commitment"
        )

    authorization = policy["authorization"]
    if authorization != {
        "implementation_and_fixture_tests": True,
        "design_1004_rebuild": False,
        "quality_audit_run": False,
        "formal_seed": False,
        "formal_500_by_4": False,
        "audit_truth_open": False,
        "model_training": False,
        "metric_generation": False,
    }:
        raise QualityChannelPolicyError("Quality-channel authorization widened")
    if set(policy["pins"]) != EXPECTED_PIN_NAMES:
        raise QualityChannelPolicyError("Quality-channel pin universe drift")
    if check_pins:
        for pin in policy["pins"].values():
            _validate_pin(pin)

    scale = policy["design_scale"]
    if (
        scale["split_order"] != ["train", "development", "audit_a", "audit_b"]
        or scale["world_counts"]
        != {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2}
        or scale["seller_count_per_world"] != 28
        or scale["pair_count_per_world"] != 378
        or scale["positive_pair_count_per_world"] != 20
        or scale["attempt_index"] != 1
        or scale["replacement_authority_forbidden"] is not True
    ):
        raise QualityChannelPolicyError("Design-scale contract drift")

    views = policy["model_views"]
    if (
        views["order"]
        != [
            "surface_full",
            "surface_code_masked",
            "surface_code_neutralized",
        ]
        or views["mask_token"] != channel.MASK_TOKEN
        or views["mask_token_code_point_length"] != len(channel.MASK_TOKEN)
        or views["neutral_render_code_ordinal_zero"] != "QAAAAAAAABA"
        or views["neutral_render_code_family"]
        != "Q_plus_item_uid_utf8_sorted_zero_based_ordinal_as_eight_base16_A_to_P_digits_plus_BA"
        or views["neutral_render_code_family_values_persisted"] is not False
        or views["neutral_render_title_modifier"] != "常规款"
        or views["neutral_render_english_tag_rule"]
        != "retain_nonempty_original_tag_and_force_visibility_independent_of_original_code_else_remain_empty"
        or views["materialization_stage"]
        != "after_candidate_irreversible_acceptance_before_build_private_truth"
        or views["quality_time_world_rebuild_forbidden"] is not True
        or views["quality_time_view_transform_forbidden"] is not True
        or views["neutralized_legal_code_permutation_byte_difference_maximum"]
        != 0
        or views["non_code_projection_commitment_required"] is not True
        or views["registered_ast_span_masking_required"] is not True
        or views["free_string_replacement_forbidden"] is not True
        or views["neutralizer_original_code_read_set_maximum"] != 0
        or views["neutralizer_input_capability"]
        != "NeutralItemProjection[NeutralItemMetadata]"
        or views["neutralizer_input_fields"]
        != ["world_uid", "seller_uid", "item_uid", "time_bucket", "category"]
        or views["neutralizer_complete_public_item_argument_forbidden"] is not True
        or views["neutralizer_original_text_value_tripwire_required"] is not True
        or views["neutralizer_runtime_field_read_guard_required"] is not True
        or views["neutralizer_receipt_read_counts_must_be_runtime_derived"] is not True
        or views["non_code_projection_structural_equality_required"] is not True
        or views["non_code_projection_relative_ast_boundaries_required"] is not True
        or views["non_code_projection_absolute_offsets_compared"] is not False
        or views["derived_symbol_exhaustive_state_count"] != 256
        or views["derived_symbol_actual_neutral_mount_integration_required"] is not True
        or views["materialized_split_files"] != EXPECTED_MATERIALIZED_SPLIT_FILES
        or len(set(views["materialized_split_files"]))
        != len(EXPECTED_MATERIALIZED_SPLIT_FILES)
        or views["recompute_profiles_from_each_item_view"] is not True
        or views["neutral_profile_recompute_order"]
        != "collapse_all_ephemeral_codes_in_items_then_rebuild_profile_safe_items_then_rebuild_complete_profiles"
        or views["recompute_all_allowed_numeric_profile_fields"] is not True
        or views["registered_code_masking_only"] is not True
        or views["unregistered_code_shaped_token_maximum"] != 0
        or views["neutral_capacity_mapping"]
        != "strict_inverse_of_frozen_capacity_index_map_to_original_0_through_7_base_skeleton"
        or views["neutral_original_code_bearing_base_skeleton"]
        != "retain_original_carrier_words_and_replace_neutral_render_code_after_production"
        or views["neutral_original_non_code_base_skeleton"]
        != "remove_v9_capacity_twin_clause_and_add_no_placeholder"
        or views["exact_clone_order"]
        != "render_both_neutral_native_titles_then_copy_registered_source_neutral_title_to_target"
    ):
        raise QualityChannelPolicyError("Three-view contract drift")

    text_family = policy["text_probe_family"]
    if (
        text_family["view_names"]
        != [
            "fs_full",
            "fs_title",
            "fs_template_surface",
            "p_full",
            "p_topic",
            "p_template_surface",
            "u_joint_full",
        ]
        or text_family["surface_view_count"] != 3
        or text_family["model_count_per_view"] != 2
        or text_family["total_model_count"] != 42
        or text_family["model_pair_keyspace_per_world"] != 378
        or text_family["eligible_pairs_per_world"] != 372
        or text_family["positive_pairs_per_world"] != 20
        or text_family["average_precision_baseline"] != 20 / 372
        or text_family["feature_widths"] != [33, 14, 30, 75, 14, 56, 124]
        or text_family["single_feature_count"] != 1038
        or text_family["row_weight"] != 1.0
        or text_family["sample_weight_argument_forbidden"] is not True
    ):
        raise QualityChannelPolicyError("Text probe family drift")

    public = policy["public_code_probe"]
    decoded = policy["decoded_slot_probe"]
    expected_public_allowed = {
        "registered_code",
        "visible_field",
        "own_or_foreign",
        "five_materialized_profile_code_occurrences",
        "six_full_minus_neutral_numeric_profile_deltas",
    }
    expected_public_forbidden = {
        "product_semantics",
        "identity_text",
        "world_uid_feature",
        "seller_uid_feature",
        "item_uid_feature",
        "pair_uid_feature",
        "controller",
        "label",
        "qrels",
        "private_code_key",
        "decoded_slot",
        "override_kind",
        "clone_direction_or_source",
        "foreign_code_legal_cause",
        "capacity_or_template_role",
        "code_derived_surface_cause",
    }
    if (
        public["feature_width"] != channel.PUBLIC_FEATURE_WIDTH
        or public["feature_names_canonical_json_sha256"]
        != _feature_names_hash(channel.PUBLIC_FEATURE_NAMES)
        or public["feature_names_generator"]
        != "step28_v13_v1_13_quality_channel_views_v9.public_feature_names"
        or public["feature_group_widths"]
        != {
            "relation_summaries": 40,
            "owned_absolute_character_composition": 320,
            "all_visible_absolute_character_composition": 320,
            "seven_field_absolute_character_composition": 2240,
            "profile_field_summaries": 60,
            "full_minus_neutral_numeric_deltas": 12,
        }
        or public["pair_exchange_invariant"] is not True
        or set(public["allowed_inputs"]) != expected_public_allowed
        or set(public["forbidden_inputs"]) != expected_public_forbidden
        or public["materialized_top_level_fields"]
        != [
            "world_uid",
            "seller_uid",
            "owned_codes",
            "item_occurrences",
            "profile_occurrences",
            "numeric_profile_deltas",
        ]
        or public["nested_occurrence_fields"] != ["field", "code", "is_own"]
        or public["nested_ast_node_or_span_or_hash_fields_maximum"] != 0
        or decoded["private_only"] is not True
        or decoded["feature_width"] != 388
        or decoded["seller_slot_pair_one_hot_width"] != 378
        or decoded["scalar_width"] != 10
        or decoded["feature_names_canonical_json_sha256"]
        != _feature_names_hash(channel.DECODED_FEATURE_NAMES)
        or decoded["feature_names_generator"]
        != "step28_v13_v1_13_quality_channel_views_v9.decoded_feature_names"
        or decoded["average_precision_baseline"] != 20 / 378
        or decoded["mode_global_ordinal_as_feature"] is not False
        or decoded["seller_slot_range"] != [0, 27]
        or decoded["item_slot_range"] != [0, 7]
        or decoded["item_slots_nonempty_unique_contiguous_from_zero"] is not True
        or decoded["decoded_values_or_matrix_persisted"] is not False
        or decoded["pair_exchange_invariant"] is not True
    ):
        raise QualityChannelPolicyError("Code or decoded-slot feature schema drift")

    models = policy["probe_models"]
    if (
        models["standardization"]
        != {
            "logistic_l2": "StandardScaler_fit_on_train_only_then_transform_train_and_development",
            "hist_gradient_boosting_depth2": "raw_unstandardized_values",
        }
        or
        models["logistic_l2"] != EXPECTED_LOGISTIC
        or models["hist_gradient_boosting_depth2"] != EXPECTED_TREE
        or models["logistic_iteration_reaching_maximum_is_failure"] is not True
        or models["class_reweighting_forbidden"] is not True
        or models["resampling_forbidden"] is not True
        or models["development_fit_forbidden"] is not True
        or models["runtime"]
        != {
            "python": "3.10.11",
            "numpy": "2.2.6",
            "scikit_learn": "1.7.2",
            "float_dtype": "float64",
            "threads": 1,
            "positive_label": 1,
            "negative_label": 0,
            "score_column": "predict_proba[:,1]",
            "nonfinite_maximum": 0,
            "sample_weight_argument_forbidden": True,
        }
    ):
        raise QualityChannelPolicyError("Frozen probe model drift")

    gates = policy["quality_gates"]
    expected_numeric = {
        "maximum_single_feature_symmetric_auc": 0.52,
        "maximum_family_symmetric_auc": 0.53,
        "maximum_family_average_precision_uplift": 0.01,
        "bootstrap_95_upper_symmetric_auc": 0.53,
        "bootstrap_95_upper_average_precision_uplift": 0.015,
        "code_character_position_maximum_absolute_deviation_from_one_sixteenth": 0.01,
    }
    if any(gates.get(name) != value for name, value in expected_numeric.items()):
        raise QualityChannelPolicyError("Quality threshold drift")
    if (
        gates.get("text_single_feature_count") != 1038
        or gates.get("code_and_slot_single_feature_count") != 3380
    ):
        raise QualityChannelPolicyError("Single-feature gate universe drift")
    if any(
        gates.get(name) != 0
        for name in (
            "literal_code_hits_in_masked",
            "literal_code_hits_in_neutralized",
            "unregistered_code_hits",
            "prior_world_code_hits",
            "unregistered_clone_foreign_code_hits",
            "view_keyset_difference_count",
            "neutralized_legal_code_permutation_byte_difference_count",
            "audit_truth_open_count",
            "audit_truth_read_count",
            "audit_truth_materialized_row_count",
            "generator_quality_result_read_count",
            "candidate_quality_result_read_count",
            "view_builder_quality_result_read_count",
        )
    ):
        raise QualityChannelPolicyError("Zero-tolerance gate drift")

    bootstrap = policy["bootstrap"]
    if (
        bootstrap["replicates"] != 9999
        or bootstrap["development_world_count"] != 500
        or bootstrap["generator"]
        != "numpy.random.Generator(numpy.random.PCG64)"
        or bootstrap["sampling"] != "world_indices_with_replacement"
        or bootstrap["generated_shape"] != [9999, 500]
        or bootstrap["dtype"] != "int64"
        or bootstrap["endpoint"] is not False
        or bootstrap["order"] != "C"
        or bootstrap["streaming_batch_size"] != 16
        or bootstrap["quantile"] != 0.95
        or bootstrap["quantile_method"] != "linear"
        or bootstrap["refit_models_inside_bootstrap"] is not False
        or bootstrap["maximum_taken_inside_each_replicate_before_quantile"]
        is not True
        or bootstrap["metadata_design_seed"] != 281320260810
        or bootstrap["text_design_seed"] != 281320260810
        or bootstrap["probe_model_seed"] != 793820367
        or bootstrap["world_order"]
        != "development_world_ordinal_0_through_499_ascending"
        or bootstrap["raw_index_dtype"] != "<i8"
        or bootstrap["raw_index_byte_length"] != 39996000
        or bootstrap["raw_index_matrix_sha256"]
        != "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661"
        or bootstrap["duplicate_world_rows_in_original_pair_order"] is not True
        or bootstrap["prediction_vector_dtype"] != "<f8"
        or bootstrap["prediction_vector_hash_required"] is not True
        or bootstrap["family_maxima_vector_hash_required"] is not True
    ):
        raise QualityChannelPolicyError("Bootstrap contract drift")

    read_order = policy["read_order"]
    if (
        read_order["audit_a_b_truth_read_before_prediction_freeze"] is not False
        or read_order["feature_rebuild_after_truth_read"] is not False
        or read_order["generator_or_candidate_reentry_after_truth_read"] is not False
        or read_order["full_private_world_passed_to_view_builder"] is not False
        or read_order["audit_truth_open_count"] != 0
        or read_order["audit_truth_read_count"] != 0
        or read_order["audit_truth_materialized_row_count"] != 0
        or read_order["stage_1"]
        != "builder_materializes_three_item_views_three_profile_views_public_code_probe_input_text_probe_eligibility_input_and_channel_structure_audit_before_private_truth_projection"
        or read_order["stage_2"]
        != "quality_preparer_validates_immutable_root_and_materialized_view_hashes_without_labels"
        or read_order["stage_3"]
        != "quality_preparer_builds_and_freezes_feature_schema_row_order_and_matrix_commitments"
        or read_order["stage_4"]
        != "pinned_supervised_validator_opens_train_and_development_truth_once"
        or read_order["uid_or_split_feature_derivation_forbidden"] is not True
        or read_order[
            "freeze_float_values_missing_bitmap_sparse_indices_shape_rows_columns_sources_before_truth"
        ]
        is not True
    ):
        raise QualityChannelPolicyError("Truth-read order widened")
    failure = policy["failure_rules"]
    for field in (
        "threshold_change_after_data_forbidden",
        "feature_or_model_change_after_data_forbidden",
        "code_or_template_change_after_data_forbidden",
        "salt_or_authority_change_after_data_forbidden",
        "world_seller_item_pair_filtering_forbidden",
        "failed_split_only_rerun_forbidden",
        "audit_truth_diagnosis_forbidden",
        "failed_payload_cleanup_required",
        "same_root_retry_mechanical_only",
        "semantic_change_requires_new_root",
        "preserve_git_commit_tree_hashes_docs_and_small_receipts",
        "delete_failed_large_payload_cache_temp_and_run_specific_worktree_code",
    ):
        if failure.get(field) is not True:
            raise QualityChannelPolicyError("Failure discipline drift")
    if failure.get("freeze_point") != "before_first_byte_of_future_1004_world_root":
        raise QualityChannelPolicyError("Quality freeze point drift")
    authorities = policy["authority_fields"]
    expected_authority_names = {
        "quality_gate_thresholds",
        "average_precision_baselines",
        "text_feature_widths",
        "public_and_private_code_widths",
        "bootstrap_constants",
        "logistic_l2_parameters",
        "hist_gradient_boosting_parameters",
    }
    if set(authorities) != expected_authority_names:
        raise QualityChannelPolicyError("Field authority universe drift")
    for name, authority in authorities.items():
        if set(authority) != {
            "value",
            "authority_path",
            "authority_sha256",
            "authority_section",
            "local_policy_key",
        }:
            raise QualityChannelPolicyError(f"Field authority schema drift: {name}")
        if (
            not isinstance(authority["authority_path"], str)
            or not isinstance(authority["authority_sha256"], str)
            or len(authority["authority_sha256"]) != 64
            or not isinstance(authority["authority_section"], str)
            or not isinstance(authority["local_policy_key"], str)
        ):
            raise QualityChannelPolicyError(f"Field authority value drift: {name}")
    expected_authority_sources = {
        "quality_gate_thresholds": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第7节",
            "quality_gates",
        ),
        "average_precision_baselines": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第4节与第7节",
            "text_probe_family.average_precision_baseline_and_decoded_slot_probe.average_precision_baseline",
        ),
        "text_feature_widths": (
            policy["pins"]["quality_audit_c_amendment"]["path"],
            policy["pins"]["quality_audit_c_amendment"]["sha256"],
            "第2节",
            "text_probe_family.feature_widths",
        ),
        "public_and_private_code_widths": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第5至7节",
            "public_code_probe.feature_width_and_decoded_slot_probe.feature_width",
        ),
        "bootstrap_constants": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第7节",
            "bootstrap",
        ),
        "logistic_l2_parameters": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第7节",
            "probe_models.logistic_l2",
        ),
        "hist_gradient_boosting_parameters": (
            policy["pins"]["v9_channel_contract"]["path"],
            policy["pins"]["v9_channel_contract"]["sha256"],
            "第7节",
            "probe_models.hist_gradient_boosting_depth2",
        ),
    }
    for name, expected in expected_authority_sources.items():
        authority = authorities[name]
        if (
            authority["authority_path"] != expected[0]
            or authority["authority_sha256"] != expected[1]
            or authority["authority_section"] != expected[2]
            or authority["local_policy_key"] != expected[3]
        ):
            raise QualityChannelPolicyError(
                f"Field authority source does not match its pinned authority: {name}"
            )
    expected_gate_authority_value = {
        name: expected_numeric[name]
        for name in (
            "maximum_single_feature_symmetric_auc",
            "maximum_family_symmetric_auc",
            "maximum_family_average_precision_uplift",
            "bootstrap_95_upper_symmetric_auc",
            "bootstrap_95_upper_average_precision_uplift",
            "code_character_position_maximum_absolute_deviation_from_one_sixteenth",
        )
    }
    expected_bootstrap_authority_value = dict(bootstrap)
    if (
        authorities["quality_gate_thresholds"]["value"]
        != expected_gate_authority_value
        or authorities["average_precision_baselines"]["value"]
        != {"text": 20 / 372, "code_and_slot": 20 / 378}
        or authorities["text_feature_widths"]["value"]
        != text_family["feature_widths"]
        or authorities["bootstrap_constants"]["value"]
        != expected_bootstrap_authority_value
        or authorities["logistic_l2_parameters"]["value"] != EXPECTED_LOGISTIC
        or authorities["hist_gradient_boosting_parameters"]["value"]
        != EXPECTED_TREE
        or authorities["public_and_private_code_widths"]["value"]
        != {"public": 2992, "decoded": 388, "combined": 3380}
    ):
        raise QualityChannelPolicyError("Field authority values drift")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY.resolve():
        raise QualityChannelPolicyError("Only the canonical v9 policy path is allowed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityChannelPolicyError("Cannot read the v9 channel policy") from exc
    if not isinstance(value, dict):
        raise QualityChannelPolicyError("The v9 channel policy must be an object")
    validate_policy(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-policy-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_policy_only:
        raise QualityChannelPolicyError(
            "Only --validate-policy-only is authorized; no data execution exists"
        )
    policy = load_policy()
    print(
        json.dumps(
            {
                "event": "step28_v13_v1_13_v9_quality_channel_policy_valid",
                "canonical_self_hash": policy["canonical_self_hash"],
                "design_1004_rebuild": False,
                "quality_audit_run": False,
                "model_training": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
