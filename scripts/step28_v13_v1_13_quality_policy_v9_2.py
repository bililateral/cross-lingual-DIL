#!/usr/bin/env python3
"""Validate the implementation-only V9.2 scientific/quality machine policy."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_quality_channel_policy_v9 as parent_validator
import step28_v13_v1_13_quality_gate_registry_v9_2 as registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_scientific_quality_policy_v9_2.json"
)
VERSION = "2026-08-23-step28-v13-v1-13-scientific-quality-policy-v9-2"
POLICY_FIELDS = (
    "version",
    "status",
    "canonical_self_hash",
    "authorization",
    "contracts",
    "parent_policies",
    "source_pins",
    "design_scale",
    "model_inputs",
    "text_probe_family_v9_2",
    "inherited_parent_sections",
    "quality_gate_overrides",
    "style_structure_gates",
    "channel_differences",
    "gate_registry",
    "complete_evidence",
    "read_order",
    "failure_rules",
    "authority_stages",
)
EXPECTED_AUTHORIZATION = {
    "implementation_and_fixture_tests": True,
    "random_authority_ceremony": False,
    "design_1004_build": False,
    "quality_audit_run": False,
    "metric_generation": False,
    "formal_seed": False,
    "formal_500_by_4": False,
    "audit_a_b_truth_open": False,
    "model_training": False,
    "model_metric_generation": False,
}
EXPECTED_CONTRACTS = {
    "scientific_experiment": {
        "path": "docs/STEP28_V13_V1_13_SCIENTIFIC_EXPERIMENT_CONTRACT_20260810.zh.md",
        "size_bytes": 19037,
        "sha256": "936ba6d52fa983307c3be654d9996814e740aea154c290c86411064d35e480f6",
    },
    "quality_audit_c_amendment": {
        "path": "docs/STEP28_V13_V1_13_QUALITY_AUDIT_C_AMENDMENT_20260811.zh.md",
        "size_bytes": 12180,
        "sha256": "1ca1e8b8cea551902cf39f42f5ea33be056c47d80c3693f5bf531c843aa94832",
    },
    "v9_quality_channel_contract": {
        "path": "docs/STEP28_V13_V1_13_V9_QUALITY_AND_CHANNEL_SENSITIVITY_CONTRACT_20260814.zh.md",
        "size_bytes": 29578,
        "sha256": "18f600b62b2b46adacab59bcca49d75b2e2694cd8bb26ec7caa98e752cb3393e",
    },
    "v9_2_reconciliation_contract": {
        "path": "docs/STEP28_V13_V1_13_V9_2_SCIENTIFIC_RECONCILIATION_CONTRACT_20260823.zh.md",
        "size_bytes": 17420,
        "sha256": "bb1a98044f91e3a9d915b842b231fd4822d305cc3121852a8f889130cc54edfe",
    },
}
PARENT_QUALITY_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_quality_channel_sensitivity_policy_v9.json"
)
PARENT_BUILDER_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_scientific_dataset_builder_policy_v9.json"
)


class QualityPolicyV92Error(ValueError):
    """Raised when V9.2 policy bytes or frozen semantics drift."""


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _verify_pin(pin: object, *, label: str) -> Path:
    if not isinstance(pin, Mapping) or set(pin) != {"path", "size_bytes", "sha256"}:
        raise QualityPolicyV92Error(f"{label} pin schema drift")
    path = common.repo_path(str(pin["path"]))
    if (
        not path.is_file()
        or isinstance(pin["size_bytes"], bool)
        or not isinstance(pin["size_bytes"], int)
        or path.stat().st_size != pin["size_bytes"]
        or common.sha256_file(path) != pin["sha256"]
    ):
        raise QualityPolicyV92Error(f"{label} pin drift")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityPolicyV92Error(f"Cannot read policy: {path}") from exc
    if not isinstance(value, dict):
        raise QualityPolicyV92Error("Policy root must be an object")
    return value


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(registry.canonical_json_bytes(policy).decode("utf-8"))
    if set(normalized) != set(POLICY_FIELDS):
        raise QualityPolicyV92Error("V9.2 policy exact top-level schema drift")
    if (
        normalized.get("version") != VERSION
        or normalized.get("status")
        != "IMPLEMENTATION_ONLY_NO_DESIGN_BUILD_NO_QUALITY_RUN_NO_FORMAL_GENERATION_NO_TRUTH_UNSEAL_NO_TRAINING"
        or normalized.get("canonical_self_hash") != _canonical_self_hash(normalized)
        or normalized.get("authorization") != EXPECTED_AUTHORIZATION
    ):
        raise QualityPolicyV92Error("V9.2 policy identity/authorization drift")
    if normalized.get("contracts") != EXPECTED_CONTRACTS:
        raise QualityPolicyV92Error("V9.2 contract pin table drift")
    for name, pin in EXPECTED_CONTRACTS.items():
        _verify_pin(pin, label=name)

    parent_policies = normalized.get("parent_policies")
    if not isinstance(parent_policies, Mapping) or set(parent_policies) != {
        "quality_v9",
        "scientific_builder_v9",
    }:
        raise QualityPolicyV92Error("V9.2 parent-policy schema drift")
    parent_quality = _load_json(
        _verify_pin(
            parent_policies["quality_v9"],
            label="parent V9 quality policy",
        )
    )
    parent_validator.validate_policy(parent_quality)
    _verify_pin(
        parent_policies["scientific_builder_v9"],
        label="parent V9 builder policy",
    )
    inherited = normalized.get("inherited_parent_sections")
    if inherited != [
        "public_code_probe",
        "decoded_slot_probe",
        "probe_models",
        "bootstrap",
    ]:
        raise QualityPolicyV92Error("Inherited V9 section list drift")
    parent_scale = parent_quality["design_scale"]
    expected_scale = {
        "split_order": parent_scale["split_order"],
        "world_counts": parent_scale["world_counts"],
        "seller_count_per_world": parent_scale["seller_count_per_world"],
        "pair_count_per_world": parent_scale["pair_count_per_world"],
        "positive_pair_count_per_world": parent_scale[
            "positive_pair_count_per_world"
        ],
        "attempt_index": None,
        "random_authority_created": False,
    }
    if normalized["design_scale"] != expected_scale:
        raise QualityPolicyV92Error("V9.2 design scale/random authority drift")
    expected_quality_gates = dict(parent_quality["quality_gates"])
    expected_quality_gates["text_single_feature_count"] = 346
    expected_quality_gates["descriptive_single_feature_count"] = 1038
    expected_quality_gates["registered_visible_occurrence_multiset_difference_count"] = 0
    if normalized.get("quality_gate_overrides") != {
        "text_single_feature_count": 346,
        "descriptive_single_feature_count": 1038,
        "registered_visible_occurrence_multiset_difference_count": 0,
    }:
        raise QualityPolicyV92Error("V9.2 quality-gate override drift")
    # Return one expanded in-memory policy to consumers while keeping the
    # persisted machine policy compact and byte-bound to the pinned parent.
    normalized = {
        key: value
        for key, value in normalized.items()
        if key not in {"inherited_parent_sections", "quality_gate_overrides"}
    }
    normalized.update(
        {
            "public_code_probe": parent_quality["public_code_probe"],
            "decoded_slot_probe": parent_quality["decoded_slot_probe"],
            "probe_models": parent_quality["probe_models"],
            "bootstrap": parent_quality["bootstrap"],
            "quality_gates": expected_quality_gates,
        }
    )

    text = normalized["text_probe_family_v9_2"]
    if text != {
        "original_author_surfaces": [
            "surface_full",
            "surface_code_masked",
            "surface_code_neutralized",
        ],
        "counterfactual_hard_surface": "surface_style_deranged_full",
        "view_names": [
            "fs_full",
            "fs_title",
            "fs_template_surface",
            "p_full",
            "p_topic",
            "p_template_surface",
            "u_joint_full",
        ],
        "feature_widths": [33, 14, 30, 75, 14, 56, 124],
        "original_author_matrix_count": 21,
        "counterfactual_matrix_count": 7,
        "total_text_matrix_count": 28,
        "descriptive_model_count": 42,
        "hard_model_count": 14,
        "descriptive_single_feature_count": 1038,
        "hard_single_feature_count": 346,
        "excluded_negative_pairs_per_world": 6,
        "model_pair_keyspace_per_world": 378,
        "eligible_pairs_per_world": 372,
        "positive_pairs_per_world": 20,
        "average_precision_baseline": 20 / 372,
        "descriptive_qualification_role": "DESCRIPTIVE_NOT_A_QUALITY_GATE",
        "descriptive_gate_status": "NOT_APPLICABLE",
    }:
        raise QualityPolicyV92Error("V9.2 text family role/width drift")
    if normalized["model_inputs"] != {
        "surface_count": 4,
        "item_profile_file_count": 8,
        "paths": [
            "observed/redacted_items.jsonl",
            "observed/model_seller_profiles.jsonl",
            "observed/redacted_items.code_masked.jsonl",
            "observed/model_seller_profiles.code_masked.jsonl",
            "observed/redacted_items.code_neutralized.jsonl",
            "observed/model_seller_profiles.code_neutralized.jsonl",
            "observed/redacted_items.style_deranged.jsonl",
            "observed/model_seller_profiles.style_deranged.jsonl",
        ],
        "counterfactual_masked_or_neutralized_file_count": 0,
    }:
        raise QualityPolicyV92Error("V9.2 eight-input contract drift")
    if normalized["style_structure_gates"] != {
        "minimum_distinct_style_factor_tuple_count": 2,
        "minimum_visible_carrier_fields_per_seller": 1,
        "minimum_actual_style_factor_reads_per_seller": 1,
        "style_probe_minimum_auc_or_ap_gate": None,
        "mapping_redraw_for_low_or_zero_dose": False,
    }:
        raise QualityPolicyV92Error("V9.2 style structure gate drift")
    if normalized["channel_differences"] != {
        "directions": [
            "full_minus_code_masked",
            "full_minus_code_neutralized",
            "code_masked_minus_code_neutralized",
        ],
        "metrics": ["symmetric_auc", "average_precision"],
        "bootstrap_draws_shared_with_text_family": True,
        "quantile_interval": [0.025, 0.975],
        "qualification_role": "DESCRIPTIVE_NOT_A_QUALITY_GATE",
    }:
        raise QualityPolicyV92Error("V9.2 channel-difference contract drift")
    gate_spec = normalized["gate_registry"]
    if gate_spec != {
        "entry_count": len(registry.GATE_REGISTRY),
        "canonical_json_sha256": registry.GATE_REGISTRY_SHA256,
        "order_bound_by_canonical_registry_hash": True,
        "failed_gate_rule": "registry_order_filter_HARD_GATE_and_passed_false",
    }:
        raise QualityPolicyV92Error("V9.2 gate registry binding drift")
    if normalized["complete_evidence"] != {
        "exclusive_create_before_outer_wrapper": True,
        "all_registry_entries_required_once": True,
        "partial_evidence_forbidden": True,
        "descriptive_passed_field_forbidden": True,
        "dataset_invalidation_survives_wrapper_failure": True,
        "pass_requires_wrapper_success": True,
        "persist_machine_terminal_exclusively": True,
        "preserve_in_memory_dataset_invalidation_if_primary_publication_fails": True,
        "cleanup_requires_documented_failure_boundary": True,
    }:
        raise QualityPolicyV92Error("V9.2 complete-evidence contract drift")
    if normalized["read_order"] != [
        "verify_all_20_split_payloads_by_raw_hash_and_physical_row_count",
        "freeze_eight_persisted_model_input_commitments",
        "freeze_28_text_matrices_2_code_slot_matrices_and_two_eligibility_masks",
        "open_train_truth_once",
        "open_development_truth_once",
        "reverify_every_label_free_byte",
        "calculate_all_42_descriptive_14_text_hard_4_code_slot_models",
        "reverify_all_frozen_state_after_last_model_family",
        "publish_complete_evidence_exclusively",
        "validate_outer_wrapper",
        "persist_machine_terminal_exclusively",
    ]:
        raise QualityPolicyV92Error("V9.2 read-order drift")
    if normalized["failure_rules"] != {
        "no_early_return_after_hard_gate_failure": True,
        "data_gate_failure": "DATASET_INVALIDATED",
        "mechanical_failure_before_complete_calculation": "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        "uncalculable_precondition_failure": "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        "calculable_gate_failure_must_continue_to_all_registry_entries": True,
        "wrapper_failure_after_complete_dataset_invalidation": "DATASET_INVALIDATED_WITH_OUTER_WRAPPER_FAILURE",
        "wrapper_failure_after_complete_pass": "AUDITOR_EXECUTION_FAILED_PASS_NOT_CERTIFIED",
        "v9_v9_1_reuse_forbidden": True,
    }:
        raise QualityPolicyV92Error("V9.2 failure semantics drift")
    if normalized["authority_stages"] != {
        "random_authority_ceremony": "SEPARATE_FUTURE_SINGLE_USE_APPROVAL_REQUIRED",
        "method_qualification_1004_build": "SEPARATE_FUTURE_SINGLE_USE_APPROVAL_REQUIRED",
        "quality_audit": "SEPARATE_FUTURE_SINGLE_USE_APPROVAL_REQUIRED_AFTER_ROOT_COMMIT",
        "formal_500_by_4": "NOT_AUTHORIZED",
        "truth_unseal": "NOT_AUTHORIZED",
        "model_training_or_metrics": "NOT_AUTHORIZED",
    }:
        raise QualityPolicyV92Error("V9.2 authority-stage separation drift")
    source_pins = normalized["source_pins"]
    if not isinstance(source_pins, list) or not source_pins:
        raise QualityPolicyV92Error("V9.2 source pin list is absent")
    paths = [value.get("path") if isinstance(value, Mapping) else None for value in source_pins]
    if (
        paths != sorted(paths, key=lambda value: str(value).encode("utf-8"))
        or len(paths) != len(set(paths))
    ):
        raise QualityPolicyV92Error("V9.2 source pin order/uniqueness drift")
    for index, pin in enumerate(source_pins):
        _verify_pin(pin, label=f"source pin {index}")
    return normalized


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise QualityPolicyV92Error("Alternate V9.2 policy path is forbidden")
    return validate_policy(_load_json(path))
