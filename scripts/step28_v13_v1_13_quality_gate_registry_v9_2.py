#!/usr/bin/env python3
"""Canonical V9.2 quality-gate registry shared by every audit layer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


VERSION = "2026-08-23-step28-v13-v1-13-quality-gate-registry-v9-2"
HARD_GATE = "HARD_GATE"
DESCRIPTIVE = "DESCRIPTIVE_NOT_A_QUALITY_GATE"
NOT_APPLICABLE = "NOT_APPLICABLE"
REQUIRED = "REQUIRED"

SURFACES = (
    "surface_full",
    "surface_code_masked",
    "surface_code_neutralized",
)
VIEWS = (
    "fs_full",
    "fs_title",
    "fs_template_surface",
    "p_full",
    "p_topic",
    "p_template_surface",
    "u_joint_full",
)
MODELS = ("logistic_l2", "hist_gradient_boosting_depth2")
FAMILY_GATES = (
    ("maximum_single_feature_symmetric_auc", 0.52),
    ("maximum_family_symmetric_auc", 0.53),
    ("maximum_family_average_precision_uplift", 0.01),
    ("bootstrap_95_upper_symmetric_auc", 0.53),
    ("bootstrap_95_upper_average_precision_uplift", 0.015),
)
ZERO_TOLERANCE_STRUCTURE_METRICS = (
    "registered_visible_occurrence_multiset_difference_count",
    "literal_code_hits_in_masked",
    "literal_code_hits_in_neutralized",
    "unregistered_code_hits",
    "unregistered_clone_foreign_code_hits",
    "prior_world_code_hits",
    "view_keyset_difference_count",
    "neutralized_legal_code_permutation_byte_difference_count",
    "audit_truth_open_count",
    "audit_truth_read_count",
    "audit_truth_materialized_row_count",
    "generator_quality_result_read_count",
    "candidate_quality_result_read_count",
    "view_builder_quality_result_read_count",
)

V9_2_STRUCTURE_METRICS = (
    ("model_input_file_count_mismatch_world_count", 0, "EQUAL"),
    ("pretruth_original_author_text_matrix_count_mismatch_split_count", 0, "EQUAL"),
    ("pretruth_counterfactual_text_matrix_count_mismatch_split_count", 0, "EQUAL"),
    ("pretruth_public_code_matrix_count_mismatch_split_count", 0, "EQUAL"),
    ("pretruth_private_slot_matrix_count_mismatch_split_count", 0, "EQUAL"),
    ("style_derangement_mapping_count_mismatch_world_count", 0, "EQUAL"),
    ("minimum_distinct_style_factor_tuple_count", 2, "GREATER_OR_EQUAL"),
    ("minimum_visible_carrier_fields_per_seller", 1, "GREATER_OR_EQUAL"),
    ("minimum_actual_style_factor_reads_per_seller", 1, "GREATER_OR_EQUAL"),
    ("style_derangement_fixed_point_count", 0, "EQUAL"),
    ("independent_production_replay_count_mismatch_world_count", 0, "EQUAL"),
    ("independent_production_replay_byte_mismatch_world_count", 0, "EQUAL"),
    ("cross_branch_invariant_mismatch_count", 0, "EQUAL"),
    ("identity_mechanism_commitment_missing_world_count", 0, "EQUAL"),
    ("shared_text_eligibility_commitment_mismatch_world_count", 0, "EQUAL"),
    ("m1_mapping_commitment_count_mismatch_world_count", 0, "EQUAL"),
    ("m1_distinct_mapping_commitment_count_mismatch_world_count", 0, "EQUAL"),
    ("counterfactual_forbidden_capability_mounted_count", 0, "EQUAL"),
    ("counterfactual_truth_or_retrieval_read_count", 0, "EQUAL"),
    ("counterfactual_quality_result_read_count", 0, "EQUAL"),
    ("persisted_model_input_hash_mismatch_count", 0, "EQUAL"),
    ("f_p_u_actual_consumption_mismatch_count", 0, "EQUAL"),
    ("full_pair_count_mismatch_world_count", 0, "EQUAL"),
    ("eligible_pair_count_mismatch_world_count", 0, "EQUAL"),
    ("positive_pair_count_mismatch_world_count", 0, "EQUAL"),
    ("excluded_positive_pair_count", 0, "EQUAL"),
    ("cross_branch_label_vector_mismatch_count", 0, "EQUAL"),
    ("train_truth_open_count", 1, "EQUAL"),
    ("development_truth_open_count", 1, "EQUAL"),
)


class GateRegistryV92Error(ValueError):
    """Raised when a consumer changes the unique gate order or role."""


def _entry(
    gate_id: str,
    *,
    family: str,
    qualification_role: str,
    metric: str,
    threshold: float | int | None,
    comparison: str | None,
    input_matrix: str,
    prediction_vector: str,
    bootstrap_family_maxima_vector: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "family": family,
        "qualification_role": qualification_role,
        "metric": metric,
        "threshold": threshold,
        "comparison": comparison,
        "input_matrix": input_matrix,
        "prediction_vector": prediction_vector,
        "bootstrap_family_maxima_vector": bootstrap_family_maxima_vector,
    }


def _structure_entries() -> list[dict[str, Any]]:
    scalar = (
        ("train_code_character_position_maximum_deviation", 0.01, "LESS_OR_EQUAL"),
        (
            "development_code_character_position_maximum_deviation",
            0.01,
            "LESS_OR_EQUAL",
        ),
        *V9_2_STRUCTURE_METRICS,
    )
    output = [
        _entry(
            f"hard.structure.{metric}",
            family="structure",
            qualification_role=HARD_GATE,
            metric=metric,
            threshold=threshold,
            comparison=comparison,
            input_matrix=NOT_APPLICABLE,
            prediction_vector=NOT_APPLICABLE,
            bootstrap_family_maxima_vector=NOT_APPLICABLE,
        )
        for metric, threshold, comparison in scalar
    ]
    output.extend(
        _entry(
            f"hard.structure.{metric}",
            family="structure",
            qualification_role=HARD_GATE,
            metric=metric,
            threshold=0,
            comparison="EQUAL",
            input_matrix=NOT_APPLICABLE,
            prediction_vector=NOT_APPLICABLE,
            bootstrap_family_maxima_vector=NOT_APPLICABLE,
        )
        for metric in ZERO_TOLERANCE_STRUCTURE_METRICS
    )
    return output


def _descriptive_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            f"descriptive.original.{surface}.{view}.{model}",
            family="original_author_text",
            qualification_role=DESCRIPTIVE,
            metric="model_auc_ap_and_preregistered_channel_differences",
            threshold=None,
            comparison=None,
            input_matrix=REQUIRED,
            prediction_vector=REQUIRED,
            bootstrap_family_maxima_vector=NOT_APPLICABLE,
        )
        for surface in SURFACES
        for view in VIEWS
        for model in MODELS
    ]


def _probe_family_entries(prefix: str, family: str) -> list[dict[str, Any]]:
    output = []
    for metric, threshold in FAMILY_GATES:
        is_single = metric == "maximum_single_feature_symmetric_auc"
        is_bootstrap = metric.startswith("bootstrap_")
        output.append(
            _entry(
                f"hard.{prefix}.{metric}",
                family=family,
                qualification_role=HARD_GATE,
                metric=metric,
                threshold=threshold,
                comparison="LESS_OR_EQUAL",
                input_matrix=REQUIRED,
                prediction_vector=NOT_APPLICABLE if is_single else REQUIRED,
                bootstrap_family_maxima_vector=(
                    REQUIRED if is_bootstrap else NOT_APPLICABLE
                ),
            )
        )
    return output


def build_gate_registry() -> tuple[dict[str, Any], ...]:
    registry = tuple(
        [
            *_structure_entries(),
            *_descriptive_entries(),
            *_probe_family_entries("text_deranged", "counterfactual_text"),
            *_probe_family_entries("code_slot", "public_code_private_slot"),
        ]
    )
    identifiers = [str(row["gate_id"]) for row in registry]
    expected_count = len(_structure_entries()) + 42 + 5 + 5
    if len(registry) != expected_count or len(identifiers) != len(set(identifiers)):
        raise GateRegistryV92Error("V9.2 gate registry cardinality drift")
    return registry


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


GATE_REGISTRY = build_gate_registry()
GATE_REGISTRY_BYTES = canonical_json_bytes(GATE_REGISTRY)
GATE_REGISTRY_SHA256 = hashlib.sha256(GATE_REGISTRY_BYTES).hexdigest()
GATE_IDS = tuple(str(row["gate_id"]) for row in GATE_REGISTRY)


def verify_registry(
    registry: Sequence[Mapping[str, Any]], *, expected_sha256: str
) -> tuple[dict[str, Any], ...]:
    normalized = tuple(json.loads(canonical_json_bytes(row)) for row in registry)
    if normalized != GATE_REGISTRY:
        raise GateRegistryV92Error("Consumer gate registry bytes drift")
    observed = hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()
    if observed != expected_sha256 or observed != GATE_REGISTRY_SHA256:
        raise GateRegistryV92Error("Consumer gate registry hash drift")
    return normalized


def ordered_failed_gate_ids(
    observations: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if set(observations) != set(GATE_IDS):
        raise GateRegistryV92Error("Gate observation keyset drift")
    failures: list[str] = []
    for spec in GATE_REGISTRY:
        gate_id = str(spec["gate_id"])
        observation = observations[gate_id]
        if spec["qualification_role"] == HARD_GATE:
            if type(observation.get("passed")) is not bool:
                raise GateRegistryV92Error("Hard gate lacks an exact boolean result")
            if observation["passed"] is False:
                failures.append(gate_id)
        elif "passed" in observation or observation.get("gate_status") != NOT_APPLICABLE:
            raise GateRegistryV92Error(
                "Descriptive gate acquired a qualification boolean"
            )
    return tuple(failures)
