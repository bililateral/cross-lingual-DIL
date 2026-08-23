#!/usr/bin/env python3
"""Assemble V9.2 structure and numerical results in the unique registry order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_quality_complete_evidence_v9_2 as evidence
import step28_v13_v1_13_quality_gate_registry_v9_2 as registry
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer_v9
import step28_v13_v1_13_quality_probe_preparer_v9_2 as preparer_v9_2


VERSION = "2026-08-23-step28-v13-v1-13-quality-result-assembler-v9-2"


class QualityResultAssemblerV92Error(ValueError):
    """Raised before partial or role-drifted evidence can be called complete."""


def matrix_structure_metric_values(
    *,
    train_text_matrices: Sequence[preparer_v9.FrozenFeatureMatrix]
    | preparer_v9_2.FrozenTextBundleV92,
    development_text_matrices: Sequence[preparer_v9.FrozenFeatureMatrix]
    | preparer_v9_2.FrozenTextBundleV92,
    train_code_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    development_code_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
) -> dict[str, int]:
    text_values = []
    for value in (train_text_matrices, development_text_matrices):
        descriptive, hard = preparer_v9_2.split_text_matrix_roles(value)
        text_values.append((len(descriptive), len(hard)))
    code_values = []
    for matrices in (train_code_matrices, development_code_matrices):
        values = tuple(matrices)
        for value in values:
            preparer_v9.verify_frozen_feature_matrix(value)
        code_values.append(
            (
                sum(value.view == "public_code_2992" for value in values),
                sum(value.view == "decoded_slot_388" for value in values),
            )
        )
    return {
        "pretruth_original_author_text_matrix_count_mismatch_split_count": sum(
            descriptive_count != 21
            for descriptive_count, _hard_count in text_values
        ),
        "pretruth_counterfactual_text_matrix_count_mismatch_split_count": sum(
            hard_count != 7 for _descriptive_count, hard_count in text_values
        ),
        "pretruth_public_code_matrix_count_mismatch_split_count": sum(
            public_count != 1 for public_count, _slot_count in code_values
        ),
        "pretruth_private_slot_matrix_count_mismatch_split_count": sum(
            slot_count != 1 for _public_count, slot_count in code_values
        ),
    }


def merge_complete_observations(
    *,
    structure_metric_values: Mapping[str, int | float],
    numerical_observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    structure = evidence.build_structure_gate_observations(structure_metric_values)
    expected_numerical = {
        str(spec["gate_id"])
        for spec in registry.GATE_REGISTRY
        if spec["family"] != "structure"
    }
    if set(numerical_observations) != expected_numerical:
        raise QualityResultAssemblerV92Error(
            "Numerical observation registry partition drift"
        )
    merged = {**structure, **{key: dict(value) for key, value in numerical_observations.items()}}
    if set(merged) != set(registry.GATE_IDS):
        raise QualityResultAssemblerV92Error("Complete observation keyset drift")
    return {gate_id: merged[gate_id] for gate_id in registry.GATE_IDS}


def assemble_formal_complete_evidence(
    *,
    structure_receipt: Mapping[str, Any],
    numerical_receipt: Mapping[str, Any],
    train_text_bundle: preparer_v9_2.FrozenTextBundleV92,
    development_text_bundle: preparer_v9_2.FrozenTextBundleV92,
    train_code_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    development_code_matrices: Sequence[preparer_v9.FrozenFeatureMatrix],
    quality_policy: Mapping[str, Any],
    root_manifest_sha256: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    if (
        structure_receipt.get("gate_registry_sha256")
        != registry.GATE_REGISTRY_SHA256
        or numerical_receipt.get("gate_registry_sha256")
        != registry.GATE_REGISTRY_SHA256
        or numerical_receipt.get("claim_boundary")
        != "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
    ):
        raise QualityResultAssemblerV92Error("Component registry/claim binding drift")
    structure_payload = dict(structure_receipt)
    supplied_structure_hash = structure_payload.pop("canonical_self_hash", None)
    if supplied_structure_hash != common.canonical_sha256(structure_payload):
        raise QualityResultAssemblerV92Error("Structure receipt self-hash drift")
    numerical_payload = dict(numerical_receipt)
    supplied_numerical_hash = numerical_payload.pop("canonical_self_hash", None)
    if supplied_numerical_hash != common.canonical_sha256(numerical_payload):
        raise QualityResultAssemblerV92Error("Numerical receipt self-hash drift")
    metric_values = dict(structure_receipt.get("metric_values", {}))
    for name, value in matrix_structure_metric_values(
        train_text_matrices=train_text_bundle,
        development_text_matrices=development_text_bundle,
        train_code_matrices=train_code_matrices,
        development_code_matrices=development_code_matrices,
    ).items():
        if name in metric_values:
            raise QualityResultAssemblerV92Error("Matrix metric was computed twice")
        metric_values[name] = value
    f_p_u_receipts = (
        preparer_v9_2.validate_counterfactual_f_p_u_consumption(train_text_bundle),
        preparer_v9_2.validate_counterfactual_f_p_u_consumption(
            development_text_bundle
        ),
    )
    metric_values["f_p_u_actual_consumption_mismatch_count"] = sum(
        int(value["f_p_u_actual_consumption_mismatch_count"])
        for value in f_p_u_receipts
    )
    for name, value in numerical_receipt.get("structure_metric_values", {}).items():
        if name in metric_values:
            raise QualityResultAssemblerV92Error("Numerical structure metric duplicated")
        metric_values[str(name)] = value
    observations = merge_complete_observations(
        structure_metric_values=metric_values,
        numerical_observations=numerical_receipt["numerical_gate_observations"],
    )
    truth_bindings = numerical_receipt.get("truth_and_order_bindings")
    if not isinstance(truth_bindings, Mapping):
        raise QualityResultAssemblerV92Error("Truth/order bindings are absent")
    bootstrap = quality_policy["bootstrap"]
    bindings = {
        "root_manifest_sha256": root_manifest_sha256,
        "quality_policy_canonical_self_hash": quality_policy[
            "canonical_self_hash"
        ],
        "source_bundle_sha256": source_bundle_sha256,
        **{field: truth_bindings[field] for field in truth_bindings},
        "model_constants_sha256": common.canonical_sha256(
            quality_policy["probe_models"]
        ),
        "bootstrap_constants_sha256": common.canonical_sha256(bootstrap),
        "bootstrap_draw_matrix_sha256": bootstrap["raw_index_matrix_sha256"],
    }
    if set(bindings) != set(evidence.BINDING_FIELDS):
        raise QualityResultAssemblerV92Error("Final evidence binding keyset drift")
    return evidence.assemble_complete_quality_evidence(
        observations=observations,
        bindings=bindings,
    )
