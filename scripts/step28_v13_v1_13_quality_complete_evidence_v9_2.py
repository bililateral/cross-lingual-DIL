#!/usr/bin/env python3
"""Durable all-gate V9.2 evidence written before any outer wrapper check."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import step28_v13_v1_13_quality_gate_registry_v9_2 as registry


VERSION = "2026-08-23-step28-v13-v1-13-quality-complete-evidence-v9-2"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
BINDING_FIELDS = (
    "root_manifest_sha256",
    "quality_policy_canonical_self_hash",
    "source_bundle_sha256",
    "train_label_vector_sha256",
    "development_label_vector_sha256",
    "train_full_row_order_sha256",
    "development_full_row_order_sha256",
    "train_eligible_row_order_sha256",
    "development_eligible_row_order_sha256",
    "train_text_eligibility_mask_sha256",
    "development_text_eligibility_mask_sha256",
    "model_constants_sha256",
    "bootstrap_constants_sha256",
    "bootstrap_draw_matrix_sha256",
)
HASH_FIELDS = (
    "input_matrix_sha256",
    "prediction_vector_sha256",
    "bootstrap_family_maxima_vector_sha256",
)
EVIDENCE_FIELDS = (
    "version",
    "status",
    "claim_boundary",
    "complete_quality_calculation",
    "gate_registry_sha256",
    "gate_entry_count",
    "ordered_gate_ids",
    "failed_gate_ids",
    "bindings",
    "entries",
    "audit_a_b_truth_open_count",
    "row_level_labels_returned",
    "row_level_predictions_returned",
    "formal_500_by_4_generated",
    "training_started",
    "canonical_self_hash",
)


class CompleteEvidenceV92Error(ValueError):
    """Raised before incomplete or order-drifted evidence can be published."""


def _canonical_json_bytes(value: object) -> bytes:
    return registry.canonical_json_bytes(value)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _required_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise CompleteEvidenceV92Error(f"{field} is not a real SHA-256")
    return value


def _not_applicable_reason(spec: Mapping[str, Any], field: str) -> str:
    if spec["family"] == "structure":
        return {
            "input_matrix_sha256": "STRUCTURE_GATE_HAS_NO_SUPERVISED_MATRIX",
            "prediction_vector_sha256": "STRUCTURE_GATE_HAS_NO_PREDICTION_VECTOR",
            "bootstrap_family_maxima_vector_sha256": (
                "STRUCTURE_GATE_HAS_NO_BOOTSTRAP_VECTOR"
            ),
        }[field]
    if field == "prediction_vector_sha256":
        return "SINGLE_FEATURE_SCAN_HAS_NO_MODEL_PREDICTION_VECTOR"
    if field == "bootstrap_family_maxima_vector_sha256":
        if spec["qualification_role"] == registry.DESCRIPTIVE:
            return "DESCRIPTIVE_MODEL_HAS_NO_FAMILY_GATE_BOOTSTRAP_VECTOR"
        return "NON_BOOTSTRAP_GATE_HAS_NO_BOOTSTRAP_FAMILY_MAXIMA_VECTOR"
    raise CompleteEvidenceV92Error("Unexpected not-applicable matrix requirement")


def _compare(observed: float, threshold: float, comparison: str) -> bool:
    if comparison == "LESS_OR_EQUAL":
        return observed <= threshold
    if comparison == "GREATER_OR_EQUAL":
        return observed >= threshold
    if comparison == "EQUAL":
        return observed == threshold
    raise CompleteEvidenceV92Error("Unknown gate comparison")


def _normalize_observation(
    spec: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    gate_id = str(spec["gate_id"])
    if str(observation.get("gate_id", "")) != gate_id:
        raise CompleteEvidenceV92Error("Gate observation identity drift")
    output: dict[str, Any] = {
        "gate_id": gate_id,
        "family": spec["family"],
        "qualification_role": spec["qualification_role"],
        "metric": spec["metric"],
        "observed": observation.get("observed"),
        "threshold": spec["threshold"],
        "comparison": spec["comparison"],
    }
    if spec["qualification_role"] == registry.HARD_GATE:
        observed = observation.get("observed")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
        ):
            raise CompleteEvidenceV92Error("Hard-gate observation is not finite")
        expected_passed = _compare(
            float(observed), float(spec["threshold"]), str(spec["comparison"])
        )
        if type(observation.get("passed")) is not bool or (
            observation["passed"] is not expected_passed
        ):
            raise CompleteEvidenceV92Error("Hard-gate boolean disagrees with threshold")
        expected_status = "PASS" if expected_passed else "FAIL"
        if observation.get("gate_status") != expected_status:
            raise CompleteEvidenceV92Error("Hard-gate status drift")
        output["gate_status"] = expected_status
        output["passed"] = expected_passed
    else:
        if "passed" in observation:
            raise CompleteEvidenceV92Error(
                "Descriptive observation must not carry a pass boolean"
            )
        if observation.get("gate_status") != registry.NOT_APPLICABLE:
            raise CompleteEvidenceV92Error("Descriptive status must be NOT_APPLICABLE")
        if not isinstance(observation.get("observed"), Mapping):
            raise CompleteEvidenceV92Error(
                "Descriptive observation must carry its metric object"
            )
        output["gate_status"] = registry.NOT_APPLICABLE

    requirement_fields = {
        "input_matrix_sha256": "input_matrix",
        "prediction_vector_sha256": "prediction_vector",
        "bootstrap_family_maxima_vector_sha256": (
            "bootstrap_family_maxima_vector"
        ),
    }
    for field, requirement_field in requirement_fields.items():
        reason_field = field.removesuffix("_sha256") + "_not_applicable_reason"
        if spec[requirement_field] == registry.REQUIRED:
            output[field] = _required_hash(observation.get(field), field=field)
            if observation.get(reason_field) is not None:
                raise CompleteEvidenceV92Error(
                    "Applicable evidence unexpectedly has a null reason"
                )
            output[reason_field] = None
        else:
            expected_reason = _not_applicable_reason(spec, field)
            if observation.get(field) is not None or observation.get(
                reason_field
            ) != expected_reason:
                raise CompleteEvidenceV92Error(
                    "Not-applicable evidence hash/reason drift"
                )
            output[field] = None
            output[reason_field] = expected_reason
    return output


def assemble_complete_quality_evidence(
    *,
    observations: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every registry entry once; never return a partial gate set."""

    if set(observations) != set(registry.GATE_IDS):
        raise CompleteEvidenceV92Error("Complete evidence gate keyset drift")
    if set(bindings) != set(BINDING_FIELDS):
        raise CompleteEvidenceV92Error("Complete evidence binding keyset drift")
    normalized_bindings = {
        field: _required_hash(bindings[field], field=field) for field in BINDING_FIELDS
    }
    entries = [
        _normalize_observation(spec, observations[str(spec["gate_id"])])
        for spec in registry.GATE_REGISTRY
    ]
    normalized_observations = {
        str(entry["gate_id"]): entry for entry in entries
    }
    failures = registry.ordered_failed_gate_ids(normalized_observations)
    evidence: dict[str, Any] = {
        "version": VERSION,
        "status": "PASS" if not failures else "DATASET_INVALIDATED",
        "claim_boundary": "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "complete_quality_calculation": True,
        "gate_registry_sha256": registry.GATE_REGISTRY_SHA256,
        "gate_entry_count": len(entries),
        "ordered_gate_ids": list(registry.GATE_IDS),
        "failed_gate_ids": list(failures),
        "bindings": normalized_bindings,
        "entries": entries,
        "audit_a_b_truth_open_count": 0,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    evidence["canonical_self_hash"] = _self_hash(evidence)
    validate_complete_quality_evidence(evidence)
    return evidence


def build_structure_gate_observations(
    metric_values: Mapping[str, int | float],
) -> dict[str, dict[str, Any]]:
    """Convert the complete structure metric map through the shared registry."""

    specs = tuple(
        value for value in registry.GATE_REGISTRY if value["family"] == "structure"
    )
    expected_metrics = {str(value["metric"]) for value in specs}
    if set(metric_values) != expected_metrics:
        raise CompleteEvidenceV92Error("Structure metric keyset drift")
    output: dict[str, dict[str, Any]] = {}
    for spec in specs:
        gate_id = str(spec["gate_id"])
        observed = metric_values[str(spec["metric"])]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
        ):
            raise CompleteEvidenceV92Error("Structure metric is not finite")
        passed = _compare(
            float(observed), float(spec["threshold"]), str(spec["comparison"])
        )
        row: dict[str, Any] = {
            "gate_id": gate_id,
            "observed": observed,
            "gate_status": "PASS" if passed else "FAIL",
            "passed": passed,
        }
        for field in HASH_FIELDS:
            reason_field = field.removesuffix("_sha256") + "_not_applicable_reason"
            row[field] = None
            row[reason_field] = _not_applicable_reason(spec, field)
        output[gate_id] = row
    return output


def validate_complete_quality_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    if set(normalized) != set(EVIDENCE_FIELDS):
        raise CompleteEvidenceV92Error("Complete evidence exact schema drift")
    if normalized.get("canonical_self_hash") != _self_hash(normalized):
        raise CompleteEvidenceV92Error("Complete evidence self-hash drift")
    if (
        normalized.get("version") != VERSION
        or normalized.get("claim_boundary")
        != "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
        or
        normalized.get("complete_quality_calculation") is not True
        or normalized.get("gate_registry_sha256") != registry.GATE_REGISTRY_SHA256
        or normalized.get("gate_entry_count") != len(registry.GATE_REGISTRY)
        or normalized.get("ordered_gate_ids") != list(registry.GATE_IDS)
        or not isinstance(normalized.get("entries"), list)
        or len(normalized["entries"]) != len(registry.GATE_REGISTRY)
    ):
        raise CompleteEvidenceV92Error("Complete evidence registry binding drift")
    observations: dict[str, dict[str, Any]] = {}
    for spec, entry in zip(registry.GATE_REGISTRY, normalized["entries"], strict=True):
        gate_id = str(spec["gate_id"])
        if not isinstance(entry, Mapping) or entry.get("gate_id") != gate_id:
            raise CompleteEvidenceV92Error("Complete evidence entry order drift")
        normalized_entry = _normalize_observation(spec, entry)
        if dict(entry) != normalized_entry:
            raise CompleteEvidenceV92Error("Complete evidence entry exact schema drift")
        observations[gate_id] = normalized_entry
    failures = list(registry.ordered_failed_gate_ids(observations))
    expected_status = "PASS" if not failures else "DATASET_INVALIDATED"
    if (
        normalized.get("failed_gate_ids") != failures
        or normalized.get("status") != expected_status
        or set(normalized.get("bindings", {})) != set(BINDING_FIELDS)
        or normalized.get("audit_a_b_truth_open_count") != 0
        or normalized.get("row_level_labels_returned") != 0
        or normalized.get("row_level_predictions_returned") != 0
        or normalized.get("formal_500_by_4_generated") is not False
        or normalized.get("training_started") is not False
    ):
        raise CompleteEvidenceV92Error("Complete evidence result drift")
    for field in BINDING_FIELDS:
        _required_hash(normalized["bindings"].get(field), field=field)
    return normalized


def publish_complete_evidence_exclusive(
    path: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_complete_quality_evidence(evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(normalized) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    if path.read_bytes() != payload:
        raise CompleteEvidenceV92Error("Published complete evidence replay drift")
    return {
        "path": path.as_posix(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_self_hash": normalized["canonical_self_hash"],
        "status": normalized["status"],
    }


def wrapper_terminal_after_complete_evidence(
    *, evidence: Mapping[str, Any], wrapper_error: BaseException | None
) -> dict[str, Any]:
    normalized = validate_complete_quality_evidence(evidence)
    if wrapper_error is None:
        status = normalized["status"]
    elif normalized["status"] == "DATASET_INVALIDATED":
        status = "DATASET_INVALIDATED_WITH_OUTER_WRAPPER_FAILURE"
    else:
        status = "AUDITOR_EXECUTION_FAILED_PASS_NOT_CERTIFIED"
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "complete_evidence_canonical_self_hash": normalized[
            "canonical_self_hash"
        ],
        "complete_evidence_status": normalized["status"],
        "wrapper_error_type": (
            None if wrapper_error is None else type(wrapper_error).__name__
        ),
        "wrapper_error_message_sha256": (
            None
            if wrapper_error is None
            else hashlib.sha256(str(wrapper_error).encode("utf-8")).hexdigest()
        ),
        "pass_certified": wrapper_error is None and normalized["status"] == "PASS",
        "dataset_invalidation_preserved": normalized["status"]
        == "DATASET_INVALIDATED",
    }
    receipt["canonical_self_hash"] = _self_hash(receipt)
    return receipt
