#!/usr/bin/env python3
"""Fail-closed helpers for the V9.4.1 successor training contract.

This module is supervision-free.  Loading the successor policy may verify
public documents, public model artifacts, and label-free compatibility output;
it must never open train/development supervision or either audit truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as predecessor_common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_model_training_policy_v2.json"
)
EXPECTED_VERSION = "2026-08-30-step28-v13-v1.13-v9.4.1-model-training-policy-v2"
EXPECTED_STATUS = (
    "DESIGN_AMENDMENT_ONLY_NO_PUBLIC_PROJECTION_NO_SUPERVISION_NO_TRAINING_"
    "NO_AUDIT_TRUTH"
)
EXPECTED_POLICY_SIZE_BYTES = 19011
EXPECTED_POLICY_SHA256 = "75805035a937f3ca1b588feaf47afc18814a1fcffb3c0bcea010a1a840749d30"
EXPECTED_POLICY_CANONICAL_SELF_HASH = (
    "0ff8acb323ffca44016a5fe2deae98353c7f9af9664ff72aa3938b29d186524d"
)
EXPECTED_BASE_ALLOWLIST = (
    "worlds.jsonl",
    "sellers.jsonl",
    "redacted_items.jsonl",
    "model_seller_profiles.jsonl",
    "complete_model_pair_endpoints.csv",
)
EXPECTED_IDENTITY_ALLOWLIST = (
    "worlds.jsonl",
    "complete_model_pair_endpoints.csv",
    "identity33_all_pairs.csv",
)
EXPECTED_MODELS = (
    "c0",
    "m0",
    "m1_r01",
    "m1_r02",
    "m1_r03",
    "m1_r04",
    "m1_r05",
    "m2",
    "m3_base",
    "m3_joint",
)
EXPECTED_SIMULTANEOUS_COMPARISONS = (
    "m2_minus_mean_five_individual_m1",
    "m2_minus_m1_r01",
    "m2_minus_m1_r02",
    "m2_minus_m1_r03",
    "m2_minus_m1_r04",
    "m2_minus_m1_r05",
)
EXPECTED_SIMULTANEOUS_MODELS = (
    "m2",
    "m1_r01",
    "m1_r02",
    "m1_r03",
    "m1_r04",
    "m1_r05",
)


class ModelTrainingContractError(ValueError):
    """Raised when the successor training contract does not replay exactly."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelTrainingContractError(f"Expected JSON object: {path}")
    return value


def parse_exact_policy_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) != EXPECTED_POLICY_SIZE_BYTES:
        raise ModelTrainingContractError("Successor policy exact byte-size drift")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_POLICY_SHA256:
        raise ModelTrainingContractError("Successor policy exact SHA-256 drift")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ModelTrainingContractError("Successor policy must be a JSON object")
    unsigned = dict(value)
    claimed = unsigned.pop("canonical_self_hash", None)
    if claimed != EXPECTED_POLICY_CANONICAL_SELF_HASH:
        raise ModelTrainingContractError("Successor policy self-hash field drift")
    if canonical_sha256(unsigned) != claimed:
        raise ModelTrainingContractError("Successor policy canonical self-hash mismatch")
    return value


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise ModelTrainingContractError(f"Missing {label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise ModelTrainingContractError(f"{label} exact byte-size drift")
    if sha256_file(path) != str(spec["sha256"]):
        raise ModelTrainingContractError(f"{label} exact SHA-256 drift")
    return path


def _verify_json_self_hash(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    value = _load_object(path)
    unsigned = dict(value)
    claimed = unsigned.pop("canonical_self_hash", None)
    if claimed != expected or canonical_sha256(unsigned) != claimed:
        raise ModelTrainingContractError(f"{label} canonical self-hash mismatch")
    return value


def _validate_predecessor(policy: Mapping[str, Any]) -> dict[str, Any]:
    pin = policy["predecessor"]
    path = _verify_pin(pin, label="predecessor model policy")
    predecessor = _verify_json_self_hash(
        path,
        str(pin["canonical_self_hash"]),
        label="predecessor model policy",
    )
    inherited = policy["inherited_section_canonical_sha256"]
    for name in (
        "dataset_qualification",
        "model_stage_truth_boundary",
        "feature_contract",
        "frozen_english_reference",
        "frozen_models",
        "folds_and_weights",
        "m1",
        "m3",
        "runtime",
        "thresholds",
        "metric_registry",
        "bootstrap",
    ):
        observed = canonical_sha256(predecessor[name])
        if observed != inherited[name]:
            raise ModelTrainingContractError(f"Inherited predecessor section drift: {name}")
    public_projection = dict(predecessor["public_projection"])
    if canonical_sha256(public_projection) != inherited[
        "public_projection_predecessor_full_historical"
    ]:
        raise ModelTrainingContractError("Historical predecessor public projection drift")
    for removed_name in (
        "allowed_observed_files",
        "m0_forbidden_inputs",
        "identity_stage_forbidden_inputs",
    ):
        if public_projection.pop(removed_name, None) is None:
            raise ModelTrainingContractError(
                f"Superseded public-projection field is absent: {removed_name}"
            )
    if canonical_sha256(public_projection) != inherited[
        "public_projection_effective_remainder"
    ]:
        raise ModelTrainingContractError("Effective public-projection remainder drift")
    m2 = dict(predecessor["m2"])
    if canonical_sha256(m2) != inherited["m2_predecessor_full_historical"]:
        raise ModelTrainingContractError("Historical predecessor M2 section drift")
    if m2.pop("active_offset_clip", None) is None:
        raise ModelTrainingContractError("Superseded M2 active-offset clip is absent")
    if canonical_sha256(m2) != inherited["m2_effective_remainder"]:
        raise ModelTrainingContractError("Effective M2 remainder drift")
    labse_without_fixture = dict(predecessor["labse_encoding"])
    removed = labse_without_fixture.pop("compatibility_fixture", None)
    if not isinstance(removed, dict):
        raise ModelTrainingContractError("Historical invalid fixture is not identifiable")
    if canonical_sha256(labse_without_fixture) != inherited[
        "labse_encoding_without_invalid_fixture"
    ]:
        raise ModelTrainingContractError("Inherited LaBSE contract drift")
    return predecessor


def _validate_full_compatibility(policy: Mapping[str, Any]) -> None:
    authority = policy["authority_registry"]
    for name, spec in authority.items():
        path = _verify_pin(spec, label=name)
        expected_self_hash = spec.get("canonical_self_hash")
        if expected_self_hash is not None:
            value = _verify_json_self_hash(path, str(expected_self_hash), label=name)
            if name == "full_english_compatibility_v2_success_manifest":
                if value.get("status") != spec["required_status"]:
                    raise ModelTrainingContractError("Full compatibility success status drift")
                if not value.get("embedding_matrix_exact_byte_match"):
                    raise ModelTrainingContractError("Full embedding matrix replay did not match")
                if not value.get("complete_733_pair_score_file_exact_byte_match"):
                    raise ModelTrainingContractError("Full score replay did not match")
                forbidden_counts = (
                    "supervised_labels_or_identity_evidence_read",
                    "identity33_read",
                    "controller_or_membership_read",
                    "qrels_or_retrieval_truth_read",
                    "audit_truth_read",
                    "model_parameters_updated",
                    "model_training_or_threshold_selection_performed",
                )
                if any(value.get(field) not in (0, False) for field in forbidden_counts):
                    raise ModelTrainingContractError(
                        "Full compatibility artifact crossed a scientific boundary"
                    )
                runtime = policy["formal_chinese_labse_runtime"]
                observed_runtime = value.get("exact_runtime")
                expected_runtime = {
                    key: runtime[key]
                    for key in (
                        "python",
                        "numpy",
                        "torch",
                        "torch_cuda_runtime",
                        "cudnn_runtime",
                        "transformers",
                        "sentence_transformers",
                        "gpu_name",
                        "gpu_compute_capability",
                        "cublas_workspace_config",
                        "tokenizers_parallelism",
                        "cuda_matmul_allow_tf32",
                        "cudnn_allow_tf32",
                        "deterministic_algorithms_enabled",
                        "cudnn_benchmark",
                        "cudnn_deterministic",
                    )
                }
                if observed_runtime != expected_runtime:
                    raise ModelTrainingContractError(
                        "Formal Chinese LaBSE runtime is not the successful v2 runtime"
                    )
                full_policy = _load_object(
                    resolve(authority["full_english_compatibility_v2_policy"]["path"])
                )
                if value.get("policy_canonical_self_hash") != full_policy.get(
                    "canonical_self_hash"
                ):
                    raise ModelTrainingContractError(
                        "Full compatibility manifest does not bind its exact policy"
                    )
                if value.get("base_model_experiment_policy_canonical_self_hash") != policy[
                    "predecessor"
                ]["canonical_self_hash"]:
                    raise ModelTrainingContractError(
                        "Full compatibility manifest does not bind the predecessor policy"
                    )
                replay = full_policy["full_replay"]
                if value.get("embedding_matrix_shape") != [
                    replay["shared_chunk_count"],
                    replay["labse_dimension"],
                ]:
                    raise ModelTrainingContractError("Full embedding matrix shape drift")
                if value.get("embedding_matrix_dtype") != replay["embedding_dtype"]:
                    raise ModelTrainingContractError("Full embedding matrix dtype drift")
                if value.get("embedding_matrix_sha256") != replay[
                    "embedding_matrix_sha256"
                ]:
                    raise ModelTrainingContractError("Full embedding matrix SHA-256 drift")
                if value.get("expected_score_sha256") != authority[
                    "full_english_compatibility_v2_score"
                ]["sha256"]:
                    raise ModelTrainingContractError("Full compatibility score pin drift")
                if value.get("numeric_tolerance_used") is not False:
                    raise ModelTrainingContractError("Full compatibility used tolerance")
                if value.get("fixture_reselection_used") is not False:
                    raise ModelTrainingContractError("Full compatibility reselected a fixture")
                if value.get("m0_m1_m2_m3_training_authorized") is not False:
                    raise ModelTrainingContractError("Full compatibility authorized training")


def _validate_english_151_inputs(
    policy: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> None:
    replay = policy["english_151_replay"]
    if replay["pair_count"] != 151 or replay["labels_or_identity_evidence_allowed"]:
        raise ModelTrainingContractError("English 151-pair replay boundary drift")
    inputs = replay["inputs"]
    expected_names = {
        "pair_manifest",
        "opaque_pair_manifest",
        "legacy18_features",
        "labse6_scores",
        "m0_model",
        "c0_model",
    }
    if set(inputs) != expected_names:
        raise ModelTrainingContractError("English 151-pair input registry drift")
    for name, spec in inputs.items():
        _verify_pin(spec, label=f"English 151-pair {name}")
    for model_name in ("m0", "c0"):
        old = predecessor["frozen_models"][model_name]
        new = inputs[f"{model_name}_model"]
        if any(new[field] != old[field] for field in ("path", "size_bytes", "sha256")):
            raise ModelTrainingContractError(f"English replay {model_name} payload drift")
    full_score = policy["authority_registry"]["full_english_compatibility_v2_score"]
    replay_score = inputs["labse6_scores"]
    if any(
        replay_score[field] != full_score[field]
        for field in ("size_bytes", "sha256")
    ):
        raise ModelTrainingContractError("English replay does not use full-v2 LaBSE scores")


def _validate_static_contract(policy: Mapping[str, Any]) -> None:
    if policy.get("version") != EXPECTED_VERSION:
        raise ModelTrainingContractError("Successor version drift")
    if policy.get("status") != EXPECTED_STATUS:
        raise ModelTrainingContractError("Successor status drift")
    if policy.get("claim_boundary") != "SUCCESSOR_MACHINE_CONTRACT_NOT_EXECUTION_AUTHORITY":
        raise ModelTrainingContractError("Successor claim boundary drift")

    supersession = policy["supersession"]
    if supersession["success_prerequisites_logical_operator"] != "AND":
        raise ModelTrainingContractError("Compatibility prerequisites are not logical AND")
    if supersession["full_v2_replaces_english_151_replay"]:
        raise ModelTrainingContractError("Full replay cannot replace the 151-pair replay")
    if supersession["v1_fixture_can_authorize_any_later_stage"]:
        raise ModelTrainingContractError("Invalidated v1 fixture retained authority")
    required_superseded = {
        "predecessor.labse_encoding.compatibility_fixture",
        "predecessor.outputs.compatibility_fixture",
        "predecessor.outputs.compatibility_fixture_linux_replay",
        "predecessor.execution_sequence[1].chunk_fixture",
        "predecessor.public_projection.allowed_observed_files",
        "predecessor.public_projection.m0_forbidden_inputs",
        "predecessor.public_projection.identity_stage_forbidden_inputs",
        "predecessor.m2.active_offset_clip",
    }
    if set(supersession["historical_invalidated_fields"]) != required_superseded:
        raise ModelTrainingContractError("Explicit predecessor supersession registry drift")

    allowlists = policy["stage_input_allowlists"]
    if tuple(allowlists["base_public_projection_observed_basenames"]) != EXPECTED_BASE_ALLOWLIST:
        raise ModelTrainingContractError("Base projection positive allowlist drift")
    if tuple(allowlists["identity_public_projection_observed_basenames"]) != EXPECTED_IDENTITY_ALLOWLIST:
        raise ModelTrainingContractError("Identity projection positive allowlist drift")
    if allowlists["base_process_may_open_identity33"]:
        raise ModelTrainingContractError("Base projection may open identity33")
    if allowlists["identity_process_may_open_sellers_items_or_profiles"]:
        raise ModelTrainingContractError("Identity projection may open text/profile inputs")
    if allowlists["either_public_process_may_open_labels_controllers_qrels_or_audit_truth"]:
        raise ModelTrainingContractError("Public projection may open supervision")

    offset = policy["p0_offset_contract"]
    if offset["training_offset_probability_clipping_allowed"]:
        raise ModelTrainingContractError("Training offset probability clipping is forbidden")
    if not offset["beta_all_zero_must_exactly_nest_m0"]:
        raise ModelTrainingContractError("M2/M1 no longer exactly nest M0")
    if offset["log_loss_evaluation_only_clip"] != [1e-15, 0.999999999999999]:
        raise ModelTrainingContractError("Evaluation-only Log Loss clipping drift")
    replay = policy["english_151_replay"]
    if not replay["prediction_csv_float64_roundtrip_must_match_probability_sha256"]:
        raise ModelTrainingContractError("English probability CSV is not numerically bound")
    if not replay["implementation_file_records_required"]:
        raise ModelTrainingContractError("English replay implementation is not recorded")
    if not replay[
        "shared_legacy18_builder_must_recompute_from_frozen_label_free_materials"
    ]:
        raise ModelTrainingContractError("English shared legacy18 reconstruction is disabled")
    if replay["shared_builder_legacy18_matrix_sha256"] != (
        "9404e755bfdccc3b8a624cddd6380f6ebac8455a4a19877f0909008ea5c729f4"
    ):
        raise ModelTrainingContractError("English shared legacy18 matrix pin drift")
    if replay["required_output_subdirectory"] != "english_151_replay_attempt2":
        raise ModelTrainingContractError("English replay output path drift")
    crosswalk = replay["canonical_to_opaque_crosswalk"]
    if (
        crosswalk["construction"]
        != "physical_row_pairs_after_exact_source_file_pins"
        or crosswalk["full_733_pair_count"] != 733
        or crosswalk["valid_151_pair_count"] != 151
        or crosswalk["uncommitted_physical_row_join_allowed"] is not False
    ):
        raise ModelTrainingContractError("English pair crosswalk contract drift")
    if policy["outputs"]["english_151_replay"] != replay[
        "required_output_subdirectory"
    ]:
        raise ModelTrainingContractError("English replay output registry mismatch")
    invalidated = policy["invalidated_prepublication_attempts"]
    if len(invalidated) != 1:
        raise ModelTrainingContractError("Invalidated prepublication attempt registry drift")
    attempt = invalidated[0]
    if (
        attempt.get("attempt_id") != "english_151_replay_attempt1"
        or attempt.get("payload_retained") is not False
        or attempt.get("path_reuse_allowed") is not False
    ):
        raise ModelTrainingContractError("Prepublication attempt boundary drift")

    authorization = policy["authorization_state"]
    if any(value for key, value in authorization.items() if key.endswith("_authorized")):
        raise ModelTrainingContractError("A successor execution capability is already open")
    for field in (
        "m0_m1_m2_m3_training_authorized",
        "audit_a_truth_authorized",
        "audit_b_truth_authorized",
    ):
        if policy[field]:
            raise ModelTrainingContractError(f"Premature authorization: {field}")

    gates = policy["confirmatory_gates"]
    current_evaluator = gates["current_common_evaluator"]
    if (
        current_evaluator["status"]
        != "NOT_IMPLEMENTED_CURRENT_COMMON_FAIL_CLOSED"
        or current_evaluator["audit_a_evaluation_allowed"] is not False
        or current_evaluator["audit_b_evaluation_allowed"] is not False
        or current_evaluator[
            "precomputed_bootstrap_ap_series_may_enter_formal_gate"
        ]
        is not False
        or current_evaluator["future_evaluator_must_be_a_separate_frozen_module"]
        is not True
    ):
        raise ModelTrainingContractError("Current confirmatory evaluator boundary drift")
    if gates["audit_a"]["primary_simultaneous_lower_bound_gt"] != 0.03:
        raise ModelTrainingContractError("Audit A primary gate drift")
    if gates["audit_b"]["primary_simultaneous_lower_bound_gt"] != 0.015:
        raise ModelTrainingContractError("Audit B primary gate drift")
    if not gates["audit_b"]["cannot_rescue_audit_a_failure"]:
        raise ModelTrainingContractError("Audit B may incorrectly rescue Audit A")
    if tuple(gates["six_simultaneous_comparison_ids"]) != EXPECTED_SIMULTANEOUS_COMPARISONS:
        raise ModelTrainingContractError("Six-comparison family drift")
    simultaneous = gates["simultaneous_lower_bound"]
    if not simultaneous["comparison_id_order_must_match_exactly"]:
        raise ModelTrainingContractError("Simultaneous comparison order is not bound")
    if simultaneous["anonymous_precomputed_comparison_arrays_allowed"]:
        raise ModelTrainingContractError("Anonymous confirmatory comparison arrays are allowed")
    if simultaneous["input_interface"] != (
        "model_keyed_ap_point_and_bootstrap_series_internal_comparison_construction"
    ):
        raise ModelTrainingContractError("Confirmatory comparison input interface drift")
    if tuple(simultaneous["model_metric_ids"]) != EXPECTED_SIMULTANEOUS_MODELS:
        raise ModelTrainingContractError("Confirmatory model-metric registry drift")
    if (
        simultaneous["actual_index_matrix_must_be_passed_and_rehashed"] is not True
        or simultaneous[
            "formal_evaluator_must_generate_bootstrap_ap_from_that_matrix"
        ]
        is not True
        or simultaneous[
            "precomputed_bootstrap_ap_series_may_be_passed_to_formal_gate"
        ]
        is not False
        or simultaneous["index_matrix_shape"] != [9999, 500]
        or simultaneous["index_matrix_dtype"] != "<i8"
        or simultaneous["index_matrix_order"] != "C"
    ):
        raise ModelTrainingContractError("Confirmatory bootstrap matrix contract drift")
    predecessor_bootstrap = {
        "audit_a": "617be9200ad55b45eda8b1800989d7e0b50579bb53ecee675713f8ba2cd4c3e4",
        "audit_b": "12565157b109301070a3648989e74a1faab05d015b5ac0dbcd772c38a5a91a87",
    }
    if simultaneous["split_bootstrap_index_sha256"] != predecessor_bootstrap:
        raise ModelTrainingContractError("Confirmatory bootstrap digest registry drift")
    if gates["audit_b"]["required_audit_a_conclusion_status"] != (
        "PASSED_AUDIT_A_CONFIRMATORY_GATE"
    ):
        raise ModelTrainingContractError("Audit B prior-conclusion status drift")
    if gates["audit_b"][
        "current_common_may_evaluate_before_exact_a_conclusion_pin"
    ]:
        raise ModelTrainingContractError("Audit B may run before exact Audit-A pins")
    required_b_pins = {
        "audit_a_conclusion_path",
        "audit_a_conclusion_size_bytes",
        "audit_a_conclusion_sha256",
        "audit_a_conclusion_canonical_self_hash",
        "audit_a_evaluation_parent_path",
        "audit_a_evaluation_parent_size_bytes",
        "audit_a_evaluation_parent_sha256",
    }
    if set(gates["audit_b"]["future_split_specific_authorization_must_pin"]) != (
        required_b_pins
    ):
        raise ModelTrainingContractError("Audit B future parent-pin registry drift")

    registry = policy["required_report_registry"]
    if tuple(registry["models"]) != EXPECTED_MODELS:
        raise ModelTrainingContractError("Required model reporting set drift")
    if set(registry["required_confusion_matrix_objects"]) != {
        "raw_confusion_matrix",
        "world_equal_confusion_matrix",
    }:
        raise ModelTrainingContractError("Confusion-matrix registry is incomplete")
    if registry["threshold_metric_aggregations"] != [
        "raw_rows",
        "world_equal_confusion",
    ]:
        raise ModelTrainingContractError("Threshold metric aggregation registry drift")
    required_retrieval = {
        "map",
        "mrr",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "ndcg_at_10",
    }
    if set(registry["retrieval_metrics"]) != required_retrieval:
        raise ModelTrainingContractError("Retrieval metric registry is incomplete")
    if not registry["average_precision_and_trapezoidal_pr_auc_are_distinct"]:
        raise ModelTrainingContractError("AP and trapezoidal PR-AUC were conflated")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if path != DEFAULT_POLICY:
        raise ModelTrainingContractError("Only the frozen successor policy path is valid")
    policy = parse_exact_policy_bytes(path.read_bytes())
    predecessor = _validate_predecessor(policy)
    _validate_full_compatibility(policy)
    _validate_english_151_inputs(policy, predecessor)
    _validate_static_contract(policy)
    return policy


def validate_p0(p0: np.ndarray | Sequence[float]) -> np.ndarray:
    values = np.asarray(p0, dtype=np.float64)
    if values.dtype.str != "<f8":
        values = values.astype("<f8", copy=False)
    if values.ndim != 1:
        raise ModelTrainingContractError("p0 must be a one-dimensional float64 vector")
    if not np.isfinite(values).all():
        raise ModelTrainingContractError("p0 contains a non-finite value")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ModelTrainingContractError("p0 must lie strictly inside (0,1)")
    return np.ascontiguousarray(values, dtype="<f8")


def raw_logit(p0: np.ndarray | Sequence[float]) -> np.ndarray:
    values = validate_p0(p0)
    result = np.log(values) - np.log1p(-values)
    if not np.isfinite(result).all():
        raise ModelTrainingContractError("Raw p0 logit is non-finite")
    return np.ascontiguousarray(result, dtype="<f8")


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def residual_probabilities(
    p0: np.ndarray | Sequence[float],
    phi: np.ndarray,
    beta: np.ndarray | Sequence[float],
    active: np.ndarray | Sequence[bool],
) -> np.ndarray:
    """Apply an identity residual while preserving exact M0 nesting.

    Inactive rows and every row whose computed residual is exactly zero retain
    the original float64 p0 bytes.  No probability clipping is used here.
    """

    base = validate_p0(p0)
    features = np.asarray(phi, dtype=np.float64)
    coefficients = np.asarray(beta, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if features.ndim != 2 or features.shape[0] != base.shape[0]:
        raise ModelTrainingContractError("phi shape does not align with p0")
    if coefficients.shape != (features.shape[1],):
        raise ModelTrainingContractError("beta shape does not align with phi")
    if mask.shape != base.shape:
        raise ModelTrainingContractError("active mask does not align with p0")
    if not np.isfinite(features).all() or not np.isfinite(coefficients).all():
        raise ModelTrainingContractError("Residual inputs contain non-finite values")
    residual = features @ coefficients
    if not np.isfinite(residual).all():
        raise ModelTrainingContractError("Identity residual is non-finite")
    changed = mask & (residual != 0.0)
    result = base.copy()
    if np.any(changed):
        logits = raw_logit(base[changed]) + residual[changed]
        result[changed] = _stable_sigmoid(logits)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ModelTrainingContractError("Residual probability is invalid")
    return np.ascontiguousarray(result, dtype="<f8")


def require_no_current_confirmatory_evaluator() -> None:
    """Fail closed until a separate frozen evaluator builds AP from real indices.

    The future audit-specific implementation must consume frozen predictions,
    world rows, truth, and the actual paired-world index matrix in one entry
    point.  This common module deliberately exposes no function that can turn
    caller-supplied precomputed AP series into a formal Audit-A/B conclusion.
    """

    raise ModelTrainingContractError(
        "Confirmatory Audit-A/B evaluation is unavailable in current common_v2; "
        "a separate frozen evaluator must generate AP series from the actual "
        "paired-world index matrix before either formal gate can run"
    )
