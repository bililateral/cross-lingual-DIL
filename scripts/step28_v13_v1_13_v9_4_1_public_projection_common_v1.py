#!/usr/bin/env python3
"""Fail-closed common contract for the V9.4.1 public projection.

This module validates public, label-free authorities only.  It deliberately
contains no authorization object and cannot execute a formal projection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "schema/step28_v13_v1_13_v9_4_1_public_projection_policy_v1.json"
)
POLICY_SIZE_BYTES = 9299
POLICY_SHA256 = "36481777088d95b492d51065b8b6c82ccc6c0ce91844ddbd39e14900c5694932"
POLICY_CANONICAL_SELF_HASH = (
    "bb1849832ffa43efd4135c33e585c81c939843f41a6026e49ae52ddddb0cbaee"
)
EXPECTED_PARENT_COMMIT = "49ffff89e01818e4f1de58a641195a0d8ef95c3e"
SPLITS = ("train", "development", "audit_a", "audit_b")


class PublicProjectionContractError(ValueError):
    """Raised when the frozen label-free projection contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PublicProjectionContractError(
            "Projection path escapes the repository root"
        ) from exc
    return candidate


def verify_file_record(spec: Mapping[str, Any], *, label: str) -> Path:
    expected_keys = {"path", "size_bytes", "sha256"}
    if not expected_keys.issubset(spec):
        raise PublicProjectionContractError(f"Incomplete file record for {label}")
    path = resolve(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise PublicProjectionContractError(f"{label} byte-size drift")
    if sha256_file(path) != str(spec["sha256"]):
        raise PublicProjectionContractError(f"{label} SHA-256 drift")
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicProjectionContractError(f"Invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PublicProjectionContractError(f"JSON authority is not an object: {path}")
    return value


def verify_canonical_self_hash(
    value: Mapping[str, Any], expected: str, *, label: str
) -> None:
    recorded = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if recorded != expected or canonical_sha256(body) != expected:
        raise PublicProjectionContractError(f"{label} canonical self-hash drift")


def _validate_semantics(policy: Mapping[str, Any]) -> None:
    if (
        policy.get("version")
        != "step28-v13-v1.13-v9.4.1-public-projection-v1"
        or policy.get("status")
        != "IMPLEMENTATION_ONLY_NO_FORMAL_PROJECTION_NO_SUPERVISION_NO_TRAINING"
        or policy.get("claim_boundary")
        != "LABEL_FREE_PUBLIC_PROJECTION_IMPLEMENTATION_ONLY"
        or policy.get("parent_training_core_commit") != EXPECTED_PARENT_COMMIT
    ):
        raise PublicProjectionContractError("Projection policy identity drift")
    if tuple(policy["formal_dataset"]["split_order"]) != SPLITS:
        raise PublicProjectionContractError("Formal split order drift")
    if any(policy["authorization_state"].values()):
        raise PublicProjectionContractError(
            "Implementation policy unexpectedly grants formal authority"
        )
    views = policy["view_isolation"]
    base_allowed = tuple(views["base_allowed_roles"])
    identity_allowed = tuple(views["identity_allowed_roles"])
    if base_allowed != (
        "worlds.jsonl",
        "sellers.jsonl",
        "redacted_items.jsonl",
        "model_seller_profiles.jsonl",
        "complete_model_pair_endpoints.csv",
    ):
        raise PublicProjectionContractError("Base-view allowlist drift")
    if identity_allowed != (
        "worlds.jsonl",
        "complete_model_pair_endpoints.csv",
        "identity33_all_pairs.csv",
    ):
        raise PublicProjectionContractError("Identity-view allowlist drift")
    if set(base_allowed) & set(views["base_forbidden_roles"]):
        raise PublicProjectionContractError("Base-view allow/deny overlap")
    if set(identity_allowed) & set(views["identity_forbidden_roles"]):
        raise PublicProjectionContractError("Identity-view allow/deny overlap")
    gpu = policy["gpu_isolation"]
    if tuple(gpu["allowed_transfer_root_files"]) != ("transfer_manifest.json",) or tuple(
        gpu["allowed_transfer_files_per_split"]
    ) != (
        "opaque_unique_texts.jsonl",
        "opaque_seller_text_index.jsonl",
        "opaque_pair_endpoints.csv",
    ) or tuple(gpu["allowed_return_root_files"]) != ("gpu_return_manifest.json",) or tuple(
        gpu["allowed_return_files_per_split"]
    ) != (
        "labse6.npy",
        "labse6_manifest.json",
    ):
        raise PublicProjectionContractError("GPU file universe drift")
    if (
        gpu["temporary_chunk_rows_retained_after_success"] is not False
        or gpu["temporary_embedding_matrix_retained_after_success"] is not False
        or gpu["model_parameters_updated"] is not False
    ):
        raise PublicProjectionContractError("GPU cleanup/update boundary drift")
    feature = policy["feature_contract"]
    if len(feature["legacy18"]) != 18 or len(feature["labse6"]) != 6:
        raise PublicProjectionContractError("Public feature registry width drift")
    if len(set(feature["legacy18"] + feature["labse6"])) != 24:
        raise PublicProjectionContractError("Public feature registry duplicates columns")
    if feature["column_name_hashes"]["base24"] != canonical_sha256(
        feature["legacy18"] + feature["labse6"]
    ):
        raise PublicProjectionContractError("base24 column-name digest drift")
    outputs = policy["formal_outputs"]
    if (
        outputs["base24_shape_per_split"] != [189000, 24]
        or outputs["identity33_shape_per_split"] != [189000, 33]
        or outputs["probability_shape_per_split"] != [189000]
        or outputs["m1_index_shape"] != [189000]
        or outputs["overwrite_allowed"] is not False
        or outputs["retain_transfer_after_final_publication"] is not False
        or outputs["retain_gpu_return_after_final_publication"] is not False
    ):
        raise PublicProjectionContractError("Formal output contract drift")
    future = policy["future_one_time_authorization"]
    if (
        future["required_exact_parent_commit"] != EXPECTED_PARENT_COMMIT
        or future["must_pin_projection_implementation_commit"] is not True
        or future["must_pin_all_implementation_files"] is not True
        or future["must_assert_output_root_absent"] is not True
        or future["may_authorize_projection_only"] is not True
        or future["may_authorize_supervision_or_training"] is not False
        or future["may_authorize_audit_truth"] is not False
    ):
        raise PublicProjectionContractError("Future one-time authority boundary drift")


def _validate_authorities(policy: Mapping[str, Any]) -> None:
    registry = policy["authority_registry"]
    expected_roles = (
        "implementation_contract",
        "training_policy_v3",
        "model_experiment_policy_v1",
        "full_english_compatibility_v2_success_manifest",
        "base24_shared_v2",
        "predecessor_public_projection_v1",
        "predecessor_identity_projection_v1",
        "step7_policy",
        "step7_common",
        "step7_encoder",
        "gpu_policy_v1",
    )
    if tuple(registry) != expected_roles:
        raise PublicProjectionContractError("Projection authority registry drift")
    paths = {
        role: verify_file_record(spec, label=f"projection authority {role}")
        for role, spec in registry.items()
    }
    for role in (
        "training_policy_v3",
        "model_experiment_policy_v1",
        "full_english_compatibility_v2_success_manifest",
        "gpu_policy_v1",
    ):
        value = load_json(paths[role])
        verify_canonical_self_hash(
            value, str(registry[role]["canonical_self_hash"]), label=role
        )
    training = load_json(paths["training_policy_v3"])
    if any(training["authorization_state"].values()):
        raise PublicProjectionContractError("Training V3 unexpectedly grants authority")
    experiment = load_json(paths["model_experiment_policy_v1"])
    qualification = experiment["dataset_qualification"]
    formal = policy["formal_dataset"]
    if (
        qualification["root"] != formal["root"]
        or qualification["root_manifest"]["canonical_self_hash"]
        != formal["root_manifest_canonical_self_hash"]
        or qualification["quality_result"]["sha256"]
        != formal["quality_result_sha256"]
        or qualification["quality_result"]["canonical_self_hash"]
        != formal["quality_result_canonical_self_hash"]
    ):
        raise PublicProjectionContractError("Formal dataset authority drift")
    compatibility = load_json(paths["full_english_compatibility_v2_success_manifest"])
    if (
        compatibility.get("status") != registry[
            "full_english_compatibility_v2_success_manifest"
        ]["required_status"]
        or compatibility.get("complete_733_pair_score_file_exact_byte_match")
        is not True
        or compatibility.get("supervised_labels_or_identity_evidence_read")
        is not False
        or compatibility.get("audit_truth_read") is not False
        or compatibility.get("model_parameters_updated") is not False
    ):
        raise PublicProjectionContractError("English compatibility success drift")
    gpu_policy = load_json(paths["gpu_policy_v1"])
    gpu_parts = gpu_policy["part_contract"]
    gpu_labse = gpu_policy["labse_contract"]
    gpu_payloads = gpu_policy["model_payloads"]
    expected_payloads = experiment["labse_encoding"]["chunk_tokenizer_payloads"]
    if (
        len(gpu_parts["part_ids"]) != len(SPLITS)
        or gpu_parts["seller_count_per_part"]
        != formal["counts_per_split"]["sellers"]
        or gpu_parts["pair_count_per_part"]
        != formal["counts_per_split"]["pairs"]
        or gpu_parts["allowed_root_files"]
        != policy["gpu_isolation"]["allowed_transfer_root_files"]
        or gpu_parts["allowed_part_files"]
        != policy["gpu_isolation"]["allowed_transfer_files_per_split"]
        or gpu_parts["return_root_files"]
        != policy["gpu_isolation"]["allowed_return_root_files"]
        or gpu_parts["return_part_files"]
        != policy["gpu_isolation"]["allowed_return_files_per_split"]
        or gpu_labse["feature_names"] != policy["feature_contract"]["labse6"]
        or gpu_labse["feature_name_canonical_sha256"]
        != policy["feature_contract"]["column_name_hashes"]["labse6"]
        or gpu_labse["output_shape_per_part"] != [
            formal["counts_per_split"]["pairs"],
            6,
        ]
        or gpu_policy["exact_runtime"] != compatibility["exact_runtime"]
        or gpu_policy["loaded_model_state"]
        != compatibility["loaded_model_state"]
        or tuple(gpu_payloads) != tuple(expected_payloads)
        or any(
            gpu_payloads[key] != expected_payloads[key]
            for key in expected_payloads
        )
    ):
        raise PublicProjectionContractError("Public/GPU projection contract drift")


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != POLICY_PATH.resolve():
        raise PublicProjectionContractError("Only the default projection policy is valid")
    raw = path.read_bytes()
    if len(raw) != POLICY_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise PublicProjectionContractError("Projection policy raw-byte pin drift")
    policy = load_json(path)
    verify_canonical_self_hash(
        policy, POLICY_CANONICAL_SELF_HASH, label="public projection policy"
    )
    _validate_semantics(policy)
    _validate_authorities(policy)
    return policy


def require_formal_projection_authorization(_policy: Mapping[str, Any]) -> None:
    raise PublicProjectionContractError(
        "Formal public projection is not authorized by the implementation policy"
    )
