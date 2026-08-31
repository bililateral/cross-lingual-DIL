#!/usr/bin/env python3
"""Common fail-closed checks for the one-time public-projection authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as projection_common


POLICY_PATH = (
    ROOT
    / "schema/step28_v13_v1_13_v9_4_1_public_projection_authority_policy_v1.json"
)
POLICY_SIZE_BYTES = 7768
POLICY_SHA256 = "4f2744a465f0a7b160be5af1f6bb45a3628a055ddc136c64620bf576ff8c2795"
POLICY_CANONICAL_SELF_HASH = (
    "b8ce8bb732099965fd28d8c4163ac2e2f94cc22aff9a13cd041d3013127f9ad9"
)
PROJECTION_IMPLEMENTATION_COMMIT = "a49151d64b8496a75d28420ac51338b3345d81df"
PROJECTION_IMPLEMENTATION_TREE = "3057823b57ad6b3f9c97743ddc7058c16130f671"
CONSUMPTION_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-consumption-v1"
)
KEY_CLEANUP_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-key-cleanup-v1"
)


class PublicProjectionAuthorityError(ValueError):
    """Raised when the one-time public-projection authority is not exact."""


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


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PublicProjectionAuthorityError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicProjectionAuthorityError(f"Invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PublicProjectionAuthorityError(f"JSON document is not an object: {path}")
    return value


def require_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if not is_sha256(claimed) or canonical_sha256(body) != claimed:
        raise PublicProjectionAuthorityError(f"{label} canonical self-hash drift")


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def resolve(relative_value: str) -> Path:
    normalized = str(relative_value).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or any(":" in part for part in relative.parts)
    ):
        raise PublicProjectionAuthorityError("Unsafe authority path")
    candidate = ROOT.resolve().joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PublicProjectionAuthorityError("Authority path escapes repository") from exc
    return candidate


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file_record(spec: Mapping[str, Any], *, label: str) -> Path:
    if set(spec) != {"path", "size_bytes", "sha256"}:
        raise PublicProjectionAuthorityError(f"{label} file record schema drift")
    path = resolve(str(spec["path"]))
    if not path.is_file() or file_record(path) != dict(spec):
        raise PublicProjectionAuthorityError(f"{label} file bytes drift")
    return path


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def git_tree() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def git_status_lines() -> list[str]:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()


def implementation_file_records(policy: Mapping[str, Any]) -> dict[str, Any]:
    paths = policy["issued_authorization_contract"]["implementation_files_to_pin"]
    return {str(relative): file_record(resolve(str(relative))) for relative in paths}


def _validate_policy_semantics(policy: Mapping[str, Any]) -> None:
    if (
        policy.get("version")
        != "step28-v13-v1.13-v9.4.1-public-projection-authority-policy-v1"
        or policy.get("status")
        != "AUTHORIZATION_IMPLEMENTATION_ONLY_NOT_ISSUED_NO_EXECUTION"
        or policy.get("claim_boundary")
        != "ONE_TIME_LABEL_FREE_PUBLIC_PROJECTION_ONLY"
        or policy.get("projection_implementation_commit")
        != PROJECTION_IMPLEMENTATION_COMMIT
        or policy.get("projection_implementation_tree")
        != PROJECTION_IMPLEMENTATION_TREE
    ):
        raise PublicProjectionAuthorityError("Authority policy identity drift")
    if any(policy["authorization_state"].values()):
        raise PublicProjectionAuthorityError("Authority implementation grants authority")
    issued = policy["issued_authorization_contract"]
    if (
        issued["issuance_ordinal"] != 1
        or issued["key_size_bytes"] != 32
        or len(issued["implementation_files_to_pin"])
        != len(set(issued["implementation_files_to_pin"]))
    ):
        raise PublicProjectionAuthorityError("Issued authority contract drift")
    paths = policy["execution_paths"]
    public_policy = projection_common.load_policy()
    if (
        paths["formal_output_root"] != public_policy["formal_outputs"]["root"]
        or paths["base_subdirectory"]
        != public_policy["formal_outputs"]["base_subdirectory"]
        or paths["identity_subdirectory"]
        != public_policy["formal_outputs"]["identity_subdirectory"]
        or paths["transfer_subdirectory"]
        != public_policy["formal_outputs"]["transfer_subdirectory"]
        or paths["gpu_return_subdirectory"]
        != public_policy["formal_outputs"]["gpu_return_subdirectory"]
    ):
        raise PublicProjectionAuthorityError("Authority/public output boundary drift")
    workflow = policy["workflow"]
    if (
        workflow["commands"]
        != [
            "prepare-windows",
            "encode-linux",
            "finalize-windows",
            "validate-output",
        ]
        or any(
            workflow[key] is not True
            for key in (
                "consume_before_first_formal_row_read",
                "raw_key_deleted_before_projection_preparation",
                "base_and_identity_formal_inputs_use_separate_processes",
                "linux_claim_before_model_load",
                "failure_consumes_attempt",
                "failure_deletes_building_root",
                "failure_deletes_gpu_workspace",
            )
        )
        or any(
            workflow[key] is not False
            for key in (
                "overwrite_allowed",
                "retain_cpu_stage_after_success",
                "retain_transfer_after_success",
                "retain_gpu_return_after_success",
                "supervision_or_audit_truth_allowed",
                "model_training_or_threshold_selection_allowed",
            )
        )
    ):
        raise PublicProjectionAuthorityError("Authority workflow drift")


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != POLICY_PATH.resolve():
        raise PublicProjectionAuthorityError("Only the frozen authority policy is valid")
    raw = path.read_bytes()
    if len(raw) != POLICY_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise PublicProjectionAuthorityError("Authority policy raw-byte pin drift")
    policy = load_json(path)
    require_self_hash(policy, label="authority policy")
    if policy.get("canonical_self_hash") != POLICY_CANONICAL_SELF_HASH:
        raise PublicProjectionAuthorityError("Authority policy self-hash pin drift")
    expected_roles = (
        "implementation_contract",
        "implementation_ready_result",
        "v6_regression_result",
        "public_policy",
        "gpu_policy",
        "training_policy_v3",
        "public_common",
        "gpu_common",
        "base_preparer",
        "gpu_materializer",
        "gpu_encoder",
        "base_finalizer",
        "identity_builder",
        "publication_protocol",
        "implementation_tests",
    )
    registry = policy["projection_registry"]
    if tuple(registry) != expected_roles:
        raise PublicProjectionAuthorityError("Projection registry drift")
    for role, spec in registry.items():
        verify_file_record(spec, label=role)
    regression = load_json(resolve(registry["v6_regression_result"]["path"]))
    require_self_hash(regression, label="V6 regression result")
    if (
        regression.get("was_successful") is not True
        or regression.get("failure_ids") != []
        or regression.get("error_ids") != []
        or regression.get("training_authorized") is not False
        or regression.get("audit_truth_authorized") is not False
    ):
        raise PublicProjectionAuthorityError("V6 regression authority drift")
    _validate_policy_semantics(policy)
    return policy


def issued_paths(policy: Mapping[str, Any]) -> dict[str, Path]:
    issued = policy["issued_authorization_contract"]
    execution = policy["execution_paths"]
    state_root = resolve(execution["state_root"])
    return {
        "authorization": resolve(issued["authorization_path"]),
        "authority_root": resolve(issued["authority_root"]),
        "key": resolve(issued["key_path"]),
        "issuance_claim": resolve(issued["issuance_claim_path"]),
        "issuance_failure": resolve(issued["issuance_failure_path"]),
        "output": resolve(execution["formal_output_root"]),
        "building": resolve(execution["building_root"]),
        "state_root": state_root,
        "consumption": state_root / execution["consumption_receipt"],
        "key_cleanup": state_root / execution["key_cleanup_receipt"],
        "prepared": state_root / execution["prepared_receipt"],
        "linux_claim": state_root / execution["linux_execution_claim"],
        "linux_completion": state_root / execution["linux_execution_completion"],
        "failure": state_root / execution["terminal_failure"],
        "completion": state_root / execution["formal_completion"],
    }


def validate_authorization(
    policy: Mapping[str, Any],
    *,
    require_raw_key: bool,
) -> dict[str, Any]:
    paths = issued_paths(policy)
    if not paths["authorization"].is_file():
        raise PublicProjectionAuthorityError("Public projection authorization is absent")
    auth = load_json(paths["authorization"])
    required = {
        "version",
        "status",
        "issuance_ordinal",
        "implementation_commit",
        "implementation_tree",
        "projection_implementation_commit",
        "projection_implementation_tree",
        "policy_sha256",
        "policy_canonical_self_hash",
        "public_policy_sha256",
        "public_policy_canonical_self_hash",
        "issuance_claim_sha256",
        "implementation_files",
        "authorization_state",
        "key_file",
        "output_root",
        "building_root",
        "state_root",
        "canonical_self_hash",
    }
    issued = policy["issued_authorization_contract"]
    if (
        set(auth) != required
        or auth["version"] != issued["version"]
        or auth["status"] != issued["status"]
        or auth["issuance_ordinal"] != 1
    ):
        raise PublicProjectionAuthorityError("Authorization schema or identity drift")
    require_self_hash(auth, label="public projection authorization")
    public_policy = projection_common.load_policy()
    expected_state = {
        "formal_public_projection_authorized": True,
        "train_development_truth_authorized": False,
        "model_training_authorized": False,
        "audit_a_prediction_authorized": False,
        "audit_a_truth_authorized": False,
        "audit_b_prediction_authorized": False,
        "audit_b_truth_authorized": False,
    }
    if (
        auth["implementation_commit"] != git_head()
        or auth["implementation_tree"] != git_tree()
        or auth["projection_implementation_commit"]
        != PROJECTION_IMPLEMENTATION_COMMIT
        or auth["projection_implementation_tree"] != PROJECTION_IMPLEMENTATION_TREE
        or auth["policy_sha256"] != sha256_file(POLICY_PATH)
        or auth["policy_canonical_self_hash"] != policy["canonical_self_hash"]
        or auth["public_policy_sha256"]
        != sha256_file(projection_common.POLICY_PATH)
        or auth["public_policy_canonical_self_hash"]
        != public_policy["canonical_self_hash"]
        or auth["implementation_files"] != implementation_file_records(policy)
        or auth["authorization_state"] != expected_state
        or auth["output_root"] != policy["execution_paths"]["formal_output_root"]
        or auth["building_root"] != policy["execution_paths"]["building_root"]
        or auth["state_root"] != policy["execution_paths"]["state_root"]
    ):
        raise PublicProjectionAuthorityError("Authorization binding drift")
    claim = load_json(paths["issuance_claim"])
    require_self_hash(claim, label="issuance claim")
    claim_required = {
        "version",
        "status",
        "issuance_ordinal",
        "implementation_commit",
        "implementation_tree",
        "projection_implementation_commit",
        "projection_implementation_tree",
        "policy_path",
        "policy_sha256",
        "policy_canonical_self_hash",
        "authorization_path",
        "authority_root",
        "key_path",
        "key_size_bytes",
        "candidate_draws_at_claim",
        "output_root",
        "building_root",
        "state_root",
        "rerun_authorized",
        "canonical_self_hash",
    }
    if (
        set(claim) != claim_required
        or sha256_file(paths["issuance_claim"]) != auth["issuance_claim_sha256"]
        or claim["version"]
        != "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-issuance-claim-v1"
        or claim["status"] != "PUBLIC_PROJECTION_AUTHORITY_ISSUANCE_CLAIMED"
        or claim["issuance_ordinal"] != 1
        or claim["implementation_commit"] != auth["implementation_commit"]
        or claim["implementation_tree"] != auth["implementation_tree"]
        or claim["projection_implementation_commit"]
        != auth["projection_implementation_commit"]
        or claim["projection_implementation_tree"]
        != auth["projection_implementation_tree"]
        or claim["policy_path"] != POLICY_PATH.relative_to(ROOT).as_posix()
        or claim["policy_sha256"] != auth["policy_sha256"]
        or claim["policy_canonical_self_hash"]
        != auth["policy_canonical_self_hash"]
        or claim["authorization_path"] != issued["authorization_path"]
        or claim["authority_root"] != issued["authority_root"]
        or claim["key_path"] != issued["key_path"]
        or claim["key_size_bytes"] != 32
        or claim["candidate_draws_at_claim"] != 0
        or claim["output_root"] != auth["output_root"]
        or claim["building_root"] != auth["building_root"]
        or claim["state_root"] != auth["state_root"]
        or claim["rerun_authorized"] is not False
    ):
        raise PublicProjectionAuthorityError("Issuance claim binding drift")
    key_spec = auth["key_file"]
    if (
        not isinstance(key_spec, dict)
        or set(key_spec) != {"path", "commitment_sha256"}
        or resolve(str(key_spec["path"])) != paths["key"]
        or not is_sha256(key_spec["commitment_sha256"])
    ):
        raise PublicProjectionAuthorityError("Authorization key schema drift")
    if require_raw_key:
        raw = paths["key"].read_bytes() if paths["key"].is_file() else b""
        if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != key_spec["commitment_sha256"]:
            raise PublicProjectionAuthorityError("Raw projection authority drift")
    return auth


def validate_consumption_claim(
    policy: Mapping[str, Any],
    auth: Mapping[str, Any],
    *,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    resolved_paths = issued_paths(policy) if paths is None else paths
    consumption = validate_receipt(
        resolved_paths["consumption"], label="projection consumption receipt"
    )
    if (
        set(consumption)
        != {
            "version",
            "status",
            "authorization_sha256",
            "authorization_canonical_self_hash",
            "implementation_commit",
            "implementation_tree",
            "key_commitment_sha256",
            "formal_rows_read_at_consumption",
            "supervision_or_audit_truth_read",
            "model_training_authorized",
            "rerun_authorized",
            "canonical_self_hash",
        }
        or consumption.get("version") != CONSUMPTION_VERSION
        or consumption.get("status")
        != "PUBLIC_PROJECTION_AUTHORITY_CONSUMED_BEFORE_FORMAL_ROW_READ"
        or consumption.get("authorization_sha256")
        != sha256_file(resolved_paths["authorization"])
        or consumption.get("authorization_canonical_self_hash")
        != auth["canonical_self_hash"]
        or consumption.get("key_commitment_sha256")
        != auth["key_file"]["commitment_sha256"]
        or consumption.get("implementation_commit") != auth["implementation_commit"]
        or consumption.get("implementation_tree") != auth["implementation_tree"]
        or consumption.get("formal_rows_read_at_consumption") != 0
        or consumption.get("supervision_or_audit_truth_read") is not False
        or consumption.get("model_training_authorized") is not False
        or consumption.get("rerun_authorized") is not False
    ):
        raise PublicProjectionAuthorityError("Projection consumption claim drift")
    return consumption


def validate_consumption(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = issued_paths(policy)
    consumption = validate_consumption_claim(policy, auth, paths=paths)
    cleanup = validate_receipt(
        paths["key_cleanup"], label="projection key-cleanup receipt"
    )
    if (
        set(cleanup)
        != {
            "version",
            "status",
            "consumption_sha256",
            "key_commitment_sha256",
            "raw_key_material_retained",
            "rerun_authorized",
            "canonical_self_hash",
        }
        or cleanup.get("version") != KEY_CLEANUP_VERSION
        or cleanup.get("status")
        != "PUBLIC_PROJECTION_RAW_AUTHORITY_DELETED_BEFORE_PREPARATION"
        or cleanup.get("consumption_sha256") != sha256_file(paths["consumption"])
        or cleanup.get("key_commitment_sha256")
        != auth["key_file"]["commitment_sha256"]
        or cleanup.get("raw_key_material_retained") is not False
        or cleanup.get("rerun_authorized") is not False
        or paths["key"].exists()
    ):
        raise PublicProjectionAuthorityError("Projection consumption lineage drift")
    return consumption, cleanup


def validate_receipt(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PublicProjectionAuthorityError(f"Missing {label}")
    value = load_json(path)
    require_self_hash(value, label=label)
    return value
