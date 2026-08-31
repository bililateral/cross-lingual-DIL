#!/usr/bin/env python3
"""Consume and execute the one-time label-free public projection in three stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as gpu_encoder
import step28_v13_v1_13_v9_4_1_finalize_base_projection_v1 as base_finalizer
import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2 as identity_builder
import step28_v13_v1_13_v9_4_1_materialize_base_gpu_workspace_v1 as materializer
import step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1 as authority
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as projection_common
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common
import step28_v13_v1_13_v9_4_1_public_projection_protocol_v1 as protocol


CONSUMPTION_VERSION = authority.CONSUMPTION_VERSION
KEY_CLEANUP_VERSION = authority.KEY_CLEANUP_VERSION
PREPARED_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-prepared-v1"
)
LINUX_CLAIM_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-linux-claim-v1"
)
LINUX_COMPLETION_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-linux-completion-v1"
)
FAILURE_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-terminal-failure-v1"
)
COMPLETION_VERSION = (
    "2026-08-31-step28-v13-v1-13-v9-4-1-public-projection-completion-v1"
)
ISOLATED_RUNNER = (
    "scripts/step28_v13_v1_13_v9_4_1_public_projection_isolated_gpu_runner_v1.py"
)
BASE_PREPARE_WORKER = (
    "scripts/step28_v13_v1_13_v9_4_1_public_projection_base_prepare_worker_v1.py"
)
IDENTITY_PREPARE_WORKER = (
    "scripts/step28_v13_v1_13_v9_4_1_public_projection_identity_prepare_worker_v1.py"
)


def _roots(policy: Mapping[str, Any]) -> dict[str, Path]:
    paths = authority.issued_paths(policy)
    execution = policy["execution_paths"]
    building = paths["building"]
    paths.update(
        {
            "cpu": building / execution["cpu_stage_subdirectory"],
            "transfer": building / execution["transfer_subdirectory"],
            "gpu_return": building / execution["gpu_return_subdirectory"],
            "base": building / execution["base_subdirectory"],
            "identity": building / execution["identity_subdirectory"],
        }
    )
    return paths


def _authorization_sha(paths: Mapping[str, Path]) -> str:
    return authority.sha256_file(paths["authorization"])


def _message_sha(exc: BaseException) -> str:
    return hashlib.sha256(str(exc).encode("utf-8")).hexdigest()


def _consumption_was_validly_claimed(
    policy: Mapping[str, Any],
    auth: Mapping[str, Any] | None,
    paths: Mapping[str, Path],
) -> bool:
    if auth is None:
        return False
    try:
        authority.validate_consumption_claim(policy, auth, paths=paths)
    except (OSError, authority.PublicProjectionAuthorityError):
        return False
    return True


def _reject_terminal_or_claimed_entry(
    paths: Mapping[str, Path], *, command: str, linux_phase: bool = False
) -> None:
    terminal_names = ("failure", "output", "completion")
    phase_names = ("linux_claim", "linux_completion") if linux_phase else ()
    existing = [
        name for name in (*terminal_names, *phase_names) if paths[name].exists()
    ]
    if existing:
        raise authority.PublicProjectionAuthorityError(
            f"{command} cannot re-enter terminal or claimed state: {','.join(existing)}"
        )


def _write_terminal_failure(
    policy: Mapping[str, Any],
    auth: Mapping[str, Any] | None,
    *,
    stage: str,
    exc: BaseException,
    workspace: Path | None = None,
) -> None:
    paths = _roots(policy)
    if not _consumption_was_validly_claimed(policy, auth, paths):
        return
    if workspace is not None:
        shutil.rmtree(workspace, ignore_errors=True)
    shutil.rmtree(paths["building"], ignore_errors=True)
    paths["key"].unlink(missing_ok=True)
    payload: dict[str, Any] = {
        "version": FAILURE_VERSION,
        "status": "PUBLIC_PROJECTION_MECHANICAL_FAILURE_NO_MODEL_OR_DATA_CONCLUSION",
        "authorization_sha256": (
            _authorization_sha(paths) if paths["authorization"].is_file() else None
        ),
        "authorization_canonical_self_hash": (
            auth.get("canonical_self_hash") if auth is not None else None
        ),
        "failure_stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": _message_sha(exc),
        "building_root_retained": paths["building"].exists(),
        "gpu_workspace_retained": workspace.exists() if workspace else False,
        "raw_key_material_retained": paths["key"].exists(),
        "formal_output_exists": paths["output"].exists(),
        "supervision_or_audit_truth_read": False,
        "model_training_performed": False,
        "scientific_result_valid": False,
        "rerun_authorized": False,
    }
    payload["canonical_self_hash"] = authority.canonical_sha256(payload)
    try:
        authority.write_json_exclusive(paths["failure"], payload)
    except FileExistsError:
        pass


def _validate_consumption(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return authority.validate_consumption(policy, auth)


def _consume_authority(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _roots(policy)
    consumption: dict[str, Any] = {
        "version": CONSUMPTION_VERSION,
        "status": "PUBLIC_PROJECTION_AUTHORITY_CONSUMED_BEFORE_FORMAL_ROW_READ",
        "authorization_sha256": _authorization_sha(paths),
        "authorization_canonical_self_hash": auth["canonical_self_hash"],
        "implementation_commit": auth["implementation_commit"],
        "implementation_tree": auth["implementation_tree"],
        "key_commitment_sha256": auth["key_file"]["commitment_sha256"],
        "formal_rows_read_at_consumption": 0,
        "supervision_or_audit_truth_read": False,
        "model_training_authorized": False,
        "rerun_authorized": False,
    }
    consumption["canonical_self_hash"] = authority.canonical_sha256(consumption)
    authority.write_json_exclusive(paths["consumption"], consumption)
    paths["key"].unlink()
    cleanup: dict[str, Any] = {
        "version": KEY_CLEANUP_VERSION,
        "status": "PUBLIC_PROJECTION_RAW_AUTHORITY_DELETED_BEFORE_PREPARATION",
        "consumption_sha256": authority.sha256_file(paths["consumption"]),
        "key_commitment_sha256": auth["key_file"]["commitment_sha256"],
        "raw_key_material_retained": paths["key"].exists(),
        "rerun_authorized": False,
    }
    cleanup["canonical_self_hash"] = authority.canonical_sha256(cleanup)
    authority.write_json_exclusive(paths["key_cleanup"], cleanup)
    return _validate_consumption(policy, auth)


def _validate_prepared_full(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _roots(policy)
    prepared = authority.validate_receipt(paths["prepared"], label="prepared receipt")
    public_policy = projection_common.load_policy()
    cpu_manifest, _cpu_parts = base_finalizer.validate_cpu_stage(
        public_policy, paths["cpu"]
    )
    transfer_manifest, _transfer_parts = gpu_encoder.validate_transfer(
        gpu_common.load_policy(), paths["transfer"]
    )
    identity_manifest = identity_builder.validate_identity_output(
        public_policy, paths["identity"]
    )
    prepared_keys = {
        "version",
        "status",
        "authorization_sha256",
        "authorization_canonical_self_hash",
        "consumption_sha256",
        "cpu_manifest_file",
        "cpu_manifest_canonical_self_hash",
        "transfer_manifest_file",
        "transfer_manifest_canonical_self_hash",
        "identity_manifest_file",
        "identity_manifest_canonical_self_hash",
        "supervision_or_audit_truth_read",
        "model_training_performed",
        "formal_output_published",
        "canonical_self_hash",
    }
    if (
        set(prepared) != prepared_keys
        or prepared.get("version") != PREPARED_VERSION
        or prepared.get("status")
        != "PREPARED_UNPUBLISHED_LABEL_FREE_PUBLIC_PROJECTION"
        or prepared.get("authorization_sha256") != _authorization_sha(paths)
        or prepared.get("authorization_canonical_self_hash")
        != auth["canonical_self_hash"]
        or prepared.get("consumption_sha256")
        != authority.sha256_file(paths["consumption"])
        or prepared.get("cpu_manifest_file")
        != authority.file_record(paths["cpu"] / "cpu_stage_manifest.json")
        or prepared.get("cpu_manifest_canonical_self_hash")
        != cpu_manifest["canonical_self_hash"]
        or prepared.get("transfer_manifest_file")
        != authority.file_record(paths["transfer"] / "transfer_manifest.json")
        or prepared.get("transfer_manifest_canonical_self_hash")
        != transfer_manifest["canonical_self_hash"]
        or prepared.get("identity_manifest_file")
        != authority.file_record(
            paths["identity"] / "identity_projection_manifest.json"
        )
        or prepared.get("identity_manifest_canonical_self_hash")
        != identity_manifest["canonical_self_hash"]
        or prepared.get("supervision_or_audit_truth_read") is not False
        or prepared.get("model_training_performed") is not False
        or prepared.get("formal_output_published") is not False
    ):
        raise authority.PublicProjectionAuthorityError("Prepared lineage drift")
    return prepared


def _validate_prepared_for_linux(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = _roots(policy)
    prepared = authority.validate_receipt(paths["prepared"], label="prepared receipt")
    transfer_manifest, transfer_parts = gpu_encoder.validate_transfer(
        gpu_common.load_policy(), paths["transfer"]
    )
    prepared_keys = {
        "version",
        "status",
        "authorization_sha256",
        "authorization_canonical_self_hash",
        "consumption_sha256",
        "cpu_manifest_file",
        "cpu_manifest_canonical_self_hash",
        "transfer_manifest_file",
        "transfer_manifest_canonical_self_hash",
        "identity_manifest_file",
        "identity_manifest_canonical_self_hash",
        "supervision_or_audit_truth_read",
        "model_training_performed",
        "formal_output_published",
        "canonical_self_hash",
    }
    if (
        set(prepared) != prepared_keys
        or prepared.get("version") != PREPARED_VERSION
        or prepared.get("status")
        != "PREPARED_UNPUBLISHED_LABEL_FREE_PUBLIC_PROJECTION"
        or prepared.get("authorization_sha256") != _authorization_sha(paths)
        or prepared.get("authorization_canonical_self_hash")
        != auth["canonical_self_hash"]
        or prepared.get("consumption_sha256")
        != authority.sha256_file(paths["consumption"])
        or prepared.get("transfer_manifest_file")
        != authority.file_record(paths["transfer"] / "transfer_manifest.json")
        or prepared.get("transfer_manifest_canonical_self_hash")
        != transfer_manifest["canonical_self_hash"]
        or prepared.get("supervision_or_audit_truth_read") is not False
        or prepared.get("model_training_performed") is not False
        or prepared.get("formal_output_published") is not False
    ):
        raise authority.PublicProjectionAuthorityError("Linux prepared lineage drift")
    return prepared, transfer_manifest, transfer_parts


def _run_prepare_worker(relative_path: str) -> None:
    subprocess.run(
        [sys.executable, str(authority.resolve(relative_path)), "--run-once"],
        cwd=authority.ROOT,
        check=True,
    )


def prepare_windows() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        raise authority.PublicProjectionAuthorityError(
            "prepare-windows may run only on Windows"
        )
    policy = authority.load_policy()
    paths = _roots(policy)
    _reject_terminal_or_claimed_entry(paths, command="prepare-windows")
    auth: dict[str, Any] | None = None
    stage = "validate_unconsumed_authorization"
    try:
        auth = authority.validate_authorization(policy, require_raw_key=True)
        expected_status = [
            f"?? {paths['authorization'].relative_to(authority.ROOT).as_posix()}"
        ]
        if authority.git_status_lines() != expected_status:
            raise authority.PublicProjectionAuthorityError(
                "Preparation requires the frozen commit plus only its authorization file"
            )
        if any(
            paths[name].exists()
            for name in (
                "output",
                "building",
                "state_root",
                "consumption",
                "failure",
                "completion",
            )
        ):
            raise authority.PublicProjectionAuthorityError(
                "Projection output, staging or state already exists"
            )
        stage = "consume_authority_before_formal_row_read"
        _consume_authority(policy, auth)
        stage = "prepare_label_free_base_inputs"
        _run_prepare_worker(BASE_PREPARE_WORKER)
        public_policy = projection_common.load_policy()
        cpu_manifest, _ = base_finalizer.validate_cpu_stage(
            public_policy, paths["cpu"]
        )
        transfer_manifest, _ = gpu_encoder.validate_transfer(
            gpu_common.load_policy(), paths["transfer"]
        )
        stage = "prepare_label_free_identity_inputs"
        _run_prepare_worker(IDENTITY_PREPARE_WORKER)
        identity_manifest = identity_builder.validate_identity_output(
            public_policy, paths["identity"]
        )
        prepared: dict[str, Any] = {
            "version": PREPARED_VERSION,
            "status": "PREPARED_UNPUBLISHED_LABEL_FREE_PUBLIC_PROJECTION",
            "authorization_sha256": _authorization_sha(paths),
            "authorization_canonical_self_hash": auth["canonical_self_hash"],
            "consumption_sha256": authority.sha256_file(paths["consumption"]),
            "cpu_manifest_file": authority.file_record(
                paths["cpu"] / "cpu_stage_manifest.json"
            ),
            "cpu_manifest_canonical_self_hash": cpu_manifest[
                "canonical_self_hash"
            ],
            "transfer_manifest_file": authority.file_record(
                paths["transfer"] / "transfer_manifest.json"
            ),
            "transfer_manifest_canonical_self_hash": transfer_manifest[
                "canonical_self_hash"
            ],
            "identity_manifest_file": authority.file_record(
                paths["identity"] / "identity_projection_manifest.json"
            ),
            "identity_manifest_canonical_self_hash": identity_manifest[
                "canonical_self_hash"
            ],
            "supervision_or_audit_truth_read": False,
            "model_training_performed": False,
            "formal_output_published": False,
        }
        prepared["canonical_self_hash"] = authority.canonical_sha256(prepared)
        authority.write_json_exclusive(paths["prepared"], prepared)
        _validate_prepared_full(policy, auth)
        return prepared
    except BaseException as exc:
        if _consumption_was_validly_claimed(policy, auth, paths):
            _write_terminal_failure(policy, auth, stage=stage, exc=exc)
        raise


def encode_linux() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise authority.PublicProjectionAuthorityError(
            "encode-linux may run only on Linux"
        )
    policy = authority.load_policy()
    paths = _roots(policy)
    _reject_terminal_or_claimed_entry(
        paths, command="encode-linux", linux_phase=True
    )
    auth: dict[str, Any] | None = None
    workspace: Path | None = None
    stage = "validate_consumed_linux_inputs"
    try:
        auth = authority.validate_authorization(policy, require_raw_key=False)
        _validate_consumption(policy, auth)
        _prepared, transfer_manifest, transfer_parts = _validate_prepared_for_linux(
            policy, auth
        )
        if (
            paths["failure"].exists()
            or paths["linux_claim"].exists()
            or paths["linux_completion"].exists()
            or paths["gpu_return"].exists()
            or paths["output"].exists()
        ):
            raise authority.PublicProjectionAuthorityError(
                "Linux execution is already claimed, terminal or published"
            )
        claim: dict[str, Any] = {
            "version": LINUX_CLAIM_VERSION,
            "status": "PUBLIC_PROJECTION_LINUX_EXECUTION_CLAIMED_ONCE",
            "authorization_sha256": _authorization_sha(paths),
            "authorization_canonical_self_hash": auth["canonical_self_hash"],
            "consumption_sha256": authority.sha256_file(paths["consumption"]),
            "prepared_sha256": authority.sha256_file(paths["prepared"]),
            "transfer_manifest_canonical_self_hash": transfer_manifest[
                "canonical_self_hash"
            ],
            "model_loads_at_claim": 0,
            "rerun_authorized": False,
        }
        claim["canonical_self_hash"] = authority.canonical_sha256(claim)
        authority.write_json_exclusive(paths["linux_claim"], claim)
        stage = "materialize_isolated_gpu_workspace"
        workspace = Path(
            tempfile.mkdtemp(
                prefix="step28-public-projection-gpu-", dir=authority.ROOT.parent
            )
        )
        workspace.rmdir()
        runner_spec = auth["implementation_files"][ISOLATED_RUNNER]
        gpu_policy = gpu_common.load_policy()
        materializer.materialize_workspace(
            gpu_policy, paths["transfer"], workspace, runner_spec
        )
        stage = "run_isolated_gpu_projection"
        subprocess.run(
            [
                sys.executable,
                str(workspace / ISOLATED_RUNNER),
                "--run-once",
            ],
            cwd=workspace,
            check=True,
        )
        stage = "collect_verified_gpu_return"
        materializer.collect_gpu_return(
            gpu_policy,
            paths["transfer"],
            workspace,
            paths["gpu_return"],
            runner_spec,
        )
        gpu_manifest, _ = gpu_encoder.validate_gpu_return(
            gpu_policy,
            transfer_manifest,
            transfer_parts,
            paths["gpu_return"],
        )
        shutil.rmtree(workspace)
        workspace = None
        completion: dict[str, Any] = {
            "version": LINUX_COMPLETION_VERSION,
            "status": "COMPLETED_ONE_TIME_OPAQUE_LINUX_GPU_PROJECTION",
            "authorization_sha256": _authorization_sha(paths),
            "linux_claim_sha256": authority.sha256_file(paths["linux_claim"]),
            "gpu_return_manifest_file": authority.file_record(
                paths["gpu_return"] / "gpu_return_manifest.json"
            ),
            "gpu_return_manifest_canonical_self_hash": gpu_manifest[
                "canonical_self_hash"
            ],
            "gpu_workspace_retained": False,
            "canonical_identifiers_or_split_names_read": False,
            "supervision_or_audit_truth_read": False,
            "model_parameters_updated": False,
            "rerun_authorized": False,
        }
        completion["canonical_self_hash"] = authority.canonical_sha256(completion)
        authority.write_json_exclusive(paths["linux_completion"], completion)
        return completion
    except BaseException as exc:
        if _consumption_was_validly_claimed(policy, auth, paths):
            shutil.rmtree(paths["gpu_return"], ignore_errors=True)
            _write_terminal_failure(
                policy, auth, stage=stage, exc=exc, workspace=workspace
            )
        raise


def _validate_linux_completion(
    policy: Mapping[str, Any], auth: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _roots(policy)
    claim = authority.validate_receipt(paths["linux_claim"], label="Linux claim")
    completion = authority.validate_receipt(
        paths["linux_completion"], label="Linux completion"
    )
    transfer_manifest, transfer_parts = gpu_encoder.validate_transfer(
        gpu_common.load_policy(), paths["transfer"]
    )
    gpu_manifest, _ = gpu_encoder.validate_gpu_return(
        gpu_common.load_policy(),
        transfer_manifest,
        transfer_parts,
        paths["gpu_return"],
    )
    claim_keys = {
        "version",
        "status",
        "authorization_sha256",
        "authorization_canonical_self_hash",
        "consumption_sha256",
        "prepared_sha256",
        "transfer_manifest_canonical_self_hash",
        "model_loads_at_claim",
        "rerun_authorized",
        "canonical_self_hash",
    }
    completion_keys = {
        "version",
        "status",
        "authorization_sha256",
        "linux_claim_sha256",
        "gpu_return_manifest_file",
        "gpu_return_manifest_canonical_self_hash",
        "gpu_workspace_retained",
        "canonical_identifiers_or_split_names_read",
        "supervision_or_audit_truth_read",
        "model_parameters_updated",
        "rerun_authorized",
        "canonical_self_hash",
    }
    if (
        set(claim) != claim_keys
        or set(completion) != completion_keys
        or claim.get("version") != LINUX_CLAIM_VERSION
        or claim.get("status") != "PUBLIC_PROJECTION_LINUX_EXECUTION_CLAIMED_ONCE"
        or claim.get("authorization_sha256") != _authorization_sha(paths)
        or claim.get("authorization_canonical_self_hash")
        != auth["canonical_self_hash"]
        or claim.get("consumption_sha256")
        != authority.sha256_file(paths["consumption"])
        or claim.get("prepared_sha256")
        != authority.sha256_file(paths["prepared"])
        or claim.get("transfer_manifest_canonical_self_hash")
        != transfer_manifest["canonical_self_hash"]
        or claim.get("model_loads_at_claim") != 0
        or claim.get("rerun_authorized") is not False
        or completion.get("version") != LINUX_COMPLETION_VERSION
        or completion.get("status")
        != "COMPLETED_ONE_TIME_OPAQUE_LINUX_GPU_PROJECTION"
        or completion.get("authorization_sha256") != _authorization_sha(paths)
        or completion.get("linux_claim_sha256")
        != authority.sha256_file(paths["linux_claim"])
        or completion.get("gpu_return_manifest_file")
        != authority.file_record(paths["gpu_return"] / "gpu_return_manifest.json")
        or completion.get("gpu_return_manifest_canonical_self_hash")
        != gpu_manifest["canonical_self_hash"]
        or completion.get("gpu_workspace_retained") is not False
        or completion.get("canonical_identifiers_or_split_names_read") is not False
        or completion.get("supervision_or_audit_truth_read") is not False
        or completion.get("model_parameters_updated") is not False
        or completion.get("rerun_authorized") is not False
    ):
        raise authority.PublicProjectionAuthorityError("Linux completion drift")
    return completion


def _completion_payload(
    policy: Mapping[str, Any], auth: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _roots(policy)
    payload: dict[str, Any] = {
        "version": COMPLETION_VERSION,
        "status": "COMPLETED_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION",
        "authorization_sha256": _authorization_sha(paths),
        "authorization_canonical_self_hash": auth["canonical_self_hash"],
        "consumption_sha256": authority.sha256_file(paths["consumption"]),
        "prepared_sha256": authority.sha256_file(paths["prepared"]),
        "linux_completion_sha256": authority.sha256_file(
            paths["linux_completion"]
        ),
        "public_projection_manifest_file": authority.file_record(
            paths["output"] / "public_projection_manifest.json"
        ),
        "public_projection_manifest_canonical_self_hash": manifest[
            "canonical_self_hash"
        ],
        "cpu_stage_retained": False,
        "transfer_retained": False,
        "gpu_return_retained": False,
        "supervision_or_audit_truth_read": False,
        "model_parameters_updated": False,
        "threshold_selected": False,
        "training_authorized": False,
        "rerun_authorized": False,
    }
    payload["canonical_self_hash"] = authority.canonical_sha256(payload)
    return payload


def finalize_windows() -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        raise authority.PublicProjectionAuthorityError(
            "finalize-windows may run only on Windows"
        )
    policy = authority.load_policy()
    paths = _roots(policy)
    if paths["failure"].exists():
        raise authority.PublicProjectionAuthorityError(
            "finalize-windows cannot re-enter a failed attempt"
        )
    auth: dict[str, Any] | None = None
    stage = "validate_completed_gpu_return"
    try:
        auth = authority.validate_authorization(policy, require_raw_key=False)
        _validate_consumption(policy, auth)
        _validate_prepared_full(policy, auth)
        _validate_linux_completion(policy, auth)
        if paths["failure"].exists() or paths["completion"].exists():
            raise authority.PublicProjectionAuthorityError(
                "Projection attempt is already terminal"
            )
        if paths["output"].exists() or paths["base"].exists():
            raise authority.PublicProjectionAuthorityError(
                "Projection output already exists"
            )
        public_policy = projection_common.load_policy()
        stage = "finalize_base24_and_frozen_probabilities"
        base_finalizer.finalize_to_temporary(
            public_policy,
            paths["cpu"],
            paths["transfer"],
            paths["gpu_return"],
            paths["base"],
        )
        stage = "delete_unpublished_intermediates"
        for name in ("cpu", "transfer", "gpu_return"):
            shutil.rmtree(paths[name])
        stage = "cross_bind_and_publish"
        protocol.freeze_combined_manifest(public_policy, paths["building"])
        paths["building"].replace(paths["output"])
        manifest = protocol.validate_publication(public_policy, paths["output"])
        completion = _completion_payload(policy, auth, manifest)
        authority.write_json_exclusive(paths["completion"], completion)
        return completion
    except BaseException as exc:
        if (
            _consumption_was_validly_claimed(policy, auth, paths)
            and auth is not None
            and paths["output"].is_dir()
        ):
            try:
                manifest = protocol.validate_publication(
                    projection_common.load_policy(), paths["output"]
                )
                completion = _completion_payload(policy, auth, manifest)
                if paths["completion"].exists():
                    observed_completion = authority.validate_receipt(
                        paths["completion"], label="projection completion"
                    )
                    if observed_completion != completion:
                        raise authority.PublicProjectionAuthorityError(
                            "Existing projection completion does not match publication"
                        )
                else:
                    authority.write_json_exclusive(paths["completion"], completion)
                return completion
            except BaseException:
                shutil.rmtree(paths["output"], ignore_errors=True)
        if _consumption_was_validly_claimed(policy, auth, paths):
            _write_terminal_failure(policy, auth, stage=stage, exc=exc)
        raise


def validate_output() -> dict[str, Any]:
    policy = authority.load_policy()
    paths = _roots(policy)
    if paths["failure"].exists():
        raise authority.PublicProjectionAuthorityError(
            "Failed public-projection attempt cannot validate as success"
        )
    auth = authority.validate_authorization(policy, require_raw_key=False)
    _validate_consumption(policy, auth)
    manifest = protocol.validate_publication(
        projection_common.load_policy(), paths["output"]
    )
    completion = authority.validate_receipt(
        paths["completion"], label="projection completion"
    )
    expected = _completion_payload(policy, auth, manifest)
    if completion != expected:
        raise authority.PublicProjectionAuthorityError(
            "Projection completion receipt drift"
        )
    return {
        "status": "VALIDATED_COMPLETED_LABEL_FREE_PUBLIC_PROJECTION",
        "public_projection_manifest_canonical_self_hash": manifest[
            "canonical_self_hash"
        ],
        "completion_canonical_self_hash": completion["canonical_self_hash"],
        "supervision_or_audit_truth_read": False,
        "model_parameters_updated": False,
        "training_authorized": False,
    }


def validate_contract() -> dict[str, Any]:
    policy = authority.load_policy()
    return {
        "status": "PASSED_PUBLIC_PROJECTION_AUTHORITY_CONTRACT_NO_EXECUTION",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "formal_projection_authorized": False,
        "formal_projection_executed": False,
        "supervision_or_audit_truth_read": False,
        "model_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-contract",
            "prepare-windows",
            "encode-linux",
            "finalize-windows",
            "validate-output",
        ),
    )
    args = parser.parse_args()
    result = {
        "validate-contract": validate_contract,
        "prepare-windows": prepare_windows,
        "encode-linux": encode_linux,
        "finalize-windows": finalize_windows,
        "validate-output": validate_output,
    }[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
