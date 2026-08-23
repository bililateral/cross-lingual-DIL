#!/usr/bin/env python3
"""One-shot authorization overlay for the frozen V9.1 design-quality audit."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import step28_v13_v1_13_quality_audit_runner_v9 as audit_runner
import step28_v13_v1_13_quality_audit_execution_adapter_v9_1 as execution_adapter
import step28_v13_v1_13_quality_channel_policy_v9 as quality_policy_module


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_POLICY_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_quality_audit_authorization_overlay_v9_1.json"
)
OVERLAY_POLICY_SIZE_BYTES = 7018
OVERLAY_POLICY_SHA256 = (
    "d9da8a2cd25b34a3d58ef5aa88151da4473eefd1190c21e92394b67706ddd442"
)
OVERLAY_POLICY_CANONICAL_SELF_HASH = (
    "3bc4cc61808e15ab087b8e0d9f274c0b55ca4f57d57c469e62938f2dda52df7a"
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REVIEWED_AT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
REVIEW_CONVERSATION_URL_RE = re.compile(
    r"^https://chatgpt\.com/c/[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "version",
    "status",
    "canonical_self_hash",
    "claim_boundary",
    "contract",
    "frozen_quality_contract",
    "design_root",
    "external_receipt",
    "execution",
    "failure_discipline",
}


class QualityAuditAuthorizationError(RuntimeError):
    """Fail-closed error for the one-shot quality-audit authorization layer."""


@dataclass(frozen=True)
class VerifiedQualityAuditReceipt:
    path: Path
    raw: bytes
    size_bytes: int
    sha256: str
    receipt_id: str
    payload: dict[str, Any]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise QualityAuditAuthorizationError(
                    f"{label} contains duplicate keys"
                )
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise QualityAuditAuthorizationError(f"{label} contains a non-finite value")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityAuditAuthorizationError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise QualityAuditAuthorizationError(f"{label} must be a JSON object")
    return value


def _repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise QualityAuditAuthorizationError("Repository path is malformed")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise QualityAuditAuthorizationError("Repository path escaped the root") from exc
    return path


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
        size_bytes = resolved.stat().st_size
        sha256 = _sha256_file(resolved)
    except (OSError, ValueError) as exc:
        raise QualityAuditAuthorizationError("Pinned file is unavailable") from exc
    return {"path": relative, "size_bytes": size_bytes, "sha256": sha256}


def _verify_file_spec(
    spec: Mapping[str, Any], *, base: Path = ROOT, label: str
) -> Path:
    if set(spec) not in (
        {"path", "size_bytes", "sha256"},
        {"path", "size_bytes", "sha256", "canonical_self_hash"},
    ):
        raise QualityAuditAuthorizationError(f"{label} binding schema drift")
    relative = spec.get("path")
    if not isinstance(relative, str) or not relative:
        raise QualityAuditAuthorizationError(f"{label} path drift")
    path = (base / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
        observed_size = path.stat().st_size
        observed_sha = _sha256_file(path)
    except (OSError, ValueError) as exc:
        raise QualityAuditAuthorizationError(f"{label} is unavailable") from exc
    if (
        type(spec.get("size_bytes")) is not int
        or spec["size_bytes"] <= 0
        or observed_size != spec["size_bytes"]
        or not isinstance(spec.get("sha256"), str)
        or HEX_SHA256_RE.fullmatch(spec["sha256"]) is None
        or observed_sha != spec["sha256"]
    ):
        raise QualityAuditAuthorizationError(f"{label} byte binding drift")
    return path


def _manifest_binding(
    *, path: Path, spec: Mapping[str, Any], base: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify_file_spec(spec, base=base, label=label)
    manifest = _strict_json_object(path.read_bytes(), label=label)
    canonical = spec.get("canonical_self_hash")
    if (
        not isinstance(canonical, str)
        or HEX_SHA256_RE.fullmatch(canonical) is None
        or manifest.get("canonical_self_hash") != canonical
        or _canonical_self_hash(manifest) != canonical
    ):
        raise QualityAuditAuthorizationError(f"{label} canonical binding drift")
    binding = {
        **_file_binding(path),
        "canonical_self_hash": canonical,
    }
    return manifest, binding


def _load_overlay_policy() -> dict[str, Any]:
    try:
        raw = OVERLAY_POLICY_PATH.read_bytes()
    except OSError as exc:
        raise QualityAuditAuthorizationError("Authorization overlay is unavailable") from exc
    if (
        len(raw) != OVERLAY_POLICY_SIZE_BYTES
        or hashlib.sha256(raw).hexdigest() != OVERLAY_POLICY_SHA256
    ):
        raise QualityAuditAuthorizationError("Authorization overlay file drift")
    policy = _strict_json_object(raw, label="Authorization overlay")
    if set(policy) != EXPECTED_TOP_LEVEL_KEYS:
        raise QualityAuditAuthorizationError("Authorization overlay schema drift")
    if (
        policy.get("canonical_self_hash")
        != OVERLAY_POLICY_CANONICAL_SELF_HASH
        or _canonical_self_hash(policy) != OVERLAY_POLICY_CANONICAL_SELF_HASH
        or policy.get("status")
        != "IMPLEMENTATION_ONLY_AWAITING_EXTERNAL_ONE_TIME_RECEIPT"
    ):
        raise QualityAuditAuthorizationError("Authorization overlay self-binding drift")
    return policy


def _validate_v9_1_equivalence_binding(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    keys = {
        "status",
        "canonical_self_hash",
        "same_random_authority",
        "unchanged_file_count",
        "changed_structure_file_count",
        "allowed_changed_json_paths",
    }
    if (
        not isinstance(expected, Mapping)
        or set(expected) != keys
        or not isinstance(observed, Mapping)
        or any(observed.get(key) != expected[key] for key in keys)
        or expected["canonical_self_hash"]
        != execution_adapter.EXPECTED_EQUIVALENCE_CANONICAL_SELF_HASH
        or _canonical_self_hash(observed)
        != execution_adapter.EXPECTED_EQUIVALENCE_CANONICAL_SELF_HASH
    ):
        raise QualityAuditAuthorizationError("V9.1 equivalence binding drift")


def _validate_static_inputs(policy: Mapping[str, Any]) -> dict[str, Any]:
    _verify_file_spec(policy["contract"], label="authorization contract")
    if policy.get("claim_boundary") != execution_adapter.EXPECTED_CLAIM_BOUNDARY:
        raise QualityAuditAuthorizationError("Quality-audit claim boundary drift")
    frozen = policy["frozen_quality_contract"]
    if set(frozen) != {
        "policy",
        "runner",
        "validator",
        "execution_adapter",
        "contract_tests",
        "base_authorization",
        "mutation_forbidden",
    } or frozen.get("mutation_forbidden") is not True:
        raise QualityAuditAuthorizationError("Frozen quality contract schema drift")
    quality_policy_path = _verify_file_spec(
        frozen["policy"], label="frozen quality policy"
    )
    _verify_file_spec(frozen["runner"], label="frozen quality runner")
    _verify_file_spec(frozen["validator"], label="frozen quality validator")
    _verify_file_spec(
        frozen["execution_adapter"], label="quality-audit execution adapter"
    )
    _verify_file_spec(frozen["contract_tests"], label="quality-audit contract tests")
    quality_policy = quality_policy_module.load_policy(quality_policy_path)
    if (
        quality_policy.get("canonical_self_hash")
        != frozen["policy"].get("canonical_self_hash")
        or quality_policy.get("authorization") != frozen["base_authorization"]
        or quality_policy["authorization"].get("implementation_and_fixture_tests")
        is not True
        or any(
            value is not False
            for key, value in quality_policy["authorization"].items()
            if key != "implementation_and_fixture_tests"
        )
    ):
        raise QualityAuditAuthorizationError("Frozen base authorization drift")

    design = policy["design_root"]
    expected_design_keys = {
        "path",
        "status",
        "world_count",
        "formal_seed_created",
        "formal_rows_created",
        "training_started",
        "scientific_use_forbidden",
        "root_manifest",
        "split_manifests",
        "data_file_count",
        "data_size_bytes",
        "physical_file_count",
        "physical_size_bytes",
        "v9_1_equivalence_replay",
    }
    if set(design) != expected_design_keys:
        raise QualityAuditAuthorizationError("Design-root overlay schema drift")
    design_root = _repo_path(design["path"])
    root_path = design_root / str(design["root_manifest"]["path"])
    root_manifest, root_binding = _manifest_binding(
        path=root_path,
        spec=design["root_manifest"],
        base=design_root,
        label="design root manifest",
    )
    if (
        root_manifest.get("status") != design["status"]
        or root_manifest.get("world_count") != design["world_count"]
        or root_manifest.get("formal_seed_created")
        is not design["formal_seed_created"]
        or root_manifest.get("formal_rows_created") != design["formal_rows_created"]
        or root_manifest.get("training_started") is not design["training_started"]
        or root_manifest.get("scientific_use_forbidden")
        is not design["scientific_use_forbidden"]
    ):
        raise QualityAuditAuthorizationError("Design-root claim boundary drift")
    expected_equivalence = design["v9_1_equivalence_replay"]
    observed_equivalence = root_manifest.get("v9_1_equivalence_replay")
    _validate_v9_1_equivalence_binding(
        expected_equivalence, observed_equivalence
    )
    split_specs = design["split_manifests"]
    if set(split_specs) != set(audit_runner.SPLITS):
        raise QualityAuditAuthorizationError("Split-manifest overlay universe drift")
    split_bindings: dict[str, dict[str, Any]] = {}
    data_file_count = 0
    data_size_bytes = 0
    for split in audit_runner.SPLITS:
        split_path = design_root / str(split_specs[split]["path"])
        manifest, binding = _manifest_binding(
            path=split_path,
            spec=split_specs[split],
            base=design_root,
            label=f"{split} split manifest",
        )
        if (
            manifest.get("split") != split
            or manifest.get("status") != design["status"]
            or root_manifest["split_manifest_self_hashes"].get(split)
            != manifest.get("canonical_self_hash")
        ):
            raise QualityAuditAuthorizationError(f"{split} split binding drift")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise QualityAuditAuthorizationError(
                f"{split} split data-file registry drift"
            )
        seen_paths: set[str] = set()
        for record in files:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "size_bytes", "sha256", "row_count"}
                or not isinstance(record.get("path"), str)
                or record["path"] in seen_paths
                or type(record.get("size_bytes")) is not int
                or record["size_bytes"] < 0
                or type(record.get("row_count")) is not int
                or record["row_count"] < 0
                or not isinstance(record.get("sha256"), str)
                or HEX_SHA256_RE.fullmatch(record["sha256"]) is None
            ):
                raise QualityAuditAuthorizationError(
                    f"{split} split data-file registry drift"
                )
            seen_paths.add(record["path"])
            data_file_count += 1
            data_size_bytes += record["size_bytes"]
        split_bindings[split] = binding
    physical_file_count = data_file_count + len(split_bindings) + 1
    physical_size_bytes = (
        data_size_bytes
        + sum(value["size_bytes"] for value in split_bindings.values())
        + root_binding["size_bytes"]
    )
    if (
        design["data_file_count"] != data_file_count
        or design["data_size_bytes"] != data_size_bytes
        or design["physical_file_count"] != physical_file_count
        or design["physical_size_bytes"] != physical_size_bytes
    ):
        raise QualityAuditAuthorizationError("Design-root aggregate size drift")
    external = policy["external_receipt"]
    expected_external_keys = {
        "version",
        "pending_path",
        "consumed_name_template",
        "status",
        "required_review_final_line",
        "review_url_prefix",
        "attempt_index",
        "required_capabilities",
        "repository_code_must_not_generate_receipt",
        "consume_before_first_design_view_read",
        "consume_before_first_train_development_truth_read",
        "consume_before_any_audit_a_b_truth_read",
        "reuse_forbidden",
    }
    if (
        set(external) != expected_external_keys
        or external["pending_path"]
        != execution_adapter.PENDING_RECEIPT_RELATIVE_PATH
        or external["status"] != execution_adapter.EXPECTED_RECEIPT_STATUS
        or external["required_review_final_line"]
        != execution_adapter.EXPECTED_REVIEW_FINAL_LINE
        or type(external["attempt_index"]) is not int
        or external["attempt_index"] != execution_adapter.EXPECTED_ATTEMPT_INDEX
        or not isinstance(external["required_capabilities"], dict)
        or any(
            type(value) is not bool
            for value in external["required_capabilities"].values()
        )
        or external["required_capabilities"]
        != execution_adapter.EXPECTED_CAPABILITIES
        or any(
            external[key] is not True
            for key in (
                "repository_code_must_not_generate_receipt",
                "consume_before_first_design_view_read",
                "consume_before_first_train_development_truth_read",
                "consume_before_any_audit_a_b_truth_read",
                "reuse_forbidden",
            )
        )
    ):
        raise QualityAuditAuthorizationError("External receipt boundary drift")
    execution = policy["execution"]
    if (
        execution.get("public_entry") != "run_quality_audit_once"
        or execution.get("public_entry_parameter_count") != 0
        or execution.get("base_policy_passed_unchanged") is not True
        or execution.get("frozen_scientific_parameters_mutated") is not False
        or execution.get("existing_result_or_temporary_path_forbids_run")
        is not True
        or execution.get("atomic_result_publication") is not True
        or execution.get("row_level_labels_returned") != 0
        or execution.get("row_level_predictions_returned") != 0
    ):
        raise QualityAuditAuthorizationError("Execution boundary drift")
    return {
        "quality_policy": dict(quality_policy),
        "design_root": design_root,
        "root_manifest": root_binding,
        "split_manifests": split_bindings,
    }


def _git_identity() -> tuple[str, str]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if status.stdout:
            raise QualityAuditAuthorizationError(
                "Git worktree is not clean for reviewed quality audit"
            )
        values: list[str] = []
        for revision in ("HEAD", "HEAD^{tree}"):
            result = subprocess.run(
                ["git", "rev-parse", revision],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            values.append(result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualityAuditAuthorizationError(
            "Reviewed Git identity cannot be verified"
        ) from exc
    if any(GIT_OBJECT_RE.fullmatch(value) is None for value in values):
        raise QualityAuditAuthorizationError("Reviewed Git identity is malformed")
    return values[0], values[1]


def _receipt_paths(policy: Mapping[str, Any]) -> tuple[Path, str]:
    pending_relative = policy["external_receipt"]["pending_path"]
    pending = _repo_path(pending_relative)
    private_root = (ROOT / "private_custody").resolve()
    if pending.parent != private_root:
        raise QualityAuditAuthorizationError("Quality-audit receipt escaped private custody")
    return pending, pending_relative


def _result_paths(policy: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    execution = policy["execution"]
    result_directory = _repo_path(execution["result_directory"])
    result_path = _repo_path(execution["result_path"])
    temporary_path = _repo_path(execution["temporary_result_path"])
    if (
        result_path.parent != result_directory
        or temporary_path.parent != result_directory.parent
        or result_path.name != "quality_audit_receipt.json"
        or not temporary_path.name.startswith(
            ".v9_1_design_preflight_attempt1_20260822."
        )
    ):
        raise QualityAuditAuthorizationError("Quality-audit result path drift")
    return result_directory, result_path, temporary_path


def _overlay_policy_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_file_binding(OVERLAY_POLICY_PATH),
        "canonical_self_hash": policy["canonical_self_hash"],
    }


def _quality_policy_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    spec = policy["frozen_quality_contract"]["policy"]
    return {
        **_file_binding(_repo_path(spec["path"])),
        "canonical_self_hash": spec["canonical_self_hash"],
    }


def _reviewed_at_is_valid(value: Any) -> bool:
    if not isinstance(value, str) or REVIEWED_AT_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _expected_receipt_bindings(
    *,
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    git_commit: str,
    git_tree: str,
) -> dict[str, Any]:
    design = policy["design_root"]
    external = policy["external_receipt"]
    return {
        "version": external["version"],
        "status": external["status"],
        "claim_boundary": policy["claim_boundary"],
        "review_final_line": external["required_review_final_line"],
        "attempt_index": external["attempt_index"],
        "input_design_root": design["path"],
        "capabilities": copy.deepcopy(external["required_capabilities"]),
        "overlay_policy": _overlay_policy_binding(policy),
        "authorization_entry_source": _file_binding(Path(__file__)),
        "frozen_quality_policy": _quality_policy_binding(policy),
        "frozen_quality_runner": _file_binding(
            _repo_path(policy["frozen_quality_contract"]["runner"]["path"])
        ),
        "frozen_quality_validator": _file_binding(
            _repo_path(policy["frozen_quality_contract"]["validator"]["path"])
        ),
        "quality_audit_execution_adapter": _file_binding(
            _repo_path(
                policy["frozen_quality_contract"]["execution_adapter"]["path"]
            )
        ),
        "root_manifest": copy.deepcopy(static["root_manifest"]),
        "split_manifests": copy.deepcopy(static["split_manifests"]),
        "git_commit": git_commit,
        "git_tree": git_tree,
        "result_path": policy["execution"]["result_path"],
    }


def _validate_receipt_payload(
    *,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    git_commit: str,
    git_tree: str,
) -> str:
    expected_keys = {
        * _expected_receipt_bindings(
            policy=policy,
            static=static,
            git_commit=git_commit,
            git_tree=git_tree,
        ),
        "review_conversation_url",
        "review_response_sha256",
        "reviewed_at_utc",
        "canonical_self_hash",
    }
    if set(payload) != expected_keys:
        raise QualityAuditAuthorizationError("Quality-audit receipt schema drift")
    capabilities = payload.get("capabilities")
    if (
        type(payload.get("attempt_index")) is not int
        or not isinstance(capabilities, dict)
        or set(capabilities) != set(execution_adapter.EXPECTED_CAPABILITIES)
        or any(type(value) is not bool for value in capabilities.values())
    ):
        raise QualityAuditAuthorizationError(
            "Quality-audit receipt JSON type drift"
        )
    unsigned = dict(payload)
    receipt_id = unsigned.pop("canonical_self_hash")
    if (
        not isinstance(receipt_id, str)
        or HEX_SHA256_RE.fullmatch(receipt_id) is None
        or hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != receipt_id
    ):
        raise QualityAuditAuthorizationError("Quality-audit receipt self-hash drift")
    expected = _expected_receipt_bindings(
        policy=policy,
        static=static,
        git_commit=git_commit,
        git_tree=git_tree,
    )
    if any(payload.get(key) != value for key, value in expected.items()):
        raise QualityAuditAuthorizationError("Quality-audit receipt binding drift")
    external = policy["external_receipt"]
    if (
        not isinstance(payload.get("review_conversation_url"), str)
        or REVIEW_CONVERSATION_URL_RE.fullmatch(
            payload["review_conversation_url"]
        )
        is None
        or not isinstance(payload.get("review_response_sha256"), str)
        or HEX_SHA256_RE.fullmatch(payload["review_response_sha256"]) is None
        or not _reviewed_at_is_valid(payload.get("reviewed_at_utc"))
    ):
        raise QualityAuditAuthorizationError("Quality-audit review metadata drift")
    return receipt_id


def _load_and_validate_pending_receipt(
    *, policy: Mapping[str, Any], static: Mapping[str, Any]
) -> VerifiedQualityAuditReceipt:
    pending, _relative = _receipt_paths(policy)
    consumed_pattern = f"{pending.stem}.consumed.*.json"
    if pending.parent.exists() and any(pending.parent.glob(consumed_pattern)):
        raise QualityAuditAuthorizationError(
            "A quality-audit authorization receipt was already consumed"
        )
    try:
        raw = pending.read_bytes()
    except OSError as exc:
        raise QualityAuditAuthorizationError(
            "Quality audit remains unauthorized: exact one-time receipt is absent"
        ) from exc
    payload = _strict_json_object(raw, label="Quality-audit receipt")
    git_commit, git_tree = _git_identity()
    receipt_id = _validate_receipt_payload(
        payload=payload,
        policy=policy,
        static=static,
        git_commit=git_commit,
        git_tree=git_tree,
    )
    return VerifiedQualityAuditReceipt(
        path=pending,
        raw=raw,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        receipt_id=receipt_id,
        payload=dict(payload),
    )


def _consume_receipt(receipt: VerifiedQualityAuditReceipt) -> dict[str, Any]:
    consumed = receipt.path.with_name(
        f"{receipt.path.stem}.consumed.{receipt.sha256}.json"
    )
    consumed_file = {
        "path": consumed.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }
    if consumed.exists():
        raise QualityAuditAuthorizationError(
            "Quality-audit authorization receipt was already consumed"
        )
    try:
        if receipt.path.read_bytes() != receipt.raw:
            raise QualityAuditAuthorizationError(
                "Quality-audit receipt changed before consumption"
            )
        receipt.path.replace(consumed)
    except OSError as exc:
        raise QualityAuditAuthorizationError(
            "Quality-audit receipt could not be consumed"
        ) from exc
    # Do not perform any fallible work after the atomic rename and before
    # returning ownership to the caller.  Full byte and lineage validation is
    # performed immediately by _validate_consumed_receipt.
    return consumed_file


def _validate_consumed_receipt(
    *,
    receipt: VerifiedQualityAuditReceipt,
    consumed_file: Mapping[str, Any],
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
) -> None:
    if set(consumed_file) != {"path", "size_bytes", "sha256"}:
        raise QualityAuditAuthorizationError("Consumed receipt binding schema drift")
    consumed = _repo_path(str(consumed_file["path"]))
    expected = receipt.path.with_name(
        f"{receipt.path.stem}.consumed.{receipt.sha256}.json"
    )
    if (
        receipt.path.exists()
        or consumed != expected
        or consumed_file["size_bytes"] != receipt.size_bytes
        or consumed_file["sha256"] != receipt.sha256
        or consumed.read_bytes() != receipt.raw
    ):
        raise QualityAuditAuthorizationError("Consumed receipt binding drift")
    payload = _strict_json_object(receipt.raw, label="Consumed quality-audit receipt")
    git_commit, git_tree = _git_identity()
    receipt_id = _validate_receipt_payload(
        payload=payload,
        policy=policy,
        static=static,
        git_commit=git_commit,
        git_tree=git_tree,
    )
    if receipt_id != receipt.receipt_id:
        raise QualityAuditAuthorizationError("Consumed receipt identity drift")


def _classified_frozen_body_failure(
    *, state: Mapping[str, str], exc: Exception
) -> dict[str, Any]:
    structure_authorization_bridge_failure = (
        type(exc)
        is audit_runner.structure_aggregator.QualityStructureAggregationError
        and str(exc) == "Formal quality audit remains unauthorized"
    )
    dataset_gate_types = (
        audit_runner.DatasetGateFailure,
        audit_runner.QualityAuditRunnerError,
        audit_runner.preparer.QualityProbePreparationError,
        audit_runner.structure_aggregator.QualityStructureAggregationError,
        audit_runner.validator.QualityProbeDatasetGateError,
        audit_runner.truth_capability.QualityTruthDatasetGateError,
        audit_runner.preparer.channel.QualityChannelViewError,
        audit_runner.preparer.text_views.QualityTextProbeViewError,
    )
    auditor_failure_types = (
        audit_runner.AuditorExecutionFailure,
        audit_runner.truth_capability.QualityTruthAuditorExecutionError,
        execution_adapter.QualityAuditExecutionAdapterError,
    )
    post_consumption_mechanical_runner_failure = (
        type(exc) is audit_runner.QualityAuditRunnerError
        and str(exc)
        in {
            "Root manifest pin drift",
            "Label-free input path is missing or unsafe",
            "Repository authority source path drift",
        }
    )
    if (
        structure_authorization_bridge_failure
        or post_consumption_mechanical_runner_failure
        or isinstance(exc, auditor_failure_types)
    ):
        status = "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
    elif isinstance(exc, dataset_gate_types):
        status = "DATASET_INVALIDATED"
    else:
        status = "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
    receipt = {
        "version": audit_runner.VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "failure_stage": state.get("stage", "authorized_overlay_entry"),
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "input_dataset_retained_at_decision": True,
        "cleanup_required": status == "DATASET_INVALIDATED",
        "cleanup_completed": False,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
    return receipt


def _execute_frozen_body(
    quality_policy: Mapping[str, Any],
    execution: execution_adapter.ConsumedQualityAuditExecution,
) -> dict[str, Any]:
    state = {"stage": "authorized_overlay_entry"}
    try:
        result = execution_adapter.run_authorized_formal_quality_audit(
            policy=quality_policy,
            execution=execution,
            state=state,
        )
    except Exception as exc:
        result = _classified_frozen_body_failure(state=state, exc=exc)
    if not isinstance(result, dict):
        raise QualityAuditAuthorizationError("Frozen audit body returned a non-object")
    return result


def _validate_no_row_level_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"row_level_labels", "row_level_predictions"}:
                raise QualityAuditAuthorizationError(
                    "Frozen audit result exposed row-level payload"
                )
            if key in {
                "row_level_labels_returned",
                "row_level_predictions_returned",
            } and nested != 0:
                raise QualityAuditAuthorizationError(
                    "Frozen audit result exposed row-level payload"
                )
            _validate_no_row_level_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_no_row_level_payload(nested)


def _require_exact_keys(
    value: Any, expected: set[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise QualityAuditAuthorizationError(f"{label} schema drift")
    return value


def _require_canonical_self_hash(
    value: Mapping[str, Any], *, label: str
) -> None:
    observed = value.get("canonical_self_hash")
    if (
        not isinstance(observed, str)
        or HEX_SHA256_RE.fullmatch(observed) is None
        or _canonical_self_hash(value) != observed
    ):
        raise QualityAuditAuthorizationError(f"{label} self-hash drift")


def _declared_data_bindings(static: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(static["design_root"])
    declared: dict[str, dict[str, Any]] = {}
    for split in audit_runner.SPLITS:
        manifest = _strict_json_object(
            (root / split / "split_manifest.json").read_bytes(),
            label=f"{split} result-validation manifest",
        )
        files = manifest.get("files")
        if not isinstance(files, list):
            raise QualityAuditAuthorizationError(
                "Result-validation manifest file registry drift"
            )
        for record in files:
            if not isinstance(record, dict):
                raise QualityAuditAuthorizationError(
                    "Result-validation manifest record drift"
                )
            path = f"{split}/{record.get('path')}"
            declared[path] = {
                "path": path,
                "size_bytes": record.get("size_bytes"),
                "sha256": record.get("sha256"),
            }
    if len(declared) != 72:
        raise QualityAuditAuthorizationError(
            "Result-validation declared file universe drift"
        )
    return declared


def _validate_structure_receipt(
    value: Any, *, require_pass: bool
) -> Mapping[str, Any]:
    receipt = _require_exact_keys(
        value,
        {
            "version",
            "status",
            "claim_boundary",
            "split_receipts",
            "zero_tolerance_counts",
            "gate_failures",
            "truth_label_row_count_read",
            "audit_truth_open_count",
            "audit_truth_read_count",
            "audit_truth_materialized_row_count",
            "forbidden_read_counts",
            "canonical_self_hash",
        },
        label="Structure receipt",
    )
    _require_canonical_self_hash(receipt, label="Structure receipt")
    if (
        receipt["version"] != audit_runner.structure_aggregator.VERSION
        or receipt["status"] not in {"PASS", "DATASET_INVALIDATED"}
        or receipt["claim_boundary"]
        != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
    ):
        raise QualityAuditAuthorizationError("Structure receipt boundary drift")
    splits = _require_exact_keys(
        receipt["split_receipts"],
        set(audit_runner.SPLITS),
        label="Structure split receipts",
    )
    for split in audit_runner.SPLITS:
        split_receipt = _require_exact_keys(
            splits[split],
            {
                "world_count",
                "seller_row_count",
                "registered_code_count",
                "code_character_position_maximum_absolute_deviation",
            },
            label=f"{split} structure receipt",
        )
        if any(
            type(split_receipt[key]) is not int or split_receipt[key] < 0
            for key in ("world_count", "seller_row_count", "registered_code_count")
        ):
            raise QualityAuditAuthorizationError(
                f"{split} structure receipt count drift"
            )
    expected_zero_keys = {
        *audit_runner.structure_aggregator.ZERO_TOLERANCE_FIELDS,
        "prior_world_code_hits",
    }
    zeros = _require_exact_keys(
        receipt["zero_tolerance_counts"],
        expected_zero_keys,
        label="Structure zero-tolerance counts",
    )
    if any(type(value) is not int or value < 0 for value in zeros.values()):
        raise QualityAuditAuthorizationError(
            "Structure zero-tolerance count type drift"
        )
    failures = receipt["gate_failures"]
    if not isinstance(failures, list) or any(
        not isinstance(value, str) for value in failures
    ):
        raise QualityAuditAuthorizationError("Structure gate-failure schema drift")
    forbidden = _require_exact_keys(
        receipt["forbidden_read_counts"],
        {
            "audit_truth",
            "generator_quality_result",
            "candidate_quality_result",
            "view_builder_quality_result",
        },
        label="Structure forbidden-read counts",
    )
    audit_truth = _require_exact_keys(
        forbidden["audit_truth"],
        {"open_count", "read_count", "materialized_row_count"},
        label="Structure audit-truth counts",
    )
    evidence_zero = (
        receipt["truth_label_row_count_read"],
        receipt["audit_truth_open_count"],
        receipt["audit_truth_read_count"],
        receipt["audit_truth_materialized_row_count"],
        *audit_truth.values(),
        forbidden["generator_quality_result"],
        forbidden["candidate_quality_result"],
        forbidden["view_builder_quality_result"],
    )
    if any(type(value) is not int or value != 0 for value in evidence_zero):
        raise QualityAuditAuthorizationError("Structure forbidden-read evidence drift")
    if require_pass and (
        receipt["status"] != "PASS"
        or failures
        or any(zeros.values())
    ):
        raise QualityAuditAuthorizationError("Structure PASS evidence drift")
    if not require_pass and receipt["status"] == "DATASET_INVALIDATED" and not failures:
        raise QualityAuditAuthorizationError("Structure invalidation evidence drift")
    return receipt


def _validate_family_receipt(
    value: Any,
    *,
    family: str,
    expected_model_count: int,
    expected_feature_count: int,
    require_pass: bool,
    overlay_policy: Mapping[str, Any],
    quality_policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
) -> Mapping[str, Any]:
    family_receipt = _require_exact_keys(
        value,
        {
            "version",
            "status",
            "claim_boundary",
            "design_claim_boundary",
            "family",
            "train_world_count",
            "development_world_count",
            "full_pair_count_per_world",
            "eligible_pair_count_per_world",
            "positive_pair_count_per_world",
            "average_precision_baseline",
            "quality_policy_canonical_self_hash",
            "execution_context",
            "input_commitments",
            "single_feature",
            "model_family",
            "bootstrap",
            "gate_checks",
            "gate_failures",
            "truth_loader_call_counts",
            "row_level_labels_returned",
            "row_level_predictions_returned",
            "canonical_self_hash",
        },
        label=f"{family} family receipt",
    )
    _require_canonical_self_hash(
        family_receipt, label=f"{family} family receipt"
    )
    failures = family_receipt["gate_failures"]
    if not isinstance(failures, list) or any(
        not isinstance(value, str) for value in failures
    ):
        raise QualityAuditAuthorizationError(f"{family} gate-failure schema drift")
    expected_status = (
        "INTERNAL_PROBE_PASS_NO_STANDALONE_CLAIM"
        if not failures
        else "INTERNAL_PROBE_GATE_TRIGGERED_NO_STANDALONE_CLAIM"
    )
    if (
        family_receipt["version"] != audit_runner.validator.VERSION
        or family_receipt["status"] != expected_status
        or family_receipt["claim_boundary"]
        != "INTERNAL_FORMAL_PROBE_CALCULATION_NO_STANDALONE_CLAIM"
        or family_receipt["design_claim_boundary"]
        != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
        or family_receipt["family"] != family
        or family_receipt["quality_policy_canonical_self_hash"]
        != quality_policy["canonical_self_hash"]
        or family_receipt["row_level_labels_returned"] != 0
        or family_receipt["row_level_predictions_returned"] != 0
    ):
        raise QualityAuditAuthorizationError(f"{family} family boundary drift")
    context = _require_exact_keys(
        family_receipt["execution_context"],
        {
            "receipt_id",
            "overlay_policy_canonical_self_hash",
            "capabilities_canonical_sha256",
            "root_manifest",
        },
        label=f"{family} execution context",
    )
    if (
        context["receipt_id"] != receipt.receipt_id
        or context["overlay_policy_canonical_self_hash"]
        != overlay_policy["canonical_self_hash"]
        or context["capabilities_canonical_sha256"]
        != execution_adapter.capabilities_canonical_sha256(
            receipt.payload["capabilities"]
        )
        or context["root_manifest"] != static["root_manifest"]
    ):
        raise QualityAuditAuthorizationError(f"{family} execution binding drift")
    single = _require_exact_keys(
        family_receipt["single_feature"],
        {
            "evaluated_feature_count",
            "maximum_symmetric_auc",
            "winner_name",
            "tie_count",
        },
        label=f"{family} single-feature receipt",
    )
    if single["evaluated_feature_count"] != expected_feature_count:
        raise QualityAuditAuthorizationError(f"{family} feature-count drift")
    model_family = _require_exact_keys(
        family_receipt["model_family"],
        {
            "model_count",
            "maximum_symmetric_auc",
            "maximum_symmetric_auc_winner",
            "maximum_symmetric_auc_tie_count",
            "maximum_average_precision_uplift",
            "maximum_average_precision_uplift_winner",
            "maximum_average_precision_uplift_tie_count",
            "models",
        },
        label=f"{family} model-family receipt",
    )
    models = model_family["models"]
    if (
        model_family["model_count"] != expected_model_count
        or not isinstance(models, Mapping)
        or len(models) != expected_model_count
    ):
        raise QualityAuditAuthorizationError(f"{family} model-count drift")
    for model_name, metrics in models.items():
        if not isinstance(model_name, str):
            raise QualityAuditAuthorizationError(f"{family} model-name drift")
        _require_exact_keys(
            metrics,
            {"symmetric_auc", "average_precision", "prediction_vector_sha256"},
            label=f"{family} model metrics",
        )
        if (
            not isinstance(metrics["prediction_vector_sha256"], str)
            or HEX_SHA256_RE.fullmatch(metrics["prediction_vector_sha256"]) is None
        ):
            raise QualityAuditAuthorizationError(
                f"{family} prediction commitment drift"
            )
    bootstrap = _require_exact_keys(
        family_receipt["bootstrap"],
        {
            "replicates",
            "world_count",
            "score_family_size",
            "draws_raw_i8_c_sha256",
            "family_max_symmetric_auc_vector_sha256",
            "family_max_average_precision_uplift_vector_sha256",
            "symmetric_auc_95_upper",
            "average_precision_uplift_95_upper",
        },
        label=f"{family} bootstrap receipt",
    )
    if (
        bootstrap["replicates"] != quality_policy["bootstrap"]["replicates"]
        or bootstrap["world_count"]
        != quality_policy["bootstrap"]["development_world_count"]
        or bootstrap["score_family_size"] != expected_model_count
        or bootstrap["draws_raw_i8_c_sha256"]
        != quality_policy["bootstrap"]["raw_index_matrix_sha256"]
    ):
        raise QualityAuditAuthorizationError(f"{family} bootstrap binding drift")
    gate_names = {
        "maximum_single_feature_symmetric_auc",
        "maximum_family_symmetric_auc",
        "maximum_family_average_precision_uplift",
        "bootstrap_95_upper_symmetric_auc",
        "bootstrap_95_upper_average_precision_uplift",
    }
    gates = _require_exact_keys(
        family_receipt["gate_checks"],
        gate_names,
        label=f"{family} gate checks",
    )
    for gate_name, gate in gates.items():
        gate_value = _require_exact_keys(
            gate,
            {"observed", "maximum_allowed", "passed"},
            label=f"{family} {gate_name} gate",
        )
        if type(gate_value["passed"]) is not bool:
            raise QualityAuditAuthorizationError(f"{family} gate type drift")
    failed_gate_names = sorted(
        (name for name, gate in gates.items() if gate["passed"] is False),
        key=str.encode,
    )
    if failures != failed_gate_names:
        raise QualityAuditAuthorizationError(f"{family} gate evidence drift")
    calls = _require_exact_keys(
        family_receipt["truth_loader_call_counts"],
        {"train", "development", "audit_a", "audit_b"},
        label=f"{family} truth-loader counts",
    )
    if calls != {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0}:
        raise QualityAuditAuthorizationError(f"{family} truth-loader count drift")
    commitments = _require_exact_keys(
        family_receipt["input_commitments"],
        {
            "train",
            "development",
            "train_text_eligibility_sha256",
            "development_text_eligibility_sha256",
        },
        label=f"{family} input commitments",
    )
    expected_views = expected_model_count // 2
    for split in ("train", "development"):
        values = commitments[split]
        if not isinstance(values, list) or len(values) != expected_views:
            raise QualityAuditAuthorizationError(f"{family} view-count drift")
        for item in values:
            binding = _require_exact_keys(
                item, {"view", "sha256"}, label=f"{family} view commitment"
            )
            if (
                not isinstance(binding["view"], str)
                or not isinstance(binding["sha256"], str)
                or HEX_SHA256_RE.fullmatch(binding["sha256"]) is None
            ):
                raise QualityAuditAuthorizationError(
                    f"{family} view commitment drift"
                )
    eligibility_values = (
        commitments["train_text_eligibility_sha256"],
        commitments["development_text_eligibility_sha256"],
    )
    if family == "text":
        if any(
            not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None
            for value in eligibility_values
        ):
            raise QualityAuditAuthorizationError(
                "Text eligibility commitment drift"
            )
    elif eligibility_values != (None, None):
        raise QualityAuditAuthorizationError(
            "Code family eligibility commitment drift"
        )
    if require_pass and (
        failures or any(gate["passed"] is not True for gate in gates.values())
    ):
        raise QualityAuditAuthorizationError(f"{family} PASS evidence drift")
    return family_receipt


def _validate_input_file_scope(
    value: Any,
    *,
    static: Mapping[str, Any],
    supervised: bool,
) -> None:
    scope = _require_exact_keys(
        value,
        {
            "declared_data_file_count",
            "label_free_actual_byte_verified_count",
            "supervised_truth_actual_byte_verified_count",
            "actual_byte_verified_count",
            "actual_byte_verified_paths",
            "actual_byte_verified_binding_sha256",
            "manifest_pin_only_count",
            "manifest_pin_only_paths",
            "manifest_pin_only_binding_sha256",
            "audit_a_b_truth_manifest_pin_only_paths",
            "audit_a_b_truth_actual_byte_read_count",
            "declared_unclassified_count",
            "scope_claim",
        },
        label="Input-file verification scope",
    )
    declared = _declared_data_bindings(static)
    actual = scope["actual_byte_verified_paths"]
    manifest_only = scope["manifest_pin_only_paths"]
    if (
        not isinstance(actual, list)
        or not isinstance(manifest_only, list)
        or actual != sorted(actual, key=str.encode)
        or manifest_only != sorted(manifest_only, key=str.encode)
        or len(set(actual)) != len(actual)
        or len(set(manifest_only)) != len(manifest_only)
        or set(actual) & set(manifest_only)
        or set(actual) | set(manifest_only) != set(declared)
    ):
        raise QualityAuditAuthorizationError("Input-file scope universe drift")
    expected_actual_count = 46 if supervised else 44
    expected_manifest_count = 26 if supervised else 28
    if (
        scope["declared_data_file_count"] != 72
        or scope["label_free_actual_byte_verified_count"] != 44
        or scope["supervised_truth_actual_byte_verified_count"]
        != (2 if supervised else 0)
        or scope["actual_byte_verified_count"] != expected_actual_count
        or len(actual) != expected_actual_count
        or scope["manifest_pin_only_count"] != expected_manifest_count
        or len(manifest_only) != expected_manifest_count
        or scope["audit_a_b_truth_actual_byte_read_count"] != 0
        or scope["declared_unclassified_count"] != 0
        or scope["scope_claim"]
        != "ACTUAL_BYTES_VERIFIED_ONLY_FOR_LISTED_PATHS_OTHER_PATHS_MANIFEST_PIN_ONLY"
    ):
        raise QualityAuditAuthorizationError("Input-file scope count drift")
    audit_truth = [
        "audit_a/private/pair_labels.csv",
        "audit_b/private/pair_labels.csv",
    ]
    if scope["audit_a_b_truth_manifest_pin_only_paths"] != audit_truth:
        raise QualityAuditAuthorizationError("Audit truth scope drift")
    if supervised and not {
        "train/private/pair_labels.csv",
        "development/private/pair_labels.csv",
    } <= set(actual):
        raise QualityAuditAuthorizationError("Supervised truth scope drift")
    for paths, field in (
        (actual, "actual_byte_verified_binding_sha256"),
        (manifest_only, "manifest_pin_only_binding_sha256"),
    ):
        bindings = [declared[path] for path in paths]
        expected_hash = hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()
        if scope[field] != expected_hash:
            raise QualityAuditAuthorizationError("Input-file scope hash drift")


def _validate_supervised_receipt(
    value: Any,
    *,
    require_pass: bool,
    overlay_policy: Mapping[str, Any],
    quality_policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
) -> None:
    supervised = _require_exact_keys(
        value,
        {
            "version",
            "status",
            "claim_boundary",
            "family_receipts",
            "truth_file_access",
            "audit_a_b_truth_remained_sealed",
            "row_level_labels_returned",
            "row_level_predictions_returned",
            "canonical_self_hash",
        },
        label="Supervised receipt",
    )
    _require_canonical_self_hash(supervised, label="Supervised receipt")
    if (
        supervised["version"] != audit_runner.validator.VERSION
        or supervised["status"] not in {"PASS", "DATASET_INVALIDATED"}
        or supervised["claim_boundary"]
        != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
        or supervised["audit_a_b_truth_remained_sealed"] is not True
        or supervised["row_level_labels_returned"] != 0
        or supervised["row_level_predictions_returned"] != 0
    ):
        raise QualityAuditAuthorizationError("Supervised receipt boundary drift")
    families = _require_exact_keys(
        supervised["family_receipts"],
        {"text", "code_and_slot"},
        label="Supervised family receipts",
    )
    text = _validate_family_receipt(
        families["text"],
        family="text",
        expected_model_count=quality_policy["text_probe_family"][
            "total_model_count"
        ],
        expected_feature_count=quality_policy["text_probe_family"][
            "single_feature_count"
        ],
        require_pass=require_pass,
        overlay_policy=overlay_policy,
        quality_policy=quality_policy,
        static=static,
        receipt=receipt,
    )
    code = _validate_family_receipt(
        families["code_and_slot"],
        family="code_and_slot",
        expected_model_count=4,
        expected_feature_count=(
            quality_policy["public_code_probe"]["feature_width"]
            + quality_policy["decoded_slot_probe"]["feature_width"]
        ),
        require_pass=require_pass,
        overlay_policy=overlay_policy,
        quality_policy=quality_policy,
        static=static,
        receipt=receipt,
    )
    expected_supervised_status = (
        "PASS"
        if not text["gate_failures"] and not code["gate_failures"]
        else "DATASET_INVALIDATED"
    )
    if supervised["status"] != expected_supervised_status:
        raise QualityAuditAuthorizationError("Supervised status evidence drift")
    truth = _require_exact_keys(
        supervised["truth_file_access"],
        {
            "version",
            "root_binding",
            "train",
            "development",
            "audit_a",
            "audit_b",
            "row_level_truth_returned_in_receipt",
        },
        label="Truth file-access receipt",
    )
    if (
        truth["version"] != audit_runner.truth_capability.VERSION
        or truth["root_binding"] != static["root_manifest"]
        or truth["row_level_truth_returned_in_receipt"] != 0
    ):
        raise QualityAuditAuthorizationError("Truth receipt root boundary drift")
    declared = _declared_data_bindings(static)
    for split in ("train", "development"):
        access = _require_exact_keys(
            truth[split],
            {
                "split",
                "file_open_count",
                "byte_read_count",
                "materialized_row_count",
                "sha256",
                "split_manifest_self_hash",
            },
            label=f"{split} truth access",
        )
        expected = declared[f"{split}/private/pair_labels.csv"]
        if (
            access["split"] != split
            or access["file_open_count"] != 1
            or access["byte_read_count"] != expected["size_bytes"]
            or access["sha256"] != expected["sha256"]
            or access["split_manifest_self_hash"]
            != static["split_manifests"][split]["canonical_self_hash"]
            or type(access["materialized_row_count"]) is not int
            or access["materialized_row_count"] <= 0
        ):
            raise QualityAuditAuthorizationError(f"{split} truth access drift")
    for split in ("audit_a", "audit_b"):
        access = _require_exact_keys(
            truth[split],
            {"file_open_count", "byte_read_count", "materialized_row_count"},
            label=f"{split} sealed truth access",
        )
        if any(value != 0 for value in access.values()):
            raise QualityAuditAuthorizationError(f"{split} truth seal drift")


def _validate_audit_result(
    *,
    audit_result: Mapping[str, Any],
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
) -> str:
    status = audit_result.get("status")
    if status == "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION":
        failure = _require_exact_keys(
            audit_result,
            {
                "version",
                "status",
                "claim_boundary",
                "failure_stage",
                "exception_type",
                "exception_message_sha256",
                "row_level_labels_returned",
                "row_level_predictions_returned",
                "input_dataset_retained_at_decision",
                "cleanup_required",
                "cleanup_completed",
                "formal_500_by_4_generated",
                "training_started",
                "canonical_self_hash",
            },
            label="Auditor failure receipt",
        )
        _require_canonical_self_hash(failure, label="Auditor failure receipt")
        if (
            failure["version"] != audit_runner.VERSION
            or failure["claim_boundary"]
            != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
            or failure["cleanup_required"] is not False
            or failure["cleanup_completed"] is not False
            or failure["input_dataset_retained_at_decision"] is not True
            or failure["row_level_labels_returned"] != 0
            or failure["row_level_predictions_returned"] != 0
            or failure["formal_500_by_4_generated"] is not False
            or failure["training_started"] is not False
            or not isinstance(failure["failure_stage"], str)
            or not isinstance(failure["exception_type"], str)
            or not isinstance(failure["exception_message_sha256"], str)
            or HEX_SHA256_RE.fullmatch(failure["exception_message_sha256"])
            is None
        ):
            raise QualityAuditAuthorizationError("Auditor failure evidence drift")
        return status
    if status == "DATASET_INVALIDATED" and "supervised" not in audit_result:
        early = _require_exact_keys(
            audit_result,
            {
                "version",
                "status",
                "claim_boundary",
                "structure",
                "input_file_verification_scope",
                "supervised_truth_opened",
                "audit_a_b_truth_open_count",
                "formal_500_by_4_generated",
                "training_started",
                "canonical_self_hash",
            },
            label="Structure-stage invalidation result",
        )
        _require_canonical_self_hash(
            early, label="Structure-stage invalidation result"
        )
        if (
            early["version"] != audit_runner.VERSION
            or early["claim_boundary"]
            != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
            or early["supervised_truth_opened"] is not False
            or early["audit_a_b_truth_open_count"] != 0
            or early["formal_500_by_4_generated"] is not False
            or early["training_started"] is not False
        ):
            raise QualityAuditAuthorizationError(
                "Structure-stage invalidation boundary drift"
            )
        _validate_structure_receipt(early["structure"], require_pass=False)
        _validate_input_file_scope(
            early["input_file_verification_scope"],
            static=static,
            supervised=False,
        )
        return status
    normal = _require_exact_keys(
        audit_result,
        {
            "version",
            "status",
            "claim_boundary",
            "root_manifest",
            "structure",
            "supervised",
            "input_file_verification_scope",
            "audit_a_b_truth_remained_sealed",
            "formal_500_by_4_generated",
            "training_started",
            "canonical_self_hash",
        },
        label="Complete audit result",
    )
    _require_canonical_self_hash(normal, label="Complete audit result")
    if status not in {"PASS", "DATASET_INVALIDATED"}:
        raise QualityAuditAuthorizationError("Complete audit status drift")
    require_pass = status == "PASS"
    if (
        normal["version"] != audit_runner.VERSION
        or normal["claim_boundary"]
        != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
        or normal["root_manifest"] != static["root_manifest"]
        or normal["audit_a_b_truth_remained_sealed"] is not True
        or normal["formal_500_by_4_generated"] is not False
        or normal["training_started"] is not False
    ):
        raise QualityAuditAuthorizationError("Complete audit boundary drift")
    structure = _validate_structure_receipt(
        normal["structure"], require_pass=True
    )
    _validate_supervised_receipt(
        normal["supervised"],
        require_pass=require_pass,
        overlay_policy=policy,
        quality_policy=static["quality_policy"],
        static=static,
        receipt=receipt,
    )
    if not require_pass and (
        structure["status"] == "PASS"
        and normal["supervised"]["status"] == "PASS"
    ):
        raise QualityAuditAuthorizationError("Dataset invalidation evidence drift")
    _validate_input_file_scope(
        normal["input_file_verification_scope"],
        static=static,
        supervised=True,
    )
    return status


def _build_result_artifact(
    *,
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
    consumed_file: Mapping[str, Any],
    audit_result: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_no_row_level_payload(audit_result)
    status = _validate_audit_result(
        audit_result=audit_result,
        policy=policy,
        static=static,
        receipt=receipt,
    )
    git_commit, git_tree = _git_identity()
    if (
        git_commit != receipt.payload["git_commit"]
        or git_tree != receipt.payload["git_tree"]
    ):
        raise QualityAuditAuthorizationError("Git identity drifted during quality audit")
    artifact: dict[str, Any] = {
        "version": policy["version"],
        "status": status,
        "claim_boundary": policy["claim_boundary"],
        "authorization": {
            "status": "CONSUMED_ONE_TIME_V9_1_DESIGN_QUALITY_AUDIT_RECEIPT",
            "receipt_id": receipt.receipt_id,
            "receipt_file": dict(consumed_file),
            "review_final_line": receipt.payload["review_final_line"],
            "review_conversation_url": receipt.payload["review_conversation_url"],
            "review_response_sha256": receipt.payload["review_response_sha256"],
            "reviewed_at_utc": receipt.payload["reviewed_at_utc"],
            "attempt_index": receipt.payload["attempt_index"],
            "capabilities": receipt.payload["capabilities"],
            "git_commit": git_commit,
            "git_tree": git_tree,
        },
        "inputs": {
            "overlay_policy": _overlay_policy_binding(policy),
            "authorization_entry_source": _file_binding(Path(__file__)),
            "frozen_quality_policy": _quality_policy_binding(policy),
            "frozen_quality_runner": _file_binding(
                _repo_path(policy["frozen_quality_contract"]["runner"]["path"])
            ),
            "frozen_quality_validator": _file_binding(
                _repo_path(policy["frozen_quality_contract"]["validator"]["path"])
            ),
            "quality_audit_execution_adapter": _file_binding(
                _repo_path(
                    policy["frozen_quality_contract"]["execution_adapter"]["path"]
                )
            ),
            "root_manifest": static["root_manifest"],
            "split_manifests": static["split_manifests"],
        },
        "audit_result": dict(audit_result),
        "audit_a_b_truth_authorized": False,
        "formal_seed_created": False,
        "formal_500_by_4_generated": False,
        "training_started": False,
        "model_metric_generation_started": False,
    }
    artifact["canonical_self_hash"] = _canonical_self_hash(artifact)
    return artifact


def _result_payload(artifact: Mapping[str, Any]) -> bytes:
    return json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"


def _publish_result(
    *, artifact: Mapping[str, Any], result_path: Path, temporary_path: Path
) -> None:
    if result_path.exists() or temporary_path.exists():
        raise QualityAuditAuthorizationError("Quality-audit result already exists")
    payload = _result_payload(artifact)
    try:
        with temporary_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if result_path.exists():
            raise QualityAuditAuthorizationError("Quality-audit result appeared concurrently")
        os.link(temporary_path, result_path)
        temporary_path.unlink()
        if result_path.read_bytes() != payload:
            raise QualityAuditAuthorizationError("Published quality-audit result drift")
    except OSError as exc:
        raise QualityAuditAuthorizationError(
            "Quality-audit result could not be published"
        ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _wrapper_terminal_failure_artifact(
    *,
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
    consumed_file: Mapping[str, Any],
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Create the only hash-only terminal state after receipt consumption."""

    truth_evidence_status = (
        "NOT_REACHED"
        if stage
        in {
            "atomic_receipt_consumption",
            "consumed_receipt_revalidation",
            "result_directory_creation",
            "consumed_execution_context_build",
        }
        else "UNAVAILABLE_AFTER_WRAPPER_EXCEPTION"
    )
    artifact: dict[str, Any] = {
        "version": policy["version"],
        "status": "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
        "claim_boundary": policy["claim_boundary"],
        "failure_stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "authorization": {
            "status": "CONSUMED_ONE_TIME_V9_1_DESIGN_QUALITY_AUDIT_RECEIPT",
            "receipt_id": receipt.receipt_id,
            "receipt_file": dict(consumed_file),
            "attempt_index": receipt.payload["attempt_index"],
            "capabilities": copy.deepcopy(receipt.payload["capabilities"]),
            "git_commit": receipt.payload["git_commit"],
            "git_tree": receipt.payload["git_tree"],
        },
        "inputs": {
            "overlay_policy": _overlay_policy_binding(policy),
            "authorization_entry_source": _file_binding(Path(__file__)),
            "frozen_quality_policy": _quality_policy_binding(policy),
            "frozen_quality_runner": _file_binding(
                _repo_path(policy["frozen_quality_contract"]["runner"]["path"])
            ),
            "frozen_quality_validator": _file_binding(
                _repo_path(policy["frozen_quality_contract"]["validator"]["path"])
            ),
            "quality_audit_execution_adapter": _file_binding(
                _repo_path(
                    policy["frozen_quality_contract"]["execution_adapter"]["path"]
                )
            ),
            "root_manifest": copy.deepcopy(static["root_manifest"]),
            "split_manifests": copy.deepcopy(static["split_manifests"]),
        },
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "audit_a_b_truth_authorized": False,
        "truth_access_evidence": {
            "status": truth_evidence_status,
            "train": None,
            "development": None,
            "audit_a": None,
            "audit_b": None,
        },
        "formal_seed_created": False,
        "formal_500_by_4_generated": False,
        "training_started": False,
        "model_metric_generation_started": False,
        "retry_authorized": False,
    }
    artifact["canonical_self_hash"] = _canonical_self_hash(artifact)
    return artifact


def _publish_wrapper_terminal_failure(
    *,
    artifact: Mapping[str, Any],
    result_directory: Path,
    result_path: Path,
    temporary_path: Path,
) -> None:
    if result_path.exists():
        try:
            if result_path.read_bytes() == _result_payload(artifact):
                return
        except OSError as exc:
            raise QualityAuditAuthorizationError(
                "Existing quality-audit terminal result cannot be verified"
            ) from exc
        raise QualityAuditAuthorizationError(
            "Existing quality-audit result differs from terminal failure"
        )
    try:
        result_directory.mkdir(parents=True, exist_ok=True)
        _publish_result(
            artifact=artifact,
            result_path=result_path,
            temporary_path=temporary_path,
        )
    except Exception as publish_exc:
        raise QualityAuditAuthorizationError(
            "Consumed receipt has no publishable terminal failure artifact"
        ) from publish_exc


def run_quality_audit_once() -> dict[str, Any]:
    """Consume one exact external receipt, then run the frozen audit body once."""

    policy = _load_overlay_policy()
    static = _validate_static_inputs(policy)
    result_directory, result_path, temporary_path = _result_paths(policy)
    if result_directory.exists() or result_path.exists() or temporary_path.exists():
        raise QualityAuditAuthorizationError(
            "Quality-audit result or temporary path already exists; resume is forbidden"
        )
    receipt = _load_and_validate_pending_receipt(policy=policy, static=static)
    anticipated_consumed = receipt.path.with_name(
        f"{receipt.path.stem}.consumed.{receipt.sha256}.json"
    )
    consumed_file = {
        "path": anticipated_consumed.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }
    stage = "atomic_receipt_consumption"
    consumed_by_this_invocation = False
    try:
        observed_consumed_file = _consume_receipt(receipt)
        consumed_by_this_invocation = True
        if observed_consumed_file != consumed_file:
            raise QualityAuditAuthorizationError(
                "Consumed quality-audit receipt binding drift"
            )
        stage = "consumed_receipt_revalidation"
        _validate_consumed_receipt(
            receipt=receipt,
            consumed_file=consumed_file,
            policy=policy,
            static=static,
        )
        stage = "result_directory_creation"
        try:
            result_directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise QualityAuditAuthorizationError(
                "Quality-audit result directory could not be created"
            ) from exc
        stage = "consumed_execution_context_build"
        execution = execution_adapter.build_consumed_execution(
            receipt_id=receipt.receipt_id,
            overlay_policy_canonical_self_hash=policy["canonical_self_hash"],
            base_policy=static["quality_policy"],
            capabilities=receipt.payload["capabilities"],
            pending_receipt_path=receipt.path,
            consumed_receipt_binding=consumed_file,
            result_path=result_path,
            dataset_root=static["design_root"],
            root_manifest_binding=static["root_manifest"],
        )
        stage = "quality_audit_body"
        audit_result = _execute_frozen_body(
            static["quality_policy"], execution
        )
        stage = "quality_audit_result_validation"
        artifact = _build_result_artifact(
            policy=policy,
            static=static,
            receipt=receipt,
            consumed_file=consumed_file,
            audit_result=audit_result,
        )
        stage = "exclusive_result_publication"
        _publish_result(
            artifact=artifact,
            result_path=result_path,
            temporary_path=temporary_path,
        )
        return artifact
    except BaseException as exc:
        if consumed_by_this_invocation:
            terminal = _wrapper_terminal_failure_artifact(
                policy=policy,
                static=static,
                receipt=receipt,
                consumed_file=consumed_file,
                stage=stage,
                exc=exc,
            )
            _publish_wrapper_terminal_failure(
                artifact=terminal,
                result_directory=result_directory,
                result_path=result_path,
                temporary_path=temporary_path,
            )
            if isinstance(exc, Exception):
                return terminal
        raise


def main() -> None:
    if len(sys.argv) != 1:
        raise QualityAuditAuthorizationError(
            "The one-shot quality-audit entry accepts no arguments"
        )
    result = run_quality_audit_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
