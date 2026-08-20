#!/usr/bin/env python3
"""One-shot authorization overlay for the frozen v9 design-quality audit."""

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
import step28_v13_v1_13_quality_channel_policy_v9 as quality_policy_module


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_POLICY_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_quality_audit_authorization_overlay_v9.json"
)
OVERLAY_POLICY_SIZE_BYTES = 5736
OVERLAY_POLICY_SHA256 = (
    "c73d85c09ff37cf4e2876ee017751bfd483dc6ed638ed2e594988cd2ffc5d266"
)
OVERLAY_POLICY_CANONICAL_SELF_HASH = (
    "9947f02b789045ea99920dbe08996a16f1b4223da0da7cd7f065000a91e021c7"
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REVIEWED_AT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
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


def _validate_static_inputs(policy: Mapping[str, Any]) -> dict[str, Any]:
    _verify_file_spec(policy["contract"], label="authorization contract")
    frozen = policy["frozen_quality_contract"]
    if set(frozen) != {
        "policy",
        "runner",
        "validator",
        "base_authorization",
        "mutation_forbidden",
    } or frozen.get("mutation_forbidden") is not True:
        raise QualityAuditAuthorizationError("Frozen quality contract schema drift")
    quality_policy_path = _verify_file_spec(
        frozen["policy"], label="frozen quality policy"
    )
    _verify_file_spec(frozen["runner"], label="frozen quality runner")
    _verify_file_spec(frozen["validator"], label="frozen quality validator")
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
    split_specs = design["split_manifests"]
    if set(split_specs) != set(audit_runner.SPLITS):
        raise QualityAuditAuthorizationError("Split-manifest overlay universe drift")
    split_bindings: dict[str, dict[str, Any]] = {}
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
        split_bindings[split] = binding
    return {
        "quality_policy": dict(quality_policy),
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
        or not temporary_path.name.startswith(".v9_design_preflight_20260820.")
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
        or not payload["review_conversation_url"].startswith(
            external["review_url_prefix"]
        )
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
        if consumed.read_bytes() != receipt.raw:
            raise QualityAuditAuthorizationError(
                "Consumed quality-audit receipt bytes drift"
            )
    except OSError as exc:
        raise QualityAuditAuthorizationError(
            "Quality-audit receipt could not be consumed"
        ) from exc
    return {
        "path": consumed.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }


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
    )
    if isinstance(exc, auditor_failure_types):
        status = "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
    elif isinstance(exc, dataset_gate_types):
        status = "DATASET_INVALIDATED"
    else:
        status = "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
    return {
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


def _execute_frozen_body(quality_policy: Mapping[str, Any]) -> dict[str, Any]:
    state = {"stage": "authorized_overlay_entry"}
    try:
        result = audit_runner._run_authorized_formal_quality_audit(
            policy=quality_policy, state=state
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


def _build_result_artifact(
    *,
    policy: Mapping[str, Any],
    static: Mapping[str, Any],
    receipt: VerifiedQualityAuditReceipt,
    consumed_file: Mapping[str, Any],
    audit_result: Mapping[str, Any],
) -> dict[str, Any]:
    status = audit_result.get("status")
    if status not in {
        "PASS",
        "DATASET_INVALIDATED",
        "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
    }:
        raise QualityAuditAuthorizationError("Frozen audit result status drift")
    if (
        audit_result.get("formal_500_by_4_generated") is not False
        or audit_result.get("training_started") is not False
    ):
        raise QualityAuditAuthorizationError("Frozen audit result claim boundary drift")
    _validate_no_row_level_payload(audit_result)
    if status == "PASS" and audit_result.get(
        "audit_a_b_truth_remained_sealed"
    ) is not True:
        raise QualityAuditAuthorizationError("Audit A/B truth seal claim drift")
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
            "status": "CONSUMED_ONE_TIME_V9_DESIGN_QUALITY_AUDIT_RECEIPT",
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


def _publish_result(
    *, artifact: Mapping[str, Any], result_path: Path, temporary_path: Path
) -> None:
    if result_path.exists() or temporary_path.exists():
        raise QualityAuditAuthorizationError("Quality-audit result already exists")
    payload = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
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
    consumed_file = _consume_receipt(receipt)
    _validate_consumed_receipt(
        receipt=receipt,
        consumed_file=consumed_file,
        policy=policy,
        static=static,
    )
    try:
        result_directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise QualityAuditAuthorizationError(
            "Quality-audit result directory could not be created"
        ) from exc
    audit_result = _execute_frozen_body(static["quality_policy"])
    artifact = _build_result_artifact(
        policy=policy,
        static=static,
        receipt=receipt,
        consumed_file=consumed_file,
        audit_result=audit_result,
    )
    _publish_result(
        artifact=artifact,
        result_path=result_path,
        temporary_path=temporary_path,
    )
    return artifact


def main() -> None:
    if len(sys.argv) != 1:
        raise QualityAuditAuthorizationError(
            "The one-shot quality-audit entry accepts no arguments"
        )
    result = run_quality_audit_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
