#!/usr/bin/env python3
"""Shared fail-closed utilities for the Step 28-v13 shortcut audit."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import step28_v13_common as dataset_common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = (
    ROOT
    / "schema"
    / "step28_v13_metadata_shortcut_audit_lock.json"
)
LOCK_VERSION = (
    "2026-07-29-step28-v13-metadata-shortcut-"
    "core-implementation-lock-v3"
)
PROJECTION_MANIFEST_VERSION = (
    "2026-07-29-step28-v13-null-nuisance-"
    "projection-manifest-v1"
)
LABEL_MANIFEST_VERSION = (
    "2026-07-29-step28-v13-classification-label-"
    "manifest-v1"
)
LABEL_FORMULA_RECEIPT_VERSION = (
    "2026-07-29-step28-v13-label-formula-validation-v1"
)
AUDIT_REPORT_VERSION = (
    "2026-07-29-step28-v13-metadata-shortcut-report-v1"
)
AUDIT_MANIFEST_VERSION = (
    "2026-07-29-step28-v13-metadata-shortcut-"
    "audit-manifest-v1"
)
BLOCKED_ACCESS_STATUS = "UNVERIFIED_FORMAL_EXECUTION_BLOCKED"
LABEL_FORMULA_ALTERNATIVE_DERIVATION = (
    "enumerate every within-controller seller combination, "
    "then test C40 pair-set membership"
)
PROJECTION_FILENAME = "null_nuisance_pairs.csv"
PROJECTION_MANIFEST_FILENAME = "projection_manifest.json"
LABEL_FILENAME = "classification_labels.csv"
LABEL_MANIFEST_FILENAME = "label_manifest.json"
LABEL_VALIDATION_FILENAME = "label_formula_validation.json"
AUDIT_REPORT_FILENAME = "metadata_shortcut_audit.json"
OOF_FILENAME = "metadata_shortcut_oof.private.csv"
AUDIT_MANIFEST_FILENAME = "audit_manifest.json"
FAILURE_MANIFEST_FILENAME = "failure_manifest.json"
SPLITS = ("train", "development", "audit_a", "audit_b")
PRETRAIN_AUDIT_SPLITS = ("train", "development")

SELLER_FEATURES = (
    "item_count",
    "title_missing_rate",
    "description_missing_rate",
    "time_bucket_probability_00",
    "time_bucket_probability_01",
    "time_bucket_probability_02",
    "time_bucket_probability_03",
)
PAIR_FEATURES = tuple(
    f"absdiff__{name}" for name in SELLER_FEATURES
) + tuple(f"sum__{name}" for name in SELLER_FEATURES)
PROJECTION_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    *PAIR_FEATURES,
)
CANDIDATE_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
REDACTED_ITEM_FIELDS = {
    "description",
    "item_uid",
    "seller_uid",
    "title",
    "world_uid",
}
HISTORY_ITEM_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
)
MEMBERSHIP_FIELDS = (
    "world_uid",
    "controller_uid",
    "seller_uid",
)
LABEL_FIELDS = ("canonical_pair_uid", "label")
OOF_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "label",
    "fold",
    "score_logistic_l2",
    "score_gradient_tree",
    "score_rbf_svm",
)


def prior_requirements(split: str) -> list[dict[str, str]]:
    train_pass = {
        "role": "train_metadata_shortcut_pass",
        "split": "train",
        "operation": "metadata_shortcut_audit",
        "status": "PASS_METADATA_SHORTCUT_ONLY",
    }
    development_pass = {
        "role": "development_metadata_shortcut_pass",
        "split": "development",
        "operation": "metadata_shortcut_audit",
        "status": "PASS_METADATA_SHORTCUT_ONLY",
    }
    if split == "train":
        return []
    if split == "development":
        return [train_pass]
    blind = {
        "role": f"{split}_blind_predictions_frozen",
        "split": split,
        "operation": "blind_prediction_freeze",
        "status": "BLIND_PREDICTIONS_FROZEN",
    }
    unseal = {
        "role": f"{split}_unseal_authorized",
        "split": split,
        "operation": "sequential_unseal",
        "status": (
            "AUDIT_A_SEQUENTIAL_RELEASE_AUTHORIZED"
            if split == "audit_a"
            else "AUDIT_B_SEQUENTIAL_RELEASE_AUTHORIZED"
        ),
    }
    output = [train_pass, development_pass, blind]
    if split == "audit_b":
        output.append(
            {
                "role": "audit_a_overall_pass",
                "split": "audit_a",
                "operation": "overall_dataset_gate",
                "status": "PASS_A_ONLY",
            }
        )
    output.append(unseal)
    return output


class ShortcutAuditError(RuntimeError):
    """Raised when a shortcut-audit contract fails closed."""


class DurabilityUnknownError(ShortcutAuditError):
    """Raised after rename when parent-directory fsync cannot confirm it."""

    def __init__(self, target: Path) -> None:
        super().__init__(
            "Artifact was renamed into place but parent fsync failed; "
            f"durability is unknown: {target}"
        )
        self.target = target


@dataclass(frozen=True)
class FileSnapshot:
    """Exact bytes used by one worker, with their immutable record."""

    basename: str
    size_bytes: int
    sha256: str
    payload: bytes

    def record(self, *, role: str) -> dict[str, Any]:
        return {
            "role": role,
            "basename": self.basename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _is_reparse_or_link(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(
        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    return stat.S_ISLNK(stat_result.st_mode) or bool(
        attributes & reparse_flag
    )


def reject_reparse_components(path: Path) -> None:
    """Reject symlinks and Windows reparse points in existing components."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts:
        raise ShortcutAuditError("Empty filesystem path")
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        current_value = dataset_common.filesystem_path(current)
        if not os.path.lexists(current_value):
            break
        result = os.lstat(current_value)
        if _is_reparse_or_link(result):
            raise ShortcutAuditError(
                f"Symlink or reparse point is forbidden: {current}"
            )


def snapshot_regular_file(path: Path) -> FileSnapshot:
    """Read one regular non-link file once; downstream uses these bytes."""

    reject_reparse_components(path)
    path_value = dataset_common.filesystem_path(path)
    with open(path_value, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ShortcutAuditError(
                f"Audit input is not a regular file: {path}"
            )
        payload = handle.read()
        after = os.fstat(handle.fileno())
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field, None) != getattr(after, field, None)
        for field in identity_fields
    ) or len(payload) != after.st_size:
        raise ShortcutAuditError(
            f"Audit input changed while being snapshotted: {path}"
        )
    return FileSnapshot(
        basename=path.name,
        size_bytes=len(payload),
        sha256=dataset_common.sha256_bytes(payload),
        payload=payload,
    )


def load_json_snapshot(path: Path) -> tuple[Any, FileSnapshot]:
    snapshot = snapshot_regular_file(path)
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ShortcutAuditError(
            f"JSON is not strict UTF-8: {path.name}"
        ) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=dataset_common._reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, dataset_common.ContractError) as error:
        raise ShortcutAuditError(
            f"Invalid canonical JSON input: {path.name}"
        ) from error
    return value, snapshot


def canonical_without_self(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: nested
        for key, nested in value.items()
        if key != "canonical_self_hash"
    }


def add_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    if "canonical_self_hash" in output:
        raise ShortcutAuditError(
            "Cannot add a second canonical_self_hash"
        )
    output["canonical_self_hash"] = dataset_common.canonical_sha256(
        output
    )
    return output


def validate_self_hash(
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed = value.get("canonical_self_hash")
    expected = dataset_common.canonical_sha256(
        canonical_without_self(value)
    )
    if observed != expected:
        raise ShortcutAuditError(f"{label} self-hash mismatch")


def manifest_identity(
    lock: Mapping[str, Any],
    *,
    lock_path: Path,
    stage: str,
    producer_relative_path: str,
    additional_parent_manifests: Sequence[
        Mapping[str, Any]
    ] = (),
) -> dict[str, Any]:
    producer_hash = lock["source_files_sha256"].get(
        producer_relative_path
    )
    if not isinstance(producer_hash, str):
        raise ShortcutAuditError(
            "Manifest producer is outside the frozen source closure"
        )
    lock_value, lock_snapshot = load_json_snapshot(lock_path)
    if lock_value != dict(lock):
        raise ShortcutAuditError(
            "In-memory lock differs from the exact lock bytes"
        )
    lock_file_hash = lock_snapshot.sha256
    lock_content_hash = str(lock["canonical_self_hash"])
    parent_manifests = [
        {
            "role": "metadata_shortcut_implementation_lock",
            "file_sha256": lock_file_hash,
            "content_sha256": lock_content_hash,
        }
    ]
    observed_roles = {"metadata_shortcut_implementation_lock"}
    for source in additional_parent_manifests:
        if (
            set(source)
            != {"role", "file_sha256", "content_sha256"}
            or not isinstance(source["role"], str)
            or not source["role"]
            or source["role"] in observed_roles
            or not isinstance(source["file_sha256"], str)
            or len(source["file_sha256"]) != 64
            or not isinstance(source["content_sha256"], str)
            or len(source["content_sha256"]) != 64
        ):
            raise ShortcutAuditError(
                "Additional parent-manifest record is invalid"
            )
        observed_roles.add(source["role"])
        parent_manifests.append(dict(source))
    parent_manifests.sort(
        key=lambda row: str(row["role"]).encode("utf-8")
    )
    return {
        "step": 28,
        "stage": stage,
        "run_id": lock["formal_run_id"],
        "policy_sha256": lock["parent_policy"]["sha256"],
        "policy_contract_sha256": lock["parent_contract"]["sha256"],
        "producer_sha256": producer_hash,
        "lock_file_sha256": lock_file_hash,
        "lock_content_sha256": lock_content_hash,
        "source_closure_sha256": lock["source_closure_sha256"],
        "parent_manifests": parent_manifests,
        "upstream_custody_parent_seal_required": True,
    }


def validate_manifest_identity(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    lock_path: Path,
    stage: str,
    producer_relative_path: str,
    additional_parent_manifests: Sequence[
        Mapping[str, Any]
    ] = (),
) -> None:
    expected = manifest_identity(
        lock,
        lock_path=lock_path,
        stage=stage,
        producer_relative_path=producer_relative_path,
        additional_parent_manifests=additional_parent_manifests,
    )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ShortcutAuditError(
            f"Manifest identity/DAG drift for stage {stage}"
        )


def exact_input_record_map(
    rows: Any,
    *,
    expected_roles: set[str],
) -> dict[str, dict[str, Any]]:
    """Validate an exact input allow-list and index it by role."""

    if not isinstance(rows, list):
        raise ShortcutAuditError("Input allow-list is not a list")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"role", "basename", "size_bytes", "sha256"}
            or not isinstance(row["role"], str)
            or not row["role"]
            or not isinstance(row["basename"], str)
            or not row["basename"]
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or row["role"] in output
        ):
            raise ShortcutAuditError(
                "Input allow-list record schema drift"
            )
        output[row["role"]] = dict(row)
    if set(output) != expected_roles:
        raise ShortcutAuditError("Input allow-list role set drift")
    return output


def validate_label_manifest_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    labels_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
    label_snapshot: FileSnapshot,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_snapshot: FileSnapshot,
) -> dict[str, dict[str, Any]]:
    """Fully validate a sealed label release used as a DAG parent."""

    if (
        split not in SPLITS
        or labels_path.name != LABEL_FILENAME
        or manifest_path.name != LABEL_MANIFEST_FILENAME
        or manifest_snapshot.basename != LABEL_MANIFEST_FILENAME
        or manifest_snapshot.size_bytes <= 0
        or os.path.abspath(
            dataset_common.filesystem_path(labels_path.parent)
        )
        != os.path.abspath(
            dataset_common.filesystem_path(manifest_path.parent)
        )
    ):
        raise ShortcutAuditError(
            "Label release path/split contract failed"
        )
    expected_manifest_keys = {
        "version",
        "status",
        "mode",
        "split",
        "row_count",
        "world_count",
        "rows_per_world",
        "label_schema",
        "formula",
        "formula_equality_required",
        "class_counts_withheld",
        "label_content_sha256",
        "step",
        "stage",
        "run_id",
        "policy_sha256",
        "policy_contract_sha256",
        "producer_sha256",
        "lock_file_sha256",
        "lock_content_sha256",
        "source_closure_sha256",
        "parent_manifests",
        "upstream_custody_parent_seal_required",
        "input_allowlist",
        "access_isolation_status",
        "forbidden_open_count_not_self_asserted",
        "files",
        "canonical_self_hash",
    }
    if set(manifest) != expected_manifest_keys:
        raise ShortcutAuditError("Label manifest schema drift")
    validate_self_hash(manifest, label="label manifest")
    validate_manifest_identity(
        manifest,
        lock,
        lock_path=lock_path,
        stage="seal_classification_labels",
        producer_relative_path=(
            "scripts/step28_v13_seal_classification_labels.py"
        ),
    )
    expected_world_count = int(
        lock["formal_world_counts"][split]
    )
    if (
        manifest.get("version") != LABEL_MANIFEST_VERSION
        or manifest.get("status")
        != "SEALED_PRIVATE_CLASSIFICATION_LABELS"
        or manifest.get("mode") != "formal"
        or manifest.get("split") != split
        or manifest.get("row_count") != expected_world_count * 40
        or manifest.get("world_count") != expected_world_count
        or manifest.get("rows_per_world") != 40
        or manifest.get("label_schema") != list(LABEL_FIELDS)
        or manifest.get("formula")
        != "int(controller(left)==controller(right))"
        or manifest.get("formula_equality_required") is not True
        or manifest.get("class_counts_withheld") is not True
        or manifest.get("access_isolation_status")
        != BLOCKED_ACCESS_STATUS
        or manifest.get("forbidden_open_count_not_self_asserted")
        is not True
        or manifest.get("label_content_sha256")
        != dataset_common.canonical_sha256(label_rows)
    ):
        raise ShortcutAuditError(
            "Label release manifest contract failed"
        )
    input_records = exact_input_record_map(
        manifest["input_allowlist"],
        expected_roles={
            "candidate_pairs",
            "controller_membership",
        },
    )
    if (
        [
            row["role"]
            for row in manifest["input_allowlist"]
        ]
        != ["candidate_pairs", "controller_membership"]
        or {
            role: row["basename"]
            for role, row in input_records.items()
        }
        != lock["label_sealer"]["input_basenames"]
    ):
        raise ShortcutAuditError(
            "Label manifest input allow-list drift"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ShortcutAuditError("Label manifest file list drift")
    validate_snapshot_file_record(
        files[0],
        snapshot=label_snapshot,
        path=LABEL_FILENAME,
        role="private_classification_labels",
    )
    if exact_file_set(labels_path.parent) != {
        LABEL_FILENAME,
        LABEL_MANIFEST_FILENAME,
    }:
        raise ShortcutAuditError(
            "Label release physical file set drift"
        )
    return input_records


def _source_closure_hashes() -> dict[str, str]:
    relative_paths = (
        "scripts/step28_v13_metadata_shortcut_common.py",
        "scripts/step28_v13_build_metadata_shortcut_lock.py",
        "scripts/step28_v13_project_null_nuisance.py",
        "scripts/step28_v13_seal_classification_labels.py",
        "scripts/step28_v13_validate_label_formula.py",
        "scripts/step28_v13_run_metadata_shortcut_audit.py",
        "tests/test_step28_v13_metadata_shortcut_audit_contracts.py",
    )
    return {
        relative: dataset_common.sha256_file(
            dataset_common.repo_path(relative)
        )
        for relative in relative_paths
    }


def load_lock(
    path: Path = DEFAULT_LOCK_PATH,
    *,
    verify_source_closure: bool = True,
) -> dict[str, Any]:
    lock, _lock_snapshot = load_json_snapshot(path)
    if not isinstance(lock, dict):
        raise ShortcutAuditError(
            "Metadata-shortcut lock is not an object"
        )
    expected_top_level_keys = {
        "version",
        "status",
        "objective",
        "parent_contract",
        "parent_policy",
        "implementation_document",
        "parent_metadata_contract_sha256",
        "engineering_smoke_release",
        "splits",
        "pretrain_audit_splits",
        "formal_run_id",
        "formal_world_counts",
        "seller_feature_order",
        "pair_feature_order",
        "projection_schema",
        "label_schema",
        "private_oof_schema",
        "float_serialization",
        "bootstrap_draw_hash_dtype",
        "projector",
        "label_sealer",
        "label_formula_validator",
        "audit_runner",
        "statistics",
        "formal_execution",
        "claim_boundary",
        "source_files_sha256",
        "source_closure_sha256",
        "dependency_closure_sha256",
        "canonical_self_hash",
    }
    if set(lock) != expected_top_level_keys:
        raise ShortcutAuditError(
            "Metadata-shortcut lock top-level schema drift"
        )
    if lock.get("version") != LOCK_VERSION:
        raise ShortcutAuditError(
            "Unexpected metadata-shortcut lock version"
        )
    validate_self_hash(lock, label="metadata-shortcut lock")
    if lock.get("status") != (
        "CORE_IMPLEMENTATION_LOCKED_FORMAL_EXECUTION_BLOCKED"
    ):
        raise ShortcutAuditError(
            "Metadata-shortcut lock status is not fail-closed"
        )
    contract_path = dataset_common.repo_path(
        str(lock["parent_contract"]["path"])
    )
    policy_path = dataset_common.repo_path(
        str(lock["parent_policy"]["path"])
    )
    document_path = dataset_common.repo_path(
        str(lock["implementation_document"]["path"])
    )
    contract_snapshot = snapshot_regular_file(contract_path)
    parent_policy, policy_snapshot = load_json_snapshot(policy_path)
    if not isinstance(parent_policy, dict):
        raise ShortcutAuditError("Parent policy is not an object")
    document_snapshot = snapshot_regular_file(document_path)
    current_files = {
        "parent_contract": contract_snapshot.sha256,
        "parent_policy": policy_snapshot.sha256,
        "implementation_document": document_snapshot.sha256,
    }
    for key, observed in current_files.items():
        if observed != lock[key]["sha256"]:
            raise ShortcutAuditError(f"{key} byte pin drift")
    smoke_record = lock["engineering_smoke_release"]
    if set(smoke_record) != {
        "path",
        "sha256",
        "content_sha256",
        "status",
        "numeric_shortcut_execution_forbidden",
        "reason",
    }:
        raise ShortcutAuditError(
            "Engineering-smoke record schema drift"
        )
    smoke_path = dataset_common.repo_path(
        str(smoke_record["path"])
    )
    smoke_value, smoke_snapshot = load_json_snapshot(smoke_path)
    if not isinstance(smoke_value, dict):
        raise ShortcutAuditError(
            "Engineering-smoke release is not an object"
        )
    if (
        smoke_snapshot.sha256 != smoke_record["sha256"]
        or smoke_value.get("canonical_self_hash")
        != smoke_record["content_sha256"]
        or smoke_value.get("status") != smoke_record["status"]
        or smoke_record["numeric_shortcut_execution_forbidden"]
        is not True
    ):
        raise ShortcutAuditError(
            "Engineering-smoke release pin drift"
        )
    parent_metadata = parent_policy["metadata_shortcut_audit"]
    metadata_projection_hash = dataset_common.canonical_sha256(
        parent_metadata
    )
    if (
        metadata_projection_hash
        != lock["parent_metadata_contract_sha256"]
    ):
        raise ShortcutAuditError(
            "Parent metadata shortcut contract drift"
        )
    if tuple(lock["splits"]) != SPLITS:
        raise ShortcutAuditError("Shortcut lock split set drift")
    if tuple(lock["seller_feature_order"]) != SELLER_FEATURES:
        raise ShortcutAuditError("Seller feature order drift")
    if tuple(lock["pair_feature_order"]) != PAIR_FEATURES:
        raise ShortcutAuditError("Pair feature order drift")
    if tuple(lock["projection_schema"]) != PROJECTION_FIELDS:
        raise ShortcutAuditError("Projection schema drift")
    if tuple(lock["label_schema"]) != LABEL_FIELDS:
        raise ShortcutAuditError("Label schema drift")
    if (
        tuple(lock["pretrain_audit_splits"])
        != PRETRAIN_AUDIT_SPLITS
        or tuple(lock["private_oof_schema"]) != OOF_FIELDS
        or lock["float_serialization"] != ".12f"
        or lock["bootstrap_draw_hash_dtype"] != ">u8"
        or lock["formal_run_id"] != parent_policy["modes"]["formal"][
            "run_id"
        ]
    ):
        raise ShortcutAuditError(
            "Shortcut auxiliary schema/serialization drift"
        )
    if lock["claim_boundary"] != {
        "pass_scope": "PASS_METADATA_SHORTCUT_ONLY",
        "pass_dataset_only_granted": False,
        "mechanism_coverage_validator_included": False,
        "absence_of_all_unknown_shortcuts_proven": False,
    }:
        raise ShortcutAuditError("Shortcut claim boundary drift")
    if lock["formal_execution"]["enabled"] is not False:
        raise ShortcutAuditError(
            "Formal execution may not be enabled in this lock"
        )
    if set(lock["formal_execution"]) != {
        "enabled",
        "formal_release_content_sha256",
        "custody_access_manifest_content_sha256",
        "execution_environment_content_sha256",
        "exact_input_bindings",
        "missing_prerequisites",
        "enablement_rule",
        "split_authorizations",
    } or any(
        lock["formal_execution"][key] is not None
        for key in (
            "formal_release_content_sha256",
            "custody_access_manifest_content_sha256",
            "execution_environment_content_sha256",
            "exact_input_bindings",
        )
    ):
        raise ShortcutAuditError(
            "Blocked formal execution placeholder drift"
        )
    split_authorizations = lock["formal_execution"].get(
        "split_authorizations"
    )
    if (
        not isinstance(split_authorizations, dict)
        or set(split_authorizations) != set(SPLITS)
    ):
        raise ShortcutAuditError(
            "Split authorization schema drift"
        )
    expected_operations = [
        "label_sealing",
        "label_formula_validation",
        "metadata_shortcut_audit",
    ]
    for split in SPLITS:
        authorization = split_authorizations[split]
        if (
            not isinstance(authorization, dict)
            or set(authorization)
            != {
                "authorized",
                "allowed_operations",
                "authorization_receipts",
                "required_prior_receipts",
                "required_prior_statuses",
            }
            or authorization["authorized"] is not False
            or authorization["allowed_operations"]
            != expected_operations
            or authorization["authorization_receipts"]
            != {operation: None for operation in expected_operations}
            or authorization["required_prior_receipts"] != []
            or authorization["required_prior_statuses"]
            != prior_requirements(split)
        ):
            raise ShortcutAuditError(
                f"Blocked split authorization drift: {split}"
            )
    expected_statistics = {
        "models": parent_metadata["models"],
        "world_grouped_folds": parent_metadata[
            "world_grouped_folds"
        ],
        "fold_random_seed": parent_metadata["random_seed"],
        "bootstrap": parent_metadata["bootstrap"],
        "point_maximum": parent_metadata[
            "auc_symmetric_point_maximum"
        ],
        "bootstrap_95_upper_maximum": parent_metadata[
            "auc_symmetric_world_bootstrap_95_upper_maximum"
        ],
        "numpy_version": parent_metadata["numpy_version"],
        "scikit_learn_version": parent_metadata[
            "sklearn_version"
        ],
    }
    if lock.get("statistics") != expected_statistics:
        raise ShortcutAuditError(
            "Shortcut statistics are not an exact parent transcription"
        )
    expected_world_counts = parent_policy["modes"]["formal"][
        "world_counts"
    ]
    if lock.get("formal_world_counts") != expected_world_counts:
        raise ShortcutAuditError("Formal world-count lock drift")
    expected_basenames = {
        "candidate_pairs": "candidate_pairs.csv",
        "redacted_items": "redacted_items.jsonl",
        "history_item_index": "history_item_index.csv",
    }
    if lock.get("projector", {}).get(
        "input_basenames"
    ) != expected_basenames:
        raise ShortcutAuditError(
            "Projector input basename allow-list drift"
        )
    if lock.get("label_sealer", {}).get(
        "input_basenames"
    ) != {
        "candidate_pairs": "candidate_pairs.csv",
        "controller_membership": "controller_membership.csv",
    }:
        raise ShortcutAuditError(
            "Label-sealer input basename allow-list drift"
        )
    if lock.get("audit_runner", {}).get(
        "input_basenames"
    ) != {
        "projection": PROJECTION_FILENAME,
        "labels": LABEL_FILENAME,
        "label_formula_receipt": LABEL_VALIDATION_FILENAME,
    }:
        raise ShortcutAuditError(
            "Audit-runner input basename allow-list drift"
        )
    if lock.get("label_formula_validator", {}).get(
        "input_basenames"
    ) != {
        "candidate_pairs": "candidate_pairs.csv",
        "controller_membership": "controller_membership.csv",
        "labels": LABEL_FILENAME,
        "label_manifest": LABEL_MANIFEST_FILENAME,
    }:
        raise ShortcutAuditError(
            "Formula-validator input basename allow-list drift"
        )
    for relative, expected_hash in lock[
        "dependency_closure_sha256"
    ].items():
        dependency_path = dataset_common.repo_path(relative)
        if dataset_common.sha256_file(dependency_path) != expected_hash:
            raise ShortcutAuditError(
                f"Metadata-shortcut dependency drift: {relative}"
            )
    if verify_source_closure:
        observed_sources = _source_closure_hashes()
        if observed_sources != lock["source_files_sha256"]:
            raise ShortcutAuditError(
                "Metadata-shortcut source closure drift"
            )
        if (
            dataset_common.canonical_sha256(observed_sources)
            != lock["source_closure_sha256"]
        ):
            raise ShortcutAuditError(
                "Metadata-shortcut source-closure hash drift"
            )
    return lock


def require_formal_execution_envelope(
    lock: Mapping[str, Any],
) -> None:
    del lock
    raise ShortcutAuditError(
        "Formal metadata-shortcut execution is blocked in this core "
        "implementation version; no in-memory lock mapping can enable "
        "release writers. Build a new versioned execution verifier, "
        "source closure, and lock after every prerequisite is sealed"
    )


def require_split_supervision_authorization(
    lock: Mapping[str, Any],
    *,
    split: str,
    operation: str,
) -> None:
    """Require a split-specific, externally sealed supervision grant."""

    if split not in SPLITS:
        raise ShortcutAuditError(
            "Unknown split for supervision authorization"
        )
    authorizations = lock["formal_execution"].get(
        "split_authorizations"
    )
    if not isinstance(authorizations, dict):
        raise ShortcutAuditError(
            "Split supervision authorizations are absent"
        )
    authorization = authorizations.get(split)
    if (
        not isinstance(authorization, dict)
        or authorization.get("authorized") is not True
        or operation
        not in authorization.get("allowed_operations", [])
    ):
        raise ShortcutAuditError(
            f"Split {split} is not authorized for {operation}"
        )
    receipts = authorization.get("authorization_receipts")
    receipt = (
        receipts.get(operation)
        if isinstance(receipts, dict)
        else None
    )
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "role",
            "path",
            "file_sha256",
            "content_sha256",
        }
        or receipt["role"] != f"{split}_{operation}_authorization"
    ):
        raise ShortcutAuditError(
            f"Split {split} lacks the exact {operation} receipt"
        )
    receipt_path = dataset_common.repo_path(str(receipt["path"]))
    receipt_value, receipt_snapshot = load_json_snapshot(receipt_path)
    if not isinstance(receipt_value, dict):
        raise ShortcutAuditError(
            "Split authorization receipt is not an object"
        )
    validate_self_hash(
        receipt_value,
        label="split operation authorization receipt",
    )
    expected_authorization_keys = {
        "version",
        "status",
        "authorized",
        "split",
        "operation",
        "run_id",
        "implementation_lock_content_sha256",
        "formal_release_content_sha256",
        "custody_access_manifest_content_sha256",
        "issuer",
        "one_shot_nonce_sha256",
        "canonical_self_hash",
    }
    if (
        set(receipt_value) != expected_authorization_keys
        or receipt_snapshot.sha256
        != receipt["file_sha256"]
        or receipt_value["canonical_self_hash"]
        != receipt["content_sha256"]
        or receipt_value.get("status")
        != "AUTHORIZED_SPLIT_OPERATION"
        or receipt_value.get("split") != split
        or receipt_value.get("operation") != operation
        or receipt_value.get("authorized") is not True
        or receipt_value.get("run_id") != lock["formal_run_id"]
        or receipt_value.get(
            "implementation_lock_content_sha256"
        )
        != lock["canonical_self_hash"]
        or receipt_value.get("formal_release_content_sha256")
        != lock["formal_execution"][
            "formal_release_content_sha256"
        ]
        or receipt_value.get(
            "custody_access_manifest_content_sha256"
        )
        != lock["formal_execution"][
            "custody_access_manifest_content_sha256"
        ]
        or not isinstance(receipt_value.get("issuer"), str)
        or not receipt_value["issuer"]
        or not isinstance(
            receipt_value.get("one_shot_nonce_sha256"),
            str,
        )
        or len(receipt_value["one_shot_nonce_sha256"]) != 64
    ):
        raise ShortcutAuditError(
            "Split supervision authorization receipt drift"
        )
    required_prior = authorization.get("required_prior_receipts")
    required_statuses = authorization.get("required_prior_statuses")
    if (
        not isinstance(required_prior, list)
        or not isinstance(required_statuses, list)
        or len(required_prior) != len(required_statuses)
    ):
        raise ShortcutAuditError(
            "Split prior-receipt contract is invalid"
        )
    expected_by_role: dict[str, dict[str, str]] = {}
    for expected in required_statuses:
        if (
            not isinstance(expected, dict)
            or set(expected)
            != {"role", "split", "operation", "status"}
            or expected["role"] in expected_by_role
            or expected["split"] not in SPLITS
            or not all(
                isinstance(expected[key], str) and expected[key]
                for key in expected
            )
        ):
            raise ShortcutAuditError(
                "Required prior-status schema drift"
            )
        expected_by_role[expected["role"]] = expected
    observed_roles: set[str] = set()
    for prior in required_prior:
        if (
            not isinstance(prior, dict)
            or set(prior)
            != {
                "role",
                "path",
                "file_sha256",
                "content_sha256",
            }
            or prior["role"] in observed_roles
            or prior["role"] not in expected_by_role
        ):
            raise ShortcutAuditError(
                "Split prior-receipt record is invalid"
            )
        observed_roles.add(prior["role"])
        expected = expected_by_role[prior["role"]]
        prior_path = dataset_common.repo_path(str(prior["path"]))
        prior_value, prior_snapshot = load_json_snapshot(prior_path)
        if not isinstance(prior_value, dict):
            raise ShortcutAuditError(
                "Prior authorization receipt is not an object"
            )
        validate_self_hash(
            prior_value,
            label="required prior receipt",
        )
        expected_prior_keys = {
            "version",
            "status",
            "role",
            "split",
            "operation",
            "run_id",
            "implementation_lock_content_sha256",
            "formal_release_content_sha256",
            "custody_access_manifest_content_sha256",
            "issuer",
            "one_shot_nonce_sha256",
            "canonical_self_hash",
        }
        if (
            set(prior_value) != expected_prior_keys
            or prior_snapshot.sha256
            != prior["file_sha256"]
            or prior_value["canonical_self_hash"]
            != prior["content_sha256"]
            or prior_value.get("role") != expected["role"]
            or prior_value.get("split") != expected["split"]
            or prior_value.get("operation")
            != expected["operation"]
            or prior_value.get("status") != expected["status"]
            or prior_value.get("run_id") != lock["formal_run_id"]
            or prior_value.get(
                "implementation_lock_content_sha256"
            )
            != lock["canonical_self_hash"]
            or prior_value.get("formal_release_content_sha256")
            != lock["formal_execution"][
                "formal_release_content_sha256"
            ]
            or prior_value.get(
                "custody_access_manifest_content_sha256"
            )
            != lock["formal_execution"][
                "custody_access_manifest_content_sha256"
            ]
            or not isinstance(prior_value.get("issuer"), str)
            or not prior_value["issuer"]
            or not isinstance(
                prior_value.get("one_shot_nonce_sha256"),
                str,
            )
            or len(prior_value["one_shot_nonce_sha256"]) != 64
        ):
            raise ShortcutAuditError(
                "Required prior authorization receipt drift"
            )
    if observed_roles != set(expected_by_role):
        raise ShortcutAuditError(
            "Required prior authorization set is incomplete"
        )


def read_csv_exact(
    path: Path,
    *,
    fieldnames: Sequence[str],
) -> list[dict[str, str]]:
    rows, _snapshot = read_csv_exact_snapshot(
        path,
        fieldnames=fieldnames,
    )
    return rows


def read_csv_exact_snapshot(
    path: Path,
    *,
    fieldnames: Sequence[str],
) -> tuple[list[dict[str, str]], FileSnapshot]:
    snapshot = snapshot_regular_file(path)
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ShortcutAuditError(
            f"CSV is not strict UTF-8: {path.name}"
        ) from error
    handle = io.StringIO(text, newline="")
    reader = csv.DictReader(handle)
    if tuple(reader.fieldnames or ()) != tuple(fieldnames):
        raise ShortcutAuditError(
            f"CSV schema mismatch for {path.name}"
        )
    rows = [dict(row) for row in reader]
    if any(tuple(row) != tuple(fieldnames) for row in rows):
        raise ShortcutAuditError(
            f"CSV row key order drift for {path.name}"
        )
    return rows, snapshot


def read_jsonl_exact(
    path: Path,
    *,
    keys: set[str],
) -> list[dict[str, Any]]:
    rows, _snapshot = read_jsonl_exact_snapshot(path, keys=keys)
    return rows


def read_jsonl_exact_snapshot(
    path: Path,
    *,
    keys: set[str],
) -> tuple[list[dict[str, Any]], FileSnapshot]:
    snapshot = snapshot_regular_file(path)
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ShortcutAuditError(
            f"JSONL is not strict UTF-8: {path.name}"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        text.splitlines(keepends=True),
        start=1,
    ):
        if not line.endswith("\n") or not line.strip():
            raise ShortcutAuditError(
                f"Malformed JSONL line {line_number}"
            )
        row = json.loads(
            line,
            object_pairs_hook=dataset_common._reject_duplicate_pairs,
        )
        if not isinstance(row, dict) or set(row) != keys:
            raise ShortcutAuditError(
                f"JSONL schema mismatch at line {line_number}"
            )
        rows.append(row)
    if not text:
        raise ShortcutAuditError(f"JSONL is empty: {path.name}")
    return rows, snapshot


def file_record(
    path: Path,
    *,
    role: str,
    root: Path,
) -> dict[str, Any]:
    return dataset_common.artifact_record(
        path,
        role=role,
        root=root,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        dataset_common.filesystem_path(path),
        os.O_RDONLY,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    reject_reparse_components(root)
    root_value = dataset_common.filesystem_path(root)
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(root_value):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        directories.append(current_path)
        for name in directory_names:
            result = os.lstat(os.path.join(current, name))
            if _is_reparse_or_link(result):
                raise ShortcutAuditError(
                    "Staging tree contains a reparse directory"
                )
        for name in file_names:
            result = os.lstat(os.path.join(current, name))
            if _is_reparse_or_link(result) or not stat.S_ISREG(
                result.st_mode
            ):
                raise ShortcutAuditError(
                    "Staging tree contains a non-regular file"
                )
            with open(
                os.path.join(current, name),
                "r+b",
            ) as handle:
                os.fsync(handle.fileno())
    for path in sorted(
        directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)


def publish_directory(
    target: Path,
    *,
    writer: Callable[[Path], None],
) -> Path:
    """Write one immutable artifact directory and publish no-replace."""

    parent = target.parent
    reject_reparse_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    reject_reparse_components(parent)
    if os.path.lexists(dataset_common.filesystem_path(target)):
        raise FileExistsError(
            f"Refusing to overwrite audit artifact directory: {target}"
        )
    stage = parent / f".staging-{target.name}-{uuid.uuid4().hex}"
    os.mkdir(dataset_common.filesystem_path(stage))
    renamed = False
    try:
        writer(stage)
        _fsync_tree(stage)
        dataset_common.atomic_rename_no_replace(stage, target)
        renamed = True
        try:
            _fsync_directory(parent)
        except OSError as error:
            raise DurabilityUnknownError(target) from error
    finally:
        if (
            not renamed
            and os.path.exists(dataset_common.filesystem_path(stage))
        ):
            shutil.rmtree(dataset_common.filesystem_path(stage))
    return target


def publish_stage_failure(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    stage: str,
    producer_relative_path: str,
    output_dir: Path,
    error: Exception,
) -> Path:
    """Publish a minimal immutable failure receipt without data values."""

    invocation_uid = uuid.uuid4().hex
    target = output_dir.parent / (
        f"{output_dir.name}.failure-{invocation_uid}"
    )
    detail_hash = dataset_common.sha256_bytes(
        (
            type(error).__name__ + "\x1f" + str(error)
        ).encode("utf-8")
    )

    def writer(stage_dir: Path) -> None:
        receipt = add_self_hash(
            {
                "version": (
                    "2026-07-29-step28-v13-stage-failure-v1"
                ),
                "status": "INVALID_STAGE_EXECUTION",
                "mode": "formal",
                "split": split,
                "failure_code": (
                    "VALIDITY_RUNTIME_OR_DURABILITY_FAILURE"
                ),
                "failure_type": type(error).__name__,
                "invocation_uid": invocation_uid,
                "private_failure_detail_sha256": detail_hash,
                "scores_released": False,
                "class_counts_withheld": True,
                **manifest_identity(
                    lock,
                    lock_path=lock_path,
                    stage=stage,
                    producer_relative_path=producer_relative_path,
                ),
            }
        )
        dataset_common.write_json(
            stage_dir / FAILURE_MANIFEST_FILENAME,
            receipt,
        )

    return publish_directory(target, writer=writer)


def exact_file_set(root: Path) -> set[str]:
    reject_reparse_components(root)
    output: set[str] = set()
    root_value = dataset_common.filesystem_path(root)
    for current, directory_names, file_names in os.walk(root_value):
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            result = os.lstat(os.path.join(current, name))
            if _is_reparse_or_link(result):
                raise ShortcutAuditError(
                    "Audit artifact directory contains a reparse point"
                )
        for name in file_names:
            result = os.lstat(os.path.join(current, name))
            if _is_reparse_or_link(result) or not stat.S_ISREG(
                result.st_mode
            ):
                raise ShortcutAuditError(
                    "Audit artifact contains a non-regular file"
                )
            output.add(
                os.path.relpath(
                    os.path.join(current, name),
                    root_value,
                ).replace(os.sep, "/")
            )
    return output


def validate_snapshot_file_record(
    row: Mapping[str, Any],
    *,
    snapshot: FileSnapshot,
    path: str,
    role: str,
) -> None:
    if set(row) != {"path", "role", "size_bytes", "sha256"}:
        raise ShortcutAuditError(
            "Manifest file-record schema drift"
        )
    if (
        row["path"] != path
        or row["role"] != role
        or row["size_bytes"] != snapshot.size_bytes
        or row["sha256"] != snapshot.sha256
    ):
        raise ShortcutAuditError(
            f"Manifest file record drift: {path}"
        )


def validate_registered_files(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    extra_files: set[str],
) -> None:
    registered: set[str] = set()
    for row in rows:
        relative = str(row["path"])
        if relative in registered:
            raise ShortcutAuditError(
                "Manifest contains a duplicate output path"
            )
        registered.add(relative)
        path = root / relative
        stat_result = os.stat(dataset_common.filesystem_path(path))
        if (
            stat_result.st_size != int(row["size_bytes"])
            or dataset_common.sha256_file(path) != row["sha256"]
        ):
            raise ShortcutAuditError(
                f"Registered audit artifact drift: {relative}"
            )
    if exact_file_set(root) != registered | extra_files:
        raise ShortcutAuditError(
            "Audit artifact physical file set drift"
        )
