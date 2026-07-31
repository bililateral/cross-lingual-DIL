#!/usr/bin/env python3
"""Prove that the Step28-v13 repair changes only C40 serialization order."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as common


VERSION = "2026-07-31-step28-v13-order-repair-equivalence-v2"
SPLITS = ("train", "development", "audit_a", "audit_b")
PARENT_RUN_ID = "v13_training_ready_v1_20260729"
PARENT_ROOT = (
    common.ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / PARENT_RUN_ID
)
PARENT_RELEASE_MANIFEST_SHA256 = (
    "6924dadb669bf056302418ac012e2f027b1bd3e9e00cf0c0e5e515258a3d3ce0"
)
PARENT_RELEASE_MANIFEST_SELF_SHA256 = (
    "f3e0b9a2de4b89613d008b7bc8ee11e4a87c0475fdd572ddea832ef543ead7aa"
)
ORDER_ONLY_DIFFERING_PATHS = {
    "observed/candidate_pairs.csv",
    "private_audit/candidate_sampling_audit.csv",
    "private_audit/world_generation_audit.jsonl",
}
FLOAT_TOLERANCE = 1e-12


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(
        common.filesystem_path(path),
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or any(tuple(row) != tuple(fields) for row in rows):
        raise common.ContractError(f"Malformed repair CSV: {path}")
    return fields, rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(
        common.filesystem_path(path),
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            value = json.loads(
                line,
                object_pairs_hook=common._reject_duplicate_pairs,
            )
            if not isinstance(value, dict):
                raise common.ContractError(
                    f"Non-object repair JSONL row: {path}"
                )
            rows.append(value)
    return rows


def _file_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise common.ContractError("Split manifest has no file records")
    by_path = {str(row["path"]): dict(row) for row in records}
    if len(by_path) != len(records):
        raise common.ContractError("Duplicate split-manifest member path")
    return by_path


def _split_semantic_paths(split: str) -> dict[str, str]:
    if split not in SPLITS:
        raise common.ContractError(
            f"Unknown repair-equivalence split: {split}"
        )
    if split in {"train", "development"}:
        supervision_root = "supervision"
        shortcut_oof = "private_audit/metadata_shortcut_oof.csv"
        shortcut_report = "audit/metadata_shortcut_audit.json"
    else:
        supervision_root = "sealed_supervision"
        shortcut_oof = (
            "sealed_supervision/metadata_shortcut_oof.private.csv"
        )
        shortcut_report = (
            "sealed_supervision/metadata_shortcut_audit.private.json"
        )
    return {
        "classification_labels": (
            f"{supervision_root}/classification_labels.csv"
        ),
        "null_nuisance_pairs": "audit/null_nuisance_pairs.csv",
        "metadata_shortcut_oof": shortcut_oof,
        "metadata_shortcut_report": shortcut_report,
    }


def _require_manifest_paths(
    *,
    split: str,
    parent_files: Mapping[str, Any],
    repaired_files: Mapping[str, Any],
    required: set[str],
) -> None:
    parent_missing = required - set(parent_files)
    repaired_missing = required - set(repaired_files)
    if parent_missing or repaired_missing:
        details: list[str] = []
        if parent_missing:
            details.append(
                "parent=" + ",".join(common.utf8_sort(parent_missing))
            )
        if repaired_missing:
            details.append(
                "repaired=" + ",".join(common.utf8_sort(repaired_missing))
            )
        raise common.ContractError(
            f"Repair-equivalence required path missing for {split}: "
            + "; ".join(details)
        )


def _verify_member(
    root: Path,
    relative: str,
    record: Mapping[str, Any],
) -> Path:
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or common.sha256_file(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("size_bytes", -1))
    ):
        raise common.ContractError(
            f"Repair-equivalence member hash drift: {path}"
        )
    return path


def _sorted_csv_equivalent(
    parent: Path,
    repaired: Path,
    *,
    ignore_fields: Sequence[str] = (),
    numeric_tolerance: float | None = None,
) -> dict[str, Any]:
    parent_fields, parent_rows = _read_csv(parent)
    repaired_fields, repaired_rows = _read_csv(repaired)
    if parent_fields != repaired_fields:
        raise common.ContractError("Repair CSV header drift")
    key = "canonical_pair_uid"
    if key not in parent_fields:
        raise common.ContractError("Repair CSV lacks canonical pair key")
    if len({row[key] for row in parent_rows}) != len(parent_rows):
        raise common.ContractError("Parent repair CSV has duplicate pair key")
    if len({row[key] for row in repaired_rows}) != len(repaired_rows):
        raise common.ContractError("Repaired CSV has duplicate pair key")
    parent_by_key = {row[key]: row for row in parent_rows}
    repaired_by_key = {row[key]: row for row in repaired_rows}
    if set(parent_by_key) != set(repaired_by_key):
        raise common.ContractError("Repair CSV pair-key set changed")
    maximum_numeric_delta = 0.0
    ignored = set(ignore_fields)
    for pair_uid in common.utf8_sort(parent_by_key):
        left = parent_by_key[pair_uid]
        right = repaired_by_key[pair_uid]
        for field in parent_fields:
            if field in ignored:
                continue
            if left[field] == right[field]:
                continue
            if numeric_tolerance is None:
                raise common.ContractError(
                    f"Repair CSV semantic drift: {parent.name}/{field}"
                )
            try:
                delta = abs(float(left[field]) - float(right[field]))
            except ValueError as exc:
                raise common.ContractError(
                    f"Repair CSV nonnumeric drift: {parent.name}/{field}"
                ) from exc
            maximum_numeric_delta = max(maximum_numeric_delta, delta)
            if delta > numeric_tolerance:
                raise common.ContractError(
                    f"Repair CSV numeric drift: {parent.name}/{field}"
                )
    return {
        "row_count": len(parent_rows),
        "pair_keyset_exact": True,
        "semantic_fields_exact": True,
        "ignored_fields": list(ignore_fields),
        "numeric_tolerance": numeric_tolerance,
        "maximum_numeric_delta": maximum_numeric_delta,
    }


def _world_audit_equivalent(
    parent: Path,
    repaired: Path,
) -> dict[str, Any]:
    def normalized(path: Path) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for source in _read_jsonl(path):
            row = copy.deepcopy(source)
            world_uid = str(row["world_uid"])
            row["candidate_summary"].pop("canonical_self_hash")
            if world_uid in output:
                raise common.ContractError("Duplicate world audit row")
            output[world_uid] = row
        return output

    if normalized(parent) != normalized(repaired):
        raise common.ContractError("World audit changed beyond C40 order hash")
    return {
        "world_keyset_exact": True,
        "all_fields_except_candidate_order_hash_exact": True,
    }


def _numbers_close(left: Any, right: Any) -> float:
    if type(left) is not type(right):
        raise common.ContractError("Repair JSON type drift")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise common.ContractError("Repair JSON key drift")
        return max(
            (_numbers_close(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list):
        if len(left) != len(right):
            raise common.ContractError("Repair JSON list-length drift")
        return max(
            (_numbers_close(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, float):
        delta = abs(left - right)
        if delta > FLOAT_TOLERANCE:
            raise common.ContractError("Repair JSON numeric drift")
        return delta
    if left != right:
        raise common.ContractError("Repair JSON semantic drift")
    return 0.0


def _metadata_report_equivalent(
    parent: Path,
    repaired: Path,
) -> dict[str, Any]:
    left = common.load_json(parent)
    right = common.load_json(repaired)
    maximum_delta = _numbers_close(left, right)
    return {
        "schema_and_nonnumeric_values_exact": True,
        "numeric_tolerance": FLOAT_TOLERANCE,
        "maximum_numeric_delta": maximum_delta,
    }


def compare_release(
    *,
    overlay: Mapping[str, Any],
    repaired_root: Path,
) -> dict[str, Any]:
    parent_manifest_path = PARENT_ROOT / "release_manifest.json"
    if (
        PARENT_ROOT.is_symlink()
        or not PARENT_ROOT.is_dir()
        or common.sha256_file(parent_manifest_path)
        != PARENT_RELEASE_MANIFEST_SHA256
    ):
        raise common.ContractError("Pinned parent release is unavailable")
    parent_release = common.load_json(parent_manifest_path)
    lineage = overlay.get("repair_lineage", {})
    if (
        parent_release.get("run_id") != PARENT_RUN_ID
        or parent_release.get("canonical_self_hash")
        != PARENT_RELEASE_MANIFEST_SELF_SHA256
        or lineage.get("parent_run_id") != PARENT_RUN_ID
        or lineage.get("parent_release_manifest_sha256")
        != PARENT_RELEASE_MANIFEST_SHA256
        or lineage.get("parent_release_manifest_self_sha256")
        != PARENT_RELEASE_MANIFEST_SELF_SHA256
    ):
        raise common.ContractError("Parent repair-lineage pin drift")
    per_split: dict[str, Any] = {}
    for split in SPLITS:
        parent_split = PARENT_ROOT / split
        repaired_split = repaired_root / split
        parent_manifest_path = parent_split / "split_manifest.json"
        repaired_manifest_path = repaired_split / "split_manifest.json"
        parent_receipt = parent_release["split_receipts"][split]
        if (
            common.sha256_file(parent_manifest_path)
            != parent_receipt["manifest_sha256"]
        ):
            raise common.ContractError("Parent split manifest hash drift")
        parent_manifest = common.load_json(parent_manifest_path)
        repaired_manifest = common.load_json(repaired_manifest_path)
        parent_files = _file_records(parent_manifest)
        repaired_files = _file_records(repaired_manifest)
        if set(parent_files) != set(repaired_files):
            raise common.ContractError("Repair split member set changed")
        semantic_paths = _split_semantic_paths(split)
        required_paths = (
            set(ORDER_ONLY_DIFFERING_PATHS)
            | set(semantic_paths.values())
        )
        _require_manifest_paths(
            split=split,
            parent_files=parent_files,
            repaired_files=repaired_files,
            required=required_paths,
        )
        differing = {
            relative
            for relative in parent_files
            if parent_files[relative]["sha256"]
            != repaired_files[relative]["sha256"]
        }
        unexpected = differing - ORDER_ONLY_DIFFERING_PATHS
        if unexpected:
            raise common.ContractError(
                "Files changed beyond the frozen order-only allow-list: "
                + ", ".join(common.utf8_sort(unexpected))
            )
        for relative in differing:
            _verify_member(
                parent_split,
                relative,
                parent_files[relative],
            )
            _verify_member(
                repaired_split,
                relative,
                repaired_files[relative],
            )
        for relative in required_paths:
            _verify_member(
                parent_split,
                relative,
                parent_files[relative],
            )
            _verify_member(
                repaired_split,
                relative,
                repaired_files[relative],
            )
        semantic = {
            "candidate_pairs": _sorted_csv_equivalent(
                parent_split / "observed/candidate_pairs.csv",
                repaired_split / "observed/candidate_pairs.csv",
            ),
            "candidate_sampling_audit": _sorted_csv_equivalent(
                parent_split / "private_audit/candidate_sampling_audit.csv",
                repaired_split
                / "private_audit/candidate_sampling_audit.csv",
                ignore_fields=("selected_rank",),
            ),
            "classification_labels": _sorted_csv_equivalent(
                parent_split
                / semantic_paths["classification_labels"],
                repaired_split
                / semantic_paths["classification_labels"],
            ),
            "null_nuisance_pairs": _sorted_csv_equivalent(
                parent_split
                / semantic_paths["null_nuisance_pairs"],
                repaired_split
                / semantic_paths["null_nuisance_pairs"],
            ),
            "metadata_shortcut_oof": _sorted_csv_equivalent(
                parent_split
                / semantic_paths["metadata_shortcut_oof"],
                repaired_split
                / semantic_paths["metadata_shortcut_oof"],
                numeric_tolerance=FLOAT_TOLERANCE,
            ),
            "metadata_shortcut_report": _metadata_report_equivalent(
                parent_split
                / semantic_paths["metadata_shortcut_report"],
                repaired_split
                / semantic_paths["metadata_shortcut_report"],
            ),
            "world_generation_audit": _world_audit_equivalent(
                parent_split
                / "private_audit/world_generation_audit.jsonl",
                repaired_split
                / "private_audit/world_generation_audit.jsonl",
            ),
        }
        if (
            parent_manifest["structure_key_sha256_commitment"]
            != repaired_manifest["structure_key_sha256_commitment"]
        ):
            raise common.ContractError("Repair structure-key commitment drift")
        per_split[split] = {
            "parent_split_manifest_sha256": common.sha256_file(
                parent_manifest_path
            ),
            "repaired_split_manifest_sha256": common.sha256_file(
                repaired_manifest_path
            ),
            "member_count": len(parent_files),
            "byte_identical_member_count": len(parent_files) - len(differing),
            "differing_members": common.utf8_sort(differing),
            "allowed_differing_members": common.utf8_sort(
                ORDER_ONLY_DIFFERING_PATHS
            ),
            "unexpected_differing_members": [],
            "structure_key_commitment_exact": True,
            "semantic_equivalence": semantic,
        }
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "PASS_C40_OUTPUT_ORDER_ONLY_REPAIR_EQUIVALENCE",
        "parent_run_id": PARENT_RUN_ID,
        "repaired_run_id": overlay["run_id"],
        "parent_release_manifest_sha256": (
            PARENT_RELEASE_MANIFEST_SHA256
        ),
        "parent_release_manifest_self_sha256": (
            PARENT_RELEASE_MANIFEST_SELF_SHA256
        ),
        "repair_lineage": copy.deepcopy(overlay["repair_lineage"]),
        "per_split": per_split,
        "all_non_order_bound_members_byte_identical": True,
        "all_order_bound_members_semantically_equivalent": True,
        "all_structure_key_commitments_exact": True,
        "world_text_candidate_membership_identity33_and_labels_unchanged": True,
        "post_outcome_resampling_performed": False,
    }
    report["canonical_self_hash"] = common.canonical_sha256(report)
    return report
