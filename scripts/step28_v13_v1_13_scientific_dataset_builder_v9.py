#!/usr/bin/env python3
"""Build the four-split Step28-v13 v1.13 V9.1 repaired design dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import step28_v13_common as common
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_quality_channel_policy_v9 as quality_policy_module
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_world_v9 as world_module


ROOT = Path(__file__).resolve().parents[1]
SPLITS = scientific.SPLITS
DESIGN_EXECUTION_MODE = "design_preflight"
AUTHORIZATION_STATUS = "ALLOW_ONE_DESIGN_PREFLIGHT_BUILD"
AUTHORIZATION_CLAIM_BOUNDARY = (
    "This receipt authorizes exactly one design_preflight build and no quality "
    "audit, formal generation, truth opening, training, or metric generation."
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")
REVIEWED_AT_RE = re.compile(r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

# These are the only seller-profile fields that the frozen M0/M3 base-feature
# adapter may mount.  ``seller_uid`` is a join key, never a feature.  The five
# text fields feed the frozen LaBSE path; the remaining values are exactly the
# source statistics required to reconstruct legacy18 without exposing market,
# split, candidate, raw-source, or audit metadata.
MODEL_PROFILE_JOIN_ONLY_FIELDS = scientific.MODEL_PROFILE_JOIN_ONLY_FIELDS
MODEL_PROFILE_TEXT_FIELDS = scientific.MODEL_PROFILE_TEXT_FIELDS
MODEL_PROFILE_NUMERIC_FIELDS = scientific.MODEL_PROFILE_NUMERIC_FIELDS
MODEL_PROFILE_FIELDS = scientific.MODEL_PROFILE_FIELDS
MODEL_PROFILE_STYLE_FIELDS = scientific.MODEL_PROFILE_STYLE_FIELDS
MODEL_REDACTED_ITEM_JOIN_ONLY_FIELDS = (
    "item_uid",
    "seller_uid",
    "world_uid",
)
MODEL_REDACTED_ITEM_TEXT_FIELDS = (
    "title",
    "description",
)
MODEL_REDACTED_ITEM_FIELDS = (
    *MODEL_REDACTED_ITEM_JOIN_ONLY_FIELDS,
    *MODEL_REDACTED_ITEM_TEXT_FIELDS,
)
GLOBAL_UID_KINDS = (
    "world",
    "seller",
    "item",
    "pair",
    "query",
    "controller",
)
EXPECTED_SPLIT_DATA_PATHS = (
    "observed/worlds.jsonl",
    "observed/redacted_items.jsonl",
    "observed/redacted_items.code_masked.jsonl",
    "observed/redacted_items.code_neutralized.jsonl",
    "observed/model_seller_profiles.jsonl",
    "observed/model_seller_profiles.code_masked.jsonl",
    "observed/model_seller_profiles.code_neutralized.jsonl",
    "observed/complete_model_pair_endpoints.csv",
    "observed/identity33_all_pairs.csv",
    "private/controller_membership.jsonl",
    "private/pair_labels.csv",
    "private/qrels.jsonl",
    "private/world_generation_audit.jsonl",
    "private/document_collision_attempts.jsonl",
    "private/identity_allocation_receipts.jsonl",
    "private/public_code_probe_input.jsonl",
    "private/text_probe_eligibility_input.jsonl",
    "private/channel_structure_audit.jsonl",
)


class DatasetBuildError(scientific.ScientificBuilderError):
    """Raised when a multi-world dataset build fails closed."""


@dataclass(frozen=True)
class _VerifiedDesignBuildReceipt:
    path: Path
    size_bytes: int
    sha256: str
    receipt_id: str
    review_response_sha256: str
    git_commit: str
    git_tree: str
    random_authority_commitment_sha256: str
    builder_policy_binding: dict[str, Any]
    quality_policy_binding: dict[str, Any]
    builder_source_file: dict[str, Any]


@dataclass(frozen=True)
class AuthorizedDesignPreflightContext:
    execution_context: scientific.ExecutionContext
    builder_policy: dict[str, Any]
    quality_policy_binding: dict[str, Any]
    builder_source_file: dict[str, Any]
    git_commit: str
    git_tree: str
    random_authority_commitment_sha256: str
    receipt_id: str
    review_response_sha256: str
    receipt_file: dict[str, Any]


def _validate_model_mount_contract(policy: Mapping[str, Any]) -> None:
    mount = policy["model_mount_contract"]
    if (
        mount["seller_profile_surface_paths"]
        != {
            "surface_full": "observed/model_seller_profiles.jsonl",
            "surface_code_masked": "observed/model_seller_profiles.code_masked.jsonl",
            "surface_code_neutralized": "observed/model_seller_profiles.code_neutralized.jsonl",
        }
        or mount["redacted_item_surface_paths"]
        != {
            "surface_full": "observed/redacted_items.jsonl",
            "surface_code_masked": "observed/redacted_items.code_masked.jsonl",
            "surface_code_neutralized": "observed/redacted_items.code_neutralized.jsonl",
        }
        or mount["public_code_probe_input_path"]
        != "private/public_code_probe_input.jsonl"
        or mount["text_probe_eligibility_input_path"]
        != "private/text_probe_eligibility_input.jsonl"
        or mount["channel_structure_audit_path"]
        != "private/channel_structure_audit.jsonl"
        or tuple(mount["seller_profile_join_only_fields"])
        != MODEL_PROFILE_JOIN_ONLY_FIELDS
        or tuple(mount["seller_profile_text_feature_source_fields"])
        != MODEL_PROFILE_TEXT_FIELDS
        or tuple(mount["seller_profile_numeric_feature_source_fields"])
        != MODEL_PROFILE_NUMERIC_FIELDS
        or tuple(mount["seller_profile_length_stat_fields"]) != ("median",)
        or tuple(mount["seller_profile_style_stat_fields"])
        != MODEL_PROFILE_STYLE_FIELDS
        or tuple(mount["redacted_item_join_only_fields"])
        != MODEL_REDACTED_ITEM_JOIN_ONLY_FIELDS
        or tuple(mount["redacted_item_text_feature_source_fields"])
        != MODEL_REDACTED_ITEM_TEXT_FIELDS
        or mount["automatic_feature_discovery_forbidden"] is not True
        or mount["full_seller_profile_mount_forbidden"] is not True
    ):
        raise DatasetBuildError("Builder/model-mount source schema drift")


@dataclass
class _JsonlWriter:
    path: Path
    handle: TextIO
    row_count: int = 0

    @classmethod
    def open(cls, path: Path) -> "_JsonlWriter":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=path, handle=path.open("x", encoding="utf-8", newline=""))

    def write(self, row: Mapping[str, Any]) -> None:
        payload = json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.handle.write(payload + "\n")
        self.row_count += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


@dataclass
class _CsvWriter:
    path: Path
    fieldnames: tuple[str, ...]
    handle: TextIO
    writer: csv.DictWriter
    row_count: int = 0

    @classmethod
    def open(cls, path: Path, fieldnames: Sequence[str]) -> "_CsvWriter":
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("x", encoding="utf-8", newline="")
        names = tuple(str(value) for value in fieldnames)
        writer = csv.DictWriter(
            handle,
            fieldnames=list(names),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        return cls(path=path, fieldnames=names, handle=handle, writer=writer)

    def write(self, row: Mapping[str, Any]) -> None:
        if set(row) != set(self.fieldnames):
            raise DatasetBuildError(f"CSV row schema drift for {self.path.name}")
        self.writer.writerow({name: row[name] for name in self.fieldnames})
        self.row_count += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


@dataclass
class _SplitWriters:
    worlds: _JsonlWriter
    redacted_items: _JsonlWriter
    masked_redacted_items: _JsonlWriter
    neutral_redacted_items: _JsonlWriter
    model_seller_profiles: _JsonlWriter
    masked_model_seller_profiles: _JsonlWriter
    neutral_model_seller_profiles: _JsonlWriter
    endpoints: _CsvWriter
    identity33: _CsvWriter
    controller_membership: _JsonlWriter
    pair_labels: _CsvWriter
    qrels: _JsonlWriter
    private_world_audit: _JsonlWriter
    collision_attempts: _JsonlWriter
    identity_allocation: _JsonlWriter
    public_code_probe_input: _JsonlWriter
    text_probe_eligibility_input: _JsonlWriter
    channel_structure_audit: _JsonlWriter

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        endpoint_fields: Sequence[str],
        identity_fields: Sequence[str],
    ) -> "_SplitWriters":
        return cls(
            worlds=_JsonlWriter.open(root / "observed" / "worlds.jsonl"),
            redacted_items=_JsonlWriter.open(
                root / "observed" / "redacted_items.jsonl"
            ),
            masked_redacted_items=_JsonlWriter.open(
                root / "observed" / "redacted_items.code_masked.jsonl"
            ),
            neutral_redacted_items=_JsonlWriter.open(
                root / "observed" / "redacted_items.code_neutralized.jsonl"
            ),
            model_seller_profiles=_JsonlWriter.open(
                root / "observed" / "model_seller_profiles.jsonl"
            ),
            masked_model_seller_profiles=_JsonlWriter.open(
                root / "observed" / "model_seller_profiles.code_masked.jsonl"
            ),
            neutral_model_seller_profiles=_JsonlWriter.open(
                root / "observed" / "model_seller_profiles.code_neutralized.jsonl"
            ),
            endpoints=_CsvWriter.open(
                root / "observed" / "complete_model_pair_endpoints.csv",
                endpoint_fields,
            ),
            identity33=_CsvWriter.open(
                root / "observed" / "identity33_all_pairs.csv", identity_fields
            ),
            controller_membership=_JsonlWriter.open(
                root / "private" / "controller_membership.jsonl"
            ),
            pair_labels=_CsvWriter.open(
                root / "private" / "pair_labels.csv",
                ("canonical_pair_uid", "world_uid", "label"),
            ),
            qrels=_JsonlWriter.open(root / "private" / "qrels.jsonl"),
            private_world_audit=_JsonlWriter.open(
                root / "private" / "world_generation_audit.jsonl"
            ),
            collision_attempts=_JsonlWriter.open(
                root / "private" / "document_collision_attempts.jsonl"
            ),
            identity_allocation=_JsonlWriter.open(
                root / "private" / "identity_allocation_receipts.jsonl"
            ),
            public_code_probe_input=_JsonlWriter.open(
                root / "private" / "public_code_probe_input.jsonl"
            ),
            text_probe_eligibility_input=_JsonlWriter.open(
                root / "private" / "text_probe_eligibility_input.jsonl"
            ),
            channel_structure_audit=_JsonlWriter.open(
                root / "private" / "channel_structure_audit.jsonl"
            ),
        )

    def all_writers(self) -> tuple[_JsonlWriter | _CsvWriter, ...]:
        return (
            self.worlds,
            self.redacted_items,
            self.masked_redacted_items,
            self.neutral_redacted_items,
            self.model_seller_profiles,
            self.masked_model_seller_profiles,
            self.neutral_model_seller_profiles,
            self.endpoints,
            self.identity33,
            self.controller_membership,
            self.pair_labels,
            self.qrels,
            self.private_world_audit,
            self.collision_attempts,
            self.identity_allocation,
            self.public_code_probe_input,
            self.text_probe_eligibility_input,
            self.channel_structure_audit,
        )

    def close(self) -> None:
        errors: list[BaseException] = []
        for writer in self.all_writers():
            try:
                writer.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise DatasetBuildError("Failed to close one or more split outputs") from errors[0]


def _file_record(path: Path, *, root: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "row_count": row_count,
    }


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _count_file_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        count = sum(1 for _ in handle)
    if path.suffix == ".csv":
        if count <= 0:
            raise DatasetBuildError(f"CSV has no header: {path.name}")
        return count - 1
    return count


V9_1_OUTER_PROFILE_HASH_FIELDS = (
    "full_profile_sha256",
    "masked_profile_sha256",
    "neutral_profile_sha256",
)
V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS = (
    "/full_profile_sha256",
    "/masked_profile_sha256",
    "/neutral_profile_sha256",
    "/neutral_receipt/neutral_profile_sha256",
)
V9_1_STRUCTURE_AUDIT_PATH = "private/channel_structure_audit.jsonl"


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(
                    f"Invalid JSONL in V9.1 equivalence replay: {path.name}"
                ) from exc
            if not isinstance(row, dict):
                raise DatasetBuildError(
                    f"Non-object JSONL in V9.1 equivalence replay: {path.name}"
                )
            rows.append(row)
    return rows


def _json_pointer_parts(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/") or pointer == "/":
        raise DatasetBuildError(f"Invalid V9.1 JSON pointer: {pointer!r}")
    parts = tuple(pointer[1:].split("/"))
    if any(not part or "~" in part for part in parts):
        raise DatasetBuildError(f"Unsupported V9.1 JSON pointer: {pointer!r}")
    return parts


def _json_pointer_value(value: Mapping[str, Any], pointer: str) -> Any:
    node: Any = value
    for part in _json_pointer_parts(pointer):
        if not isinstance(node, Mapping) or part not in node:
            raise DatasetBuildError(f"Missing V9.1 JSON pointer: {pointer}")
        node = node[part]
    return node


def _structure_invariant_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(row))
    for pointer in V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS:
        node: Any = projected
        parts = _json_pointer_parts(pointer)
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise DatasetBuildError(f"Missing V9.1 JSON pointer: {pointer}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise DatasetBuildError(f"Missing V9.1 JSON pointer: {pointer}")
        del node[parts[-1]]
    return projected


def _structure_allowed_path_projection(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "world_uid": str(row["world_uid"]),
        "values": [
            {
                "json_pointer": pointer,
                "value": _json_pointer_value(row, pointer),
            }
            for pointer in V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS
        ],
    }


def _ordered_mapping_key_schema_projection(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    mappings: list[dict[str, Any]] = []

    def visit(node: Any, pointer: str) -> None:
        if isinstance(node, Mapping):
            keys = [str(key) for key in node]
            mappings.append({"json_pointer": pointer, "keys": keys})
            for key, child in node.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                visit(child, f"{pointer}/{escaped}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{pointer}/{index}")

    visit(row, "")
    return {"world_uid": str(row["world_uid"]), "mappings": mappings}


def _validate_v9_invalidated_equivalence_commitment_payload(
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    expected_payload_keys = {
        "version",
        "status",
        "claim_boundary",
        "historical_structure_source_commit",
        "invalidated_root",
        "allowed_changed_json_path_count",
        "semantic_profile_hash_count",
        "unchanged_file_count",
        "changed_structure_file_count",
        "splits",
        "canonical_self_hash",
    }
    unsigned = dict(payload)
    claimed = unsigned.pop("canonical_self_hash", None)
    if (
        set(payload) != expected_payload_keys
        or payload.get("version")
        != "2026-08-21-step28-v13-v1-13-v9-invalidated-equivalence-commitment-v2"
        or payload.get("status")
        != "V9_DATASET_INVALIDATED_REFERENCE_HASHES_ONLY"
        or payload.get("claim_boundary")
        != "V9_1_MAY_CHANGE_ONLY_FOUR_EXACT_PROFILE_HASH_JSON_PATHS_PER_STRUCTURE_ROW"
        or payload.get("allowed_changed_json_path_count") != 4
        or payload.get("semantic_profile_hash_count") != 3
        or payload.get("unchanged_file_count") != 68
        or payload.get("changed_structure_file_count") != 4
        or claimed != common.canonical_sha256(unsigned)
        or set(payload.get("splits", {})) != set(SPLITS)
        or payload.get("invalidated_root", {}).get(
            "random_authority_commitment_sha256"
        )
        != common.canonical_sha256(
            policy["public_preflight_keys"][DESIGN_EXECUTION_MODE]
        )
        or payload.get("historical_structure_source_commit")
        != "8bb3276de6d84c0aad8d7475af5ca0b41a86b959"
    ):
        raise DatasetBuildError("V9 invalidated equivalence commitment drift")
    invalidated_root = payload.get("invalidated_root")
    if (
        not isinstance(invalidated_root, Mapping)
        or set(invalidated_root)
        != {
            "path",
            "size_bytes",
            "sha256",
            "canonical_self_hash",
            "quality_attempt2_result_sha256",
            "random_authority_commitment_sha256",
        }
        or invalidated_root.get("path")
        != "reports/step28_v13_v1_13_scientific_builder/design_preflight_v9_20260814"
        or isinstance(invalidated_root.get("size_bytes"), bool)
        or not isinstance(invalidated_root.get("size_bytes"), int)
        or int(invalidated_root["size_bytes"]) <= 0
        or any(
            not HEX_SHA256_RE.fullmatch(str(invalidated_root.get(field, "")))
            for field in (
                "sha256",
                "canonical_self_hash",
                "quality_attempt2_result_sha256",
                "random_authority_commitment_sha256",
            )
        )
    ):
        raise DatasetBuildError("V9 invalidated equivalence commitment drift")
    expected_unchanged = set(EXPECTED_SPLIT_DATA_PATHS) - {
        V9_1_STRUCTURE_AUDIT_PATH
    }
    expected_world_counts = policy["execution_modes"][DESIGN_EXECUTION_MODE][
        "world_counts"
    ]
    for split in SPLITS:
        split_reference = payload["splits"].get(split)
        if (
            not isinstance(split_reference, Mapping)
            or set(split_reference)
            != {"split_manifest", "structure_audit", "unchanged_files"}
        ):
            raise DatasetBuildError("V9 invalidated equivalence commitment drift")
        split_manifest = split_reference.get("split_manifest")
        unchanged = split_reference.get("unchanged_files")
        structure = split_reference.get("structure_audit")
        if (
            not isinstance(split_manifest, Mapping)
            or set(split_manifest)
            != {"size_bytes", "sha256", "canonical_self_hash"}
            or isinstance(split_manifest.get("size_bytes"), bool)
            or not isinstance(split_manifest.get("size_bytes"), int)
            or int(split_manifest["size_bytes"]) <= 0
            or any(
                not HEX_SHA256_RE.fullmatch(str(split_manifest.get(field, "")))
                for field in ("sha256", "canonical_self_hash")
            )
            or not isinstance(unchanged, Mapping)
            or set(unchanged) != expected_unchanged
            or not isinstance(structure, Mapping)
            or set(structure)
            != {
                "path",
                "old_record",
                "allowed_changed_json_paths",
                "persisted_structure_version",
                "neutral_receipt_persisted_version",
                "invariant_canonical_jsonl_sha256",
                "world_uid_utf8_lines_sha256",
                "ordered_mapping_key_schema_canonical_jsonl_sha256",
                "old_allowed_path_commitment_canonical_jsonl_sha256",
                "old_outer_profile_commitment_canonical_jsonl_sha256",
                "old_inner_neutral_profile_commitment_canonical_jsonl_sha256",
            }
            or structure.get("path") != V9_1_STRUCTURE_AUDIT_PATH
            or structure.get("allowed_changed_json_paths")
            != list(V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS)
            or structure.get("persisted_structure_version")
            != scientific.PERSISTED_STRUCTURE_VERSION
            or structure.get("neutral_receipt_persisted_version")
            != scientific.PERSISTED_STRUCTURE_VERSION
        ):
            raise DatasetBuildError("V9 invalidated equivalence commitment drift")
        for relative, record in unchanged.items():
            if (
                not isinstance(record, Mapping)
                or set(record) != {"path", "size_bytes", "sha256", "row_count"}
                or record.get("path") != relative
                or any(
                    isinstance(record.get(field), bool)
                    or not isinstance(record.get(field), int)
                    or int(record[field]) < 0
                    for field in ("size_bytes", "row_count")
                )
                or not HEX_SHA256_RE.fullmatch(str(record.get("sha256", "")))
            ):
                raise DatasetBuildError(
                    "V9 invalidated equivalence commitment drift"
                )
        old_record = structure.get("old_record")
        hash_fields = (
            "invariant_canonical_jsonl_sha256",
            "world_uid_utf8_lines_sha256",
            "ordered_mapping_key_schema_canonical_jsonl_sha256",
            "old_allowed_path_commitment_canonical_jsonl_sha256",
            "old_outer_profile_commitment_canonical_jsonl_sha256",
            "old_inner_neutral_profile_commitment_canonical_jsonl_sha256",
        )
        if (
            not isinstance(old_record, Mapping)
            or set(old_record) != {"path", "size_bytes", "sha256", "row_count"}
            or old_record.get("path") != V9_1_STRUCTURE_AUDIT_PATH
            or old_record.get("row_count") != expected_world_counts[split]
            or any(
                isinstance(old_record.get(field), bool)
                or not isinstance(old_record.get(field), int)
                or int(old_record[field]) < 0
                for field in ("size_bytes", "row_count")
            )
            or not HEX_SHA256_RE.fullmatch(str(old_record.get("sha256", "")))
            or any(
                not HEX_SHA256_RE.fullmatch(str(structure.get(field, "")))
                for field in hash_fields
            )
        ):
            raise DatasetBuildError("V9 invalidated equivalence commitment drift")
    return dict(payload)


def _load_v9_invalidated_equivalence_commitment(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    spec = policy["v9_invalidated_equivalence_commitment"]
    path = common.verify_file_pin(
        spec, label="V9 invalidated equivalence commitment"
    )
    return _validate_v9_invalidated_equivalence_commitment_payload(
        common.load_json(path), policy
    )


def _canonical_jsonl_projection_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(common.canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_v9_1_persisted_equivalence(
    *,
    temp_root: Path,
    split_manifests: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the repair changes only four exact persisted JSON paths."""

    reference = _load_v9_invalidated_equivalence_commitment(policy)
    split_receipts: dict[str, dict[str, Any]] = {}
    unchanged_count = 0
    changed_count = 0
    surface_contract = (
        (
            "observed/model_seller_profiles.jsonl",
            "full_profile_sha256",
        ),
        (
            "observed/model_seller_profiles.code_masked.jsonl",
            "masked_profile_sha256",
        ),
        (
            "observed/model_seller_profiles.code_neutralized.jsonl",
            "neutral_profile_sha256",
        ),
    )
    for split in SPLITS:
        split_root = temp_root / split
        current_files = {
            str(row["path"]): dict(row)
            for row in split_manifests[split]["files"]
        }
        expected = reference["splits"][split]
        old_unchanged = expected["unchanged_files"]
        if set(current_files) != set(EXPECTED_SPLIT_DATA_PATHS):
            raise DatasetBuildError(
                f"V9.1 persisted equivalence file universe drift: {split}"
            )
        for relative, old_record in old_unchanged.items():
            if current_files.get(relative) != old_record:
                raise DatasetBuildError(
                    f"V9.1 changed a forbidden persisted file: {split}/{relative}"
                )
            unchanged_count += 1

        structure_reference = expected["structure_audit"]
        structure_record = current_files[V9_1_STRUCTURE_AUDIT_PATH]
        if (
            structure_reference.get("path") != V9_1_STRUCTURE_AUDIT_PATH
            or structure_reference.get("allowed_changed_json_paths")
            != list(V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS)
            or structure_record.get("row_count")
            != structure_reference["old_record"].get("row_count")
            or structure_record.get("sha256")
            == structure_reference["old_record"].get("sha256")
        ):
            raise DatasetBuildError(
                f"V9.1 structure-audit change boundary drift: {split}"
            )
        structure_rows = _read_jsonl_objects(
            split_root / V9_1_STRUCTURE_AUDIT_PATH
        )
        if len(structure_rows) != structure_record.get("row_count"):
            raise DatasetBuildError(
                f"V9.1 structure-audit row count drift: {split}"
            )
        invariant_rows: list[dict[str, Any]] = []
        profile_commitments: list[dict[str, Any]] = []
        mapping_key_schemas: list[dict[str, Any]] = []
        for row in structure_rows:
            if (
                row.get("version") != scientific.PERSISTED_STRUCTURE_VERSION
                or not isinstance(row.get("neutral_receipt"), Mapping)
                or row["neutral_receipt"].get("version")
                != scientific.PERSISTED_STRUCTURE_VERSION
                or row["neutral_receipt"].get("neutral_profile_sha256")
                != row.get("neutral_profile_sha256")
            ):
                raise DatasetBuildError(
                    f"V9.1 persisted version/neutral binding drift: {split}"
                )
            invariant_rows.append(_structure_invariant_projection(row))
            profile_commitments.append(_structure_allowed_path_projection(row))
            mapping_key_schemas.append(
                _ordered_mapping_key_schema_projection(row)
            )
        world_uid_lines = hashlib.sha256(
            b"".join(
                str(row["world_uid"]).encode("utf-8") + b"\n"
                for row in structure_rows
            )
        ).hexdigest()
        if (
            _canonical_jsonl_projection_sha256(invariant_rows)
            != structure_reference["invariant_canonical_jsonl_sha256"]
            or world_uid_lines
            != structure_reference["world_uid_utf8_lines_sha256"]
            or _canonical_jsonl_projection_sha256(mapping_key_schemas)
            != structure_reference[
                "ordered_mapping_key_schema_canonical_jsonl_sha256"
            ]
            or _canonical_jsonl_projection_sha256(profile_commitments)
            == structure_reference[
                "old_allowed_path_commitment_canonical_jsonl_sha256"
            ]
        ):
            raise DatasetBuildError(
                f"V9.1 structure-audit invariant replay drift: {split}"
            )

        full_items = _read_jsonl_objects(
            split_root / "observed/redacted_items.jsonl"
        )
        seller_world: dict[str, str] = {}
        for row in full_items:
            seller_uid = str(row["seller_uid"])
            world_uid = str(row["world_uid"])
            if seller_uid in seller_world and seller_world[seller_uid] != world_uid:
                raise DatasetBuildError(
                    f"V9.1 seller/world ownership drift: {split}"
                )
            seller_world[seller_uid] = world_uid
        structure_by_world = {
            str(row["world_uid"]): row for row in structure_rows
        }
        if len(structure_by_world) != len(structure_rows):
            raise DatasetBuildError(f"V9.1 duplicate world UID: {split}")
        for relative, hash_field in surface_contract:
            grouped = {world_uid: [] for world_uid in structure_by_world}
            for profile in _read_jsonl_objects(split_root / relative):
                try:
                    world_uid = seller_world[str(profile["seller_uid"])]
                except KeyError as exc:
                    raise DatasetBuildError(
                        f"V9.1 seller/profile world join drift: {split}"
                    ) from exc
                grouped[world_uid].append(profile)
            if any(
                common.canonical_sha256(grouped[world_uid])
                != structure_by_world[world_uid][hash_field]
                for world_uid in structure_by_world
            ):
                raise DatasetBuildError(
                    f"V9.1 persisted profile commitment drift: {split}/{hash_field}"
                )
            if hash_field == "neutral_profile_sha256" and any(
                structure_by_world[world_uid]["neutral_receipt"][
                    "neutral_profile_sha256"
                ]
                != structure_by_world[world_uid][hash_field]
                for world_uid in structure_by_world
            ):
                raise DatasetBuildError(
                    f"V9.1 inner/outer neutral commitment drift: {split}"
                )
        changed_count += 1
        split_receipts[split] = {
            "unchanged_file_count": len(old_unchanged),
            "changed_structure_file_count": 1,
            "structure_row_count": len(structure_rows),
            "structure_invariant_canonical_jsonl_sha256": (
                _canonical_jsonl_projection_sha256(invariant_rows)
            ),
            "new_profile_commitment_canonical_jsonl_sha256": (
                _canonical_jsonl_projection_sha256(profile_commitments)
            ),
        }
    if unchanged_count != 68 or changed_count != 4:
        raise DatasetBuildError("V9.1 persisted equivalence aggregate drift")
    receipt = {
        "version": "2026-08-21-step28-v13-v1-13-v9-1-equivalence-replay-v1",
        "status": "PASS_EXACT_MECHANICAL_PROFILE_COMMITMENT_REPAIR",
        "invalidated_reference_canonical_self_hash": reference[
            "canonical_self_hash"
        ],
        "same_random_authority": True,
        "unchanged_file_count": unchanged_count,
        "changed_structure_file_count": changed_count,
        "allowed_changed_json_paths": list(
            V9_1_ALLOWED_STRUCTURE_HASH_JSON_PATHS
        ),
        "splits": split_receipts,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def _verify_output_tree(root: Path, root_manifest: Mapping[str, Any]) -> None:
    """Re-read all persisted bytes before the temporary tree may be published."""

    if common.load_json(root / "root_manifest.json") != root_manifest:
        raise DatasetBuildError("Root manifest replay drift")
    if _canonical_self_hash(root_manifest) != root_manifest.get(
        "canonical_self_hash"
    ):
        raise DatasetBuildError("Root manifest self-hash drift")
    for split in SPLITS:
        split_root = root / split
        manifest = common.load_json(split_root / "split_manifest.json")
        if _canonical_self_hash(manifest) != manifest.get("canonical_self_hash"):
            raise DatasetBuildError(f"Split manifest self-hash drift: {split}")
        if (
            root_manifest["split_manifest_self_hashes"].get(split)
            != manifest["canonical_self_hash"]
        ):
            raise DatasetBuildError(f"Root/split manifest binding drift: {split}")
        records = manifest.get("files")
        if not isinstance(records, list) or {
            str(record.get("path")) for record in records if isinstance(record, Mapping)
        } != set(EXPECTED_SPLIT_DATA_PATHS):
            raise DatasetBuildError(f"Split file universe drift: {split}")
        for record in records:
            if set(record) != {"path", "size_bytes", "sha256", "row_count"}:
                raise DatasetBuildError(f"Split file-record schema drift: {split}")
            path = (split_root / str(record["path"])).resolve()
            if split_root.resolve() not in path.parents or not path.is_file():
                raise DatasetBuildError(f"Unsafe or missing split file: {split}")
            if (
                path.stat().st_size != record["size_bytes"]
                or common.sha256_file(path) != record["sha256"]
                or _count_file_rows(path) != record["row_count"]
            ):
                raise DatasetBuildError(f"Split file replay drift: {split}/{path.name}")


def _project_model_seller_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact model-facing seller projection, with no audit columns."""

    try:
        return scientific.project_model_seller_profile(profile)
    except scientific.ScientificBuilderError as exc:
        raise DatasetBuildError(str(exc)) from exc


def _project_model_redacted_item(row: Mapping[str, Any]) -> dict[str, Any]:
    if any(name not in row for name in MODEL_REDACTED_ITEM_FIELDS):
        raise DatasetBuildError("Redacted item lacks a frozen model source field")
    if any(
        not isinstance(row[name], str) or not row[name]
        for name in MODEL_REDACTED_ITEM_JOIN_ONLY_FIELDS
    ) or any(
        not isinstance(row[name], str) for name in MODEL_REDACTED_ITEM_TEXT_FIELDS
    ):
        raise DatasetBuildError("Redacted item model-source field type drift")
    projected = {name: row[name] for name in MODEL_REDACTED_ITEM_FIELDS}
    if tuple(projected) != MODEL_REDACTED_ITEM_FIELDS:
        raise DatasetBuildError("Model redacted-item projection order drift")
    return projected


def _world_uid_sets(
    accepted: world_module.AcceptedScientificWorld,
) -> dict[str, set[str]]:
    """Collect and close all identifier universes before any row is written."""

    world = accepted.world
    values = {
        "world": {str(accepted.world_uid)},
        "seller": {
            str(row["seller_uid"]) for row in world["public"]["sellers"]
        },
        "item": {str(row["item_uid"]) for row in accepted.redacted_items},
        "pair": {str(row["canonical_pair_uid"]) for row in accepted.pair_labels},
        "query": {str(row["query_uid"]) for row in accepted.qrels},
        "controller": {
            str(row["controller_uid"]) for row in accepted.controller_membership
        },
    }
    expected = {
        "world": 1,
        "seller": 28,
        "item": len(accepted.redacted_items),
        "pair": 378,
        "query": 28,
        "controller": 12,
    }
    if set(values) != set(GLOBAL_UID_KINDS):
        raise DatasetBuildError("Global UID universe drift")
    for kind, identifiers in values.items():
        if "" in identifiers or len(identifiers) != expected[kind]:
            raise DatasetBuildError(f"World {kind} UID cardinality drift")
    if values["seller"] != {
        str(row["seller_uid"]) for row in accepted.seller_profiles
    }:
        raise DatasetBuildError("Seller profile/public UID keyset drift")
    if values["item"] != {
        str(row["item_uid"]) for row in world["public"]["items"]
    }:
        raise DatasetBuildError("Redacted/public item UID keyset drift")
    if any(
        str(row["seller_uid"]) not in values["seller"]
        or str(row["world_uid"]) not in values["world"]
        for row in accepted.redacted_items
    ):
        raise DatasetBuildError("Redacted item join-key drift")
    endpoint_pairs = {
        str(row["canonical_pair_uid"])
        for row in world["public"]["complete_model_pair_endpoints"]
    }
    if endpoint_pairs != values["pair"]:
        raise DatasetBuildError("Endpoint/private pair UID keyset drift")
    if {
        str(row["seller_uid"]) for row in accepted.controller_membership
    } != values["seller"]:
        raise DatasetBuildError("Controller membership seller keyset drift")
    if {
        str(row["query_seller_uid"]) for row in accepted.qrels
    } != values["seller"]:
        raise DatasetBuildError("Qrel query seller keyset drift")
    return values


def _commit_unique_uid_sets(
    accepted: world_module.AcceptedScientificWorld,
    *,
    seen: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    values = _world_uid_sets(accepted)
    _commit_uid_values(values, seen=seen)
    return values


def _commit_uid_values(
    values: Mapping[str, set[str]], *, seen: Mapping[str, set[str]]
) -> None:
    """Atomically commit already validated UID sets to global registries."""

    if set(seen) != set(GLOBAL_UID_KINDS):
        raise DatasetBuildError("Global UID registry schema drift")
    if set(values) != set(GLOBAL_UID_KINDS):
        raise DatasetBuildError("World UID registry schema drift")
    collisions = {
        kind: sorted(values[kind] & seen[kind], key=lambda value: value.encode("utf-8"))
        for kind in GLOBAL_UID_KINDS
        if values[kind] & seen[kind]
    }
    if collisions:
        kind = next(name for name in GLOBAL_UID_KINDS if name in collisions)
        raise DatasetBuildError(
            f"Cross-world {kind} UID reuse: {collisions[kind][0]}"
        )
    for kind in GLOBAL_UID_KINDS:
        seen[kind].update(values[kind])


def _private_world_audit_row(
    accepted: world_module.AcceptedScientificWorld,
) -> dict[str, Any]:
    private = accepted.world["private"]
    return {
        "world_uid": accepted.world_uid,
        "split": accepted.split,
        "split_ordinal": accepted.split_ordinal,
        "structural_parent_sha256": accepted.structural_parent_sha256,
        "candidate_zero_lineage_reference_sha256": (
            accepted.candidate_zero_lineage_reference_sha256
        ),
        "document_capacity_receipt": accepted.document_capacity_receipt,
        "document_capacity_audit": accepted.document_capacity_audit,
        "profile_provenance_sha256": accepted.profile_provenance_sha256,
        "identity33_sha256": accepted.identity33_sha256,
        "controller_style_groups": private["controller_style_groups"],
        "mechanism_assignments": private["mechanism_assignments"],
        "identity_assets": private["identity_assets"],
        "identity_slots_audit": private["identity_slots_audit"],
        "positive_targets": private["positive_targets"],
        "negative_flags": private["negative_flags"],
        "override_audit": private["override_audit"],
        "exact_title_clone_endpoint_qualification": private[
            "exact_title_clone_endpoint_qualification"
        ],
        "solver_audit": private["solver_audit"],
    }


def _write_world(
    writers: _SplitWriters,
    accepted: world_module.AcceptedScientificWorld,
) -> None:
    writers.worlds.write(
        {
            "world_uid": accepted.world_uid,
            "split_ordinal": accepted.split_ordinal,
        }
    )
    for row in sorted(
        accepted.redacted_items,
        key=lambda value: str(value["item_uid"]).encode("utf-8"),
    ):
        writers.redacted_items.write(_project_model_redacted_item(row))
    for row in sorted(
        accepted.masked_redacted_items,
        key=lambda value: str(value["item_uid"]).encode("utf-8"),
    ):
        writers.masked_redacted_items.write(_project_model_redacted_item(row))
    for row in sorted(
        accepted.neutral_redacted_items,
        key=lambda value: str(value["item_uid"]).encode("utf-8"),
    ):
        writers.neutral_redacted_items.write(_project_model_redacted_item(row))
    for row in sorted(
        accepted.seller_profiles,
        key=lambda value: str(value["seller_uid"]).encode("utf-8"),
    ):
        writers.model_seller_profiles.write(_project_model_seller_profile(row))
    for row in sorted(
        accepted.masked_seller_profiles,
        key=lambda value: str(value["seller_uid"]).encode("utf-8"),
    ):
        writers.masked_model_seller_profiles.write(
            _project_model_seller_profile(row)
        )
    for row in sorted(
        accepted.neutral_seller_profiles,
        key=lambda value: str(value["seller_uid"]).encode("utf-8"),
    ):
        writers.neutral_model_seller_profiles.write(
            _project_model_seller_profile(row)
        )
    for row in sorted(
        accepted.world["public"]["complete_model_pair_endpoints"],
        key=lambda value: (
            str(value["world_uid"]).encode("utf-8"),
            str(value["canonical_pair_uid"]).encode("utf-8"),
        ),
    ):
        writers.endpoints.write(row)
    for row in accepted.identity33:
        writers.identity33.write(row)
    for row in accepted.controller_membership:
        writers.controller_membership.write(row)
    for row in accepted.pair_labels:
        writers.pair_labels.write(row)
    for row in accepted.qrels:
        writers.qrels.write(row)
    writers.private_world_audit.write(_private_world_audit_row(accepted))
    writers.collision_attempts.write(
        {
            "world_uid": accepted.world_uid,
            "split": accepted.split,
            "split_ordinal": accepted.split_ordinal,
            "accepted_candidate_index": accepted.candidate_index,
            "candidates_examined": accepted.candidates_examined,
            "rejection_counts": accepted.rejection_counts,
            "code_registry_delta_count": len(accepted.code_registry_delta),
            "code_registry_delta_sha256": common.canonical_sha256(
                accepted.code_registry_delta
            ),
            "item_registry_delta_count": len(accepted.item_registry_delta),
            "item_registry_delta_sha256": common.canonical_sha256(
                accepted.item_registry_delta
            ),
            "seller_registry_delta_count": len(accepted.seller_registry_delta),
            "seller_registry_delta_sha256": common.canonical_sha256(
                accepted.seller_registry_delta
            ),
            "natural_output_sha256": accepted.natural_output_sha256,
        }
    )
    writers.identity_allocation.write(
        {
            "world_uid": accepted.world_uid,
            "split": accepted.split,
            "split_ordinal": accepted.split_ordinal,
            "identity_registry_delta_count": len(accepted.identity_registry_delta),
            "identity_registry_delta_sha256": common.canonical_sha256(
                accepted.identity_registry_delta
            ),
            "receipt": accepted.identity_allocation_receipt,
        }
    )
    for row in accepted.public_code_probe_input:
        writers.public_code_probe_input.write(row)
    for row in accepted.text_probe_eligibility_input:
        writers.text_probe_eligibility_input.write(row)
    writers.channel_structure_audit.write(accepted.channel_structure_audit)


def _validate_split_counts(
    *,
    split: str,
    world_count: int,
    writers: _SplitWriters,
    positive_count: int,
    expected_item_count: int,
    item_document_hashes: set[str],
    seller_document_hashes: set[str],
    identity_value_hashes: set[str],
) -> None:
    expected_pairs = world_count * 378
    expected_sellers = world_count * 28
    exact = {
        "worlds": (writers.worlds.row_count, world_count),
        "model_seller_profiles": (
            writers.model_seller_profiles.row_count,
            expected_sellers,
        ),
        "masked_model_seller_profiles": (
            writers.masked_model_seller_profiles.row_count,
            expected_sellers,
        ),
        "neutral_model_seller_profiles": (
            writers.neutral_model_seller_profiles.row_count,
            expected_sellers,
        ),
        "redacted_items": (writers.redacted_items.row_count, expected_item_count),
        "masked_redacted_items": (
            writers.masked_redacted_items.row_count,
            expected_item_count,
        ),
        "neutral_redacted_items": (
            writers.neutral_redacted_items.row_count,
            expected_item_count,
        ),
        "endpoints": (writers.endpoints.row_count, expected_pairs),
        "identity33": (writers.identity33.row_count, expected_pairs),
        "controller_membership": (
            writers.controller_membership.row_count,
            expected_sellers,
        ),
        "pair_labels": (writers.pair_labels.row_count, expected_pairs),
        "qrels": (writers.qrels.row_count, expected_sellers),
        "private_world_audit": (writers.private_world_audit.row_count, world_count),
        "collision_attempts": (writers.collision_attempts.row_count, world_count),
        "identity_allocation": (writers.identity_allocation.row_count, world_count),
        "public_code_probe_input": (
            writers.public_code_probe_input.row_count,
            expected_sellers,
        ),
        "text_probe_eligibility_input": (
            writers.text_probe_eligibility_input.row_count,
            expected_pairs,
        ),
        "channel_structure_audit": (
            writers.channel_structure_audit.row_count,
            world_count,
        ),
    }
    failures = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in exact.items()
        if observed != expected
    }
    if failures:
        raise DatasetBuildError(f"Split row-count drift {split}: {failures}")
    if positive_count != world_count * 20:
        raise DatasetBuildError(f"Split positive-pair count drift: {split}")
    if expected_item_count <= 0 or len(item_document_hashes) != expected_item_count:
        raise DatasetBuildError(f"Split item-document registry drift: {split}")
    if len(seller_document_hashes) != expected_sellers:
        raise DatasetBuildError(f"Split seller-document registry drift: {split}")
    if not identity_value_hashes:
        raise DatasetBuildError(f"Split identity-value registry is empty: {split}")


def _safe_remove_temp(temp_root: Path, *, expected_output: Path) -> None:
    if (
        temp_root.parent != expected_output.parent
        or temp_root.name != f".{expected_output.name}.building"
        or expected_output.parent == ROOT
    ):
        raise DatasetBuildError("Refusing to remove an unexpected temporary path")
    if temp_root.exists():
        shutil.rmtree(temp_root)


def _repo_file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
        size_bytes = resolved.stat().st_size
    except (OSError, ValueError) as exc:
        raise DatasetBuildError("Pinned authorization source is unavailable") from exc
    return {
        "path": relative,
        "size_bytes": size_bytes,
        "sha256": common.sha256_file(resolved),
    }


def _policy_binding(path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_repo_file_binding(path),
        "canonical_self_hash": str(policy["canonical_self_hash"]),
    }


def _git_identity() -> tuple[str, str]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if status.stdout:
            raise DatasetBuildError(
                "Tracked Git worktree is not clean for the reviewed build"
            )
        values = []
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
        raise DatasetBuildError("Reviewed Git identity cannot be verified") from exc
    if any(GIT_OBJECT_RE.fullmatch(value) is None for value in values):
        raise DatasetBuildError("Reviewed Git identity is malformed")
    return values[0], values[1]


def _authorization_receipt_path(policy: Mapping[str, Any]) -> Path:
    overlay = policy["design_build_authorization_overlay"]
    path = common.repo_path(str(overlay["receipt_path"]))
    private_root = (ROOT / "private_custody").resolve()
    if private_root not in path.parents or path.parent != private_root:
        raise DatasetBuildError("Design-build receipt path escaped private custody")
    return path


def _load_strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise DatasetBuildError("Design-build receipt has duplicate keys")
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise DatasetBuildError("Design-build receipt has a non-finite value")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetBuildError("Design-build receipt is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DatasetBuildError("Design-build receipt must be a JSON object")
    return value


def _load_and_validate_design_build_receipt(
    policy: Mapping[str, Any],
) -> tuple[_VerifiedDesignBuildReceipt, dict[str, Any]]:
    path = _authorization_receipt_path(policy)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetBuildError(
            "Design build remains unauthorized: exact one-time receipt is absent"
        ) from exc
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    receipt = _load_strict_json_object(raw)
    expected_keys = {
        "version",
        "status",
        "claim_boundary",
        "review_final_line",
        "review_conversation_url",
        "review_response_sha256",
        "reviewed_at_utc",
        "execution_mode",
        "attempt_index",
        "world_counts",
        "output_root",
        "random_authority_commitment_sha256",
        "builder_policy",
        "quality_policy",
        "builder_source",
        "git_commit",
        "git_tree",
        "canonical_self_hash",
    }
    if set(receipt) != expected_keys:
        raise DatasetBuildError("Design-build receipt schema drift")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("canonical_self_hash")
    if (
        not isinstance(receipt_id, str)
        or HEX_SHA256_RE.fullmatch(receipt_id) is None
        or common.canonical_sha256(unsigned) != receipt_id
    ):
        raise DatasetBuildError("Design-build receipt self-hash drift")

    overlay = policy["design_build_authorization_overlay"]
    quality_policy = quality_policy_module.load_policy()
    quality_authorization = quality_policy["authorization"]
    if quality_authorization.get("implementation_and_fixture_tests") is not True or any(
        value is not False
        for key, value in quality_authorization.items()
        if key != "implementation_and_fixture_tests"
    ):
        raise DatasetBuildError("Quality or training authorization unexpectedly opened")
    git_commit, git_tree = _git_identity()
    builder_policy_binding = _policy_binding(scientific.DEFAULT_POLICY_PATH, policy)
    quality_policy_binding = _policy_binding(
        quality_policy_module.DEFAULT_POLICY, quality_policy
    )
    builder_source = _repo_file_binding(Path(__file__))
    mode_spec = policy["execution_modes"][DESIGN_EXECUTION_MODE]
    random_authority_commitment = common.canonical_sha256(
        policy["public_preflight_keys"][DESIGN_EXECUTION_MODE]
    )
    fixed_expected = {
        "version": overlay["receipt_version"],
        "status": AUTHORIZATION_STATUS,
        "claim_boundary": AUTHORIZATION_CLAIM_BOUNDARY,
        "review_final_line": overlay["required_review_final_line"],
        "execution_mode": DESIGN_EXECUTION_MODE,
        "attempt_index": policy["single_attempt_random_authority"]["attempt_index"],
        "world_counts": mode_spec["world_counts"],
        "output_root": mode_spec["output_root"],
        "random_authority_commitment_sha256": random_authority_commitment,
        "builder_policy": builder_policy_binding,
        "quality_policy": quality_policy_binding,
        "builder_source": builder_source,
        "git_commit": git_commit,
        "git_tree": git_tree,
    }
    for key, expected in fixed_expected.items():
        if receipt.get(key) != expected:
            raise DatasetBuildError("Design-build receipt binding drift")
    if (
        not isinstance(receipt["review_conversation_url"], str)
        or not receipt["review_conversation_url"].startswith(
            "https://chatgpt.com/c/"
        )
        or not isinstance(receipt["review_response_sha256"], str)
        or HEX_SHA256_RE.fullmatch(receipt["review_response_sha256"]) is None
        or not isinstance(receipt["reviewed_at_utc"], str)
        or REVIEWED_AT_RE.fullmatch(receipt["reviewed_at_utc"]) is None
    ):
        raise DatasetBuildError("Design-build review metadata drift")
    return (
        _VerifiedDesignBuildReceipt(
            path=path,
            size_bytes=len(raw),
            sha256=receipt_sha256,
            receipt_id=receipt_id,
            review_response_sha256=receipt["review_response_sha256"],
            git_commit=git_commit,
            git_tree=git_tree,
            random_authority_commitment_sha256=random_authority_commitment,
            builder_policy_binding=builder_policy_binding,
            quality_policy_binding=quality_policy_binding,
            builder_source_file=builder_source,
        ),
        dict(quality_policy),
    )


def _expected_receipt_relative_path(
    policy: Mapping[str, Any], *, receipt_sha256: str, consumed: bool
) -> str:
    source = Path(policy["design_build_authorization_overlay"]["receipt_path"])
    if not consumed:
        return source.as_posix()
    return source.with_name(
        f"{source.stem}.consumed.{receipt_sha256}.json"
    ).as_posix()


def _validate_design_preflight_context_lineage(
    context: AuthorizedDesignPreflightContext,
    *,
    receipt_consumed: bool,
) -> None:
    scientific.validate_policy(context.builder_policy)
    execution = context.execution_context
    mode_spec = context.builder_policy["execution_modes"][DESIGN_EXECUTION_MODE]
    observed_counts = Counter(str(row["split"]) for row in execution.world_records)
    if (
        execution.execution_mode != DESIGN_EXECUTION_MODE
        or execution.base_mode != "development_smoke"
        or execution.scientific_use_forbidden is not True
        or execution.output_root
        != common.repo_path(str(mode_spec["output_root"]))
        or dict(observed_counts) != mode_spec["world_counts"]
        or len(execution.world_records)
        != context.builder_policy["single_attempt_random_authority"][
            "total_world_count"
        ]
    ):
        raise DatasetBuildError("Authorized design-preflight context drift")
    current_builder_policy = scientific.load_policy()
    current_quality_policy = quality_policy_module.load_policy()
    current_commit, current_tree = _git_identity()
    receipt_file = context.receipt_file
    if not isinstance(receipt_file, Mapping):
        raise DatasetBuildError("Authorized design-preflight lineage drift")
    receipt_sha256 = receipt_file.get("sha256")
    receipt_path = receipt_file.get("path")
    if (
        current_builder_policy != context.builder_policy
        or _policy_binding(scientific.DEFAULT_POLICY_PATH, current_builder_policy)
        != {
            **_repo_file_binding(scientific.DEFAULT_POLICY_PATH),
            "canonical_self_hash": context.builder_policy["canonical_self_hash"],
        }
        or _policy_binding(
            quality_policy_module.DEFAULT_POLICY, current_quality_policy
        )
        != context.quality_policy_binding
        or _repo_file_binding(Path(__file__)) != context.builder_source_file
        or (current_commit, current_tree) != (context.git_commit, context.git_tree)
        or context.random_authority_commitment_sha256
        != common.canonical_sha256(
            context.builder_policy["public_preflight_keys"][DESIGN_EXECUTION_MODE]
        )
        or not isinstance(context.receipt_id, str)
        or HEX_SHA256_RE.fullmatch(context.receipt_id) is None
        or not isinstance(context.review_response_sha256, str)
        or HEX_SHA256_RE.fullmatch(context.review_response_sha256) is None
        or set(receipt_file) != {"path", "size_bytes", "sha256"}
        or type(receipt_file["size_bytes"]) is not int
        or receipt_file["size_bytes"] <= 0
        or not isinstance(receipt_sha256, str)
        or HEX_SHA256_RE.fullmatch(receipt_sha256) is None
        or receipt_path
        != _expected_receipt_relative_path(
            context.builder_policy,
            receipt_sha256=receipt_sha256,
            consumed=receipt_consumed,
        )
    ):
        raise DatasetBuildError("Authorized design-preflight lineage drift")


def _validate_pending_design_preflight_context(
    context: AuthorizedDesignPreflightContext,
) -> None:
    _validate_design_preflight_context_lineage(
        context, receipt_consumed=False
    )


def _validate_authorized_design_preflight_context(
    context: AuthorizedDesignPreflightContext,
) -> None:
    _validate_design_preflight_context_lineage(
        context, receipt_consumed=True
    )


def _consume_design_build_receipt(
    receipt: _VerifiedDesignBuildReceipt,
) -> dict[str, Any]:
    consumed = receipt.path.with_name(
        f"{receipt.path.stem}.consumed.{receipt.sha256}.json"
    )
    if consumed.exists():
        raise DatasetBuildError("Design-build receipt was already consumed")
    try:
        if common.sha256_file(receipt.path) != receipt.sha256:
            raise DatasetBuildError("Design-build receipt changed before consumption")
        receipt.path.replace(consumed)
        if common.sha256_file(consumed) != receipt.sha256:
            raise DatasetBuildError("Consumed design-build receipt bytes drift")
    except OSError as exc:
        raise DatasetBuildError("Design-build receipt could not be consumed") from exc
    return {
        "path": consumed.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }


def run_design_preflight_once() -> dict[str, Any]:
    """Run the sole write-capable mode after consuming an exact external receipt."""

    policy = scientific.load_policy()
    _validate_model_mount_contract(policy)
    receipt, _quality_policy = _load_and_validate_design_build_receipt(policy)
    execution = scientific.build_execution_context(
        policy, execution_mode=DESIGN_EXECUTION_MODE
    )
    original_receipt_file = {
        "path": receipt.path.relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": receipt.size_bytes,
        "sha256": receipt.sha256,
    }
    pending = AuthorizedDesignPreflightContext(
        execution_context=execution,
        builder_policy=dict(policy),
        quality_policy_binding=receipt.quality_policy_binding,
        builder_source_file=receipt.builder_source_file,
        git_commit=receipt.git_commit,
        git_tree=receipt.git_tree,
        random_authority_commitment_sha256=(
            receipt.random_authority_commitment_sha256
        ),
        receipt_id=receipt.receipt_id,
        review_response_sha256=receipt.review_response_sha256,
        receipt_file=original_receipt_file,
    )
    _validate_pending_design_preflight_context(pending)
    output_root = execution.output_root
    temp_root = output_root.parent / f".{output_root.name}.building"
    if output_root.exists() or temp_root.exists():
        raise DatasetBuildError(
            "Design output or temporary root already exists; resume is forbidden"
        )
    consumed_receipt_file = _consume_design_build_receipt(receipt)
    authorized = AuthorizedDesignPreflightContext(
        execution_context=execution,
        builder_policy=dict(policy),
        quality_policy_binding=receipt.quality_policy_binding,
        builder_source_file=receipt.builder_source_file,
        git_commit=receipt.git_commit,
        git_tree=receipt.git_tree,
        random_authority_commitment_sha256=(
            receipt.random_authority_commitment_sha256
        ),
        receipt_id=receipt.receipt_id,
        review_response_sha256=receipt.review_response_sha256,
        receipt_file=consumed_receipt_file,
    )
    return _run_design_preflight_transaction(context=authorized)


def _run_design_preflight_transaction(
    *, context: AuthorizedDesignPreflightContext
) -> dict[str, Any]:
    """Write one complete design root from a previously authorized context."""

    _validate_authorized_design_preflight_context(context)
    authorization = context
    policy = authorization.builder_policy
    context = authorization.execution_context
    execution_mode = context.execution_mode
    output_root = context.output_root
    temp_root = output_root.parent / f".{output_root.name}.building"
    if output_root.exists() or temp_root.exists():
        raise DatasetBuildError(
            "Immutable output or temporary root already exists; resume is forbidden"
        )
    temp_root.mkdir(parents=True, exist_ok=False)
    completed = False
    writers_by_split: dict[str, _SplitWriters] = {}
    try:
        template, fixture, style_profile = scientific.load_release_inputs(context)
        historical = collision.load_historical_exclusion_registries()
        endpoint_fields = tuple(
            context.effective_policy["relational_integrity"][
                "pair_projection_contract"
            ]["complete_model_pair_endpoints_schema"]
        )
        identity_fields = (
            "canonical_pair_uid",
            "world_uid",
            *tuple(context.effective_policy["history_features"]["feature_names"]),
        )
        for split in SPLITS:
            writers_by_split[split] = _SplitWriters.open(
                temp_root / split,
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
            )

        current_item_hashes: set[str] = set()
        current_seller_hashes: set[str] = set()
        current_identity_hashes: set[str] = set()
        current_item_codes: set[str] = set()
        seen_uids: dict[str, set[str]] = {
            kind: set() for kind in GLOBAL_UID_KINDS
        }
        split_uid_sets: dict[str, dict[str, set[str]]] = {
            split: {kind: set() for kind in GLOBAL_UID_KINDS} for split in SPLITS
        }
        split_item_document_hashes: dict[str, set[str]] = {
            split: set() for split in SPLITS
        }
        split_seller_document_hashes: dict[str, set[str]] = {
            split: set() for split in SPLITS
        }
        split_identity_value_hashes: dict[str, set[str]] = {
            split: set() for split in SPLITS
        }
        split_item_codes: dict[str, set[str]] = {split: set() for split in SPLITS}
        positive_counts: Counter[str] = Counter()
        candidate_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        rejection_totals: dict[str, Counter[str]] = defaultdict(Counter)
        split_world_counts: Counter[str] = Counter()
        split_ordinals: dict[str, set[int]] = {split: set() for split in SPLITS}
        records = sorted(
            context.world_records,
            key=lambda row: (
                SPLITS.index(str(row["split"])),
                int(row["split_ordinal"]),
            ),
        )
        for position, record in enumerate(records, start=1):
            split = str(record["split"])
            split_ordinal = record["split_ordinal"]
            expected_split_worlds = int(
                policy["execution_modes"][execution_mode]["world_counts"][split]
            )
            if (
                isinstance(split_ordinal, bool)
                or not isinstance(split_ordinal, int)
                or not 0 <= split_ordinal < expected_split_worlds
                or split_ordinal in split_ordinals[split]
            ):
                raise DatasetBuildError(f"Split world ordinal drift: {split}")
            split_ordinals[split].add(split_ordinal)
            structure_key = common.structure_key_for_split(
                context.effective_policy,
                mode=context.base_mode,
                split=split,
            )
            accepted = world_module.build_scientific_world(
                policy=context.effective_policy,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                mode=context.base_mode,
                world_record=record,
                structure_key_hex=structure_key,
                document_variation_key=context.document_variation_key,
                anonymous_handle_key=context.anonymous_handle_key,
                historical_item_hashes=historical.item_document_hashes,
                historical_seller_hashes=historical.seller_document_hashes,
                historical_identity_hashes=historical.identity_value_hashes,
                current_item_hashes=current_item_hashes,
                current_seller_hashes=current_seller_hashes,
                current_identity_hashes=current_identity_hashes,
                current_item_codes=current_item_codes,
                candidate_limit=int(policy["candidate_selection"]["candidate_limit"]),
                identity_maximum_counter=int(
                    policy["candidate_selection"][
                        "identity_value_maximum_counter"
                    ]
                ),
            )
            world_uid_sets = _commit_unique_uid_sets(accepted, seen=seen_uids)
            for kind in GLOBAL_UID_KINDS:
                split_uid_sets[split][kind].update(world_uid_sets[kind])
            if (
                split_item_document_hashes[split]
                & set(accepted.item_registry_delta)
                or split_seller_document_hashes[split]
                & set(accepted.seller_registry_delta)
                or split_identity_value_hashes[split]
                & set(accepted.identity_registry_delta)
                or split_item_codes[split] & set(accepted.code_registry_delta)
            ):
                raise DatasetBuildError("Within-split registry delta reuse")
            split_item_document_hashes[split].update(accepted.item_registry_delta)
            split_seller_document_hashes[split].update(
                accepted.seller_registry_delta
            )
            split_identity_value_hashes[split].update(
                accepted.identity_registry_delta
            )
            split_item_codes[split].update(accepted.code_registry_delta)
            _write_world(writers_by_split[split], accepted)
            split_world_counts[split] += 1
            positive_counts[split] += sum(row["label"] for row in accepted.pair_labels)
            candidate_histograms[split][accepted.candidate_index] += 1
            rejection_totals[split].update(accepted.rejection_counts)
            print(
                json.dumps(
                    {
                        "event": "world_complete",
                        "execution_mode": execution_mode,
                        "position": position,
                        "total_worlds": len(records),
                        "split": split,
                        "split_ordinal": accepted.split_ordinal,
                        "candidate_index": accepted.candidate_index,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        split_manifests: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            writers = writers_by_split[split]
            expected_worlds = int(
                policy["execution_modes"][execution_mode]["world_counts"][split]
            )
            if split_ordinals[split] != set(range(expected_worlds)):
                raise DatasetBuildError(f"Split world ordinals are not contiguous: {split}")
            _validate_split_counts(
                split=split,
                world_count=expected_worlds,
                writers=writers,
                positive_count=positive_counts[split],
                expected_item_count=len(split_item_document_hashes[split]),
                item_document_hashes=split_item_document_hashes[split],
                seller_document_hashes=split_seller_document_hashes[split],
                identity_value_hashes=split_identity_value_hashes[split],
            )
            writers.close()
            files = [
                _file_record(
                    writer.path, root=temp_root / split, row_count=writer.row_count
                )
                for writer in writers.all_writers()
            ]
            files.sort(key=lambda row: row["path"].encode("utf-8"))
            manifest = {
                "version": policy["version"],
                "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
                "execution_mode": execution_mode,
                "split": split,
                "world_count": expected_worlds,
                "world_ordinal_count": len(split_ordinals[split]),
                "world_ordinals_sha256": common.canonical_sha256(
                    sorted(split_ordinals[split])
                ),
                "seller_count": expected_worlds * 28,
                "pair_count": expected_worlds * 378,
                "positive_pair_count": expected_worlds * 20,
                "negative_pair_count": expected_worlds * 358,
                "item_count": len(split_item_document_hashes[split]),
                "item_code_registry_count": len(split_item_codes[split]),
                "item_code_registry_sha256": common.canonical_sha256(
                    sorted(split_item_codes[split])
                ),
                "item_document_registry_count": len(
                    split_item_document_hashes[split]
                ),
                "item_document_registry_sha256": common.canonical_sha256(
                    sorted(split_item_document_hashes[split])
                ),
                "seller_document_registry_count": len(
                    split_seller_document_hashes[split]
                ),
                "seller_document_registry_sha256": common.canonical_sha256(
                    sorted(split_seller_document_hashes[split])
                ),
                "identity_value_registry_count": len(
                    split_identity_value_hashes[split]
                ),
                "identity_value_registry_sha256": common.canonical_sha256(
                    sorted(split_identity_value_hashes[split])
                ),
                "uid_registries": {
                    kind: {
                        "count": len(split_uid_sets[split][kind]),
                        "sha256": common.canonical_sha256(
                            sorted(split_uid_sets[split][kind])
                        ),
                    }
                    for kind in GLOBAL_UID_KINDS
                },
                "candidate_index_histogram": {
                    str(key): value
                    for key, value in sorted(candidate_histograms[split].items())
                },
                "collision_rejection_totals": {
                    name: int(rejection_totals[split][name])
                    for name in world_module.COLLISION_CATEGORIES
                },
                "files": files,
            }
            manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
            common.write_json(temp_root / split / "split_manifest.json", manifest)
            split_manifests[split] = manifest
        writers_by_split.clear()

        v9_1_equivalence_replay = _validate_v9_1_persisted_equivalence(
            temp_root=temp_root,
            split_manifests=split_manifests,
            policy=policy,
        )

        expected_total = sum(
            int(value)
            for value in policy["execution_modes"][execution_mode][
                "world_counts"
            ].values()
        )
        if len(seen_uids["world"]) != expected_total:
            raise DatasetBuildError("Root world cardinality drift")
        for kind in GLOBAL_UID_KINDS:
            if len(seen_uids[kind]) != sum(
                len(split_uid_sets[split][kind]) for split in SPLITS
            ):
                raise DatasetBuildError(
                    f"Cross-split {kind} UID registry intersection"
                )
        for name, global_values, split_values in (
            (
                "item document",
                current_item_hashes,
                split_item_document_hashes,
            ),
            (
                "seller document",
                current_seller_hashes,
                split_seller_document_hashes,
            ),
            (
                "identity value",
                current_identity_hashes,
                split_identity_value_hashes,
            ),
            ("item code", current_item_codes, split_item_codes),
        ):
            if len(global_values) != sum(
                len(split_values[split]) for split in SPLITS
            ):
                raise DatasetBuildError(f"Cross-split {name} registry intersection")
        root_manifest = {
            "version": policy["version"],
            "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
            "execution_mode": execution_mode,
            "scientific_use_forbidden": context.scientific_use_forbidden,
            "formal_seed_created": False,
            "formal_rows_created": 0,
            "training_started": False,
            "scientific_contract": policy["scientific_contract"],
            "builder_policy_canonical_self_hash": policy["canonical_self_hash"],
            "builder_policy_file": _repo_file_binding(
                scientific.DEFAULT_POLICY_PATH
            ),
            "quality_policy_canonical_self_hash": authorization.quality_policy_binding[
                "canonical_self_hash"
            ],
            "quality_policy_file": {
                key: authorization.quality_policy_binding[key]
                for key in ("path", "size_bytes", "sha256")
            },
            "builder_source_file": authorization.builder_source_file,
            "design_build_authorization": {
                "status": "CONSUMED_ONE_TIME_DESIGN_PREFLIGHT_RECEIPT",
                "receipt_id": authorization.receipt_id,
                "receipt_file": authorization.receipt_file,
                "review_response_sha256": (
                    authorization.review_response_sha256
                ),
                "review_final_line": policy[
                    "design_build_authorization_overlay"
                ]["required_review_final_line"],
                "git_commit": authorization.git_commit,
                "git_tree": authorization.git_tree,
                "execution_mode": DESIGN_EXECUTION_MODE,
                "attempt_index": policy["single_attempt_random_authority"][
                    "attempt_index"
                ],
                "world_counts": policy["execution_modes"][
                    DESIGN_EXECUTION_MODE
                ]["world_counts"],
                "output_root": policy["execution_modes"][
                    DESIGN_EXECUTION_MODE
                ]["output_root"],
                "random_authority_commitment_sha256": (
                    authorization.random_authority_commitment_sha256
                ),
                "base_policy_alone_authorized_run": False,
            },
            "split_order": list(SPLITS),
            "world_count": expected_total,
            "seller_count": expected_total * 28,
            "pair_count": expected_total * 378,
            "positive_pair_count": expected_total * 20,
            "negative_pair_count": expected_total * 358,
            "uid_registries": {
                kind: {
                    "count": len(seen_uids[kind]),
                    "sha256": common.canonical_sha256(sorted(seen_uids[kind])),
                }
                for kind in GLOBAL_UID_KINDS
            },
            "item_document_registry_count": len(current_item_hashes),
            "item_document_registry_sha256": common.canonical_sha256(
                sorted(current_item_hashes)
            ),
            "seller_document_registry_count": len(current_seller_hashes),
            "seller_document_registry_sha256": common.canonical_sha256(
                sorted(current_seller_hashes)
            ),
            "item_code_registry_count": len(current_item_codes),
            "item_code_registry_sha256": common.canonical_sha256(
                sorted(current_item_codes)
            ),
            "identity_value_registry_count": len(current_identity_hashes),
            "identity_value_registry_sha256": common.canonical_sha256(
                sorted(current_identity_hashes)
            ),
            "historical_exclusion_counts": {
                "item_documents": len(historical.item_document_hashes),
                "seller_documents": len(historical.seller_document_hashes),
                "identity_values": len(historical.identity_value_hashes),
            },
            "v9_1_equivalence_replay": v9_1_equivalence_replay,
            "split_manifest_self_hashes": {
                split: split_manifests[split]["canonical_self_hash"]
                for split in SPLITS
            },
        }
        root_manifest["canonical_self_hash"] = common.canonical_sha256(root_manifest)
        common.write_json(temp_root / "root_manifest.json", root_manifest)
        _verify_output_tree(temp_root, root_manifest)
        temp_root.rename(output_root)
        completed = True
        return root_manifest
    finally:
        if not completed:
            for writers in writers_by_split.values():
                for writer in writers.all_writers():
                    if not writer.handle.closed:
                        writer.handle.close()
            _safe_remove_temp(temp_root, expected_output=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-policy-only",
        action="store_true",
        help=(
            "Validate the fixed design_preflight policy and formal closure "
            "without generating rows."
        ),
    )
    args = parser.parse_args()
    if args.validate_policy_only:
        policy = scientific.load_policy()
        context = scientific.build_execution_context(
            policy, execution_mode=DESIGN_EXECUTION_MODE
        )
        try:
            scientific.build_execution_context(policy, execution_mode="formal")
        except scientific.ScientificBuilderError:
            formal_closed = True
        else:
            raise DatasetBuildError("Formal mode unexpectedly became available")
        output = {
            "status": "PASS_FIXED_DESIGN_POLICY_ONLY_NO_BUILD_AUTHORIZATION",
            "execution_mode": DESIGN_EXECUTION_MODE,
            "world_count": len(context.world_records),
            "formal_generation_closed": formal_closed,
            "formal_seed_created": False,
            "formal_rows_created": 0,
        }
    else:
        output = run_design_preflight_once()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Step28-v13 v1.13 scientific build failed: {exc}", file=sys.stderr)
        raise
