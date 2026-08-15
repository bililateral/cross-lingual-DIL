#!/usr/bin/env python3
"""Build the four-split Step28-v13 v1.13 v9 design Chinese dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import step28_v13_common as common
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_world_v9 as world_module


ROOT = Path(__file__).resolve().parents[1]
SPLITS = scientific.SPLITS

# These are the only seller-profile fields that the frozen M0/M3 base-feature
# adapter may mount.  ``seller_uid`` is a join key, never a feature.  The five
# text fields feed the frozen LaBSE path; the remaining values are exactly the
# source statistics required to reconstruct legacy18 without exposing market,
# split, candidate, raw-source, or audit metadata.
MODEL_PROFILE_JOIN_ONLY_FIELDS = ("seller_uid",)
MODEL_PROFILE_TEXT_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
MODEL_PROFILE_NUMERIC_FIELDS = (
    "item_count",
    "title_length_stats",
    "description_length_stats",
    "style_stats",
)
MODEL_PROFILE_FIELDS = (
    *MODEL_PROFILE_JOIN_ONLY_FIELDS,
    *MODEL_PROFILE_TEXT_FIELDS,
    *MODEL_PROFILE_NUMERIC_FIELDS,
)
MODEL_PROFILE_STYLE_FIELDS = (
    "digit_ratio_mean",
    "punct_ratio_mean",
    "repeated_title_share",
    "repeated_description_share",
    "max_category_share",
)
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

    missing = [name for name in MODEL_PROFILE_FIELDS if name not in profile]
    if missing:
        raise DatasetBuildError(
            f"Seller profile lacks frozen M0 source field: {missing[0]}"
        )
    title_stats = profile["title_length_stats"]
    description_stats = profile["description_length_stats"]
    style_stats = profile["style_stats"]
    if not isinstance(title_stats, Mapping) or "median" not in title_stats:
        raise DatasetBuildError("Seller title-length statistics are incomplete")
    if not isinstance(description_stats, Mapping) or "median" not in description_stats:
        raise DatasetBuildError("Seller description-length statistics are incomplete")
    if not isinstance(style_stats, Mapping) or any(
        name not in style_stats for name in MODEL_PROFILE_STYLE_FIELDS
    ):
        raise DatasetBuildError("Seller style statistics are incomplete")
    seller_uid = profile["seller_uid"]
    if not isinstance(seller_uid, str) or not seller_uid:
        raise DatasetBuildError("Seller join key must be a non-empty string")
    if any(not isinstance(profile[name], str) for name in MODEL_PROFILE_TEXT_FIELDS):
        raise DatasetBuildError("Seller model text field type drift")
    item_count = profile["item_count"]
    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count <= 0:
        raise DatasetBuildError("Seller item count must be a positive integer")
    for label, value in (
        ("title median", title_stats["median"]),
        ("description median", description_stats["median"]),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise DatasetBuildError(f"Seller {label} is not finite and non-negative")
    for name in MODEL_PROFILE_STYLE_FIELDS:
        value = style_stats[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise DatasetBuildError(f"Seller style statistic out of range: {name}")
    projected = {
        "seller_uid": seller_uid,
        **{name: str(profile[name]) for name in MODEL_PROFILE_TEXT_FIELDS},
        "item_count": profile["item_count"],
        "title_length_stats": {"median": title_stats["median"]},
        "description_length_stats": {"median": description_stats["median"]},
        "style_stats": {
            name: style_stats[name] for name in MODEL_PROFILE_STYLE_FIELDS
        },
    }
    if tuple(projected) != MODEL_PROFILE_FIELDS:
        raise DatasetBuildError("Model seller-profile projection order drift")
    return projected


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


def run_build(*, execution_mode: str) -> dict[str, Any]:
    raise DatasetBuildError(
        "V9 is implementation-only: dataset rebuild remains unauthorized; "
        "use the in-memory causal replay contract through train ordinal 283"
    )


def _run_build_unreachable_until_fresh_review(*, execution_mode: str) -> dict[str, Any]:
    """Retain the reviewed writer transaction without making it executable."""

    raise DatasetBuildError(
        "V9 writer transaction is sealed until a fresh review changes the policy"
    )

    # The unreachable transaction below remains a static implementation target;
    # no Python entry point can cross the fail-closed guard above in this release.
    policy = scientific.load_policy()
    _validate_model_mount_contract(policy)
    context = scientific.build_execution_context(policy, execution_mode=execution_mode)
    output_root = context.output_root
    temp_root = output_root.parent / f".{output_root.name}.building"
    if output_root.exists():
        raise DatasetBuildError(
            f"Immutable output already exists; remove only if documented as failed: {output_root}"
        )
    _safe_remove_temp(temp_root, expected_output=output_root)
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
        "--mode",
        choices=scientific.EXECUTION_MODES,
        default="small_smoke",
    )
    parser.add_argument(
        "--validate-policy-only",
        action="store_true",
        help="Validate frozen design inputs without generating rows.",
    )
    args = parser.parse_args()
    if args.validate_policy_only:
        policy = scientific.load_policy()
        if args.mode == "formal":
            try:
                scientific.build_execution_context(policy, execution_mode=args.mode)
            except scientific.ScientificBuilderError as exc:
                output = {
                    "status": "PASS_FORMAL_FAIL_CLOSED",
                    "reason": str(exc),
                    "formal_seed_created": False,
                    "formal_rows_created": 0,
                }
            else:
                raise DatasetBuildError("Formal mode unexpectedly became available")
        else:
            context = scientific.build_execution_context(
                policy, execution_mode=args.mode
            )
            output = {
                "status": "PASS_DESIGN_POLICY_ONLY",
                "execution_mode": args.mode,
                "world_count": len(context.world_records),
                "formal_seed_created": False,
                "formal_rows_created": 0,
            }
    else:
        output = run_build(execution_mode=args.mode)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Step28-v13 v1.13 scientific build failed: {exc}", file=sys.stderr)
        raise
