#!/usr/bin/env python3
"""Persist and consume a Step28-v13 v1.12 full-378 split stage.

The current entry point is design-only.  Formal execution remains impossible
until a separately frozen post-ceremony execution lock authorizes one split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_TABLES = {
    "worlds": ("observed/worlds.csv", "csv"),
    "sellers": ("observed/sellers.csv", "csv"),
    "redacted_items": ("observed/redacted_items.jsonl", "jsonl"),
    "seller_profiles": ("observed/seller_profiles.jsonl", "jsonl"),
    "complete_model_pair_endpoints": (
        "observed/complete_model_pair_endpoints.csv",
        "csv",
    ),
    "identity33_all_pairs": ("features/identity33_all_pairs.csv", "csv"),
    "retrieval_queries": ("retrieval/queries.csv", "csv"),
}

PRIVATE_TABLES = {
    "controller_membership": ("oracle/controller_membership.csv", "csv"),
    "controller_style_groups": (
        "oracle/controller_style_groups.csv",
        "csv",
    ),
    "mechanism_assignments": ("oracle/mechanism_assignments.csv", "csv"),
    "identity_assets": ("oracle/identity_assets.jsonl", "jsonl"),
    "positive_targets": ("oracle/positive_targets.csv", "csv"),
    "negative_flags": ("oracle/negative_flags.csv", "csv"),
    "classification_labels": (
        "oracle/classification_labels.csv",
        "csv",
    ),
    "retrieval_qrels": ("oracle/retrieval_qrels.csv", "csv"),
    "raw_identity_bearing_items": (
        "audit/raw_identity_bearing_items.jsonl",
        "jsonl",
    ),
    "history_safe_occurrences": (
        "audit/history_safe_occurrences.csv",
        "csv",
    ),
    "history_item_index": ("audit/history_item_index.csv", "csv"),
    "parsed_identity_occurrences": (
        "audit/parsed_identity_occurrences.csv",
        "csv",
    ),
    "renderer_identity_slots": (
        "audit/renderer_identity_slots.jsonl",
        "jsonl",
    ),
    "renderer_identity_slot_edits": (
        "audit/renderer_identity_slot_edits.jsonl",
        "jsonl",
    ),
    "renderer_noise_slots": (
        "audit/renderer_noise_slots.jsonl",
        "jsonl",
    ),
    "registered_override_audit": (
        "audit/registered_override_audit.jsonl",
        "jsonl",
    ),
    "render_asts": ("audit/render_asts.jsonl", "jsonl"),
    "solver_audit": ("audit/solver_audit.jsonl", "jsonl"),
    "projection_attestations": (
        "audit/history_projection_attestations.jsonl",
        "jsonl",
    ),
}


class SplitStageError(ValueError):
    """Raised when a staged split is incomplete or noncanonical."""


class StreamingTable:
    def __init__(self, path: Path, *, format_name: str) -> None:
        if format_name not in {"csv", "jsonl"}:
            raise SplitStageError("Unknown streaming table format")
        self.path = path
        self.format_name = format_name
        self.fields: list[str] | None = None
        self.row_count = 0
        os.makedirs(
            preceremony._filesystem_path(path.parent),
            exist_ok=True,
        )
        if preceremony.exists_long_path(path):
            raise SplitStageError(f"Refusing to overwrite staged table: {path}")
        self.handle: TextIO = open(
            preceremony._filesystem_path(path),
            "x",
            encoding="utf-8",
            newline="",
        )
        self.writer: csv.DictWriter[str] | None = None

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for source in rows:
            row = dict(source)
            fields = list(row)
            if self.fields is None:
                if not fields or len(fields) != len(set(fields)):
                    raise SplitStageError("First staged row has an invalid schema")
                self.fields = fields
                if self.format_name == "csv":
                    self.writer = csv.DictWriter(
                        self.handle,
                        fieldnames=self.fields,
                        extrasaction="raise",
                        lineterminator="\n",
                    )
                    self.writer.writeheader()
            elif fields != self.fields:
                raise SplitStageError(
                    f"Staged row schema/order drift: {self.path.name}"
                )
            if self.format_name == "csv":
                if any(
                    isinstance(value, (dict, list, tuple, set))
                    for value in row.values()
                ):
                    raise SplitStageError(
                        f"Nested value cannot enter CSV: {self.path.name}"
                    )
                assert self.writer is not None
                self.writer.writerow(row)
            else:
                self.handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            self.row_count += 1

    def close(self) -> None:
        if self.handle.closed:
            return
        if self.fields is None or self.row_count == 0:
            self.handle.close()
            raise SplitStageError(f"Staged table is empty: {self.path}")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    preceremony.write_bytes_no_replace_long_path(path, payload)


def _file_records(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in preceremony.walk_files_long_path(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        result = preceremony.stat_long_path(path)
        records.append(
            {
                "path": relative,
                "size_bytes": int(result.st_size),
                "sha256": preceremony.sha256_file(path),
            }
        )
    records.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    if len({str(row["path"]) for row in records}) != len(records):
        raise SplitStageError("Stage file-record path collision")
    return records


def _mount_contract(split: str) -> dict[str, Any]:
    return preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-mount-contract-v1",
            "split": split,
            "m0_allowlist": [
                "observed/seller_profiles.jsonl",
                "observed/complete_model_pair_endpoints.csv",
            ],
            "m0_text_fields_in_order": [
                "category_concat_top",
                "signature_title_concat",
                "title_concat_top",
                "signature_description_concat",
                "description_concat_top",
            ],
            "m0_all_other_profile_fields_forbidden": True,
            "identity_adapter_allowlist": [
                "features/identity33_all_pairs.csv",
                "observed/complete_model_pair_endpoints.csv",
                "external_frozen_p0_scores",
            ],
            "synthetic_labels_forbidden_to_m0": True,
            "identity33_forbidden_to_m0": True,
            "uid_endpoint_world_order_removed_before_fit": True,
            "audit_truth_mounted": False,
            "c40_mounted": False,
        }
    )


def _build_core_stage_impl(
    *,
    output_root: Path,
    split: str,
    world_count: int,
    design_only: bool,
    draft: Mapping[str, Any],
    validated_baseline: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    structure_commitments: Mapping[str, str],
    authority_reference: Mapping[str, Any],
    force_first_candidate_collision: bool = False,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Write one fresh core stage from an already validated capability."""

    if split not in formal.SPLITS or not 1 <= world_count <= 500:
        raise SplitStageError("Stage split/world count is invalid")
    if (
        set(capabilities) != {"split", "generator", "m1"}
        or capabilities["split"] != split
        or set(capabilities["generator"]) != set(formal.GENERATOR_ROLES)
        or set(structure_commitments) != set(formal.SPLITS)
        or not authority_reference
    ):
        raise SplitStageError("Stage capability/authority contract drift")
    if preceremony.exists_long_path(output_root):
        raise SplitStageError(f"Refusing an existing stage root: {output_root}")
    os.makedirs(preceremony._filesystem_path(output_root), exist_ok=False)
    public_root = output_root / "public"
    private_root = output_root / "private"
    os.makedirs(preceremony._filesystem_path(public_root), exist_ok=False)
    os.makedirs(preceremony._filesystem_path(private_root), exist_ok=False)

    execution_policy = formal.build_execution_policy(
        draft=draft,
        split=split,
        generator_capabilities=capabilities["generator"],
        structure_commitments=structure_commitments,
    )
    records = formal.split_world_records(execution_policy, split=split)[
        :world_count
    ]
    template, fixture, style_profile = formal.load_release_inputs(
        execution_policy
    )

    public_writers = {
        name: StreamingTable(public_root / relative, format_name=format_name)
        for name, (relative, format_name) in PUBLIC_TABLES.items()
    }
    if split in {"train", "development"}:
        public_writers["classification_labels"] = StreamingTable(
            public_root / "supervision/classification_labels.csv",
            format_name="csv",
        )
    private_writers = {
        name: StreamingTable(private_root / relative, format_name=format_name)
        for name, (relative, format_name) in PRIVATE_TABLES.items()
    }
    identity_allocation_writer = StreamingTable(
        private_root / "audit/identity_allocation_receipts.jsonl",
        format_name="jsonl",
    )
    allocated_identity_hashes: set[str] = set()
    all_identity_hashes: list[str] = []
    mechanism_rows: list[dict[str, Any]] = []
    aggregate_counts = {
        "world_count": 0,
        "pair_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "seller_count": 0,
        "identity33_row_count": 0,
        "identity_asset_count": 0,
        "retrieval_query_count": 0,
        "retrieval_qrel_count": 0,
    }
    all_writers = [
        *public_writers.values(),
        *private_writers.values(),
        identity_allocation_writer,
    ]
    try:
        for index, record in enumerate(records):
            bundle = formal.materialize_world_bundle(
                execution_policy=execution_policy,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                split=split,
                world_record=record,
                generator_capabilities=capabilities["generator"],
                historical_forbidden_hashes=validated_baseline[
                    "failed_identity_hashes"
                ],
                allocated_identity_hashes=allocated_identity_hashes,
                maximum_identity_counter=int(
                    draft["identity_collision_resolution"]["maximum_counter"]
                ),
                force_first_candidate_collision=(
                    force_first_candidate_collision and index == 0
                ),
            )
            for name, writer in public_writers.items():
                if name == "classification_labels":
                    writer.append(bundle["private"]["classification_labels"])
                else:
                    writer.append(bundle["public"][name])
            for name, writer in private_writers.items():
                writer.append(bundle["private"][name])
            identity_allocation_writer.append(
                [bundle["audit"]["identity_allocation"]]
            )
            all_identity_hashes.extend(bundle["audit"]["identity_value_hashes"])
            mechanism_rows.extend(bundle["private"]["mechanism_assignments"])
            labels = bundle["private"]["classification_labels"]
            aggregate_counts["world_count"] += 1
            aggregate_counts["pair_count"] += len(labels)
            aggregate_counts["positive_count"] += sum(
                int(row["label"]) for row in labels
            )
            aggregate_counts["negative_count"] += sum(
                1 - int(row["label"]) for row in labels
            )
            aggregate_counts["seller_count"] += len(
                bundle["public"]["sellers"]
            )
            aggregate_counts["identity33_row_count"] += len(
                bundle["public"]["identity33_all_pairs"]
            )
            aggregate_counts["identity_asset_count"] += len(
                bundle["private"]["identity_assets"]
            )
            aggregate_counts["retrieval_query_count"] += len(
                bundle["public"]["retrieval_queries"]
            )
            aggregate_counts["retrieval_qrel_count"] += len(
                bundle["private"]["retrieval_qrels"]
            )
            if progress_every > 0 and (index + 1) % progress_every == 0:
                prefix = "DESIGN" if design_only else "FORMAL"
                print(f"{prefix}_STAGE_PROGRESS {split} {index + 1}/{world_count}")
    except BaseException:
        for writer in all_writers:
            if not writer.handle.closed:
                writer.handle.close()
        raise
    else:
        for writer in all_writers:
            writer.close()

    expected = {
        "world_count": world_count,
        "pair_count": world_count * 378,
        "positive_count": world_count * 20,
        "negative_count": world_count * 358,
        "seller_count": world_count * 28,
        "identity33_row_count": world_count * 378,
        "identity_asset_count": formal.expected_identity_asset_count(
            draft, split=split, world_count=world_count
        ),
        "retrieval_query_count": world_count * 28,
        "retrieval_qrel_count": world_count * 40,
    }
    if aggregate_counts != expected:
        raise SplitStageError(
            f"Core stage aggregate count drift: {aggregate_counts} != {expected}"
        )
    if (
        len(all_identity_hashes) != len(set(all_identity_hashes))
        or set(all_identity_hashes) != allocated_identity_hashes
    ):
        raise SplitStageError("Core stage identity hash closure failed")
    mechanism_scope = preceremony.validate_world_scoped_mechanism_slots(
        mechanism_rows,
        expected_world_count=world_count,
    )
    identity_hash_document = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-identity-hashes-v1",
            "split": split,
            "design_only": design_only,
            "hash_count": len(all_identity_hashes),
            "hashes": sorted(all_identity_hashes),
        }
    )
    _write_json_no_replace(
        private_root / "audit/identity_value_hashes.json",
        identity_hash_document,
    )
    generation_receipt = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-core-stage-receipt-v1",
            "status": (
                "PASS_DESIGN_ONLY_CORE_STAGE"
                if design_only
                else "PASS_FORMAL_CORE_STAGE"
            ),
            "run_id": draft["run_id"],
            "split": split,
            "design_only": design_only,
            "formal_authorization_used": not design_only,
            "master_seed_access": False,
            "authority_reference": dict(authority_reference),
            "generator_capability_commitments": formal.capability_commitments(
                capabilities
            )["generator"],
            "c40_generated_or_read": False,
            "aggregate_counts": aggregate_counts,
            "mechanism_scope": mechanism_scope,
            "identity_hash_count": len(all_identity_hashes),
            "identity_hashes_sha256": preceremony.canonical_sha256(
                sorted(all_identity_hashes)
            ),
            "draft_sha256": preceremony.sha256_file(
                formal.DEFAULT_DRAFT_PATH
            ),
            "draft_canonical_self_hash": draft["canonical_self_hash"],
            "runtime_versions": formal.runtime_versions(),
            "m1_receipts_complete": False,
            "publication_occurred_during_core_generation": False,
        }
    )
    _write_json_no_replace(
        public_root / "audit/generation_integrity_receipt.json",
        generation_receipt,
    )
    _write_json_no_replace(
        public_root / "audit/mount_contract.json",
        _mount_contract(split),
    )
    return {
        "output_root": output_root,
        "public_root": public_root,
        "private_root": private_root,
        "generation_receipt": generation_receipt,
        "m1_capabilities": dict(capabilities["m1"]),
    }


def build_core_stage(
    *,
    output_root: Path,
    split: str,
    world_count: int,
    design_only: bool,
    force_first_candidate_collision: bool = False,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Write a design stage; formal work requires the separate executor."""

    if not design_only:
        raise SplitStageError(
            "Formal generation is unavailable without the execution-lock executor"
        )
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    master = bytes.fromhex(draft["randomness"]["design_only_master_hex"])
    all_capabilities = {
        name: formal.derive_capabilities(master, split=name)
        for name in formal.SPLITS
    }
    structure_commitments = {
        name: formal.capability_commitments(all_capabilities[name])[
            "generator"
        ]["structure"]
        for name in formal.SPLITS
    }
    return _build_core_stage_impl(
        output_root=output_root,
        split=split,
        world_count=world_count,
        design_only=True,
        draft=draft,
        validated_baseline=validated["baseline"],
        capabilities=all_capabilities[split],
        structure_commitments=structure_commitments,
        authority_reference={
            "kind": "design_only_formal_build_draft",
            "path": formal.DEFAULT_DRAFT_PATH.relative_to(ROOT).as_posix(),
            "sha256": preceremony.sha256_file(formal.DEFAULT_DRAFT_PATH),
        },
        force_first_candidate_collision=force_first_candidate_collision,
        progress_every=progress_every,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(
        preceremony._filesystem_path(path),
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_m1_structural_receipt(
    *,
    private_root: Path,
    public_root: Path,
    replicate: int,
    rewire_key_hex: str,
    design_only: bool,
    authority_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-read staged rows and validate one whole-row M1 capability."""

    if (
        not 1 <= replicate <= 5
        or preceremony.HEX_SHA256_RE.fullmatch(rewire_key_hex) is None
        or (not design_only and not authority_reference)
    ):
        raise SplitStageError("M1 structural replay authorization drift")
    endpoints = _read_csv(
        public_root / "observed/complete_model_pair_endpoints.csv"
    )
    identity_rows = _read_csv(public_root / "features/identity33_all_pairs.csv")
    endpoints_by_world: dict[str, list[dict[str, str]]] = {}
    identity_by_world: dict[str, list[dict[str, str]]] = {}
    for row in endpoints:
        endpoints_by_world.setdefault(row["world_uid"], []).append(row)
    for row in identity_rows:
        identity_by_world.setdefault(row["world_uid"], []).append(row)
    if set(endpoints_by_world) != set(identity_by_world):
        raise SplitStageError("M1 endpoint/identity world keyset drift")
    mapping_hashes: list[str] = []
    rewired_hashes: list[str] = []
    for world_uid in sorted(endpoints_by_world, key=str.encode):
        mapping = preceremony.build_endpoint_disjoint_derangement(
            endpoints_by_world[world_uid],
            world_uid=world_uid,
            key_hex=rewire_key_hex,
        )
        rewired = preceremony.rewire_identity33_rows(
            identity_by_world[world_uid], mapping
        )
        mapping_hashes.append(preceremony.canonical_sha256(mapping))
        rewired_hashes.append(preceremony.canonical_sha256(rewired))
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-m1-structural-v1",
            "status": (
                "PASS_DESIGN_ONLY_M1_STRUCTURAL_REPLAY"
                if design_only
                else "PASS_FORMAL_M1_STRUCTURAL_REPLAY"
            ),
            "replicate": f"r{replicate:02d}",
            "split": "train",
            "design_only": design_only,
            "formal_authorization_used": not design_only,
            "authority_reference": (
                dict(authority_reference or {}) if not design_only else None
            ),
            "world_count": len(endpoints_by_world),
            "pair_count": len(endpoints),
            "mapping_count": len(endpoints),
            "fixed_point_count": 0,
            "endpoint_overlap_count": 0,
            "whole_identity33_multiset_preserved": True,
            "rewire_key_commitment": hashlib.sha256(
                bytes.fromhex(rewire_key_hex)
            ).hexdigest(),
            "mapping_hashes_sha256": preceremony.canonical_sha256(
                mapping_hashes
            ),
            "rewired_hashes_sha256": preceremony.canonical_sha256(
                rewired_hashes
            ),
            "raw_rewire_key_persisted": False,
        }
    )
    target = private_root / f"m1/r{replicate:02d}/structural_receipt.json"
    _write_json_no_replace(target, receipt)
    return receipt


def finalize_stage(
    *,
    output_root: Path,
    split: str,
    world_count: int,
    design_only: bool,
) -> dict[str, Any]:
    public_root = output_root / "public"
    private_root = output_root / "private"
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    expected_public = set(draft["release"]["public_common_members"])
    expected_public.update(
        draft["release"]["public_supervision_members"][split]
    )
    expected_private = set(draft["release"]["private_common_members"])
    if split == "train":
        expected_private.update(
            draft["release"]["train_private_additional_members"]
        )
    observed_public_before_manifest = {
        row["path"]
        for row in _file_records(public_root, excluded={"split_manifest.json"})
    }
    observed_private_before_manifest = {
        row["path"]
        for row in _file_records(private_root, excluded={"private_manifest.json"})
    }
    if observed_public_before_manifest != expected_public - {"split_manifest.json"}:
        raise SplitStageError(
            "Design public producer/member contract is not exact"
        )
    if observed_private_before_manifest != expected_private - {"private_manifest.json"}:
        raise SplitStageError(
            "Design private producer/member contract is not exact"
        )
    private_records = _file_records(
        private_root, excluded={"private_manifest.json"}
    )
    expected_private_manifest = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-private-manifest-v1",
            "status": (
                "PASS_DESIGN_ONLY_PRIVATE_STAGE"
                if design_only
                else "PASS_FORMAL_PRIVATE_STAGE"
            ),
            "split": split,
            "design_only": design_only,
            "world_count": world_count,
            "file_count": len(private_records),
            "files": private_records,
        }
    )
    private_manifest_path = private_root / "private_manifest.json"
    public_manifest_path = public_root / "split_manifest.json"
    if preceremony.exists_long_path(
        public_manifest_path
    ) and not preceremony.exists_long_path(private_manifest_path):
        raise SplitStageError(
            "Public split manifest exists without its earlier private manifest"
        )
    if preceremony.exists_long_path(private_manifest_path):
        private_manifest = preceremony.load_json_strict(private_manifest_path)
        preceremony.validate_canonical_self_hash(
            private_manifest, label="recoverable v1.12 private manifest"
        )
        if private_manifest != expected_private_manifest:
            raise SplitStageError("Existing private manifest differs from replay")
    else:
        private_manifest = expected_private_manifest
        _write_json_no_replace(private_manifest_path, private_manifest)
    public_records = _file_records(
        public_root, excluded={"split_manifest.json"}
    )
    expected_split_manifest = preceremony.with_canonical_self_hash(
        {
            "version": draft["release"]["split_manifest_version"],
            "status": (
                "PASS_DESIGN_ONLY_PERSISTED_STAGE"
                if design_only
                else "PASS_FORMAL_PERSISTED_STAGE"
            ),
            "run_id": draft["run_id"],
            "split": split,
            "design_only": design_only,
            "formal_authorization_used": not design_only,
            "world_count": world_count,
            "pair_count": world_count * 378,
            "positive_count": world_count * 20,
            "negative_count": world_count * 358,
            "public_file_count": len(public_records),
            "public_files": public_records,
            "private_manifest_sha256": preceremony.sha256_file(
                private_root / "private_manifest.json"
            ),
            "private_manifest_canonical_self_hash": private_manifest[
                "canonical_self_hash"
            ],
            "c40_member_count": 0,
            "scientific_metrics_produced": False,
            "publication_occurred_before_manifest_freeze": False,
        }
    )
    if preceremony.exists_long_path(public_manifest_path):
        split_manifest = preceremony.load_json_strict(public_manifest_path)
        preceremony.validate_canonical_self_hash(
            split_manifest, label="recoverable v1.12 split manifest"
        )
        if split_manifest != expected_split_manifest:
            raise SplitStageError("Existing split manifest differs from replay")
    else:
        split_manifest = expected_split_manifest
        _write_json_no_replace(public_manifest_path, split_manifest)
    validate_stage(
        output_root=output_root,
        split=split,
        world_count=world_count,
        design_only=design_only,
    )
    return split_manifest


def finalize_design_stage(
    *, output_root: Path, split: str, world_count: int
) -> dict[str, Any]:
    return finalize_stage(
        output_root=output_root,
        split=split,
        world_count=world_count,
        design_only=True,
    )


def _validate_records(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_unlisted: set[str],
) -> None:
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise SplitStageError("Manifest file records are malformed")
    observed = _file_records(root, excluded=allowed_unlisted)
    expected_paths = {str(row["path"]) for row in records}
    if len(expected_paths) != len(records):
        raise SplitStageError("Manifest file records contain duplicate paths")
    observed_by_path = {str(row["path"]): row for row in observed}
    for record in records:
        path = str(record["path"])
        if observed_by_path.get(path) != dict(record):
            raise SplitStageError(f"Manifest file record drift: {path}")
    extras = set(observed_by_path) - expected_paths
    if extras:
        raise SplitStageError(f"Manifest tree contains extra files: {extras}")


def validate_stage(
    *,
    output_root: Path,
    split: str,
    world_count: int,
    design_only: bool,
) -> dict[str, Any]:
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    public_root = output_root / "public"
    private_root = output_root / "private"
    split_manifest = preceremony.load_json_strict(
        public_root / "split_manifest.json"
    )
    private_manifest = preceremony.load_json_strict(
        private_root / "private_manifest.json"
    )
    preceremony.validate_canonical_self_hash(
        split_manifest, label="v1.12 design split manifest"
    )
    preceremony.validate_canonical_self_hash(
        private_manifest, label="v1.12 design private manifest"
    )
    expected_public = set(draft["release"]["public_common_members"])
    expected_public.update(
        draft["release"]["public_supervision_members"][split]
    )
    expected_private = set(draft["release"]["private_common_members"])
    if split == "train":
        expected_private.update(
            draft["release"]["train_private_additional_members"]
        )
    public_records = split_manifest.get("public_files")
    private_records = private_manifest.get("files")
    if (
        split_manifest.get("version")
        != draft["release"]["split_manifest_version"]
        or private_manifest.get("version")
        != "2026-08-03-step28-v13-v1-12-private-manifest-v1"
        or split_manifest.get("status")
        != (
            "PASS_DESIGN_ONLY_PERSISTED_STAGE"
            if design_only
            else "PASS_FORMAL_PERSISTED_STAGE"
        )
        or private_manifest.get("status")
        != (
            "PASS_DESIGN_ONLY_PRIVATE_STAGE"
            if design_only
            else "PASS_FORMAL_PRIVATE_STAGE"
        )
        or split_manifest.get("split") != split
        or private_manifest.get("split") != split
        or split_manifest.get("run_id") != draft["run_id"]
        or int(split_manifest.get("world_count", -1)) != world_count
        or int(private_manifest.get("world_count", -1)) != world_count
        or int(split_manifest.get("pair_count", -1)) != world_count * 378
        or int(split_manifest.get("positive_count", -1)) != world_count * 20
        or int(split_manifest.get("negative_count", -1)) != world_count * 358
        or not isinstance(public_records, list)
        or not isinstance(private_records, list)
        or not all(isinstance(record, Mapping) for record in public_records)
        or not all(isinstance(record, Mapping) for record in private_records)
        or int(split_manifest.get("public_file_count", -1))
        != len(public_records or [])
        or int(private_manifest.get("file_count", -1))
        != len(private_records or [])
        or {str(record.get("path", "")) for record in public_records or []}
        != expected_public - {"split_manifest.json"}
        or {str(record.get("path", "")) for record in private_records or []}
        != expected_private - {"private_manifest.json"}
        or split_manifest.get("design_only") is not design_only
        or private_manifest.get("design_only") is not design_only
        or split_manifest.get("formal_authorization_used") is not (
            not design_only
        )
        or split_manifest.get("publication_occurred_before_manifest_freeze")
        is not False
        or int(split_manifest.get("c40_member_count", -1)) != 0
        or preceremony.sha256_file(private_root / "private_manifest.json")
        != split_manifest.get("private_manifest_sha256")
        or private_manifest.get("canonical_self_hash")
        != split_manifest.get("private_manifest_canonical_self_hash")
    ):
        raise SplitStageError("Design stage manifest semantic drift")
    _validate_records(
        public_root,
        public_records,
        allowed_unlisted={"split_manifest.json"},
    )
    _validate_records(
        private_root,
        private_records,
        allowed_unlisted={"private_manifest.json"},
    )
    return {
        "status": (
            "PASS_DESIGN_ONLY_PERSISTED_STAGE_REPLAY"
            if design_only
            else "PASS_FORMAL_PERSISTED_STAGE_REPLAY"
        ),
        "split": split,
        "world_count": world_count,
        "pair_count": world_count * 378,
        "public_manifest_sha256": preceremony.sha256_file(
            public_root / "split_manifest.json"
        ),
        "private_manifest_sha256": preceremony.sha256_file(
            private_root / "private_manifest.json"
        ),
    }


def validate_design_stage(
    *, output_root: Path, split: str, world_count: int
) -> dict[str, Any]:
    return validate_stage(
        output_root=output_root,
        split=split,
        world_count=world_count,
        design_only=True,
    )


def run_two_world_design_replay() -> dict[str, Any]:
    """Persist train/development one-world stages and consume every byte."""

    with tempfile.TemporaryDirectory(prefix="step28-v1-12-formal-stage-") as raw:
        root = Path(raw)
        results: list[dict[str, Any]] = []
        for split in ("train", "development"):
            stage = root / split
            built = build_core_stage(
                output_root=stage,
                split=split,
                world_count=1,
                design_only=True,
                force_first_candidate_collision=(split == "train"),
                progress_every=0,
            )
            if split == "train":
                for replicate, role in enumerate(formal.M1_ROLES, start=1):
                    write_m1_structural_receipt(
                        private_root=built["private_root"],
                        public_root=built["public_root"],
                        replicate=replicate,
                        rewire_key_hex=built["m1_capabilities"][role],
                        design_only=True,
                    )
            finalize_design_stage(
                output_root=stage, split=split, world_count=1
            )
            results.append(
                validate_design_stage(
                    output_root=stage, split=split, world_count=1
                )
            )
        receipt = preceremony.with_canonical_self_hash(
            {
                "version": "2026-08-03-step28-v13-v1-12-two-world-persisted-v1",
                "status": "PASS_DESIGN_ONLY_TWO_WORLD_PERSISTED_REPLAY",
                "formal_authorization_used": False,
                "formal_seed_or_key_access": False,
                "formal_rows_produced": 0,
                "design_world_count": 2,
                "design_pair_count": 756,
                "m1_structural_receipt_count": 5,
                "producer_path": (
                    "scripts/step28_v13_v1_12_generate_split.py"
                ),
                "producer_sha256": preceremony.sha256_file(Path(__file__)),
                "formal_common_sha256": preceremony.sha256_file(
                    ROOT / "scripts" / "step28_v13_v1_12_formal_common.py"
                ),
                "formal_build_draft_sha256": preceremony.sha256_file(
                    formal.DEFAULT_DRAFT_PATH
                ),
                "runtime_versions": formal.runtime_versions(),
                "stages": results,
                "temporary_stage_deleted_on_exit": True,
            }
        )
        return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-draft", action="store_true")
    parser.add_argument("--run-two-world-design-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_two_world_design_replay:
        if args.output is None:
            raise SplitStageError(
                "--run-two-world-design-replay requires a fresh --output receipt"
            )
        receipt = run_two_world_design_replay()
        _write_json_no_replace(args.output, receipt)
        print(
            receipt["status"],
            receipt["design_world_count"],
            receipt["design_pair_count"],
            preceremony.sha256_file(args.output),
        )
        return
    if args.validate_draft:
        formal.load_and_validate_draft()
        print("PASS_V1_12_FORMAL_BUILD_DRAFT_NO_AUTHORIZATION")
        return
    raise SplitStageError("Choose --validate-draft or --run-two-world-design-replay")


if __name__ == "__main__":
    main()
