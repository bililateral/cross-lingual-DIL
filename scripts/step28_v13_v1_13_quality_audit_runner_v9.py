#!/usr/bin/env python3
"""Manifest-bound v9 design-quality runner; formal execution is policy-gated."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import step28_v13_structure as structure
import step28_v13_common as common
import step28_v13_v1_13_document_capacity_v9 as document_capacity
import step28_v13_v1_13_quality_channel_policy_v9 as quality_policy_module
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_probe_validator_v9 as validator
import step28_v13_v1_13_quality_structure_aggregator_v9 as structure_aggregator
import step28_v13_v1_13_quality_truth_capability_v9 as truth_capability
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_dataset_builder_v9 as dataset_builder


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-08-14-step28-v13-v1-13-quality-audit-runner-v9"
SPLITS = ("train", "development", "audit_a", "audit_b")
SURFACE_FILES = {
    "surface_full": (
        "observed/redacted_items.jsonl",
        "observed/model_seller_profiles.jsonl",
    ),
    "surface_code_masked": (
        "observed/redacted_items.code_masked.jsonl",
        "observed/model_seller_profiles.code_masked.jsonl",
    ),
    "surface_code_neutralized": (
        "observed/redacted_items.code_neutralized.jsonl",
        "observed/model_seller_profiles.code_neutralized.jsonl",
    ),
}
ENDPOINT_PATH = "observed/complete_model_pair_endpoints.csv"
WORLDS_PATH = "observed/worlds.jsonl"
PUBLIC_CODE_PATH = "private/public_code_probe_input.jsonl"
ELIGIBILITY_PATH = "private/text_probe_eligibility_input.jsonl"
STRUCTURE_AUDIT_PATH = "private/channel_structure_audit.jsonl"
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


class QualityAuditRunnerError(ValueError):
    """Raised when the formal runner boundary or persisted input drifts."""


class DatasetGateFailure(QualityAuditRunnerError):
    """A preregistered persisted-data hard gate failed."""


class AuditorExecutionFailure(QualityAuditRunnerError):
    """The auditor, dependency, runtime, or I/O failed mechanically."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditorExecutionFailure(f"JSON I/O failed: {path.name}") from exc
    except UnicodeError as exc:
        raise DatasetGateFailure(f"JSON bytes are invalid: {path.name}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetGateFailure(f"JSON bytes are invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise DatasetGateFailure("JSON root must be an object")
    return value


def _load_jsonl(path: Path, *, expected_rows: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.endswith("\n") or not line.strip():
                    raise QualityAuditRunnerError("JSONL line framing drift")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QualityAuditRunnerError("JSONL row must be an object")
                rows.append(value)
    except OSError as exc:
        raise AuditorExecutionFailure(f"JSONL I/O failed: {path.name}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetGateFailure(f"Invalid JSONL bytes: {path.name}") from exc
    if len(rows) != expected_rows:
        raise QualityAuditRunnerError("JSONL row count drift")
    return tuple(rows)


def _load_csv(
    path: Path, *, expected_rows: int, expected_fields: Sequence[str]
) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise QualityAuditRunnerError("CSV header drift")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise AuditorExecutionFailure(f"CSV I/O failed: {path.name}") from exc
    except (UnicodeError, csv.Error) as exc:
        raise DatasetGateFailure(f"Invalid CSV bytes: {path.name}") from exc
    if len(rows) != expected_rows or any(
        tuple(row) != tuple(expected_fields) or None in row for row in rows
    ):
        raise QualityAuditRunnerError("CSV row schema/count drift")
    return rows


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise QualityAuditRunnerError("Split manifest files drift")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "size_bytes", "sha256", "row_count"}
            or not isinstance(row.get("path"), str)
            or row["path"] in records
        ):
            raise QualityAuditRunnerError("Split manifest record drift")
        records[row["path"]] = row
    return records


def _verified_source(
    *, dataset_root: Path, split: str, relative: str, record: Mapping[str, Any]
) -> tuple[Path, preparer.SourceCommitment]:
    split_root = (dataset_root / split).resolve()
    path = (split_root / relative).resolve()
    if split_root not in path.parents or not path.is_file():
        raise QualityAuditRunnerError("Label-free input path is missing or unsafe")
    size = record.get("size_bytes")
    sha256 = record.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or path.stat().st_size != size
        or _sha256_file(path) != sha256
    ):
        raise QualityAuditRunnerError("Label-free input bytes drift")
    return path, preparer.SourceCommitment(
        path=f"{split}/{relative}", size_bytes=size, sha256=sha256
    )


def _source_tuple(*values: preparer.SourceCommitment) -> tuple[preparer.SourceCommitment, ...]:
    return tuple(sorted(values, key=lambda value: value.path.encode("utf-8")))


def _repo_source(path: Path) -> preparer.SourceCommitment:
    resolved = path.resolve()
    if ROOT not in resolved.parents or not resolved.is_file():
        raise QualityAuditRunnerError("Repository authority source path drift")
    return preparer.SourceCommitment(
        path=resolved.relative_to(ROOT).as_posix(),
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
    )


def _load_split_label_free(
    *, dataset_root: Path, split: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    records = _manifest_records(manifest)
    required = {
        WORLDS_PATH,
        ENDPOINT_PATH,
        PUBLIC_CODE_PATH,
        ELIGIBILITY_PATH,
        STRUCTURE_AUDIT_PATH,
        *(path for pair in SURFACE_FILES.values() for path in pair),
    }
    if not required <= set(records):
        raise QualityAuditRunnerError("Required label-free file is absent from manifest")
    paths: dict[str, Path] = {}
    sources: dict[str, preparer.SourceCommitment] = {}
    for relative in sorted(required, key=lambda value: value.encode("utf-8")):
        paths[relative], sources[relative] = _verified_source(
            dataset_root=dataset_root,
            split=split,
            relative=relative,
            record=records[relative],
        )
    worlds = _load_jsonl(
        paths[WORLDS_PATH], expected_rows=int(records[WORLDS_PATH]["row_count"])
    )
    endpoints = _load_csv(
        paths[ENDPOINT_PATH],
        expected_rows=int(records[ENDPOINT_PATH]["row_count"]),
        expected_fields=preparer.ENDPOINT_FIELDS,
    )
    return {
        "worlds": worlds,
        "endpoints": endpoints,
        "public_code": _load_jsonl(
            paths[PUBLIC_CODE_PATH],
            expected_rows=int(records[PUBLIC_CODE_PATH]["row_count"]),
        ),
        "eligibility": _load_jsonl(
            paths[ELIGIBILITY_PATH],
            expected_rows=int(records[ELIGIBILITY_PATH]["row_count"]),
        ),
        "structure_audit": _load_jsonl(
            paths[STRUCTURE_AUDIT_PATH],
            expected_rows=int(records[STRUCTURE_AUDIT_PATH]["row_count"]),
        ),
        "surface_rows": {
            surface: (
                _load_jsonl(
                    paths[item_path],
                    expected_rows=int(records[item_path]["row_count"]),
                ),
                _load_jsonl(
                    paths[profile_path],
                    expected_rows=int(records[profile_path]["row_count"]),
                ),
            )
            for surface, (item_path, profile_path) in SURFACE_FILES.items()
        },
        "sources": sources,
    }


def _load_root_manifests(
    *, dataset_root: Path, root_pin: truth_capability.RootManifestPin
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = dataset_root.resolve()
    manifest_path = root / "root_manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != root_pin.size_bytes
        or _sha256_file(manifest_path) != root_pin.sha256
    ):
        raise QualityAuditRunnerError("Root manifest pin drift")
    root_manifest = _load_json(manifest_path)
    if (
        root_manifest.get("canonical_self_hash") != root_pin.canonical_self_hash
        or _canonical_self_hash(root_manifest) != root_pin.canonical_self_hash
    ):
        raise QualityAuditRunnerError("Root manifest self-hash drift")
    split_hashes = root_manifest.get("split_manifest_self_hashes")
    if not isinstance(split_hashes, dict) or set(split_hashes) != set(SPLITS):
        raise QualityAuditRunnerError("Root/split binding drift")
    manifests: dict[str, dict[str, Any]] = {}
    expected_files = {"root_manifest.json"}
    for split in SPLITS:
        manifest = _load_json(root / split / "split_manifest.json")
        if (
            _canonical_self_hash(manifest) != manifest.get("canonical_self_hash")
            or split_hashes[split] != manifest["canonical_self_hash"]
        ):
            raise QualityAuditRunnerError("Split manifest binding drift")
        if set(_manifest_records(manifest)) != set(EXPECTED_SPLIT_DATA_PATHS):
            raise QualityAuditRunnerError("Split manifest file universe drift")
        manifests[split] = manifest
        expected_files.add(f"{split}/split_manifest.json")
        expected_files.update(
            f"{split}/{relative}" for relative in EXPECTED_SPLIT_DATA_PATHS
        )
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise QualityAuditRunnerError("Design root physical file universe drift")
    return root_manifest, manifests


def _registry_sha256(values: set[str]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(sorted(values, key=lambda value: value.encode("utf-8")))
    ).hexdigest()


def _validate_public_uid_registries(
    *,
    root_manifest: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    loaded: Mapping[str, Mapping[str, Any]],
    expected_pairs_per_world: int = 378,
    expected_sellers_per_world: int = 28,
    expected_excluded_pairs_per_world: int = 6,
) -> None:
    if (
        isinstance(expected_pairs_per_world, bool)
        or not isinstance(expected_pairs_per_world, int)
        or expected_pairs_per_world <= 0
        or isinstance(expected_sellers_per_world, bool)
        or not isinstance(expected_sellers_per_world, int)
        or expected_sellers_per_world <= 1
        or expected_pairs_per_world
        != expected_sellers_per_world * (expected_sellers_per_world - 1) // 2
        or isinstance(expected_excluded_pairs_per_world, bool)
        or not isinstance(expected_excluded_pairs_per_world, int)
        or not 0 <= expected_excluded_pairs_per_world <= expected_pairs_per_world
    ):
        raise QualityAuditRunnerError("Endpoint scale contract drift")
    registries: dict[str, dict[str, set[str]]] = {
        split: {kind: set() for kind in ("world", "seller", "item", "pair")}
        for split in SPLITS
    }
    for split in SPLITS:
        split_data = loaded[split]
        world_rows = tuple(split_data["worlds"])
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"world_uid", "split_ordinal"}
            or type(row["world_uid"]) is not str
            or not row["world_uid"]
            or type(row["split_ordinal"]) is not int
            for row in world_rows
        ):
            raise QualityAuditRunnerError("World registry schema drift")
        ordered_world_uids = tuple(str(row["world_uid"]) for row in world_rows)
        if [int(row["split_ordinal"]) for row in world_rows] != list(
            range(len(world_rows))
        ):
            raise QualityAuditRunnerError("World split-ordinal sequence drift")
        worlds = set(ordered_world_uids)
        try:
            _row_keys, _ordered_worlds, endpoint_sellers_by_world = (
                preparer._validate_endpoints(
                    split_data["endpoints"],
                    ordered_world_uids=ordered_world_uids,
                    expected_pairs_per_world=expected_pairs_per_world,
                )
            )
        except preparer.QualityProbePreparationError as exc:
            raise QualityAuditRunnerError(
                "Per-split endpoint closure drift"
            ) from exc
        if any(
            len(values) != expected_sellers_per_world
            for values in endpoint_sellers_by_world.values()
        ):
            raise QualityAuditRunnerError("Per-world seller count drift")
        full_items = split_data["surface_rows"]["surface_full"][0]
        item_uids: set[str] = set()
        seller_uids: set[str] = set()
        item_worlds: set[str] = set()
        seller_world: dict[str, str] = {}
        for row in full_items:
            if (
                not isinstance(row, Mapping)
                or set(row) != set(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)
            ):
                raise QualityAuditRunnerError("Full item exact schema drift")
            item_uid = row.get("item_uid")
            seller_uid = row.get("seller_uid")
            world_uid = row.get("world_uid")
            if any(not isinstance(value, str) or not value for value in (item_uid, seller_uid, world_uid)):
                raise QualityAuditRunnerError("Full item UID schema drift")
            if item_uid in item_uids:
                raise QualityAuditRunnerError("Duplicate item UID within split")
            item_uids.add(item_uid)
            seller_uids.add(seller_uid)
            item_worlds.add(world_uid)
            previous_world = seller_world.setdefault(seller_uid, world_uid)
            if previous_world != world_uid:
                raise QualityAuditRunnerError("Seller UID crosses world boundary")
        full_profile_sellers: set[str] = set()
        expected_item_keys = {
            (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
            for row in full_items
        }
        policy_surface_order = tuple(SURFACE_FILES)
        for surface in policy_surface_order:
            items, profiles = split_data["surface_rows"][surface]
            if any(
                not isinstance(row, Mapping)
                or set(row) != set(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)
                for row in items
            ):
                raise QualityAuditRunnerError("Surface item exact schema drift")
            item_keys = {
                (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
                for row in items
            }
            if len(item_keys) != len(items) or item_keys != expected_item_keys:
                raise QualityAuditRunnerError("Surface item keyset drift")
            if any(
                not isinstance(row, Mapping)
                or set(row) != set(dataset_builder.MODEL_PROFILE_FIELDS)
                for row in profiles
            ):
                raise QualityAuditRunnerError("Surface profile exact schema drift")
            profile_sellers = {str(row["seller_uid"]) for row in profiles}
            if len(profile_sellers) != len(profiles) or profile_sellers != seller_uids:
                raise QualityAuditRunnerError("Surface profile seller keyset drift")
            item_count_by_seller: dict[str, int] = {
                seller_uid: 0 for seller_uid in seller_uids
            }
            for item in items:
                item_count_by_seller[str(item["seller_uid"])] += 1
            if any(
                isinstance(row["item_count"], bool)
                or not isinstance(row["item_count"], int)
                or row["item_count"] != item_count_by_seller[str(row["seller_uid"])]
                for row in profiles
            ):
                raise QualityAuditRunnerError("Surface profile item-count drift")
            if surface == "surface_full":
                full_profile_sellers = profile_sellers
        if policy_surface_order != (
            "surface_full",
            "surface_code_masked",
            "surface_code_neutralized",
        ):
            raise QualityAuditRunnerError("Surface order drift")
        pair_uids: set[str] = set()
        endpoint_worlds: set[str] = set()
        endpoint_sellers: set[str] = set()
        for row in split_data["endpoints"]:
            pair_uid = row["canonical_pair_uid"]
            if pair_uid in pair_uids:
                raise QualityAuditRunnerError("Duplicate pair UID within split")
            pair_uids.add(pair_uid)
            endpoint_worlds.add(row["world_uid"])
            endpoint_sellers.update((row["seller_uid_left"], row["seller_uid_right"]))
        public_sellers = {str(row["seller_uid"]) for row in split_data["public_code"]}
        try:
            preparer._parse_public_rows(
                split_data["public_code"],
                ordered_worlds=ordered_world_uids,
                sellers_by_world=endpoint_sellers_by_world,
                expected_sellers_per_world=expected_sellers_per_world,
            )
        except (
            preparer.QualityProbePreparationError,
            preparer.channel.QualityChannelViewError,
        ) as exc:
            raise QualityAuditRunnerError("Public code schema/owner drift") from exc
        eligibility_rows = tuple(split_data["eligibility"])
        endpoint_keys = {
            (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            for row in split_data["endpoints"]
        }
        eligibility_keys = {
            (str(row.get("world_uid")), str(row.get("canonical_pair_uid")))
            for row in eligibility_rows
            if isinstance(row, Mapping)
        }
        if (
            any(
                not isinstance(row, Mapping)
                or set(row) != set(preparer.ELIGIBILITY_FIELDS)
                or type(row["text_probe_eligible"]) is not bool
                for row in eligibility_rows
            )
            or len(eligibility_keys) != len(eligibility_rows)
            or eligibility_keys != endpoint_keys
        ):
            raise QualityAuditRunnerError("Text eligibility exact closure drift")
        eligibility_by_world: dict[str, list[Mapping[str, Any]]] = {
            world_uid: [] for world_uid in ordered_world_uids
        }
        for row in eligibility_rows:
            eligibility_by_world[str(row["world_uid"])].append(row)
        if any(
            len(rows) != expected_pairs_per_world
            or sum(row["text_probe_eligible"] is False for row in rows)
            != expected_excluded_pairs_per_world
            for rows in eligibility_by_world.values()
        ):
            raise QualityAuditRunnerError(
                "Per-world text eligibility cardinality drift"
            )
        structure_rows = tuple(split_data["structure_audit"])
        if (
            any(
                not isinstance(row, Mapping)
                or set(row) != set(structure_aggregator.STRUCTURE_AUDIT_FIELDS)
                for row in structure_rows
            )
            or {str(row["world_uid"]) for row in structure_rows} != worlds
            or len(structure_rows) != len(worlds)
        ):
            raise QualityAuditRunnerError("Structure audit exact schema/world drift")
        if (
            not worlds
            or worlds != item_worlds
            or worlds != endpoint_worlds
            or seller_uids != endpoint_sellers
            or seller_uids != public_sellers
            or seller_uids != full_profile_sellers
        ):
            raise QualityAuditRunnerError("Public UID join universe drift")
        observed = {
            "world": worlds,
            "seller": seller_uids,
            "item": item_uids,
            "pair": pair_uids,
        }
        expected_registries = manifests[split].get("uid_registries")
        if not isinstance(expected_registries, Mapping):
            raise QualityAuditRunnerError("Split UID registry manifest drift")
        for kind, values in observed.items():
            spec = expected_registries.get(kind)
            if (
                not isinstance(spec, Mapping)
                or spec.get("count") != len(values)
                or spec.get("sha256") != _registry_sha256(values)
            ):
                raise QualityAuditRunnerError(f"Split {kind} UID registry drift")
            registries[split][kind] = values
    root_registries = root_manifest.get("uid_registries")
    if not isinstance(root_registries, Mapping):
        raise QualityAuditRunnerError("Root UID registry manifest drift")
    for kind in ("world", "seller", "item", "pair"):
        values_by_split = [registries[split][kind] for split in SPLITS]
        merged = set().union(*values_by_split)
        if len(merged) != sum(len(values) for values in values_by_split):
            raise QualityAuditRunnerError(f"Cross-split {kind} UID intersection")
        spec = root_registries.get(kind)
        if (
            not isinstance(spec, Mapping)
            or spec.get("count") != len(merged)
            or spec.get("sha256") != _registry_sha256(merged)
        ):
            raise QualityAuditRunnerError(f"Root {kind} UID registry drift")


def _validate_builder_seller_authority(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    id_key: str,
    expected_sellers_per_world: int,
) -> None:
    """Bind every persisted seller join key to the frozen builder authority."""

    for split in SPLITS:
        expected_by_world = {
            str(record["world_uid"]): {
                structure.base_uid(
                    key_hex=id_key,
                    entity_kind="seller",
                    parent_uid_or_mode=str(record["world_uid"]),
                    ordinal=slot,
                )
                for slot in range(expected_sellers_per_world)
            }
            for record in records_by_split[split]
        }
        expected_all = set().union(*expected_by_world.values())
        split_data = loaded[split]
        endpoint_by_world: dict[str, set[str]] = {
            world_uid: set() for world_uid in expected_by_world
        }
        for row in split_data["endpoints"]:
            endpoint_by_world[str(row["world_uid"])].update(
                (str(row["seller_uid_left"]), str(row["seller_uid_right"]))
            )
        if endpoint_by_world != expected_by_world:
            raise QualityAuditRunnerError("Persisted seller authority replay drift")
        for surface in SURFACE_FILES:
            items, profiles = split_data["surface_rows"][surface]
            item_by_world: dict[str, set[str]] = {
                world_uid: set() for world_uid in expected_by_world
            }
            for row in items:
                item_by_world[str(row["world_uid"])].add(str(row["seller_uid"]))
            profile_sellers = {str(row["seller_uid"]) for row in profiles}
            if item_by_world != expected_by_world or profile_sellers != expected_all:
                raise QualityAuditRunnerError("Persisted seller authority replay drift")
        public_by_world: dict[str, set[str]] = {
            world_uid: set() for world_uid in expected_by_world
        }
        for row in split_data["public_code"]:
            public_by_world[str(row["world_uid"])].add(str(row["seller_uid"]))
        if public_by_world != expected_by_world:
            raise QualityAuditRunnerError("Persisted seller authority replay drift")


def _validate_loaded_structure_bindings(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    expected_clone_count_per_world: int,
) -> None:
    """Recompute structure receipts from the six model-view files actually loaded."""

    hash_fields = {
        "surface_full": ("full_item_sha256", "full_profile_sha256"),
        "surface_code_masked": ("masked_item_sha256", "masked_profile_sha256"),
        "surface_code_neutralized": (
            "neutral_item_sha256",
            "neutral_profile_sha256",
        ),
    }
    for split in SPLITS:
        split_data = loaded[split]
        structure_by_world = {
            str(row["world_uid"]): row for row in split_data["structure_audit"]
        }
        full_items = split_data["surface_rows"]["surface_full"][0]
        seller_world = {
            str(row["seller_uid"]): str(row["world_uid"]) for row in full_items
        }
        items_by_surface_world: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
        profiles_by_surface_world: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
        for surface in SURFACE_FILES:
            items, profiles = split_data["surface_rows"][surface]
            item_groups = {world_uid: [] for world_uid in structure_by_world}
            profile_groups = {world_uid: [] for world_uid in structure_by_world}
            for row in items:
                item_groups[str(row["world_uid"])].append(row)
            for row in profiles:
                profile_groups[seller_world[str(row["seller_uid"])]].append(row)
            items_by_surface_world[surface] = item_groups
            profiles_by_surface_world[surface] = profile_groups
        public_by_world = {world_uid: [] for world_uid in structure_by_world}
        for row in split_data["public_code"]:
            public_by_world[str(row["world_uid"])].append(row)
        endpoints_by_world = {world_uid: [] for world_uid in structure_by_world}
        for row in split_data["endpoints"]:
            endpoints_by_world[str(row["world_uid"])].append(row)
        eligibility = {
            (str(row["world_uid"]), str(row["canonical_pair_uid"])): bool(
                row["text_probe_eligible"]
            )
            for row in split_data["eligibility"]
        }
        for world_uid, audit in structure_by_world.items():
            full_world_items = items_by_surface_world["surface_full"][world_uid]
            item_index = {
                str(row["item_uid"]): row for row in full_world_items
            }
            if len(item_index) != len(full_world_items):
                raise QualityAuditRunnerError("Structure item universe drift")
            for surface, (item_hash_field, profile_hash_field) in hash_fields.items():
                if (
                    common.canonical_sha256(items_by_surface_world[surface][world_uid])
                    != audit[item_hash_field]
                    or common.canonical_sha256(
                        profiles_by_surface_world[surface][world_uid]
                    )
                    != audit[profile_hash_field]
                ):
                    raise QualityAuditRunnerError(
                        "Persisted model-view structure hash drift"
                    )
            public_rows = public_by_world[world_uid]
            codes = {
                str(code) for row in public_rows for code in row["owned_codes"]
            }
            item_occurrences = sum(
                len(row["item_occurrences"]) for row in public_rows
            )
            visible_occurrences = item_occurrences + sum(
                len(row["profile_occurrences"]) for row in public_rows
            )
            if (
                audit["item_count"] != len(full_world_items)
                or audit["registered_code_count"] != len(codes)
                or audit["registered_item_occurrence_count"] != item_occurrences
                or audit["registered_visible_occurrence_expected_count"]
                != visible_occurrences
                or audit["registered_visible_occurrence_actual_count"]
                != visible_occurrences
            ):
                raise QualityAuditRunnerError("Persisted structure count drift")
            clones = audit["clone_directions"]
            if len(clones) != expected_clone_count_per_world:
                raise QualityAuditRunnerError("Exact-title clone count drift")
            pair_uid_by_sellers = {
                frozenset((str(row["seller_uid_left"]), str(row["seller_uid_right"]))): str(
                    row["canonical_pair_uid"]
                )
                for row in endpoints_by_world[world_uid]
            }
            for clone in clones:
                source_uid = str(clone["source_item_uid"])
                target_uid = str(clone["target_item_uid"])
                if source_uid not in item_index or target_uid not in item_index:
                    raise QualityAuditRunnerError("Clone item authority drift")
                source_seller = str(item_index[source_uid]["seller_uid"])
                target_seller = str(item_index[target_uid]["seller_uid"])
                pair_uid = pair_uid_by_sellers.get(
                    frozenset((source_seller, target_seller))
                )
                if (
                    source_seller == target_seller
                    or pair_uid is None
                    or eligibility[(world_uid, pair_uid)] is not False
                    or any(
                        next(
                            row for row in items_by_surface_world[surface][world_uid]
                            if str(row["item_uid"]) == source_uid
                        )["title"]
                        != next(
                            row for row in items_by_surface_world[surface][world_uid]
                            if str(row["item_uid"]) == target_uid
                        )["title"]
                        for surface in SURFACE_FILES
                    )
                ):
                    raise QualityAuditRunnerError("Clone/view/eligibility closure drift")
            neutral = audit["neutral_receipt"]
            expected_item_uids = set(item_index)
            if (
                {
                    str(row["item_uid"])
                    for row in neutral["per_item_template_mapping"]
                }
                != expected_item_uids
                or {
                    str(row["item_uid"])
                    for row in neutral["non_code_projection_nodes"]
                }
                != expected_item_uids
            ):
                raise QualityAuditRunnerError("Neutral receipt item authority drift")


def _root_pin_from_policy(policy: Mapping[str, Any]) -> tuple[Path, truth_capability.RootManifestPin]:
    spec = policy.get("pins", {}).get("design_root_manifest")
    if not isinstance(spec, Mapping) or set(spec) != {
        "path",
        "size_bytes",
        "sha256",
        "canonical_self_hash",
    }:
        raise AuditorExecutionFailure("Design root manifest is not pinned")
    path = (ROOT / str(spec["path"])).resolve()
    if ROOT not in path.parents or path.name != "root_manifest.json":
        raise AuditorExecutionFailure("Design root manifest path drift")
    return path.parent, truth_capability.RootManifestPin(
        path="root_manifest.json",
        size_bytes=int(spec["size_bytes"]),
        sha256=str(spec["sha256"]),
        canonical_self_hash=str(spec["canonical_self_hash"]),
    )


def _run_authorized_formal_quality_audit(
    *, policy: Mapping[str, Any], state: dict[str, str]
) -> dict[str, Any]:
    """Execute the authorized body; the public wrapper classifies failures."""

    state["stage"] = "root_manifest_and_physical_universe"
    dataset_root, root_pin = _root_pin_from_policy(policy)
    root_manifest, manifests = _load_root_manifests(
        dataset_root=dataset_root, root_pin=root_pin
    )
    if (
        root_manifest.get("execution_mode") != "design_preflight"
        or root_manifest.get("scientific_use_forbidden") is not True
        or root_manifest.get("formal_rows_created") != 0
        or root_manifest.get("training_started") is not False
    ):
        raise QualityAuditRunnerError("Design-only root claim boundary drift")
    state["stage"] = "label_free_split_loading"
    loaded = {
        split: _load_split_label_free(
            dataset_root=dataset_root, split=split, manifest=manifests[split]
        )
        for split in SPLITS
    }
    state["stage"] = "four_split_uid_endpoint_and_view_closure"
    _validate_public_uid_registries(
        root_manifest=root_manifest, manifests=manifests, loaded=loaded
    )
    state["stage"] = "builder_authority_replay"
    builder_policy = scientific.load_policy()
    context = scientific.build_execution_context(
        builder_policy, execution_mode="design_preflight"
    )
    if context.output_root.resolve() != dataset_root:
        raise QualityAuditRunnerError("Builder authority/output root binding drift")
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in context.world_records:
        records_by_split[str(record["split"])].append(dict(record))
    for split in SPLITS:
        records_by_split[split].sort(key=lambda row: int(row["split_ordinal"]))
        observed_worlds = loaded[split]["worlds"]
        expected_world_projection = [
            {
                "world_uid": str(row["world_uid"]),
                "split_ordinal": int(row["split_ordinal"]),
            }
            for row in records_by_split[split]
        ]
        if list(observed_worlds) != expected_world_projection:
            raise QualityAuditRunnerError("Persisted world authority replay drift")

    builder_policy_source = _repo_source(scientific.DEFAULT_POLICY_PATH)
    code_key = document_capacity.derive_code_key(context.document_variation_key)
    id_key = str(context.effective_policy["randomness"][context.base_mode]["id_key_hex"])
    _validate_builder_seller_authority(
        loaded=loaded,
        records_by_split=records_by_split,
        id_key=id_key,
        expected_sellers_per_world=policy["design_scale"][
            "seller_count_per_world"
        ],
    )
    # Close the persisted nested receipt schema and every zero-tolerance
    # counter before dereferencing receipt internals in the six-view replay.
    # This keeps malformed persisted structure bytes classified as a dataset
    # gate rather than leaking a KeyError into the auditor-failure branch.
    state["stage"] = "label_free_structure_schema_and_zero_gates"
    structure_receipt = structure_aggregator.aggregate_formal_structure(
        public_rows_by_split={split: loaded[split]["public_code"] for split in SPLITS},
        structure_rows_by_split={
            split: loaded[split]["structure_audit"] for split in SPLITS
        },
        policy=policy,
    )
    if structure_receipt["status"] != "PASS":
        receipt: dict[str, Any] = {
            "version": VERSION,
            "status": "DATASET_INVALIDATED",
            "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
            "structure": structure_receipt,
            "supervised_truth_opened": False,
            "audit_a_b_truth_open_count": 0,
            "formal_500_by_4_generated": False,
            "training_started": False,
        }
        receipt["canonical_self_hash"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        return receipt

    state["stage"] = "loaded_model_view_structure_binding"
    _validate_loaded_structure_bindings(
        loaded=loaded,
        expected_clone_count_per_world=builder_policy[
            "exact_title_clone_endpoint_qualification"
        ]["expected_exact_title_clone_count_per_world"],
    )
    state["stage"] = "label_free_feature_freeze"
    text_matrices: dict[str, tuple[preparer.FrozenFeatureMatrix, ...]] = {}
    code_matrices: dict[str, tuple[preparer.FrozenFeatureMatrix, ...]] = {}
    eligibilities: dict[str, preparer.FrozenTextEligibility] = {}
    for split in ("train", "development"):
        endpoints = loaded[split]["endpoints"]
        ordered_world_uids = tuple(str(row["world_uid"]) for row in loaded[split]["worlds"])
        source_map = loaded[split]["sources"]
        surface_values: list[preparer.FrozenFeatureMatrix] = []
        for surface in policy["model_views"]["order"]:
            item_path, profile_path = SURFACE_FILES[surface]
            items, profiles = loaded[split]["surface_rows"][surface]
            surface_values.extend(
                preparer.prepare_text_surface_matrices(
                    surface=surface,
                    items=items,
                    profiles=profiles,
                    endpoints=endpoints,
                    ordered_world_uids=ordered_world_uids,
                    sources=_source_tuple(
                        source_map[ENDPOINT_PATH],
                        source_map[item_path],
                        source_map[profile_path],
                    ),
                )
            )
        text_matrices[split] = tuple(surface_values)
        eligibilities[split] = preparer.freeze_text_eligibility(
            eligibility_rows=loaded[split]["eligibility"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            sources=_source_tuple(
                source_map[ELIGIBILITY_PATH], source_map[ENDPOINT_PATH]
            ),
        )
        public_matrix = preparer.prepare_public_code_matrix(
            public_rows=loaded[split]["public_code"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            sources=_source_tuple(
                source_map[PUBLIC_CODE_PATH], source_map[ENDPOINT_PATH]
            ),
        )
        expected_ordinals = {
            str(row["world_uid"]): int(row["mode_global_ordinal"])
            for row in records_by_split[split]
        }
        expected_seller_slots = {
            (
                world_uid,
                structure.base_uid(
                    key_hex=id_key,
                    entity_kind="seller",
                    parent_uid_or_mode=world_uid,
                    ordinal=slot,
                ),
            ): slot
            for world_uid in ordered_world_uids
            for slot in range(policy["design_scale"]["seller_count_per_world"])
        }
        decoded_matrix = preparer.prepare_decoded_slot_matrix(
            public_rows=loaded[split]["public_code"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            expected_mode_global_ordinal_by_world=expected_ordinals,
            expected_seller_slot_by_world_and_seller=expected_seller_slots,
            decode_coordinate=lambda _world_uid, code: document_capacity.decode_code(
                code_key=code_key, code=code
            ),
            sources=_source_tuple(
                builder_policy_source,
                source_map[PUBLIC_CODE_PATH],
                source_map[ENDPOINT_PATH],
                source_map[WORLDS_PATH],
            ),
        )
        code_matrices[split] = (public_matrix, decoded_matrix)

    state["stage"] = "train_development_truth_and_supervised_gates"
    supervised_receipt = validator.evaluate_formal_probe_families(
        text_train_matrices=text_matrices["train"],
        text_development_matrices=text_matrices["development"],
        code_train_matrices=code_matrices["train"],
        code_development_matrices=code_matrices["development"],
        dataset_root=dataset_root,
        root_manifest_pin=root_pin,
        policy=policy,
        train_text_eligibility=eligibilities["train"],
        development_text_eligibility=eligibilities["development"],
    )
    status = (
        "PASS"
        if structure_receipt["status"] == "PASS"
        and supervised_receipt["status"] == "PASS"
        else "DATASET_INVALIDATED"
    )
    state["stage"] = "aggregate_success_receipt"
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "root_manifest": {
            "size_bytes": root_pin.size_bytes,
            "sha256": root_pin.sha256,
            "canonical_self_hash": root_pin.canonical_self_hash,
        },
        "structure": structure_receipt,
        "supervised": supervised_receipt,
        "audit_a_b_truth_remained_sealed": True,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    receipt["canonical_self_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def _classified_failure_receipt(
    *, status: str, stage: str, exc: BaseException
) -> dict[str, Any]:
    message_sha256 = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "failure_stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": message_sha256,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "input_dataset_retained_at_decision": True,
        "cleanup_required": status == "DATASET_INVALIDATED",
        "cleanup_completed": False,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    receipt["canonical_self_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def run_formal_quality_audit(
    *, policy_path: Path = quality_policy_module.DEFAULT_POLICY
) -> dict[str, Any]:
    """Run the complete transaction or return a classified, hash-only failure."""

    policy = quality_policy_module.load_policy(policy_path)
    authorization = policy["authorization"]
    if (
        authorization["quality_audit_run"] is not True
        or authorization["metric_generation"] is not True
    ):
        raise QualityAuditRunnerError("Formal quality audit remains unauthorized")
    state = {"stage": "authorized_entry"}
    try:
        return _run_authorized_formal_quality_audit(policy=policy, state=state)
    except AuditorExecutionFailure as exc:
        return _classified_failure_receipt(
            status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            stage=state["stage"],
            exc=exc,
        )
    except truth_capability.QualityTruthAuditorExecutionError as exc:
        return _classified_failure_receipt(
            status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            stage=state["stage"],
            exc=exc,
        )
    except (
        DatasetGateFailure,
        QualityAuditRunnerError,
        preparer.QualityProbePreparationError,
        structure_aggregator.QualityStructureAggregationError,
        validator.QualityProbeDatasetGateError,
        truth_capability.QualityTruthDatasetGateError,
        preparer.channel.QualityChannelViewError,
        preparer.text_views.QualityTextProbeViewError,
    ) as exc:
        return _classified_failure_receipt(
            status="DATASET_INVALIDATED", stage=state["stage"], exc=exc
        )
    except Exception as exc:
        return _classified_failure_receipt(
            status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            stage=state["stage"],
            exc=exc,
        )


def main() -> None:
    result = run_formal_quality_audit()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
