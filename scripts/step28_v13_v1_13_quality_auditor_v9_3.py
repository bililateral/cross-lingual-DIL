#!/usr/bin/env python3
"""Run the frozen V9.3 1,004-world shortcut-quality audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

import step28_v13_common as common
import step28_v13_v1_13_method_dataset_builder_v9_3 as dataset_builder
import step28_v13_v1_13_method_policy_v9_3 as method_policy_module
import step28_v13_v1_13_method_world_v9_3 as method_world
import step28_v13_v1_13_prebuild_structure_gate_v9_3_r2 as prebuild_gate
import step28_v13_v1_13_quality_probe_core_v9_3 as probe_core
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_structure_matrix_v9_3 as structure_matrix


VERSION = (
    "2026-08-27-step28-v13-v1-13-quality-auditor-"
    "v9-3-r2-user-accepted-residual-22"
)
DEFAULT_ROOT = dataset_builder.DEFAULT_OUTPUT
DEFAULT_OUTPUT = common.repo_path(
    "reports/step28_v13_v1_13_quality_audit/"
    "v9_3_r2_method_qualification_20260827/complete_quality_evidence.json"
)
ARTIFICIAL_CODE = re.compile(r"Q[A-P]{10}")
PRIVATE_COORDINATE_UID = re.compile(
    r"(?:w|ctl|sel|ias|itm|slt|nsl)_[0-9a-f]{64}"
)
MODEL_ITEM_FIELDS = ("world_uid", "seller_uid", "item_uid", "title", "description")
SPLITS = tuple(dataset_builder.WORLD_COUNTS)
DEFERRED_TRUTH_PATHS = {
    f"{split}/private/controller_membership.jsonl" for split in SPLITS
}


class QualityAuditorV93Error(common.ContractError):
    """Raised for an auditor execution failure, not a data-quality failure."""


@dataclass
class SplitData:
    split: str
    worlds: list[dict[str, Any]]
    sellers: list[dict[str, Any]]
    endpoints: list[dict[str, str]]
    original_items: list[dict[str, Any]]
    original_profiles: list[dict[str, Any]]
    deranged_items: list[dict[str, Any]]
    deranged_profiles: list[dict[str, Any]]
    identity33: list[dict[str, str]]
    overrides: list[dict[str, Any]]
    seller_structure: list[dict[str, Any]]
    noise_structure: list[dict[str, Any]]
    world_audits: list[dict[str, Any]]


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = deepcopy(dict(payload))
    value["canonical_self_sha256"] = None
    return common.canonical_sha256(value)


def _prebuild_gate_self_hash(payload: Mapping[str, Any]) -> str:
    """Reproduce the structure gate's remove-field self-hash convention."""

    value = deepcopy(dict(payload))
    value.pop("canonical_self_sha256", None)
    return common.canonical_sha256(value)


def _load_jsonl(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise QualityAuditorV93Error(
                    f"JSONL line boundary drift: {path.name}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise QualityAuditorV93Error(f"JSONL object drift: {path.name}")
            rows.append(value)
    if len(rows) != expected_rows:
        raise QualityAuditorV93Error(f"JSONL row count drift: {path.name}")
    return rows


def _load_csv(path: Path, *, expected_rows: int) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or len(rows) != expected_rows or any(tuple(row) != fields for row in rows):
        raise QualityAuditorV93Error(f"CSV closure drift: {path.name}")
    return fields, rows


def _manifest_records(root_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = root_manifest.get("files")
    if not isinstance(rows, list):
        raise QualityAuditorV93Error("Root file registry is absent")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise QualityAuditorV93Error("Root file registry schema drift")
        path = str(row["path"])
        if path in output:
            raise QualityAuditorV93Error("Root file registry path collision")
        output[path] = dict(row)
    return output


def _verify_root(
    root: Path, policy: Mapping[str, Any]
) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]
]:
    manifest_path = root / "root_manifest.json"
    if root.resolve() != DEFAULT_ROOT.resolve() or not manifest_path.is_file():
        raise QualityAuditorV93Error("Frozen V9.3 method root is absent or misplaced")
    manifest = common.load_json(manifest_path)
    if (
        manifest.get("version") != dataset_builder.VERSION
        or manifest.get("status") != "BUILT_NOT_QUALITY_AUDITED_NOT_TRAINING_QUALIFIED"
        or manifest.get("canonical_self_sha256") != _self_hash(manifest)
        or manifest.get("method_policy_canonical_self_sha256")
        != policy["canonical_self_sha256"]
        or manifest.get("world_counts") != dataset_builder.WORLD_COUNTS
        or manifest.get("pair_truth_materialized") is not False
        or manifest.get("audit_truth_read_count") != 0
    ):
        raise QualityAuditorV93Error("Root manifest scientific boundary drift")
    gate_result_path = common.repo_path(
        str(policy["prebuild_structure_gate_contract"]["result_path"])
    )
    if not gate_result_path.is_file():
        raise QualityAuditorV93Error("Prebuild structure-gate result is absent")
    gate_result = common.load_json(gate_result_path)
    gate_summary = manifest.get("prebuild_structure_gate")
    if (
        not isinstance(gate_summary, Mapping)
        or gate_result.get("canonical_self_sha256")
        != _prebuild_gate_self_hash(gate_result)
        or gate_result.get("scientific_pass") is not True
        or gate_result.get("status")
        != policy["prebuild_structure_gate_contract"]["required_pass_status"]
        or gate_summary.get("version") != gate_result.get("version")
        or gate_summary.get("status") != gate_result.get("status")
        or gate_summary.get("scientific_pass") is not True
        or gate_summary.get("file_sha256") != common.sha256_file(gate_result_path)
        or gate_summary.get("canonical_self_sha256")
        != gate_result.get("canonical_self_sha256")
        or gate_summary.get("hard_gates") != gate_result.get("hard_gates")
        or gate_summary.get("probe_contract_audit")
        != gate_result.get("probe_contract_audit")
        or gate_summary.get("finite_preregistered_projection_map_sha256")
        != policy["prebuild_structure_gate_contract"][
            "finite_preregistered_projection_map_sha256"
        ]
    ):
        raise QualityAuditorV93Error(
            "Method root/prebuild structure-gate binding drift"
        )
    records = _manifest_records(manifest)
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "root_manifest.json"
    }
    if set(observed) != set(records):
        raise QualityAuditorV93Error("Root file universe drift")
    for relative, path in observed.items():
        record = records[relative]
        if path.stat().st_size != record.get("size_bytes"):
            raise QualityAuditorV93Error(f"Root file size drift: {relative}")
        if relative not in DEFERRED_TRUTH_PATHS and common.sha256_file(path) != record.get("sha256"):
            raise QualityAuditorV93Error(f"Root file hash drift: {relative}")
    for split in SPLITS:
        split_path = root / split / "split_manifest.json"
        split_manifest = common.load_json(split_path)
        if (
            split_manifest.get("canonical_self_sha256") != _self_hash(split_manifest)
            or split_manifest.get("split") != split
            or split_manifest.get("pair_truth_materialized") is not False
            or split_manifest.get("canonical_self_sha256")
            != manifest["split_manifest_self_hashes"][split]
        ):
            raise QualityAuditorV93Error(f"Split manifest drift: {split}")
    return manifest, records, gate_result


def _record(records: Mapping[str, Mapping[str, Any]], relative: str) -> Mapping[str, Any]:
    value = records.get(relative)
    if value is None:
        raise QualityAuditorV93Error(f"Required method-root file is absent: {relative}")
    return value


def _load_split(root: Path, records: Mapping[str, Mapping[str, Any]], split: str) -> SplitData:
    prefix = f"{split}/"

    def jsonl(relative: str) -> list[dict[str, Any]]:
        full = prefix + relative
        record = _record(records, full)
        return _load_jsonl(root / full, expected_rows=int(record["row_count"]))

    def csv_rows(relative: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
        full = prefix + relative
        record = _record(records, full)
        return _load_csv(root / full, expected_rows=int(record["row_count"]))

    worlds = jsonl("observed/worlds.jsonl")
    sellers = jsonl("observed/sellers.jsonl")
    endpoint_fields, endpoints_raw = csv_rows("observed/complete_pair_endpoints.csv")
    if endpoint_fields != tuple(method_world.PAIR_KEY_FIELDS):
        raise QualityAuditorV93Error(f"Endpoint schema drift: {split}")
    endpoints = [dict(row) for row in endpoints_raw]
    identity_fields, identity33 = csv_rows("observed/identity33_all_pairs.csv")
    if (
        identity_fields[:2] != tuple(method_world.PAIR_KEY_FIELDS[:2])
        or len(identity_fields) != 35
        or [
            (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            for row in identity33
        ]
        != _row_keys(endpoints)
    ):
        raise QualityAuditorV93Error(f"Identity33 public schema/order drift: {split}")
    seller_fields, seller_raw = csv_rows("private/seller_slot_structure.csv")
    noise_fields, noise_raw = csv_rows("private/noise_visible_structure.csv")
    expected_seller_fields = (*method_world.PAIR_KEY_FIELDS, *structure_matrix.SELLER_SLOT_RAW_FIELDS)
    expected_noise_fields = (*method_world.PAIR_KEY_FIELDS, *structure_matrix.NOISE_VISIBLE_RAW_FIELDS)
    if seller_fields != expected_seller_fields or noise_fields != expected_noise_fields:
        raise QualityAuditorV93Error(f"Structure raw schema drift: {split}")

    def numeric_rows(rows: Sequence[Mapping[str, str]], raw_fields: Sequence[str]) -> list[dict[str, Any]]:
        return [
            {
                **{name: str(row[name]) for name in method_world.PAIR_KEY_FIELDS},
                **{name: int(row[name]) for name in raw_fields},
            }
            for row in rows
        ]

    expected_worlds = dataset_builder.WORLD_COUNTS[split]
    if (
        len(worlds) != expected_worlds
        or len(sellers) != expected_worlds * 28
        or len(endpoints) != expected_worlds * 378
        or len({row["world_uid"] for row in worlds}) != expected_worlds
        or Counter(str(row["split"]) for row in worlds) != Counter({split: expected_worlds})
    ):
        raise QualityAuditorV93Error(f"Split public cardinality drift: {split}")
    return SplitData(
        split=split,
        worlds=worlds,
        sellers=sellers,
        endpoints=endpoints,
        original_items=jsonl("observed/original_redacted_items.jsonl"),
        original_profiles=jsonl("observed/original_model_seller_profiles.jsonl"),
        deranged_items=jsonl("observed/deranged_redacted_items.jsonl"),
        deranged_profiles=jsonl("observed/deranged_model_seller_profiles.jsonl"),
        identity33=identity33,
        overrides=jsonl("private/override_audit.jsonl"),
        seller_structure=numeric_rows(seller_raw, structure_matrix.SELLER_SLOT_RAW_FIELDS),
        noise_structure=numeric_rows(noise_raw, structure_matrix.NOISE_VISIBLE_RAW_FIELDS),
        world_audits=jsonl("private/world_generation_audit.jsonl"),
    )


def _row_keys(endpoints: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    keys = tuple(
        (str(row["world_uid"]), str(row["canonical_pair_uid"])) for row in endpoints
    )
    if len(keys) != len(set(keys)):
        raise QualityAuditorV93Error("Endpoint row-key collision")
    return keys


def _freeze_structure(data: SplitData) -> dict[str, probe_core.FrozenMatrix]:
    endpoint_keys = _row_keys(data.endpoints)
    output: dict[str, probe_core.FrozenMatrix] = {}
    for view, rows, raw_fields, builder, names in (
        (
            "seller_slot",
            data.seller_structure,
            structure_matrix.SELLER_SLOT_RAW_FIELDS,
            structure_matrix.seller_matrix,
            structure_matrix.seller_matrix_feature_names(),
        ),
        (
            "noise_visible",
            data.noise_structure,
            structure_matrix.NOISE_VISIBLE_RAW_FIELDS,
            structure_matrix.noise_matrix,
            structure_matrix.noise_matrix_feature_names(),
        ),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["world_uid"])].append(row)
        matrices: list[np.ndarray] = []
        observed_keys: list[tuple[str, str]] = []
        for world in data.worlds:
            world_uid = str(world["world_uid"])
            world_rows = grouped[world_uid]
            structure_matrix.validate_world_rows(world_rows, raw_fields=raw_fields, label=view)
            matrices.append(builder(world_rows))
            observed_keys.extend(
                (world_uid, str(row["canonical_pair_uid"])) for row in world_rows
            )
        if tuple(observed_keys) != endpoint_keys:
            raise QualityAuditorV93Error(f"Structure/endpoint row-order drift: {view}")
        output[view] = probe_core.freeze_matrix(
            view=view,
            values=np.concatenate(matrices, axis=0),
            row_keys=endpoint_keys,
            column_names=names,
            take_ownership=True,
        )
    return output


def _freeze_text(data: SplitData, *, surface: str) -> dict[str, probe_core.FrozenMatrix]:
    if surface == "original_author":
        items, profiles = data.original_items, data.original_profiles
    elif surface == "style_deranged":
        items, profiles = data.deranged_items, data.deranged_profiles
    else:
        raise QualityAuditorV93Error("Unknown text surface")
    matrices, names = text_views.build_text_probe_views(
        items=items, profiles=profiles, endpoints=data.endpoints
    )
    excluded = {str(row["canonical_pair_uid"]) for row in data.overrides}
    if len(excluded) != len(data.worlds) * 6:
        raise QualityAuditorV93Error(f"Registered text exclusion drift: {data.split}")
    mask = np.fromiter(
        (str(row["canonical_pair_uid"]) not in excluded for row in data.endpoints),
        dtype=np.bool_,
        count=len(data.endpoints),
    )
    if int(mask.sum()) != len(data.worlds) * 372:
        raise QualityAuditorV93Error(f"Text eligibility cardinality drift: {data.split}")
    keys = tuple(key for key, keep in zip(_row_keys(data.endpoints), mask, strict=True) if keep)
    return {
        view: probe_core.freeze_matrix(
            view=view,
            values=matrix[mask],
            row_keys=keys,
            column_names=names[view],
            take_ownership=True,
        )
        for view, matrix in matrices.items()
    }


def _scan_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for child in value.values() for text in _scan_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _scan_strings(child)]
    return []


def _validate_private_coordinate_nonintervention(
    splits: Mapping[str, SplitData],
    *,
    forbidden_fields: Sequence[str],
) -> bool:
    """Reject private coordinates or private UIDs on every model-visible surface."""

    forbidden = set(forbidden_fields)
    if not forbidden or any(not isinstance(name, str) or not name for name in forbidden):
        raise QualityAuditorV93Error("Private-coordinate field contract drift")

    def validate_value(value: Any, *, row_links: set[str]) -> None:
        if isinstance(value, Mapping):
            leaked = forbidden.intersection(str(name) for name in value)
            if leaked:
                raise QualityAuditorV93Error(
                    "Private coordinate reached a model-visible field: "
                    + ",".join(sorted(leaked))
                )
            for child in value.values():
                validate_value(child, row_links=row_links)
        elif isinstance(value, list):
            for child in value:
                validate_value(child, row_links=row_links)
        elif isinstance(value, str):
            if PRIVATE_COORDINATE_UID.search(value) or any(
                link and link in value for link in row_links
            ):
                raise QualityAuditorV93Error(
                    "Private UID reached a model-visible feature value"
                )

    for data in splits.values():
        world_uids = {str(row["world_uid"]) for row in data.worlds}
        seller_to_world = {
            str(row["seller_uid"]): str(row["world_uid"])
            for row in data.sellers
        }
        if (
            len(world_uids) != len(data.worlds)
            or len(seller_to_world) != len(data.sellers)
            or set(seller_to_world.values()) != world_uids
        ):
            raise QualityAuditorV93Error(
                "Private-coordinate public row-link universe drift"
            )
        if any(
            type(world.get("candidate_index")) is not int
            or world["candidate_index"] != 0
            for world in data.worlds
        ):
            raise QualityAuditorV93Error(
                "Private candidate coordinate changed across the method root"
            )
        original_items = {
            str(row["item_uid"]): row for row in data.original_items
        }
        deranged_items = {
            str(row["item_uid"]): row for row in data.deranged_items
        }
        original_profiles = {
            str(row["seller_uid"]): row for row in data.original_profiles
        }
        deranged_profiles = {
            str(row["seller_uid"]): row for row in data.deranged_profiles
        }
        if (
            any(set(row) != set(MODEL_ITEM_FIELDS) for row in data.original_items)
            or any(set(row) != set(MODEL_ITEM_FIELDS) for row in data.deranged_items)
            or any(
                set(row) != set(scientific.MODEL_PROFILE_FIELDS)
                for row in data.original_profiles
            )
            or any(
                set(row) != set(scientific.MODEL_PROFILE_FIELDS)
                for row in data.deranged_profiles
            )
            or any(
                tuple(row)[:2] != tuple(method_world.PAIR_KEY_FIELDS[:2])
                or len(row) != 35
                for row in data.identity33
            )
            or len(original_items) != len(data.original_items)
            or len(deranged_items) != len(data.deranged_items)
            or set(original_items) != set(deranged_items)
            or len(original_profiles) != len(data.original_profiles)
            or len(deranged_profiles) != len(data.deranged_profiles)
            or set(original_profiles) != set(deranged_profiles)
            or set(original_profiles) != set(seller_to_world)
        ):
            raise QualityAuditorV93Error(
                "Private-coordinate model schema or counterfactual row alignment drift"
            )
        for item_uid in original_items:
            left = original_items[item_uid]
            right = deranged_items[item_uid]
            if any(
                str(left[name]) != str(right[name])
                for name in ("world_uid", "seller_uid", "item_uid")
            ) or seller_to_world.get(str(left["seller_uid"])) != str(
                left["world_uid"]
            ):
                raise QualityAuditorV93Error(
                    "Private row links changed across the style counterfactual"
                )
            row_links = {
                str(left["world_uid"]),
                str(left["seller_uid"]),
                str(left["item_uid"]),
            }
            for row in (left, right):
                feature_projection = {
                    name: value
                    for name, value in row.items()
                    if name not in {"world_uid", "seller_uid", "item_uid"}
                }
                validate_value(feature_projection, row_links=row_links)
        for seller_uid in original_profiles:
            left = original_profiles[seller_uid]
            right = deranged_profiles[seller_uid]
            if str(left["seller_uid"]) != str(right["seller_uid"]):
                raise QualityAuditorV93Error(
                    "Private seller link changed across the style counterfactual"
                )
            row_links = {str(left["seller_uid"])}
            for row in (left, right):
                feature_projection = {
                    name: value
                    for name, value in row.items()
                    if name not in {"world_uid", "seller_uid"}
                }
                validate_value(feature_projection, row_links=row_links)
        for row in data.identity33:
            if str(row["world_uid"]) not in world_uids:
                raise QualityAuditorV93Error(
                    "Identity33 world link is outside the public world universe"
                )
            feature_projection = {
                name: value
                for name, value in row.items()
                if name not in {"canonical_pair_uid", "world_uid"}
            }
            validate_value(
                feature_projection,
                row_links={
                    str(row["canonical_pair_uid"]),
                    str(row["world_uid"]),
                },
            )
    return True


def _validate_label_free_surfaces(splits: Mapping[str, SplitData]) -> dict[str, bool]:
    seller_uids: set[str] = set()
    item_uids: set[str] = set()
    world_uids: set[str] = set()
    for data in splits.values():
        clone_pairs = [row for row in data.overrides if row["override_kind"] == "exact_title_clone"]
        if len(clone_pairs) != len(data.worlds) * 2:
            raise QualityAuditorV93Error("Exact-title clone dose drift")
        for world in data.worlds:
            uid = str(world["world_uid"])
            if uid in world_uids:
                raise QualityAuditorV93Error("Cross-split world UID collision")
            world_uids.add(uid)
        for seller in data.sellers:
            uid = str(seller["seller_uid"])
            if uid in seller_uids:
                raise QualityAuditorV93Error("Cross-split seller UID collision")
            seller_uids.add(uid)
        for surface_items, surface_profiles in (
            (data.original_items, data.original_profiles),
            (data.deranged_items, data.deranged_profiles),
        ):
            title_owners: dict[str, set[str]] = defaultdict(set)
            description_owners: dict[str, set[str]] = defaultdict(set)
            item_documents: set[tuple[str, str]] = set()
            seller_documents: set[tuple[str, ...]] = set()
            registered_clone_titles: set[tuple[str, frozenset[str]]] = set()
            items_by_uid = {str(row["item_uid"]): row for row in surface_items}
            for row in clone_pairs:
                left = str(row["item_uid_left"])
                right = str(row["item_uid_right"])
                title = str(items_by_uid[left]["title"])
                if not title or title != str(items_by_uid[right]["title"]):
                    raise QualityAuditorV93Error("Registered clone title lineage drift")
                registered_clone_titles.add((title, frozenset((left, right))))
            local_item_uids: set[str] = set()
            for row in surface_items:
                uid = str(row["item_uid"])
                local_item_uids.add(uid)
                if ARTIFICIAL_CODE.search(str(row["title"])) or ARTIFICIAL_CODE.search(str(row["description"])):
                    raise QualityAuditorV93Error("Artificial code reached a text surface")
                title = str(row["title"])
                description = str(row["description"])
                if title:
                    title_owners[title].add(uid)
                if description:
                    description_owners[description].add(uid)
                document = (title, description)
                if document in item_documents:
                    raise QualityAuditorV93Error("Cross-item document collision")
                item_documents.add(document)
            if surface_items is data.original_items:
                if item_uids & local_item_uids:
                    raise QualityAuditorV93Error("Cross-split item UID collision")
                item_uids.update(local_item_uids)
            for profile in surface_profiles:
                if any(ARTIFICIAL_CODE.search(text) for text in _scan_strings(profile)):
                    raise QualityAuditorV93Error("Artificial code reached a seller profile")
                document = tuple(
                    str(profile[name])
                    for name in (
                        "category_concat_top",
                        "signature_title_concat",
                        "title_concat_top",
                        "signature_description_concat",
                        "description_concat_top",
                    )
                )
                if document in seller_documents:
                    raise QualityAuditorV93Error("Cross-seller five-field document collision")
                seller_documents.add(document)
            for title, owners in title_owners.items():
                if len(owners) > 1 and (title, frozenset(owners)) not in registered_clone_titles:
                    raise QualityAuditorV93Error("Unregistered nonempty title collision")
            if any(len(owners) > 1 for owners in description_owners.values()):
                raise QualityAuditorV93Error("Nonempty description collision")
        for audit in data.world_audits:
            if (
                audit.get("artificial_code_occurrence_count") != 0
                or audit.get("truth_materialized") is not False
                or audit["counterfactual_intervention"].get(
                    "labels_or_controller_membership_read"
                )
                is not False
                or not audit.get("noise_time_counterfactual_identity33_unchanged")
                or any(
                    int(count) <= 0
                    for count in audit["counterfactual_intervention"][
                        "style_factor_changed_seller_counts"
                    ].values()
                )
            ):
                raise QualityAuditorV93Error("World mechanism audit drift")
    for item_attribute, profile_attribute in (
        ("original_items", "original_profiles"),
        ("deranged_items", "deranged_profiles"),
    ):
        global_titles: dict[str, set[str]] = defaultdict(set)
        global_descriptions: dict[str, set[str]] = defaultdict(set)
        global_documents: set[tuple[str, str]] = set()
        global_seller_documents: set[tuple[str, ...]] = set()
        registered: set[tuple[str, frozenset[str]]] = set()
        for data in splits.values():
            surface_items = getattr(data, item_attribute)
            by_uid = {str(row["item_uid"]): row for row in surface_items}
            for override in data.overrides:
                if override["override_kind"] != "exact_title_clone":
                    continue
                left = str(override["item_uid_left"])
                right = str(override["item_uid_right"])
                registered.add(
                    (str(by_uid[left]["title"]), frozenset((left, right)))
                )
            for row in surface_items:
                uid = str(row["item_uid"])
                title = str(row["title"])
                description = str(row["description"])
                if title:
                    global_titles[title].add(uid)
                if description:
                    global_descriptions[description].add(uid)
                document = (title, description)
                if document in global_documents:
                    raise QualityAuditorV93Error("Cross-split item document collision")
                global_documents.add(document)
            for profile in getattr(data, profile_attribute):
                document = tuple(
                    str(profile[name])
                    for name in (
                        "category_concat_top",
                        "signature_title_concat",
                        "title_concat_top",
                        "signature_description_concat",
                        "description_concat_top",
                    )
                )
                if document in global_seller_documents:
                    raise QualityAuditorV93Error(
                        "Cross-split seller five-field document collision"
                    )
                global_seller_documents.add(document)
        if any(
            len(owners) > 1 and (title, frozenset(owners)) not in registered
            for title, owners in global_titles.items()
        ):
            raise QualityAuditorV93Error("Cross-split unregistered title collision")
        if any(len(owners) > 1 for owners in global_descriptions.values()):
            raise QualityAuditorV93Error("Cross-split description collision")
    private_coordinate_nonintervention = _validate_private_coordinate_nonintervention(
        splits,
        forbidden_fields=method_policy_module.load_policy()[
            "public_model_surfaces"
        ]["forbidden_fields"],
    )
    return {
        "public_and_private_schema_closure": True,
        "artificial_code_zero_occurrence": True,
        "private_coordinate_nonintervention": private_coordinate_nonintervention,
        "document_collision_closure": True,
        "split_isolation": True,
        "identity_asset_and_shared_relation_positive_control": True,
        "style_counterfactual_positive_control": True,
        "noise_counterfactual_identity_invariance": True,
    }


def _verify_and_load_controller_membership(
    *,
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    if split not in {"train", "development"}:
        raise QualityAuditorV93Error("Audit controller truth read is forbidden")
    relative = f"{split}/private/controller_membership.jsonl"
    record = _record(records, relative)
    path = root / relative
    payload = path.read_bytes()
    if (
        len(payload) != int(record["size_bytes"])
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
    ):
        raise QualityAuditorV93Error(f"Deferred controller hash drift: {split}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualityAuditorV93Error(
            f"Deferred controller encoding drift: {split}"
        ) from exc
    lines = text.splitlines(keepends=True)
    if (
        len(lines) != int(record["row_count"])
        or any(not line.endswith("\n") or not line.strip() for line in lines)
    ):
        raise QualityAuditorV93Error(f"Deferred controller line drift: {split}")
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise QualityAuditorV93Error(f"Deferred controller schema drift: {split}")
    return rows


def _materialize_labels(
    *,
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    data: SplitData,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    membership = _verify_and_load_controller_membership(
        root=root, records=records, split=data.split
    )
    members_by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    endpoints_by_world: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in membership:
        members_by_world[str(row["world_uid"])].append(row)
    for row in data.endpoints:
        endpoints_by_world[str(row["world_uid"])].append(row)
    label_by_key: dict[tuple[str, str], int] = {}
    for world in data.worlds:
        world_uid = str(world["world_uid"])
        truth = method_world.materialize_private_truth(
            world_uid=world_uid,
            controller_membership=members_by_world[world_uid],
            endpoints=endpoints_by_world[world_uid],
        )
        labels = truth["pair_labels"]
        if len(labels) != 378 or sum(int(row["label"]) for row in labels) != 20:
            raise QualityAuditorV93Error(f"Per-world truth cardinality drift: {data.split}")
        for row in labels:
            key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            if key in label_by_key:
                raise QualityAuditorV93Error("Materialized truth key collision")
            label_by_key[key] = int(row["label"])
    keys = _row_keys(data.endpoints)
    if set(keys) != set(label_by_key):
        raise QualityAuditorV93Error("Truth/endpoint keyspace drift")
    labels = np.asarray([label_by_key[key] for key in keys], dtype=np.int8)
    override_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in data.overrides
    }
    if any(label_by_key[key] != 0 for key in override_keys):
        raise QualityAuditorV93Error("Registered hard negative is not negative")
    return labels, membership


def _validate_identity_positive_control(
    *,
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    data: SplitData,
    labels: np.ndarray,
    membership: Sequence[Mapping[str, Any]],
) -> None:
    prefix = f"{data.split}/"
    identity_record = _record(records, prefix + "observed/identity33_all_pairs.csv")
    fields, identity_rows = _load_csv(
        root / prefix / "observed/identity33_all_pairs.csv",
        expected_rows=int(identity_record["row_count"]),
    )
    if fields[:2] != tuple(method_world.PAIR_KEY_FIELDS[:2]) or len(fields) != 35:
        raise QualityAuditorV93Error("Identity33 persisted schema drift")
    label_by_key = dict(zip(_row_keys(data.endpoints), labels, strict=True))
    positive_signal_worlds: set[str] = set()
    for row in identity_rows:
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        if label_by_key.get(key) == 1 and any(float(row[name]) != 0.0 for name in fields[2:]):
            positive_signal_worlds.add(key[0])
    if positive_signal_worlds != {str(row["world_uid"]) for row in data.worlds}:
        raise QualityAuditorV93Error("Identity33 lacks a positive signal in a world")

    asset_record = _record(records, prefix + "private/identity_assets.jsonl")
    assets = _load_jsonl(
        root / prefix / "private/identity_assets.jsonl",
        expected_rows=int(asset_record["row_count"]),
    )
    mechanism_record = _record(records, prefix + "private/mechanism_assignments.jsonl")
    mechanisms = _load_jsonl(
        root / prefix / "private/mechanism_assignments.jsonl",
        expected_rows=int(mechanism_record["row_count"]),
    )
    controller_by_seller = {
        str(row["seller_uid"]): str(row["controller_uid"]) for row in membership
    }
    world_by_seller = {
        str(row["seller_uid"]): str(row["world_uid"]) for row in data.sellers
    }
    shared_worlds: set[str] = set()
    for asset in assets:
        sellers = tuple(str(value) for value in asset["sellers"])
        if len(sellers) >= 2:
            controllers = {controller_by_seller[value] for value in sellers}
            worlds = {world_by_seller[value] for value in sellers}
            if len(controllers) == 1 and len(worlds) == 1:
                shared_worlds.update(worlds)
    expected_worlds = {str(row["world_uid"]) for row in data.worlds}
    if (
        shared_worlds != expected_worlds
        or len(mechanisms) != len(data.worlds) * 12
        or Counter(str(row["world_uid"]) for row in mechanisms)
        != Counter({world: 12 for world in expected_worlds})
    ):
        raise QualityAuditorV93Error("Identity asset/mechanism positive control drift")


def _text_labels(data: SplitData, complete_labels: np.ndarray) -> np.ndarray:
    excluded = {str(row["canonical_pair_uid"]) for row in data.overrides}
    mask = np.fromiter(
        (str(row["canonical_pair_uid"]) not in excluded for row in data.endpoints),
        dtype=np.bool_,
        count=len(data.endpoints),
    )
    labels = np.ascontiguousarray(complete_labels[mask], dtype=np.int8)
    if (
        labels.shape != (len(data.worlds) * 372,)
        or int(labels.sum()) != len(data.worlds) * 20
    ):
        raise QualityAuditorV93Error(f"Text truth cardinality drift: {data.split}")
    return labels


def _observation(
    registry: Mapping[str, Mapping[str, Any]],
    identifier: str,
    value: bool | float,
    *,
    passed: bool | None,
) -> dict[str, Any]:
    contract = registry.get(identifier)
    if contract is None:
        raise QualityAuditorV93Error(f"Unregistered quality observation: {identifier}")
    return {**dict(contract), "value": value, "passed": passed}


def _family_observations(
    *,
    registry: Mapping[str, Mapping[str, Any]],
    prefix: str,
    result: Mapping[str, Any],
    baseline: float,
    gates: Mapping[str, Any],
    hard_gate: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for view, value in result["single_feature_maximum_symmetric_roc_auc_by_view"].items():
        identifier = f"{prefix}::{view}::univariate::maximum_symmetric_roc_auc"
        output.append(
            _observation(
                registry,
                identifier,
                float(value),
                passed=(
                    float(value) <= float(gates["maximum_single_feature_symmetric_roc_auc"])
                    if hard_gate
                    else None
                ),
            )
        )
    for qualified, metrics in result["model_results"].items():
        view, model = qualified.split("::", 1)
        auc = float(metrics["symmetric_roc_auc"])
        ap = float(metrics["average_precision"])
        output.append(
            _observation(
                registry,
                f"{prefix}::{view}::{model}::symmetric_roc_auc",
                auc,
                passed=(
                    auc <= float(gates["maximum_family_symmetric_roc_auc"])
                    if hard_gate
                    else None
                ),
            )
        )
        output.append(
            _observation(
                registry,
                f"{prefix}::{view}::{model}::average_precision",
                ap,
                passed=(
                    ap - baseline
                    <= float(gates["maximum_family_average_precision_uplift"])
                    if hard_gate
                    else None
                ),
            )
        )
    if hard_gate:
        family_prefix = f"{prefix}::family"
        output.extend(
            (
                _observation(
                    registry,
                    f"{family_prefix}::maximum_symmetric_roc_auc",
                    float(result["maximum_symmetric_roc_auc"]),
                    passed=float(result["maximum_symmetric_roc_auc"])
                    <= float(gates["maximum_family_symmetric_roc_auc"]),
                ),
                _observation(
                    registry,
                    f"{family_prefix}::maximum_average_precision_uplift",
                    float(result["maximum_average_precision_uplift"]),
                    passed=float(result["maximum_average_precision_uplift"])
                    <= float(gates["maximum_family_average_precision_uplift"]),
                ),
                _observation(
                    registry,
                    f"{family_prefix}::bootstrap_95_upper_symmetric_roc_auc",
                    float(result["bootstrap"]["symmetric_auc_95_upper"]),
                    passed=float(result["bootstrap"]["symmetric_auc_95_upper"])
                    <= float(gates["bootstrap_95_upper_symmetric_roc_auc"]),
                ),
                _observation(
                    registry,
                    f"{family_prefix}::bootstrap_95_upper_average_precision_uplift",
                    float(result["bootstrap"]["average_precision_uplift_95_upper"]),
                    passed=float(result["bootstrap"]["average_precision_uplift_95_upper"])
                    <= float(gates["bootstrap_95_upper_average_precision_uplift"]),
                ),
            )
        )
    return output


def run(*, root: Path, output: Path) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve() or output.exists():
        raise QualityAuditorV93Error("Quality evidence path is wrong or already used")
    policy = method_policy_module.load_policy()
    if (
        policy.get("status")
        != "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED"
    ):
        raise QualityAuditorV93Error("Quality audit received an unfrozen method policy")
    root_manifest, file_records, prebuild_gate_result = _verify_root(
        root.resolve(), policy
    )
    splits = {
        split: _load_split(root.resolve(), file_records, split) for split in SPLITS
    }
    structural_checks = _validate_label_free_surfaces(splits)
    print(json.dumps({"event": "label_free_root_closure_complete"}), flush=True)

    frozen_structure = {split: _freeze_structure(data) for split, data in splits.items()}
    observed_prebuild_structure_commitments = {
        split: {
            view: frozen_structure[split][view].commitment
            for view in ("seller_slot", "noise_visible")
        }
        for split in ("train", "development")
    }
    if (
        observed_prebuild_structure_commitments
        != prebuild_gate_result["matrix_commitments"]
    ):
        raise QualityAuditorV93Error(
            "Method-root structure matrices differ from the prebuild gate"
        )
    print(json.dumps({"event": "structure_matrices_frozen"}), flush=True)
    frozen_text: dict[str, dict[str, dict[str, probe_core.FrozenMatrix]]] = {}
    for split, data in splits.items():
        frozen_text[split] = {}
        for surface in ("style_deranged", "original_author"):
            frozen_text[split][surface] = _freeze_text(data, surface=surface)
            print(
                json.dumps(
                    {"event": "text_matrices_frozen", "split": split, "surface": surface}
                ),
                flush=True,
            )
    matrix_commitments = {
        "structure": {
            split: {view: frozen.commitment for view, frozen in values.items()}
            for split, values in frozen_structure.items()
        },
        "text": {
            split: {
                surface: {view: frozen.commitment for view, frozen in values.items()}
                for surface, values in surfaces.items()
            }
            for split, surfaces in frozen_text.items()
        },
    }
    matrix_commitment_sha256 = common.canonical_sha256(matrix_commitments)
    print(
        json.dumps(
            {
                "event": "all_label_free_matrices_frozen_before_truth",
                "matrix_commitment_sha256": matrix_commitment_sha256,
                "audit_a_truth_read_count": 0,
                "audit_b_truth_read_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    truth_materializations = {
        split: _materialize_labels(
            root=root.resolve(), records=file_records, data=splits[split]
        )
        for split in ("train", "development")
    }
    complete_labels = {
        split: values[0] for split, values in truth_materializations.items()
    }
    observed_prebuild_label_commitments = {
        split: {
            "row_count": len(complete_labels[split]),
            "positive_count": int(complete_labels[split].sum()),
            "raw_i1_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    complete_labels[split], dtype=np.int8
                ).tobytes(order="C")
            ).hexdigest(),
        }
        for split in ("train", "development")
    }
    if (
        observed_prebuild_label_commitments
        != prebuild_gate_result["label_commitments"]
    ):
        raise QualityAuditorV93Error(
            "Method-root labels differ from the prebuild abstract labels"
        )
    for split, (_labels, membership) in truth_materializations.items():
        _validate_identity_positive_control(
            root=root.resolve(),
            records=file_records,
            data=splits[split],
            labels=_labels,
            membership=membership,
        )
    text_label_values = {
        split: _text_labels(splits[split], complete_labels[split])
        for split in ("train", "development")
    }
    print(json.dumps({"event": "train_development_truth_materialized_once"}), flush=True)

    structure_result = probe_core.evaluate_family(
        train=frozen_structure["train"],
        development=frozen_structure["development"],
        train_labels=complete_labels["train"],
        development_labels=complete_labels["development"],
        policy=policy,
        average_precision_baseline=20 / 378,
        bootstrap=True,
    )
    prebuild_gate.validate_probe_result_contract(
        structure_result,
        policy=policy,
        average_precision_baseline=20 / 378,
    )
    if structure_result != prebuild_gate_result["probe_result"]:
        raise QualityAuditorV93Error(
            "Full-audit structure probe differs from the prebuild gate"
        )
    print(json.dumps({"event": "structure_probe_family_complete"}), flush=True)
    hard_text_result = probe_core.evaluate_family(
        train=frozen_text["train"]["style_deranged"],
        development=frozen_text["development"]["style_deranged"],
        train_labels=text_label_values["train"],
        development_labels=text_label_values["development"],
        policy=policy,
        average_precision_baseline=20 / 372,
        bootstrap=True,
    )
    print(json.dumps({"event": "style_deranged_probe_family_complete"}), flush=True)
    descriptive_text_result = probe_core.evaluate_family(
        train=frozen_text["train"]["original_author"],
        development=frozen_text["development"]["original_author"],
        train_labels=text_label_values["train"],
        development_labels=text_label_values["development"],
        policy=policy,
        average_precision_baseline=20 / 372,
        bootstrap=False,
    )
    print(json.dumps({"event": "original_author_descriptive_family_complete"}), flush=True)

    registry_rows = policy["observation_registry"]
    registry = {str(row["observation_id"]): row for row in registry_rows}
    observations = [
        _observation(
            registry,
            f"structural::{name}::boolean_pass",
            value,
            passed=value,
        )
        for name, value in structural_checks.items()
    ]
    gates = policy["quality_gates"]
    observations.extend(
        _family_observations(
            registry=registry,
            prefix="structure",
            result=structure_result,
            baseline=20 / 378,
            gates=gates,
            hard_gate=True,
        )
    )
    observations.extend(
        _family_observations(
            registry=registry,
            prefix="text::style_deranged",
            result=hard_text_result,
            baseline=20 / 372,
            gates=gates,
            hard_gate=True,
        )
    )
    observations.extend(
        _family_observations(
            registry=registry,
            prefix="text::original_author",
            result=descriptive_text_result,
            baseline=20 / 372,
            gates=gates,
            hard_gate=False,
        )
    )
    observed_ids = [str(row["observation_id"]) for row in observations]
    expected_ids = [str(row["observation_id"]) for row in registry_rows]
    if Counter(observed_ids) != Counter(expected_ids) or len(observed_ids) != len(expected_ids):
        raise QualityAuditorV93Error("Quality observation registry closure drift")
    ordered = {row["observation_id"]: row for row in observations}
    observations = [ordered[identifier] for identifier in expected_ids]
    hard_failures = [
        row["observation_id"]
        for row in observations
        if row["role"] == "qualification_hard_gate" and row["passed"] is not True
    ]
    result = {
        "version": VERSION,
        "status": "PASS_METHOD_QUALIFICATION_NOT_FORMAL_TRAINING_DATA"
        if not hard_failures
        else "DATASET_INVALIDATED",
        "canonical_self_sha256": None,
        "method_root_manifest_canonical_self_sha256": root_manifest[
            "canonical_self_sha256"
        ],
        "method_policy_canonical_self_sha256": policy["canonical_self_sha256"],
        "matrix_commitment_sha256": matrix_commitment_sha256,
        "matrix_commitments": matrix_commitments,
        "train_truth_read_count": 1,
        "development_truth_read_count": 1,
        "audit_a_truth_read_count": 0,
        "audit_b_truth_read_count": 0,
        "observation_count": len(observations),
        "observations": observations,
        "hard_failure_count": len(hard_failures),
        "hard_failure_observation_ids": hard_failures,
        "structure_family_summary": structure_result,
        "style_deranged_family_summary": hard_text_result,
        "original_author_descriptive_summary": descriptive_text_result,
        "m0_m1_m2_m3_training_authorized": False,
        "formal_500x4_generation_authorized": False,
    }
    result["canonical_self_sha256"] = _self_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(output, result)
    persisted = common.load_json(output)
    if persisted != result or persisted["canonical_self_sha256"] != _self_hash(persisted):
        raise QualityAuditorV93Error("Published quality evidence closure drift")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(root=args.root.resolve(), output=args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "canonical_self_sha256": result["canonical_self_sha256"],
                "observation_count": result["observation_count"],
                "hard_failure_count": result["hard_failure_count"],
                "audit_a_truth_read_count": 0,
                "audit_b_truth_read_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
