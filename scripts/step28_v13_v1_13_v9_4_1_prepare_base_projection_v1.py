#!/usr/bin/env python3
"""Prepare a label-free, opaque GPU workload for the formal base projection.

The implementation policy grants no formal execution authority.  This module
therefore exposes deterministic building blocks but its command line can only
validate the contract; a future exact-commit one-time wrapper must own formal
publication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_base24_shared_v2 as shared_base24
import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as experiment_common
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as training_common
import step28_v13_v1_13_v9_4_1_prepare_public_projection_v1 as predecessor
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as common
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common


ROW_KEY_FIELDS = (
    "split",
    "world_ordinal",
    "world_uid",
    "canonical_pair_uid",
    "seller_uid_left",
    "seller_uid_right",
)
ITEM_FIELDS = ("world_uid", "seller_uid", "item_uid", "title", "description")
GPU_PAIR_FIELDS = ("pair_uid", "seller_uid_left", "seller_uid_right")
GPU_SELLER_TEXT_FIELDS = ("seller_uid", "field_name", "text_uid", "multiplicity")
GPU_TEXT_FIELDS = ("text_uid", "text", "text_sha256")
FIELD_ORDER = {"title": 0, "description": 1}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise common.PublicProjectionContractError(
                    f"Blank JSONL line at {path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise common.PublicProjectionContractError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise common.PublicProjectionContractError(
                    f"Non-object JSONL row at {path}:{line_number}"
                )
            yield value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise common.PublicProjectionContractError(f"Missing CSV header: {path}")
        return list(reader)


def render_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if tuple(row) != tuple(fields):
                raise common.PublicProjectionContractError("CSV row schema/order drift")
            writer.writerow(row)


def render_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def render_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def matrix_value_sha256(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(matrix)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_pair_uid(left: str, right: str) -> str:
    first, second = experiment_common.canonical_pair_endpoints(left, right)
    return f"{first}||{second}"


def validate_public_row_order(
    split: str,
    worlds: Sequence[Mapping[str, Any]],
    sellers: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, str]],
    expected: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    if len(worlds) != expected["worlds"] or len(sellers) != expected["sellers"]:
        raise common.PublicProjectionContractError("Formal world/seller count drift")
    world_order: dict[str, int] = {}
    sellers_by_world: dict[str, list[str]] = defaultdict(list)
    for ordinal, world in enumerate(worlds):
        if (
            tuple(world)
            != (
                "world_uid",
                "split",
                "world_ordinal",
                "seller_count",
                "item_count",
                "pair_count",
            )
            or world["split"] != split
            or int(world["world_ordinal"]) != ordinal
            or int(world["seller_count"]) != expected["sellers_per_world"]
            or int(world["item_count"]) != expected["items"] // expected["worlds"]
            or int(world["pair_count"]) != expected["pairs_per_world"]
        ):
            raise common.PublicProjectionContractError("Formal world schema/order drift")
        uid = str(world["world_uid"])
        if uid in world_order:
            raise common.PublicProjectionContractError("Duplicate formal world UID")
        world_order[uid] = ordinal
    seller_uids: list[str] = []
    seller_worlds: dict[str, str] = {}
    for seller in sellers:
        if tuple(seller) != ("world_uid", "seller_uid", "market"):
            raise common.PublicProjectionContractError("Public seller schema/order drift")
        world_uid = str(seller["world_uid"])
        seller_uid = str(seller["seller_uid"])
        if world_uid not in world_order or not isinstance(seller["market"], str):
            raise common.PublicProjectionContractError("Seller references another world")
        sellers_by_world[world_uid].append(seller_uid)
        seller_uids.append(seller_uid)
        seller_worlds[seller_uid] = world_uid
    if len(set(seller_uids)) != len(seller_uids):
        raise common.PublicProjectionContractError("Duplicate public seller UID")
    for world_uid, ordinal in world_order.items():
        observed = sellers_by_world[world_uid]
        if len(observed) != expected["sellers_per_world"] or observed != sorted(
            observed, key=lambda value: value.encode("utf-8")
        ):
            raise common.PublicProjectionContractError(
                f"Seller order/universe drift at world {ordinal}"
            )

    if len(pairs) != expected["pairs"]:
        raise common.PublicProjectionContractError("Formal pair count drift")
    row_keys: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    pair_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    previous_ordinal = -1
    for pair in pairs:
        if tuple(pair) != (
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        ):
            raise common.PublicProjectionContractError("Pair endpoint schema/order drift")
        world_uid = str(pair["world_uid"])
        if world_uid not in world_order:
            raise common.PublicProjectionContractError("Pair references another world")
        ordinal = world_order[world_uid]
        if ordinal < previous_ordinal:
            raise common.PublicProjectionContractError("Pair world order drift")
        previous_ordinal = ordinal
        left = str(pair["seller_uid_left"])
        right = str(pair["seller_uid_right"])
        endpoints = experiment_common.canonical_pair_endpoints(left, right)
        if (
            endpoints != (left, right)
            or pair["canonical_pair_uid"] != canonical_pair_uid(left, right)
            or left not in sellers_by_world[world_uid]
            or right not in sellers_by_world[world_uid]
        ):
            raise common.PublicProjectionContractError("Canonical pair identity drift")
        pair_counts[world_uid] += 1
        pair_sets[world_uid].add(endpoints)
        row_keys.append(
            {
                "split": split,
                "world_ordinal": ordinal,
                "world_uid": world_uid,
                "canonical_pair_uid": str(pair["canonical_pair_uid"]),
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
        )
    for world_uid, ordinal in world_order.items():
        universe = sellers_by_world[world_uid]
        expected_pairs = {
            (left, right)
            for index, left in enumerate(universe)
            for right in universe[index + 1 :]
        }
        if (
            pair_counts[world_uid] != expected["pairs_per_world"]
            or pair_sets[world_uid] != expected_pairs
        ):
            raise common.PublicProjectionContractError(
                f"K28 pair universe drift at world {ordinal}"
            )
    return row_keys, seller_uids, seller_worlds


def build_opaque_text_workload(
    items: Iterable[Mapping[str, Any]],
    *,
    valid_worlds: set[str],
    seller_uids: Sequence[str],
    seller_worlds: Mapping[str, str],
    expected_item_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    seller_set = set(seller_uids)
    text_by_uid: dict[str, str] = {}
    mappings: set[tuple[str, str, str]] = set()
    item_uids: set[str] = set()
    world_item_counts: Counter[str] = Counter()
    item_count = 0
    for item in items:
        item_count += 1
        if tuple(item) != ITEM_FIELDS:
            raise common.PublicProjectionContractError("Redacted item schema/order drift")
        world_uid = str(item["world_uid"])
        seller_uid = str(item["seller_uid"])
        item_uid = str(item["item_uid"])
        if (
            world_uid not in valid_worlds
            or seller_uid not in seller_set
            or seller_worlds.get(seller_uid) != world_uid
            or item_uid in item_uids
        ):
            raise common.PublicProjectionContractError("Redacted item identity drift")
        item_uids.add(item_uid)
        world_item_counts[world_uid] += 1
        for field in ("title", "description"):
            text = item[field]
            if not isinstance(text, str):
                raise common.PublicProjectionContractError("Redacted text is not a string")
            if not text.strip():
                continue
            text_uid = hashlib.sha256(text.encode("utf-8")).hexdigest()
            previous = text_by_uid.setdefault(text_uid, text)
            if previous != text:
                raise common.PublicProjectionContractError("Text SHA-256 collision")
            mappings.add((seller_uid, field, text_uid))
    if item_count != expected_item_count or len(item_uids) != expected_item_count:
        raise common.PublicProjectionContractError("Redacted item count drift")
    expected_per_world = expected_item_count // len(valid_worlds)
    if expected_per_world * len(valid_worlds) != expected_item_count or any(
        world_item_counts[world_uid] != expected_per_world
        for world_uid in valid_worlds
    ):
        raise common.PublicProjectionContractError("Per-world item count drift")
    represented = {seller_uid for seller_uid, _field, _text_uid in mappings}
    if represented != seller_set:
        raise common.PublicProjectionContractError("Seller without nonempty public text")
    opaque_sellers = {
        seller_uid: f"seller_{ordinal:06d}"
        for ordinal, seller_uid in enumerate(
            sorted(seller_uids, key=lambda value: value.encode("utf-8")), start=1
        )
    }
    unique_rows = [
        {"text_uid": text_uid, "text": text, "text_sha256": text_uid}
        for text_uid, text in sorted(text_by_uid.items())
    ]
    seller_rows = [
        {
            "seller_uid": opaque_sellers[seller_uid],
            "field_name": field,
            "text_uid": text_uid,
            "multiplicity": 1,
        }
        for seller_uid, field, text_uid in sorted(
            mappings,
            key=lambda value: (
                opaque_sellers[value[0]],
                FIELD_ORDER[value[1]],
                value[2],
            ),
        )
    ]
    if any(tuple(row) != GPU_TEXT_FIELDS for row in unique_rows) or any(
        tuple(row) != GPU_SELLER_TEXT_FIELDS for row in seller_rows
    ):
        raise common.PublicProjectionContractError("Opaque text workload schema drift")
    return unique_rows, seller_rows, opaque_sellers


def opaque_pair_rows(
    pairs: Sequence[Mapping[str, str]], opaque_sellers: Mapping[str, str]
) -> list[dict[str, str]]:
    rows = [
        {
            "pair_uid": f"pair_{index:06d}",
            "seller_uid_left": opaque_sellers[str(pair["seller_uid_left"])],
            "seller_uid_right": opaque_sellers[str(pair["seller_uid_right"])],
        }
        for index, pair in enumerate(pairs, start=1)
    ]
    if any(tuple(row) != GPU_PAIR_FIELDS for row in rows):
        raise common.PublicProjectionContractError("Opaque pair schema drift")
    return rows


def prepare_split(
    policy: Mapping[str, Any],
    experiment_policy: Mapping[str, Any],
    split: str,
    part_id: str,
    reference: Mapping[str, Any],
    cpu_root: Path,
    transfer_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = predecessor.verify_split_public_inputs(
        experiment_policy, split, predecessor.BASE_PUBLIC_ROLES
    )
    worlds = list(iter_jsonl(paths["worlds.jsonl"]))
    sellers = list(iter_jsonl(paths["sellers.jsonl"]))
    pairs = read_csv(paths["complete_model_pair_endpoints.csv"])
    row_keys, seller_uids, seller_worlds = validate_public_row_order(
        split,
        worlds,
        sellers,
        pairs,
        policy["formal_dataset"]["counts_per_split"],
    )
    profiles = {
        str(row["seller_uid"]): shared_base24.project_model_profile(row)
        for row in iter_jsonl(paths["model_seller_profiles.jsonl"])
    }
    if set(profiles) != set(seller_uids) or len(profiles) != len(seller_uids):
        raise common.PublicProjectionContractError("Model profile seller universe drift")
    legacy18 = shared_base24.legacy18_matrix(
        pairs, profiles, reference, policy["feature_contract"]["legacy18"]
    )
    if legacy18.shape != (len(pairs), 18) or not np.isfinite(legacy18).all():
        raise common.PublicProjectionContractError("legacy18 matrix contract drift")

    unique_rows, seller_rows, opaque_sellers = build_opaque_text_workload(
        iter_jsonl(paths["redacted_items.jsonl"]),
        valid_worlds={str(row["world_uid"]) for row in worlds},
        seller_uids=seller_uids,
        seller_worlds=seller_worlds,
        expected_item_count=int(policy["formal_dataset"]["counts_per_split"]["items"]),
    )
    gpu_pairs = opaque_pair_rows(pairs, opaque_sellers)

    cpu_split = cpu_root / split
    cpu_split.mkdir(parents=True, exist_ok=False)
    row_path = cpu_split / "row_keys.csv"
    legacy_path = cpu_split / "legacy18.npy"
    render_csv(row_path, row_keys, ROW_KEY_FIELDS)
    np.save(legacy_path, np.ascontiguousarray(legacy18, dtype="<f8"), allow_pickle=False)

    transfer_part = transfer_root / part_id
    transfer_part.mkdir(parents=True, exist_ok=False)
    text_path = transfer_part / "opaque_unique_texts.jsonl"
    seller_path = transfer_part / "opaque_seller_text_index.jsonl"
    pair_path = transfer_part / "opaque_pair_endpoints.csv"
    render_jsonl(text_path, unique_rows)
    render_jsonl(seller_path, seller_rows)
    render_csv(pair_path, gpu_pairs, GPU_PAIR_FIELDS)
    cpu_record = {
        "split": split,
        "part_id": part_id,
        "world_count": len(worlds),
        "seller_count": len(seller_uids),
        "item_count": int(policy["formal_dataset"]["counts_per_split"]["items"]),
        "pair_count": len(pairs),
        "row_keys_file": file_record(row_path, cpu_root),
        "legacy18_file": file_record(legacy_path, cpu_root),
        "legacy18_shape": list(legacy18.shape),
        "legacy18_dtype": legacy18.dtype.str,
        "legacy18_value_sha256": matrix_value_sha256(legacy18),
    }
    transfer_record = {
        "part_id": part_id,
        "unique_text_count": len(unique_rows),
        "seller_text_row_count": len(seller_rows),
        "opaque_seller_count": len(opaque_sellers),
        "opaque_pair_count": len(gpu_pairs),
        "files": {
            "opaque_unique_texts": file_record(text_path, transfer_root),
            "opaque_seller_text_index": file_record(seller_path, transfer_root),
            "opaque_pair_endpoints": file_record(pair_path, transfer_root),
        },
    }
    return cpu_record, transfer_record


def prepare_to_temporary(
    policy: Mapping[str, Any], cpu_root: Path, transfer_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build unpublished temporary payloads for a future authorized wrapper."""

    if cpu_root.exists() or transfer_root.exists():
        raise common.PublicProjectionContractError("Temporary projection root exists")
    experiment_policy = experiment_common.load_policy()
    successor_policy = training_common.load_policy()
    reference, _seller_records, _pairs = shared_base24.reconstruct_frozen_english_public(
        successor_policy
    )
    cpu_root.mkdir(parents=True)
    transfer_root.mkdir(parents=True)
    cpu_splits: list[dict[str, Any]] = []
    transfer_parts: list[dict[str, Any]] = []
    for ordinal, split in enumerate(common.SPLITS):
        cpu, transfer = prepare_split(
            policy,
            experiment_policy,
            split,
            f"part_{ordinal:03d}",
            reference,
            cpu_root,
            transfer_root,
        )
        cpu_splits.append(cpu)
        transfer_parts.append(transfer)
    cpu_manifest = {
        "step": "step28_v13_v1_13_v9_4_1_prepare_base_projection_v1",
        "status": "PREPARED_UNPUBLISHED_LABEL_FREE_CPU_STAGE",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "splits": cpu_splits,
        "total_pair_count": sum(row["pair_count"] for row in cpu_splits),
        "labels_controllers_membership_qrels_or_audit_truth_read": False,
        "identity33_read": False,
        "model_training_or_scoring_performed": False,
    }
    cpu_manifest["canonical_self_hash"] = common.canonical_sha256(cpu_manifest)
    render_json(cpu_root / "cpu_stage_manifest.json", cpu_manifest)
    transfer_manifest = {
        "step": "step28_v13_v1_13_v9_4_1_opaque_gpu_transfer_v1",
        "status": "FROZEN_OPAQUE_LABEL_FREE_TRANSFER_NO_CANONICAL_IDENTITIES",
        "public_policy_canonical_self_hash": policy["canonical_self_hash"],
        "gpu_policy_canonical_self_hash": gpu_common.load_policy()[
            "canonical_self_hash"
        ],
        "cpu_stage_canonical_self_hash": cpu_manifest["canonical_self_hash"],
        "parts": transfer_parts,
        "part_count": len(transfer_parts),
        "split_names_or_canonical_identifiers_present": False,
        "labels_controllers_membership_qrels_or_audit_truth_present": False,
        "identity33_or_legacy18_present": False,
    }
    transfer_manifest["canonical_self_hash"] = common.canonical_sha256(
        transfer_manifest
    )
    render_json(transfer_root / "transfer_manifest.json", transfer_manifest)
    return cpu_manifest, transfer_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    result = {
        "status": "PASSED_BASE_PREPARATION_CONTRACT_NO_FORMAL_EXECUTION",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "formal_projection_executed": False,
        "supervision_or_audit_truth_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
