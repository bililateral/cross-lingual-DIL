#!/usr/bin/env python3
"""Create V9.4 proxy items from only a truth-free public nuisance schedule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
from itertools import combinations
from types import MappingProxyType
from typing import Any

import step28_v13_v1_13_model_visible_projection_v9_4 as projection_v94


VERSION = "2026-08-27-step28-v13-v1-13-model-visible-prebuild-source-v9-4"
WORLD_FIELDS = (
    "split",
    "world_ordinal",
    "world_uid",
    "seller_uids",
    "noise_slot_by_seller_slot",
)
SIGNATURE_FIELDS = (
    "noise_slot",
    "item_count",
    "title_present_mask",
    "description_present_mask",
    "joint_empty_mask",
)
ALLOWED_SPLITS = ("train", "development")


class ModelVisiblePrebuildSourceV94Error(ValueError):
    """Raised when the truth-free prebuild source surface drifts."""


def _time_bucket(
    *,
    time_key_hex: str,
    split: str,
    world_ordinal: int,
    noise_slot: int,
    logical_item_ordinal: int,
) -> int:
    if (
        not isinstance(time_key_hex, str)
        or len(time_key_hex) != 64
        or any(character not in "0123456789abcdef" for character in time_key_hex)
    ):
        raise ModelVisiblePrebuildSourceV94Error("Time key commitment input drift")
    payload = (
        f"v9.4-time-bucket::{split}::{world_ordinal:03d}::"
        f"noise_{noise_slot:02d}::item_{logical_item_ordinal:02d}"
    ).encode("utf-8")
    digest = hmac.new(bytes.fromhex(time_key_hex), payload, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % 4


def _signature_by_slot(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[int, str, str]]:
    signatures: dict[int, tuple[int, str, str]] = {}
    for source in rows:
        if not isinstance(source, Mapping) or tuple(source) != SIGNATURE_FIELDS:
            raise ModelVisiblePrebuildSourceV94Error("Noise signature schema/order drift")
        slot = source["noise_slot"]
        count = source["item_count"]
        title_mask = source["title_present_mask"]
        description_mask = source["description_present_mask"]
        joint_empty_mask = source["joint_empty_mask"]
        if (
            type(slot) is not int
            or not 0 <= slot < 28
            or type(count) is not int
            or not 2 <= count <= 8
            or not all(isinstance(value, str) for value in (
                title_mask,
                description_mask,
                joint_empty_mask,
            ))
            or not all(
                len(value) == count and set(value) <= {"0", "1"}
                for value in (title_mask, description_mask, joint_empty_mask)
            )
            or title_mask.count("1") < 1
            or description_mask.count("1") < 2
            or any(
                int(joint_empty_mask[index])
                != int(title_mask[index] == "0" and description_mask[index] == "0")
                for index in range(count)
            )
            or slot in signatures
        ):
            raise ModelVisiblePrebuildSourceV94Error("Noise signature value drift")
        signatures[slot] = (count, title_mask, description_mask)
    if set(signatures) != set(range(28)):
        raise ModelVisiblePrebuildSourceV94Error("Noise signature slot closure drift")
    return signatures


def build_truth_free_world_source(
    *,
    world: Mapping[str, Any],
    noise_signatures: Sequence[Mapping[str, Any]],
    time_key_hex: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Return exact endpoint/item projections without any controller or label input."""

    if not isinstance(world, Mapping) or tuple(world) != WORLD_FIELDS:
        raise ModelVisiblePrebuildSourceV94Error("World nuisance schema/order drift")
    split = world["split"]
    world_ordinal = world["world_ordinal"]
    world_uid = world["world_uid"]
    seller_uids = world["seller_uids"]
    noise_slots = world["noise_slot_by_seller_slot"]
    if (
        split not in ALLOWED_SPLITS
        or type(world_ordinal) is not int
        or not 0 <= world_ordinal < 500
        or not isinstance(world_uid, str)
        or not world_uid
        or "|" in world_uid
        or type(seller_uids) is not list
        or len(seller_uids) != 28
        or any(not isinstance(value, str) or not value or "|" in value for value in seller_uids)
        or len(set(seller_uids)) != 28
        or seller_uids != sorted(seller_uids, key=lambda value: value.encode("utf-8"))
        or type(noise_slots) is not list
        or len(noise_slots) != 28
        or any(type(value) is not int for value in noise_slots)
        or set(noise_slots) != set(range(28))
    ):
        raise ModelVisiblePrebuildSourceV94Error("World nuisance value drift")
    signatures = _signature_by_slot(noise_signatures)
    items: list[dict[str, Any]] = []
    for seller_slot, seller_uid in enumerate(seller_uids):
        noise_slot = noise_slots[seller_slot]
        count, title_mask, description_mask = signatures[noise_slot]
        for ordinal in range(count):
            items.append({
                "world_uid": world_uid,
                "seller_uid": seller_uid,
                "logical_item_ordinal": ordinal,
                "title_nonempty": title_mask[ordinal] == "1",
                "description_nonempty": description_mask[ordinal] == "1",
                "time_bucket": _time_bucket(
                    time_key_hex=time_key_hex,
                    split=split,
                    world_ordinal=world_ordinal,
                    noise_slot=noise_slot,
                    logical_item_ordinal=ordinal,
                ),
            })
    endpoints = [
        {
            "world_uid": world_uid,
            "canonical_pair_uid": f"{left}||{right}",
            "seller_uid_left": left,
            "seller_uid_right": right,
        }
        for left, right in combinations(seller_uids, 2)
    ]
    return (
        tuple(MappingProxyType(dict(row)) for row in endpoints),
        tuple(MappingProxyType(dict(row)) for row in items),
    )


def build_truth_free_world_projection(
    *,
    world: Mapping[str, Any],
    noise_signatures: Sequence[Mapping[str, Any]],
    time_key_hex: str,
) -> tuple[Mapping[str, str], ...]:
    endpoints, items = build_truth_free_world_source(
        world=world,
        noise_signatures=noise_signatures,
        time_key_hex=time_key_hex,
    )
    return projection_v94.build_world_projection(
        endpoint_rows=endpoints,
        item_rows=items,
    )


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "world_fields": list(WORLD_FIELDS),
        "signature_fields": list(SIGNATURE_FIELDS),
        "allowed_splits": list(ALLOWED_SPLITS),
        "time_bucket_derivation": "HMAC-SHA256-first-u64-big-endian-modulo-4",
        "reads_controller_membership": False,
        "reads_registered_negative_plan": False,
        "reads_labels": False,
        "reads_model_outputs": False,
    }
