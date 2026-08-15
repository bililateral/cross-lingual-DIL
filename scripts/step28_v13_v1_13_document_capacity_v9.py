#!/usr/bin/env python3
"""Label-isolated visible-document capacity repair for Step28-v13 v1.13 v9.

This module runs only after the base world, registered override endpoints,
identity slots, noise slots, and baseline Identity33 have been frozen.  It may
change only render-AST item codes and an originally joint-empty item's two
visibility flags.  It never allocates or moves identity-bearing structure.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure


VERSION = "2026-08-14-step28-v13-v1-13-document-capacity-v9"
FEISTEL_DOMAIN = b"step28-v13-v1.13-v9-item-code-feistel-v1"
FEISTEL_KEY_LABEL = b"key"
CARRIER_DOMAIN = b"step28-v13-v1.13-v9-joint-empty-carrier-v1"
FEISTEL_ROUNDS = 6
HALF_BITS = 20
HALF_MASK = (1 << HALF_BITS) - 1
WORLD_STRIDE = 256
SELLER_STRIDE = 8
SELLER_COUNT = 28
CODE_RE = re.compile(r"^Q[A-P]{10}$")
CODE_TOKEN_RE = re.compile(r"Q[A-P]{10}")


class DocumentCapacityError(common.ContractError):
    """Raised when the v9 representation-only repair cannot close exactly."""


def _canonical_clone(value: Any) -> Any:
    return json.loads(common.canonical_json_bytes(value).decode("utf-8"))


def _validated_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise DocumentCapacityError("Document-variation key must contain 32 bytes")
    return value


def derive_code_key(document_variation_key: bytes) -> bytes:
    key = _validated_key(document_variation_key)
    return hmac.new(
        key,
        common.FIELD_SEPARATOR.join((FEISTEL_DOMAIN, FEISTEL_KEY_LABEL)),
        hashlib.sha256,
    ).digest()


def _round_function(*, code_key: bytes, round_index: int, right_half: int) -> int:
    if (
        not isinstance(code_key, bytes)
        or len(code_key) != 32
        or type(round_index) is not int
        or not 0 <= round_index < FEISTEL_ROUNDS
        or type(right_half) is not int
        or not 0 <= right_half <= HALF_MASK
    ):
        raise DocumentCapacityError("Feistel round input drift")
    message = common.FIELD_SEPARATOR.join(
        (
            FEISTEL_DOMAIN,
            round_index.to_bytes(1, "big"),
            right_half.to_bytes(3, "big"),
        )
    )
    return int.from_bytes(
        hmac.new(code_key, message, hashlib.sha256).digest()[:3], "big"
    ) >> 4


def permute_40(*, code_key: bytes, value: int) -> int:
    if type(value) is not int or not 0 <= value < (1 << 40):
        raise DocumentCapacityError("Feistel input is outside the 40-bit domain")
    left = value >> HALF_BITS
    right = value & HALF_MASK
    for round_index in range(FEISTEL_ROUNDS):
        left, right = (
            right,
            left
            ^ _round_function(
                code_key=code_key,
                round_index=round_index,
                right_half=right,
            ),
        )
    return (left << HALF_BITS) | right


def invert_40(*, code_key: bytes, value: int) -> int:
    if type(value) is not int or not 0 <= value < (1 << 40):
        raise DocumentCapacityError("Feistel output is outside the 40-bit domain")
    left = value >> HALF_BITS
    right = value & HALF_MASK
    for round_index in reversed(range(FEISTEL_ROUNDS)):
        left, right = (
            right
            ^ _round_function(
                code_key=code_key,
                round_index=round_index,
                right_half=left,
            ),
            left,
        )
    return (left << HALF_BITS) | right


def coordinate(
    *,
    mode_global_ordinal: int,
    seller_slot_ordinal: int,
    item_slot_ordinal: int,
) -> int:
    if (
        type(mode_global_ordinal) is not int
        or not 0 <= mode_global_ordinal < (1 << 32)
        or type(seller_slot_ordinal) is not int
        or not 0 <= seller_slot_ordinal < SELLER_COUNT
        or type(item_slot_ordinal) is not int
        or not 0 <= item_slot_ordinal < SELLER_STRIDE
    ):
        raise DocumentCapacityError("Item-code creation slot drift")
    value = (
        mode_global_ordinal * WORLD_STRIDE
        + seller_slot_ordinal * SELLER_STRIDE
        + item_slot_ordinal
    )
    if value >= 1 << 40:
        raise DocumentCapacityError("Item-code coordinate exceeds 40 bits")
    return value


def encode_code(*, code_key: bytes, value: int) -> str:
    permuted = permute_40(code_key=code_key, value=value)
    output = "Q" + "".join(
        chr(ord("A") + int(symbol, 16)) for symbol in f"{permuted:010x}"
    )
    if CODE_RE.fullmatch(output) is None:
        raise DocumentCapacityError("Encoded item code has the wrong format")
    return output


def decode_code(*, code_key: bytes, code: str) -> int:
    if not isinstance(code, str) or CODE_RE.fullmatch(code) is None:
        raise DocumentCapacityError("Item code has the wrong format")
    permuted = int("".join(f"{ord(symbol) - ord('A'):x}" for symbol in code[1:]), 16)
    return invert_40(code_key=code_key, value=permuted)


def _carrier_projection(*, document_variation_key: bytes, item_uid: str) -> str:
    if not isinstance(item_uid, str) or not item_uid:
        raise DocumentCapacityError("Carrier projection lacks an item UID")
    digest = hmac.new(
        _validated_key(document_variation_key),
        common.FIELD_SEPARATOR.join((CARRIER_DOMAIN, item_uid.encode("utf-8"))),
        hashlib.sha256,
    ).digest()
    return "title_only" if digest[0] & 1 else "description_only"


def _creation_slot_maps(
    *,
    policy: Mapping[str, Any],
    mode: str,
    world_uid: str,
    world: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    try:
        id_key = str(policy["randomness"][mode]["id_key_hex"])
    except (KeyError, TypeError) as exc:
        raise DocumentCapacityError("ID-key custody is malformed") from exc
    expected_sellers = {
        structure.base_uid(
            key_hex=id_key,
            entity_kind="seller",
            parent_uid_or_mode=world_uid,
            ordinal=ordinal,
        ): ordinal
        for ordinal in range(SELLER_COUNT)
    }
    actual_sellers = {str(row["seller_uid"]) for row in world["public"]["sellers"]}
    if actual_sellers != set(expected_sellers):
        raise DocumentCapacityError("Seller creation slots do not reconstruct UIDs")

    items_by_seller: defaultdict[str, set[str]] = defaultdict(set)
    for row in world["public"]["items"]:
        seller_uid = str(row["seller_uid"])
        item_uid = str(row["item_uid"])
        if seller_uid not in expected_sellers or item_uid in items_by_seller[seller_uid]:
            raise DocumentCapacityError("Public item ownership or multiplicity drift")
        items_by_seller[seller_uid].add(item_uid)
    if set(items_by_seller) != actual_sellers:
        raise DocumentCapacityError("A seller has no public item rows")

    item_slots: dict[str, tuple[int, int]] = {}
    for seller_uid, seller_slot in expected_sellers.items():
        actual_items = items_by_seller[seller_uid]
        if not 1 <= len(actual_items) <= SELLER_STRIDE:
            raise DocumentCapacityError("Seller item count is outside 1..8")
        expected_items = {
            structure.base_uid(
                key_hex=id_key,
                entity_kind="item",
                parent_uid_or_mode=seller_uid,
                ordinal=item_slot,
            ): item_slot
            for item_slot in range(len(actual_items))
        }
        if actual_items != set(expected_items):
            raise DocumentCapacityError("Item creation slots do not reconstruct UIDs")
        for item_uid, item_slot in expected_items.items():
            item_slots[item_uid] = (seller_slot, item_slot)
    if len(item_slots) != len(world["public"]["items"]):
        raise DocumentCapacityError("Item creation-slot closure failed")
    return expected_sellers, item_slots


def apply_capacity_parent(
    *,
    policy: Mapping[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    document_variation_key: bytes,
    world: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a representation-only parent and a private deterministic receipt."""

    world_uid = str(world_record.get("world_uid", ""))
    mode_global_ordinal = world_record.get("mode_global_ordinal")
    if (
        not world_uid
        or world_uid != str(world["public"]["world"]["world_uid"])
        or type(mode_global_ordinal) is not int
    ):
        raise DocumentCapacityError("Capacity parent world binding drift")
    _seller_slots, item_slots = _creation_slot_maps(
        policy=policy,
        mode=mode,
        world_uid=world_uid,
        world=world,
    )
    output = copy.deepcopy(dict(world))
    ast_by_item = {
        str(row["item_uid"]): row for row in output["private"]["render_asts"]
    }
    if len(ast_by_item) != len(output["private"]["render_asts"]):
        raise DocumentCapacityError("Render AST contains duplicate item UIDs")
    if set(ast_by_item) != set(item_slots):
        raise DocumentCapacityError("Render AST and creation-slot universes differ")
    public_by_item = {
        str(row["item_uid"]): row for row in output["public"]["items"]
    }
    if len(public_by_item) != len(output["public"]["items"]) or set(
        public_by_item
    ) != set(ast_by_item):
        raise DocumentCapacityError("Public item and render-AST universes differ")

    identity_audit = output["private"]["identity_slots_audit"]
    identity_edit = output["private"]["identity_slots_edit"]
    edit_fields = (
        "slot_uid",
        "item_uid",
        "seller_uid",
        "field_name",
        "start",
        "end",
        "identity_type",
        "downstream_canonical_value",
        "raw_surface",
        "time_bucket",
    )
    projected_edits = [
        {field: row[field] for field in edit_fields} for row in identity_audit
    ]
    if common.canonical_json_bytes(projected_edits) != common.canonical_json_bytes(
        identity_edit
    ):
        raise DocumentCapacityError(
            "Identity audit/edit projections disagree before capacity repair"
        )
    identity_items = {str(row["item_uid"]) for row in identity_audit}
    identity_edit_items = {str(row["item_uid"]) for row in identity_edit}
    noise_items = {
        str(row["item_uid"]) for row in output["private"]["noise_slots_audit"]
    }
    override_items = {
        str(row[field])
        for row in output["private"]["override_audit"]
        for field in ("item_uid_left", "item_uid_right")
    }
    code_key = derive_code_key(document_variation_key)
    slot_rows: list[dict[str, Any]] = []
    codes: set[str] = set()
    projection_counts = {"none": 0, "title_only": 0, "description_only": 0}
    for item_uid in sorted(ast_by_item, key=lambda value: value.encode("utf-8")):
        ast = ast_by_item[item_uid]
        seller_slot, item_slot = item_slots[item_uid]
        value = coordinate(
            mode_global_ordinal=mode_global_ordinal,
            seller_slot_ordinal=seller_slot,
            item_slot_ordinal=item_slot,
        )
        code = encode_code(code_key=code_key, value=value)
        if decode_code(code_key=code_key, code=code) != value or code in codes:
            raise DocumentCapacityError("Item-code bijection or uniqueness failed")
        codes.add(code)
        original_title = ast.get("title_nonempty")
        original_description = ast.get("description_nonempty")
        if not isinstance(original_title, bool) or not isinstance(
            original_description, bool
        ):
            raise DocumentCapacityError("Render visibility flags are not Boolean")
        projection = "none"
        if not original_title and not original_description:
            if (
                item_uid in identity_items
                or item_uid in identity_edit_items
                or item_uid in noise_items
                or item_uid in override_items
                or ast.get("identity_slot_uids") != []
                or ast.get("noise_slot_uid") != ""
                or str(public_by_item[item_uid].get("title", "")) != ""
                or str(public_by_item[item_uid].get("description", "")) != ""
            ):
                raise DocumentCapacityError(
                    "Joint-empty item already carries text, identity, noise, or override state"
                )
            projection = _carrier_projection(
                document_variation_key=document_variation_key,
                item_uid=item_uid,
            )
            ast["title_nonempty"] = projection == "title_only"
            ast["description_nonempty"] = projection == "description_only"
        ast["code"] = code
        projection_counts[projection] += 1
        slot_rows.append(
            {
                "item_uid": item_uid,
                "seller_slot_ordinal": seller_slot,
                "item_slot_ordinal": item_slot,
                "coordinate": value,
                "code": code,
                "original_title_nonempty": original_title,
                "original_description_nonempty": original_description,
                "carrier_projection": projection,
            }
        )

    restored = copy.deepcopy(output)
    restored_ast = {
        str(row["item_uid"]): row for row in restored["private"]["render_asts"]
    }
    original_ast = {
        str(row["item_uid"]): row for row in world["private"]["render_asts"]
    }
    for item_uid, row in restored_ast.items():
        for field in ("code", "title_nonempty", "description_nonempty"):
            row[field] = original_ast[item_uid][field]
    if common.canonical_json_bytes(restored) != common.canonical_json_bytes(world):
        raise DocumentCapacityError("Capacity repair changed a forbidden base-world field")
    receipt = {
        "version": VERSION,
        "world_uid": world_uid,
        "mode_global_ordinal": mode_global_ordinal,
        "item_count": len(slot_rows),
        "unique_code_count": len(codes),
        "projection_counts": projection_counts,
        "slot_rows_sha256": common.canonical_sha256(slot_rows),
        "base_world_sha256": common.canonical_sha256(world),
        "capacity_parent_sha256": common.canonical_sha256(output),
        "labels_controllers_candidates_or_registries_read": False,
        "changed_render_fields": [
            "code",
            "title_nonempty",
            "description_nonempty",
        ],
    }
    return _canonical_clone(output), receipt
