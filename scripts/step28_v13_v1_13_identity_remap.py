#!/usr/bin/env python3
"""V1.13-only deterministic identity remapping for trial parent allocation.

This module has no filesystem, seed, capability, dataset, or training entry.
It remaps the temporary fixed-width identity surfaces of one already-built
world while resolving collisions against caller-provided immutable registries.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_identity_values as identity_values
import step28_v13_text_renderer as renderer
import step7_v3_1_source_data as source


HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_VALUE_DOMAIN = b"step28-v13-v1.13-identity-value"


class IdentityRemapError(common.ContractError):
    """Raised when the v1.13 identity remap cannot close exactly."""


def _require_hash_set(values: frozenset[str] | set[str], *, label: str) -> None:
    if type(values) not in {frozenset, set}:
        raise IdentityRemapError(f"{label} must be a set or frozenset")
    if any(
        not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None
        for value in values
    ):
        raise IdentityRemapError(f"{label} contains a malformed SHA-256 value")


def _candidate_identity_value(
    *,
    key_hex: str,
    world_uid: str,
    identity_asset_uid: str,
    identity_type: str,
    counter: int,
) -> str:
    if (
        not isinstance(key_hex, str)
        or HEX_SHA256_RE.fullmatch(key_hex) is None
        or isinstance(counter, bool)
        or not isinstance(counter, int)
        or counter < 0
    ):
        raise IdentityRemapError("Identity candidate counter/key is malformed")
    digest = hmac.new(
        bytes.fromhex(key_hex),
        IDENTITY_VALUE_DOMAIN
        + b"\x1f"
        + world_uid.encode("utf-8")
        + b"\x1f"
        + identity_asset_uid.encode("utf-8")
        + b"\x1f"
        + identity_type.encode("ascii")
        + b"\x1f"
        + str(counter).encode("ascii"),
        hashlib.sha256,
    ).digest()
    modulus = identity_values.domain_size(identity_type, "parser_safe_hex_v2")
    return identity_values.encode_identity_value(
        identity_type,
        int.from_bytes(digest, "big", signed=False) % modulus,
        handle_encoding="parser_safe_hex_v2",
    )


def _identity_value_collides_with_visible_text(
    value: str,
    *,
    visible_texts: Sequence[str],
    visible_compacts: Sequence[str],
) -> bool:
    normalized = str(value).strip().casefold()
    compact = source.compact_identifier(value)
    if not normalized or not compact:
        raise IdentityRemapError("Identity candidate has an empty visible projection")
    return any(normalized in text for text in visible_texts) or any(
        compact in text for text in visible_compacts
    )


def _earliest_context_guard_boundary(
    value: str, *, guards: Sequence[str]
) -> tuple[int | None, int]:
    positions = [position for guard in guards if (position := value.find(guard)) >= 0]
    return (min(positions) if positions else None), sum(
        value.count(guard) for guard in guards
    )


def _select_first_admissible_candidate(
    candidate_rows: Iterable[tuple[str, str]],
    *,
    historical_forbidden: frozenset[str],
    allocated_in_trial: set[str],
    visible_texts: Sequence[str],
    visible_compacts: Sequence[str],
) -> tuple[int, str, str, int]:
    visible_rejections = 0
    for counter, (value, value_hash) in enumerate(candidate_rows):
        if HEX_SHA256_RE.fullmatch(value_hash) is None:
            raise IdentityRemapError("Identity candidate hash is malformed")
        if _identity_value_collides_with_visible_text(
            value,
            visible_texts=visible_texts,
            visible_compacts=visible_compacts,
        ):
            visible_rejections += 1
            continue
        if value_hash in historical_forbidden or value_hash in allocated_in_trial:
            continue
        allocated_in_trial.add(value_hash)
        return counter, value, value_hash, visible_rejections
    raise IdentityRemapError("Per-asset deterministic candidate domain exhausted")


def remap_world_identity_values(
    world: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    key_hex: str,
    historical_forbidden: frozenset[str],
    allocated_in_trial: set[str],
    maximum_counter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remap one copied world with the v1.13 domain and no fault hook."""

    if (
        not isinstance(key_hex, str)
        or HEX_SHA256_RE.fullmatch(key_hex) is None
        or isinstance(maximum_counter, bool)
        or not isinstance(maximum_counter, int)
        or maximum_counter < 1
    ):
        raise IdentityRemapError("Identity remap key/counter budget is invalid")
    _require_hash_set(historical_forbidden, label="historical identity registry")
    _require_hash_set(allocated_in_trial, label="trial identity registry")
    if historical_forbidden & allocated_in_trial:
        raise IdentityRemapError("Trial identity registry intersects history")
    output = copy.deepcopy(dict(world))
    public = output.get("public")
    private = output.get("private")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise IdentityRemapError("World identity remap boundary is malformed")
    assets = private.get("identity_assets")
    slot_audit = private.get("identity_slots_audit")
    slot_edit = private.get("identity_slots_edit")
    items = public.get("items")
    if not all(
        isinstance(value, list) for value in (assets, slot_audit, slot_edit, items)
    ):
        raise IdentityRemapError("World identity remap tables are missing")
    world_uids = {
        str(row["world_uid"])
        for row in [*items, *private.get("mechanism_assignments", [])]
    }
    if len(world_uids) != 1:
        raise IdentityRemapError("World identity remap received mixed worlds")
    world_uid = next(iter(world_uids))
    guards = renderer.context_guard_pool(template)
    if not guards or len(guards) != len(set(guards)):
        raise IdentityRemapError("Identity context-guard registry is invalid")
    visible_texts: list[str] = []
    visible_compacts: list[str] = []
    for item in items:
        raw_description = str(item["description"])
        boundary, _count = _earliest_context_guard_boundary(
            raw_description, guards=guards
        )
        for text in (
            str(item["title"]),
            raw_description if boundary is None else raw_description[:boundary],
        ):
            normalized = source.normalize_redacted_text(text).casefold()
            visible_texts.append(normalized)
            visible_compacts.append(source.compact_identifier(normalized))
    ordered_assets = sorted(
        assets, key=lambda row: str(row["identity_asset_uid"]).encode("utf-8")
    )
    if (
        not ordered_assets
        or len({str(row["identity_asset_uid"]) for row in ordered_assets})
        != len(ordered_assets)
        or len({str(row["identity_uid"]) for row in ordered_assets})
        != len(ordered_assets)
    ):
        raise IdentityRemapError("Identity asset UID/value lineage is not one-to-one")

    allocation_start_count = len(allocated_in_trial)
    old_uid_to_new: dict[str, tuple[str, str, str]] = {}
    allocation_audit_rows: list[dict[str, Any]] = []
    selected_counters: list[int] = []
    visible_rejection_count = 0
    for asset in ordered_assets:
        identity_type = str(asset["identity_type"])
        asset_uid = str(asset["identity_asset_uid"])
        old_identity_uid = str(asset["identity_uid"])
        old_value = str(asset["identity_value"])

        def candidates() -> Iterable[tuple[str, str]]:
            for counter in range(maximum_counter + 1):
                value = _candidate_identity_value(
                    key_hex=key_hex,
                    world_uid=world_uid,
                    identity_asset_uid=asset_uid,
                    identity_type=identity_type,
                    counter=counter,
                )
                yield value, identity_values.value_hash(value)

        counter, new_value, selected_hash, rejected = (
            _select_first_admissible_candidate(
                candidates(),
                historical_forbidden=historical_forbidden,
                allocated_in_trial=allocated_in_trial,
                visible_texts=visible_texts,
                visible_compacts=visible_compacts,
            )
        )
        visible_rejection_count += rejected
        if len(new_value) != len(old_value):
            raise IdentityRemapError("Identity remap changed fixed-width surface length")
        new_identity_uid = "id_" + common.canonical_sha256(
            {
                "contact_type": identity_type.strip().lower(),
                "normalized_value": new_value.strip().lower(),
            }
        )
        if old_identity_uid in old_uid_to_new:
            raise IdentityRemapError("Temporary identity UID is duplicated")
        old_uid_to_new[old_identity_uid] = (
            new_identity_uid,
            new_value,
            selected_hash,
        )
        allocation_audit_rows.append(
            {
                "identity_asset_uid": asset_uid,
                "selected_counter": counter,
                "visible_text_candidate_rejection_count": rejected,
                "selected_value_hash": selected_hash,
            }
        )
        selected_counters.append(counter)
        asset["identity_value"] = new_value
        asset["identity_uid"] = new_identity_uid

    slot_by_uid: dict[str, Mapping[str, Any]] = {}
    edits_by_item: dict[str, list[tuple[int, int, str, str]]] = {}
    for row in slot_audit:
        slot_uid = str(row["slot_uid"])
        replacement = old_uid_to_new.get(str(row["identity_uid"]))
        if replacement is None or slot_uid in slot_by_uid:
            raise IdentityRemapError("Identity slot references an unknown/duplicate asset")
        new_identity_uid, new_value, _value_hash = replacement
        old_surface = str(row["raw_surface"])
        if len(old_surface) != len(new_value):
            raise IdentityRemapError("Identity slot replacement length drift")
        row["identity_uid"] = new_identity_uid
        row["raw_surface"] = new_value
        row["downstream_canonical_value"] = new_value.strip().lower()
        slot_by_uid[slot_uid] = row
        edits_by_item.setdefault(str(row["item_uid"]), []).append(
            (int(row["start"]), int(row["end"]), old_surface, new_value)
        )
    if len(slot_by_uid) != len(slot_audit):
        raise IdentityRemapError("Identity slot audit keyset drift during remap")
    seen_edit_slots: set[str] = set()
    for row in slot_edit:
        slot_uid = str(row["slot_uid"])
        source_row = slot_by_uid.get(slot_uid)
        if source_row is None or slot_uid in seen_edit_slots:
            raise IdentityRemapError("Identity edit row lacks a unique authoritative slot")
        seen_edit_slots.add(slot_uid)
        for field in (
            "item_uid",
            "seller_uid",
            "field_name",
            "start",
            "end",
            "identity_type",
            "time_bucket",
        ):
            if row[field] != source_row[field]:
                raise IdentityRemapError("Identity edit/slot immutable lineage drift")
        row["raw_surface"] = source_row["raw_surface"]
        row["downstream_canonical_value"] = source_row[
            "downstream_canonical_value"
        ]
    if seen_edit_slots != set(slot_by_uid):
        raise IdentityRemapError("Identity edit/slot keyset is not one-to-one")

    changed_item_uids: set[str] = set()
    item_uids = {str(row["item_uid"]) for row in items}
    if len(item_uids) != len(items) or not set(edits_by_item).issubset(item_uids):
        raise IdentityRemapError("Identity slot edit references an invalid item")
    for item in items:
        item_uid = str(item["item_uid"])
        description = str(item["description"])
        edits = sorted(edits_by_item.get(item_uid, []), reverse=True)
        for start, end, old_surface, new_surface in edits:
            if description[start:end] != old_surface:
                raise IdentityRemapError("Identity remap offset does not round-trip")
            description = description[:start] + new_surface + description[end:]
        if edits:
            changed_item_uids.add(item_uid)
            item["description"] = description

    if len(allocated_in_trial) - allocation_start_count != len(assets):
        raise IdentityRemapError("Identity remap allocation count drift")
    selected_hashes = {value[2] for value in old_uid_to_new.values()}
    if (
        len(selected_hashes) != len(assets)
        or selected_hashes & historical_forbidden
    ):
        raise IdentityRemapError("Identity remap collision closure failed")
    return output, {
        "world_uid": world_uid,
        "identity_asset_count": len(assets),
        "identity_slot_count": len(slot_audit),
        "changed_item_count": len(changed_item_uids),
        "maximum_counter": maximum_counter,
        "maximum_selected_counter": max(selected_counters),
        "nonzero_counter_count": sum(value > 0 for value in selected_counters),
        "forced_design_collision_count": 0,
        "visible_text_candidate_rejection_count": visible_rejection_count,
        "historical_intersection_count": 0,
        "same_run_intersection_count": 0,
        "selected_value_hashes_sha256": common.canonical_sha256(
            common.utf8_sort(selected_hashes)
        ),
        "allocation_audit_rows": allocation_audit_rows,
        "allocation_audit_rows_sha256": common.canonical_sha256(
            allocation_audit_rows
        ),
    }
