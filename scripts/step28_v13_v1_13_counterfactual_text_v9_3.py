#!/usr/bin/env python3
"""Create the V9.3 controller-blind, code-free style counterfactual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_world_builder_v9_3 as world_builder
from step28_v13_v1_13_style_derangement import build_style_source_derangement


VERSION = "2026-08-26-step28-v13-v1-13-counterfactual-text-v9-3"


class CounterfactualTextV93Error(common.ContractError):
    """Raised when the style intervention changes a non-style object."""


def _logical_style_source_mapping(
    *, split: str, world_ordinal: int, seller_uids: Sequence[str]
) -> dict[str, str]:
    actual = tuple(common.utf8_sort(seller_uids))
    if len(actual) != 28 or len(set(actual)) != 28:
        raise CounterfactualTextV93Error("Counterfactual seller universe drift")
    logical = tuple(f"logical_seller_{slot:02d}" for slot in range(28))
    logical_to_actual = dict(zip(logical, actual, strict=True))
    mapping = build_style_source_derangement(
        split=split,
        world_uid=f"v9_3_logical_world_{split}_{world_ordinal:03d}",
        seller_uids=logical,
    ).as_mapping()
    if (
        set(mapping) != set(logical)
        or set(mapping.values()) != set(logical)
        or any(target == source for target, source in mapping.items())
    ):
        raise CounterfactualTextV93Error("Logical style mapping is not a derangement")
    return {
        logical_to_actual[target]: logical_to_actual[source]
        for target, source in mapping.items()
    }


def _slot_clauses(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    slot_uids: Sequence[str],
    identity_by_uid: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    role_to_family = policy["identity_design"]["role_to_template_family"]
    clauses: list[str] = []
    for slot_uid in slot_uids:
        slot = identity_by_uid.get(slot_uid)
        if slot is None:
            raise CounterfactualTextV93Error("Counterfactual identity slot is absent")
        clauses.append(
            renderer.identity_clause(
                template_family=str(role_to_family[str(slot["planned_role"])]),
                identity_type=str(slot["identity_type"]),
                normalized_value=str(slot["raw_surface"]),
                template=template,
            )
        )
    return clauses


def rerender_counterfactual_world(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    split: str,
    world_ordinal: int,
    world: Mapping[str, Any],
) -> dict[str, Any]:
    """Swap author styles on logical slots while preserving all identity content."""

    public = world["public"]
    private = world["private"]
    sellers = [dict(row) for row in public["sellers"]]
    items = [dict(row) for row in public["items"]]
    asts = [dict(row) for row in private["render_asts"]]
    identity_rows = [dict(row) for row in private["identity_slots_audit"]]
    noise_rows = [dict(row) for row in private["noise_slots_audit"]]
    world_uid = str(public["world"]["world_uid"])
    if (
        len(sellers) != 28
        or any(str(row["world_uid"]) != world_uid for row in sellers)
        or any(str(row["world_uid"]) != world_uid for row in items)
    ):
        raise CounterfactualTextV93Error("Counterfactual world boundary drift")
    seller_uids = tuple(str(row["seller_uid"]) for row in sellers)
    source_by_target = _logical_style_source_mapping(
        split=split,
        world_ordinal=world_ordinal,
        seller_uids=seller_uids,
    )
    reachable = {
        str(row["effective_style_uid"]): dict(row)
        for row in renderer.reachable_effective_styles(dict(template))
    }
    original_style: dict[str, str] = {}
    ast_by_item: dict[str, dict[str, Any]] = {}
    for ast in asts:
        seller_uid = str(ast["seller_uid"])
        item_uid = str(ast["item_uid"])
        style_uid = str(ast["effective_style_uid"])
        if seller_uid not in source_by_target or style_uid not in reachable:
            raise CounterfactualTextV93Error("Counterfactual AST style drift")
        if seller_uid in original_style and original_style[seller_uid] != style_uid:
            raise CounterfactualTextV93Error("Seller has multiple original styles")
        if item_uid in ast_by_item:
            raise CounterfactualTextV93Error("Counterfactual AST item collision")
        original_style[seller_uid] = style_uid
        ast_by_item[item_uid] = ast
    if set(original_style) != set(seller_uids) or set(ast_by_item) != {
        str(row["item_uid"]) for row in items
    }:
        raise CounterfactualTextV93Error("Counterfactual AST universe is incomplete")
    factor_order = tuple(template["renderer_contract"]["style_factor_order"])
    original_multiset = sorted(
        tuple(reachable[original_style[seller]][name] for name in factor_order)
        for seller in seller_uids
    )
    mapped_multiset = sorted(
        tuple(
            reachable[original_style[source_by_target[seller]]][name]
            for name in factor_order
        )
        for seller in seller_uids
    )
    if original_multiset != mapped_multiset:
        raise CounterfactualTextV93Error("Style derangement changed the style multiset")

    counterfactual_asts: list[dict[str, Any]] = []
    for source in asts:
        ast = dict(source)
        ast["effective_style_uid"] = original_style[
            source_by_target[str(source["seller_uid"])]
        ]
        counterfactual_asts.append(ast)
    counterfactual_ast_by_item = {
        str(row["item_uid"]): row for row in counterfactual_asts
    }
    identity_by_uid = {str(row["slot_uid"]): row for row in identity_rows}
    noise_by_uid = {str(row["noise_slot_uid"]): row for row in noise_rows}
    if len(identity_by_uid) != len(identity_rows) or len(noise_by_uid) != len(
        noise_rows
    ):
        raise CounterfactualTextV93Error("Counterfactual private slot collision")

    titles: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    updated_identity: dict[str, dict[str, Any]] = {}
    updated_noise: dict[str, dict[str, Any]] = {}
    title_skeletons = template["split_libraries"][split]["title_skeletons"]
    description_skeletons = template["split_libraries"][split][
        "description_skeletons"
    ]
    for item_uid in common.utf8_sort(counterfactual_ast_by_item):
        ast = counterfactual_ast_by_item[item_uid]
        style = reachable[str(ast["effective_style_uid"])]
        phrase = str(ast["natural_variation_phrase"])
        title = (
            world_builder._render_code_free_title(
                skeleton=str(title_skeletons[int(ast["title_skeleton_index"])]),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                style=style,
                template=template,
                natural_variation_phrase=phrase,
            )
            if bool(ast["title_nonempty"])
            else ""
        )
        base_description = (
            world_builder._render_code_free_description(
                skeleton=str(
                    description_skeletons[
                        int(ast["description_skeleton_index"])
                    ]
                ),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                delivery=str(ast["delivery"]),
                service=str(ast["service"]),
                style=style,
                template=template,
                natural_variation_phrase=phrase,
            )
            if bool(ast["description_nonempty"])
            else ""
        )
        noise_uid = str(ast["noise_slot_uid"])
        noise_clause = ""
        if noise_uid:
            source_noise = noise_by_uid.get(noise_uid)
            if source_noise is None:
                raise CounterfactualTextV93Error("Counterfactual noise slot is absent")
            noise_clause = str(source_noise["raw_surface"])
            noise = dict(source_noise)
            noise["start"] = len(base_description)
            noise["end"] = len(base_description) + len(noise_clause)
            updated_noise[noise_uid] = noise
        slot_uids = tuple(str(value) for value in ast["identity_slot_uids"])
        clauses = _slot_clauses(
            policy=policy,
            template=template,
            slot_uids=slot_uids,
            identity_by_uid=identity_by_uid,
        )
        selector = world_builder._logical_render_selector(
            split=split,
            world_ordinal=world_ordinal,
            noise_slot=int(ast["noise_slot"]),
            logical_item_ordinal=int(ast["logical_item_ordinal"]),
        )
        description = renderer.render_description(
            base_description=base_description,
            noise_clause=noise_clause,
            identity_clauses=clauses,
            selector_uid=selector,
            template=template,
        )
        if clauses:
            guards = renderer.context_guard_sequence(
                selector_uid=selector,
                count=len(clauses) + 1,
                template=template,
            )
            cursor = len(base_description) + len(noise_clause) + len(guards[0])
            for slot_uid, clause, guard in zip(
                slot_uids, clauses, guards[1:], strict=True
            ):
                row = dict(identity_by_uid[slot_uid])
                raw_surface = str(row["raw_surface"])
                local_start = clause.find(raw_surface)
                if local_start < 0 or clause.find(raw_surface, local_start + 1) >= 0:
                    raise CounterfactualTextV93Error(
                        "Counterfactual identity surface is not unique"
                    )
                row["start"] = cursor + local_start
                row["end"] = row["start"] + len(raw_surface)
                updated_identity[slot_uid] = row
                cursor += len(clause) + len(guard)
            if cursor != len(description):
                raise CounterfactualTextV93Error(
                    "Counterfactual identity offsets did not close"
                )
        titles[item_uid] = title
        descriptions[item_uid] = description

    for override in private["override_audit"]:
        if str(override["override_kind"]) == "exact_title_clone":
            left = str(override["item_uid_left"])
            right = str(override["item_uid_right"])
            if left not in titles or right not in titles or not titles[left]:
                raise CounterfactualTextV93Error("Counterfactual clone lineage drift")
            titles[right] = titles[left]
    if set(updated_identity) != set(identity_by_uid) or set(updated_noise) != set(
        noise_by_uid
    ):
        raise CounterfactualTextV93Error("Counterfactual private slots are incomplete")

    counterfactual_items: list[dict[str, Any]] = []
    for source_item in items:
        item_uid = str(source_item["item_uid"])
        item = dict(source_item)
        item["title"] = titles[item_uid]
        item["description"] = descriptions[item_uid]
        counterfactual_items.append(item)
    style_changed = sum(
        original_style[seller]
        != original_style[source_by_target[seller]]
        for seller in seller_uids
    )
    factor_changed_counts = {
        factor: sum(
            reachable[original_style[seller]][factor]
            != reachable[original_style[source_by_target[seller]]][factor]
            for seller in seller_uids
        )
        for factor in factor_order
    }
    if any(count <= 0 for count in factor_changed_counts.values()):
        raise CounterfactualTextV93Error(
            "Style derangement left an entire frozen factor unchanged"
        )
    visible_changed = sum(
        any(
            before[field] != after[field]
            for field in ("title", "description")
        )
        for before, after in zip(items, counterfactual_items, strict=True)
    )
    return {
        "version": VERSION,
        "world_uid": world_uid,
        "public_items": counterfactual_items,
        "private_render_asts": counterfactual_asts,
        "private_identity_slots_audit": [
            updated_identity[str(row["slot_uid"])] for row in identity_rows
        ],
        "private_noise_slots_audit": [
            updated_noise[str(row["noise_slot_uid"])] for row in noise_rows
        ],
        "audit": {
            "source_mapping_sha256": common.canonical_sha256(source_by_target),
            "style_multiset_preserved": True,
            "style_changed_seller_count": style_changed,
            "style_factor_changed_seller_counts": factor_changed_counts,
            "visible_changed_item_count": visible_changed,
            "identity_slot_count": len(identity_rows),
            "noise_slot_count": len(noise_rows),
            "labels_or_controller_membership_read": False,
        },
    }
