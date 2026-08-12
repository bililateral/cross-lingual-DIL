#!/usr/bin/env python3
"""Re-render one accepted v1.13 world after a controller-blind style swap."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_text_renderer as renderer
from step28_v13_v1_13_style_derangement import (
    StyleSourceDerangement,
    build_style_source_derangement,
)


class CounterfactualTextError(common.ContractError):
    """Raised if an intervention changes anything except effective author style."""


def _single_world_uid(
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> str:
    worlds = {
        str(row.get("world_uid", ""))
        for rows in (sellers, items, render_asts)
        for row in rows
    }
    if len(sellers) != 28 or len(worlds) != 1 or not next(iter(worlds)):
        raise CounterfactualTextError("Counterfactual requires one 28-seller world")
    return next(iter(worlds))


def _one_style_per_seller(
    *,
    sellers: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    reachable_style_uids: set[str],
) -> dict[str, str]:
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    styles: dict[str, set[str]] = {seller_uid: set() for seller_uid in seller_uids}
    counts: Counter[str] = Counter()
    for ast in render_asts:
        seller_uid = str(ast.get("seller_uid", ""))
        style_uid = str(ast.get("effective_style_uid", ""))
        if seller_uid not in styles or style_uid not in reachable_style_uids:
            raise CounterfactualTextError("Render AST seller/style universe drift")
        styles[seller_uid].add(style_uid)
        counts[seller_uid] += 1
    if (
        set(counts) != seller_uids
        or any(count <= 0 for count in counts.values())
        or any(len(values) != 1 for values in styles.values())
    ):
        raise CounterfactualTextError("Each seller must have exactly one style")
    return {seller_uid: next(iter(styles[seller_uid])) for seller_uid in seller_uids}


def _validate_mapping(
    source_by_target: Mapping[str, str], *, seller_uids: set[str]
) -> None:
    if (
        set(source_by_target) != seller_uids
        or set(source_by_target.values()) != seller_uids
        or any(target == source for target, source in source_by_target.items())
    ):
        raise CounterfactualTextError(
            "Style-source mapping must be a fixed-point-free bijection"
        )


def _rerender_descriptions(
    *,
    policy: Mapping[str, Any],
    split: str,
    template: Mapping[str, Any],
    ast_index: Mapping[str, Mapping[str, Any]],
    styles: Mapping[str, Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    identity_by_uid = {
        str(row["slot_uid"]): dict(row) for row in identity_slots_audit
    }
    noise_by_uid = {
        str(row["noise_slot_uid"]): dict(row) for row in noise_slots_audit
    }
    if (
        len(identity_by_uid) != len(identity_slots_audit)
        or len(noise_by_uid) != len(noise_slots_audit)
    ):
        raise CounterfactualTextError("Private slot UID collision")
    role_to_family = policy["identity_design"]["role_to_template_family"]
    descriptions: dict[str, str] = {}
    updated_identity: dict[str, dict[str, Any]] = {}
    updated_noise: dict[str, dict[str, Any]] = {}
    for item_uid in common.utf8_sort(ast_index):
        ast = ast_index[item_uid]
        base = production._base_description(
            ast=ast,
            split=split,
            template=template,
            styles=styles,
        )
        noise_uid = str(ast["noise_slot_uid"])
        noise = ""
        if noise_uid:
            if noise_uid not in noise_by_uid:
                raise CounterfactualTextError("AST references unknown noise slot")
            noise_row = dict(noise_by_uid[noise_uid])
            noise = str(noise_row["raw_surface"])
            noise_row["start"] = len(base)
            noise_row["end"] = len(base) + len(noise)
            updated_noise[noise_uid] = noise_row
        slot_uids = [str(value) for value in ast["identity_slot_uids"]]
        if len(slot_uids) != len(set(slot_uids)):
            raise CounterfactualTextError("AST repeats an identity slot")
        clauses: list[str] = []
        for slot_uid in slot_uids:
            if slot_uid not in identity_by_uid:
                raise CounterfactualTextError("AST references unknown identity slot")
            slot = identity_by_uid[slot_uid]
            clause = renderer.identity_clause(
                template_family=str(role_to_family[str(slot["planned_role"])]),
                identity_type=str(slot["identity_type"]),
                normalized_value=str(slot["raw_surface"]),
                template=template,
            )
            if clause.count(str(slot["raw_surface"])) != 1:
                raise CounterfactualTextError("Identity surface is not unique")
            clauses.append(clause)
        rendered = renderer.render_description(
            base_description=base,
            noise_clause=noise,
            identity_clauses=clauses,
            selector_uid=item_uid,
            template=template,
        )
        descriptions[item_uid] = rendered
        if clauses:
            guards = renderer.context_guard_sequence(
                selector_uid=item_uid,
                count=len(clauses) + 1,
                template=template,
            )
            cursor = len(base) + len(noise) + len(guards[0])
            for slot_uid, clause, guard in zip(
                slot_uids, clauses, guards[1:], strict=True
            ):
                slot = dict(identity_by_uid[slot_uid])
                relative = clause.index(str(slot["raw_surface"]))
                slot["start"] = cursor + relative
                slot["end"] = slot["start"] + len(str(slot["raw_surface"]))
                updated_identity[slot_uid] = slot
                cursor += len(clause) + len(guard)
            if cursor != len(rendered):
                raise CounterfactualTextError("Identity offset replay drift")
    if set(updated_identity) != set(identity_by_uid) or set(updated_noise) != set(
        noise_by_uid
    ):
        raise CounterfactualTextError("Counterfactual slot replay is incomplete")
    return (
        descriptions,
        [updated_identity[str(row["slot_uid"])] for row in identity_slots_audit],
        [updated_noise[str(row["noise_slot_uid"])] for row in noise_slots_audit],
    )


def rerender_counterfactual_world(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Swap only styles, then replay the production parser/redactor/profile chain."""

    world_uid = _single_world_uid(sellers, items, render_asts)
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    mapping: StyleSourceDerangement = build_style_source_derangement(
        split=split,
        world_uid=world_uid,
        seller_uids=tuple(seller_uids),
    )
    source_by_target = mapping.as_mapping()
    _validate_mapping(source_by_target, seller_uids=seller_uids)
    production._validate_observed_raw_against_private_ast(
        policy,
        split=split,
        template=template,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
        override_audit=override_audit,
    )
    style_rows = renderer.reachable_effective_styles(template)
    styles = {str(row["effective_style_uid"]): row for row in style_rows}
    original_style = _one_style_per_seller(
        sellers=sellers,
        render_asts=render_asts,
        reachable_style_uids=set(styles),
    )
    factor_order = tuple(template["renderer_contract"]["style_factor_order"])
    style_uid_changed_count = sum(
        original_style[target] != original_style[source_by_target[target]]
        for target in seller_uids
    )
    style_factor_changed_count = sum(
        tuple(styles[original_style[target]][name] for name in factor_order)
        != tuple(
            styles[original_style[source_by_target[target]]][name]
            for name in factor_order
        )
        for target in seller_uids
    )
    if style_uid_changed_count != style_factor_changed_count:
        raise CounterfactualTextError("Style UID/factor dose mismatch")
    original_style_uid_multiset = common.utf8_sort(original_style.values())
    mapped_style_uid_multiset = common.utf8_sort(
        original_style[source_by_target[target]] for target in seller_uids
    )
    original_style_factor_multiset = sorted(
        (
            tuple(styles[original_style[target]][name] for name in factor_order)
            for target in seller_uids
        ),
        key=common.canonical_json_bytes,
    )
    mapped_style_factor_multiset = sorted(
        (
            tuple(
                styles[original_style[source_by_target[target]]][name]
                for name in factor_order
            )
            for target in seller_uids
        ),
        key=common.canonical_json_bytes,
    )
    if (
        original_style_uid_multiset != mapped_style_uid_multiset
        or original_style_factor_multiset != mapped_style_factor_multiset
    ):
        raise CounterfactualTextError("Style derangement changed a style multiset")
    counterfactual_asts: list[dict[str, Any]] = []
    for source_ast in render_asts:
        ast = dict(source_ast)
        target_seller = str(ast["seller_uid"])
        ast["effective_style_uid"] = original_style[source_by_target[target_seller]]
        if any(
            ast[name] != value
            for name, value in source_ast.items()
            if name != "effective_style_uid"
        ):
            raise CounterfactualTextError("A non-style AST field changed")
        counterfactual_asts.append(ast)
    ast_index = {str(row["item_uid"]): row for row in counterfactual_asts}
    item_index = {str(row["item_uid"]): dict(row) for row in items}
    if len(ast_index) != len(counterfactual_asts) or set(ast_index) != set(item_index):
        raise CounterfactualTextError("Counterfactual item/AST keyset drift")
    titles = {
        item_uid: production._base_title(
            ast=ast,
            split=split,
            template=template,
            styles=styles,
        )
        for item_uid, ast in ast_index.items()
    }
    for override in sorted(
        override_audit,
        key=lambda row: (
            str(row["override_kind"]).encode("utf-8"),
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        ),
    ):
        if str(override["override_kind"]) == "exact_title_clone":
            source_uid = str(override["item_uid_left"])
            target_uid = str(override["item_uid_right"])
            if source_uid not in titles or target_uid not in titles or not titles[source_uid]:
                raise CounterfactualTextError("Exact-title clone lineage drift")
            titles[target_uid] = titles[source_uid]
    descriptions, updated_identity, updated_noise = _rerender_descriptions(
        policy=policy,
        split=split,
        template=template,
        ast_index=ast_index,
        styles=styles,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
    )
    counterfactual_items: list[dict[str, Any]] = []
    for source_item in items:
        item_uid = str(source_item["item_uid"])
        item = dict(source_item)
        item["title"] = titles[item_uid]
        item["description"] = descriptions[item_uid]
        if any(
            item[name] != value
            for name, value in source_item.items()
            if name not in {"title", "description"}
        ):
            raise CounterfactualTextError("Non-text item metadata changed")
        counterfactual_items.append(item)
    production._validate_observed_raw_against_private_ast(
        policy,
        split=split,
        template=template,
        items=counterfactual_items,
        identity_slots_audit=updated_identity,
        noise_slots_audit=updated_noise,
        render_asts=counterfactual_asts,
        override_audit=override_audit,
    )
    parsed = production.parse_observed_world(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=counterfactual_items,
    )
    parser_audit = production.validate_parser_against_private_plan(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=counterfactual_items,
        parsed_rows=parsed,
        identity_slots_audit=updated_identity,
        noise_slots_audit=updated_noise,
        render_asts=counterfactual_asts,
    )
    registry_profiles = production.registry_profiles_from_sellers(
        policy, sellers=sellers
    )
    redaction = production.redact_observed_world(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=sellers,
        items=counterfactual_items,
        registry_profiles=registry_profiles,
        parsed_rows=parsed,
    )
    redaction_audit = production.validate_redaction_against_private_plan(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=sellers,
        items=counterfactual_items,
        redacted_items=redaction["redacted_items"],
        parsed_rows=parsed,
        identity_slots_audit=updated_identity,
        noise_slots_audit=updated_noise,
        render_asts=counterfactual_asts,
        override_audit=override_audit,
    )
    profile_safe_items = production.build_profile_safe_items(
        policy,
        items=counterfactual_items,
        redacted_items=redaction["redacted_items"],
    )
    seller_profiles, profile_audit = profiles_module.build_world_profiles(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=profile_safe_items,
    )
    original_identity_core = [
        {name: value for name, value in row.items() if name not in {"start", "end"}}
        for row in identity_slots_audit
    ]
    updated_identity_core = [
        {name: value for name, value in row.items() if name not in {"start", "end"}}
        for row in updated_identity
    ]
    original_noise_core = [
        {name: value for name, value in row.items() if name not in {"start", "end"}}
        for row in noise_slots_audit
    ]
    updated_noise_core = [
        {name: value for name, value in row.items() if name not in {"start", "end"}}
        for row in updated_noise
    ]
    if (
        common.canonical_json_bytes(original_identity_core)
        != common.canonical_json_bytes(updated_identity_core)
        or common.canonical_json_bytes(original_noise_core)
        != common.canonical_json_bytes(updated_noise_core)
    ):
        raise CounterfactualTextError("Identity or noise lineage changed")
    original_items_by_seller: dict[str, list[Mapping[str, Any]]] = {
        seller_uid: [] for seller_uid in seller_uids
    }
    counterfactual_items_by_seller: dict[str, list[Mapping[str, Any]]] = {
        seller_uid: [] for seller_uid in seller_uids
    }
    for original, updated in zip(items, counterfactual_items, strict=True):
        seller_uid = str(original["seller_uid"])
        if seller_uid != str(updated["seller_uid"]):
            raise CounterfactualTextError("Counterfactual item seller changed")
        original_items_by_seller[seller_uid].append(original)
        counterfactual_items_by_seller[seller_uid].append(updated)
    visible_seller_changed_count = sum(
        any(
            str(original[field]) != str(updated[field])
            for original, updated in zip(
                sorted(
                    original_items_by_seller[seller_uid],
                    key=lambda row: str(row["item_uid"]).encode("utf-8"),
                ),
                sorted(
                    counterfactual_items_by_seller[seller_uid],
                    key=lambda row: str(row["item_uid"]).encode("utf-8"),
                ),
                strict=True,
            )
            for field in ("title", "description")
        )
        for seller_uid in seller_uids
    )
    return {
        "world_uid": world_uid,
        "public": {
            "raw_items": counterfactual_items,
            "redacted_items": redaction["redacted_items"],
            "profile_safe_items": profile_safe_items,
            "seller_profiles": seller_profiles,
        },
        "private": {
            "render_asts": counterfactual_asts,
            "identity_slots_audit": updated_identity,
            "noise_slots_audit": updated_noise,
            "parsed_identity_occurrences": parsed,
        },
        "audit": {
            "parser": parser_audit,
            "redaction": redaction_audit,
            "profile": profile_audit,
            "derangement_attempt": mapping.attempt,
            "seller_set_sha256": mapping.seller_set_sha256,
            "mapping_sha256": mapping.mapping_sha256,
            "target_source_pairs": [list(row) for row in mapping.target_source_pairs],
            "source_seller_changed_count": len(source_by_target),
            "effective_style_uid_changed_count": style_uid_changed_count,
            "effective_style_factor_tuple_changed_count": style_factor_changed_count,
            "zero_dose_seller_count": len(source_by_target)
            - style_factor_changed_count,
            "visible_seller_changed_count": visible_seller_changed_count,
            "zero_visible_dose_seller_count": len(source_by_target)
            - visible_seller_changed_count,
            "original_style_uid_multiset_sha256": common.canonical_sha256(
                original_style_uid_multiset
            ),
            "mapped_style_uid_multiset_sha256": common.canonical_sha256(
                mapped_style_uid_multiset
            ),
            "original_style_factor_multiset_sha256": common.canonical_sha256(
                original_style_factor_multiset
            ),
            "mapped_style_factor_multiset_sha256": common.canonical_sha256(
                mapped_style_factor_multiset
            ),
            "changed_title_count": sum(
                str(original["title"]) != str(updated["title"])
                for original, updated in zip(items, counterfactual_items, strict=True)
            ),
            "changed_description_count": sum(
                str(original["description"]) != str(updated["description"])
                for original, updated in zip(items, counterfactual_items, strict=True)
            ),
        },
    }
