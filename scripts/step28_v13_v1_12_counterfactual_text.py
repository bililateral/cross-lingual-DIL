#!/usr/bin/env python3
"""Re-render one Step28-v13 world after a controller-blind style derangement."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_production_chain as production
import step28_v13_profiles as profiles
import step28_v13_text_renderer as renderer
import step28_v13_v1_12_preceremony as preceremony
from step28_v13_v1_12_style_derangement import (
    StyleSourceDerangement,
    build_style_source_derangement,
)


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
        raise common.ContractError("Counterfactual re-render requires one 28-seller world")
    return next(iter(worlds))


def _one_effective_style_per_seller(
    *,
    sellers: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    reachable_style_uids: set[str],
) -> dict[str, str]:
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    styles: dict[str, set[str]] = {seller_uid: set() for seller_uid in seller_uids}
    item_counts: Counter[str] = Counter()
    for ast in render_asts:
        seller_uid = str(ast.get("seller_uid", ""))
        style_uid = str(ast.get("effective_style_uid", ""))
        if seller_uid not in styles or style_uid not in reachable_style_uids:
            raise common.ContractError("Counterfactual render AST seller/style drift")
        styles[seller_uid].add(style_uid)
        item_counts[seller_uid] += 1
    if (
        set(item_counts) != seller_uids
        or any(count <= 0 for count in item_counts.values())
        or any(len(values) != 1 for values in styles.values())
    ):
        raise common.ContractError(
            "Each counterfactual seller must have exactly one original effective style"
        )
    return {seller_uid: next(iter(styles[seller_uid])) for seller_uid in seller_uids}


def _validate_source_mapping(
    source_by_target: Mapping[str, str],
    *,
    seller_uids: set[str],
    identity_fixture: bool,
) -> None:
    if (
        set(source_by_target) != seller_uids
        or set(source_by_target.values()) != seller_uids
    ):
        raise common.ContractError("Counterfactual source mapping is not a bijection")
    fixed_points = {
        target for target, source in source_by_target.items() if target == source
    }
    if identity_fixture:
        if fixed_points != seller_uids:
            raise common.ContractError("Identity fixture must be the exact identity mapping")
    elif fixed_points:
        raise common.ContractError("Counterfactual source mapping contains a fixed point")


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
        raise common.ContractError("Counterfactual slot UID collision")
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
                raise common.ContractError("Counterfactual AST references unknown noise slot")
            noise_row = dict(noise_by_uid[noise_uid])
            noise = str(noise_row["raw_surface"])
            noise_row["start"] = len(base)
            noise_row["end"] = len(base) + len(noise)
            updated_noise[noise_uid] = noise_row

        slot_uids = [str(value) for value in ast["identity_slot_uids"]]
        if len(slot_uids) != len(set(slot_uids)):
            raise common.ContractError("Counterfactual AST repeats an identity slot")
        clauses: list[str] = []
        for slot_uid in slot_uids:
            if slot_uid not in identity_by_uid:
                raise common.ContractError(
                    "Counterfactual AST references unknown identity slot"
                )
            slot = identity_by_uid[slot_uid]
            family = str(role_to_family[str(slot["planned_role"])])
            clause = renderer.identity_clause(
                template_family=family,
                identity_type=str(slot["identity_type"]),
                normalized_value=str(slot["raw_surface"]),
                template=template,
            )
            if clause.count(str(slot["raw_surface"])) != 1:
                raise common.ContractError(
                    "Identity surface is not unique inside its registered clause"
                )
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
                raise common.ContractError("Counterfactual identity offset replay drift")

    if set(updated_identity) != set(identity_by_uid) or set(updated_noise) != set(
        noise_by_uid
    ):
        raise common.ContractError("Counterfactual slot replay is incomplete")
    return (
        descriptions,
        [updated_identity[str(row["slot_uid"])] for row in identity_slots_audit],
        [updated_noise[str(row["noise_slot_uid"])] for row in noise_slots_audit],
    )


def _rerender_with_mapping(
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
    source_by_target: Mapping[str, str],
    identity_fixture: bool,
) -> dict[str, Any]:
    world_uid = _single_world_uid(sellers, items, render_asts)
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    _validate_source_mapping(
        source_by_target,
        seller_uids=seller_uids,
        identity_fixture=identity_fixture,
    )
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
    original_style_by_seller = _one_effective_style_per_seller(
        sellers=sellers,
        render_asts=render_asts,
        reachable_style_uids=set(styles),
    )

    counterfactual_asts: list[dict[str, Any]] = []
    for source_ast in render_asts:
        ast = dict(source_ast)
        target_seller = str(ast["seller_uid"])
        ast["effective_style_uid"] = original_style_by_seller[
            source_by_target[target_seller]
        ]
        for name, value in source_ast.items():
            if name != "effective_style_uid" and ast[name] != value:
                raise common.ContractError("Counterfactual changed a non-style AST field")
        counterfactual_asts.append(ast)
    ast_index = {str(row["item_uid"]): row for row in counterfactual_asts}
    item_index = {str(row["item_uid"]): dict(row) for row in items}
    if len(ast_index) != len(counterfactual_asts) or set(ast_index) != set(item_index):
        raise common.ContractError("Counterfactual item/AST keyset drift")

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
                raise common.ContractError("Counterfactual exact-title lineage drift")
            titles[target_uid] = titles[source_uid]

    descriptions, counterfactual_identity_slots, counterfactual_noise_slots = (
        _rerender_descriptions(
            policy=policy,
            split=split,
            template=template,
            ast_index=ast_index,
            styles=styles,
            identity_slots_audit=identity_slots_audit,
            noise_slots_audit=noise_slots_audit,
        )
    )
    counterfactual_items: list[dict[str, Any]] = []
    for source_item in items:
        item_uid = str(source_item["item_uid"])
        item = dict(source_item)
        item["title"] = titles[item_uid]
        item["description"] = descriptions[item_uid]
        for name, value in source_item.items():
            if name not in {"title", "description"} and item[name] != value:
                raise common.ContractError("Counterfactual changed non-text item metadata")
        counterfactual_items.append(item)

    production._validate_observed_raw_against_private_ast(
        policy,
        split=split,
        template=template,
        items=counterfactual_items,
        identity_slots_audit=counterfactual_identity_slots,
        noise_slots_audit=counterfactual_noise_slots,
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
        identity_slots_audit=counterfactual_identity_slots,
        noise_slots_audit=counterfactual_noise_slots,
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
        identity_slots_audit=counterfactual_identity_slots,
        noise_slots_audit=counterfactual_noise_slots,
        render_asts=counterfactual_asts,
        override_audit=override_audit,
    )
    profile_safe_items = production.build_profile_safe_items(
        policy,
        items=counterfactual_items,
        redacted_items=redaction["redacted_items"],
    )
    visible_projection = preceremony.project_registered_visible_text(
        policy=policy,
        template=template,
        sellers=sellers,
        items=counterfactual_items,
        parsed_rows=parsed,
    )
    projected_profile_index = {
        str(row["item_uid"]): row
        for row in visible_projection["profile_safe_items"]
    }
    production_profile_index = {
        str(row["item_uid"]): row for row in profile_safe_items
    }
    if (
        visible_projection["redacted_items"] != redaction["redacted_items"]
        or len(projected_profile_index) != len(visible_projection["profile_safe_items"])
        or len(production_profile_index) != len(profile_safe_items)
        or projected_profile_index != production_profile_index
    ):
        raise common.ContractError(
            "Registered projection and production redactor disagree by item UID"
        )
    seller_profiles, profile_audit = profiles.build_world_profiles(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=profile_safe_items,
    )
    return {
        "world_uid": world_uid,
        "public": {
            "sellers": [dict(row) for row in sellers],
            "raw_items": counterfactual_items,
            "redacted_items": redaction["redacted_items"],
            "profile_safe_items": profile_safe_items,
            "seller_profiles": seller_profiles,
        },
        "private": {
            "render_asts": counterfactual_asts,
            "identity_slots_audit": counterfactual_identity_slots,
            "noise_slots_audit": counterfactual_noise_slots,
            "parsed_identity_occurrences": parsed,
        },
        "audit": {
            "identity_fixture": identity_fixture,
            "parser": parser_audit,
            "redaction": redaction_audit,
            "profile": profile_audit,
            "source_style_changed_seller_count": sum(
                original_style_by_seller[target]
                != original_style_by_seller[source_by_target[target]]
                for target in seller_uids
            ),
            "raw_title_changed_item_count": sum(
                str(original["title"]) != str(updated["title"])
                for original, updated in zip(items, counterfactual_items, strict=True)
            ),
            "raw_description_changed_item_count": sum(
                str(original["description"]) != str(updated["description"])
                for original, updated in zip(items, counterfactual_items, strict=True)
            ),
        },
    }


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
    """Apply the frozen public derangement, then replay the full production chain."""

    world_uid = _single_world_uid(sellers, items, render_asts)
    mapping: StyleSourceDerangement = build_style_source_derangement(
        split=split,
        world_uid=world_uid,
        seller_uids=[str(row["seller_uid"]) for row in sellers],
    )
    output = _rerender_with_mapping(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=sellers,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
        override_audit=override_audit,
        source_by_target=mapping.as_mapping(),
        identity_fixture=False,
    )
    output["audit"]["derangement"] = {
        "attempt": mapping.attempt,
        "seller_set_sha256": mapping.seller_set_sha256,
        "mapping_sha256": mapping.mapping_sha256,
        "target_source_pairs": [list(row) for row in mapping.target_source_pairs],
    }
    return output


def rerender_identity_mapping_fixture(
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
    """Unit-test-only identity mapping proving byte-equal production replay."""

    seller_uids = {str(row["seller_uid"]) for row in sellers}
    return _rerender_with_mapping(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=sellers,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
        override_audit=override_audit,
        source_by_target={seller_uid: seller_uid for seller_uid in seller_uids},
        identity_fixture=True,
    )
