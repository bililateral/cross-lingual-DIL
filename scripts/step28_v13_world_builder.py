#!/usr/bin/env python3
"""Build one deterministic Step 28-v13 world with strict public/oracle separation."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import step28_v13_common as common
import step28_v13_identity_plan as identity_plan
import step28_v13_nonidentity as nonidentity
import step28_v13_structure as structure
import step28_v13_text_renderer as renderer


FLAG_NAMES = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)


def _complete_pair_endpoints(
    world_uid: str, seller_uids: list[str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for left, right in itertools.combinations(common.utf8_sort(seller_uids), 2):
        rows.append(
            {
                "canonical_pair_uid": common.canonical_pair_uid(left, right),
                "world_uid": world_uid,
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
        )
    rows.sort(key=lambda row: row["canonical_pair_uid"].encode("utf-8"))
    if len(rows) != 378 or len(
        {row["canonical_pair_uid"] for row in rows}
    ) != 378:
        raise common.ContractError("Complete pair endpoint universe is not 378")
    return rows


def _render_identity_slots(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    noise_records_by_item: Mapping[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    role_to_family = policy["identity_design"]["role_to_template_family"]
    identity_audit: list[dict[str, Any]] = []
    identity_edit: list[dict[str, Any]] = []
    noise_audit: list[dict[str, Any]] = []
    for seller_uid in common.utf8_sort(items_by_seller):
        for item in sorted(
            items_by_seller[seller_uid],
            key=lambda row: row["item_uid"].encode("utf-8"),
        ):
            slots = sorted(
                item["identity_slots"],
                key=lambda row: row["slot_uid"].encode("utf-8"),
            )
            clauses: list[str] = []
            for slot in slots:
                family = str(role_to_family[slot["role"]])
                clauses.append(
                    renderer.identity_clause(
                        template_family=family,
                        identity_type=slot["identity_type"],
                        normalized_value=slot["identity_value"],
                        template=template,
                    )
                )
            description = renderer.render_description(
                base_description=item["base_description"],
                noise_clause=item["noise_clause"],
                identity_clauses=clauses,
                selector_uid=item["item_uid"],
                template=template,
            )
            item["description"] = description

            cursor = len(item["base_description"])
            noise_record = noise_records_by_item.get(item["item_uid"])
            if item["noise_clause"]:
                if noise_record is None:
                    raise common.ContractError("Noise clause lacks its private record")
                raw_noise = str(noise_record["raw_surface"])
                if item["noise_clause"] != raw_noise:
                    raise common.ContractError("Noise clause/private record drift")
                noise_audit.append(
                    {
                        "noise_slot_uid": noise_record["noise_slot_uid"],
                        "item_uid": item["item_uid"],
                        "seller_uid": seller_uid,
                        "field_name": "description",
                        "start": cursor,
                        "end": cursor + len(raw_noise),
                        "raw_surface": raw_noise,
                        "parser_expectation": "must_ignore",
                    }
                )
                cursor += len(raw_noise)
            elif noise_record is not None:
                raise common.ContractError("Noise private record lacks rendered clause")

            if not item["description_nonempty"]:
                if slots or item["noise_clause"] or description:
                    raise common.ContractError("Empty description received a slot")
                continue
            if not slots:
                if cursor != len(description):
                    raise common.ContractError(
                        "Identity-free description has an unregistered suffix"
                    )
                continue
            guards = renderer.context_guard_sequence(
                selector_uid=item["item_uid"],
                count=len(slots) + 1,
                template=template,
            )
            if description[cursor : cursor + len(guards[0])] != guards[0]:
                raise common.ContractError("Initial identity context guard drift")
            cursor += len(guards[0])
            for slot, clause, guard in zip(
                slots,
                clauses,
                guards[1:],
                strict=True,
            ):
                value = str(slot["identity_value"])
                local_start = clause.find(value)
                if local_start < 0 or clause.find(value, local_start + 1) >= 0:
                    raise common.ContractError(
                        "Identity value is not unique inside its rendered clause"
                    )
                value_start = cursor + local_start
                value_end = value_start + len(value)
                if description[value_start:value_end] != value:
                    raise common.ContractError("Identity slot offsets do not round-trip")
                family = str(role_to_family[slot["role"]])
                flag_values = fixture["expected_role_flags"][family][
                    slot["identity_type"]
                ]
                if len(flag_values) != len(FLAG_NAMES):
                    raise common.ContractError("Parser role-flag fixture drift")
                expected_flags = dict(zip(FLAG_NAMES, flag_values, strict=True))
                audit_row = {
                    "slot_uid": slot["slot_uid"],
                    "bundle_uid": slot["bundle_uid"],
                    "item_uid": item["item_uid"],
                    "seller_uid": seller_uid,
                    "field_name": "description",
                    "start": value_start,
                    "end": value_end,
                    "identity_uid": slot["identity_uid"],
                    "identity_type": slot["identity_type"],
                    "downstream_canonical_value": value.strip().lower(),
                    "raw_surface": value,
                    "parser_expectation": "must_extract",
                    "expected_seller_facing_context": int(
                        expected_flags["seller_facing_context"]
                    ),
                    "expected_product_data_risk_context": int(
                        expected_flags["product_data_risk_context"]
                    ),
                    "expected_direct_identity_eligible": int(
                        expected_flags["direct_identity_eligible"]
                    ),
                    "expected_support_only": int(expected_flags["support_only"]),
                    "planned_role": slot["role"],
                    "time_bucket": item["time_bucket"],
                }
                identity_audit.append(audit_row)
                identity_edit.append(
                    {
                        key: audit_row[key]
                        for key in (
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
                    }
                )
                cursor += len(clause)
                if description[cursor : cursor + len(guard)] != guard:
                    raise common.ContractError("Inter-identity context guard drift")
                cursor += len(guard)
            if cursor != len(description):
                raise common.ContractError("Rendered description cursor did not close")
    identity_audit.sort(
        key=lambda row: (
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
            row["slot_uid"].encode("utf-8"),
        )
    )
    identity_edit.sort(
        key=lambda row: (
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
            row["slot_uid"].encode("utf-8"),
        )
    )
    noise_audit.sort(
        key=lambda row: (
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
            row["noise_slot_uid"].encode("utf-8"),
        )
    )
    return identity_audit, identity_edit, noise_audit


def build_world(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    style_profile: dict[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    structure_key_hex: str,
) -> dict[str, Any]:
    """Build a world in memory; this function performs no filesystem writes."""

    common.validate_policy(policy, mode=mode)
    expected_template = common.load_json(
        common.repo_path(str(policy["template_library"]["path"]))
    )
    expected_fixture = common.load_json(
        common.repo_path(
            str(
                policy["identity_design"][
                    "role_template_parser_flag_fixture"
                ]["path"]
            )
        )
    )
    expected_style_profile = common.load_json(
        common.verify_file_pin(
            policy["style_reference_boundary"]["generator_release_inputs"][
                "profile"
            ],
            label="synthetic style reference",
        )
    )
    if (
        common.canonical_json_bytes(template)
        != common.canonical_json_bytes(expected_template)
        or common.canonical_json_bytes(fixture)
        != common.canonical_json_bytes(expected_fixture)
        or common.canonical_json_bytes(style_profile)
        != common.canonical_json_bytes(expected_style_profile)
    ):
        raise common.ContractError(
            "World builder received an unregistered release input object"
        )
    common.validate_independent_replay_public_domains(
        policy,
        template=template,
        style_profile=style_profile,
    )
    expected_records = {
        str(row["world_uid"]): row
        for row in structure.build_mode_world_pool(policy, mode=mode)
    }
    world_uid = str(world_record["world_uid"])
    expected_record = expected_records.get(world_uid)
    if (
        set(world_record)
        != {"world_uid", "mode_global_ordinal", "split", "split_ordinal"}
        or not isinstance(world_record["world_uid"], str)
        or type(world_record["mode_global_ordinal"]) is not int
        or type(world_record["split_ordinal"]) is not int
        or not isinstance(world_record["split"], str)
        or expected_record is None
        or common.canonical_json_bytes(dict(world_record))
        != common.canonical_json_bytes(expected_record)
    ):
        raise common.ContractError(
            "World record is not an exact member of the registered mode pool"
        )
    split = str(world_record["split"])
    expected_structure_key = common.structure_key_for_split(
        policy, mode=mode, split=split
    )
    if structure_key_hex != expected_structure_key:
        raise common.ContractError(
            "World builder received a structure key outside split custody"
        )
    graph_name = str(policy["identity_design"]["mechanism_by_split"][split])
    membership = structure.build_world_membership(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
    )
    markets, market_proposal = structure.assign_markets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        membership=membership,
    )
    mechanisms = structure.assign_controller_mechanisms(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
    )
    controller_styles = nonidentity.world_controller_styles(
        policy=policy,
        template=template,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        controller_uids=membership["controller_uids"],
        mode=mode,
    )
    items_by_seller: dict[str, list[dict[str, Any]]] = {}
    for seller_uid in membership["seller_uids"]:
        controller_uid = membership["seller_to_controller"][seller_uid]
        effective_style = nonidentity.seller_effective_style(
            policy=policy,
            template=template,
            mode=mode,
            seller_uid=seller_uid,
            controller_style=controller_styles[controller_uid],
        )
        items_by_seller[seller_uid] = nonidentity.build_seller_items(
            policy=policy,
            template=template,
            style_profile=style_profile,
            mode=mode,
            split=split,
            world_uid=world_uid,
            seller_uid=seller_uid,
            effective_style=effective_style,
        )

    override_audit: list[dict[str, Any]] = []
    noise_records_by_item: dict[str, dict[str, Any]] = {}
    id_key = policy["randomness"][mode]["id_key_hex"]
    pre_slot_call_count = 0

    def materialize_observed_overrides_and_noise(
        final_negative_flags: Any,
    ) -> dict[str, Any]:
        nonlocal pre_slot_call_count
        pre_slot_call_count += 1
        if pre_slot_call_count != 1:
            raise common.ContractError("Pre-slot materializer called more than once")
        override_audit.extend(
            nonidentity.apply_registered_overrides(
                policy=policy,
                template=template,
                style_profile=style_profile,
                split=split,
                world_uid=world_uid,
                structure_key_hex=structure_key_hex,
                items_by_seller=items_by_seller,
                negative_flags=final_negative_flags,
            )
        )
        for seller_uid in common.utf8_sort(membership["seller_uids"]):
            record = nonidentity.assign_must_ignore_noise(
                policy=policy,
                template=template,
                mode=mode,
                seller_uid=seller_uid,
                items=items_by_seller[seller_uid],
            )
            item_uid = str(record["item_uid"])
            if item_uid in noise_records_by_item:
                raise common.ContractError("Noise item reused across sellers")
            noise_records_by_item[item_uid] = {
                **record,
                "noise_slot_uid": structure.base_uid(
                    key_hex=id_key,
                    entity_kind="noise_slot",
                    parent_uid_or_mode=item_uid,
                    ordinal=0,
                ),
            }
        override_sellers = {
            row[key]
            for row in override_audit
            for key in ("seller_uid_left", "seller_uid_right")
        }
        override_items = {
            row[key]
            for row in override_audit
            for key in ("item_uid_left", "item_uid_right")
        }
        return {
            "override_audit_count": len(override_audit),
            "high_semantic_count": sum(
                row["override_kind"] == "high_semantic_similarity"
                for row in override_audit
            ),
            "exact_title_clone_count": sum(
                row["override_kind"] == "exact_title_clone"
                for row in override_audit
            ),
            "unique_override_seller_count": len(override_sellers),
            "unique_override_item_count": len(override_items),
            "noise_record_count": len(noise_records_by_item),
            "override_audit_sha256": common.canonical_sha256(override_audit),
            "noise_record_sha256": common.canonical_sha256(
                [
                    noise_records_by_item[item_uid]
                    for item_uid in common.utf8_sort(noise_records_by_item)
                ]
            ),
        }

    solved = identity_plan.solve_identity_plan(
        policy,
        mode=mode,
        split=split,
        world_uid=world_uid,
        world_mode_global_ordinal=int(world_record["mode_global_ordinal"]),
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
        mechanisms=mechanisms,
        items_by_seller=items_by_seller,
        pre_slot_callback=materialize_observed_overrides_and_noise,
    )
    if pre_slot_call_count != 1 or len(noise_records_by_item) != 28:
        raise common.ContractError("Pre-slot observed materialization did not close")

    identity_audit, identity_edit, noise_audit = _render_identity_slots(
        policy=policy,
        template=template,
        fixture=fixture,
        items_by_seller=items_by_seller,
        noise_records_by_item=noise_records_by_item,
    )
    items = [
        item
        for seller_uid in common.utf8_sort(items_by_seller)
        for item in sorted(
            items_by_seller[seller_uid],
            key=lambda row: row["item_uid"].encode("utf-8"),
        )
    ]
    observed_items = [
        {
            key: item[key]
            for key in (
                "world_uid",
                "seller_uid",
                "item_uid",
                "time_bucket",
                "category",
                "title",
                "description",
            )
        }
        for item in items
    ]
    render_asts = [
        {
            "world_uid": item["world_uid"],
            "seller_uid": item["seller_uid"],
            "item_uid": item["item_uid"],
            "time_bucket": item["time_bucket"],
            "category": item["category"],
            "product": item["product"],
            "attribute": item["attribute"],
            "delivery": item["delivery"],
            "service": item["service"],
            "code": item["code"],
            "title_skeleton_index": item["title_skeleton_index"],
            "description_skeleton_index": item["description_skeleton_index"],
            "effective_style_uid": item["effective_style_uid"],
            "title_nonempty": item["title_nonempty"],
            "description_nonempty": item["description_nonempty"],
            "identity_slot_uids": [
                row["slot_uid"]
                for row in sorted(
                    item["identity_slots"],
                    key=lambda row: row["slot_uid"].encode("utf-8"),
                )
            ],
            "noise_slot_uid": noise_records_by_item.get(
                item["item_uid"], {}
            ).get("noise_slot_uid", ""),
        }
        for item in items
    ]
    controller_membership = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "seller_uid": seller_uid,
        }
        for controller_uid in common.utf8_sort(membership["controller_members"])
        for seller_uid in membership["controller_members"][controller_uid]
    ]
    controller_style_groups = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "style_id": str(controller_styles[controller_uid]["style_id"]),
        }
        for controller_uid in common.utf8_sort(controller_styles)
    ]
    mechanism_rows = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            **mechanisms[controller_uid],
        }
        for controller_uid in common.utf8_sort(mechanisms)
    ]
    return {
        "public": {
            "world": {"world_uid": world_uid},
            "sellers": [
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "market": markets[seller_uid],
                }
                for seller_uid in common.utf8_sort(membership["seller_uids"])
            ],
            "items": observed_items,
            "complete_model_pair_endpoints": _complete_pair_endpoints(
                world_uid, membership["seller_uids"]
            ),
        },
        "private": {
            "controller_membership": controller_membership,
            "controller_style_groups": controller_style_groups,
            "mechanism_assignments": mechanism_rows,
            "identity_assets": solved["assets"],
            "identity_slots_audit": identity_audit,
            "identity_slots_edit": identity_edit,
            "noise_slots_audit": noise_audit,
            "render_asts": render_asts,
            "positive_targets": solved["positive_targets"],
            "negative_flags": solved["negative_flags"],
            "override_audit": override_audit,
            "solver_audit": {
                "world_uid": world_uid,
                "split": split,
                "graph_name": graph_name,
                "market_proposal_counter": market_proposal,
                **solved["solver_audit"],
            },
        },
    }
