#!/usr/bin/env python3
"""Materialize one V9.3 method world without opening or returning pair truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_history_features as history_features
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_counterfactual_text_v9_3 as counterfactual
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_structure_matrix_v9_3 as structure_matrix
import step28_v13_v1_13_world_builder_v9_3 as world_builder


VERSION = "2026-08-26-step28-v13-v1-13-method-world-v9-3"
PAIR_KEY_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
ROLE_NAMES = (
    "clone_source",
    "clone_target",
    "semantic_left",
    "semantic_right",
)


class MethodWorldV93Error(common.ContractError):
    """Raised when a method-world scientific boundary does not close."""


def _pair_ordinal(left: int, right: int, *, width: int) -> int:
    if type(left) is not int or type(right) is not int or not 0 <= left < right < width:
        raise MethodWorldV93Error("Pair ordinal endpoints are malformed")
    return sum(width - first - 1 for first in range(left)) + right - left - 1


def _parser_plan_audit(
    *,
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots: Sequence[Mapping[str, Any]],
    noise_slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    items_by_uid = {str(row["item_uid"]): row for row in items}
    if len(items_by_uid) != len(items):
        raise MethodWorldV93Error("Parser audit item UID collision")

    def key_from_plan(row: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(row["seller_uid"]),
            str(row["item_uid"]),
            str(row["field_name"]),
            str(row["identity_type"]),
            str(row["downstream_canonical_value"]),
        )

    def key_from_parser(row: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(row["seller_uid"]),
            str(row["item_uid"]),
            str(row["source_field"]),
            str(row["contact_type"]),
            str(row["normalized_value"]),
        )

    planned = {key_from_plan(row): row for row in identity_slots}
    observed = {key_from_parser(row): row for row in parsed_rows}
    if (
        len(planned) != len(identity_slots)
        or len(observed) != len(parsed_rows)
        or set(planned) != set(observed)
    ):
        raise MethodWorldV93Error("Parser rows do not equal the identity-slot plan")
    flag_pairs = (
        ("seller_facing_context", "expected_seller_facing_context"),
        ("product_data_risk_context", "expected_product_data_risk_context"),
        ("direct_identity_eligible", "expected_direct_identity_eligible"),
        ("support_only", "expected_support_only"),
    )
    for key in planned:
        plan = planned[key]
        parsed = observed[key]
        item = items_by_uid.get(str(plan["item_uid"]))
        if item is None:
            raise MethodWorldV93Error("Planned parser item is absent")
        start = int(plan["start"])
        end = int(plan["end"])
        if (
            str(item["seller_uid"]) != str(plan["seller_uid"])
            or str(item["description"])[start:end] != str(plan["raw_surface"])
            or str(parsed["raw_value"]) != str(plan["raw_surface"])
            or any(int(parsed[left]) != int(plan[right]) for left, right in flag_pairs)
        ):
            raise MethodWorldV93Error("Parser value or role flags drifted")
    for row in noise_slots:
        surface = str(row["raw_surface"])
        item = items_by_uid.get(str(row["item_uid"]))
        if item is None or str(item["description"]).count(surface) != 1:
            raise MethodWorldV93Error("Noise surface does not occur exactly once")
        if any(
            str(parsed["raw_value"]) == surface
            and str(parsed["item_uid"]) == str(row["item_uid"])
            for parsed in parsed_rows
        ):
            raise MethodWorldV93Error("Must-ignore noise reached the parser output")
    return {
        "planned_identity_slot_count": len(identity_slots),
        "parsed_identity_row_count": len(parsed_rows),
        "noise_slot_count": len(noise_slots),
        "exact_rows_and_flags": True,
    }


def _expected_clean_items(
    *,
    template: Mapping[str, Any],
    split: str,
    items: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    noise_slots: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str]]:
    styles = {
        str(row["effective_style_uid"]): row
        for row in renderer.reachable_effective_styles(dict(template))
    }
    ast_by_item = {str(row["item_uid"]): row for row in render_asts}
    noise_by_item = {str(row["item_uid"]): str(row["raw_surface"]) for row in noise_slots}
    if len(ast_by_item) != len(items) or len(noise_by_item) != len(noise_slots):
        raise MethodWorldV93Error("Clean reconstruction private key collision")
    title_skeletons = template["split_libraries"][split]["title_skeletons"]
    description_skeletons = template["split_libraries"][split][
        "description_skeletons"
    ]
    expected: dict[str, tuple[str, str]] = {}
    for item in items:
        item_uid = str(item["item_uid"])
        ast = ast_by_item.get(item_uid)
        if ast is None:
            raise MethodWorldV93Error("Clean reconstruction AST is absent")
        style = styles[str(ast["effective_style_uid"])]
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
        description = (
            world_builder._render_code_free_description(
                skeleton=str(
                    description_skeletons[int(ast["description_skeleton_index"])]
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
        description += noise_by_item.get(item_uid, "")
        expected[item_uid] = (
            production.source.normalize_redacted_text(title),
            production.source.normalize_redacted_text(description),
        )
    for row in override_audit:
        if str(row["override_kind"]) == "exact_title_clone":
            source_uid = str(row["item_uid_left"])
            target_uid = str(row["item_uid_right"])
            source_title = expected[source_uid][0]
            expected[target_uid] = (source_title, expected[target_uid][1])
    return expected


def _process_surface(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    identity_slots: Sequence[Mapping[str, Any]],
    noise_slots: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parsed = production.parse_observed_world(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=items,
    )
    parser_audit = _parser_plan_audit(
        items=items,
        parsed_rows=parsed,
        identity_slots=identity_slots,
        noise_slots=noise_slots,
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
        items=items,
        registry_profiles=registry_profiles,
        parsed_rows=parsed,
    )
    expected_clean = _expected_clean_items(
        template=template,
        split=split,
        items=items,
        render_asts=render_asts,
        noise_slots=noise_slots,
        override_audit=override_audit,
    )
    for row in redaction["redacted_items"]:
        expected = expected_clean.get(str(row["item_uid"]))
        if expected is None or (str(row["title"]), str(row["description"])) != expected:
            raise MethodWorldV93Error("Redacted text is not the code-free clean replay")
    profile_safe_items = production.build_profile_safe_items(
        policy,
        items=items,
        redacted_items=redaction["redacted_items"],
    )
    profiles, profile_audit = profiles_module.build_world_profiles(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=profile_safe_items,
    )
    projected_profiles = scientific.project_model_seller_profiles(profiles)
    history_rows = production.project_history_safe_occurrences(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=items,
        parsed_rows=parsed,
    )
    return {
        "redacted_items": [dict(row) for row in redaction["redacted_items"]],
        "model_seller_profiles": [dict(row) for row in projected_profiles],
        "history_safe_occurrences": [dict(row) for row in history_rows],
        "parsed_identity_occurrences": [dict(row) for row in parsed],
        "audit": {
            "parser": parser_audit,
            "profile": profile_audit,
            "redacted_item_count": len(redaction["redacted_items"]),
            "history_safe_occurrence_count": len(history_rows),
        },
    }


def _identity33_without_truth(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    history_rows: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    history_times: dict[str, int] = {}
    for row in history_rows:
        item_uid = str(row["item_uid"])
        time_bucket = int(row["time_bucket"])
        if item_uid in history_times and history_times[item_uid] != time_bucket:
            raise MethodWorldV93Error(
                "One item has multiple identity-layer time buckets"
            )
        history_times[item_uid] = time_bucket
    history_item_index = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": history_times.get(
                str(row["item_uid"]), int(row["time_bucket"])
            ),
        }
        for row in items
    ]
    history_item_index.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
        )
    )
    feature_names, excluded_names = history_features._feature_contract(policy)
    pair_rows_by_world, sellers_by_world = history_features._validate_pair_rows(
        policy, endpoints
    )
    item_index = history_features._validate_history_item_index(
        policy,
        rows=history_item_index,
        sellers_by_world=sellers_by_world,
    )
    history_by_world = history_features._validate_history_rows(
        policy,
        mode=mode,
        split=split,
        rows=history_rows,
        sellers_by_world=sellers_by_world,
        item_index=item_index,
    )
    rows, audit = history_features._compute_identity33(
        policy,
        feature_names=feature_names,
        excluded_names=excluded_names,
        pair_rows_by_world=pair_rows_by_world,
        history_rows_by_world=history_by_world,
        history_safe_occurrence_count=len(history_rows),
        history_item_index_count=len(history_item_index),
    )
    return rows, {**audit, "truth_or_controller_membership_read": False}


def _symmetric_counts(
    left: Sequence[int], right: Sequence[int], *, prefix: str
) -> dict[str, int]:
    if len(left) != len(right):
        raise MethodWorldV93Error("Symmetric count vector width drift")
    output: dict[str, int] = {}
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        output[f"{prefix}_{index:02d}_absdiff"] = abs(left_value - right_value)
        output[f"{prefix}_{index:02d}_sum"] = left_value + right_value
    return output


def _structure_rows(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    split: str,
    world: Mapping[str, Any],
    candidate_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public = world["public"]
    private = world["private"]
    sellers = tuple(common.utf8_sort(str(row["seller_uid"]) for row in public["sellers"]))
    seller_slot = {seller: index for index, seller in enumerate(sellers)}
    market_names = tuple(str(value) for value in policy["world_design"]["markets"])
    if len(market_names) != len(set(market_names)):
        raise MethodWorldV93Error("Frozen market domain contains duplicates")
    market_index = {name: index for index, name in enumerate(market_names)}
    market_by_seller = {
        str(row["seller_uid"]): market_index[str(row["market"])]
        for row in public["sellers"]
    }
    ast_by_seller: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    noise_by_seller: dict[str, int] = {}
    for ast in private["render_asts"]:
        seller = str(ast["seller_uid"])
        ast_by_seller[seller].append(ast)
        noise = int(ast["noise_slot"])
        if seller in noise_by_seller and noise_by_seller[seller] != noise:
            raise MethodWorldV93Error("Seller spans multiple noise slots")
        noise_by_seller[seller] = noise
    if set(ast_by_seller) != set(sellers):
        raise MethodWorldV93Error("Structure AST seller universe drift")
    controller_size: dict[str, int] = {}
    members_by_controller: defaultdict[str, list[str]] = defaultdict(list)
    for row in private["controller_membership"]:
        members_by_controller[str(row["controller_uid"])].append(str(row["seller_uid"]))
    for members in members_by_controller.values():
        for seller in members:
            controller_size[seller] = len(members)
    if set(controller_size) != set(sellers):
        raise MethodWorldV93Error("Controller-size seller universe drift")
    treatment_by_pair: dict[str, int] = {}
    role_by_seller = {seller: [0, 0, 0, 0] for seller in sellers}
    registered_item_ordinal: dict[str, int] = {}
    for row in private["override_audit"]:
        treatment = str(row["override_kind"])
        treatment_by_pair[str(row["canonical_pair_uid"])] = (
            1 if treatment == "exact_title_clone" else 2
        )
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        if treatment == "exact_title_clone":
            roles = (0, 1)
        else:
            roles = (2, 3)
        role_by_seller[left][roles[0]] += 1
        role_by_seller[right][roles[1]] += 1
        registered_item_ordinal[left] = int(row["logical_item_ordinal_left"])
        registered_item_ordinal[right] = int(row["logical_item_ordinal_right"])
    if len(treatment_by_pair) != 6 or sum(map(sum, role_by_seller.values())) != 12:
        raise MethodWorldV93Error("Registered-treatment structure drift")

    lexicon = template["generic_lexicon"]
    if split not in template["split_libraries"]:
        raise MethodWorldV93Error("Structure split library is absent")
    categories = tuple(str(value) for value in lexicon["categories"])
    category_index = {value: index for index, value in enumerate(categories)}
    # The matrix schema is global and frozen before any world is generated.
    # Split-local libraries may have different lengths, so every split uses the
    # maximum registered width and leaves unused trailing ordinals at zero.
    title_width = max(
        len(library["title_skeletons"])
        for library in template["split_libraries"].values()
    )
    description_width = max(
        len(library["description_skeletons"])
        for library in template["split_libraries"].values()
    )
    service_values = tuple(str(value) for value in lexicon["service"])
    delivery_values = tuple(str(value) for value in lexicon["delivery"])
    if any(
        len(values) != len(set(values))
        for values in (categories, service_values, delivery_values)
    ):
        raise MethodWorldV93Error("Frozen visible-structure domain contains duplicates")

    def seller_vector(seller: str) -> dict[str, Any]:
        rows = ast_by_seller[seller]
        category = [0] * len(categories)
        title = [0] * title_width
        description = [0] * description_width
        service = [0] * len(service_values)
        delivery = [0] * len(delivery_values)
        time = [0] * 4
        for ast in rows:
            logical_ordinal = int(ast["logical_item_ordinal"])
            if not 0 <= logical_ordinal < 8:
                raise MethodWorldV93Error("Logical item ordinal is outside 0..7")
            category[category_index[str(ast["category"])]] += 1
            title[int(ast["title_skeleton_index"])] += 1
            description[int(ast["description_skeleton_index"])] += 1
            service[service_values.index(str(ast["service"]))] += 1
            delivery[delivery_values.index(str(ast["delivery"]))] += 1
            time[int(ast["time_bucket"])] += 1
        variations = [int(ast["natural_variation_ordinal"]) for ast in rows]
        logical_ordinals = sorted(int(ast["logical_item_ordinal"]) for ast in rows)
        if logical_ordinals != list(range(len(rows))):
            raise MethodWorldV93Error("Logical item ordinals are not contiguous")
        return {
            "item_count": len(rows),
            "title_present_mask": sum(
                (1 << int(ast["logical_item_ordinal"]))
                for ast in rows
                if bool(ast["title_nonempty"])
            ),
            "description_present_mask": sum(
                (1 << int(ast["logical_item_ordinal"]))
                for ast in rows
                if bool(ast["description_nonempty"])
            ),
            "joint_empty_mask": sum(
                (1 << int(ast["logical_item_ordinal"]))
                for ast in rows
                if not bool(ast["title_nonempty"])
                and not bool(ast["description_nonempty"])
            ),
            "category": category,
            "title_template": title,
            "description_template": description,
            "service": service,
            "delivery": delivery,
            "time_bucket": time,
            "variation_min": min(variations),
            "variation_max": max(variations),
        }

    seller_vectors = {seller: seller_vector(seller) for seller in sellers}
    seller_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    for endpoint in public["complete_model_pair_endpoints"]:
        left = str(endpoint["seller_uid_left"])
        right = str(endpoint["seller_uid_right"])
        left_slot, right_slot = sorted((seller_slot[left], seller_slot[right]))
        left_noise, right_noise = sorted((noise_by_seller[left], noise_by_seller[right]))
        pair_uid = str(endpoint["canonical_pair_uid"])
        base = {name: endpoint[name] for name in PAIR_KEY_FIELDS}
        seller_rows.append(
            {
                **base,
                "seller_pair_ordinal": _pair_ordinal(left_slot, right_slot, width=28),
                "seller_slot_min": left_slot,
                "seller_slot_max": right_slot,
                "seller_slot_diff": right_slot - left_slot,
                "seller_slot_sum": right_slot + left_slot,
            }
        )
        left_vector = seller_vectors[left]
        right_vector = seller_vectors[right]
        left_size, right_size = sorted((controller_size[left], controller_size[right]))
        left_market, right_market = sorted((market_by_seller[left], market_by_seller[right]))
        row = {
            **base,
            "noise_pair_ordinal": _pair_ordinal(left_noise, right_noise, width=28),
            "noise_slot_min": left_noise,
            "noise_slot_max": right_noise,
            "noise_slot_diff": right_noise - left_noise,
            "noise_slot_sum": right_noise + left_noise,
            "item_count_absdiff": abs(left_vector["item_count"] - right_vector["item_count"]),
            "item_count_sum": left_vector["item_count"] + right_vector["item_count"],
            "controller_size_min": left_size,
            "controller_size_max": right_size,
            "controller_size_diff": right_size - left_size,
            "controller_size_sum": right_size + left_size,
            "market_pair_ordinal": left_market * len(market_names) + right_market,
            "candidate_index": candidate_index,
            "collision_fallback_type": 0,
            "registered_treatment": treatment_by_pair.get(pair_uid, 0),
            "registered_endpoint_count": int(any(role_by_seller[left]))
            + int(any(role_by_seller[right])),
            "registered_clone_endpoint_count": sum(role_by_seller[left][:2])
            + sum(role_by_seller[right][:2]),
            "registered_semantic_endpoint_count": sum(role_by_seller[left][2:])
            + sum(role_by_seller[right][2:]),
            "registered_item_ordinal_min": min(
                registered_item_ordinal.get(left, -1),
                registered_item_ordinal.get(right, -1),
            ),
            "registered_item_ordinal_max": max(
                registered_item_ordinal.get(left, -1),
                registered_item_ordinal.get(right, -1),
            ),
            "variation_min_absdiff": abs(
                left_vector["variation_min"] - right_vector["variation_min"]
            ),
            "variation_min_sum": left_vector["variation_min"] + right_vector["variation_min"],
            "variation_max_absdiff": abs(
                left_vector["variation_max"] - right_vector["variation_max"]
            ),
            "variation_max_sum": left_vector["variation_max"] + right_vector["variation_max"],
            **_symmetric_counts(role_by_seller[left], role_by_seller[right], prefix="role"),
        }
        for mask_name in (
            "title_present_mask",
            "description_present_mask",
            "joint_empty_mask",
        ):
            left_mask, right_mask = sorted(
                (int(left_vector[mask_name]), int(right_vector[mask_name]))
            )
            row[f"{mask_name}_min"] = left_mask
            row[f"{mask_name}_max"] = right_mask
            row[f"{mask_name}_diff"] = right_mask - left_mask
            row[f"{mask_name}_sum"] = right_mask + left_mask
        for name in (
            "category",
            "title_template",
            "description_template",
            "service",
            "delivery",
            "time_bucket",
        ):
            row.update(
                _symmetric_counts(
                    left_vector[name], right_vector[name], prefix=name
                )
            )
        noise_rows.append(row)
    seller_rows.sort(key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"))
    noise_rows.sort(key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"))
    if len(seller_rows) != 378 or len(noise_rows) != 378:
        raise MethodWorldV93Error("Structure row cardinality drift")
    structure_matrix.validate_world_rows(
        seller_rows,
        raw_fields=structure_matrix.SELLER_SLOT_RAW_FIELDS,
        label="seller-slot",
    )
    structure_matrix.validate_world_rows(
        noise_rows,
        raw_fields=structure_matrix.NOISE_VISIBLE_RAW_FIELDS,
        label="noise-visible",
    )
    return seller_rows, noise_rows


def build_method_world(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    style_profile: dict[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    structure_key_hex: str,
    balanced_schedule: Mapping[str, Any] | None,
    registered_negative_plan: Mapping[str, Any] | None,
    joint_signatures: Mapping[str, Any],
    blind_audit_design: Mapping[str, Any] | None = None,
    candidate_index: int = 0,
) -> dict[str, Any]:
    """Build all label-free inputs for one method world; truth stays absent."""

    split = str(world_record["split"])
    world_ordinal = int(world_record["split_ordinal"])
    world = world_builder.build_world(
        policy=policy,
        template=template,
        fixture=fixture,
        style_profile=style_profile,
        mode=mode,
        world_record=world_record,
        structure_key_hex=structure_key_hex,
        balanced_schedule=balanced_schedule,
        registered_negative_plan=registered_negative_plan,
        joint_signatures=joint_signatures,
        blind_audit_design=blind_audit_design,
        candidate_index=candidate_index,
    )
    original = _process_surface(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        sellers=world["public"]["sellers"],
        items=world["public"]["items"],
        render_asts=world["private"]["render_asts"],
        identity_slots=world["private"]["identity_slots_audit"],
        noise_slots=world["private"]["noise_slots_audit"],
        override_audit=world["private"]["override_audit"],
    )
    counterfactual_world = counterfactual.rerender_counterfactual_world(
        policy=policy,
        template=template,
        split=split,
        world_ordinal=world_ordinal,
        world=world,
    )
    deranged = _process_surface(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        sellers=world["public"]["sellers"],
        items=counterfactual_world["public_items"],
        render_asts=counterfactual_world["private_render_asts"],
        identity_slots=counterfactual_world["private_identity_slots_audit"],
        noise_slots=counterfactual_world["private_noise_slots_audit"],
        override_audit=world["private"]["override_audit"],
    )
    if common.canonical_json_bytes(original["history_safe_occurrences"]) != common.canonical_json_bytes(
        deranged["history_safe_occurrences"]
    ):
        raise MethodWorldV93Error("Style derangement changed identity history")
    identity33, identity33_audit = _identity33_without_truth(
        policy=policy,
        mode=mode,
        split=split,
        history_rows=original["history_safe_occurrences"],
        items=world["public"]["items"],
        endpoints=world["public"]["complete_model_pair_endpoints"],
    )
    noise_time_counterfactual = [
        {
            **dict(row),
            "time_bucket": (int(row["time_bucket"]) + 1) % 4,
        }
        for row in world["public"]["items"]
    ]
    counterfactual_identity33, _counterfactual_identity33_audit = (
        _identity33_without_truth(
            policy=policy,
            mode=mode,
            split=split,
            history_rows=original["history_safe_occurrences"],
            items=noise_time_counterfactual,
            endpoints=world["public"]["complete_model_pair_endpoints"],
        )
    )
    if common.canonical_json_bytes(identity33) != common.canonical_json_bytes(
        counterfactual_identity33
    ):
        raise MethodWorldV93Error(
            "Noise-time intervention changed the identity33 projection"
        )
    seller_matrix, noise_matrix = _structure_rows(
        policy=policy,
        template=template,
        split=split,
        world=world,
        candidate_index=candidate_index,
    )
    forbidden_surfaces: list[str] = []

    def collect_strings(value: Any) -> None:
        if isinstance(value, str):
            forbidden_surfaces.append(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                collect_strings(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_strings(child)

    for surface in (
        original["redacted_items"],
        original["model_seller_profiles"],
        deranged["redacted_items"],
        deranged["model_seller_profiles"],
    ):
        collect_strings(surface)
    if any(world_builder.ARTIFICIAL_CODE_PATTERN.search(value) for value in forbidden_surfaces):
        raise MethodWorldV93Error("Artificial code reached a model surface")
    return {
        "version": VERSION,
        "world_uid": str(world["public"]["world"]["world_uid"]),
        "split": split,
        "split_ordinal": world_ordinal,
        "candidate_index": candidate_index,
        "public": {
            "sellers": [dict(row) for row in world["public"]["sellers"]],
            "complete_pair_endpoints": [
                dict(row) for row in world["public"]["complete_model_pair_endpoints"]
            ],
            "original_redacted_items": original["redacted_items"],
            "original_model_seller_profiles": original["model_seller_profiles"],
            "deranged_redacted_items": deranged["redacted_items"],
            "deranged_model_seller_profiles": deranged["model_seller_profiles"],
            "identity33": identity33,
            "seller_slot_structure_rows": seller_matrix,
            "noise_visible_structure_rows": noise_matrix,
        },
        "private_without_truth": {
            "controller_membership": [
                dict(row) for row in world["private"]["controller_membership"]
            ],
            "override_audit": [dict(row) for row in world["private"]["override_audit"]],
            "render_asts": [dict(row) for row in world["private"]["render_asts"]],
            "identity_slots_audit": [
                dict(row) for row in world["private"]["identity_slots_audit"]
            ],
            "noise_slots_audit": [
                dict(row) for row in world["private"]["noise_slots_audit"]
            ],
            "identity_assets": [dict(row) for row in world["private"]["identity_assets"]],
            "mechanism_assignments": [
                dict(row) for row in world["private"]["mechanism_assignments"]
            ],
        },
        "audit": {
            "original_surface": original["audit"],
            "counterfactual_surface": deranged["audit"],
            "counterfactual_intervention": counterfactual_world["audit"],
            "identity33": identity33_audit,
            "noise_time_counterfactual_identity33_unchanged": True,
            "artificial_code_occurrence_count": 0,
            "truth_materialized": False,
        },
    }


def materialize_private_truth(
    *,
    world_uid: str,
    controller_membership: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create labels and qrels only after all label-free matrices are frozen."""

    seller_controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in controller_membership
    }
    if len(seller_controller) != 28:
        raise MethodWorldV93Error("Truth controller membership is incomplete")
    members: defaultdict[str, list[str]] = defaultdict(list)
    for seller, controller in seller_controller.items():
        members[controller].append(seller)
    labels: list[dict[str, Any]] = []
    for row in endpoints:
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        labels.append(
            {
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "world_uid": world_uid,
                "label": int(seller_controller[left] == seller_controller[right]),
            }
        )
    qrels = []
    for seller in common.utf8_sort(seller_controller):
        relevant = common.utf8_sort(
            value for value in members[seller_controller[seller]] if value != seller
        )
        qrels.append(
            {
                "world_uid": world_uid,
                "query_uid": common.query_uid(world_uid, seller),
                "query_seller_uid": seller,
                "relevant_seller_uids": relevant,
            }
        )
    if len(labels) != 378 or sum(row["label"] for row in labels) != 20 or len(qrels) != 28:
        raise MethodWorldV93Error("Private truth cardinality drift")
    return {"pair_labels": labels, "qrels": qrels}
