#!/usr/bin/env python3
"""Build one code-free V9.3 world from frozen seller/noise/negative plans."""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_identity_plan as identity_plan
import step28_v13_nonidentity as nonidentity
import step28_v13_structure as structure
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_audit_design_v9_3 as audit_design_module
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as joint_noise
import step28_v13_v1_13_registered_negative_plan_v9_3 as negative_plan


VERSION = "2026-08-26-step28-v13-v1-13-world-builder-v9-3"
ARTIFICIAL_CODE_PATTERN = re.compile(r"Q[A-P]{10}")
SPLIT_VARIATION_ORDINAL = {
    "train": 0,
    "development": 1,
    "audit_a": 2,
    "audit_b": 3,
}
MAXIMUM_WORLDS_PER_SPLIT = 500
MAXIMUM_LOGICAL_ITEMS_PER_NOISE_SLOT = 8
MAXIMUM_CANDIDATES = 32
NATURAL_VARIATION_RADIX = 64
NATURAL_VARIATION_CAPACITY = NATURAL_VARIATION_RADIX**4
NATURAL_VARIATION_SEGMENTS = (
    tuple(
        left + right
        for left in ("居家", "办公", "出行", "日常", "批量", "临时", "长期", "灵活")
        for right in ("自用", "补货", "换新", "周转", "搭配", "备选", "常备", "体验")
    ),
    tuple(
        left + right
        for left in ("标准", "精选", "稳定", "实用", "耐用", "便捷", "清爽", "灵活")
        for right in ("配置", "组合", "规格", "版本", "方案", "套装", "选项", "款式")
    ),
    tuple(
        left + right
        for left in ("简约", "稳妥", "常规", "细致", "轻便", "独立", "整洁", "加固")
        for right in ("包装", "封装", "整理", "检查", "配套", "防护", "装箱", "核对")
    ),
    tuple(
        left + right
        for left in ("及时", "耐心", "清晰", "主动", "细致", "灵活", "稳妥", "持续")
        for right in ("答疑", "跟进", "沟通", "确认", "协助", "响应", "说明", "反馈")
    ),
)


FLAG_NAMES = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)


def _validate_registered_plan(
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either the closed strict format or the V9.3-R2 successor."""

    if plan.get("version") == negative_plan.BOUNDED_RESIDUAL_VERSION:
        audit = negative_plan.validate_plan(
            plan,
            schedule,
            joint_signatures,
            expected_version=negative_plan.BOUNDED_RESIDUAL_VERSION,
            require_exact_balance=False,
            success_status=negative_plan.BOUNDED_RESIDUAL_STATUS,
        )
        if audit.get("role_eligibility_predicates") != list(
            negative_plan.ROLE_ELIGIBILITY_PREDICATE_NAMES
        ):
            raise common.ContractError("V9.3-R2 role-eligibility authority drift")
        return audit
    return negative_plan.validate_plan(plan, schedule, joint_signatures)


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


def _design_index(key_hex: str, modulus: int, *atoms: object) -> int:
    if modulus <= 0:
        raise common.ContractError("V9.3 design modulus must be positive")
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise common.ContractError("V9.3 design key is not hexadecimal") from exc
    payload = common.FIELD_SEPARATOR.join(
        str(atom).encode("utf-8") for atom in (VERSION, *atoms)
    )
    return int.from_bytes(hmac.new(key, payload, hashlib.sha256).digest(), "big") % modulus


def _normalize_code_free_text(value: str) -> str:
    output = re.sub(r"[ \t]+", " ", value)
    output = re.sub(r" +([，。；！、｜])", r"\1", output)
    output = re.sub(r"([（【]) +", r"\1", output)
    output = re.sub(r" +([）】])", r"\1", output)
    return output.strip()


def _natural_variation(
    *,
    split: str,
    world_ordinal: int,
    noise_slot: int,
    logical_item_ordinal: int,
    candidate_index: int,
    render_phrase: bool = True,
) -> tuple[int, str]:
    if type(render_phrase) is not bool:
        raise common.ContractError("V9.3 natural-variation phrase control drift")
    if split not in SPLIT_VARIATION_ORDINAL:
        raise common.ContractError("V9.3 natural-variation split is invalid")
    coordinates = (
        (world_ordinal, MAXIMUM_WORLDS_PER_SPLIT, "world"),
        (noise_slot, 28, "noise slot"),
        (
            logical_item_ordinal,
            MAXIMUM_LOGICAL_ITEMS_PER_NOISE_SLOT,
            "logical item",
        ),
        (candidate_index, MAXIMUM_CANDIDATES, "candidate"),
    )
    for value, upper, label in coordinates:
        if type(value) is not int or not 0 <= value < upper:
            raise common.ContractError(
                f"V9.3 natural-variation {label} coordinate is invalid"
            )
    ordinal = SPLIT_VARIATION_ORDINAL[split]
    for value, upper, _label in coordinates:
        ordinal = ordinal * upper + value
    if not 0 <= ordinal < NATURAL_VARIATION_CAPACITY:
        raise common.ContractError("V9.3 natural-variation capacity exhausted")
    remainder = ordinal
    digits: list[int] = []
    for _segment in NATURAL_VARIATION_SEGMENTS:
        digits.append(remainder % NATURAL_VARIATION_RADIX)
        remainder //= NATURAL_VARIATION_RADIX
    if remainder:
        raise common.ContractError("V9.3 natural-variation encoding overflow")
    phrase = (
        "、".join(
            segment[digit]
            for segment, digit in zip(
                NATURAL_VARIATION_SEGMENTS, digits, strict=True
            )
        )
        if render_phrase
        else ""
    )
    return ordinal, phrase


def _logical_render_selector(
    *,
    split: str,
    world_ordinal: int,
    noise_slot: int,
    logical_item_ordinal: int,
) -> str:
    if split not in {"train", "development", "audit_a", "audit_b"}:
        raise common.ContractError("V9.3 logical render split is invalid")
    if world_ordinal < 0 or not 0 <= noise_slot < 28 or logical_item_ordinal < 0:
        raise common.ContractError("V9.3 logical render coordinate is invalid")
    return (
        f"v9_3_render::{split}::{world_ordinal:03d}::"
        f"noise_{noise_slot:02d}::item_{logical_item_ordinal:02d}"
    )


def _render_code_free_title(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    style: Mapping[str, Any],
    template: Mapping[str, Any],
    natural_variation_phrase: str,
) -> str:
    output = skeleton.format(
        product=product,
        attribute=attribute,
        title_modifier="",
        code="",
    )
    output = renderer._transform_base(
        _normalize_code_free_text(
            f"{output}（{natural_variation_phrase}）"
        ),
        style=style,
        template=template,
        description=False,
    )
    tag = str(style["english_tag"])
    if tag:
        output = f"{output} {tag}"
    if ARTIFICIAL_CODE_PATTERN.search(output):
        raise common.ContractError("Artificial code reached a V9.3 title")
    return output


def _render_code_free_description(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    delivery: str,
    service: str,
    style: Mapping[str, Any],
    template: Mapping[str, Any],
    natural_variation_phrase: str,
) -> str:
    if not skeleton.endswith(renderer.DESCRIPTION_SUFFIX):
        raise common.ContractError("V9.3 description skeleton suffix drift")
    separator, ending = renderer._style_values(style)
    output = skeleton.format(
        product=product,
        attribute=attribute,
        code="",
        delivery=delivery,
        service=service,
        separator=separator,
        ending=ending,
        noise_clause="",
        context_guard="",
        identity_clause="",
    )
    output = renderer._transform_base(
        _normalize_code_free_text(
            f"{output}{separator}商品说明：{natural_variation_phrase}{ending}"
        ),
        style=style,
        template=template,
        description=True,
    )
    if ARTIFICIAL_CODE_PATTERN.search(output):
        raise common.ContractError("Artificial code reached a V9.3 description")
    return output


def _planned_membership(
    *,
    base: Mapping[str, Any],
    schedule_world: Mapping[str, Any],
    world_ordinal: int,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    groups, noise_by_seller = balanced._validate_world(
        schedule_world, expected_ordinal=world_ordinal
    )
    sellers = tuple(common.utf8_sort(base["seller_uids"]))
    controllers = tuple(common.utf8_sort(base["controller_uids"]))
    if len(sellers) != 28 or len(controllers) != 12:
        raise common.ContractError("V9.3 membership UID universe drift")
    controller_members: dict[str, list[str]] = {}
    seller_to_controller: dict[str, str] = {}
    for controller_uid, group in zip(controllers, groups, strict=True):
        members = [sellers[slot] for slot in group]
        controller_members[controller_uid] = common.utf8_sort(members)
        for seller_uid in members:
            seller_to_controller[seller_uid] = controller_uid
    if set(seller_to_controller) != set(sellers):
        raise common.ContractError("V9.3 planned membership is not a partition")
    return {
        "controller_uids": list(controllers),
        "seller_uids": list(sellers),
        "controller_partition_order": list(controllers),
        "seller_partition_order": list(sellers),
        "controller_members": controller_members,
        "seller_to_controller": seller_to_controller,
    }, noise_by_seller


def _logical_nonidentity_assignments(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    split: str,
    world_ordinal: int,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    int,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Assign visible nuisance variables from logical slots, never private UIDs."""

    actual_sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    actual_controllers = tuple(common.utf8_sort(membership["controller_uids"]))
    if len(actual_sellers) != 28 or len(actual_controllers) != 12:
        raise common.ContractError("V9.3 logical assignment universe drift")
    logical_sellers = tuple(f"logical_seller_{slot:02d}" for slot in range(28))
    logical_controllers = tuple(
        f"logical_controller_{slot:02d}" for slot in range(12)
    )
    actual_to_logical_seller = dict(
        zip(actual_sellers, logical_sellers, strict=True)
    )
    logical_to_actual_seller = {
        logical: actual for actual, logical in actual_to_logical_seller.items()
    }
    actual_to_logical_controller = dict(
        zip(actual_controllers, logical_controllers, strict=True)
    )
    logical_to_actual_controller = {
        logical: actual for actual, logical in actual_to_logical_controller.items()
    }
    logical_controller_members = {
        actual_to_logical_controller[controller_uid]: common.utf8_sort(
            actual_to_logical_seller[seller_uid]
            for seller_uid in membership["controller_members"][controller_uid]
        )
        for controller_uid in actual_controllers
    }
    logical_seller_to_controller = {
        seller_uid: controller_uid
        for controller_uid, members in logical_controller_members.items()
        for seller_uid in members
    }
    logical_membership = {
        "controller_uids": list(logical_controllers),
        "seller_uids": list(logical_sellers),
        "controller_partition_order": list(logical_controllers),
        "seller_partition_order": list(logical_sellers),
        "controller_members": logical_controller_members,
        "seller_to_controller": logical_seller_to_controller,
    }
    logical_world_uid = f"v9_3_logical_world_{split}_{world_ordinal:03d}"
    logical_markets, market_proposal = structure.assign_markets(
        policy,
        world_uid=logical_world_uid,
        structure_key_hex=structure_key_hex,
        membership=logical_membership,
    )
    logical_mechanisms = structure.assign_controller_mechanisms(
        policy,
        world_uid=logical_world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=logical_membership,
        markets=logical_markets,
    )
    logical_controller_styles = nonidentity.world_controller_styles(
        policy=dict(policy),
        template=dict(template),
        world_uid=logical_world_uid,
        structure_key_hex=structure_key_hex,
        controller_uids=list(logical_controllers),
        mode=mode,
    )
    logical_effective_styles: dict[str, dict[str, Any]] = {}
    for logical_seller_uid in logical_sellers:
        logical_controller_uid = logical_seller_to_controller[logical_seller_uid]
        logical_effective_styles[logical_seller_uid] = (
            nonidentity.seller_effective_style(
                policy=dict(policy),
                template=dict(template),
                mode=mode,
                seller_uid=logical_seller_uid,
                controller_style=logical_controller_styles[
                    logical_controller_uid
                ],
            )
        )
    markets = {
        logical_to_actual_seller[logical_seller_uid]: market
        for logical_seller_uid, market in logical_markets.items()
    }
    mechanisms = {
        logical_to_actual_controller[logical_controller_uid]: dict(value)
        for logical_controller_uid, value in logical_mechanisms.items()
    }
    controller_styles = {
        logical_to_actual_controller[logical_controller_uid]: dict(value)
        for logical_controller_uid, value in logical_controller_styles.items()
    }
    effective_styles = {
        logical_to_actual_seller[logical_seller_uid]: dict(value)
        for logical_seller_uid, value in logical_effective_styles.items()
    }
    if not (
        set(markets) == set(actual_sellers)
        and set(mechanisms) == set(actual_controllers)
        and set(controller_styles) == set(actual_controllers)
        and set(effective_styles) == set(actual_sellers)
    ):
        raise common.ContractError("V9.3 logical assignment remap did not close")
    return (
        markets,
        market_proposal,
        mechanisms,
        controller_styles,
        effective_styles,
    )


def _solve_identity_plan_logically(
    *,
    policy: dict[str, Any],
    mode: str,
    split: str,
    world_uid: str,
    world_ordinal: int,
    world_mode_global_ordinal: int,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
    mechanisms: Mapping[str, Mapping[str, Any]],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    pre_slot_callback: Callable[
        [Sequence[Mapping[str, Any]]], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Solve identity placement on logical coordinates and remap only row links."""

    actual_sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    actual_controllers = tuple(common.utf8_sort(membership["controller_uids"]))
    if len(actual_sellers) != 28 or len(actual_controllers) != 12:
        raise common.ContractError("V9.3 logical identity universe drift")
    logical_sellers = tuple(f"logical_seller_{slot:02d}" for slot in range(28))
    logical_controllers = tuple(
        f"logical_controller_{slot:02d}" for slot in range(12)
    )
    actual_to_logical_seller = dict(
        zip(actual_sellers, logical_sellers, strict=True)
    )
    logical_to_actual_seller = {
        logical: actual for actual, logical in actual_to_logical_seller.items()
    }
    actual_to_logical_controller = dict(
        zip(actual_controllers, logical_controllers, strict=True)
    )
    logical_to_actual_controller = {
        logical: actual for actual, logical in actual_to_logical_controller.items()
    }
    logical_controller_members = {
        actual_to_logical_controller[actual_controller]: common.utf8_sort(
            actual_to_logical_seller[actual_seller]
            for actual_seller in membership["controller_members"][actual_controller]
        )
        for actual_controller in actual_controllers
    }
    logical_seller_to_controller = {
        seller: controller
        for controller, members in logical_controller_members.items()
        for seller in members
    }
    logical_membership = {
        "controller_uids": list(logical_controllers),
        "seller_uids": list(logical_sellers),
        "controller_partition_order": list(logical_controllers),
        "seller_partition_order": list(logical_sellers),
        "controller_members": logical_controller_members,
        "seller_to_controller": logical_seller_to_controller,
    }
    logical_markets = {
        actual_to_logical_seller[actual_seller]: markets[actual_seller]
        for actual_seller in actual_sellers
    }
    logical_mechanisms = {
        actual_to_logical_controller[actual_controller]: dict(
            mechanisms[actual_controller]
        )
        for actual_controller in actual_controllers
    }
    logical_world_uid = f"v9_3_logical_world_{split}_{world_ordinal:03d}"
    logical_items_by_seller: dict[str, list[dict[str, Any]]] = {}
    logical_to_actual_item: dict[str, str] = {}
    logical_item_objects: dict[str, dict[str, Any]] = {}
    actual_item_objects: dict[str, dict[str, Any]] = {}
    for seller_slot, actual_seller in enumerate(actual_sellers):
        logical_seller = actual_to_logical_seller[actual_seller]
        actual_items = sorted(
            items_by_seller[actual_seller],
            key=lambda row: int(row["logical_item_ordinal"]),
        )
        logical_rows: list[dict[str, Any]] = []
        for expected_ordinal, actual_item in enumerate(actual_items):
            if int(actual_item["logical_item_ordinal"]) != expected_ordinal:
                raise common.ContractError(
                    "V9.3 logical identity item ordinals are not contiguous"
                )
            logical_item_uid = (
                f"logical_item_{seller_slot:02d}_{expected_ordinal:02d}"
            )
            logical_item = {
                **actual_item,
                "world_uid": logical_world_uid,
                "seller_uid": logical_seller,
                "item_uid": logical_item_uid,
                "identity_slots": [],
            }
            logical_rows.append(logical_item)
            logical_to_actual_item[logical_item_uid] = str(actual_item["item_uid"])
            logical_item_objects[logical_item_uid] = logical_item
            actual_item_objects[str(actual_item["item_uid"])] = actual_item
        logical_items_by_seller[logical_seller] = logical_rows
    exact_remap: dict[str, str] = {
        logical_world_uid: world_uid,
        **logical_to_actual_seller,
        **logical_to_actual_controller,
        **logical_to_actual_item,
    }
    for left_slot, right_slot in itertools.combinations(range(28), 2):
        logical_pair_uid = common.canonical_pair_uid(
            logical_sellers[left_slot], logical_sellers[right_slot]
        )
        exact_remap[logical_pair_uid] = common.canonical_pair_uid(
            actual_sellers[left_slot], actual_sellers[right_slot]
        )

    def remap(value: Any) -> Any:
        if isinstance(value, str):
            return exact_remap.get(value, value)
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(remap(item) for item in value)
        if isinstance(value, dict):
            output: dict[Any, Any] = {}
            for key, item in value.items():
                mapped_key = exact_remap.get(key, key) if isinstance(key, str) else key
                if mapped_key in output:
                    raise common.ContractError(
                        "V9.3 logical identity remap produced a duplicate key"
                    )
                output[mapped_key] = remap(item)
            return output
        return value

    solved = identity_plan.solve_identity_plan(
        policy,
        mode=mode,
        split=split,
        world_uid=logical_world_uid,
        world_mode_global_ordinal=world_mode_global_ordinal,
        structure_key_hex=structure_key_hex,
        membership=logical_membership,
        markets=logical_markets,
        mechanisms=logical_mechanisms,
        items_by_seller=logical_items_by_seller,
        pre_slot_callback=pre_slot_callback,
    )
    for logical_item_uid, logical_item in logical_item_objects.items():
        actual_item = actual_item_objects[logical_to_actual_item[logical_item_uid]]
        actual_item["identity_slots"] = remap(logical_item["identity_slots"])
    remapped = remap(solved)
    if any(
        str(row.get("seller_uid", "")) in logical_sellers
        or str(row.get("item_uid", "")) in logical_to_actual_item
        for row in remapped["slots"]
    ):
        raise common.ContractError("V9.3 logical identity row-link remap did not close")
    return remapped


def _planned_items(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    split: str,
    world_uid: str,
    world_ordinal: int,
    candidate_index: int,
    membership: Mapping[str, Any],
    noise_by_seller: tuple[int, ...],
    joint_signatures: Mapping[str, Any],
    effective_styles: Mapping[str, Mapping[str, Any]],
    render_text: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    if type(render_text) is not bool:
        raise common.ContractError("V9.3 render-text control drift")
    joint_noise.validate_payload(joint_signatures)
    signature_by_noise = {
        int(row["noise_slot"]): row["signature"]
        for row in joint_signatures["noise_slot_multiset"]
    }
    if set(signature_by_noise) != set(range(28)):
        raise common.ContractError("V9.3 joint-noise slot universe drift")
    sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    if set(effective_styles) != set(sellers):
        raise common.ContractError("V9.3 effective-style seller universe drift")
    text_key = str(policy["randomness"][mode]["text_key_hex"])
    id_key = str(policy["randomness"][mode]["id_key_hex"])
    lexicon = template["generic_lexicon"]
    categories = tuple(lexicon["categories"])
    attributes = tuple(lexicon["attributes"])
    deliveries = tuple(lexicon["delivery"])
    services = tuple(lexicon["service"])
    library = template["split_libraries"][split]
    title_skeletons = tuple(library["title_skeletons"])
    description_skeletons = tuple(library["description_skeletons"])
    output: dict[str, list[dict[str, Any]]] = {}
    for seller_slot, seller_uid in enumerate(sellers):
        noise_slot = int(noise_by_seller[seller_slot])
        signature = signature_by_noise[noise_slot]
        item_count = int(signature["item_count"])
        title_mask = str(signature["title_present_mask"])
        description_mask = str(signature["description_present_mask"])
        joint_empty_mask = str(signature["joint_empty_mask"])
        if not (
            len(title_mask)
            == len(description_mask)
            == len(joint_empty_mask)
            == item_count
        ):
            raise common.ContractError("V9.3 joint-noise mask length drift")
        style = dict(effective_styles[seller_uid])
        rows: list[dict[str, Any]] = []
        for logical_ordinal in range(item_count):
            selector = (
                split,
                world_ordinal,
                noise_slot,
                logical_ordinal,
                candidate_index,
            )
            variation_ordinal, variation_phrase = _natural_variation(
                split=split,
                world_ordinal=world_ordinal,
                noise_slot=noise_slot,
                logical_item_ordinal=logical_ordinal,
                candidate_index=candidate_index,
                render_phrase=render_text,
            )
            category = categories[
                _design_index(text_key, len(categories), *selector, "category")
            ]
            products = tuple(lexicon["category_products"][category])
            product = products[
                _design_index(text_key, len(products), *selector, "product")
            ]
            attribute = attributes[
                _design_index(text_key, len(attributes), *selector, "attribute")
            ]
            delivery = deliveries[
                _design_index(text_key, len(deliveries), *selector, "delivery")
            ]
            service = services[
                _design_index(text_key, len(services), *selector, "service")
            ]
            title_index = _design_index(
                text_key, len(title_skeletons), *selector, "title-skeleton"
            )
            description_index = _design_index(
                text_key,
                len(description_skeletons),
                *selector,
                "description-skeleton",
            )
            item_uid = structure.base_uid(
                key_hex=id_key,
                entity_kind="item",
                parent_uid_or_mode=seller_uid,
                ordinal=logical_ordinal,
            )
            title_nonempty = title_mask[logical_ordinal] == "1"
            description_nonempty = description_mask[logical_ordinal] == "1"
            if (not title_nonempty and not description_nonempty) != (
                joint_empty_mask[logical_ordinal] == "1"
            ):
                raise common.ContractError("V9.3 joint-empty mask drift")
            title = (
                _render_code_free_title(
                    skeleton=title_skeletons[title_index],
                    product=product,
                    attribute=attribute,
                    style=style,
                    template=template,
                    natural_variation_phrase=variation_phrase,
                )
                if title_nonempty and render_text
                else ""
            )
            base_description = (
                _render_code_free_description(
                    skeleton=description_skeletons[description_index],
                    product=product,
                    attribute=attribute,
                    delivery=delivery,
                    service=service,
                    style=style,
                    template=template,
                    natural_variation_phrase=variation_phrase,
                )
                if description_nonempty and render_text
                else ""
            )
            rows.append(
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "item_uid": item_uid,
                    "item_ordinal": logical_ordinal,
                    "logical_item_ordinal": logical_ordinal,
                    "seller_slot": seller_slot,
                    "noise_slot": noise_slot,
                    "time_bucket": _design_index(
                        text_key, 4, *selector, "time-bucket"
                    ),
                    "category": category,
                    "product": product,
                    "attribute": attribute,
                    "delivery": delivery,
                    "service": service,
                    "title_skeleton_index": title_index,
                    "description_skeleton_index": description_index,
                    "natural_variation_ordinal": variation_ordinal,
                    "natural_variation_phrase": variation_phrase,
                    "effective_style_uid": style["effective_style_uid"],
                    "effective_style": style,
                    "title_nonempty": title_nonempty,
                    "description_nonempty": description_nonempty,
                    "title": title,
                    "base_description": base_description,
                    "noise_clause": "",
                    "identity_slots": [],
                }
            )
        output[seller_uid] = rows
    return output


def _rerender_code_free_item(
    item: dict[str, Any], *, template: Mapping[str, Any], split: str
) -> None:
    library = template["split_libraries"][split]
    if item["title_nonempty"]:
        item["title"] = _render_code_free_title(
            skeleton=library["title_skeletons"][item["title_skeleton_index"]],
            product=item["product"],
            attribute=item["attribute"],
            style=item["effective_style"],
            template=template,
            natural_variation_phrase=item["natural_variation_phrase"],
        )
    if item["description_nonempty"]:
        item["base_description"] = _render_code_free_description(
            skeleton=library["description_skeletons"][
                item["description_skeleton_index"]
            ],
            product=item["product"],
            attribute=item["attribute"],
            delivery=item["delivery"],
            service=item["service"],
            style=item["effective_style"],
            template=template,
            natural_variation_phrase=item["natural_variation_phrase"],
        )


def _apply_registered_overrides(
    *,
    template: Mapping[str, Any],
    split: str,
    world_uid: str,
    world_ordinal: int,
    membership: Mapping[str, Any],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    plan_world: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    render_text: bool = True,
) -> list[dict[str, Any]]:
    if type(render_text) is not bool:
        raise common.ContractError("V9.3 override render-text control drift")
    joint_noise.validate_payload(joint_signatures)
    signature_by_noise = {
        int(row["noise_slot"]): row["signature"]
        for row in joint_signatures["noise_slot_multiset"]
    }
    sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    if (
        plan_world.get("world_ordinal") != world_ordinal
        or not isinstance(plan_world.get("assignments"), list)
        or len(plan_world["assignments"]) != 6
    ):
        raise common.ContractError("V9.3 registered-negative world binding drift")
    category_values = tuple(template["generic_lexicon"]["categories"])
    attributes = tuple(template["generic_lexicon"]["attributes"])
    title_skeletons = tuple(template["split_libraries"][split]["title_skeletons"])
    used_sellers: set[int] = set()
    audit: list[dict[str, Any]] = []
    for assignment in plan_world["assignments"]:
        treatment = str(assignment["treatment"])
        endpoints = assignment["endpoints"]
        if len(endpoints) != 2:
            raise common.ContractError("V9.3 registered-negative endpoint drift")
        endpoint_items: list[dict[str, Any]] = []
        endpoint_sellers: list[int] = []
        for endpoint in endpoints:
            seller_slot = int(endpoint["seller_slot"])
            logical_ordinal = int(endpoint["logical_item_ordinal"])
            role = str(endpoint["role"])
            if seller_slot in used_sellers:
                raise common.ContractError("V9.3 registered endpoint reused")
            used_sellers.add(seller_slot)
            seller_uid = sellers[seller_slot]
            try:
                item = items_by_seller[seller_uid][logical_ordinal]
            except IndexError as exc:
                raise common.ContractError(
                    "V9.3 registered logical item is absent"
                ) from exc
            if not item["title_nonempty"]:
                raise common.ContractError("V9.3 registered item lacks a title")
            eligible = negative_plan._role_eligible_logical_item_ordinals(
                treatment=treatment,
                role=role,
                signature=signature_by_noise[int(item["noise_slot"])],
            )
            if logical_ordinal not in eligible:
                raise common.ContractError(
                    "V9.3 registered item is not eligible for its named role"
                )
            endpoint_sellers.append(seller_slot)
            endpoint_items.append(item)
        if treatment == "exact_title_clone":
            if render_text:
                endpoint_items[1]["title"] = endpoint_items[0]["title"]
        elif treatment == "high_semantic_similarity":
            asset = assignment["semantic_asset"]
            category = category_values[int(asset["category_ordinal"])]
            products = tuple(template["generic_lexicon"]["category_products"][category])
            product = products[int(asset["product_ordinal"])]
            attribute = attributes[int(asset["attribute_ordinal"])]
            for item, skeleton_key in zip(
                endpoint_items,
                ("left_title_skeleton_ordinal", "right_title_skeleton_ordinal"),
                strict=True,
            ):
                item["category"] = category
                item["product"] = product
                item["attribute"] = attribute
                item["title_skeleton_index"] = int(asset[skeleton_key])
                if not 0 <= item["title_skeleton_index"] < len(title_skeletons):
                    raise common.ContractError("V9.3 semantic skeleton drift")
                if render_text:
                    _rerender_code_free_item(item, template=template, split=split)
        else:
            raise common.ContractError("Unknown V9.3 registered treatment")
        audit.append(
            {
                "world_uid": world_uid,
                "override_kind": treatment,
                "asset_index": int(assignment["instance_ordinal"]),
                "seller_uid_left": sellers[endpoint_sellers[0]],
                "seller_uid_right": sellers[endpoint_sellers[1]],
                "canonical_pair_uid": common.canonical_pair_uid(
                    sellers[endpoint_sellers[0]], sellers[endpoint_sellers[1]]
                ),
                "item_uid_left": endpoint_items[0]["item_uid"],
                "item_uid_right": endpoint_items[1]["item_uid"],
                "logical_item_ordinal_left": endpoint_items[0][
                    "logical_item_ordinal"
                ],
                "logical_item_ordinal_right": endpoint_items[1][
                    "logical_item_ordinal"
                ],
            }
        )
    if len(used_sellers) != 12:
        raise common.ContractError("V9.3 registered endpoint count drift")
    return audit


def _registered_negative_flags(
    *,
    world_uid: str,
    membership: Mapping[str, Any],
    plan_world: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    rows: list[dict[str, Any]] = []
    for assignment in plan_world["assignments"]:
        endpoints = assignment["endpoints"]
        left = sellers[int(endpoints[0]["seller_slot"])]
        right = sellers[int(endpoints[1]["seller_slot"])]
        treatment = str(assignment["treatment"])
        flag = {
            "exact_title_clone": "exact_title_clone_target",
            "high_semantic_similarity": "high_semantic_similarity_target",
        }.get(treatment)
        if flag is None:
            raise common.ContractError("Unknown V9.3 registered flag treatment")
        rows.append(
            {
                "canonical_pair_uid": common.canonical_pair_uid(left, right),
                "flag": flag,
                "asset_index": int(assignment["instance_ordinal"]),
            }
        )
    if len(rows) != 6 or len({row["canonical_pair_uid"] for row in rows}) != 6:
        raise common.ContractError("V9.3 registered flag universe drift")
    return rows


def _assign_code_free_noise(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    mode: str,
    split: str,
    world_ordinal: int,
    membership: Mapping[str, Any],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    text_key = str(policy["randomness"][mode]["text_key_hex"])
    id_key = str(policy["randomness"][mode]["id_key_hex"])
    values = tuple(fixture["must_ignore_adversarial_values"])
    templates = tuple(template["identity_clause_templates"]["must_ignore"])
    output: dict[str, dict[str, Any]] = {}
    for seller_slot, seller_uid in enumerate(common.utf8_sort(membership["seller_uids"])):
        eligible = [
            item for item in items_by_seller[seller_uid] if item["description_nonempty"]
        ]
        if not eligible:
            continue
        item = eligible[
            _design_index(
                text_key,
                len(eligible),
                split,
                world_ordinal,
                int(eligible[0]["noise_slot"]),
                "noise-item",
            )
        ]
        template_index = _design_index(
            text_key,
            len(templates),
            split,
            world_ordinal,
            int(item["noise_slot"]),
            "noise-template",
        )
        value = values[
            _design_index(
                text_key,
                len(values),
                split,
                world_ordinal,
                int(item["noise_slot"]),
                "noise-value",
            )
        ]
        clause = renderer.must_ignore_clause(
            template_index=template_index,
            value=value,
            template=template,
        )
        item["noise_clause"] = clause
        output[item["item_uid"]] = {
            "item_uid": item["item_uid"],
            "template_index": template_index,
            "value": value,
            "raw_surface": clause,
            "noise_slot_uid": structure.base_uid(
                key_hex=id_key,
                entity_kind="noise_slot",
                parent_uid_or_mode=item["item_uid"],
                ordinal=0,
            ),
            "seller_slot": seller_slot,
            "noise_slot": int(item["noise_slot"]),
        }
    return output


def _logical_identity_slot_order(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    logical_keys = [
        (
            int(row["global_asset_index"]),
            str(row["identity_type"]).encode("utf-8"),
            str(row["role"]).encode("utf-8"),
            str(row["identity_value"]).encode("utf-8"),
        )
        for row in rows
    ]
    if len(logical_keys) != len(set(logical_keys)):
        raise common.ContractError(
            "Identity slots lack a private-coordinate-free logical order"
        )
    return [
        row
        for _key, row in sorted(
            zip(logical_keys, rows, strict=True), key=lambda value: value[0]
        )
    ]


def _render_identity_slots(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    mode: str,
    split: str,
    world_ordinal: int,
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    noise_records_by_item: Mapping[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    role_to_family = policy["identity_design"]["role_to_template_family"]
    identity_time_key = str(
        policy["randomness"][mode]["identity_value_key_hex"]
    )
    identity_audit: list[dict[str, Any]] = []
    identity_edit: list[dict[str, Any]] = []
    noise_audit: list[dict[str, Any]] = []
    for seller_uid in common.utf8_sort(items_by_seller):
        for item in sorted(
            items_by_seller[seller_uid],
            key=lambda row: row["item_uid"].encode("utf-8"),
        ):
            logical_selector = _logical_render_selector(
                split=split,
                world_ordinal=world_ordinal,
                noise_slot=int(item["noise_slot"]),
                logical_item_ordinal=int(item["logical_item_ordinal"]),
            )
            slots = _logical_identity_slot_order(item["identity_slots"])
            identity_time_bucket = (
                _design_index(
                    identity_time_key,
                    4,
                    split,
                    world_ordinal,
                    common.canonical_sha256(
                        [
                            {
                                "identity_type": str(slot["identity_type"]),
                                "identity_value": str(slot["identity_value"]),
                                "role": str(slot["role"]),
                            }
                            for slot in slots
                        ]
                    ),
                    "identity-occurrence-time",
                )
                if slots
                else None
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
                selector_uid=logical_selector,
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
                selector_uid=logical_selector,
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
                    "time_bucket": identity_time_bucket,
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


def build_structure_blueprint(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    structure_key_hex: str,
    balanced_schedule: Mapping[str, Any],
    registered_negative_plan: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    candidate_index: int = 0,
) -> dict[str, Any]:
    """Build only the pre-text structure authority for one train/dev world."""

    common.validate_policy(policy, mode=mode)
    expected_template = common.load_json(
        common.repo_path(str(policy["template_library"]["path"]))
    )
    expected_style_profile = common.load_json(
        common.verify_file_pin(
            policy["style_reference_boundary"]["generator_release_inputs"][
                "profile"
            ],
            label="synthetic style reference",
        )
    )
    if common.canonical_json_bytes(template) != common.canonical_json_bytes(
        expected_template
    ):
        raise common.ContractError(
            "Structure blueprint received an unregistered template object"
        )
    common.validate_independent_replay_public_domains(
        policy,
        template=template,
        style_profile=expected_style_profile,
    )
    expected_records = {
        str(row["world_uid"]): row
        for row in structure.build_mode_world_pool(policy, mode=mode)
    }
    world_uid = str(world_record.get("world_uid", ""))
    expected_record = expected_records.get(world_uid)
    if (
        set(world_record)
        != {"world_uid", "mode_global_ordinal", "split", "split_ordinal"}
        or expected_record is None
        or common.canonical_json_bytes(dict(world_record))
        != common.canonical_json_bytes(expected_record)
    ):
        raise common.ContractError(
            "Structure blueprint world record is not registered"
        )
    split = str(world_record["split"])
    split_ordinal = int(world_record["split_ordinal"])
    if split not in ("train", "development"):
        raise common.ContractError(
            "Structure blueprint is limited to train/development"
        )
    if type(candidate_index) is not int or candidate_index != 0:
        raise common.ContractError(
            "Structure blueprint candidate index must remain frozen at zero"
        )
    schedule_audit = balanced.validate_schedule(balanced_schedule)
    plan_audit = _validate_registered_plan(
        registered_negative_plan,
        balanced_schedule,
        joint_signatures,
    )
    if schedule_audit["split"] != split or plan_audit["split"] != split:
        raise common.ContractError("Structure blueprint plan split drift")
    expected_structure_key = common.structure_key_for_split(
        policy, mode=mode, split=split
    )
    if structure_key_hex != expected_structure_key:
        raise common.ContractError(
            "Structure blueprint received an out-of-custody key"
        )
    graph_name = str(policy["identity_design"]["mechanism_by_split"][split])
    base_membership = structure.build_world_membership(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
    )
    membership, noise_by_seller = _planned_membership(
        base=base_membership,
        schedule_world=balanced_schedule["worlds"][split_ordinal],
        world_ordinal=split_ordinal,
    )
    (
        markets,
        _market_proposal,
        _mechanisms,
        _controller_styles,
        effective_styles,
    ) = _logical_nonidentity_assignments(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        world_ordinal=split_ordinal,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
    )
    items_by_seller = _planned_items(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        world_uid=world_uid,
        world_ordinal=split_ordinal,
        candidate_index=0,
        membership=membership,
        noise_by_seller=noise_by_seller,
        joint_signatures=joint_signatures,
        effective_styles=effective_styles,
        render_text=False,
    )
    override_audit = _apply_registered_overrides(
        template=template,
        split=split,
        world_uid=world_uid,
        world_ordinal=split_ordinal,
        membership=membership,
        items_by_seller=items_by_seller,
        plan_world=registered_negative_plan["worlds"][split_ordinal],
        joint_signatures=joint_signatures,
        render_text=False,
    )
    items = [
        item
        for seller_uid in common.utf8_sort(items_by_seller)
        for item in sorted(
            items_by_seller[seller_uid],
            key=lambda row: str(row["item_uid"]).encode("utf-8"),
        )
    ]
    if any(
        item["title"] or item["base_description"] or item["natural_variation_phrase"]
        for item in items
    ):
        raise common.ContractError("Structure blueprint produced natural text")
    render_asts = [
        {
            "world_uid": item["world_uid"],
            "seller_uid": item["seller_uid"],
            "item_uid": item["item_uid"],
            "time_bucket": item["time_bucket"],
            "category": item["category"],
            "logical_item_ordinal": item["logical_item_ordinal"],
            "noise_slot": item["noise_slot"],
            "title_skeleton_index": item["title_skeleton_index"],
            "description_skeleton_index": item[
                "description_skeleton_index"
            ],
            "natural_variation_ordinal": item["natural_variation_ordinal"],
            "title_nonempty": item["title_nonempty"],
            "description_nonempty": item["description_nonempty"],
            "service": item["service"],
            "delivery": item["delivery"],
        }
        for item in items
    ]
    controller_membership = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "seller_uid": seller_uid,
        }
        for controller_uid in common.utf8_sort(
            membership["controller_members"]
        )
        for seller_uid in membership["controller_members"][controller_uid]
    ]
    blueprint = {
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
            "complete_model_pair_endpoints": _complete_pair_endpoints(
                world_uid, membership["seller_uids"]
            ),
        },
        "private": {
            "controller_membership": controller_membership,
            "render_asts": render_asts,
            "override_audit": override_audit,
        },
        "audit": {
            "split": split,
            "split_ordinal": split_ordinal,
            "candidate_index": 0,
            "natural_text_field_count": 0,
            "identity_asset_count": 0,
            "pair_label_count": 0,
            "qrels_count": 0,
            "role_eligibility_predicates": list(
                negative_plan.ROLE_ELIGIBILITY_PREDICATE_NAMES
            ),
            "balanced_schedule_canonical_self_sha256": balanced_schedule[
                "canonical_self_sha256"
            ],
            "registered_negative_plan_canonical_self_sha256": (
                registered_negative_plan["canonical_self_sha256"]
            ),
        },
    }
    return blueprint


def build_world(
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
    """Build one train/development V9.3 world without filesystem writes."""

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
    if split not in ("train", "development", "audit_a", "audit_b"):
        raise common.ContractError("Unknown V9.3 method split")
    if (
        type(candidate_index) is not int
        or not 0 <= candidate_index < 32
    ):
        raise common.ContractError("V9.3 candidate index is outside 0..31")
    split_ordinal = int(world_record["split_ordinal"])
    if split in ("train", "development"):
        if balanced_schedule is None or registered_negative_plan is None:
            raise common.ContractError("V9.3 train/development plan is absent")
        schedule_audit = balanced.validate_schedule(balanced_schedule)
        plan_audit = _validate_registered_plan(
            registered_negative_plan, balanced_schedule, joint_signatures
        )
        if schedule_audit["split"] != split or plan_audit["split"] != split:
            raise common.ContractError("V9.3 planned input split drift")
        schedule_world = balanced_schedule["worlds"][split_ordinal]
        plan_world = registered_negative_plan["worlds"][split_ordinal]
        schedule_commitment = str(balanced_schedule["canonical_self_sha256"])
        plan_commitment = str(
            registered_negative_plan["canonical_self_sha256"]
        )
    else:
        if balanced_schedule is not None or registered_negative_plan is not None:
            raise common.ContractError("Blind-audit split received a train/dev plan")
        if blind_audit_design is None:
            raise common.ContractError("Blind-audit design is absent")
        audit_design_module.validate_payload(blind_audit_design, joint_signatures)
        design_block = blind_audit_design["splits"][split]
        if (
            design_block.get("world_count") != 2
            or not 0 <= split_ordinal < 2
        ):
            raise common.ContractError("Blind-audit world ordinal drift")
        schedule_world = design_block["schedule_worlds"][split_ordinal]
        plan_world = design_block["plan_worlds"][split_ordinal]
        schedule_commitment = str(blind_audit_design["canonical_self_sha256"])
        plan_commitment = schedule_commitment
    expected_structure_key = common.structure_key_for_split(
        policy, mode=mode, split=split
    )
    if structure_key_hex != expected_structure_key:
        raise common.ContractError(
            "World builder received a structure key outside split custody"
        )
    graph_name = str(policy["identity_design"]["mechanism_by_split"][split])
    base_membership = structure.build_world_membership(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
    )
    membership, noise_by_seller = _planned_membership(
        base=base_membership,
        schedule_world=schedule_world,
        world_ordinal=split_ordinal,
    )
    (
        markets,
        market_proposal,
        mechanisms,
        controller_styles,
        effective_styles,
    ) = _logical_nonidentity_assignments(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        world_ordinal=split_ordinal,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
    )
    items_by_seller = _planned_items(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        world_uid=world_uid,
        world_ordinal=split_ordinal,
        candidate_index=candidate_index,
        membership=membership,
        noise_by_seller=noise_by_seller,
        joint_signatures=joint_signatures,
        effective_styles=effective_styles,
    )

    override_audit: list[dict[str, Any]] = []
    noise_records_by_item: dict[str, dict[str, Any]] = {}
    pre_slot_call_count = 0

    def materialize_observed_overrides_and_noise(
        final_negative_flags: Any,
    ) -> dict[str, Any]:
        nonlocal pre_slot_call_count
        pre_slot_call_count += 1
        if pre_slot_call_count != 1:
            raise common.ContractError("Pre-slot materializer called more than once")
        if not isinstance(final_negative_flags, (list, tuple)):
            raise common.ContractError("V9.3 parent negative flags are malformed")
        override_audit.extend(
            _apply_registered_overrides(
                template=template,
                split=split,
                world_uid=world_uid,
                world_ordinal=split_ordinal,
                membership=membership,
                items_by_seller=items_by_seller,
                plan_world=plan_world,
                joint_signatures=joint_signatures,
            )
        )
        noise_records_by_item.update(
            _assign_code_free_noise(
                policy=policy,
                template=template,
                fixture=fixture,
                mode=mode,
                split=split,
                world_ordinal=split_ordinal,
                membership=membership,
                items_by_seller=items_by_seller,
            )
        )
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

    expected_noise_count = sum(
        any(item["description_nonempty"] for item in items)
        for items in items_by_seller.values()
    )
    solved = _solve_identity_plan_logically(
        policy=policy,
        mode=mode,
        split=split,
        world_uid=world_uid,
        world_ordinal=split_ordinal,
        world_mode_global_ordinal=int(world_record["mode_global_ordinal"]),
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
        mechanisms=mechanisms,
        items_by_seller=items_by_seller,
        pre_slot_callback=materialize_observed_overrides_and_noise,
    )
    if (
        pre_slot_call_count != 1
        or len(noise_records_by_item) != expected_noise_count
    ):
        raise common.ContractError("Pre-slot observed materialization did not close")
    parent_flags = [
        dict(row)
        for row in solved["negative_flags"]
        if row["flag"]
        not in {"exact_title_clone_target", "high_semantic_similarity_target"}
    ]
    solved["negative_flags"] = [
        *parent_flags,
        *_registered_negative_flags(
            world_uid=world_uid,
            membership=membership,
            plan_world=plan_world,
        ),
    ]

    identity_audit, identity_edit, noise_audit = _render_identity_slots(
        policy=policy,
        template=template,
        fixture=fixture,
        mode=mode,
        split=split,
        world_ordinal=split_ordinal,
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
            "logical_item_ordinal": item["logical_item_ordinal"],
            "seller_slot": item["seller_slot"],
            "noise_slot": item["noise_slot"],
            "title_skeleton_index": item["title_skeleton_index"],
            "description_skeleton_index": item["description_skeleton_index"],
            "natural_variation_ordinal": item["natural_variation_ordinal"],
            "natural_variation_phrase": item["natural_variation_phrase"],
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
                "balanced_schedule_canonical_self_sha256": schedule_commitment,
                "registered_negative_plan_canonical_self_sha256": plan_commitment,
                "joint_noise_signature_canonical_self_sha256": joint_signatures[
                    "canonical_self_sha256"
                ],
                "candidate_index": candidate_index,
                **solved["solver_audit"],
            },
        },
    }
