#!/usr/bin/env python3
"""Label-blind nonidentity item DGP for Step 28-v13."""

from __future__ import annotations

import hashlib
import hmac
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_text_renderer as renderer


PROBABILITY_SCALE = 10**12


def integerized_probabilities(probabilities: list[float]) -> list[int]:
    if not probabilities:
        raise common.ContractError("Categorical probability list is empty")
    output: list[int] = []
    with localcontext() as context:
        context.prec = 80
        for value in probabilities[:-1]:
            probability = Decimal(str(value))
            if not probability.is_finite() or probability < 0:
                raise common.ContractError("Invalid categorical probability")
            output.append(
                int(
                    (probability * Decimal(PROBABILITY_SCALE)).to_integral_value(
                        rounding=ROUND_HALF_EVEN
                    )
                )
            )
    final = PROBABILITY_SCALE - sum(output)
    if final < 0:
        raise common.ContractError("Categorical integerization has negative final cell")
    output.append(final)
    if sum(output) != PROBABILITY_SCALE:
        raise common.ContractError("Categorical integerization does not sum to scale")
    return output


def categorical_choice(
    rng: common.DeterministicRng,
    values: list[Any],
    probabilities: list[float],
) -> Any:
    if len(values) != len(probabilities):
        raise common.ContractError("Categorical values/probabilities length mismatch")
    weights = integerized_probabilities(probabilities)
    draw = rng.randbelow(PROBABILITY_SCALE)
    cursor = 0
    for value, weight in zip(values, weights, strict=True):
        cursor += weight
        if draw < cursor:
            return value
    raise common.ContractError("Categorical draw fell outside integerized mass")


def inverse_quantile(
    quantiles: dict[str, float], draw: Decimal
) -> Decimal:
    knots = sorted(
        (Decimal(key), Decimal(str(value))) for key, value in quantiles.items()
    )
    if not knots or draw < 0 or draw >= 1:
        raise common.ContractError("Invalid inverse-quantile input")
    if draw <= knots[0][0]:
        return knots[0][1]
    if draw >= knots[-1][0]:
        return knots[-1][1]
    for (left_p, left_v), (right_p, right_v) in zip(knots, knots[1:]):
        if left_p <= draw <= right_p:
            with localcontext() as context:
                context.prec = 80
                fraction = (draw - left_p) / (right_p - left_p)
                return left_v + fraction * (right_v - left_v)
    raise common.ContractError("Inverse-quantile interpolation failed")


def _text_rank(
    key_hex: str,
    *parts: str,
    candidates: list[str],
) -> list[str]:
    if not candidates or len(candidates) != len(set(candidates)):
        raise common.ContractError("Text rank candidates are empty or duplicated")
    prefix = common.FIELD_SEPARATOR.join(part.encode("utf-8") for part in parts)
    key = bytes.fromhex(key_hex)
    return [
        value
        for _digest, _value_bytes, value in sorted(
            (
                hmac.new(
                    key,
                    prefix + common.FIELD_SEPARATOR + value.encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
                value.encode("utf-8"),
                value,
            )
            for value in candidates
        )
    ]


def world_controller_styles(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    world_uid: str,
    structure_key_hex: str,
    controller_uids: list[str],
    mode: str,
) -> dict[str, dict[str, Any]]:
    text_key = policy["randomness"][mode]["text_key_hex"]
    style_by_id = {
        row["style_id"]: row for row in template["style_prototypes"]
    }
    selected = _text_rank(
        text_key,
        world_uid,
        "selected_style",
        candidates=list(style_by_id),
    )[:4]
    controller_order = structure.rank_candidates(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_style_assignment",
        candidates=controller_uids,
    )
    output: dict[str, dict[str, Any]] = {}
    for index, controller_uid in enumerate(controller_order):
        output[controller_uid] = dict(style_by_id[selected[index // 3]])
    if len(output) != 12 or any(
        sum(row["style_id"] == style_id for row in output.values()) != 3
        for style_id in selected
    ):
        raise common.ContractError("Controller style allocation drift")
    return output


def seller_effective_style(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    mode: str,
    seller_uid: str,
    controller_style: dict[str, Any],
) -> dict[str, Any]:
    contract = template["renderer_contract"]
    factors = list(contract["style_factor_order"])
    selected = _text_rank(
        policy["randomness"][mode]["text_key_hex"],
        seller_uid,
        "style_perturbation",
        candidates=factors,
    )[:2]
    values = {factor: controller_style[factor] for factor in factors}
    for factor in selected:
        domain = list(contract["style_factor_domains"][factor])
        index = domain.index(values[factor])
        values[factor] = domain[(index + 1) % len(domain)]
    return {"effective_style_uid": renderer.effective_style_uid(values), **values}


def _rng(key_hex: str, entity_uid: str, draw_name: str, *suffix: int) -> common.DeterministicRng:
    context = ["step28-v13-item-dgp", entity_uid, draw_name]
    context.extend(str(value) for value in suffix)
    return common.DeterministicRng(key_hex, *context)


def _item_code(key_hex: str, item_uid: str) -> str:
    digest = hmac.new(
        bytes.fromhex(key_hex),
        item_uid.encode("utf-8") + common.FIELD_SEPARATOR + b"code",
        hashlib.sha256,
    ).digest()
    return renderer.nibble_code(digest)


def build_seller_items(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    style_profile: dict[str, Any],
    mode: str,
    split: str,
    world_uid: str,
    seller_uid: str,
    effective_style: dict[str, Any],
) -> list[dict[str, Any]]:
    text_key = policy["randomness"][mode]["text_key_hex"]
    id_key = policy["randomness"][mode]["id_key_hex"]
    item_count_pmf = style_profile["item_count_pmf"]
    item_count_values = [int(value) for value in item_count_pmf]
    item_count_probabilities = [
        float(item_count_pmf[str(value)]) for value in item_count_values
    ]
    item_count = int(
        categorical_choice(
            _rng(text_key, seller_uid, "item_count"),
            item_count_values,
            item_count_probabilities,
        )
    )
    quantiles = style_profile["seller_equal_weight_quantiles"]
    title_rate = inverse_quantile(
        quantiles["title_missing"],
        _rng(text_key, seller_uid, "title_missing_rate").uniform01_decimal(),
    )
    description_rate = inverse_quantile(
        quantiles["description_missing"],
        _rng(text_key, seller_uid, "description_missing_rate").uniform01_decimal(),
    )
    masks: list[tuple[bool, bool]] | None = None
    for proposal in range(
        int(policy["nonidentity_item_dgp"]["missingness"]["maximum_proposals"])
    ):
        current: list[tuple[bool, bool]] = []
        for item_index in range(item_count):
            title_nonempty = _rng(
                text_key, seller_uid, "title_nonempty", proposal, item_index
            ).bernoulli(Decimal(1) - title_rate)
            description_nonempty = _rng(
                text_key, seller_uid, "description_nonempty", proposal, item_index
            ).bernoulli(Decimal(1) - description_rate)
            current.append((title_nonempty, description_nonempty))
        if (
            sum(title for title, _description in current) >= 1
            and sum(description for _title, description in current) >= 2
        ):
            masks = current
            break
    if masks is None:
        raise common.ContractError(f"Missingness proposal exhaustion for {seller_uid}")

    lexicon = template["generic_lexicon"]
    category_probabilities = list(
        style_profile["anonymous_category_rank_probability"]
    )
    category_values = list(lexicon["categories"])
    library = template["split_libraries"][split]
    output: list[dict[str, Any]] = []
    for item_index, (title_nonempty, description_nonempty) in enumerate(masks):
        item_uid = structure.base_uid(
            key_hex=id_key,
            entity_kind="item",
            parent_uid_or_mode=seller_uid,
            ordinal=item_index,
        )
        category = categorical_choice(
            _rng(text_key, item_uid, "category"),
            category_values,
            category_probabilities,
        )
        product = _rng(text_key, item_uid, "product").choice(
            lexicon["category_products"][category]
        )
        attribute = _rng(text_key, item_uid, "attribute").choice(
            lexicon["attributes"]
        )
        delivery = _rng(text_key, item_uid, "delivery").choice(
            lexicon["delivery"]
        )
        service = _rng(text_key, item_uid, "service").choice(
            lexicon["service"]
        )
        title_index = _rng(text_key, item_uid, "title_skeleton").randbelow(
            len(library["title_skeletons"])
        )
        description_index = _rng(
            text_key, item_uid, "description_skeleton"
        ).randbelow(len(library["description_skeletons"]))
        time_bucket = _rng(text_key, item_uid, "time_bucket").randbelow(4)
        code = _item_code(text_key, item_uid)
        title = (
            renderer.render_base_title(
                skeleton=library["title_skeletons"][title_index],
                product=product,
                attribute=attribute,
                code=code,
                style=effective_style,
                template=template,
            )
            if title_nonempty
            else ""
        )
        base_description = (
            renderer.render_base_description(
                skeleton=library["description_skeletons"][description_index],
                product=product,
                attribute=attribute,
                code=code,
                delivery=delivery,
                service=service,
                style=effective_style,
                template=template,
            )
            if description_nonempty
            else ""
        )
        output.append(
            {
                "world_uid": world_uid,
                "seller_uid": seller_uid,
                "item_uid": item_uid,
                "item_ordinal": item_index,
                "time_bucket": time_bucket,
                "category": category,
                "product": product,
                "attribute": attribute,
                "delivery": delivery,
                "service": service,
                "code": code,
                "title_skeleton_index": title_index,
                "description_skeleton_index": description_index,
                "effective_style_uid": effective_style["effective_style_uid"],
                "effective_style": dict(effective_style),
                "title_nonempty": title_nonempty,
                "description_nonempty": description_nonempty,
                "title": title,
                "base_description": base_description,
                "noise_clause": "",
                "identity_slots": [],
            }
        )
    return output


def assign_must_ignore_noise(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    mode: str,
    seller_uid: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    text_key = policy["randomness"][mode]["text_key_hex"]
    eligible = [row for row in items if row["description_nonempty"]]
    if not eligible:
        raise common.ContractError("Seller lacks a nonempty description for noise")
    ranked = _text_rank(
        text_key,
        seller_uid,
        "must_ignore_item",
        candidates=[row["item_uid"] for row in eligible],
    )
    by_uid = {row["item_uid"]: row for row in eligible}
    item = by_uid[ranked[0]]
    fixture = common.load_json(
        common.repo_path(
            policy["identity_design"]["role_template_parser_flag_fixture"]["path"]
        )
    )
    template_index = _rng(text_key, seller_uid, "must_ignore_template").randbelow(
        len(template["identity_clause_templates"]["must_ignore"])
    )
    value = _rng(text_key, seller_uid, "must_ignore_value").choice(
        fixture["must_ignore_adversarial_values"]
    )
    clause = renderer.must_ignore_clause(
        template_index=template_index,
        value=value,
        template=template,
    )
    item["noise_clause"] = clause
    return {
        "item_uid": item["item_uid"],
        "template_index": template_index,
        "value": value,
        "raw_surface": clause,
    }


def _override_rank(
    structure_key_hex: str,
    *,
    world_uid: str,
    draw_name: str,
    asset_index: int,
    candidates: Sequence[str],
    prefix_atoms: Sequence[str] = (),
) -> list[str]:
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise common.ContractError(f"Override candidates invalid for {draw_name}")
    key = bytes.fromhex(structure_key_hex)
    prefix = (
        world_uid.encode("utf-8"),
        draw_name.encode("ascii"),
        str(asset_index).encode("ascii"),
        *(value.encode("utf-8") for value in prefix_atoms),
    )
    return [
        candidate
        for _digest, _candidate_bytes, candidate in sorted(
            (
                hmac.new(
                    key,
                    common.FIELD_SEPARATOR.join((*prefix, candidate.encode("utf-8"))),
                    hashlib.sha256,
                ).digest(),
                candidate.encode("utf-8"),
                candidate,
            )
            for candidate in values
        )
    ]


def _rerender_item_base(
    item: dict[str, Any],
    *,
    template: Mapping[str, Any],
    split: str,
) -> None:
    library = template["split_libraries"][split]
    if item["title_nonempty"]:
        item["title"] = renderer.render_base_title(
            skeleton=library["title_skeletons"][item["title_skeleton_index"]],
            product=item["product"],
            attribute=item["attribute"],
            code=item["code"],
            style=item["effective_style"],
            template=template,
        )
    else:
        item["title"] = ""
    if item["description_nonempty"]:
        item["base_description"] = renderer.render_base_description(
            skeleton=library["description_skeletons"][
                item["description_skeleton_index"]
            ],
            product=item["product"],
            attribute=item["attribute"],
            code=item["code"],
            delivery=item["delivery"],
            service=item["service"],
            style=item["effective_style"],
            template=template,
        )
    else:
        item["base_description"] = ""


def apply_registered_overrides(
    *,
    policy: dict[str, Any],
    template: dict[str, Any],
    style_profile: dict[str, Any],
    split: str,
    world_uid: str,
    structure_key_hex: str,
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    negative_flags: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply registered semantic overrides, then exact title clones."""

    item_maps = {
        seller_uid: {row["item_uid"]: row for row in rows}
        for seller_uid, rows in items_by_seller.items()
    }
    reserved_override_items: defaultdict[str, set[str]] = defaultdict(set)

    def ordered_endpoints(
        row: Mapping[str, Any], *, draw_name: str
    ) -> tuple[str, str]:
        pair_uid = str(row["canonical_pair_uid"])
        endpoints = pair_uid.split("||")
        if (
            len(endpoints) != 2
            or common.canonical_pair_uid(endpoints[0], endpoints[1])
            != pair_uid
        ):
            raise common.ContractError(
                "Registered override flag has an invalid canonical pair"
            )
        ranked = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name=draw_name,
            asset_index=int(row["asset_index"]),
            candidates=endpoints,
        )
        return ranked[0], ranked[1]

    def selected_item(
        seller_uid: str,
        *,
        draw_name: str,
        asset_index: int,
        side_name: str,
    ) -> dict[str, Any]:
        candidates = [
            row["item_uid"]
            for row in items_by_seller[seller_uid]
            if row["title_nonempty"]
            and row["item_uid"] not in reserved_override_items[seller_uid]
        ]
        if not candidates:
            raise common.ContractError(
                f"Registered override item capacity exhausted for {seller_uid}"
            )
        ranked = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name=draw_name,
            asset_index=asset_index,
            candidates=candidates,
            prefix_atoms=(side_name,),
        )
        chosen = ranked[0]
        reserved_override_items[seller_uid].add(chosen)
        return item_maps[seller_uid][chosen]

    semantic_flags = sorted(
        (
            row
            for row in negative_flags
            if row["flag"] == "high_semantic_similarity_target"
        ),
        key=lambda row: (
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        ),
    )
    lexicon = template["generic_lexicon"]
    category_values = list(lexicon["categories"])
    category_probabilities = list(
        style_profile["anonymous_category_rank_probability"]
    )
    title_skeleton_count = len(template["split_libraries"][split]["title_skeletons"])
    semantic_updates: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for flag in semantic_flags:
        asset_index = int(flag["asset_index"])
        left_seller, right_seller = ordered_endpoints(
            flag, draw_name="high_semantic_side"
        )
        left_item = selected_item(
            left_seller,
            draw_name="high_semantic_item",
            asset_index=asset_index,
            side_name="left",
        )
        right_item = selected_item(
            right_seller,
            draw_name="high_semantic_item",
            asset_index=asset_index,
            side_name="right",
        )
        category = categorical_choice(
            common.DeterministicRng(
                structure_key_hex,
                world_uid,
                "high_semantic_category",
                str(asset_index),
            ),
            category_values,
            category_probabilities,
        )
        product = common.DeterministicRng(
            structure_key_hex,
            world_uid,
            "high_semantic_product",
            str(asset_index),
        ).choice(lexicon["category_products"][category])
        attribute = common.DeterministicRng(
            structure_key_hex,
            world_uid,
            "high_semantic_attribute",
            str(asset_index),
        ).choice(lexicon["attributes"])
        skeleton_candidates = [
            structure.TUPLE_SEPARATOR.join((str(left), str(right)))
            for left in range(title_skeleton_count)
            for right in range(title_skeleton_count)
            if left != right
        ]
        skeleton_pair = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name="high_semantic_skeleton_pair",
            asset_index=asset_index,
            candidates=skeleton_candidates,
        )[0]
        left_skeleton, right_skeleton = (
            int(value) for value in skeleton_pair.split(structure.TUPLE_SEPARATOR)
        )
        for item, skeleton_index in (
            (left_item, left_skeleton),
            (right_item, right_skeleton),
        ):
            update = {
                "category": category,
                "product": product,
                "attribute": attribute,
                "title_skeleton_index": skeleton_index,
            }
            prior = semantic_updates.setdefault(item["item_uid"], update)
            if prior != update:
                raise common.ContractError(
                    "Conflicting high-semantic overrides selected the same item"
                )
        audit.append(
            {
                "override_kind": "high_semantic_similarity",
                "asset_index": asset_index,
                "canonical_pair_uid": flag["canonical_pair_uid"],
                "seller_uid_left": left_seller,
                "seller_uid_right": right_seller,
                "item_uid_left": left_item["item_uid"],
                "item_uid_right": right_item["item_uid"],
            }
        )
    all_items = {
        item["item_uid"]: item for rows in items_by_seller.values() for item in rows
    }
    for item_uid, update in semantic_updates.items():
        all_items[item_uid].update(update)
        _rerender_item_base(all_items[item_uid], template=template, split=split)

    clone_flags = sorted(
        (
            row
            for row in negative_flags
            if row["flag"] == "exact_title_clone_target"
        ),
        key=lambda row: (
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        ),
    )
    title_snapshot = {uid: str(item["title"]) for uid, item in all_items.items()}
    clone_updates: dict[str, str] = {}
    for flag in clone_flags:
        asset_index = int(flag["asset_index"])
        source_seller, destination_seller = ordered_endpoints(
            flag, draw_name="exact_clone_side"
        )
        source_item = selected_item(
            source_seller,
            draw_name="exact_clone_item",
            asset_index=asset_index,
            side_name="source",
        )
        destination_item = selected_item(
            destination_seller,
            draw_name="exact_clone_item",
            asset_index=asset_index,
            side_name="destination",
        )
        source_title = title_snapshot[source_item["item_uid"]]
        prior = clone_updates.setdefault(destination_item["item_uid"], source_title)
        if prior != source_title:
            raise common.ContractError(
                "Conflicting exact-title clones selected the same destination item"
            )
        audit.append(
            {
                "override_kind": "exact_title_clone",
                "asset_index": asset_index,
                "canonical_pair_uid": flag["canonical_pair_uid"],
                "seller_uid_left": source_seller,
                "seller_uid_right": destination_seller,
                "item_uid_left": source_item["item_uid"],
                "item_uid_right": destination_item["item_uid"],
            }
        )
    for item_uid, title in clone_updates.items():
        all_items[item_uid]["title"] = title
    if len(audit) != 6:
        raise common.ContractError("Registered override audit count drift")
    endpoint_sellers = {
        row[key]
        for row in audit
        for key in ("seller_uid_left", "seller_uid_right")
    }
    endpoint_items = {
        row[key]
        for row in audit
        for key in ("item_uid_left", "item_uid_right")
    }
    if len(endpoint_sellers) != 12 or len(endpoint_items) != 12:
        raise common.ContractError("Registered override endpoint uniqueness drift")
    for row in audit:
        left = all_items[row["item_uid_left"]]
        right = all_items[row["item_uid_right"]]
        if row["override_kind"] == "high_semantic_similarity":
            if (
                left["category"] != right["category"]
                or left["product"] != right["product"]
                or left["attribute"] != right["attribute"]
                or left["title_skeleton_index"] == right["title_skeleton_index"]
                or left["code"] == right["code"]
                or left["title"] == right["title"]
            ):
                raise common.ContractError(
                    "High-semantic registered override integrity failed"
                )
            if left["product"] not in lexicon["category_products"][left["category"]]:
                raise common.ContractError(
                    "High-semantic product/category integrity failed"
                )
        elif row["override_kind"] == "exact_title_clone":
            if not left["title"] or left["title"] != right["title"]:
                raise common.ContractError("Exact-title clone integrity failed")
        else:
            raise common.ContractError("Unknown registered override kind")
    return audit
