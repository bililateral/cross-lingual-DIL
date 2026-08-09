#!/usr/bin/env python3
"""Exact conditional assignment-null audit for v1.12 style counterfactuals."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_text_renderer as renderer


SELLER_COUNT = 28
STYLE_FACTORS = (
    "separator",
    "ending",
    "line_mode",
    "english_tag",
    "traditional_variant",
    "repeat_punctuation",
)


class AssignmentNullError(ValueError):
    """Raised when the exact conditional assignment null cannot close."""


def _completion_count(*, remaining_count: int, available_diagonal_count: int) -> int:
    if not 0 <= available_diagonal_count <= remaining_count:
        raise AssignmentNullError("Invalid derangement completion dimensions")
    return sum(
        (-1) ** selected
        * math.comb(available_diagonal_count, selected)
        * math.factorial(remaining_count - selected)
        for selected in range(available_diagonal_count + 1)
    )


def derangement_count(size: int) -> int:
    if type(size) is not int or size < 0:
        raise AssignmentNullError("Derangement size must be a nonnegative int")
    return _completion_count(
        remaining_count=size, available_diagonal_count=size
    )


def _pair_relation_expectation(
    *,
    left_target: str,
    right_target: str,
    attribute_by_seller: Mapping[str, Any],
) -> float:
    """Exact P[g(source-left)==g(source-right) | uniform derangement]."""

    sellers = set(attribute_by_seller)
    if (
        len(sellers) != SELLER_COUNT
        or left_target not in sellers
        or right_target not in sellers
        or left_target == right_target
    ):
        raise AssignmentNullError("Pair expectation seller universe drift")
    outside = sellers - {left_target, right_target}
    counts = Counter(attribute_by_seller[seller_uid] for seller_uid in outside)
    left_value = attribute_by_seller[left_target]
    right_value = attribute_by_seller[right_target]
    swap_equal = int(left_value == right_value)
    one_cross_equal_count = counts[right_value] + counts[left_value]
    outside_ordered_equal_count = sum(
        count * (count - 1) for count in counts.values()
    )
    remaining = SELLER_COUNT - 2
    swap_weight = _completion_count(
        remaining_count=remaining,
        available_diagonal_count=SELLER_COUNT - 2,
    )
    one_cross_weight = _completion_count(
        remaining_count=remaining,
        available_diagonal_count=SELLER_COUNT - 3,
    )
    outside_weight = _completion_count(
        remaining_count=remaining,
        available_diagonal_count=SELLER_COUNT - 4,
    )
    denominator = derangement_count(SELLER_COUNT)
    enumerated_denominator = (
        swap_weight
        + 2 * (SELLER_COUNT - 2) * one_cross_weight
        + (SELLER_COUNT - 2) * (SELLER_COUNT - 3) * outside_weight
    )
    if enumerated_denominator != denominator:
        raise AssignmentNullError("Pair derangement completion mass did not close")
    numerator = (
        swap_equal * swap_weight
        + one_cross_equal_count * one_cross_weight
        + outside_ordered_equal_count * outside_weight
    )
    if not 0 <= numerator <= denominator:
        raise AssignmentNullError("Pair derangement relation mass is invalid")
    return numerator / denominator


def _seller_relation_expectation(
    *, target: str, attribute_by_seller: Mapping[str, Any]
) -> float:
    if len(attribute_by_seller) != SELLER_COUNT or target not in attribute_by_seller:
        raise AssignmentNullError("Seller expectation universe drift")
    value = attribute_by_seller[target]
    matching_nonself = sum(
        seller_uid != target and candidate == value
        for seller_uid, candidate in attribute_by_seller.items()
    )
    return matching_nonself / float(SELLER_COUNT - 1)


def build_assignment_null_rows(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    controller_membership: Sequence[Mapping[str, Any]],
    controller_style_groups: Sequence[Mapping[str, Any]],
    target_source_pairs: Sequence[Sequence[str]],
    eligible_pair_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> dict[str, Any]:
    """Return observed and exact-null seller/pair relations for one world."""

    contract = policy["assignment_null_audit"]
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    world_uids = {str(row["world_uid"]) for row in sellers}
    if len(seller_uids) != SELLER_COUNT or len(world_uids) != 1:
        raise AssignmentNullError("Assignment audit requires one 28-seller world")
    world_uid = next(iter(world_uids))
    source_by_target = {
        str(row[0]): str(row[1]) for row in target_source_pairs
    }
    if (
        len(source_by_target) != SELLER_COUNT
        or set(source_by_target) != seller_uids
        or set(source_by_target.values()) != seller_uids
        or any(target == source for target, source in source_by_target.items())
    ):
        raise AssignmentNullError("Assignment audit mapping is not a derangement")
    controller_by_seller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in controller_membership
    }
    base_style_by_controller = {
        str(row["controller_uid"]): str(row["style_id"])
        for row in controller_style_groups
    }
    if (
        len(controller_by_seller) != SELLER_COUNT
        or set(controller_by_seller) != seller_uids
        or set(controller_by_seller.values()) != set(base_style_by_controller)
    ):
        raise AssignmentNullError("Assignment controller/base-style closure failed")
    base_style_by_seller = {
        seller_uid: base_style_by_controller[controller_uid]
        for seller_uid, controller_uid in controller_by_seller.items()
    }
    reachable = {
        str(row["effective_style_uid"]): dict(row)
        for row in renderer.reachable_effective_styles(template)
    }
    effective_uids_by_seller: dict[str, set[str]] = {
        seller_uid: set() for seller_uid in seller_uids
    }
    for ast in render_asts:
        seller_uid = str(ast["seller_uid"])
        style_uid = str(ast["effective_style_uid"])
        if seller_uid not in effective_uids_by_seller or style_uid not in reachable:
            raise AssignmentNullError("Assignment render AST style drift")
        effective_uids_by_seller[seller_uid].add(style_uid)
    if any(len(values) != 1 for values in effective_uids_by_seller.values()):
        raise AssignmentNullError("Assignment seller effective style is not unique")
    effective_uid_by_seller = {
        seller_uid: next(iter(values))
        for seller_uid, values in effective_uids_by_seller.items()
    }
    factor_by_seller = {
        factor: {
            seller_uid: reachable[effective_uid_by_seller[seller_uid]][factor]
            for seller_uid in seller_uids
        }
        for factor in STYLE_FACTORS
    }
    attributes: dict[str, Mapping[str, Any]] = {
        "controller": controller_by_seller,
        "base_style": base_style_by_seller,
        "effective_style": effective_uid_by_seller,
        **{f"factor__{factor}": values for factor, values in factor_by_seller.items()},
    }

    seller_rows: list[dict[str, Any]] = []
    for target in sorted(seller_uids, key=lambda value: value.encode("utf-8")):
        source = source_by_target[target]
        row: dict[str, Any] = {
            "world_uid": world_uid,
            "target_seller_uid": target,
            "source_seller_uid": source,
        }
        for relation_name, attribute_name in (
            ("source_controller_equals_target_controller", "controller"),
            ("source_base_style_equals_target_original_base_style", "base_style"),
            (
                "source_effective_style_equals_target_original_effective_style",
                "effective_style",
            ),
        ):
            attribute = attributes[attribute_name]
            row[relation_name] = int(attribute[source] == attribute[target])
            row[f"expected__{relation_name}"] = _seller_relation_expectation(
                target=target, attribute_by_seller=attribute
            )
        seller_rows.append(row)

    if len(eligible_pair_rows) != len(labels):
        raise AssignmentNullError("Assignment pair/label length drift")
    pair_rows: list[dict[str, Any]] = []
    pair_relation_attributes = (
        ("same_source_controller", "controller"),
        ("same_source_base_style", "base_style"),
        ("same_source_effective_style", "effective_style"),
        *((f"same_source_factor__{factor}", f"factor__{factor}") for factor in STYLE_FACTORS),
    )
    for pair, label in zip(eligible_pair_rows, labels, strict=True):
        left = str(pair["seller_uid_left"])
        right = str(pair["seller_uid_right"])
        if left not in seller_uids or right not in seller_uids or left == right:
            raise AssignmentNullError("Assignment eligible endpoint drift")
        source_left = source_by_target[left]
        source_right = source_by_target[right]
        row = {
            "world_uid": world_uid,
            "canonical_pair_uid": str(pair["canonical_pair_uid"]),
            "label": int(label),
        }
        factor_observed: list[float] = []
        factor_expected: list[float] = []
        for relation_name, attribute_name in pair_relation_attributes:
            attribute = attributes[attribute_name]
            observed = float(attribute[source_left] == attribute[source_right])
            expected = _pair_relation_expectation(
                left_target=left,
                right_target=right,
                attribute_by_seller=attribute,
            )
            row[relation_name] = observed
            row[f"expected__{relation_name}"] = expected
            if relation_name.startswith("same_source_factor__"):
                factor_observed.append(observed)
                factor_expected.append(expected)
        row["same_source_factor_proportion"] = sum(factor_observed) / len(
            STYLE_FACTORS
        )
        row["expected__same_source_factor_proportion"] = sum(
            factor_expected
        ) / len(STYLE_FACTORS)
        pair_rows.append(row)
    required_relations = tuple(str(value) for value in contract["pair_gate_relations_in_order"])
    if (
        len(pair_rows) != 372
        or sum(int(row["label"]) for row in pair_rows) != 20
        or any(
            name not in row or f"expected__{name}" not in row
            for row in pair_rows
            for name in required_relations
        )
    ):
        raise AssignmentNullError("Assignment pair relation closure failed")
    return {
        "world_uid": world_uid,
        "seller_rows": seller_rows,
        "pair_rows": pair_rows,
        "analytic_denominator": derangement_count(SELLER_COUNT),
        "classifier_fitted": False,
    }
