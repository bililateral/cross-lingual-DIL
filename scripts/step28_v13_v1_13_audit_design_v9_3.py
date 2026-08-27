#!/usr/bin/env python3
"""Build the two-world V9.3 blind-audit schedules and registered plans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import hashlib
from typing import Any

import numpy as np

import step28_v13_common as common
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as signatures_module
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as constructor


VERSION = "2026-08-26-step28-v13-v1-13-blind-audit-design-v9-3"
SPLITS = ("audit_a", "audit_b")
WORLD_COUNT = 2
PUBLIC_SEEDS = {"audit_a": 281320260831, "audit_b": 281320260833}


class AuditDesignV93Error(common.ContractError):
    """Raised when a blind-audit abstract design drifts."""


def _with_self_hash(payload: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(payload)
    output["canonical_self_sha256"] = None
    output["canonical_self_sha256"] = common.canonical_sha256(output)
    return output


def _deterministic_permutation(*, split: str, world: int, purpose: str, counter: int) -> list[int]:
    if split not in SPLITS or not 0 <= world < WORLD_COUNT or counter < 0:
        raise AuditDesignV93Error("Blind-audit permutation coordinate drift")
    seed = int.from_bytes(
        hashlib.sha256(
            common.canonical_json_bytes(
                [VERSION, PUBLIC_SEEDS[split], split, world, purpose, counter]
            )
        ).digest()[:16],
        "big",
    )
    return [
        int(value)
        for value in np.random.Generator(np.random.PCG64(seed)).permutation(28)
    ]


def _schedule_world(*, split: str, world: int) -> dict[str, Any]:
    membership = _deterministic_permutation(
        split=split, world=world, purpose="membership", counter=0
    )
    noise = _deterministic_permutation(
        split=split, world=world, purpose="noise", counter=0
    )
    triads = [
        sorted(membership[offset : offset + 3])
        for offset in range(0, 12, 3)
    ]
    dyads = [
        sorted(membership[offset : offset + 2])
        for offset in range(12, 28, 2)
    ]
    groups = [
        *sorted(triads, key=lambda row: tuple(row)),
        *sorted(dyads, key=lambda row: tuple(row)),
    ]
    if sorted(value for group in groups for value in group) != list(range(28)):
        raise AuditDesignV93Error("Blind-audit membership is not a partition")
    return {
        "world_ordinal": world,
        "controller_groups": groups,
        "noise_slot_by_seller_slot": noise,
    }


def _controller_by_seller(schedule_world: Mapping[str, Any]) -> list[int]:
    controller = [-1] * 28
    for controller_index, group in enumerate(schedule_world["controller_groups"]):
        for seller in group:
            seller = int(seller)
            if not 0 <= seller < 28 or controller[seller] != -1:
                raise AuditDesignV93Error("Blind-audit controller group drift")
            controller[seller] = controller_index
    if any(value < 0 for value in controller):
        raise AuditDesignV93Error("Blind-audit controller partition is incomplete")
    return controller


def _valid_endpoint_row(
    row: list[int],
    *,
    schedule_world: Mapping[str, Any],
    signature_by_noise: Mapping[int, Mapping[str, Any]],
) -> bool:
    if len(row) != 12 or len(set(row)) != 12:
        return False
    controllers = _controller_by_seller(schedule_world)
    noise_by_seller = [int(value) for value in schedule_world["noise_slot_by_seller_slot"]]
    for pair_index, (left_position, right_position) in enumerate(constructor.PAIR_POSITIONS):
        left = int(row[left_position])
        right = int(row[right_position])
        if controllers[left] == controllers[right]:
            return False
        left_signature = signature_by_noise[noise_by_seller[left]]
        right_signature = signature_by_noise[noise_by_seller[right]]
        treatment = "exact_title_clone" if pair_index < 2 else "high_semantic_similarity"
        if not constructor._eligible_logical_item_ordinals(
            treatment=treatment, signature=left_signature
        ) or not constructor._eligible_logical_item_ordinals(
            treatment=treatment, signature=right_signature
        ):
            return False
        if pair_index < 2 and not any(
            title == description == "1"
            for signature in (left_signature, right_signature)
            for title, description in zip(
                signature["title_present_mask"],
                signature["description_present_mask"],
                strict=True,
            )
        ):
            return False
    return True


def _endpoint_row(
    *,
    split: str,
    world: int,
    schedule_world: Mapping[str, Any],
    signature_by_noise: Mapping[int, Mapping[str, Any]],
) -> tuple[list[int], int]:
    for counter in range(100_000):
        row = _deterministic_permutation(
            split=split, world=world, purpose="registered-endpoints", counter=counter
        )[:12]
        if _valid_endpoint_row(
            row,
            schedule_world=schedule_world,
            signature_by_noise=signature_by_noise,
        ):
            return row, counter
    raise AuditDesignV93Error("Blind-audit endpoint search exhausted")


@lru_cache(maxsize=2)
def _audit_semantic_asset_sequence(
    split: str,
) -> tuple[tuple[int, int, int, int, int], ...]:
    if split not in SPLITS:
        raise AuditDesignV93Error("Unknown blind-audit semantic split")
    category_counts = constructor._largest_remainder_counts(
        constructor.SEMANTIC_ASSET_COUNT_PER_SPLIT,
        constructor.SEMANTIC_CATEGORY_WEIGHTS,
    )
    categories = constructor._permuted_balanced_values(
        split=split, namespace="category", counts=category_counts
    )
    attributes = constructor._permuted_balanced_values(
        split=split,
        namespace="attribute",
        counts=(
            constructor.SEMANTIC_ASSET_COUNT_PER_SPLIT
            // constructor.SEMANTIC_ATTRIBUTE_COUNT,
        )
        * constructor.SEMANTIC_ATTRIBUTE_COUNT,
    )
    skeleton_pairs = tuple(
        (left, right)
        for left in range(constructor.SEMANTIC_TITLE_SKELETON_COUNT)
        for right in range(constructor.SEMANTIC_TITLE_SKELETON_COUNT)
        if left != right
    )
    skeleton_counts = constructor._largest_remainder_counts(
        constructor.SEMANTIC_ASSET_COUNT_PER_SPLIT,
        (1,) * len(skeleton_pairs),
    )
    skeleton_ordinals = constructor._permuted_balanced_values(
        split=split,
        namespace="title_skeleton_pair",
        counts=skeleton_counts,
    )
    category_seen = [0] * len(constructor.SEMANTIC_CATEGORY_PRODUCT_COUNTS)
    output: list[tuple[int, int, int, int, int]] = []
    for index, category in enumerate(categories):
        product_count = constructor.SEMANTIC_CATEGORY_PRODUCT_COUNTS[category]
        product_start = int.from_bytes(
            hashlib.sha256(
                "\0".join(
                    (
                        constructor.SEMANTIC_ASSET_SELECTOR_VERSION,
                        split,
                        "product_start",
                        str(category),
                    )
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        ) % product_count
        product = (product_start + category_seen[category]) % product_count
        category_seen[category] += 1
        left_skeleton, right_skeleton = skeleton_pairs[skeleton_ordinals[index]]
        output.append(
            (category, product, attributes[index], left_skeleton, right_skeleton)
        )
    return tuple(output)


def _audit_semantic_asset(
    *, split: str, world_ordinal: int, instance_ordinal: int
) -> dict[str, int]:
    index = world_ordinal * 4 + instance_ordinal
    category, product, attribute, left_skeleton, right_skeleton = (
        _audit_semantic_asset_sequence(split)[index]
    )
    return {
        "category_ordinal": category,
        "product_ordinal": product,
        "attribute_ordinal": attribute,
        "left_title_skeleton_ordinal": left_skeleton,
        "right_title_skeleton_ordinal": right_skeleton,
    }


def build_payload(joint_signatures: Mapping[str, Any]) -> dict[str, Any]:
    signatures_module.validate_payload(joint_signatures)
    signature_by_noise = {
        int(row["noise_slot"]): row["signature"]
        for row in joint_signatures["noise_slot_multiset"]
    }
    payload: dict[str, Any] = {
        "version": VERSION,
        "canonical_self_sha256": None,
        "joint_noise_signature_canonical_self_sha256": joint_signatures[
            "canonical_self_sha256"
        ],
        "public_seeds": dict(PUBLIC_SEEDS),
        "truth_or_model_result_read": False,
        "splits": {},
    }
    for split in SPLITS:
        schedule_worlds = [
            _schedule_world(split=split, world=world)
            for world in range(WORLD_COUNT)
        ]
        occurrence_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        plan_worlds: list[dict[str, Any]] = []
        search_counters: list[int] = []
        for world, schedule_world in enumerate(schedule_worlds):
            seller_row, search_counter = _endpoint_row(
                split=split,
                world=world,
                schedule_world=schedule_world,
                signature_by_noise=signature_by_noise,
            )
            search_counters.append(search_counter)
            noise_by_seller = schedule_world["noise_slot_by_seller_slot"]
            assignments: list[dict[str, Any]] = []
            for pair_index, (left_position, right_position) in enumerate(
                constructor.PAIR_POSITIONS
            ):
                if pair_index < 2:
                    treatment = "exact_title_clone"
                    instance_ordinal = pair_index
                    roles = ("source", "target")
                else:
                    treatment = "high_semantic_similarity"
                    instance_ordinal = pair_index - 2
                    roles = ("left", "right")
                endpoints: list[dict[str, Any]] = []
                for position, role in zip(
                    (left_position, right_position), roles, strict=True
                ):
                    seller_slot = int(seller_row[position])
                    signature = signature_by_noise[int(noise_by_seller[seller_slot])]
                    signature_text = constructor._canonical_signature_text(signature)
                    occurrence_key = (treatment, role, signature_text)
                    occurrence_index = occurrence_counts[occurrence_key]
                    occurrence_counts[occurrence_key] += 1
                    eligible = constructor._eligible_logical_item_ordinals(
                        treatment=treatment, signature=signature
                    )
                    start = constructor._logical_item_cycle_start(
                        split=split,
                        treatment=treatment,
                        role=role,
                        signature=signature,
                        modulus=len(eligible),
                    )
                    endpoints.append(
                        {
                            "role": role,
                            "seller_slot": seller_slot,
                            "logical_item_ordinal": eligible[
                                (start + occurrence_index) % len(eligible)
                            ],
                        }
                    )
                assignments.append(
                    {
                        "treatment": treatment,
                        "instance_ordinal": instance_ordinal,
                        "endpoints": endpoints,
                        "semantic_asset": (
                            None
                            if treatment == "exact_title_clone"
                            else _audit_semantic_asset(
                                split=split,
                                world_ordinal=world,
                                instance_ordinal=instance_ordinal,
                            )
                        ),
                    }
                )
            plan_worlds.append(
                {"world_ordinal": world, "assignments": assignments}
            )
        payload["splits"][split] = {
            "world_count": WORLD_COUNT,
            "endpoint_search_counters": search_counters,
            "schedule_worlds": schedule_worlds,
            "plan_worlds": plan_worlds,
        }
    return _with_self_hash(payload)


def validate_payload(
    payload: Mapping[str, Any], joint_signatures: Mapping[str, Any]
) -> dict[str, Any]:
    expected = build_payload(joint_signatures)
    if common.canonical_json_bytes(dict(payload)) != common.canonical_json_bytes(
        expected
    ):
        raise AuditDesignV93Error("Blind-audit design does not exactly replay")
    return {
        "canonical_self_sha256": expected["canonical_self_sha256"],
        "split_count": 2,
        "world_count": 4,
        "truth_or_model_result_read": False,
    }
