#!/usr/bin/env python3
"""Validate an abstract V9.3 registered-negative joint selection plan."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as noise_signatures


VERSION = (
    "2026-08-26-step28-v13-v1-13-registered-negative-plan-"
    "v9-3-training-ready-noise"
)
BOUNDED_RESIDUAL_VERSION = (
    "2026-08-27-step28-v13-v1-13-registered-negative-plan-"
    "v9-3-r2-user-accepted-residual-22"
)
BOUNDED_RESIDUAL_STATUS = (
    "PASS_DETERMINISTIC_USER_ACCEPTED_RESIDUAL_22_PENDING_STRUCTURE_GATE"
)
ITEM_SELECTOR_VERSION = "2026-08-25-step28-v13-v1-13-item-selector-v9-3"
SEMANTIC_POLICY_PATH = "schema/step28_v13_synthetic_chinese_dataset_policy.json"
SEMANTIC_POLICY_SHA256 = "ce18015199c864df0f76a240df782c331020e5e76d483c5440cea6a673c74729"
SEMANTIC_DOMAIN_SHA256 = "79e15d3757920fc2c03e1934f49de45100d5edb61dc9fe154308197f5776c539"
SEMANTIC_CATEGORY_PRODUCT_COUNTS = (2, 1, 2, 2, 1, 2, 1, 1)
SEMANTIC_CATEGORY_WEIGHTS = (
    279_543_120_782,
    191_368_123_410,
    127_175_533_935,
    58_396_734_260,
    41_082_852_696,
    40_316_625_550,
    38_065_064_649,
    224_051_944_718,
)
SEMANTIC_ATTRIBUTE_COUNT = 10
SEMANTIC_TITLE_SKELETON_COUNT = 8
SEMANTIC_ASSET_COUNT_PER_SPLIT = 2_000
SEMANTIC_ASSET_SELECTOR_VERSION = (
    "2026-08-25-step28-v13-v1-13-semantic-asset-selector-v9-3"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORLD_COUNT = 500
TREATMENTS = ("exact_title_clone", "high_semantic_similarity")
TREATMENT_INSTANCE_COUNTS = {
    "exact_title_clone": 2,
    "high_semantic_similarity": 4,
}
TREATMENT_ROLES = {
    "exact_title_clone": ("source", "target"),
    "high_semantic_similarity": ("left", "right"),
}
ROLE_ORDER = (
    ("exact_title_clone", "source"),
    ("exact_title_clone", "target"),
    ("high_semantic_similarity", "left"),
    ("high_semantic_similarity", "right"),
)
ROLE_ELIGIBILITY_PREDICATE_NAMES = (
    "clone_source_eligible",
    "clone_target_eligible",
    "semantic_left_eligible",
    "semantic_right_eligible",
)
PAIR_HISTOGRAMS = {
    "exact_title_clone": {2: 134, 3: 244},
    "high_semantic_similarity": {5: 268, 6: 110},
}
DIRECTED_PAIR_HISTOGRAMS = {
    "exact_title_clone": {1: 512, 2: 244},
    "high_semantic_similarity": {2: 268, 3: 488},
}
ROLE_HISTOGRAMS = {
    ("exact_title_clone", "source"): {35: 8, 36: 20},
    ("exact_title_clone", "target"): {35: 8, 36: 20},
    ("high_semantic_similarity", "left"): {71: 16, 72: 12},
    ("high_semantic_similarity", "right"): {71: 16, 72: 12},
}
ENDPOINT_HISTOGRAMS = {
    "exact_title_clone": {71: 16, 72: 12},
    "high_semantic_similarity": {142: 4, 143: 24},
}
ROLE_TRIAD_TOTALS = {
    ("exact_title_clone", "source"): 429,
    ("exact_title_clone", "target"): 429,
    ("high_semantic_similarity", "left"): 857,
    ("high_semantic_similarity", "right"): 857,
}


class RegisteredNegativePlanError(ValueError):
    """Raised when a registered-negative joint plan violates its contract."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_self_sha256(payload: Mapping[str, Any]) -> str:
    projection = copy.deepcopy(dict(payload))
    projection.pop("canonical_self_sha256", None)
    return canonical_sha256(projection)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    if set(value) != expected:
        raise RegisteredNegativePlanError(
            f"{name} key drift: expected={sorted(expected)} observed={sorted(value)}"
        )


def _require_slot(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < balanced.SELLER_SLOT_COUNT
    ):
        raise RegisteredNegativePlanError(f"{name} is not a valid slot ordinal")
    return value


def _histogram(values: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(value) for value in values).items()))


def _pair(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise RegisteredNegativePlanError("Registered negative contains a self-pair")
    return (left, right) if left < right else (right, left)


def _signature_key(signature: Mapping[str, Any]) -> str:
    return canonical_json_bytes(signature).decode("utf-8")


def _eligible_logical_item_ordinals(
    *,
    treatment: str,
    signature: Mapping[str, Any],
) -> list[int]:
    title_eligible = [
        index
        for index, bit in enumerate(signature["title_present_mask"])
        if bit == "1"
    ]
    if not title_eligible:
        raise RegisteredNegativePlanError("Selected noise slot is title-ineligible")
    if treatment != "exact_title_clone":
        return title_eligible
    title_and_description = [
        index
        for index in title_eligible
        if signature["description_present_mask"][index] == "1"
    ]
    return title_and_description or title_eligible


def _role_eligible_logical_item_ordinals(
    *,
    treatment: str,
    role: str,
    signature: Mapping[str, Any],
) -> list[int]:
    role_key = (treatment, role)
    if role_key not in ROLE_ORDER:
        raise RegisteredNegativePlanError(
            f"Unknown registered-negative endpoint role: {role_key}"
        )
    return _eligible_logical_item_ordinals(
        treatment=treatment,
        signature=signature,
    )


def _logical_item_cycle_start(
    *, split: str, treatment: str, role: str, signature: Mapping[str, Any], modulus: int
) -> int:
    material = "\0".join(
        (
            ITEM_SELECTOR_VERSION,
            split,
            treatment,
            role,
            canonical_json_bytes(signature).decode("utf-8"),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % modulus


def _largest_remainder_counts(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    denominator = sum(weights)
    counts = [total * weight // denominator for weight in weights]
    remainder_count = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(total * weights[index] % denominator), index),
    )
    for index in order[:remainder_count]:
        counts[index] += 1
    return tuple(counts)


def _permuted_balanced_values(
    *, split: str, namespace: str, counts: tuple[int, ...]
) -> tuple[int, ...]:
    tokens = [
        (value, occurrence)
        for value, count in enumerate(counts)
        for occurrence in range(count)
    ]
    tokens.sort(
        key=lambda token: hashlib.sha256(
            "\0".join(
                (
                    SEMANTIC_ASSET_SELECTOR_VERSION,
                    split,
                    namespace,
                    str(token[0]),
                    str(token[1]),
                )
            ).encode("utf-8")
        ).digest()
    )
    return tuple(value for value, _occurrence in tokens)


@lru_cache(maxsize=2)
def _semantic_asset_sequence(split: str) -> tuple[tuple[int, int, int, int, int], ...]:
    if split not in ("train", "development"):
        raise RegisteredNegativePlanError("Unknown semantic-asset split")
    categories = _permuted_balanced_values(
        split=split,
        namespace="category",
        counts=_largest_remainder_counts(
            SEMANTIC_ASSET_COUNT_PER_SPLIT, SEMANTIC_CATEGORY_WEIGHTS
        ),
    )
    attributes = _permuted_balanced_values(
        split=split,
        namespace="attribute",
        counts=(
            SEMANTIC_ASSET_COUNT_PER_SPLIT // SEMANTIC_ATTRIBUTE_COUNT,
        )
        * SEMANTIC_ATTRIBUTE_COUNT,
    )
    skeleton_pairs = [
        (left, right)
        for left in range(SEMANTIC_TITLE_SKELETON_COUNT)
        for right in range(SEMANTIC_TITLE_SKELETON_COUNT)
        if left != right
    ]
    skeleton_pair_ordinals = _permuted_balanced_values(
        split=split,
        namespace="title_skeleton_pair",
        counts=_largest_remainder_counts(
            SEMANTIC_ASSET_COUNT_PER_SPLIT, (1,) * len(skeleton_pairs)
        ),
    )
    category_seen = [0] * len(SEMANTIC_CATEGORY_PRODUCT_COUNTS)
    sequence = []
    for index, category in enumerate(categories):
        product_count = SEMANTIC_CATEGORY_PRODUCT_COUNTS[category]
        product_start = int.from_bytes(
            hashlib.sha256(
                "\0".join(
                    (
                        SEMANTIC_ASSET_SELECTOR_VERSION,
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
        left_skeleton, right_skeleton = skeleton_pairs[
            skeleton_pair_ordinals[index]
        ]
        sequence.append(
            (
                category,
                product,
                attributes[index],
                left_skeleton,
                right_skeleton,
            )
        )
    return tuple(sequence)


def _expected_semantic_asset(
    *, split: str, world_ordinal: int, instance_ordinal: int
) -> dict[str, int]:
    index = world_ordinal * 4 + instance_ordinal
    if not 0 <= index < SEMANTIC_ASSET_COUNT_PER_SPLIT:
        raise RegisteredNegativePlanError("Semantic asset index drift")
    category, product, attribute, left_skeleton, right_skeleton = (
        _semantic_asset_sequence(split)[index]
    )
    return {
        "category_ordinal": category,
        "product_ordinal": product,
        "attribute_ordinal": attribute,
        "left_title_skeleton_ordinal": left_skeleton,
        "right_title_skeleton_ordinal": right_skeleton,
    }


def replay_semantic_public_domain() -> dict[str, Any]:
    """Independently reopen and validate the public semantic decision domain."""
    policy_path = REPOSITORY_ROOT / SEMANTIC_POLICY_PATH
    raw = policy_path.read_bytes()
    observed_policy_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_policy_sha256 != SEMANTIC_POLICY_SHA256:
        raise RegisteredNegativePlanError(
            "Semantic policy file hash drift: "
            f"expected={SEMANTIC_POLICY_SHA256} observed={observed_policy_sha256}"
        )
    policy = json.loads(raw.decode("utf-8"))
    if not isinstance(policy, Mapping):
        raise RegisteredNegativePlanError("Semantic policy root is not an object")
    domain = policy.get("independent_replay_public_domains")
    if not isinstance(domain, Mapping):
        raise RegisteredNegativePlanError("Semantic public domain is missing")
    observed_domain_sha256 = canonical_sha256(domain)
    if observed_domain_sha256 != SEMANTIC_DOMAIN_SHA256:
        raise RegisteredNegativePlanError(
            "Semantic public-domain hash drift: "
            f"expected={SEMANTIC_DOMAIN_SHA256} observed={observed_domain_sha256}"
        )
    categories = domain.get("categories_in_registered_order")
    probabilities = domain.get("anonymous_category_rank_probability")
    products = domain.get("category_products")
    attributes = domain.get("attributes")
    skeleton_counts = domain.get("title_skeleton_count_by_split")
    if (
        not isinstance(categories, list)
        or not categories
        or len(categories) != len(set(categories))
        or not all(isinstance(value, str) and value for value in categories)
        or not isinstance(probabilities, list)
        or len(probabilities) != len(categories)
        or not isinstance(products, Mapping)
        or set(products) != set(categories)
        or not isinstance(attributes, list)
        or not isinstance(skeleton_counts, Mapping)
    ):
        raise RegisteredNegativePlanError("Semantic public-domain schema drift")
    try:
        category_integer_weights = tuple(
            int(format(float(value), ".12f").replace(".", ""))
            for value in probabilities
        )
    except (TypeError, ValueError) as exc:
        raise RegisteredNegativePlanError(
            "Semantic category-probability domain drift"
        ) from exc
    if category_integer_weights != SEMANTIC_CATEGORY_WEIGHTS:
        raise RegisteredNegativePlanError(
            "Semantic category-probability integer weights drift"
        )
    category_product_counts = []
    for category in categories:
        values = products[category]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise RegisteredNegativePlanError(
                f"Semantic product domain drift for category {category!r}"
            )
        category_product_counts.append(len(values))
    if tuple(category_product_counts) != SEMANTIC_CATEGORY_PRODUCT_COUNTS:
        raise RegisteredNegativePlanError("Semantic category/product cardinality drift")
    if (
        len(attributes) != SEMANTIC_ATTRIBUTE_COUNT
        or len(attributes) != len(set(attributes))
        or not all(isinstance(value, str) and value for value in attributes)
    ):
        raise RegisteredNegativePlanError("Semantic attribute domain drift")
    if any(
        skeleton_counts.get(split) != SEMANTIC_TITLE_SKELETON_COUNT
        for split in ("train", "development")
    ):
        raise RegisteredNegativePlanError("Semantic title-skeleton domain drift")
    return {
        "policy_path": SEMANTIC_POLICY_PATH,
        "policy_sha256": observed_policy_sha256,
        "domain_sha256": observed_domain_sha256,
        "category_product_counts": category_product_counts,
        "category_integer_weights": list(category_integer_weights),
        "attribute_count": len(attributes),
        "train_title_skeleton_count": skeleton_counts["train"],
        "development_title_skeleton_count": skeleton_counts["development"],
    }


def _schedule_world_context(
    raw_world: Mapping[str, Any]
) -> tuple[dict[int, int], dict[int, bool], tuple[int, ...]]:
    groups, noise = balanced._validate_world(
        raw_world, expected_ordinal=int(raw_world["world_ordinal"])
    )
    controller: dict[int, int] = {}
    triad: dict[int, bool] = {}
    for group_index, group in enumerate(groups):
        for seller_slot in group:
            controller[seller_slot] = group_index
            triad[seller_slot] = len(group) == 3
    return controller, triad, noise


def _validate_endpoint(
    row: Mapping[str, Any], *, expected_role: str, name: str
) -> tuple[int, int]:
    _require_exact_keys(
        row,
        {"role", "seller_slot", "logical_item_ordinal"},
        name=name,
    )
    if row["role"] != expected_role:
        raise RegisteredNegativePlanError(f"{name} role drift")
    seller_slot = _require_slot(row["seller_slot"], name=f"{name} seller_slot")
    ordinal = row["logical_item_ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise RegisteredNegativePlanError(f"{name} logical item ordinal drift")
    return seller_slot, ordinal


def validate_plan(
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    *,
    expected_version: str = VERSION,
    require_exact_balance: bool = True,
    success_status: str = (
        "PASS_ABSTRACT_REGISTERED_NEGATIVE_PLAN_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED"
    ),
) -> dict[str, Any]:
    schedule_audit = balanced.validate_schedule(schedule)
    signature_audit = noise_signatures.validate_payload(joint_signatures)
    semantic_domain_audit = replay_semantic_public_domain()
    _require_exact_keys(
        plan,
        {
            "version",
            "split",
            "world_count",
            "balanced_schedule_sha256",
            "joint_noise_signature_sha256",
            "semantic_domain_contract",
            "worlds",
            "canonical_self_sha256",
        },
        name="registered-negative plan",
    )
    if plan["version"] != expected_version:
        raise RegisteredNegativePlanError("Registered-negative plan version drift")
    split = plan["split"]
    if split not in ("train", "development") or split != schedule_audit["split"]:
        raise RegisteredNegativePlanError("Registered-negative split drift")
    if plan["world_count"] != WORLD_COUNT:
        raise RegisteredNegativePlanError("Registered-negative world count drift")
    if plan["balanced_schedule_sha256"] != schedule["canonical_self_sha256"]:
        raise RegisteredNegativePlanError("Registered-negative schedule pin drift")
    if (
        plan["joint_noise_signature_sha256"]
        != joint_signatures["canonical_self_sha256"]
    ):
        raise RegisteredNegativePlanError("Registered-negative noise-signature pin drift")
    expected_semantic_domain_contract = {
        "policy_path": SEMANTIC_POLICY_PATH,
        "policy_sha256": SEMANTIC_POLICY_SHA256,
        "domain_sha256": SEMANTIC_DOMAIN_SHA256,
        "category_product_counts": list(SEMANTIC_CATEGORY_PRODUCT_COUNTS),
        "category_integer_weights": list(SEMANTIC_CATEGORY_WEIGHTS),
        "attribute_count": SEMANTIC_ATTRIBUTE_COUNT,
        "title_skeleton_count": SEMANTIC_TITLE_SKELETON_COUNT,
        "asset_count_per_split": SEMANTIC_ASSET_COUNT_PER_SPLIT,
        "asset_selector_version": SEMANTIC_ASSET_SELECTOR_VERSION,
        "allocation": "integer_maximum_remainder_then_split_specific_sha256_permutation",
        "shared_fields_per_pair": [
            "category_ordinal",
            "product_ordinal",
            "attribute_ordinal",
        ],
        "distinct_fields_per_pair": [
            "left_title_skeleton_ordinal",
            "right_title_skeleton_ordinal",
        ],
    }
    if plan["semantic_domain_contract"] != expected_semantic_domain_contract:
        raise RegisteredNegativePlanError("Semantic public-domain contract drift")
    supplied_self = plan["canonical_self_sha256"]
    if (
        not isinstance(supplied_self, str)
        or len(supplied_self) != 64
        or supplied_self != canonical_self_sha256(plan)
    ):
        raise RegisteredNegativePlanError("Registered-negative canonical self-hash drift")
    raw_worlds = plan["worlds"]
    if not isinstance(raw_worlds, list) or len(raw_worlds) != WORLD_COUNT:
        raise RegisteredNegativePlanError("Registered-negative world rows drift")

    noise_slots = {
        int(row["noise_slot"]): row["signature"]
        for row in joint_signatures["noise_slot_multiset"]
    }
    seller_pairs = {treatment: Counter() for treatment in TREATMENTS}
    noise_pairs = {treatment: Counter() for treatment in TREATMENTS}
    seller_directed_pairs = {treatment: Counter() for treatment in TREATMENTS}
    noise_directed_pairs = {treatment: Counter() for treatment in TREATMENTS}
    seller_roles = {role: [0] * 28 for role in ROLE_ORDER}
    noise_roles = {role: [0] * 28 for role in ROLE_ORDER}
    seller_endpoints = {treatment: [0] * 28 for treatment in TREATMENTS}
    noise_endpoints = {treatment: [0] * 28 for treatment in TREATMENTS}
    role_triads = {role: 0 for role in ROLE_ORDER}
    seller_size_cross = {role: [[0, 0] for _slot in range(28)] for role in ROLE_ORDER}
    noise_size_cross = {role: [[0, 0] for _slot in range(28)] for role in ROLE_ORDER}
    logical_occurrences: dict[
        tuple[str, str, str], list[tuple[int, int, int, int, int]]
    ] = defaultdict(list)
    semantic_assets: list[dict[str, int]] = []

    for expected_world, (raw_plan_world, raw_schedule_world) in enumerate(
        zip(raw_worlds, schedule["worlds"])
    ):
        if not isinstance(raw_plan_world, Mapping):
            raise RegisteredNegativePlanError("Registered-negative world is not an object")
        _require_exact_keys(
            raw_plan_world,
            {"world_ordinal", "assignments"},
            name=f"plan world[{expected_world}]",
        )
        if raw_plan_world["world_ordinal"] != expected_world:
            raise RegisteredNegativePlanError("Registered-negative world order drift")
        assignments = raw_plan_world["assignments"]
        if not isinstance(assignments, list) or len(assignments) != 6:
            raise RegisteredNegativePlanError("Each world must register exactly six negatives")
        controller, triad, noise_by_seller = _schedule_world_context(raw_schedule_world)
        used_sellers: set[int] = set()
        cursor = 0
        for treatment in TREATMENTS:
            expected_roles = TREATMENT_ROLES[treatment]
            for instance_ordinal in range(TREATMENT_INSTANCE_COUNTS[treatment]):
                assignment = assignments[cursor]
                cursor += 1
                if not isinstance(assignment, Mapping):
                    raise RegisteredNegativePlanError("Assignment is not an object")
                _require_exact_keys(
                    assignment,
                    {
                        "treatment",
                        "instance_ordinal",
                        "endpoints",
                        "semantic_asset",
                    },
                    name=f"world[{expected_world}] assignment[{cursor - 1}]",
                )
                if (
                    assignment["treatment"] != treatment
                    or assignment["instance_ordinal"] != instance_ordinal
                ):
                    raise RegisteredNegativePlanError("Assignment canonical order drift")
                if treatment == "exact_title_clone":
                    if assignment["semantic_asset"] is not None:
                        raise RegisteredNegativePlanError(
                            "Exact-title clone must not carry a semantic asset"
                        )
                else:
                    expected_asset = _expected_semantic_asset(
                        split=split,
                        world_ordinal=expected_world,
                        instance_ordinal=instance_ordinal,
                    )
                    if assignment["semantic_asset"] != expected_asset:
                        raise RegisteredNegativePlanError(
                            "High-semantic public-domain asset replay drift"
                        )
                    if (
                        expected_asset["left_title_skeleton_ordinal"]
                        == expected_asset["right_title_skeleton_ordinal"]
                    ):
                        raise RegisteredNegativePlanError(
                            "High-semantic endpoints use the same title skeleton"
                        )
                    semantic_assets.append(expected_asset)
                endpoints = assignment["endpoints"]
                if not isinstance(endpoints, list) or len(endpoints) != 2:
                    raise RegisteredNegativePlanError("Assignment endpoint count drift")
                parsed = [
                    _validate_endpoint(
                        endpoint,
                        expected_role=role,
                        name=f"world[{expected_world}] {treatment}[{instance_ordinal}] {role}",
                    )
                    for endpoint, role in zip(endpoints, expected_roles)
                ]
                left_seller, right_seller = parsed[0][0], parsed[1][0]
                if left_seller in used_sellers or right_seller in used_sellers:
                    raise RegisteredNegativePlanError(
                        "A world does not have twelve distinct registered endpoints"
                    )
                used_sellers.update((left_seller, right_seller))
                if controller[left_seller] == controller[right_seller]:
                    raise RegisteredNegativePlanError(
                        "A registered hard-negative pair is not cross-controller"
                    )
                left_noise = noise_by_seller[left_seller]
                right_noise = noise_by_seller[right_seller]
                seller_pairs[treatment][_pair(left_seller, right_seller)] += 1
                noise_pairs[treatment][_pair(left_noise, right_noise)] += 1
                seller_directed_pairs[treatment][(left_seller, right_seller)] += 1
                noise_directed_pairs[treatment][(left_noise, right_noise)] += 1

                descriptions_present: list[bool] = []
                for (seller_slot, logical_ordinal), role in zip(parsed, expected_roles):
                    role_key = (treatment, role)
                    noise_slot = noise_by_seller[seller_slot]
                    signature = noise_slots[noise_slot]
                    title_mask = signature["title_present_mask"]
                    description_mask = signature["description_present_mask"]
                    role_eligible = _role_eligible_logical_item_ordinals(
                        treatment=treatment,
                        role=role,
                        signature=signature,
                    )
                    if logical_ordinal not in role_eligible:
                        raise RegisteredNegativePlanError(
                            "Registered endpoint selected a role-ineligible logical item"
                        )
                    descriptions_present.append(description_mask[logical_ordinal] == "1")
                    signature_key = _signature_key(signature)
                    logical_occurrences[(treatment, role, signature_key)].append(
                        (
                            expected_world,
                            instance_ordinal,
                            seller_slot,
                            noise_slot,
                            logical_ordinal,
                        )
                    )
                    seller_roles[role_key][seller_slot] += 1
                    noise_roles[role_key][noise_slot] += 1
                    seller_endpoints[treatment][seller_slot] += 1
                    noise_endpoints[treatment][noise_slot] += 1
                    size_index = 1 if triad[seller_slot] else 0
                    role_triads[role_key] += size_index
                    seller_size_cross[role_key][seller_slot][size_index] += 1
                    noise_size_cross[role_key][noise_slot][size_index] += 1
                if treatment == "exact_title_clone" and not any(descriptions_present):
                    raise RegisteredNegativePlanError(
                        "An exact-title clone would duplicate two description-empty item documents"
                    )
        if len(used_sellers) != 12:
            raise RegisteredNegativePlanError("Registered endpoint cardinality drift")

    pair_order = balanced.PAIR_ORDER
    for treatment in TREATMENTS:
        for coordinate_name, counters in (
            ("seller", seller_pairs),
            ("noise", noise_pairs),
        ):
            observed = _histogram([counters[treatment][pair] for pair in pair_order])
            if require_exact_balance and observed != PAIR_HISTOGRAMS[treatment]:
                raise RegisteredNegativePlanError(
                    f"{coordinate_name} {treatment} pair-balance drift: {observed}"
                )
        for coordinate_name, counters in (
            ("seller", seller_directed_pairs),
            ("noise", noise_directed_pairs),
        ):
            observed = _histogram(
                [
                    counters[treatment][(left, right)]
                    for left in range(28)
                    for right in range(28)
                    if left != right
                ]
            )
            if (
                require_exact_balance
                and observed != DIRECTED_PAIR_HISTOGRAMS[treatment]
            ):
                raise RegisteredNegativePlanError(
                    f"{coordinate_name} {treatment} directed-pair balance drift: {observed}"
                )
        for coordinate_name, values in (
            ("seller", seller_endpoints[treatment]),
            ("noise", noise_endpoints[treatment]),
        ):
            observed = _histogram(values)
            if require_exact_balance and observed != ENDPOINT_HISTOGRAMS[treatment]:
                raise RegisteredNegativePlanError(
                    f"{coordinate_name} {treatment} endpoint-balance drift: {observed}"
                )

    for role_key in ROLE_ORDER:
        for coordinate_name, values in (
            ("seller", seller_roles[role_key]),
            ("noise", noise_roles[role_key]),
        ):
            observed = _histogram(values)
            if require_exact_balance and observed != ROLE_HISTOGRAMS[role_key]:
                raise RegisteredNegativePlanError(
                    f"{coordinate_name} {role_key} role-balance drift: {observed}"
                )
        if (
            require_exact_balance
            and role_triads[role_key] != ROLE_TRIAD_TOTALS[role_key]
        ):
            raise RegisteredNegativePlanError(
                f"{role_key} triad/dyad exposure drift: {role_triads[role_key]}"
            )
        role_index = ROLE_ORDER.index(role_key)
        expected_bounds = (
            ((20, 21), (15, 16))
            if role_index < 2
            else ((40, 41), (30, 31))
        )
        for coordinate_name, table in (
            ("seller", seller_size_cross),
            ("noise", noise_size_cross),
        ):
            for slot, (dyad, triad_count) in enumerate(table[role_key]):
                if (
                    require_exact_balance
                    and not expected_bounds[0][0] <= dyad <= expected_bounds[0][1]
                ):
                    raise RegisteredNegativePlanError(
                        f"{coordinate_name} {role_key} slot {slot} dyad exposure drift"
                    )
                if (
                    require_exact_balance
                    and not expected_bounds[1][0]
                    <= triad_count
                    <= expected_bounds[1][1]
                ):
                    raise RegisteredNegativePlanError(
                        f"{coordinate_name} {role_key} slot {slot} triad exposure drift"
                    )

    for coordinate_name, roles in (("seller", seller_roles), ("noise", noise_roles)):
        clone_low_source = {
            slot for slot, value in enumerate(roles[("exact_title_clone", "source")]) if value == 35
        }
        clone_low_target = {
            slot for slot, value in enumerate(roles[("exact_title_clone", "target")]) if value == 35
        }
        semantic_high_left = {
            slot for slot, value in enumerate(roles[("high_semantic_similarity", "left")]) if value == 72
        }
        semantic_high_right = {
            slot for slot, value in enumerate(roles[("high_semantic_similarity", "right")]) if value == 72
        }
        if require_exact_balance and clone_low_source & clone_low_target:
            raise RegisteredNegativePlanError(
                f"{coordinate_name} clone low-count role sets overlap"
            )
        if require_exact_balance and semantic_high_left & semantic_high_right:
            raise RegisteredNegativePlanError(
                f"{coordinate_name} semantic high-count role sets overlap"
            )

    if len(semantic_assets) != WORLD_COUNT * TREATMENT_INSTANCE_COUNTS[
        "high_semantic_similarity"
    ]:
        raise RegisteredNegativePlanError("Semantic asset cardinality drift")
    category_counter = Counter(
        asset["category_ordinal"] for asset in semantic_assets
    )
    expected_category_counts = _largest_remainder_counts(
        SEMANTIC_ASSET_COUNT_PER_SPLIT, SEMANTIC_CATEGORY_WEIGHTS
    )
    observed_category_counts = tuple(
        category_counter[index]
        for index in range(len(SEMANTIC_CATEGORY_PRODUCT_COUNTS))
    )
    if observed_category_counts != expected_category_counts:
        raise RegisteredNegativePlanError("Semantic category-law balance drift")
    product_count_rows = []
    for category, product_count in enumerate(SEMANTIC_CATEGORY_PRODUCT_COUNTS):
        counter = Counter(
            asset["product_ordinal"]
            for asset in semantic_assets
            if asset["category_ordinal"] == category
        )
        counts = [counter[product] for product in range(product_count)]
        if max(counts) - min(counts) > 1 or sum(counts) != category_counter[category]:
            raise RegisteredNegativePlanError(
                "Semantic conditional product balance drift"
            )
        product_count_rows.append(counts)
    attribute_counter = Counter(
        asset["attribute_ordinal"] for asset in semantic_assets
    )
    attribute_histogram = _histogram(list(attribute_counter.values()))
    if attribute_histogram != {200: 10}:
        raise RegisteredNegativePlanError("Semantic attribute balance drift")
    skeleton_pair_counter = Counter(
        (
            asset["left_title_skeleton_ordinal"],
            asset["right_title_skeleton_ordinal"],
        )
        for asset in semantic_assets
    )
    skeleton_pair_histogram = _histogram(list(skeleton_pair_counter.values()))
    if skeleton_pair_histogram != {35: 16, 36: 40}:
        raise RegisteredNegativePlanError("Semantic skeleton-pair balance drift")

    logical_selector_rows = []
    for (treatment, role, signature_key), occurrences in sorted(
        logical_occurrences.items(), key=lambda row: tuple(value.encode("utf-8") for value in row[0])
    ):
        signature = json.loads(signature_key)
        eligible = _eligible_logical_item_ordinals(
            treatment=treatment, signature=signature
        )
        start = _logical_item_cycle_start(
            split=split,
            treatment=treatment,
            role=role,
            signature=signature,
            modulus=len(eligible),
        )
        ordinal_counts: Counter[int] = Counter()
        for occurrence_index, occurrence in enumerate(occurrences):
            expected_ordinal = eligible[(start + occurrence_index) % len(eligible)]
            if occurrence[4] != expected_ordinal:
                raise RegisteredNegativePlanError(
                    "Logical-item stratified cycle replay drift"
                )
            ordinal_counts[expected_ordinal] += 1
        if max(ordinal_counts.values()) - min(ordinal_counts.values()) > 1:
            raise RegisteredNegativePlanError(
                "Logical-item cycle is not floor/ceiling balanced"
            )
        logical_selector_rows.append(
            {
                "treatment": treatment,
                "role": role,
                "signature_sha256": hashlib.sha256(signature_key.encode("utf-8")).hexdigest(),
                "eligible_logical_item_ordinals": eligible,
                "cycle_start_index": start,
                "occurrence_count": sum(ordinal_counts.values()),
                "ordinal_count_histogram": {
                    str(key): value for key, value in _histogram(list(ordinal_counts.values())).items()
                },
            }
        )

    def cross_table_payload(
        table: Mapping[tuple[str, str], list[list[int]]]
    ) -> dict[str, list[dict[str, int]]]:
        return {
            f"{treatment}:{role}": [
                {"slot": slot, "dyad": values[0], "triad": values[1]}
                for slot, values in enumerate(table[(treatment, role)])
            ]
            for treatment, role in ROLE_ORDER
        }

    seller_cross_payload = cross_table_payload(seller_size_cross)
    noise_cross_payload = cross_table_payload(noise_size_cross)
    observed_pair_histograms = {
        coordinate_name: {
            treatment: _histogram(
                [counters[treatment][pair] for pair in pair_order]
            )
            for treatment in TREATMENTS
        }
        for coordinate_name, counters in (
            ("seller", seller_pairs),
            ("noise", noise_pairs),
        )
    }
    observed_directed_pair_histograms = {
        coordinate_name: {
            treatment: _histogram(
                [
                    counters[treatment][(left, right)]
                    for left in range(28)
                    for right in range(28)
                    if left != right
                ]
            )
            for treatment in TREATMENTS
        }
        for coordinate_name, counters in (
            ("seller", seller_directed_pairs),
            ("noise", noise_directed_pairs),
        )
    }
    observed_role_histograms = {
        coordinate_name: {
            role_key: _histogram(values[role_key])
            for role_key in ROLE_ORDER
        }
        for coordinate_name, values in (
            ("seller", seller_roles),
            ("noise", noise_roles),
        )
    }
    observed_endpoint_histograms = {
        coordinate_name: {
            treatment: _histogram(values[treatment])
            for treatment in TREATMENTS
        }
        for coordinate_name, values in (
            ("seller", seller_endpoints),
            ("noise", noise_endpoints),
        )
    }

    def stringify_histogram(histogram: Mapping[int, int]) -> dict[str, int]:
        return {str(key): value for key, value in histogram.items()}

    return {
        "version": expected_version,
        "split": split,
        "world_count": WORLD_COUNT,
        "registered_pair_count_per_world": 6,
        "registered_endpoint_count_per_world": 12,
        "pair_histograms": {
            treatment: stringify_histogram(
                observed_pair_histograms["seller"][treatment]
            )
            for treatment in TREATMENTS
        },
        "noise_pair_histograms": {
            treatment: stringify_histogram(
                observed_pair_histograms["noise"][treatment]
            )
            for treatment in TREATMENTS
        },
        "directed_pair_histograms": {
            treatment: stringify_histogram(
                observed_directed_pair_histograms["seller"][treatment]
            )
            for treatment in TREATMENTS
        },
        "noise_directed_pair_histograms": {
            treatment: stringify_histogram(
                observed_directed_pair_histograms["noise"][treatment]
            )
            for treatment in TREATMENTS
        },
        "role_histograms": {
            f"{treatment}:{role}": stringify_histogram(
                observed_role_histograms["seller"][(treatment, role)]
            )
            for treatment, role in ROLE_ORDER
        },
        "noise_role_histograms": {
            f"{treatment}:{role}": stringify_histogram(
                observed_role_histograms["noise"][(treatment, role)]
            )
            for treatment, role in ROLE_ORDER
        },
        "endpoint_histograms": {
            treatment: stringify_histogram(
                observed_endpoint_histograms["seller"][treatment]
            )
            for treatment in TREATMENTS
        },
        "noise_endpoint_histograms": {
            treatment: stringify_histogram(
                observed_endpoint_histograms["noise"][treatment]
            )
            for treatment in TREATMENTS
        },
        "role_triad_totals": {
            f"{treatment}:{role}": role_triads[(treatment, role)]
            for treatment, role in ROLE_ORDER
        },
        "exact_balance_required": require_exact_balance,
        "role_eligibility_predicates": list(ROLE_ELIGIBILITY_PREDICATE_NAMES),
        "seller_role_controller_size_cross_table_sha256": canonical_sha256(
            seller_cross_payload
        ),
        "noise_role_controller_size_cross_table_sha256": canonical_sha256(
            noise_cross_payload
        ),
        "seller_role_controller_size_cross_table": seller_cross_payload,
        "noise_role_controller_size_cross_table": noise_cross_payload,
        "logical_item_selector_rows": logical_selector_rows,
        "logical_item_selector_rows_sha256": canonical_sha256(logical_selector_rows),
        "semantic_asset_count": len(semantic_assets),
        "semantic_assets_sha256": canonical_sha256(semantic_assets),
        "semantic_category_counts": list(observed_category_counts),
        "semantic_expected_category_counts": list(expected_category_counts),
        "semantic_conditional_product_counts": product_count_rows,
        "semantic_attribute_count_histogram": {
            str(key): value for key, value in attribute_histogram.items()
        },
        "semantic_skeleton_pair_count_histogram": {
            str(key): value for key, value in skeleton_pair_histogram.items()
        },
        "semantic_public_domain_replay": semantic_domain_audit,
        "plan_canonical_self_sha256": supplied_self,
        "balanced_schedule_sha256": plan["balanced_schedule_sha256"],
        "joint_noise_signature_sha256": signature_audit["canonical_self_sha256"],
        "status": success_status,
    }


def _global_relabel_audit(
    left: Sequence[int], right: Sequence[int], *, cardinality: int
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise RegisteredNegativePlanError("Relabel audit sequence length drift")
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    conflict_count = 0
    exact_position_match_count = 0
    for left_value, right_value in zip(left, right):
        if (
            isinstance(left_value, bool)
            or isinstance(right_value, bool)
            or not isinstance(left_value, int)
            or not isinstance(right_value, int)
            or not 0 <= left_value < cardinality
            or not 0 <= right_value < cardinality
        ):
            raise RegisteredNegativePlanError("Relabel audit value-domain drift")
        exact_position_match_count += int(left_value == right_value)
        if left_value in forward and forward[left_value] != right_value:
            conflict_count += 1
        if right_value in reverse and reverse[right_value] != left_value:
            conflict_count += 1
        forward.setdefault(left_value, right_value)
        reverse.setdefault(right_value, left_value)
    complete_bijection = (
        len(forward) == cardinality
        and len(reverse) == cardinality
        and conflict_count == 0
    )
    return {
        "sequence_length": len(left),
        "exact_position_match_count": exact_position_match_count,
        "mapping_conflict_count": conflict_count,
        "forward_domain_size": len(forward),
        "reverse_domain_size": len(reverse),
        "complete_global_relabel": complete_bijection,
        "candidate_mapping_sha256": canonical_sha256(
            [[key, forward[key]] for key in sorted(forward)]
        ),
    }


def _endpoint_sequences(
    plan: Mapping[str, Any], schedule: Mapping[str, Any]
) -> tuple[list[int], list[int], list[int]]:
    seller_sequence: list[int] = []
    noise_sequence: list[int] = []
    logical_item_sequence: list[int] = []
    for plan_world, schedule_world in zip(plan["worlds"], schedule["worlds"]):
        _controllers, _triads, noise_by_seller = _schedule_world_context(
            schedule_world
        )
        for assignment in plan_world["assignments"]:
            for endpoint in assignment["endpoints"]:
                seller_slot = int(endpoint["seller_slot"])
                seller_sequence.append(seller_slot)
                noise_sequence.append(int(noise_by_seller[seller_slot]))
                logical_item_sequence.append(int(endpoint["logical_item_ordinal"]))
    return seller_sequence, noise_sequence, logical_item_sequence


def validate_train_development_plan_pair(
    train_plan: Mapping[str, Any],
    development_plan: Mapping[str, Any],
    train_schedule: Mapping[str, Any],
    development_schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    *,
    expected_version: str = VERSION,
    require_exact_balance: bool = True,
    plan_success_status: str = (
        "PASS_ABSTRACT_REGISTERED_NEGATIVE_PLAN_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED"
    ),
    pair_success_status: str = (
        "PASS_INDEPENDENT_SPLIT_PLAN_PAIR_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED"
    ),
) -> dict[str, Any]:
    """Reject exact or globally relabelled reuse across the two split plans."""
    train_audit = validate_plan(
        train_plan,
        train_schedule,
        joint_signatures,
        expected_version=expected_version,
        require_exact_balance=require_exact_balance,
        success_status=plan_success_status,
    )
    development_audit = validate_plan(
        development_plan,
        development_schedule,
        joint_signatures,
        expected_version=expected_version,
        require_exact_balance=require_exact_balance,
        success_status=plan_success_status,
    )
    if train_plan["split"] != "train" or development_plan["split"] != "development":
        raise RegisteredNegativePlanError("Plan-pair split order drift")
    if train_plan["canonical_self_sha256"] == development_plan["canonical_self_sha256"]:
        raise RegisteredNegativePlanError("Train/development plans are byte-identical")
    train_seller, train_noise, train_items = _endpoint_sequences(
        train_plan, train_schedule
    )
    development_seller, development_noise, development_items = _endpoint_sequences(
        development_plan, development_schedule
    )
    seller_relabel = _global_relabel_audit(
        train_seller, development_seller, cardinality=balanced.SELLER_SLOT_COUNT
    )
    noise_relabel = _global_relabel_audit(
        train_noise, development_noise, cardinality=balanced.SELLER_SLOT_COUNT
    )
    if seller_relabel["complete_global_relabel"]:
        raise RegisteredNegativePlanError(
            "Development registered-negative plan is a global seller-slot relabel of train"
        )
    if noise_relabel["complete_global_relabel"]:
        raise RegisteredNegativePlanError(
            "Development registered-negative plan is a global noise-slot relabel of train"
        )
    return {
        "train_plan_sha256": train_plan["canonical_self_sha256"],
        "development_plan_sha256": development_plan["canonical_self_sha256"],
        "seller_endpoint_relabel_audit": seller_relabel,
        "noise_endpoint_relabel_audit": noise_relabel,
        "train_logical_item_sequence_sha256": canonical_sha256(train_items),
        "development_logical_item_sequence_sha256": canonical_sha256(
            development_items
        ),
        "logical_item_sequence_exact_match_count": sum(
            int(left == right)
            for left, right in zip(train_items, development_items)
        ),
        "train_audit_sha256": canonical_sha256(train_audit),
        "development_audit_sha256": canonical_sha256(development_audit),
        "status": pair_success_status,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegisteredNegativePlanError(f"JSON root must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--joint-signatures", required=True, type=Path)
    args = parser.parse_args()
    audit = validate_plan(
        load_json(args.plan),
        load_json(args.schedule),
        load_json(args.joint_signatures),
    )
    print(canonical_json_bytes(audit).decode("utf-8"))


if __name__ == "__main__":
    main()
