#!/usr/bin/env python3
"""Construct V9.3 abstract registered-negative plans by one public trajectory."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as noise_signatures
import step28_v13_v1_13_registered_negative_plan_v9_3 as plan_contract


VERSION = (
    "2026-08-26-step28-v13-v1-13-construct-registered-negative-plan-"
    "v9-3-successor-unassigned-stop-guard"
)
PUBLIC_DESIGN_SEEDS = {
    "train": 281320260826,
    "development": 281320260828,
}
FORMAL_OUTPUT_RELATIVE = Path(
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "__successor_registered_negative_plan_unassigned__"
)
FORMAL_INPUT_PINS = {
    "train_schedule": (
        Path(
            "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
            "design_preflight_v2_20260825/train_balanced_schedule.json"
        ),
        "c4a058fb97e351b033d2f4c595985251cf790b1693123bdec01fc64bcc18a4c2",
    ),
    "development_schedule": (
        Path(
            "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
            "design_preflight_v2_20260825/development_balanced_schedule.json"
        ),
        "d29197a5c052e6a450ea838291f740fbc39e434dd754d2bd3c91dd2bf5ba4087",
    ),
    "joint_signatures": (
        Path(
            "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
            "joint_noise_signature_preflight_v2_20260826.json"
        ),
        "05caa6a2b19591fdbf042384b1d31dccfde4a3f1096275ec44671f9f4b3c4d37",
    ),
}
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
ANNEALING_ITERATIONS = 40_000_000
COLD_POLISHING_ITERATIONS = 15_000_000
MAXIMUM_ITERATIONS = ANNEALING_ITERATIONS + COLD_POLISHING_ITERATIONS
LOCAL_REPAIR_MAX_ROUNDS = 128
LOCAL_REPAIR_MAX_PROPOSALS_PER_ROUND = 500_000
LOCAL_REPAIR_MAX_CANDIDATES_PER_ROUND = 20_000
LOCAL_REPAIR_MAX_TARGETED_CANDIDATES_PER_CELL = 180
LOCAL_REPAIR_MAX_COMPENSATING_CANDIDATES_PER_ROUND = 4_000
LOCAL_REPAIR_COARSE_MILP_THRESHOLD = 22
LOCAL_REPAIR_COARSE_MILP_MAX_HELPFUL_CANDIDATES = 1_500
LOCAL_REPAIR_COARSE_MILP_MAX_COMPENSATING_CANDIDATES = 500
LOCAL_REPAIR_COARSE_MILP_TIME_LIMIT_SECONDS = 300
LOCAL_REPAIR_COARSE_MILP_SLACK_COST = 10_000.0
LOCAL_REPAIR_MAX_POSITIVE_DELTA = 4
LOCAL_REPAIR_TARGETED_MAX_POSITIVE_DELTA = 12
ROLE_BY_POSITION = np.asarray(
    [0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3], dtype=np.int8
)
TREATMENT_BY_POSITION = np.asarray(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8
)
PAIR_BY_POSITION = np.asarray(
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int8
)
PAIR_POSITIONS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
ROLE_NAMES = (
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


class RegisteredNegativeConstructionError(RuntimeError):
    """Raised when the abstract joint construction cannot be published."""


RepairCandidate = tuple[
    dict[tuple[str, tuple[int, ...]], int],
    dict[int, np.ndarray],
    int,
    str,
]


def _solve_sparse_candidate_milp(
    *,
    current_bounds: list[tuple[int, int, int]],
    candidate_cell_changes: list[dict[int, int]],
    candidate_worlds: list[tuple[int, ...]],
    candidate_costs: list[float],
    time_limit_seconds: int,
    slack_unit_cost_override: float | None = None,
) -> dict[str, Any]:
    """Lexicographically minimize total slack then deterministic move cost."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    candidate_count = len(candidate_cell_changes)
    cell_count = len(current_bounds)
    if (
        not candidate_count
        or len(candidate_worlds) != candidate_count
        or len(candidate_costs) != candidate_count
    ):
        raise RegisteredNegativeConstructionError("MILP candidate input drift")
    variable_count = candidate_count + cell_count
    lower_row_offset = 0
    upper_row_offset = cell_count
    world_row_offset = 2 * cell_count
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    for candidate_index, (changes, worlds) in enumerate(
        zip(candidate_cell_changes, candidate_worlds)
    ):
        if not worlds or len(worlds) != len(set(worlds)):
            raise RegisteredNegativeConstructionError("MILP candidate world drift")
        for cell_index, coefficient in changes.items():
            if not 0 <= cell_index < cell_count or not coefficient:
                raise RegisteredNegativeConstructionError("MILP cell change drift")
            row_indices.extend(
                (lower_row_offset + cell_index, upper_row_offset + cell_index)
            )
            column_indices.extend((candidate_index, candidate_index))
            coefficients.extend((float(coefficient), float(coefficient)))
        for world in worlds:
            if not 0 <= world < balanced.WORLD_COUNT:
                raise RegisteredNegativeConstructionError("MILP world index drift")
            row_indices.append(world_row_offset + world)
            column_indices.append(candidate_index)
            coefficients.append(1.0)
    for cell_index in range(cell_count):
        slack_column = candidate_count + cell_index
        row_indices.extend(
            (lower_row_offset + cell_index, upper_row_offset + cell_index)
        )
        column_indices.extend((slack_column, slack_column))
        coefficients.extend((1.0, -1.0))
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    for current, lower, _upper in current_bounds:
        constraint_lower.append(float(lower - current))
        constraint_upper.append(np.inf)
    for current, _lower, upper in current_bounds:
        constraint_lower.append(-np.inf)
        constraint_upper.append(float(upper - current))
    constraint_lower.extend([0.0] * balanced.WORLD_COUNT)
    constraint_upper.extend([1.0] * balanced.WORLD_COUNT)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(2 * cell_count + balanced.WORLD_COUNT, variable_count),
        dtype=np.float64,
    ).tocsc()
    costs = np.empty(variable_count, dtype=np.float64)
    costs[:candidate_count] = np.asarray(candidate_costs, dtype=np.float64)
    if np.any(costs[:candidate_count] <= 0.0) or not np.all(
        np.isfinite(costs[:candidate_count])
    ):
        raise RegisteredNegativeConstructionError(
            "MILP candidate costs must be finite and strictly positive"
        )
    if slack_unit_cost_override is not None:
        if (
            not math.isfinite(float(slack_unit_cost_override))
            or float(slack_unit_cost_override) <= 0.0
        ):
            raise RegisteredNegativeConstructionError(
                "MILP slack cost override must be finite and positive"
            )
        slack_cost = float(slack_unit_cost_override)
    else:
        # World mutual exclusion allows at most 500 selected columns. One
        # extra slack unit therefore dominates the largest possible secondary
        # move cost without scaling by every unselected column in the pool.
        maximum_secondary_cost = sum(
            sorted(map(float, costs[:candidate_count]), reverse=True)[
                : balanced.WORLD_COUNT
            ]
        )
        slack_cost = float(maximum_secondary_cost + 1.0)
    costs[candidate_count:] = slack_cost
    lower_variable_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_variable_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    upper_variable_bounds[:candidate_count] = 1.0
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:candidate_count] = 1
    result = milp(
        c=costs,
        integrality=integrality,
        bounds=Bounds(lower_variable_bounds, upper_variable_bounds),
        constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
        options={
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    feasible = result.x is not None
    if feasible and any(
        not math.isclose(
            float(value),
            float(round(float(value))),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for value in result.x[:candidate_count]
    ):
        raise RegisteredNegativeConstructionError(
            "MILP returned a nonintegral candidate incumbent"
        )
    selected_candidate_indices = (
        [
            index
            for index, value in enumerate(result.x[:candidate_count])
            if float(value) > 0.5
        ]
        if feasible
        else []
    )
    total_slack = (
        None
        if not feasible
        else int(round(float(np.sum(result.x[candidate_count:]))))
    )
    if feasible and not math.isclose(
        float(np.sum(result.x[candidate_count:])),
        float(total_slack),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RegisteredNegativeConstructionError(
            "MILP returned nonintegral residual slack"
        )
    return {
        "feasible": feasible,
        "selected_candidate_indices": selected_candidate_indices,
        "total_slack": total_slack,
        "slack_cost": slack_cost,
        "solver_status": int(result.status),
        "solver_status_name": str(result.message),
        "solver_objective": None if result.fun is None else float(result.fun),
        "solver_best_objective_bound": None,
        "solver_wall_time_seconds": None,
        "solver_conflict_count": None,
        "solver_branch_count": None,
        "scipy_version": __import__("scipy").__version__,
    }

def _penalty(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return (lower - value) ** 2
    if value > upper:
        return (value - upper) ** 2
    return 0


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _canonical_signature_text(signature: Mapping[str, Any]) -> str:
    return json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
        raise RegisteredNegativeConstructionError(
            "A selected noise slot has no title-eligible logical item"
        )
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
    """Return the endpoint-role-specific logical-item eligibility authority."""

    role_key = (treatment, role)
    if role_key not in ROLE_NAMES:
        raise RegisteredNegativeConstructionError(
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
            _canonical_signature_text(signature),
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
    if split not in PUBLIC_DESIGN_SEEDS:
        raise RegisteredNegativeConstructionError("Unknown semantic-asset split")
    category_counts = _largest_remainder_counts(
        SEMANTIC_ASSET_COUNT_PER_SPLIT, SEMANTIC_CATEGORY_WEIGHTS
    )
    categories = _permuted_balanced_values(
        split=split, namespace="category", counts=category_counts
    )
    attribute_counts = (SEMANTIC_ASSET_COUNT_PER_SPLIT // SEMANTIC_ATTRIBUTE_COUNT,) * (
        SEMANTIC_ATTRIBUTE_COUNT
    )
    attributes = _permuted_balanced_values(
        split=split, namespace="attribute", counts=attribute_counts
    )
    skeleton_pairs = [
        (left, right)
        for left in range(SEMANTIC_TITLE_SKELETON_COUNT)
        for right in range(SEMANTIC_TITLE_SKELETON_COUNT)
        if left != right
    ]
    skeleton_pair_counts = _largest_remainder_counts(
        SEMANTIC_ASSET_COUNT_PER_SPLIT, (1,) * len(skeleton_pairs)
    )
    skeleton_pair_ordinals = _permuted_balanced_values(
        split=split, namespace="title_skeleton_pair", counts=skeleton_pair_counts
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


def _semantic_asset(*, split: str, world_ordinal: int, instance_ordinal: int) -> dict[str, int]:
    index = world_ordinal * 4 + instance_ordinal
    if not 0 <= index < SEMANTIC_ASSET_COUNT_PER_SPLIT:
        raise RegisteredNegativeConstructionError("Semantic asset index drift")
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


def _world_arrays(
    schedule: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    balanced.validate_schedule(schedule)
    controllers = np.zeros((balanced.WORLD_COUNT, 28), dtype=np.int8)
    triads = np.zeros((balanced.WORLD_COUNT, 28), dtype=np.int8)
    noise = np.zeros((balanced.WORLD_COUNT, 28), dtype=np.int8)
    for world_ordinal, raw_world in enumerate(schedule["worlds"]):
        groups, world_noise = balanced._validate_world(
            raw_world, expected_ordinal=world_ordinal
        )
        for group_index, group in enumerate(groups):
            for seller_slot in group:
                controllers[world_ordinal, seller_slot] = group_index
                triads[world_ordinal, seller_slot] = int(len(group) == 3)
        noise[world_ordinal] = np.asarray(world_noise, dtype=np.int8)
    return controllers, triads, noise


class JointSearch:
    """Incremental simulated annealing over 500 six-edge matchings."""

    def __init__(
        self,
        schedule: Mapping[str, Any],
        joint_signatures: Mapping[str, Any],
        *,
        split: str,
    ) -> None:
        if split not in PUBLIC_DESIGN_SEEDS:
            raise RegisteredNegativeConstructionError("Unknown frozen search split")
        noise_signatures.validate_payload(joint_signatures)
        self.split = split
        self.public_design_seed = PUBLIC_DESIGN_SEEDS[split]
        self.rng = np.random.Generator(np.random.PCG64(self.public_design_seed))
        self.controllers, self.triads, self.noise = _world_arrays(schedule)
        self.noise_slot_signatures = {
            int(row["noise_slot"]): row["signature"]
            for row in joint_signatures["noise_slot_multiset"]
        }
        self.clone_description_capable = np.asarray(
            [
                any(
                    title == description == "1"
                    for title, description in zip(
                        self.noise_slot_signatures[noise_slot]["title_present_mask"],
                        self.noise_slot_signatures[noise_slot]["description_present_mask"],
                    )
                )
                for noise_slot in range(28)
            ],
            dtype=np.bool_,
        )
        self.role_eligible = np.zeros(
            (balanced.WORLD_COUNT, len(ROLE_BY_POSITION), 28),
            dtype=np.bool_,
        )
        for world in range(balanced.WORLD_COUNT):
            for position in range(12):
                role_index = int(ROLE_BY_POSITION[position])
                treatment, role = ROLE_NAMES[role_index]
                for seller_slot in range(28):
                    noise_slot = int(self.noise[world, seller_slot])
                    signature = self.noise_slot_signatures[noise_slot]
                    self.role_eligible[world, position, seller_slot] = bool(
                        _role_eligible_logical_item_ordinals(
                            treatment=treatment,
                            role=role,
                            signature=signature,
                        )
                    )
        self.assignments = np.zeros((balanced.WORLD_COUNT, 12), dtype=np.int8)
        for world in range(balanced.WORLD_COUNT):
            while True:
                row = self.rng.permutation(28)[:12].astype(np.int8, copy=False)
                if self._valid_row(world, row):
                    self.assignments[world] = row
                    break

        self.pair_seller = np.zeros((2, 28, 28), dtype=np.int16)
        self.pair_noise = np.zeros((2, 28, 28), dtype=np.int16)
        self.directed_seller = np.zeros((2, 28, 28), dtype=np.int16)
        self.directed_noise = np.zeros((2, 28, 28), dtype=np.int16)
        self.role_seller = np.zeros((4, 28), dtype=np.int16)
        self.role_noise = np.zeros((4, 28), dtype=np.int16)
        self.endpoint_seller = np.zeros((2, 28), dtype=np.int16)
        self.endpoint_noise = np.zeros((2, 28), dtype=np.int16)
        self.role_triad = np.zeros(4, dtype=np.int16)
        self.role_size_seller = np.zeros((4, 28, 2), dtype=np.int16)
        self.role_size_noise = np.zeros((4, 28, 2), dtype=np.int16)
        self.arrays = {
            "pair_seller": self.pair_seller,
            "pair_noise": self.pair_noise,
            "directed_seller": self.directed_seller,
            "directed_noise": self.directed_noise,
            "role_seller": self.role_seller,
            "role_noise": self.role_noise,
            "endpoint_seller": self.endpoint_seller,
            "endpoint_noise": self.endpoint_noise,
            "role_triad": self.role_triad,
            "role_size_seller": self.role_size_seller,
            "role_size_noise": self.role_size_noise,
        }
        for world, row in enumerate(self.assignments):
            self._accumulate_world(world, row, 1)
        self.objective = self._full_objective()

    def _accumulate_world(self, world: int, row: np.ndarray, delta: int) -> None:
        for pair_index, (left_position, right_position) in enumerate(PAIR_POSITIONS):
            treatment = 0 if pair_index < 2 else 1
            left_seller = int(row[left_position])
            right_seller = int(row[right_position])
            seller_pair = _pair(left_seller, right_seller)
            noise_pair = _pair(
                int(self.noise[world, left_seller]),
                int(self.noise[world, right_seller]),
            )
            self.pair_seller[(treatment, *seller_pair)] += delta
            self.pair_noise[(treatment, *noise_pair)] += delta
            self.directed_seller[treatment, left_seller, right_seller] += delta
            self.directed_noise[
                treatment,
                int(self.noise[world, left_seller]),
                int(self.noise[world, right_seller]),
            ] += delta
        for position, seller_value in enumerate(row):
            seller = int(seller_value)
            noise = int(self.noise[world, seller])
            role = int(ROLE_BY_POSITION[position])
            treatment = int(TREATMENT_BY_POSITION[position])
            self.role_seller[role, seller] += delta
            self.role_noise[role, noise] += delta
            self.endpoint_seller[treatment, seller] += delta
            self.endpoint_noise[treatment, noise] += delta
            self.role_triad[role] += delta * int(self.triads[world, seller])
            size_index = int(self.triads[world, seller])
            self.role_size_seller[role, seller, size_index] += delta
            self.role_size_noise[role, noise, size_index] += delta

    @staticmethod
    def _bounds(family: str, index: tuple[int, ...]) -> tuple[int, int]:
        if family == "role_triad":
            target = 429 if index[0] < 2 else 857
            return target, target
        if family.startswith("directed_"):
            return (1, 2) if index[0] == 0 else (2, 3)
        if family.startswith("role_size_"):
            role, _slot, size_index = index
            if role < 2:
                return (15, 16) if size_index == 1 else (20, 21)
            return (30, 31) if size_index == 1 else (40, 41)
        if family.startswith("pair_"):
            return (2, 3) if index[0] == 0 else (5, 6)
        if family.startswith("role_"):
            return (35, 36) if index[0] < 2 else (71, 72)
        if family.startswith("endpoint_"):
            return (71, 72) if index[0] == 0 else (142, 143)
        raise AssertionError(f"Unknown count family: {family}")

    def _full_objective(self) -> int:
        total = 0
        for family in ("pair_seller", "pair_noise"):
            array = self.arrays[family]
            for treatment in range(2):
                lower, upper = self._bounds(family, (treatment, 0, 1))
                total += sum(
                    _penalty(int(array[treatment, left, right]), lower, upper)
                    for left in range(28)
                    for right in range(left + 1, 28)
                )
        for family in ("directed_seller", "directed_noise"):
            array = self.arrays[family]
            for treatment in range(2):
                lower, upper = self._bounds(family, (treatment, 0, 1))
                total += sum(
                    _penalty(int(array[treatment, left, right]), lower, upper)
                    for left in range(28)
                    for right in range(28)
                    if left != right
                )
        for family in (
            "role_seller",
            "role_noise",
            "endpoint_seller",
            "endpoint_noise",
        ):
            array = self.arrays[family]
            for first in range(array.shape[0]):
                lower, upper = self._bounds(family, (first, 0))
                total += sum(
                    _penalty(int(array[first, slot]), lower, upper)
                    for slot in range(28)
                )
        for role in range(4):
            lower, upper = self._bounds("role_triad", (role,))
            total += _penalty(int(self.role_triad[role]), lower, upper)
        for family in ("role_size_seller", "role_size_noise"):
            array = self.arrays[family]
            for role in range(4):
                for slot in range(28):
                    for size_index in range(2):
                        lower, upper = self._bounds(
                            family, (role, slot, size_index)
                        )
                        total += _penalty(
                            int(array[role, slot, size_index]), lower, upper
                        )
        return int(total)

    def _objective_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for family in self.arrays:
            subtotal = 0
            array = self.arrays[family]
            if family.startswith("pair_"):
                for treatment in range(2):
                    lower, upper = self._bounds(family, (treatment, 0, 1))
                    subtotal += sum(
                        _penalty(int(array[treatment, left, right]), lower, upper)
                        for left in range(28)
                        for right in range(left + 1, 28)
                    )
            elif family.startswith("directed_"):
                for treatment in range(2):
                    lower, upper = self._bounds(family, (treatment, 0, 1))
                    subtotal += sum(
                        _penalty(int(array[treatment, left, right]), lower, upper)
                        for left in range(28)
                        for right in range(28)
                        if left != right
                    )
            elif family == "role_triad":
                for role in range(4):
                    lower, upper = self._bounds(family, (role,))
                    subtotal += _penalty(int(array[role]), lower, upper)
            elif family.startswith("role_size_"):
                for role in range(4):
                    for slot in range(28):
                        for size_index in range(2):
                            lower, upper = self._bounds(
                                family, (role, slot, size_index)
                            )
                            subtotal += _penalty(
                                int(array[role, slot, size_index]), lower, upper
                            )
            else:
                for first in range(array.shape[0]):
                    lower, upper = self._bounds(family, (first, 0))
                    subtotal += sum(
                        _penalty(int(array[first, slot]), lower, upper)
                        for slot in range(28)
                    )
            breakdown[family] = int(subtotal)
        if sum(breakdown.values()) != self._full_objective():
            raise RegisteredNegativeConstructionError(
                "Objective breakdown disagrees with independent full objective"
            )
        return breakdown

    def _constraint_cells(
        self,
    ) -> list[tuple[str, tuple[int, ...], int, int]]:
        cells: list[tuple[str, tuple[int, ...], int, int]] = []
        for family in ("pair_seller", "pair_noise"):
            for treatment in range(2):
                for left in range(28):
                    for right in range(left + 1, 28):
                        index = (treatment, left, right)
                        cells.append((family, index, *self._bounds(family, index)))
        for family in ("directed_seller", "directed_noise"):
            for treatment in range(2):
                for left in range(28):
                    for right in range(28):
                        if left == right:
                            continue
                        index = (treatment, left, right)
                        cells.append((family, index, *self._bounds(family, index)))
        for family in (
            "role_seller",
            "role_noise",
            "endpoint_seller",
            "endpoint_noise",
        ):
            array = self.arrays[family]
            for first in range(array.shape[0]):
                for slot in range(28):
                    index = (first, slot)
                    cells.append((family, index, *self._bounds(family, index)))
        for role in range(4):
            index = (role,)
            cells.append(("role_triad", index, *self._bounds("role_triad", index)))
        for family in ("role_size_seller", "role_size_noise"):
            for role in range(4):
                for slot in range(28):
                    for size_index in range(2):
                        index = (role, slot, size_index)
                        cells.append((family, index, *self._bounds(family, index)))
        return cells

    def _option_contributions(
        self,
        world: int,
        pair_index: int,
        left_seller: int,
        right_seller: int,
    ) -> dict[tuple[str, tuple[int, ...]], int]:
        """Return every global count cell touched by one ordered endpoint pair."""

        left_position, right_position = PAIR_POSITIONS[pair_index]
        treatment = 0 if pair_index < 2 else 1
        left_noise = int(self.noise[world, left_seller])
        right_noise = int(self.noise[world, right_seller])
        seller_pair = _pair(left_seller, right_seller)
        noise_pair = _pair(left_noise, right_noise)
        contributions: defaultdict[
            tuple[str, tuple[int, ...]], int
        ] = defaultdict(int)

        def add(family: str, index: tuple[int, ...]) -> None:
            contributions[(family, index)] += 1

        add("pair_seller", (treatment, *seller_pair))
        add("pair_noise", (treatment, *noise_pair))
        add("directed_seller", (treatment, left_seller, right_seller))
        add("directed_noise", (treatment, left_noise, right_noise))
        for position, seller, noise in (
            (left_position, left_seller, left_noise),
            (right_position, right_seller, right_noise),
        ):
            role = int(ROLE_BY_POSITION[position])
            endpoint_treatment = int(TREATMENT_BY_POSITION[position])
            size_index = int(self.triads[world, seller])
            add("role_seller", (role, seller))
            add("role_noise", (role, noise))
            add("endpoint_seller", (endpoint_treatment, seller))
            add("endpoint_noise", (endpoint_treatment, noise))
            if size_index:
                add("role_triad", (role,))
            add("role_size_seller", (role, seller, size_index))
            add("role_size_noise", (role, noise, size_index))
        return dict(contributions)

    @staticmethod
    def _row_replacements_for_targets(
        world: int, row: np.ndarray, targets: Mapping[int, int]
    ) -> list[tuple[int, int, int]]:
        """Set target positions while preserving the row's endpoint uniqueness."""
        if len(set(targets.values())) != len(targets):
            return []
        new_row = row.copy()
        for position, target in sorted(targets.items()):
            target = int(target)
            if int(new_row[position]) == target:
                continue
            matches = np.flatnonzero(new_row == target)
            if len(matches) > 1:
                raise RegisteredNegativeConstructionError(
                    "Endpoint row was not unique before local repair"
                )
            if len(matches) == 1:
                other_position = int(matches[0])
                new_row[other_position] = new_row[position]
            new_row[position] = target
        if any(int(new_row[position]) != int(target) for position, target in targets.items()):
            return []
        return [
            (world, position, int(new_row[position]))
            for position in range(12)
            if int(new_row[position]) != int(row[position])
        ]

    def _coordinate_seller(self, family: str, world: int, slot: int) -> int:
        if family.endswith("_seller"):
            return slot
        matches = np.flatnonzero(self.noise[world] == slot)
        if len(matches) != 1:
            raise RegisteredNegativeConstructionError(
                "Noise schedule is not a per-world bijection"
            )
        return int(matches[0])

    def _targeted_repair_proposals(
        self,
        violated_cells: list[tuple[str, tuple[int, ...], int, int]],
    ):
        """Yield deterministic proposals aimed at every currently violated cell."""
        for family, index, lower, upper in violated_cells:
            current = int(self.arrays[family][index])
            under = current < lower
            source_key = (family, index)
            if family.startswith("pair_") or family.startswith("directed_"):
                treatment, left_slot, right_slot = index
                pair_indices = range(0, 2) if treatment == 0 else range(2, 6)
                directed = family.startswith("directed_")
                if under:
                    orientations = (
                        ((left_slot, right_slot),)
                        if directed
                        else ((left_slot, right_slot), (right_slot, left_slot))
                    )
                    for world in range(balanced.WORLD_COUNT):
                        for left_value, right_value in orientations:
                            left_seller = self._coordinate_seller(
                                family, world, left_value
                            )
                            right_seller = self._coordinate_seller(
                                family, world, right_value
                            )
                            for pair_index in pair_indices:
                                left_position, right_position = PAIR_POSITIONS[pair_index]
                                replacements = self._row_replacements_for_targets(
                                    world,
                                    self.assignments[world],
                                    {
                                        left_position: left_seller,
                                        right_position: right_seller,
                                    },
                                )
                                if replacements:
                                    yield source_key, "targeted_pair_insertion", replacements
                else:
                    for world in range(balanced.WORLD_COUNT):
                        row = self.assignments[world]
                        for pair_index in pair_indices:
                            left_position, right_position = PAIR_POSITIONS[pair_index]
                            left_seller = int(row[left_position])
                            right_seller = int(row[right_position])
                            if family.endswith("_noise"):
                                observed = (
                                    int(self.noise[world, left_seller]),
                                    int(self.noise[world, right_seller]),
                                )
                            else:
                                observed = (left_seller, right_seller)
                            matches = (
                                observed == (left_slot, right_slot)
                                if directed
                                else _pair(*observed) == (left_slot, right_slot)
                            )
                            if not matches:
                                continue
                            for position in (left_position, right_position):
                                for candidate_seller in range(28):
                                    replacements = self._row_replacements_for_targets(
                                        world,
                                        row,
                                        {position: candidate_seller},
                                    )
                                    if replacements:
                                        yield source_key, "targeted_pair_removal", replacements
                continue

            if family == "role_triad":
                role = index[0]
                desired_size = 1 if under else 0
                for world in range(balanced.WORLD_COUNT):
                    for position in np.flatnonzero(ROLE_BY_POSITION == role):
                        for candidate_seller in np.flatnonzero(
                            self.triads[world] == desired_size
                        ):
                            replacements = self._row_replacements_for_targets(
                                world,
                                self.assignments[world],
                                {int(position): int(candidate_seller)},
                            )
                            if replacements:
                                yield source_key, "targeted_role_triad", replacements
                continue

            if family.startswith("role_size_"):
                role, slot, size_index = index
                positions = np.flatnonzero(ROLE_BY_POSITION == role)
                for world in range(balanced.WORLD_COUNT):
                    row = self.assignments[world]
                    target_seller = self._coordinate_seller(family, world, slot)
                    if under:
                        if int(self.triads[world, target_seller]) != size_index:
                            continue
                        candidate_sellers = (target_seller,)
                    else:
                        candidate_sellers = range(28)
                    for position in positions:
                        observed_seller = int(row[position])
                        observed_slot = (
                            observed_seller
                            if family.endswith("_seller")
                            else int(self.noise[world, observed_seller])
                        )
                        if not under and not (
                            observed_slot == slot
                            and int(self.triads[world, observed_seller]) == size_index
                        ):
                            continue
                        for candidate_seller in candidate_sellers:
                            replacements = self._row_replacements_for_targets(
                                world,
                                row,
                                {int(position): int(candidate_seller)},
                            )
                            if replacements:
                                yield source_key, "targeted_role_size", replacements
                continue

            if family.startswith("role_"):
                role, slot = index
                positions = np.flatnonzero(ROLE_BY_POSITION == role)
            elif family.startswith("endpoint_"):
                treatment, slot = index
                positions = np.flatnonzero(TREATMENT_BY_POSITION == treatment)
            else:
                raise RegisteredNegativeConstructionError(
                    f"Unhandled local-repair family: {family}"
                )
            for world in range(balanced.WORLD_COUNT):
                row = self.assignments[world]
                target_seller = self._coordinate_seller(family, world, slot)
                for position in positions:
                    observed_seller = int(row[position])
                    observed_slot = (
                        observed_seller
                        if family.endswith("_seller")
                        else int(self.noise[world, observed_seller])
                    )
                    if not under and observed_slot != slot:
                        continue
                    candidate_sellers = (target_seller,) if under else range(28)
                    for candidate_seller in candidate_sellers:
                        replacements = self._row_replacements_for_targets(
                            world,
                            row,
                            {int(position): int(candidate_seller)},
                        )
                        if replacements:
                            yield source_key, "targeted_marginal", replacements

    @staticmethod
    def _cell_violation(value: int, lower: int, upper: int) -> int:
        return max(lower - value, 0) + max(value - upper, 0)

    def _l1_bound_violation(
        self, cells: list[tuple[str, tuple[int, ...], int, int]]
    ) -> int:
        return sum(
            self._cell_violation(
                int(self.arrays[family][index]), lower, upper
            )
            for family, index, lower, upper in cells
        )

    def _l1_delta_for_changes(
        self,
        changes: Mapping[tuple[str, tuple[int, ...]], int],
        cell_bounds: Mapping[tuple[str, tuple[int, ...]], tuple[int, int]],
    ) -> int:
        delta = 0
        for key, count_delta in changes.items():
            lower, upper = cell_bounds[key]
            family, index = key
            current = int(self.arrays[family][index])
            delta += self._cell_violation(
                current + int(count_delta), lower, upper
            ) - self._cell_violation(current, lower, upper)
        return int(delta)

    def _objective_delta_for_changes(
        self,
        changes: Mapping[tuple[str, tuple[int, ...]], int],
    ) -> int:
        delta = 0
        for (family, index), count_delta in changes.items():
            current = int(self.arrays[family][index])
            lower, upper = self._bounds(family, index)
            delta += _penalty(
                current + int(count_delta), lower, upper
            ) - _penalty(current, lower, upper)
        return int(delta)

    def _rebuild_all_counts(self) -> None:
        for array in self.arrays.values():
            array.fill(0)
        for world, row in enumerate(self.assignments):
            self._accumulate_world(world, row, 1)
        self.objective = self._full_objective()

    @staticmethod
    def _candidate_equivalence_key(
        candidate: RepairCandidate,
        cell_index: Mapping[tuple[str, tuple[int, ...]], int],
    ) -> tuple[Any, ...]:
        changes, new_rows, _objective_delta, _kind = candidate
        return (
            tuple(sorted(new_rows)),
            tuple(
                sorted(
                    (cell_index[key], int(value))
                    for key, value in changes.items()
                    if value
                )
            ),
        )

    @staticmethod
    def _candidate_row_key(candidate: RepairCandidate) -> tuple[Any, ...]:
        _changes, new_rows, objective_delta, kind = candidate
        return (
            int(objective_delta),
            len(new_rows),
            kind,
            tuple(
                (world, new_rows[world].tobytes())
                for world in sorted(new_rows)
            ),
        )

    def _collect_targeted_candidate_columns(
        self,
        *,
        target_cells: list[tuple[str, tuple[int, ...], int, int]],
        cells: list[tuple[str, tuple[int, ...], int, int]],
        cell_index: Mapping[tuple[str, tuple[int, ...]], int],
        excluded_worlds: frozenset[int] = frozenset(),
    ) -> tuple[list[RepairCandidate], dict[str, int]]:
        """Collect all exact-delta columns for the explicit target cells.

        Equivalent columns with the same touched worlds and the same full
        5,324-cell delta are compressed canonically. Nothing else is truncated.
        """

        cell_bounds = {
            (family, index): (lower, upper)
            for family, index, lower, upper in cells
        }
        selected: list[RepairCandidate] = []
        enumerated_count = 0
        valid_count = 0
        equivalent_count = 0
        for target_cell in target_cells:
            family, index, lower, upper = target_cell
            source_key = (family, index)
            source_current = int(self.arrays[family][index])
            retained: dict[tuple[Any, ...], RepairCandidate] = {}
            for yielded_source, kind, replacements in self._targeted_repair_proposals(
                [target_cell]
            ):
                enumerated_count += 1
                if yielded_source != source_key:
                    raise RegisteredNegativeConstructionError(
                        "Targeted candidate source-key drift"
                    )
                proposal = self._change_delta(replacements)
                if proposal is None:
                    continue
                objective_delta, changes, new_rows = proposal
                if excluded_worlds.intersection(new_rows):
                    continue
                source_delta = int(changes.get(source_key, 0))
                source_after = source_current + source_delta
                if self._cell_violation(
                    source_after, lower, upper
                ) >= self._cell_violation(source_current, lower, upper):
                    continue
                candidate: RepairCandidate = (
                    dict(changes),
                    {world: row.copy() for world, row in new_rows.items()},
                    int(objective_delta),
                    kind,
                )
                valid_count += 1
                equivalence = self._candidate_equivalence_key(
                    candidate, cell_index
                )
                existing = retained.get(equivalence)
                if existing is None or self._candidate_row_key(
                    candidate
                ) < self._candidate_row_key(existing):
                    if existing is not None:
                        equivalent_count += 1
                    retained[equivalence] = candidate
                else:
                    equivalent_count += 1

            def rank(candidate: RepairCandidate) -> tuple[Any, ...]:
                changes, _new_rows, objective_delta, _kind = candidate
                source_after = source_current + int(changes.get(source_key, 0))
                return (
                    self._cell_violation(source_after, lower, upper),
                    self._l1_delta_for_changes(changes, cell_bounds),
                    int(objective_delta),
                    self._candidate_row_key(candidate),
                )

            ranked = sorted(retained.values(), key=rank)
            selected.extend(ranked)

        globally_retained: dict[tuple[Any, ...], RepairCandidate] = {}
        for candidate in selected:
            equivalence = self._candidate_equivalence_key(candidate, cell_index)
            existing = globally_retained.get(equivalence)
            if existing is None or self._candidate_row_key(
                candidate
            ) < self._candidate_row_key(existing):
                if existing is not None:
                    equivalent_count += 1
                globally_retained[equivalence] = candidate
            else:
                equivalent_count += 1
        columns = sorted(
            globally_retained.values(), key=self._candidate_row_key
        )
        return columns, {
            "target_cell_count": len(target_cells),
            "enumerated_proposal_count": enumerated_count,
            "valid_proposal_count": valid_count,
            "equivalent_proposal_count": equivalent_count,
            "omitted_by_search_budget_count": 0,
            "retained_column_count": len(columns),
        }
    def _exact_local_repair(
        self, *, stop_at_residual_checkpoint: bool = False
    ) -> dict[str, Any]:
        """Iteratively reduce slack with audited greedy and sparse MILP moves."""

        cells = self._constraint_cells()
        cell_index = {
            (family, index): ordinal
            for ordinal, (family, index, _lower, _upper) in enumerate(cells)
        }

        def l1_bound_violation() -> int:
            return sum(
                max(lower - int(self.arrays[family][index]), 0)
                + max(int(self.arrays[family][index]) - upper, 0)
                for family, index, lower, upper in cells
            )

        round_receipts: list[dict[str, Any]] = []
        for round_ordinal in range(1, LOCAL_REPAIR_MAX_ROUNDS + 1):
            violated_cells = [
                cell
                for cell in cells
                if not cell[2] <= int(self.arrays[cell[0]][cell[1]]) <= cell[3]
            ]
            if not violated_cells:
                return {
                    "attempted": bool(round_receipts),
                    "round_count": len(round_receipts),
                    "rounds": round_receipts,
                    "final_l1_bound_violation": 0,
                    "final_objective": 0,
                }
            if l1_bound_violation() <= LOCAL_REPAIR_COARSE_MILP_THRESHOLD:
                if stop_at_residual_checkpoint:
                    return {
                        "attempted": bool(round_receipts),
                        "checkpoint_only": True,
                        "round_count": len(round_receipts),
                        "rounds": round_receipts,
                        "final_l1_bound_violation": l1_bound_violation(),
                        "final_objective": self.objective,
                    }
                raise RegisteredNegativeConstructionError(
                    "Terminal V17 residual pricing is disabled; build and use "
                    "the version-neutral residual checkpoint"
                )

            candidates: list[
                tuple[
                    dict[tuple[str, tuple[int, ...]], int],
                    dict[int, np.ndarray],
                    int,
                    str,
                ]
            ] = []
            seen: set[tuple[tuple[int, bytes], ...]] = set()

            def is_helpful(
                changes: Mapping[tuple[str, tuple[int, ...]], int]
            ) -> bool:
                for family, index, lower, upper in violated_cells:
                    delta = changes.get((family, index), 0)
                    current = int(self.arrays[family][index])
                    if (current < lower and delta > 0) or (
                        current > upper and delta < 0
                    ):
                        return True
                return False

            def add_candidate(
                replacements: list[tuple[int, int, int]],
                kind: str,
                maximum_positive_delta: int,
            ) -> bool:
                proposal = self._change_delta(replacements)
                if proposal is None:
                    return False
                objective_delta, changes, new_rows = proposal
                if objective_delta > maximum_positive_delta:
                    return False
                if not is_helpful(changes):
                    return False
                key = tuple(
                    (world, new_rows[world].tobytes()) for world in sorted(new_rows)
                )
                if key in seen:
                    return False
                seen.add(key)
                candidates.append(
                    (
                        dict(changes),
                        {world: row.copy() for world, row in new_rows.items()},
                        objective_delta,
                        kind,
                    )
                )
                return True

            targeted_counts: defaultdict[
                tuple[str, tuple[int, ...]], int
            ] = defaultdict(int)
            for source_key, kind, replacements in self._targeted_repair_proposals(
                violated_cells
            ):
                if (
                    targeted_counts[source_key]
                    >= LOCAL_REPAIR_MAX_TARGETED_CANDIDATES_PER_CELL
                ):
                    continue
                if add_candidate(
                    replacements, kind, LOCAL_REPAIR_TARGETED_MAX_POSITIVE_DELTA
                ):
                    targeted_counts[source_key] += 1

            def candidate_canonical_key(candidate_index: int) -> tuple[Any, ...]:
                _changes, new_rows, objective_delta, kind = candidates[
                    candidate_index
                ]
                return (
                    objective_delta,
                    len(new_rows),
                    kind,
                    tuple(
                        (world, new_rows[world].tobytes())
                        for world in sorted(new_rows)
                    ),
                )

            def candidate_l1_help(candidate_index: int) -> int:
                changes = candidates[candidate_index][0]
                improvement = 0
                for family, index, lower, upper in violated_cells:
                    current = int(self.arrays[family][index])
                    changed = current + changes.get((family, index), 0)
                    before = max(lower - current, 0) + max(current - upper, 0)
                    after = max(lower - changed, 0) + max(changed - upper, 0)
                    improvement += before - after
                return int(improvement)

            def apply_best_strict_improver(
                *, valid_random_proposals: int, compensating_candidates: int
            ) -> bool:
                improving = [
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate[2] < 0
                ]
                if not improving:
                    return False
                selected_index = min(improving, key=candidate_canonical_key)
                _changes, new_rows, declared_delta, kind = candidates[selected_index]
                before_objective = self.objective
                before_l1 = l1_bound_violation()
                for world, row in new_rows.items():
                    self.assignments[world] = row
                for array in self.arrays.values():
                    array.fill(0)
                for world, row in enumerate(self.assignments):
                    self._accumulate_world(world, row, 1)
                self.objective = self._full_objective()
                after_l1 = l1_bound_violation()
                if self.objective != before_objective + declared_delta:
                    raise RegisteredNegativeConstructionError(
                        "Greedy local repair delta disagrees with independent rebuild"
                    )
                if self.objective >= before_objective:
                    raise RegisteredNegativeConstructionError(
                        "Greedy local repair did not strictly reduce the full objective"
                    )
                round_receipt = {
                    "round_ordinal": round_ordinal,
                    "repair_mode": "deterministic_steepest_single_move",
                    "before_objective": before_objective,
                    "after_objective": self.objective,
                    "before_l1_bound_violation": before_l1,
                    "after_l1_bound_violation": after_l1,
                    "violated_cell_count": len(violated_cells),
                    "candidate_count": len(candidates),
                    "targeted_candidate_count": sum(targeted_counts.values()),
                    "compensating_candidate_count": compensating_candidates,
                    "valid_random_proposal_count": valid_random_proposals,
                    "selected_candidate_count": 1,
                    "selected_world_count": len(new_rows),
                    "selected_moves_by_proposal_kind": {kind: 1},
                    "declared_objective_delta": declared_delta,
                }
                round_receipts.append(round_receipt)
                print(
                    json.dumps(
                        {"event": "exact_local_greedy_round_complete", **round_receipt},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return True

            if apply_best_strict_improver(
                valid_random_proposals=0, compensating_candidates=0
            ):
                continue

            valid_random_proposals = 0
            compensating_candidates = 0
            for _proposal_ordinal in range(
                LOCAL_REPAIR_MAX_PROPOSALS_PER_ROUND
            ):
                proposal = self._change_delta(self._proposal())
                if proposal is None:
                    continue
                objective_delta, changes, new_rows = proposal
                valid_random_proposals += 1
                if objective_delta > LOCAL_REPAIR_MAX_POSITIVE_DELTA:
                    continue
                helpful = is_helpful(changes)
                if not helpful and (
                    compensating_candidates
                    >= LOCAL_REPAIR_MAX_COMPENSATING_CANDIDATES_PER_ROUND
                ):
                    continue
                key = tuple(
                    (world, new_rows[world].tobytes()) for world in sorted(new_rows)
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    (
                        dict(changes),
                        {world: row.copy() for world, row in new_rows.items()},
                        objective_delta,
                        (
                            self.last_proposal_kind
                            if helpful
                            else "random_compensating_" + self.last_proposal_kind
                        ),
                    )
                )
                if not helpful:
                    compensating_candidates += 1
                if len(candidates) >= LOCAL_REPAIR_MAX_CANDIDATES_PER_ROUND:
                    break
            if not candidates:
                raise RegisteredNegativeConstructionError(
                    "Exact local repair produced no eligible candidate moves"
                )

            if apply_best_strict_improver(
                valid_random_proposals=valid_random_proposals,
                compensating_candidates=compensating_candidates,
            ):
                continue

            before_objective = self.objective
            before_l1 = l1_bound_violation()
            helpful_indices = sorted(
                (
                    index
                    for index in range(len(candidates))
                    if candidate_l1_help(index) > 0
                ),
                key=lambda index: (
                    -candidate_l1_help(index),
                    *candidate_canonical_key(index),
                ),
            )[:LOCAL_REPAIR_COARSE_MILP_MAX_HELPFUL_CANDIDATES]
            coarse_compensating_indices = sorted(
                (
                    index
                    for index in range(len(candidates))
                    if candidate_l1_help(index) <= 0
                ),
                key=candidate_canonical_key,
            )[:LOCAL_REPAIR_COARSE_MILP_MAX_COMPENSATING_CANDIDATES]
            candidates = [
                candidates[index]
                for index in helpful_indices + coarse_compensating_indices
            ]
            repair_mode = "coarse_compressed_slack_milp"
            start_event = "exact_local_coarse_milp_round_start"
            time_limit_seconds = LOCAL_REPAIR_COARSE_MILP_TIME_LIMIT_SECONDS
            candidate_count = len(candidates)
            candidate_cell_changes = [
                {
                    cell_index[key]: int(value)
                    for key, value in changes.items()
                    if key in cell_index
                }
                for changes, _new_rows, _delta, _kind in candidates
            ]
            candidate_worlds = [
                tuple(sorted(new_rows))
                for _changes, new_rows, _delta, _kind in candidates
            ]
            print(
                json.dumps(
                    {
                        "event": start_event,
                        "round_ordinal": round_ordinal,
                        "before_objective": before_objective,
                        "before_l1_bound_violation": before_l1,
                        "violated_cell_count": len(violated_cells),
                        "candidate_count": candidate_count,
                        "targeted_candidate_count": sum(targeted_counts.values()),
                        "compensating_candidate_count": compensating_candidates,
                        "valid_random_proposal_count": valid_random_proposals,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            current_bounds = [
                (int(self.arrays[family][index]), lower, upper)
                for family, index, lower, upper in cells
            ]
            result = _solve_sparse_candidate_milp(
                current_bounds=current_bounds,
                candidate_cell_changes=candidate_cell_changes,
                candidate_worlds=candidate_worlds,
                candidate_costs=[
                    1.0 + max(0, int(objective_delta)) * 0.01
                    for _changes, _new_rows, objective_delta, _kind in candidates
                ],
                time_limit_seconds=time_limit_seconds,
                slack_unit_cost_override=LOCAL_REPAIR_COARSE_MILP_SLACK_COST,
            )
            if not result["feasible"]:
                raise RegisteredNegativeConstructionError(
                    "Exact local solver found no candidate set: "
                    f"mode={repair_mode} round={round_ordinal} "
                    f"status={result['solver_status_name']}"
                )
            selected = result["selected_candidate_indices"]
            if not selected:
                raise RegisteredNegativeConstructionError(
                    "Exact local solver returned an empty improvement set: "
                    f"mode={repair_mode}"
                )
            touched_worlds: set[int] = set()
            selected_kinds: defaultdict[str, int] = defaultdict(int)
            for index in selected:
                _changes, new_rows, _delta, kind = candidates[index]
                if touched_worlds & set(new_rows):
                    raise RegisteredNegativeConstructionError(
                        "Exact local repair selected overlapping worlds"
                    )
                touched_worlds.update(new_rows)
                selected_kinds[kind] += 1
                for world, row in new_rows.items():
                    self.assignments[world] = row
            for array in self.arrays.values():
                array.fill(0)
            for world, row in enumerate(self.assignments):
                self._accumulate_world(world, row, 1)
            self.objective = self._full_objective()
            after_l1 = l1_bound_violation()
            if self.objective >= before_objective and after_l1 >= before_l1:
                raise RegisteredNegativeConstructionError(
                    "Exact local solver did not strictly improve either audited "
                    f"measure: round={round_ordinal} objective={before_objective}->"
                    f"{self.objective} l1={before_l1}->{after_l1}"
                )
            round_receipt = {
                "round_ordinal": round_ordinal,
                "repair_mode": repair_mode,
                "before_objective": before_objective,
                "after_objective": self.objective,
                "before_l1_bound_violation": before_l1,
                "after_l1_bound_violation": after_l1,
                "violated_cell_count": len(violated_cells),
                "candidate_count": candidate_count,
                "targeted_candidate_count": sum(targeted_counts.values()),
                "compensating_candidate_count": compensating_candidates,
                "valid_random_proposal_count": valid_random_proposals,
                "selected_candidate_count": len(selected),
                "selected_world_count": len(touched_worlds),
                "selected_moves_by_proposal_kind": dict(
                    sorted(selected_kinds.items())
                ),
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"feasible", "selected_candidate_indices"}
                },
                "time_limit_seconds": time_limit_seconds,
            }
            round_receipts.append(round_receipt)
            print(
                json.dumps(
                    {"event": "exact_local_solver_round_complete", **round_receipt},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        raise RegisteredNegativeConstructionError(
            "Exact local repair exhausted its frozen round count without zero "
            f"violation: objective={self.objective} l1={l1_bound_violation()} "
            f"breakdown={self._objective_breakdown()}"
        )

    def _valid_row(self, world: int, row: np.ndarray) -> bool:
        if len(set(map(int, row))) != 12:
            return False
        if not all(
            self.controllers[world, int(row[left])]
            != self.controllers[world, int(row[right])]
            for left, right in PAIR_POSITIONS
        ):
            return False
        for pair_index, (left_position, right_position) in enumerate(PAIR_POSITIONS):
            left_noise = int(self.noise[world, int(row[left_position])])
            right_noise = int(self.noise[world, int(row[right_position])])
            if not self.role_eligible[
                world, left_position, int(row[left_position])
            ] or not self.role_eligible[
                world, right_position, int(row[right_position])
            ]:
                return False
            if pair_index < 2 and not (
                self.clone_description_capable[left_noise]
                or self.clone_description_capable[right_noise]
            ):
                return False
        return True

    def _change_delta(
        self, replacements: list[tuple[int, int, int]]
    ) -> tuple[int, dict[tuple[str, tuple[int, ...]], int], dict[int, np.ndarray]] | None:
        new_rows: dict[int, np.ndarray] = {}
        for world, position, new_seller in replacements:
            row = new_rows.setdefault(world, self.assignments[world].copy())
            row[position] = new_seller
        if any(not self._valid_row(world, row) for world, row in new_rows.items()):
            return None

        changes: defaultdict[tuple[str, tuple[int, ...]], int] = defaultdict(int)

        def add(family: str, index: tuple[int, ...], delta: int) -> None:
            changes[(family, index)] += delta

        for world, new_row in new_rows.items():
            old_row = self.assignments[world]
            for pair_index, (left_position, right_position) in enumerate(PAIR_POSITIONS):
                treatment = 0 if pair_index < 2 else 1
                old_seller_pair = _pair(
                    int(old_row[left_position]), int(old_row[right_position])
                )
                new_seller_pair = _pair(
                    int(new_row[left_position]), int(new_row[right_position])
                )
                old_noise_pair = _pair(
                    int(self.noise[world, old_row[left_position]]),
                    int(self.noise[world, old_row[right_position]]),
                )
                new_noise_pair = _pair(
                    int(self.noise[world, new_row[left_position]]),
                    int(self.noise[world, new_row[right_position]]),
                )
                if old_seller_pair != new_seller_pair:
                    add("pair_seller", (treatment, *old_seller_pair), -1)
                    add("pair_seller", (treatment, *new_seller_pair), 1)
                if old_noise_pair != new_noise_pair:
                    add("pair_noise", (treatment, *old_noise_pair), -1)
                    add("pair_noise", (treatment, *new_noise_pair), 1)
                old_left_seller = int(old_row[left_position])
                old_right_seller = int(old_row[right_position])
                new_left_seller = int(new_row[left_position])
                new_right_seller = int(new_row[right_position])
                if (old_left_seller, old_right_seller) != (
                    new_left_seller,
                    new_right_seller,
                ):
                    add(
                        "directed_seller",
                        (treatment, old_left_seller, old_right_seller),
                        -1,
                    )
                    add(
                        "directed_seller",
                        (treatment, new_left_seller, new_right_seller),
                        1,
                    )
                old_left_noise = int(self.noise[world, old_left_seller])
                old_right_noise = int(self.noise[world, old_right_seller])
                new_left_noise = int(self.noise[world, new_left_seller])
                new_right_noise = int(self.noise[world, new_right_seller])
                if (old_left_noise, old_right_noise) != (
                    new_left_noise,
                    new_right_noise,
                ):
                    add(
                        "directed_noise",
                        (treatment, old_left_noise, old_right_noise),
                        -1,
                    )
                    add(
                        "directed_noise",
                        (treatment, new_left_noise, new_right_noise),
                        1,
                    )
            for position in range(12):
                old_seller = int(old_row[position])
                new_seller = int(new_row[position])
                if old_seller == new_seller:
                    continue
                old_noise = int(self.noise[world, old_seller])
                new_noise = int(self.noise[world, new_seller])
                role = int(ROLE_BY_POSITION[position])
                treatment = int(TREATMENT_BY_POSITION[position])
                add("role_seller", (role, old_seller), -1)
                add("role_seller", (role, new_seller), 1)
                add("role_noise", (role, old_noise), -1)
                add("role_noise", (role, new_noise), 1)
                add("endpoint_seller", (treatment, old_seller), -1)
                add("endpoint_seller", (treatment, new_seller), 1)
                add("endpoint_noise", (treatment, old_noise), -1)
                add("endpoint_noise", (treatment, new_noise), 1)
                add(
                    "role_triad",
                    (role,),
                    int(self.triads[world, new_seller])
                    - int(self.triads[world, old_seller]),
                )
                old_size = int(self.triads[world, old_seller])
                new_size = int(self.triads[world, new_seller])
                add("role_size_seller", (role, old_seller, old_size), -1)
                add("role_size_seller", (role, new_seller, new_size), 1)
                add("role_size_noise", (role, old_noise, old_size), -1)
                add("role_size_noise", (role, new_noise, new_size), 1)

        objective_delta = 0
        cleaned = {key: value for key, value in changes.items() if value}
        for (family, index), count_delta in cleaned.items():
            array = self.arrays[family]
            current = int(array[index])
            lower, upper = self._bounds(family, index)
            objective_delta += _penalty(current + count_delta, lower, upper) - _penalty(
                current, lower, upper
            )
        return int(objective_delta), cleaned, new_rows

    def _proposal(self) -> list[tuple[int, int, int]]:
        draw = float(self.rng.random())
        world = int(self.rng.integers(balanced.WORLD_COUNT))
        position = int(self.rng.integers(12))
        if draw < 0.30:
            self.last_proposal_kind = "single_endpoint_replacement"
            used = set(map(int, self.assignments[world]))
            candidates = [slot for slot in range(28) if slot not in used]
            return [(world, position, int(self.rng.choice(candidates)))]
        if draw < 0.50:
            self.last_proposal_kind = "within_world_two_position_swap"
            other_position = int(self.rng.integers(11))
            other_position += int(other_position >= position)
            return [
                (world, position, int(self.assignments[world, other_position])),
                (world, other_position, int(self.assignments[world, position])),
            ]
        if draw < 0.76:
            self.last_proposal_kind = (
                "cross_world_same_position_swap"
                if draw < 0.68
                else "cross_world_random_position_swap"
            )
            other_world = int(self.rng.integers(balanced.WORLD_COUNT - 1))
            other_world += int(other_world >= world)
            other_position = (
                position if draw < 0.68 else int(self.rng.integers(12))
            )
            return [
                (world, position, int(self.assignments[other_world, other_position])),
                (other_world, other_position, int(self.assignments[world, position])),
            ]
        if draw < 0.92:
            self.last_proposal_kind = (
                "cross_world_same_position_three_cycle"
                if draw < 0.86
                else "cross_world_random_position_three_cycle"
            )
            worlds = [int(value) for value in self.rng.choice(500, 3, replace=False)]
            positions = (
                [position, position, position]
                if draw < 0.86
                else [int(value) for value in self.rng.integers(12, size=3)]
            )
            values = [
                int(self.assignments[target_world, target_position])
                for target_world, target_position in zip(worlds, positions)
            ]
            return [
                (worlds[index], positions[index], values[(index - 1) % 3])
                for index in range(3)
            ]
        if draw < 0.97:
            self.last_proposal_kind = "cross_world_complete_pair_swap"
            other_world = int(self.rng.integers(balanced.WORLD_COUNT - 1))
            other_world += int(other_world >= world)
            pair_index = int(self.rng.integers(len(PAIR_POSITIONS)))
            positions = PAIR_POSITIONS[pair_index]
            return [
                (
                    world,
                    pair_position,
                    int(self.assignments[other_world, pair_position]),
                )
                for pair_position in positions
            ] + [
                (
                    other_world,
                    pair_position,
                    int(self.assignments[world, pair_position]),
                )
                for pair_position in positions
            ]
        self.last_proposal_kind = "within_world_three_position_cycle"
        positions = [
            int(value) for value in self.rng.choice(12, 3, replace=False)
        ]
        values = [int(self.assignments[world, value]) for value in positions]
        return [
            (world, positions[index], values[(index - 1) % 3])
            for index in range(3)
        ]

    def run(self, *, stop_at_residual_checkpoint: bool = False) -> dict[str, Any]:
        initial_objective = self.objective
        accepted_moves = 0
        accepted_by_proposal_kind: defaultdict[str, int] = defaultdict(int)
        solved_iteration: int | None = None
        for iteration in range(1, MAXIMUM_ITERATIONS + 1):
            proposal = self._change_delta(self._proposal())
            if proposal is None:
                continue
            objective_delta, changes, new_rows = proposal
            if iteration <= ANNEALING_ITERATIONS:
                temperature = max(
                    0.01, 8.0 * (1.0 - iteration / ANNEALING_ITERATIONS)
                )
            else:
                temperature = 0.01
            if objective_delta <= 0 or self.rng.random() < math.exp(
                -objective_delta / temperature
            ):
                for (family, index), count_delta in changes.items():
                    self.arrays[family][index] += count_delta
                for world, row in new_rows.items():
                    self.assignments[world] = row
                self.objective += objective_delta
                accepted_moves += 1
                accepted_by_proposal_kind[self.last_proposal_kind] += 1
                if self.objective == 0:
                    solved_iteration = iteration
                    break
            if iteration % 1_000_000 == 0:
                print(
                    json.dumps(
                        {
                            "accepted_moves": accepted_moves,
                            "iteration": iteration,
                            "objective": self.objective,
                            "objective_breakdown": self._objective_breakdown(),
                            "temperature": temperature,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
        exact_local_repair: dict[str, Any] = {
            "attempted": False,
            "reason": "annealing_reached_objective_zero",
        }
        solution_stage = "annealing_or_cold_polishing"
        if solved_iteration is None:
            print(
                json.dumps(
                        {
                            "event": "starting_coarse_then_residual_checkpoint",
                        "pre_repair_objective": self.objective,
                        "objective_breakdown": self._objective_breakdown(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            exact_local_repair = self._exact_local_repair(
                stop_at_residual_checkpoint=stop_at_residual_checkpoint
            )
            solution_stage = (
                "coarse_candidate_repair_then_residual_checkpoint"
                if stop_at_residual_checkpoint
                else "residual_successor_unassigned_stop_guard"
            )
        full_objective = self._full_objective()
        if full_objective != self.objective:
            raise RegisteredNegativeConstructionError(
                "Incremental objective and independent full objective disagree"
            )
        if stop_at_residual_checkpoint:
            if not 0 < full_objective <= LOCAL_REPAIR_COARSE_MILP_THRESHOLD:
                raise RegisteredNegativeConstructionError(
                    "Residual checkpoint did not stop inside its frozen boundary"
                )
        elif full_objective != 0:
            raise RegisteredNegativeConstructionError(
                "Terminal V17 exact-zero control flow must not be reused"
            )
        return {
            "initial_objective": initial_objective,
            "solved_iteration": solved_iteration,
            "solution_stage": solution_stage,
            "accepted_moves": accepted_moves,
            "accepted_moves_by_proposal_kind": dict(
                sorted(accepted_by_proposal_kind.items())
            ),
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "annealing_iterations": ANNEALING_ITERATIONS,
            "cold_polishing_iterations": COLD_POLISHING_ITERATIONS,
            "exact_local_repair": exact_local_repair,
            "final_objective_breakdown": self._objective_breakdown(),
        }


def materialize_plan(
    *,
    split: str,
    assignments: np.ndarray,
    schedule: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
    plan_version: str = plan_contract.VERSION,
) -> dict[str, Any]:
    noise_slot_signatures = {
        int(row["noise_slot"]): row["signature"]
        for row in joint_signatures["noise_slot_multiset"]
    }
    occurrence_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    worlds = []
    for world_ordinal, seller_row in enumerate(assignments):
        noise_by_seller = schedule["worlds"][world_ordinal][
            "noise_slot_by_seller_slot"
        ]
        world_assignments = []
        for pair_index, (left_position, right_position) in enumerate(PAIR_POSITIONS):
            if pair_index < 2:
                treatment = "exact_title_clone"
                instance_ordinal = pair_index
                roles = ("source", "target")
            else:
                treatment = "high_semantic_similarity"
                instance_ordinal = pair_index - 2
                roles = ("left", "right")
            endpoints = []
            for position, role in zip((left_position, right_position), roles):
                seller_slot = int(seller_row[position])
                noise_slot = int(noise_by_seller[seller_slot])
                signature = noise_slot_signatures[noise_slot]
                signature_text = _canonical_signature_text(signature)
                stratum = (treatment, role, signature_text)
                occurrence_index = occurrence_counts[stratum]
                occurrence_counts[stratum] += 1
                eligible = _role_eligible_logical_item_ordinals(
                    treatment=treatment,
                    role=role,
                    signature=signature,
                )
                start = _logical_item_cycle_start(
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
            world_assignments.append(
                {
                    "treatment": treatment,
                    "instance_ordinal": instance_ordinal,
                    "endpoints": endpoints,
                    "semantic_asset": (
                        None
                        if treatment == "exact_title_clone"
                        else _semantic_asset(
                            split=split,
                            world_ordinal=world_ordinal,
                            instance_ordinal=instance_ordinal,
                        )
                    ),
                }
            )
        worlds.append(
            {"world_ordinal": world_ordinal, "assignments": world_assignments}
        )
    plan: dict[str, Any] = {
        "version": plan_version,
        "split": split,
        "world_count": balanced.WORLD_COUNT,
        "balanced_schedule_sha256": schedule["canonical_self_sha256"],
        "joint_noise_signature_sha256": joint_signatures[
            "canonical_self_sha256"
        ],
        "semantic_domain_contract": {
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
        },
        "worlds": worlds,
    }
    plan["canonical_self_sha256"] = plan_contract.canonical_self_sha256(plan)
    return plan


def _write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(plan_contract.canonical_json_bytes(value))
        stream.write(b"\n")


def validate_formal_invocation(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    development_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    """Fail closed until a successor to terminal V17 is frozen."""

    del (
        output_directory,
        train_schedule_path,
        development_schedule_path,
        joint_signature_path,
    )
    raise RegisteredNegativeConstructionError(
        "The V9.3 registered-negative successor is unassigned; terminal V17 "
        "must not be rerun"
    )


def publish(
    *,
    output_directory: Path,
    train_schedule_path: Path,
    development_schedule_path: Path,
    joint_signature_path: Path,
) -> dict[str, Any]:
    validate_formal_invocation(
        output_directory=output_directory,
        train_schedule_path=train_schedule_path,
        development_schedule_path=development_schedule_path,
        joint_signature_path=joint_signature_path,
    )
    if output_directory.exists():
        raise RegisteredNegativeConstructionError("Output directory already exists")
    building = output_directory.with_name(output_directory.name + ".building")
    if building.exists():
        raise RegisteredNegativeConstructionError("Stale building directory exists")
    train_schedule = plan_contract.load_json(train_schedule_path)
    development_schedule = plan_contract.load_json(development_schedule_path)
    joint_signatures = plan_contract.load_json(joint_signature_path)
    balanced.validate_train_development_pair(train_schedule, development_schedule)
    noise_signatures.validate_payload(joint_signatures)

    train_search = JointSearch(
        train_schedule, joint_signatures, split="train"
    )
    train_search_receipt = train_search.run()
    train_plan = materialize_plan(
        split="train",
        assignments=train_search.assignments,
        schedule=train_schedule,
        joint_signatures=joint_signatures,
    )
    development_search = JointSearch(
        development_schedule, joint_signatures, split="development"
    )
    development_search_receipt = development_search.run()
    development_plan = materialize_plan(
        split="development",
        assignments=development_search.assignments,
        schedule=development_schedule,
        joint_signatures=joint_signatures,
    )
    train_audit = plan_contract.validate_plan(
        train_plan, train_schedule, joint_signatures
    )
    development_audit = plan_contract.validate_plan(
        development_plan, development_schedule, joint_signatures
    )
    plan_pair_audit = plan_contract.validate_train_development_plan_pair(
        train_plan,
        development_plan,
        train_schedule,
        development_schedule,
        joint_signatures,
    )

    try:
        building.mkdir(parents=True, exist_ok=False)
        names_and_values = {
            "train_registered_negative_plan.json": train_plan,
            "development_registered_negative_plan.json": development_plan,
        }
        for name, value in names_and_values.items():
            _write_new_json(building / name, value)
        receipt: dict[str, Any] = {
            "version": VERSION,
            "public_design_seeds": PUBLIC_DESIGN_SEEDS,
            "numpy_version": np.__version__,
            "train_search": train_search_receipt,
            "development_search": development_search_receipt,
            "train_audit": train_audit,
            "development_audit": development_audit,
            "plan_pair_audit": plan_pair_audit,
            "inputs": {
                "train_schedule_sha256": hashlib.sha256(
                    train_schedule_path.read_bytes()
                ).hexdigest(),
                "development_schedule_sha256": hashlib.sha256(
                    development_schedule_path.read_bytes()
                ).hexdigest(),
                "joint_signature_file_sha256": hashlib.sha256(
                    joint_signature_path.read_bytes()
                ).hexdigest(),
            },
            "published_files": {
                name: {
                    "size_bytes": (building / name).stat().st_size,
                    "sha256": hashlib.sha256((building / name).read_bytes()).hexdigest(),
                }
                for name in sorted(names_and_values)
            },
            "status": "PASS_ABSTRACT_CONSTRUCTION_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED",
        }
        receipt["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
            receipt
        )
        _write_new_json(building / "construction_receipt.json", receipt)
        building.replace(output_directory)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--train-schedule", required=True, type=Path)
    parser.add_argument("--development-schedule", required=True, type=Path)
    parser.add_argument("--joint-signatures", required=True, type=Path)
    args = parser.parse_args()
    receipt = publish(
        output_directory=args.output_directory,
        train_schedule_path=args.train_schedule,
        development_schedule_path=args.development_schedule,
        joint_signature_path=args.joint_signatures,
    )
    print(plan_contract.canonical_json_bytes(receipt).decode("utf-8"))


if __name__ == "__main__":
    main()
