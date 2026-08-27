#!/usr/bin/env python3
"""Rebuild the V9.4 train/development abstract world schedules from zero."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np


VERSION = "2026-08-27-step28-v13-v1-13-balanced-world-schedule-v9-4"
WORLD_COUNT = 500
SELLER_COUNT = 28
GROUP_SIZES = (3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2)
PAIR_COUNT = SELLER_COUNT * (SELLER_COUNT - 1) // 2
POSITIVE_COUNT = sum(size * (size - 1) // 2 for size in GROUP_SIZES)
PUBLIC_DESIGN_SEEDS = MappingProxyType({
    "train": 281320260825,
    "development": 281320260827,
})
MAX_ITERATIONS = 12_000_000
PAIR_HISTOGRAM = {26: 206, 27: 172}
TRIAD_HISTOGRAM = {214: 20, 215: 8}
NOISE_ASSIGNMENT_HISTOGRAM = {17: 4, 18: 24}
_SCHEDULE_ISSUER = object()
GROUP_BY_POSITION = np.asarray(
    [group_index for group_index, size in enumerate(GROUP_SIZES) for _ in range(size)],
    dtype=np.int8,
)
GROUP_POSITIONS = tuple(
    np.flatnonzero(GROUP_BY_POSITION == group_index)
    for group_index in range(len(GROUP_SIZES))
)


class BalancedWorldScheduleV94Error(ValueError):
    """Raised when the frozen abstract schedule trajectory or closure drifts."""


@dataclass(frozen=True)
class SplitSchedule:
    split: str
    public_worlds: tuple[Mapping[str, Any], ...]
    controller_groups_by_world: tuple[tuple[tuple[str, ...], ...], ...]
    commitment: Mapping[str, Any]
    _issuer: object


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _latent_schedule_sha256(
    permutations: np.ndarray,
    noise_by_seller: np.ndarray,
) -> str:
    canonical_controller_groups = []
    for permutation in permutations:
        groups = [
            tuple(sorted(int(permutation[position]) for position in positions))
            for positions in GROUP_POSITIONS
        ]
        groups.sort(key=lambda group: (-len(group), group))
        canonical_controller_groups.append(groups)
    return _canonical_sha256({
        "controller_groups_by_seller_slot": canonical_controller_groups,
        "noise_by_seller": noise_by_seller.tolist(),
    })


def _penalty(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return (lower - value) ** 2
    if value > upper:
        return (value - upper) ** 2
    return 0


def _pair_penalty(matrix: np.ndarray) -> int:
    return sum(
        _penalty(int(matrix[left, right]), 26, 27)
        for left in range(SELLER_COUNT)
        for right in range(left + 1, SELLER_COUNT)
    )


def _triad_penalty(values: np.ndarray) -> int:
    return sum(_penalty(int(value), 214, 215) for value in values)


def _balanced_noise_assignments(rng: np.random.Generator) -> np.ndarray:
    """Create 500 bijections with exact 17/18 row and column frequencies."""

    rows: list[np.ndarray] = []
    for round_index in range(18):
        seller_order = rng.permutation(SELLER_COUNT)
        noise_order = rng.permutation(SELLER_COUNT)
        shifts: Sequence[int] = range(SELLER_COUNT)
        if round_index == 17:
            shifts = sorted(
                int(value)
                for value in rng.choice(SELLER_COUNT, size=24, replace=False)
            )
        for shift in shifts:
            mapping = np.empty(SELLER_COUNT, dtype=np.int16)
            for position, seller_slot in enumerate(seller_order):
                mapping[int(seller_slot)] = int(
                    noise_order[(position + shift) % SELLER_COUNT]
                )
            rows.append(mapping)
    order = rng.permutation(len(rows))
    result = np.vstack([rows[int(index)] for index in order])
    if result.shape != (WORLD_COUNT, SELLER_COUNT):
        raise BalancedWorldScheduleV94Error("Noise assignment cardinality drift")
    return result


def _initialize_counts(
    permutations: np.ndarray,
    noise_by_seller: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seller_pairs = np.zeros((SELLER_COUNT, SELLER_COUNT), dtype=np.int16)
    noise_pairs = np.zeros((SELLER_COUNT, SELLER_COUNT), dtype=np.int16)
    seller_triad = np.zeros(SELLER_COUNT, dtype=np.int16)
    noise_triad = np.zeros(SELLER_COUNT, dtype=np.int16)
    for world, permutation in enumerate(permutations):
        for positions in GROUP_POSITIONS:
            sellers = permutation[positions]
            noises = noise_by_seller[world, sellers]
            if len(positions) == 3:
                seller_triad[sellers] += 1
                noise_triad[noises] += 1
            for left_index in range(len(sellers)):
                for right_index in range(left_index + 1, len(sellers)):
                    seller_pair = tuple(sorted((
                        int(sellers[left_index]),
                        int(sellers[right_index]),
                    )))
                    noise_pair = tuple(sorted((
                        int(noises[left_index]),
                        int(noises[right_index]),
                    )))
                    seller_pairs[seller_pair] += 1
                    noise_pairs[noise_pair] += 1
    return seller_pairs, noise_pairs, seller_triad, noise_triad


def _construct_arrays(split: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if split not in PUBLIC_DESIGN_SEEDS:
        raise BalancedWorldScheduleV94Error("Unknown schedule split")
    rng = np.random.Generator(np.random.PCG64(PUBLIC_DESIGN_SEEDS[split]))
    noise_by_seller = _balanced_noise_assignments(rng)
    permutations = np.vstack([
        rng.permutation(SELLER_COUNT) for _ in range(WORLD_COUNT)
    ]).astype(np.int16, copy=False)
    seller_pairs, noise_pairs, seller_triad, noise_triad = _initialize_counts(
        permutations,
        noise_by_seller,
    )
    objectives = [
        _pair_penalty(seller_pairs),
        _pair_penalty(noise_pairs),
        _triad_penalty(seller_triad),
        _triad_penalty(noise_triad),
    ]
    accepted_moves = 0
    solved_iteration: int | None = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        world = int(rng.integers(WORLD_COUNT))
        left_position = int(rng.integers(SELLER_COUNT))
        right_position = int(rng.integers(SELLER_COUNT - 1))
        right_position += int(right_position >= left_position)
        left_group = int(GROUP_BY_POSITION[left_position])
        right_group = int(GROUP_BY_POSITION[right_position])
        if left_group == right_group:
            continue
        permutation = permutations[world]
        left_seller = int(permutation[left_position])
        right_seller = int(permutation[right_position])
        left_peers = [
            int(permutation[position])
            for position in GROUP_POSITIONS[left_group]
            if int(position) != left_position
        ]
        right_peers = [
            int(permutation[position])
            for position in GROUP_POSITIONS[right_group]
            if int(position) != right_position
        ]
        seller_changes: dict[tuple[int, int], int] = {}
        noise_changes: dict[tuple[int, int], int] = {}

        def add_change(
            target: dict[tuple[int, int], int],
            first: int,
            second: int,
            delta: int,
        ) -> None:
            pair = tuple(sorted((first, second)))
            target[pair] = target.get(pair, 0) + delta

        for peer in left_peers:
            add_change(seller_changes, left_seller, peer, -1)
            add_change(seller_changes, right_seller, peer, 1)
            add_change(
                noise_changes,
                int(noise_by_seller[world, left_seller]),
                int(noise_by_seller[world, peer]),
                -1,
            )
            add_change(
                noise_changes,
                int(noise_by_seller[world, right_seller]),
                int(noise_by_seller[world, peer]),
                1,
            )
        for peer in right_peers:
            add_change(seller_changes, right_seller, peer, -1)
            add_change(seller_changes, left_seller, peer, 1)
            add_change(
                noise_changes,
                int(noise_by_seller[world, right_seller]),
                int(noise_by_seller[world, peer]),
                -1,
            )
            add_change(
                noise_changes,
                int(noise_by_seller[world, left_seller]),
                int(noise_by_seller[world, peer]),
                1,
            )
        seller_delta = sum(
            _penalty(int(seller_pairs[pair]) + delta, 26, 27)
            - _penalty(int(seller_pairs[pair]), 26, 27)
            for pair, delta in seller_changes.items()
        )
        noise_delta = sum(
            _penalty(int(noise_pairs[pair]) + delta, 26, 27)
            - _penalty(int(noise_pairs[pair]), 26, 27)
            for pair, delta in noise_changes.items()
        )
        seller_triad_changes: dict[int, int] = {}
        noise_triad_changes: dict[int, int] = {}
        seller_triad_delta = 0
        noise_triad_delta = 0
        if len(GROUP_POSITIONS[left_group]) != len(GROUP_POSITIONS[right_group]):
            if len(GROUP_POSITIONS[left_group]) == 3:
                seller_triad_changes = {left_seller: -1, right_seller: 1}
            else:
                seller_triad_changes = {left_seller: 1, right_seller: -1}
            noise_triad_changes = {
                int(noise_by_seller[world, seller]): delta
                for seller, delta in seller_triad_changes.items()
            }
            seller_triad_delta = sum(
                _penalty(int(seller_triad[slot]) + delta, 214, 215)
                - _penalty(int(seller_triad[slot]), 214, 215)
                for slot, delta in seller_triad_changes.items()
            )
            noise_triad_delta = sum(
                _penalty(int(noise_triad[slot]) + delta, 214, 215)
                - _penalty(int(noise_triad[slot]), 214, 215)
                for slot, delta in noise_triad_changes.items()
            )
        delta = seller_delta + noise_delta + 2 * (
            seller_triad_delta + noise_triad_delta
        )
        temperature = max(0.015, 6.0 * (1.0 - iteration / MAX_ITERATIONS))
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            permutation[left_position], permutation[right_position] = (
                permutation[right_position],
                permutation[left_position],
            )
            accepted_moves += 1
            for pair, change in seller_changes.items():
                seller_pairs[pair] += change
            for pair, change in noise_changes.items():
                noise_pairs[pair] += change
            for slot, change in seller_triad_changes.items():
                seller_triad[slot] += change
            for slot, change in noise_triad_changes.items():
                noise_triad[slot] += change
            objectives[0] += seller_delta
            objectives[1] += noise_delta
            objectives[2] += seller_triad_delta
            objectives[3] += noise_triad_delta
            if objectives == [0, 0, 0, 0]:
                solved_iteration = iteration
                break
    if solved_iteration is None:
        raise BalancedWorldScheduleV94Error(
            "Frozen schedule trajectory did not reach exact balance"
        )
    return permutations, noise_by_seller, {
        "public_design_seed": PUBLIC_DESIGN_SEEDS[split],
        "maximum_iterations": MAX_ITERATIONS,
        "solved_iteration": solved_iteration,
        "accepted_moves": accepted_moves,
        "numpy_version": np.__version__,
    }


def _histogram(values: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(value) for value in values).items()))


def _validate_arrays(permutations: np.ndarray, noise: np.ndarray) -> dict[str, Any]:
    if (
        permutations.shape != (WORLD_COUNT, SELLER_COUNT)
        or noise.shape != (WORLD_COUNT, SELLER_COUNT)
        or any(set(row.tolist()) != set(range(SELLER_COUNT)) for row in permutations)
        or any(set(row.tolist()) != set(range(SELLER_COUNT)) for row in noise)
    ):
        raise BalancedWorldScheduleV94Error("Schedule array closure drift")
    seller_pairs, noise_pairs, seller_triad, noise_triad = _initialize_counts(
        permutations,
        noise,
    )
    upper = np.triu_indices(SELLER_COUNT, k=1)
    assignment = np.zeros((SELLER_COUNT, SELLER_COUNT), dtype=np.int16)
    for row in noise:
        assignment[np.arange(SELLER_COUNT), row] += 1
    audits = {
        "seller_pair_histogram": _histogram(seller_pairs[upper]),
        "noise_pair_histogram": _histogram(noise_pairs[upper]),
        "seller_triad_histogram": _histogram(seller_triad),
        "noise_triad_histogram": _histogram(noise_triad),
        "noise_assignment_row_histograms": tuple(
            tuple(sorted(_histogram(row).items())) for row in assignment
        ),
        "noise_assignment_column_histograms": tuple(
            tuple(sorted(_histogram(column).items())) for column in assignment.T
        ),
    }
    if (
        audits["seller_pair_histogram"] != PAIR_HISTOGRAM
        or audits["noise_pair_histogram"] != PAIR_HISTOGRAM
        or audits["seller_triad_histogram"] != TRIAD_HISTOGRAM
        or audits["noise_triad_histogram"] != TRIAD_HISTOGRAM
        or any(dict(value) != NOISE_ASSIGNMENT_HISTOGRAM for value in audits[
            "noise_assignment_row_histograms"
        ])
        or any(dict(value) != NOISE_ASSIGNMENT_HISTOGRAM for value in audits[
            "noise_assignment_column_histograms"
        ])
    ):
        raise BalancedWorldScheduleV94Error("Schedule balance closure drift")
    return audits


def build_split_schedule(split: str) -> SplitSchedule:
    permutations, noise, construction = _construct_arrays(split)
    audit = _validate_arrays(permutations, noise)
    public_worlds: list[Mapping[str, Any]] = []
    private_groups: list[tuple[tuple[str, ...], ...]] = []
    for world_ordinal, permutation in enumerate(permutations):
        world_uid = f"v9_4_{split}_world_{world_ordinal:03d}"
        seller_uids = [
            f"{world_uid}_seller_{seller_slot:02d}"
            for seller_slot in range(SELLER_COUNT)
        ]
        public_worlds.append(MappingProxyType({
            "split": split,
            "world_ordinal": world_ordinal,
            "world_uid": world_uid,
            "seller_uids": tuple(seller_uids),
            "noise_slot_by_seller_slot": tuple(
                int(value) for value in noise[world_ordinal]
            ),
        }))
        groups = [
            tuple(sorted(
                (seller_uids[int(permutation[position])] for position in positions),
                key=lambda value: value.encode("utf-8"),
            ))
            for positions in GROUP_POSITIONS
        ]
        triads = sorted(
            (group for group in groups if len(group) == 3),
            key=lambda value: tuple(item.encode("utf-8") for item in value),
        )
        dyads = sorted(
            (group for group in groups if len(group) == 2),
            key=lambda value: tuple(item.encode("utf-8") for item in value),
        )
        private_groups.append(tuple([*triads, *dyads]))
    public_payload = [
        {
            **dict(row),
            "seller_uids": list(row["seller_uids"]),
            "noise_slot_by_seller_slot": list(row["noise_slot_by_seller_slot"]),
        }
        for row in public_worlds
    ]
    private_payload = [
        [list(group) for group in world_groups]
        for world_groups in private_groups
    ]
    latent_schedule_sha256 = _latent_schedule_sha256(permutations, noise)
    commitment_payload = {
        "version": VERSION,
        "split": split,
        "world_count": WORLD_COUNT,
        "seller_count": SELLER_COUNT,
        "pair_count": PAIR_COUNT,
        "positive_count_per_world": POSITIVE_COUNT,
        "public_worlds_sha256": _canonical_sha256(public_payload),
        "private_controller_truth_sha256": _canonical_sha256(private_payload),
        "latent_schedule_sha256": latent_schedule_sha256,
        "construction": tuple(construction.items()),
        "balance_audit_sha256": _canonical_sha256(audit),
    }
    commitment_payload["split_schedule_commitment_sha256"] = _canonical_sha256(
        commitment_payload
    )
    result = SplitSchedule(
        split=split,
        public_worlds=tuple(public_worlds),
        controller_groups_by_world=tuple(private_groups),
        commitment=MappingProxyType(commitment_payload),
        _issuer=_SCHEDULE_ISSUER,
    )
    verify_split_schedule(result)
    return result


def verify_split_schedule(schedule: SplitSchedule) -> None:
    expected_fields = (
        "version",
        "split",
        "world_count",
        "seller_count",
        "pair_count",
        "positive_count_per_world",
        "public_worlds_sha256",
        "private_controller_truth_sha256",
        "latent_schedule_sha256",
        "construction",
        "balance_audit_sha256",
        "split_schedule_commitment_sha256",
    )
    if (
        type(schedule) is not SplitSchedule
        or schedule._issuer is not _SCHEDULE_ISSUER
        or schedule.split not in PUBLIC_DESIGN_SEEDS
        or len(schedule.public_worlds) != WORLD_COUNT
        or len(schedule.controller_groups_by_world) != WORLD_COUNT
        or type(schedule.commitment) is not MappingProxyType
        or tuple(schedule.commitment) != expected_fields
    ):
        raise BalancedWorldScheduleV94Error("Split schedule capability drift")
    public_payload = [
        {
            **dict(row),
            "seller_uids": list(row["seller_uids"]),
            "noise_slot_by_seller_slot": list(row["noise_slot_by_seller_slot"]),
        }
        for row in schedule.public_worlds
    ]
    private_payload = [
        [list(group) for group in world_groups]
        for world_groups in schedule.controller_groups_by_world
    ]
    construction = schedule.commitment.get("construction")
    if (
        type(construction) is not tuple
        or tuple(key for key, _ in construction) != (
            "public_design_seed",
            "maximum_iterations",
            "solved_iteration",
            "accepted_moves",
            "numpy_version",
        )
    ):
        raise BalancedWorldScheduleV94Error("Split construction receipt drift")
    construction_values = dict(construction)
    if (
        construction_values["public_design_seed"]
        != PUBLIC_DESIGN_SEEDS[schedule.split]
        or construction_values["maximum_iterations"] != MAX_ITERATIONS
        or type(construction_values["solved_iteration"]) is not int
        or not 1 <= construction_values["solved_iteration"] <= MAX_ITERATIONS
        or type(construction_values["accepted_moves"]) is not int
        or construction_values["accepted_moves"] <= 0
        or construction_values["numpy_version"] != np.__version__
    ):
        raise BalancedWorldScheduleV94Error("Split construction value drift")
    for world_ordinal, (public, groups) in enumerate(zip(
        schedule.public_worlds,
        schedule.controller_groups_by_world,
        strict=True,
    )):
        if (
            type(public) is not MappingProxyType
            or tuple(public) != (
                "split",
                "world_ordinal",
                "world_uid",
                "seller_uids",
                "noise_slot_by_seller_slot",
            )
            or public["split"] != schedule.split
            or public["world_ordinal"] != world_ordinal
            or type(public["seller_uids"]) is not tuple
            or len(public["seller_uids"]) != SELLER_COUNT
            or tuple(sorted(
                public["seller_uids"], key=lambda value: value.encode("utf-8")
            )) != public["seller_uids"]
            or type(public["noise_slot_by_seller_slot"]) is not tuple
            or set(public["noise_slot_by_seller_slot"]) != set(range(SELLER_COUNT))
            or len(groups) != len(GROUP_SIZES)
            or tuple(len(group) for group in groups) != GROUP_SIZES
            or set(item for group in groups for item in group)
            != set(public["seller_uids"])
        ):
            raise BalancedWorldScheduleV94Error("Split schedule world closure drift")
    permutations = np.empty((WORLD_COUNT, SELLER_COUNT), dtype=np.int16)
    noise = np.empty((WORLD_COUNT, SELLER_COUNT), dtype=np.int16)
    for world_ordinal, (public, groups) in enumerate(zip(
        schedule.public_worlds,
        schedule.controller_groups_by_world,
        strict=True,
    )):
        seller_slot = {
            seller_uid: index for index, seller_uid in enumerate(public["seller_uids"])
        }
        permutations[world_ordinal] = np.asarray([
            seller_slot[seller_uid]
            for group in groups
            for seller_uid in group
        ], dtype=np.int16)
        noise[world_ordinal] = np.asarray(
            public["noise_slot_by_seller_slot"], dtype=np.int16
        )
    expected_commitment = {
        "version": VERSION,
        "split": schedule.split,
        "world_count": WORLD_COUNT,
        "seller_count": SELLER_COUNT,
        "pair_count": PAIR_COUNT,
        "positive_count_per_world": POSITIVE_COUNT,
        "public_worlds_sha256": _canonical_sha256(public_payload),
        "private_controller_truth_sha256": _canonical_sha256(private_payload),
        "latent_schedule_sha256": _latent_schedule_sha256(
            permutations,
            noise,
        ),
        "construction": schedule.commitment["construction"],
        "balance_audit_sha256": _canonical_sha256(
            _validate_arrays(permutations, noise)
        ),
    }
    if (
        any(
            type(schedule.commitment[key]) is not type(value)
            or schedule.commitment[key] != value
            for key, value in expected_commitment.items()
        )
        or schedule.commitment["split_schedule_commitment_sha256"]
        != _canonical_sha256(expected_commitment)
    ):
        raise BalancedWorldScheduleV94Error("Split schedule commitment drift")


def _schedule_arrays(schedule: SplitSchedule) -> tuple[np.ndarray, np.ndarray]:
    verify_split_schedule(schedule)
    permutations = np.empty((WORLD_COUNT, SELLER_COUNT), dtype=np.int16)
    noise = np.empty((WORLD_COUNT, SELLER_COUNT), dtype=np.int16)
    for world_ordinal, (public, groups) in enumerate(zip(
        schedule.public_worlds,
        schedule.controller_groups_by_world,
        strict=True,
    )):
        seller_slot = {
            seller_uid: index for index, seller_uid in enumerate(public["seller_uids"])
        }
        permutations[world_ordinal] = np.asarray([
            seller_slot[seller_uid]
            for group in groups
            for seller_uid in group
        ], dtype=np.int16)
        noise[world_ordinal] = np.asarray(
            public["noise_slot_by_seller_slot"], dtype=np.int16
        )
    return permutations, noise


def _high_set(values: np.ndarray, high: int) -> set[int]:
    return {
        index for index, value in enumerate(values.tolist()) if int(value) == high
    }


def _triad_patterns(
    schedule: SplitSchedule, *, coordinate: str
) -> tuple[tuple[int, ...], ...]:
    patterns = [[0] * WORLD_COUNT for _ in range(SELLER_COUNT)]
    for world_ordinal, (public, groups) in enumerate(zip(
        schedule.public_worlds,
        schedule.controller_groups_by_world,
        strict=True,
    )):
        seller_slot = {
            seller_uid: index for index, seller_uid in enumerate(public["seller_uids"])
        }
        noise = public["noise_slot_by_seller_slot"]
        for group in groups[:4]:
            for seller_uid in group:
                slot = seller_slot[seller_uid]
                if coordinate == "noise":
                    slot = int(noise[slot])
                patterns[slot][world_ordinal] = 1
    result = tuple(tuple(pattern) for pattern in patterns)
    if len(set(result)) != SELLER_COUNT:
        raise BalancedWorldScheduleV94Error(
            f"{coordinate} triad patterns are not unique"
        )
    return result


def validate_train_development_pair(
    train: SplitSchedule,
    development: SplitSchedule,
) -> MappingProxyType[str, Any]:
    if train.split != "train" or development.split != "development":
        raise BalancedWorldScheduleV94Error("Schedule pair role drift")
    train_permutations, train_noise = _schedule_arrays(train)
    development_permutations, development_noise = _schedule_arrays(development)
    if (
        train.commitment["latent_schedule_sha256"]
        == development.commitment["latent_schedule_sha256"]
    ):
        raise BalancedWorldScheduleV94Error(
            "Train/development latent schedule identity drift"
        )
    train_counts = _initialize_counts(train_permutations, train_noise)
    development_counts = _initialize_counts(development_permutations, development_noise)
    upper = np.triu_indices(SELLER_COUNT, k=1)
    vector_pairs = (
        ("seller_pair", train_counts[0][upper], development_counts[0][upper], 27),
        ("noise_pair", train_counts[1][upper], development_counts[1][upper], 27),
        ("seller_triad", train_counts[2], development_counts[2], 215),
        ("noise_triad", train_counts[3], development_counts[3], 215),
    )
    distances: list[tuple[str, int, int]] = []
    for name, left, right, high in vector_pairs:
        left_high = _high_set(left, high)
        right_high = _high_set(right, high)
        hamming = len(left_high ^ right_high)
        if not left_high or len(left_high) != len(right_high) or hamming == 0:
            raise BalancedWorldScheduleV94Error(
                f"Train/development {name} distance drift"
            )
        distances.append((name, len(left_high & right_high), hamming))
    global_relabel: list[tuple[str, int]] = []
    for coordinate in ("seller", "noise"):
        train_patterns = _triad_patterns(train, coordinate=coordinate)
        development_patterns = _triad_patterns(
            development, coordinate=coordinate
        )
        shared = len(set(train_patterns) & set(development_patterns))
        if shared == SELLER_COUNT:
            raise BalancedWorldScheduleV94Error(
                f"Train/development {coordinate} global relabel drift"
            )
        global_relabel.append((coordinate, shared))
    receipt = {
        "version": VERSION,
        "train_commitment_sha256": train.commitment[
            "split_schedule_commitment_sha256"
        ],
        "development_commitment_sha256": development.commitment[
            "split_schedule_commitment_sha256"
        ],
        "train_latent_schedule_sha256": train.commitment[
            "latent_schedule_sha256"
        ],
        "development_latent_schedule_sha256": development.commitment[
            "latent_schedule_sha256"
        ],
        "indicator_intersection_and_hamming": tuple(distances),
        "shared_exact_triad_patterns": tuple(global_relabel),
        "fixed_global_relabel_rejected": True,
    }
    receipt["pair_audit_commitment_sha256"] = _canonical_sha256(receipt)
    return MappingProxyType(receipt)


def public_world_dicts(schedule: SplitSchedule) -> list[dict[str, Any]]:
    verify_split_schedule(schedule)
    return [
        {
            **dict(world),
            "seller_uids": list(world["seller_uids"]),
            "noise_slot_by_seller_slot": list(
                world["noise_slot_by_seller_slot"]
            ),
        }
        for world in schedule.public_worlds
    ]


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "world_count": WORLD_COUNT,
        "seller_count": SELLER_COUNT,
        "controller_group_sizes": list(GROUP_SIZES),
        "positive_count_per_world": POSITIVE_COUNT,
        "public_design_seeds": dict(PUBLIC_DESIGN_SEEDS),
        "maximum_iterations": MAX_ITERATIONS,
        "direct_r2_plan_read": False,
        "issued_schedule_required": True,
        "latent_schedule_commitment": True,
        "train_development_pair_audit_required": True,
    }
