#!/usr/bin/env python3
"""Construct and publish the small abstract V9.3 train/development schedules.

The constructor uses a public design constant and never opens text, identities,
truth, model outputs, or an experiment random authority.  Published schedules
remain abstract design inputs and are not datasets or training-qualified roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import step28_v13_v1_13_balanced_schedule_v9_3 as contract


VERSION = "2026-08-25-step28-v13-v1-13-construct-balanced-schedule-v9-3"
PUBLIC_DESIGN_SEEDS = {
    "train": 281320260825,
    "development": 281320260827,
}
DEFAULT_MAX_ITERATIONS = 12_000_000
REPORT_VERSION = "v1_20260825"

GROUP_BY_POSITION = np.asarray(
    [0] * 3
    + [1] * 3
    + [2] * 3
    + [3] * 3
    + [4] * 2
    + [5] * 2
    + [6] * 2
    + [7] * 2
    + [8] * 2
    + [9] * 2
    + [10] * 2
    + [11] * 2,
    dtype=np.int8,
)
GROUP_POSITIONS = tuple(
    np.flatnonzero(GROUP_BY_POSITION == group_index)
    for group_index in range(12)
)


class BalancedScheduleConstructionError(RuntimeError):
    """Raised when the one frozen construction trajectory fails."""


def _penalty(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return (lower - value) ** 2
    if value > upper:
        return (value - upper) ** 2
    return 0


def _matrix_penalty(matrix: np.ndarray) -> int:
    return sum(
        _penalty(int(matrix[left, right]), 26, 27)
        for left in range(contract.SELLER_SLOT_COUNT)
        for right in range(left + 1, contract.SELLER_SLOT_COUNT)
    )


def _triad_penalty(values: np.ndarray) -> int:
    return sum(_penalty(int(value), 214, 215) for value in values)


def _balanced_noise_assignments(rng: np.random.Generator) -> np.ndarray:
    """Build 500 varied bijections with exact 17/18 row and column counts."""
    slot_count = contract.SELLER_SLOT_COUNT
    rows: list[np.ndarray] = []
    for round_index in range(18):
        seller_order = rng.permutation(slot_count)
        noise_order = rng.permutation(slot_count)
        shifts = list(range(slot_count))
        if round_index == 17:
            shifts = sorted(rng.choice(slot_count, size=24, replace=False).tolist())
        for shift in shifts:
            mapping = np.empty(slot_count, dtype=np.int16)
            for position, seller_slot in enumerate(seller_order):
                mapping[int(seller_slot)] = int(
                    noise_order[(position + shift) % slot_count]
                )
            rows.append(mapping)
    order = rng.permutation(len(rows))
    return np.vstack([rows[int(index)] for index in order])


def construct_base(
    *, split: str, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if max_iterations != DEFAULT_MAX_ITERATIONS:
        raise BalancedScheduleConstructionError(
            "The frozen construction iteration limit cannot be overridden"
        )
    if split not in PUBLIC_DESIGN_SEEDS:
        raise BalancedScheduleConstructionError("Unknown frozen construction split")
    public_design_seed = PUBLIC_DESIGN_SEEDS[split]
    rng = np.random.Generator(np.random.PCG64(public_design_seed))
    world_count = contract.WORLD_COUNT
    slot_count = contract.SELLER_SLOT_COUNT
    noise_by_seller = _balanced_noise_assignments(rng)
    if noise_by_seller.shape != (world_count, slot_count):
        raise BalancedScheduleConstructionError("Noise assignment cardinality drift")
    permutations = np.vstack(
        [rng.permutation(slot_count) for _world in range(world_count)]
    ).astype(np.int16, copy=False)

    seller_pairs = np.zeros((slot_count, slot_count), dtype=np.int16)
    noise_pairs = np.zeros((slot_count, slot_count), dtype=np.int16)
    seller_triad = np.zeros(slot_count, dtype=np.int16)
    noise_triad = np.zeros(slot_count, dtype=np.int16)
    for world, permutation in enumerate(permutations):
        for positions in GROUP_POSITIONS:
            sellers = permutation[positions]
            noises = noise_by_seller[world, sellers]
            if len(positions) == 3:
                seller_triad[sellers] += 1
                noise_triad[noises] += 1
            for left_index in range(len(sellers)):
                for right_index in range(left_index + 1, len(sellers)):
                    seller_pair = tuple(
                        sorted(
                            (
                                int(sellers[left_index]),
                                int(sellers[right_index]),
                            )
                        )
                    )
                    noise_pair = tuple(
                        sorted(
                            (
                                int(noises[left_index]),
                                int(noises[right_index]),
                            )
                        )
                    )
                    seller_pairs[seller_pair] += 1
                    noise_pairs[noise_pair] += 1

    seller_objective = _matrix_penalty(seller_pairs)
    noise_objective = _matrix_penalty(noise_pairs)
    seller_triad_objective = _triad_penalty(seller_triad)
    noise_triad_objective = _triad_penalty(noise_triad)
    accepted_moves = 0
    solved_iteration: int | None = None

    for iteration in range(1, max_iterations + 1):
        world = int(rng.integers(world_count))
        left_position = int(rng.integers(slot_count))
        right_position = int(rng.integers(slot_count - 1))
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

        delta = (
            seller_delta
            + noise_delta
            + 2 * (seller_triad_delta + noise_triad_delta)
        )
        temperature = max(
            0.015, 6.0 * (1.0 - iteration / DEFAULT_MAX_ITERATIONS)
        )
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
            seller_objective += seller_delta
            noise_objective += noise_delta
            seller_triad_objective += seller_triad_delta
            noise_triad_objective += noise_triad_delta
            if (
                seller_objective
                == noise_objective
                == seller_triad_objective
                == noise_triad_objective
                == 0
            ):
                solved_iteration = iteration
                break
        if iteration % 1_000_000 == 0:
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "seller_pair_penalty": seller_objective,
                        "noise_pair_penalty": noise_objective,
                        "seller_triad_penalty": seller_triad_objective,
                        "noise_triad_penalty": noise_triad_objective,
                        "accepted_moves": accepted_moves,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    if solved_iteration is None:
        raise BalancedScheduleConstructionError(
            "Frozen trajectory did not reach exact dual balance; do not rerun"
        )
    receipt = {
        "version": VERSION,
        "split": split,
        "public_design_seed": public_design_seed,
        "maximum_iterations": DEFAULT_MAX_ITERATIONS,
        "solved_iteration": solved_iteration,
        "accepted_moves": accepted_moves,
        "numpy_version": np.__version__,
        "status": "PASS_ABSTRACT_CONSTRUCTION_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED",
    }
    return permutations, noise_by_seller, receipt


def _schedule_payload(
    *,
    split: str,
    permutations: np.ndarray,
    noise_by_seller: np.ndarray,
) -> dict[str, Any]:
    worlds: list[dict[str, Any]] = []
    for world_ordinal, permutation in enumerate(permutations):
        groups: list[list[int]] = []
        for positions in GROUP_POSITIONS:
            groups.append(
                sorted(int(permutation[position]) for position in positions)
            )
        triads = sorted(group for group in groups if len(group) == 3)
        dyads = sorted(group for group in groups if len(group) == 2)
        mapped_noise = [0] * contract.SELLER_SLOT_COUNT
        for seller_slot in range(contract.SELLER_SLOT_COUNT):
            mapped_noise[seller_slot] = int(
                noise_by_seller[world_ordinal, seller_slot]
            )
        worlds.append(
            {
                "world_ordinal": world_ordinal,
                "controller_groups": [*triads, *dyads],
                "noise_slot_by_seller_slot": mapped_noise,
            }
        )
    payload: dict[str, Any] = {
        "version": contract.VERSION,
        "split": split,
        "world_count": contract.WORLD_COUNT,
        "seller_slot_count": contract.SELLER_SLOT_COUNT,
        "worlds": worlds,
    }
    payload["canonical_self_sha256"] = contract.canonical_self_sha256(payload)
    return payload


def build_train_development_payloads(
    *, max_iterations: int = DEFAULT_MAX_ITERATIONS
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_permutations, train_noise_by_seller, train_receipt = construct_base(
        split="train", max_iterations=max_iterations
    )
    train = _schedule_payload(
        split="train",
        permutations=train_permutations,
        noise_by_seller=train_noise_by_seller,
    )
    development_permutations, development_noise_by_seller, development_receipt = (
        construct_base(split="development", max_iterations=max_iterations)
    )
    development = _schedule_payload(
        split="development",
        permutations=development_permutations,
        noise_by_seller=development_noise_by_seller,
    )
    train_audit = contract.validate_schedule(train)
    development_audit = contract.validate_schedule(development)
    pair_audit = contract.validate_train_development_pair(train, development)
    receipt = {
        "version": VERSION,
        "public_design_seeds": PUBLIC_DESIGN_SEEDS,
        "train_construction": train_receipt,
        "development_construction": development_receipt,
        "train_audit": train_audit,
        "development_audit": development_audit,
        "train_development_pair_audit": pair_audit,
    }
    return train, development, receipt


def _write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(contract.canonical_json_bytes(value))
        stream.write(b"\n")


def publish(output_directory: Path) -> dict[str, Any]:
    if output_directory.exists():
        raise BalancedScheduleConstructionError("Output directory already exists")
    building = output_directory.with_name(output_directory.name + ".building")
    if building.exists():
        raise BalancedScheduleConstructionError("Stale building directory exists")
    train, development, receipt = build_train_development_payloads()
    try:
        building.mkdir(parents=True, exist_ok=False)
        _write_new_json(building / "train_balanced_schedule.json", train)
        _write_new_json(
            building / "development_balanced_schedule.json", development
        )
        receipt["published_files"] = {
            name: {
                "size_bytes": (building / name).stat().st_size,
                "sha256": hashlib.sha256((building / name).read_bytes()).hexdigest(),
            }
            for name in (
                "development_balanced_schedule.json",
                "train_balanced_schedule.json",
            )
        }
        receipt["canonical_self_sha256"] = contract.canonical_self_sha256(receipt)
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
    args = parser.parse_args()
    receipt = publish(args.output_directory)
    print(contract.canonical_json_bytes(receipt).decode("utf-8"))


if __name__ == "__main__":
    main()
