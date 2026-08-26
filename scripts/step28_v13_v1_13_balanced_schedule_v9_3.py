#!/usr/bin/env python3
"""Validate the label-free 500-world V9.3 structural schedule.

This module does not construct worlds, render text, open truth files, or fit a
model.  It only validates a frozen abstract seller-slot/noise-slot design.
"""

from __future__ import annotations

from collections import Counter
import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "2026-08-25-step28-v13-v1-13-balanced-schedule-v9-3"
WORLD_COUNT = 500
SELLER_SLOT_COUNT = 28
TRIAD_COUNT = 4
DYAD_COUNT = 8
PAIR_COUNT = 378
POSITIVE_PAIR_COUNT_PER_WORLD = 20
EXPECTED_GROUP_SIZES = (3,) * TRIAD_COUNT + (2,) * DYAD_COUNT
EXPECTED_PAIR_COUNT_HISTOGRAM = {26: 206, 27: 172}
EXPECTED_TRIAD_EXPOSURE_HISTOGRAM = {214: 20, 215: 8}
EXPECTED_NOISE_ASSIGNMENT_HISTOGRAM = {17: 4, 18: 24}
SPLITS = ("train", "development", "audit_a", "audit_b")


class BalancedScheduleError(ValueError):
    """Raised when a frozen V9.3 structural schedule is invalid."""


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
        raise BalancedScheduleError(
            f"{name} key drift: expected={sorted(expected)} observed={sorted(value)}"
        )


def _require_int(value: object, *, name: str, lower: int, upper: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ):
        raise BalancedScheduleError(
            f"{name} must be an integer in [{lower}, {upper}]"
        )
    return value


def _pair(left: int, right: int) -> tuple[int, int]:
    if left == right:
        raise BalancedScheduleError("A structural pair cannot be a self-pair")
    return (left, right) if left < right else (right, left)


def _pair_order() -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(SELLER_SLOT_COUNT)
        for right in range(left + 1, SELLER_SLOT_COUNT)
    )


PAIR_ORDER = _pair_order()


def _counter_vector(counter: Counter[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(int(counter[pair]) for pair in PAIR_ORDER)


def _histogram(values: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(value) for value in values).items()))


def _validate_world(
    row: Mapping[str, Any], *, expected_ordinal: int
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    _require_exact_keys(
        row,
        {"world_ordinal", "controller_groups", "noise_slot_by_seller_slot"},
        name=f"world[{expected_ordinal}]",
    )
    ordinal = _require_int(
        row["world_ordinal"],
        name="world_ordinal",
        lower=0,
        upper=WORLD_COUNT - 1,
    )
    if ordinal != expected_ordinal:
        raise BalancedScheduleError("World ordinals are not exactly 0..499 in order")

    raw_groups = row["controller_groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != TRIAD_COUNT + DYAD_COUNT:
        raise BalancedScheduleError("A world must contain exactly 12 controller groups")
    groups: list[tuple[int, ...]] = []
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, list):
            raise BalancedScheduleError("Controller groups must be JSON arrays")
        expected_size = EXPECTED_GROUP_SIZES[group_index]
        if len(raw_group) != expected_size:
            raise BalancedScheduleError(
                "Controller groups must be four triads followed by eight dyads"
            )
        group = tuple(
            _require_int(
                value,
                name=f"controller_groups[{group_index}] seller slot",
                lower=0,
                upper=SELLER_SLOT_COUNT - 1,
            )
            for value in raw_group
        )
        if tuple(sorted(group)) != group or len(set(group)) != len(group):
            raise BalancedScheduleError(
                "Each controller group must be strictly ascending and duplicate-free"
            )
        groups.append(group)
    if tuple(sorted(groups[:TRIAD_COUNT])) != tuple(groups[:TRIAD_COUNT]) or tuple(
        sorted(groups[TRIAD_COUNT:])
    ) != tuple(groups[TRIAD_COUNT:]):
        raise BalancedScheduleError("Controller groups are not canonically ordered")
    seller_slots = [slot for group in groups for slot in group]
    if sorted(seller_slots) != list(range(SELLER_SLOT_COUNT)):
        raise BalancedScheduleError(
            "Controller groups are not an exact partition of seller slots 0..27"
        )

    raw_noise = row["noise_slot_by_seller_slot"]
    if not isinstance(raw_noise, list) or len(raw_noise) != SELLER_SLOT_COUNT:
        raise BalancedScheduleError("Noise-slot mapping must contain exactly 28 values")
    noise = tuple(
        _require_int(
            value,
            name="noise_slot_by_seller_slot value",
            lower=0,
            upper=SELLER_SLOT_COUNT - 1,
        )
        for value in raw_noise
    )
    if sorted(noise) != list(range(SELLER_SLOT_COUNT)):
        raise BalancedScheduleError(
            "Each world must map seller slots bijectively onto noise slots"
        )
    return tuple(groups), noise


def _validate_schedule_internal(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    _require_exact_keys(
        payload,
        {
            "version",
            "split",
            "world_count",
            "seller_slot_count",
            "worlds",
            "canonical_self_sha256",
        },
        name="schedule",
    )
    if payload["version"] != VERSION:
        raise BalancedScheduleError("Balanced schedule version drift")
    split = payload["split"]
    if split not in SPLITS:
        raise BalancedScheduleError("Balanced schedule split is invalid")
    if payload["world_count"] != WORLD_COUNT:
        raise BalancedScheduleError("Balanced schedule must contain 500 worlds")
    if payload["seller_slot_count"] != SELLER_SLOT_COUNT:
        raise BalancedScheduleError("Balanced schedule must use 28 seller slots")
    supplied_self = payload["canonical_self_sha256"]
    if (
        not isinstance(supplied_self, str)
        or len(supplied_self) != 64
        or supplied_self != canonical_self_sha256(payload)
    ):
        raise BalancedScheduleError("Balanced schedule canonical self-hash drift")
    raw_worlds = payload["worlds"]
    if not isinstance(raw_worlds, list) or len(raw_worlds) != WORLD_COUNT:
        raise BalancedScheduleError("Balanced schedule world cardinality drift")

    seller_pairs: Counter[tuple[int, int]] = Counter()
    noise_pairs: Counter[tuple[int, int]] = Counter()
    seller_triad = [0] * SELLER_SLOT_COUNT
    noise_triad = [0] * SELLER_SLOT_COUNT
    assignment = [[0] * SELLER_SLOT_COUNT for _ in range(SELLER_SLOT_COUNT)]
    for expected_ordinal, raw_world in enumerate(raw_worlds):
        if not isinstance(raw_world, Mapping):
            raise BalancedScheduleError("World rows must be JSON objects")
        groups, noise = _validate_world(raw_world, expected_ordinal=expected_ordinal)
        for seller_slot, noise_slot in enumerate(noise):
            assignment[seller_slot][noise_slot] += 1
        positive_count = 0
        for group in groups:
            if len(group) == 3:
                for seller_slot in group:
                    seller_triad[seller_slot] += 1
                    noise_triad[noise[seller_slot]] += 1
            for left_index in range(len(group)):
                for right_index in range(left_index + 1, len(group)):
                    left = group[left_index]
                    right = group[right_index]
                    seller_pairs[_pair(left, right)] += 1
                    noise_pairs[_pair(noise[left], noise[right])] += 1
                    positive_count += 1
        if positive_count != POSITIVE_PAIR_COUNT_PER_WORLD:
            raise BalancedScheduleError("World positive structural pair count drift")

    seller_vector = _counter_vector(seller_pairs)
    noise_vector = _counter_vector(noise_pairs)
    if _histogram(seller_vector) != EXPECTED_PAIR_COUNT_HISTOGRAM:
        raise BalancedScheduleError("Seller-slot pair balance is not exact 26/27")
    if _histogram(noise_vector) != EXPECTED_PAIR_COUNT_HISTOGRAM:
        raise BalancedScheduleError("Noise-slot pair balance is not exact 26/27")
    if _histogram(seller_triad) != EXPECTED_TRIAD_EXPOSURE_HISTOGRAM:
        raise BalancedScheduleError("Seller-slot triad exposure is not exact 214/215")
    if _histogram(noise_triad) != EXPECTED_TRIAD_EXPOSURE_HISTOGRAM:
        raise BalancedScheduleError("Noise-slot triad exposure is not exact 214/215")
    for row in assignment:
        if _histogram(row) != EXPECTED_NOISE_ASSIGNMENT_HISTOGRAM:
            raise BalancedScheduleError(
                "A seller slot does not receive noise slots exactly 17/18 times"
            )
    for noise_slot in range(SELLER_SLOT_COUNT):
        column = [assignment[seller_slot][noise_slot] for seller_slot in range(28)]
        if _histogram(column) != EXPECTED_NOISE_ASSIGNMENT_HISTOGRAM:
            raise BalancedScheduleError(
                "A noise slot is not assigned to seller slots exactly 17/18 times"
            )

    audit = {
        "version": VERSION,
        "split": split,
        "world_count": WORLD_COUNT,
        "seller_slot_count": SELLER_SLOT_COUNT,
        "pair_count_per_world": PAIR_COUNT,
        "positive_pair_count_per_world": POSITIVE_PAIR_COUNT_PER_WORLD,
        "seller_pair_count_histogram": {
            str(key): value for key, value in EXPECTED_PAIR_COUNT_HISTOGRAM.items()
        },
        "noise_pair_count_histogram": {
            str(key): value for key, value in EXPECTED_PAIR_COUNT_HISTOGRAM.items()
        },
        "seller_triad_exposure_histogram": {
            str(key): value
            for key, value in EXPECTED_TRIAD_EXPOSURE_HISTOGRAM.items()
        },
        "noise_triad_exposure_histogram": {
            str(key): value
            for key, value in EXPECTED_TRIAD_EXPOSURE_HISTOGRAM.items()
        },
        "noise_assignment_histogram_per_row_and_column": {
            str(key): value
            for key, value in EXPECTED_NOISE_ASSIGNMENT_HISTOGRAM.items()
        },
        "seller_pair_count_vector_sha256": canonical_sha256(seller_vector),
        "noise_pair_count_vector_sha256": canonical_sha256(noise_vector),
        "seller_triad_exposure_vector_sha256": canonical_sha256(
            tuple(seller_triad)
        ),
        "noise_triad_exposure_vector_sha256": canonical_sha256(tuple(noise_triad)),
        "schedule_canonical_self_sha256": supplied_self,
        "status": "PASS_ABSTRACT_BALANCE_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED",
    }
    return (
        audit,
        seller_vector,
        noise_vector,
        tuple(seller_triad),
        tuple(noise_triad),
    )


def validate_schedule(payload: Mapping[str, Any]) -> dict[str, Any]:
    audit, *_vectors = _validate_schedule_internal(payload)
    return audit


def _indicator_distance(
    left: Sequence[int], right: Sequence[int], *, high_value: int, name: str
) -> dict[str, int]:
    if len(left) != len(right):
        raise BalancedScheduleError(f"{name} indicator vector length drift")
    left_high = {index for index, value in enumerate(left) if value == high_value}
    right_high = {index for index, value in enumerate(right) if value == high_value}
    if not left_high or len(left_high) != len(right_high):
        raise BalancedScheduleError(f"{name} high-count set cardinality drift")
    intersection = len(left_high & right_high)
    hamming = len(left_high ^ right_high)
    if hamming == 0:
        raise BalancedScheduleError(f"{name} high-count indicators are identical")
    return {
        "vector_length": len(left),
        "high_value": high_value,
        "high_count_per_split": len(left_high),
        "high_set_intersection": intersection,
        "indicator_hamming_distance": hamming,
    }


def _triad_membership_patterns(
    payload: Mapping[str, Any], *, coordinate: str
) -> tuple[tuple[int, ...], ...]:
    """Return the exact 500-world triad-membership pattern of every slot.

    A fixed global relabel cannot change this multiset.  Requiring all patterns
    to be unique makes the resulting non-isomorphism certificate unambiguous
    instead of relying on a heuristic graph matcher.
    """
    patterns = [[0] * WORLD_COUNT for _slot in range(SELLER_SLOT_COUNT)]
    for world_ordinal, raw_world in enumerate(payload["worlds"]):
        groups, noise = _validate_world(raw_world, expected_ordinal=world_ordinal)
        for group in groups[:TRIAD_COUNT]:
            for seller_slot in group:
                slot = seller_slot if coordinate == "seller" else noise[seller_slot]
                patterns[slot][world_ordinal] = 1
    frozen = tuple(tuple(row) for row in patterns)
    if len(set(frozen)) != SELLER_SLOT_COUNT:
        raise BalancedScheduleError(
            f"{coordinate} triad-membership patterns are not unique; "
            "global-relabel audit is inconclusive"
        )
    return frozen


def _reject_global_relabel_isomorphism(
    train_payload: Mapping[str, Any],
    development_payload: Mapping[str, Any],
    *,
    coordinate: str,
) -> dict[str, Any]:
    train_patterns = _triad_membership_patterns(train_payload, coordinate=coordinate)
    development_patterns = _triad_membership_patterns(
        development_payload, coordinate=coordinate
    )
    shared = len(set(train_patterns) & set(development_patterns))
    if shared == SELLER_SLOT_COUNT:
        raise BalancedScheduleError(
            f"Train/development {coordinate} structures are a fixed global "
            "slot relabel of one another"
        )
    return {
        "coordinate": coordinate,
        "slot_count": SELLER_SLOT_COUNT,
        "unique_pattern_count_per_split": SELLER_SLOT_COUNT,
        "shared_exact_pattern_count": shared,
        "fixed_global_relabel_isomorphism_rejected": True,
    }


def validate_train_development_pair(
    train_payload: Mapping[str, Any], development_payload: Mapping[str, Any]
) -> dict[str, Any]:
    (
        train_audit,
        train_seller,
        train_noise,
        train_seller_triad,
        train_noise_triad,
    ) = _validate_schedule_internal(train_payload)
    (
        development_audit,
        development_seller,
        development_noise,
        development_seller_triad,
        development_noise_triad,
    ) = _validate_schedule_internal(development_payload)
    if train_audit["split"] != "train" or development_audit["split"] != "development":
        raise BalancedScheduleError("Train/development schedule roles drift")
    if (
        train_audit["schedule_canonical_self_sha256"]
        == development_audit["schedule_canonical_self_sha256"]
    ):
        raise BalancedScheduleError("Train/development schedule bytes are identical")
    seller_pair_distance = _indicator_distance(
        train_seller,
        development_seller,
        high_value=27,
        name="Train/development seller-pair",
    )
    noise_pair_distance = _indicator_distance(
        train_noise,
        development_noise,
        high_value=27,
        name="Train/development noise-pair",
    )
    seller_triad_distance = _indicator_distance(
        train_seller_triad,
        development_seller_triad,
        high_value=215,
        name="Train/development seller-triad",
    )
    noise_triad_distance = _indicator_distance(
        train_noise_triad,
        development_noise_triad,
        high_value=215,
        name="Train/development noise-triad",
    )
    seller_isomorphism_audit = _reject_global_relabel_isomorphism(
        train_payload, development_payload, coordinate="seller"
    )
    noise_isomorphism_audit = _reject_global_relabel_isomorphism(
        train_payload, development_payload, coordinate="noise"
    )
    return {
        "version": VERSION,
        "train_schedule_sha256": train_audit["schedule_canonical_self_sha256"],
        "development_schedule_sha256": development_audit[
            "schedule_canonical_self_sha256"
        ],
        "seller_pair_count_vectors_distinct": True,
        "noise_pair_count_vectors_distinct": True,
        "seller_pair_indicator_distance": seller_pair_distance,
        "noise_pair_indicator_distance": noise_pair_distance,
        "seller_triad_indicator_distance": seller_triad_distance,
        "noise_triad_indicator_distance": noise_triad_distance,
        "seller_global_relabel_audit": seller_isomorphism_audit,
        "noise_global_relabel_audit": noise_isomorphism_audit,
        "status": "PASS_ABSTRACT_TRAIN_DEVELOPMENT_PAIR_ONLY_NOT_DATA_OR_TRAINING_QUALIFIED",
    }


def load_schedule(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BalancedScheduleError("Balanced schedule root must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--paired-schedule", type=Path)
    args = parser.parse_args()
    payload = load_schedule(args.schedule)
    if args.paired_schedule is None:
        audit = validate_schedule(payload)
    else:
        audit = validate_train_development_pair(
            payload, load_schedule(args.paired_schedule)
        )
    print(canonical_json_bytes(audit).decode("utf-8"))


if __name__ == "__main__":
    main()
