#!/usr/bin/env python3
"""Project V9.4's 14 nuisance features from a narrow truth-free surface."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94


VERSION = "2026-08-27-step28-v13-v1-13-model-visible-projection-v9-4"
ENDPOINT_FIELDS = (
    "world_uid",
    "canonical_pair_uid",
    "seller_uid_left",
    "seller_uid_right",
)
ITEM_FIELDS = (
    "world_uid",
    "seller_uid",
    "logical_item_ordinal",
    "title_nonempty",
    "description_nonempty",
    "time_bucket",
)
SELLERS_PER_WORLD = 28
PAIRS_PER_WORLD = 378


class ModelVisibleProjectionV94Error(ValueError):
    """Raised when the truth-free projection surface drifts."""


def _canonical_pair_uid(left: str, right: str) -> str:
    if left == right or "|" in left or "|" in right:
        raise ModelVisibleProjectionV94Error("Endpoint seller key is invalid")
    ordered = sorted((left, right), key=lambda value: value.encode("utf-8"))
    return f"{ordered[0]}||{ordered[1]}"


def _validate_item_rows(
    rows: Sequence[Mapping[str, Any]], *, world_uid: str
) -> dict[str, dict[str, float]]:
    item_counts: Counter[str] = Counter()
    title_missing: Counter[str] = Counter()
    description_missing: Counter[str] = Counter()
    time_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    ordinals: defaultdict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for source in rows:
        if not isinstance(source, Mapping):
            raise ModelVisibleProjectionV94Error("Item projection row is not a mapping")
        if tuple(source) != ITEM_FIELDS:
            raise ModelVisibleProjectionV94Error("Item projection schema/order drift")
        if source["world_uid"] != world_uid:
            raise ModelVisibleProjectionV94Error("Item projection world drift")
        seller_uid = source["seller_uid"]
        ordinal = source["logical_item_ordinal"]
        title_nonempty = source["title_nonempty"]
        description_nonempty = source["description_nonempty"]
        time_bucket = source["time_bucket"]
        if (
            not isinstance(seller_uid, str)
            or not seller_uid
            or type(ordinal) is not int
            or type(title_nonempty) is not bool
            or type(description_nonempty) is not bool
            or type(time_bucket) is not int
            or not 0 <= time_bucket < 4
            or (seller_uid, ordinal) in seen
        ):
            raise ModelVisibleProjectionV94Error("Item projection value drift")
        seen.add((seller_uid, ordinal))
        item_counts[seller_uid] += 1
        title_missing[seller_uid] += int(not title_nonempty)
        description_missing[seller_uid] += int(not description_nonempty)
        time_counts[seller_uid][time_bucket] += 1
        ordinals[seller_uid].append(ordinal)
    if len(item_counts) != SELLERS_PER_WORLD:
        raise ModelVisibleProjectionV94Error("Item projection seller count drift")
    output: dict[str, dict[str, float]] = {}
    for seller_uid in sorted(item_counts, key=lambda value: value.encode("utf-8")):
        count = item_counts[seller_uid]
        if not 2 <= count <= 8 or sorted(ordinals[seller_uid]) != list(range(count)):
            raise ModelVisibleProjectionV94Error("Logical item ordinal closure drift")
        output[seller_uid] = {
            "item_count": float(count),
            "title_missing_rate": title_missing[seller_uid] / count,
            "description_missing_rate": description_missing[seller_uid] / count,
            **{
                f"time_bucket_probability_{index:02d}": time_counts[seller_uid][index]
                / count
                for index in range(4)
            },
        }
    return output


def build_world_projection(
    *,
    endpoint_rows: Sequence[Mapping[str, Any]],
    item_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, str], ...]:
    """Build all 378 rows for one world without labels or private mechanisms."""

    if len(endpoint_rows) != PAIRS_PER_WORLD:
        raise ModelVisibleProjectionV94Error("Endpoint row count drift")
    world_values: set[str] = set()
    for source in endpoint_rows:
        if not isinstance(source, Mapping):
            raise ModelVisibleProjectionV94Error("Endpoint row is not a mapping")
        if tuple(source) != ENDPOINT_FIELDS:
            raise ModelVisibleProjectionV94Error("Endpoint schema/order drift")
        source_world_uid = source["world_uid"]
        if not isinstance(source_world_uid, str):
            raise ModelVisibleProjectionV94Error("Endpoint world domain drift")
        world_values.add(source_world_uid)
    if len(world_values) != 1:
        raise ModelVisibleProjectionV94Error("Endpoint world domain drift")
    world_uid = next(iter(world_values))
    if not world_uid:
        raise ModelVisibleProjectionV94Error("Endpoint world UID is empty")
    seller_vectors = _validate_item_rows(item_rows, world_uid=world_uid)
    pair_uids: set[str] = set()
    endpoint_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for source in endpoint_rows:
        if tuple(source) != ENDPOINT_FIELDS:
            raise ModelVisibleProjectionV94Error("Endpoint schema/order drift")
        pair_uid = source["canonical_pair_uid"]
        left = source["seller_uid_left"]
        right = source["seller_uid_right"]
        if (
            source["world_uid"] != world_uid
            or not isinstance(pair_uid, str)
            or not pair_uid
            or not isinstance(left, str)
            or not isinstance(right, str)
            or left not in seller_vectors
            or right not in seller_vectors
            or left == right
            or pair_uid != _canonical_pair_uid(left, right)
            or pair_uid in pair_uids
        ):
            raise ModelVisibleProjectionV94Error("Endpoint key/value drift")
        ordered_pair = tuple(sorted((left, right), key=lambda value: value.encode("utf-8")))
        if ordered_pair in endpoint_pairs:
            raise ModelVisibleProjectionV94Error("Duplicate unordered seller pair")
        pair_uids.add(pair_uid)
        endpoint_pairs.add(ordered_pair)
        rows.append(
            matrix_v94.pair_feature_row(
                world_uid=world_uid,
                canonical_pair_uid=pair_uid,
                left=seller_vectors[left],
                right=seller_vectors[right],
            )
        )
    expected_sellers = set(seller_vectors)
    if (
        {seller for pair in endpoint_pairs for seller in pair} != expected_sellers
        or len(endpoint_pairs) != len(expected_sellers) * (len(expected_sellers) - 1) // 2
    ):
        raise ModelVisibleProjectionV94Error("Endpoint rows are not a complete graph")
    rows.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["canonical_pair_uid"].encode("utf-8"),
        )
    )
    frozen_rows = tuple(MappingProxyType(dict(row)) for row in rows)
    matrix_v94.freeze_matrix(frozen_rows, expected_row_count=PAIRS_PER_WORLD)
    return frozen_rows


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "endpoint_fields": list(ENDPOINT_FIELDS),
        "item_fields": list(ITEM_FIELDS),
        "sellers_per_world": SELLERS_PER_WORLD,
        "pairs_per_world": PAIRS_PER_WORLD,
        "matrix_contract": matrix_v94.contract_payload(),
        "reads_controller_membership": False,
        "reads_registered_override_audit": False,
        "reads_pair_labels": False,
    }
