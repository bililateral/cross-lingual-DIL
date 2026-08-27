#!/usr/bin/env python3
"""Replay V9.4 nuisance features from actual public item rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94
import step28_v13_v1_13_model_visible_projection_v9_4 as projection_v94


VERSION = "2026-08-27-step28-v13-v1-13-model-visible-public-replay-v9-4"
PUBLIC_ITEM_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "logical_item_ordinal",
    "title",
    "description",
    "time_bucket",
)
PUBLIC_ENDPOINT_FIELDS = projection_v94.ENDPOINT_FIELDS


class ModelVisiblePublicReplayV94Error(ValueError):
    """Raised when actual public rows cannot replay the registered projection."""


def replay_public_world(
    *,
    endpoint_rows: Sequence[Mapping[str, Any]],
    item_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, str], ...]:
    item_uids: set[str] = set()
    world_uids: set[str] = set()
    seller_uids: set[str] = set()
    projected_items: list[dict[str, Any]] = []
    for source in item_rows:
        if not isinstance(source, Mapping) or tuple(source) != PUBLIC_ITEM_FIELDS:
            raise ModelVisiblePublicReplayV94Error("Public item schema/order drift")
        world_uid = source["world_uid"]
        seller_uid = source["seller_uid"]
        item_uid = source["item_uid"]
        ordinal = source["logical_item_ordinal"]
        title = source["title"]
        description = source["description"]
        time_bucket = source["time_bucket"]
        if (
            not isinstance(world_uid, str)
            or not world_uid
            or not isinstance(seller_uid, str)
            or not seller_uid
            or "|" in seller_uid
            or not isinstance(item_uid, str)
            or not item_uid
            or item_uid in item_uids
            or type(ordinal) is not int
            or not isinstance(title, str)
            or not isinstance(description, str)
            or type(time_bucket) is not int
            or not 0 <= time_bucket < 4
        ):
            raise ModelVisiblePublicReplayV94Error("Public item value drift")
        item_uids.add(item_uid)
        world_uids.add(world_uid)
        seller_uids.add(seller_uid)
        projected_items.append({
            "world_uid": world_uid,
            "seller_uid": seller_uid,
            "logical_item_ordinal": ordinal,
            "title_nonempty": title != "",
            "description_nonempty": description != "",
            "time_bucket": time_bucket,
        })
    if len(world_uids) != 1 or len(seller_uids) != 28:
        raise ModelVisiblePublicReplayV94Error("Public world/seller closure drift")
    world_uid = next(iter(world_uids))
    endpoints: list[dict[str, Any]] = []
    for source in endpoint_rows:
        if not isinstance(source, Mapping) or tuple(source) != PUBLIC_ENDPOINT_FIELDS:
            raise ModelVisiblePublicReplayV94Error("Public endpoint schema/order drift")
        endpoints.append(dict(source))
    return projection_v94.build_world_projection(
        endpoint_rows=endpoints,
        item_rows=projected_items,
    )


def require_exact_replay(
    *,
    registered_rows: Sequence[Mapping[str, Any]],
    public_endpoint_rows: Sequence[Mapping[str, Any]],
    public_item_rows: Sequence[Mapping[str, Any]],
) -> matrix_v94.FrozenMatrix:
    registered = matrix_v94.freeze_matrix(
        registered_rows,
        expected_row_count=projection_v94.PAIRS_PER_WORLD,
    )
    replayed_rows = replay_public_world(
        endpoint_rows=public_endpoint_rows,
        item_rows=public_item_rows,
    )
    replayed = matrix_v94.freeze_matrix(
        replayed_rows,
        expected_row_count=projection_v94.PAIRS_PER_WORLD,
    )
    if (
        registered.row_keys != replayed.row_keys
        or registered.column_names != replayed.column_names
        or registered.joint_commitment_sha256 != replayed.joint_commitment_sha256
        or registered.raw_f8_c_sha256 != replayed.raw_f8_c_sha256
        or [dict(row) for row in registered_rows]
        != [dict(row) for row in replayed_rows]
    ):
        raise ModelVisiblePublicReplayV94Error(
            "Registered/public model-visible projection drift"
        )
    return replayed


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "public_endpoint_fields": list(PUBLIC_ENDPOINT_FIELDS),
        "public_item_fields": list(PUBLIC_ITEM_FIELDS),
        "title_nonempty": "exact_UTF-8_string_is_not_empty",
        "description_nonempty": "exact_UTF-8_string_is_not_empty",
        "time_bucket_source": "actual_public_time_index",
        "exact_replay_bindings": [
            "canonical_rows",
            "row_keys",
            "column_names",
            "little_endian_float64_C_order_bytes",
            "joint_commitment_sha256",
            "actual_persisted_endpoint_rows",
        ],
    }
