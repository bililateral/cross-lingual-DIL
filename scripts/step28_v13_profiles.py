#!/usr/bin/env python3
"""Build per-world synthetic seller profiles with frozen Step3 primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step3_build_seller_profiles as step3


def _profile_schema(policy: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    spec = policy["frozen_inputs"]["step3_profile_schema"]
    path = common.verify_file_pin(spec, label="Step3 seller-profile schema")
    schema = common.load_json(path)
    fields = [str(value) for value in schema["profile_fields"]]
    compression = dict(schema["compression_policy"])
    if not fields or len(fields) != len(set(fields)) or not compression:
        raise common.ContractError("Step3 seller-profile schema is malformed")
    return fields, compression


def build_world_profiles(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay Step3 profile aggregation inside one independent 28-seller world."""

    if mode not in policy["modes"]:
        raise common.ContractError(f"Unknown profile mode: {mode}")
    if split not in {"train", "development", "audit_a", "audit_b"}:
        raise common.ContractError(f"Unknown profile split: {split}")
    seller_schema = policy["relational_integrity"]["observed_core_schemas"][
        "sellers.csv"
    ]
    item_schema = policy["relational_integrity"]["observed_core_schemas"][
        "items.jsonl"
    ]
    if len(sellers) != 28:
        raise common.ContractError("Per-world Step3 replay requires 28 sellers")
    seller_index: dict[str, dict[str, Any]] = {}
    world_uids: set[str] = set()
    for source_row in sellers:
        if set(source_row) != set(seller_schema):
            raise common.ContractError("Seller schema drift in Step3 replay")
        row = dict(source_row)
        seller_uid = str(row["seller_uid"])
        if not seller_uid or seller_uid in seller_index or not str(row["market"]):
            raise common.ContractError("Invalid seller in Step3 replay")
        seller_index[seller_uid] = row
        world_uids.add(str(row["world_uid"]))
    if len(world_uids) != 1:
        raise common.ContractError("Step3 replay must be world-local")
    world_uid = next(iter(world_uids))

    public_items: list[dict[str, Any]] = []
    item_uids: set[str] = set()
    for source_row in items:
        if set(source_row) != set(item_schema):
            raise common.ContractError("Item schema drift in Step3 replay")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        seller_uid = str(row["seller_uid"])
        if (
            str(row["world_uid"]) != world_uid
            or seller_uid not in seller_index
            or not item_uid
            or item_uid in item_uids
        ):
            raise common.ContractError("Invalid item lineage in Step3 replay")
        item_uids.add(item_uid)
        public_items.append(row)
    public_items.sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        )
    )

    source_dataset = common.source_dataset_name(
        policy,
        mode=mode,
        split=split,
    )
    raw_profiles: dict[str, dict[str, Any]] = {}
    for item in public_items:
        seller_uid = str(item["seller_uid"])
        seller = seller_index[seller_uid]
        meta = {
            "data_bucket": source_dataset,
            "source_dataset": source_dataset,
            "source_row_number": str(item["item_uid"]),
            "seller_uid": seller_uid,
            "source_market_raw": str(seller["market"]),
            "source_seller_raw": seller_uid,
            "source_seller_id_raw": seller_uid,
            "alias_normalized": seller_uid,
        }
        profile = step3.ensure_profile(raw_profiles, meta)
        step3.update_profile(
            profile,
            meta,
            title_raw=item["title"],
            description_raw=item["description"],
            category_raw=item["category"],
            price_raw="",
            ship_from_raw="",
            structured_snapshot="",
            parsed_rating=None,
            source_specific_numeric=None,
        )
    if set(raw_profiles) != set(seller_index):
        raise common.ContractError("Step3 raw profile seller keyset drift")

    fields, compression = _profile_schema(policy)
    specificity = step3.build_specificity_catalog(raw_profiles)
    if (
        int(specificity["seller_count"]) != 28
        or any(
            not isinstance(specificity[name], dict)
            for name in ("title_df", "description_segment_df")
        )
    ):
        raise common.ContractError("Per-world Step3 specificity catalog drift")
    profiles = [
        step3.finalize_profile(
            raw_profiles[seller_uid], compression, specificity
        )
        for seller_uid in common.utf8_sort(raw_profiles)
    ]
    for profile in profiles:
        if list(profile) != fields:
            raise common.ContractError(
                "Finalized synthetic Step3 profile schema/order drift"
            )
        seller_uid = str(profile["seller_uid"])
        expected_item_count = sum(
            str(row["seller_uid"]) == seller_uid for row in public_items
        )
        if (
            str(profile["source_seller_raw"]) != seller_uid
            or str(profile["source_seller_id_raw"]) != seller_uid
            or str(profile["alias_normalized"]) != seller_uid
            or str(profile["source_dataset"]) != source_dataset
            or str(profile["data_bucket"]) != source_dataset
            or int(profile["item_count"]) != expected_item_count
            or not str(profile["profile_text"])
        ):
            raise common.ContractError("Finalized synthetic Step3 profile drift")
    audit = {
        "world_uid": world_uid,
        "seller_count": len(profiles),
        "item_count": len(public_items),
        "specificity_scope": "one_world_only",
        "specificity_title_df_count": len(specificity["title_df"]),
        "specificity_description_segment_df_count": len(
            specificity["description_segment_df"]
        ),
        "profile_bytes_sha256": common.canonical_sha256(profiles),
        "labels_or_private_structure_read": False,
    }
    return profiles, audit
