#!/usr/bin/env python3
"""Build the frozen mechanism-covered, class-stratified Step28-v13 C40."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common


SAFE_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
AUDIT_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "label_stratum",
    "covered_positive_mechanisms",
    "covered_negative_flags",
    "selection_role",
    "hmac_digest_hex",
    "selected_rank",
)
POSITIVE_COUNT_BY_SPLIT = {
    "train": 16,
    "development": 10,
    "audit_a": 10,
    "audit_b": 10,
}
POSITIVE_MECHANISMS = frozenset(
    {
        "single_identity_stable_reuse",
        "multi_type_identity_reuse",
        "cross_market_stable_reuse",
        "single_hop_rotation",
        "corroborated_two_hop_rotation",
        "sparse_history",
        "same_controller_no_direct_share",
        "zero_visible_identity_history",
    }
)
NEGATIVE_FLAGS = frozenset(
    {
        "support_hub_pair",
        "high_frequency_direct_hub_pair",
        "risky_shared_token_pair",
        "private_collision_target",
        "false_rotation_target",
        "exact_title_clone_target",
        "high_semantic_similarity_target",
    }
)
MEMBERSHIP_FIELDS = {"world_uid", "controller_uid", "seller_uid"}
POSITIVE_TARGET_FIELDS = {
    "controller_uid",
    "mechanism",
    "mechanism_slot_uid",
    "seller_uid_left",
    "seller_uid_right",
    "canonical_pair_uid",
}
NEGATIVE_FLAG_FIELDS = {
    "canonical_pair_uid",
    "flag",
    "asset_index",
}


def _utf8(value: str) -> bytes:
    return value.encode("utf-8")


def _safe_pair(row: Mapping[str, Any], *, world_uid: str) -> dict[str, str]:
    if set(row) != set(SAFE_FIELDS):
        raise common.ContractError("Complete pair schema drift in C40 sampler")
    output = {name: str(row[name]) for name in SAFE_FIELDS}
    if (
        output["world_uid"] != world_uid
        or output["seller_uid_left"] == output["seller_uid_right"]
        or output["canonical_pair_uid"]
        != common.canonical_pair_uid(
            output["seller_uid_left"],
            output["seller_uid_right"],
        )
    ):
        raise common.ContractError("Complete pair lineage drift in C40 sampler")
    return output


def _rank_key(
    *,
    key_hex: str,
    world_uid: str,
    pair_uid: str,
) -> tuple[bytes, bytes]:
    return (
        common.hmac_digest(
            key_hex,
            world_uid,
            "mechanism_stratified_c40",
            pair_uid,
        ),
        _utf8(pair_uid),
    )


def build_world_c40(
    *,
    split: str,
    candidate_key_hex: str,
    world_uid: str,
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    controller_membership: Sequence[Mapping[str, Any]],
    positive_targets: Sequence[Mapping[str, Any]],
    negative_flags: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Cover every mechanism/flag, then HMAC-fill exact class budgets."""

    if split not in POSITIVE_COUNT_BY_SPLIT:
        raise common.ContractError("Unknown C40 split")
    positive_budget = POSITIVE_COUNT_BY_SPLIT[split]
    negative_budget = 40 - positive_budget
    controller: dict[str, str] = {}
    for row in controller_membership:
        seller_uid = str(row["seller_uid"])
        if (
            set(row) != MEMBERSHIP_FIELDS
            or seller_uid in controller
            or str(row["world_uid"]) != world_uid
            or not str(row["controller_uid"])
        ):
            raise common.ContractError("Controller membership drift in C40")
        controller[seller_uid] = str(row["controller_uid"])
    if len(controller) != 28:
        raise common.ContractError("C40 requires exactly 28 sellers")

    pairs = [
        _safe_pair(row, world_uid=world_uid)
        for row in complete_pair_endpoints
    ]
    pair_by_uid = {
        row["canonical_pair_uid"]: row
        for row in pairs
    }
    if len(pairs) != 378 or len(pair_by_uid) != 378:
        raise common.ContractError("C40 complete pair universe is not 378")
    if {
        endpoint
        for row in pairs
        for endpoint in (
            row["seller_uid_left"],
            row["seller_uid_right"],
        )
    } != set(controller):
        raise common.ContractError("C40 pair/membership seller universes differ")

    rank_key_by_uid = {
        pair_uid: _rank_key(
            key_hex=candidate_key_hex,
            world_uid=world_uid,
            pair_uid=pair_uid,
        )
        for pair_uid in pair_by_uid
    }
    ranked_uids = sorted(pair_by_uid, key=rank_key_by_uid.__getitem__)
    label_by_uid = {
        pair_uid: int(
            controller[pair["seller_uid_left"]]
            == controller[pair["seller_uid_right"]]
        )
        for pair_uid, pair in pair_by_uid.items()
    }
    if (
        sum(label_by_uid.values()) != 20
        or len(label_by_uid) - sum(label_by_uid.values()) != 358
    ):
        raise common.ContractError(
            "Controller topology is not exact 20 positive/358 negative"
        )

    positive_groups: dict[str, set[str]] = defaultdict(set)
    negative_groups: dict[str, set[str]] = defaultdict(set)
    positive_by_pair: dict[str, set[str]] = defaultdict(set)
    negative_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in positive_targets:
        allowed_fields = (
            POSITIVE_TARGET_FIELDS
            if "world_uid" not in row
            else POSITIVE_TARGET_FIELDS | {"world_uid"}
        )
        if set(row) != allowed_fields:
            raise common.ContractError("Positive target schema drift")
        if "world_uid" in row and str(row["world_uid"]) != world_uid:
            raise common.ContractError("Positive target world drift")
        pair_uid = str(row["canonical_pair_uid"])
        mechanism = str(row["mechanism"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        if (
            pair_uid not in pair_by_uid
            or label_by_uid[pair_uid] != 1
            or mechanism not in POSITIVE_MECHANISMS
            or not str(row["mechanism_slot_uid"])
            or pair_uid != common.canonical_pair_uid(left, right)
            or {
                left,
                right,
            }
            != {
                pair_by_uid[pair_uid]["seller_uid_left"],
                pair_by_uid[pair_uid]["seller_uid_right"],
            }
            or controller[left] != controller[right]
            or controller[left] != str(row["controller_uid"])
        ):
            raise common.ContractError("Invalid positive mechanism target")
        positive_groups[mechanism].add(pair_uid)
        positive_by_pair[pair_uid].add(mechanism)
    for row in negative_flags:
        allowed_fields = (
            NEGATIVE_FLAG_FIELDS
            if "world_uid" not in row
            else NEGATIVE_FLAG_FIELDS | {"world_uid"}
        )
        if set(row) != allowed_fields:
            raise common.ContractError("Negative flag schema drift")
        if "world_uid" in row and str(row["world_uid"]) != world_uid:
            raise common.ContractError("Negative flag world drift")
        pair_uid = str(row["canonical_pair_uid"])
        flag = str(row["flag"])
        if (
            pair_uid not in pair_by_uid
            or label_by_uid[pair_uid] != 0
            or flag not in NEGATIVE_FLAGS
            or type(row["asset_index"]) is not int
            or int(row["asset_index"]) < 0
        ):
            raise common.ContractError("Invalid negative mechanism target")
        negative_groups[flag].add(pair_uid)
        negative_by_pair[pair_uid].add(flag)
    if (
        set(positive_groups) != POSITIVE_MECHANISMS
        or set(negative_groups) != NEGATIVE_FLAGS
    ):
        raise common.ContractError(
            "Registered C40 mechanism/flag group count drift"
        )

    selected_positive = {
        min(pair_uids, key=rank_key_by_uid.__getitem__)
        for pair_uids in positive_groups.values()
    }
    selected_negative = {
        min(pair_uids, key=rank_key_by_uid.__getitem__)
        for pair_uids in negative_groups.values()
    }
    coverage_uids = selected_positive | selected_negative
    if (
        len(selected_positive) > positive_budget
        or len(selected_negative) > negative_budget
    ):
        raise common.ContractError("C40 mechanism coverage exceeds class budget")
    for pair_uid in ranked_uids:
        label = label_by_uid[pair_uid]
        if label == 1 and len(selected_positive) < positive_budget:
            selected_positive.add(pair_uid)
        elif label == 0 and len(selected_negative) < negative_budget:
            selected_negative.add(pair_uid)
        if (
            len(selected_positive) == positive_budget
            and len(selected_negative) == negative_budget
        ):
            break
    selected_uids = selected_positive | selected_negative
    if (
        len(selected_uids) != 40
        or len(selected_positive) != positive_budget
        or len(selected_negative) != negative_budget
        or any(
            selected_uids.isdisjoint(pair_uids)
            for pair_uids in positive_groups.values()
        )
        or any(
            selected_uids.isdisjoint(pair_uids)
            for pair_uids in negative_groups.values()
        )
    ):
        raise common.ContractError("C40 class fill or mechanism coverage failed")

    selected_order = sorted(selected_uids, key=rank_key_by_uid.__getitem__)
    safe_rows = [pair_by_uid[pair_uid] for pair_uid in selected_order]
    audit_rows: list[dict[str, Any]] = []
    for selected_rank, pair_uid in enumerate(selected_order, start=1):
        row = {
            "canonical_pair_uid": pair_uid,
            "world_uid": world_uid,
            "label_stratum": label_by_uid[pair_uid],
            "covered_positive_mechanisms": "|".join(
                common.utf8_sort(positive_by_pair.get(pair_uid, set()))
            ),
            "covered_negative_flags": "|".join(
                common.utf8_sort(negative_by_pair.get(pair_uid, set()))
            ),
            "selection_role": (
                "mechanism_coverage"
                if pair_uid in coverage_uids
                else "class_hmac_fill"
            ),
            "hmac_digest_hex": rank_key_by_uid[pair_uid][0].hex(),
            "selected_rank": selected_rank,
        }
        if tuple(row) != AUDIT_FIELDS:
            raise common.ContractError("C40 private audit schema drift")
        audit_rows.append(row)
    if any(set(row) != set(SAFE_FIELDS) for row in safe_rows):
        raise common.ContractError("C40 public projection leaked audit fields")
    summary = {
        "world_uid": world_uid,
        "split": split,
        "pair_count": 40,
        "positive_count": positive_budget,
        "negative_count": negative_budget,
        "positive_mechanism_count": len(positive_groups),
        "negative_flag_count": len(negative_groups),
        "all_positive_mechanisms_covered": True,
        "all_negative_flags_covered": True,
        "model_visible_sampling_fields": False,
        "candidate_input_items_or_text_read": False,
        "canonical_self_hash": common.canonical_sha256(
            {
                "safe_rows": safe_rows,
                "audit_rows": audit_rows,
            }
        ),
    }
    return safe_rows, audit_rows, summary
