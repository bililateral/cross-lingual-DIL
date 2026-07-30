#!/usr/bin/env python3
"""Project parser-validated identity slots for label-free M1 rewiring."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_history_common as history
import step28_v13_common as common
import step28_v13_production_chain as production


FIELD_ORDER = {"title": 0, "description": 1}


def _identity_uid(contact_type: str, normalized_value: str) -> str:
    return "id_" + common.canonical_sha256(
        {
            "contact_type": contact_type.strip().lower(),
            "normalized_value": normalized_value.strip().lower(),
        }
    )


def _bundle_uid(world_uid: str, seller_uid: str, identity_uid: str) -> str:
    return "bundle0_" + common.canonical_sha256(
        {
            "world_uid": world_uid,
            "seller_uid": seller_uid,
            "identity_uid": identity_uid,
        }
    )


def _parser_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    token = history.token_key(dict(row))
    if token is None:
        raise common.ContractError("Safe-slot parser row lacks a token")
    return (
        str(row["item_uid"]),
        str(row["source_field"]),
        token[0],
        token[1],
    )


def _edit_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["item_uid"]),
        str(row["field_name"]),
        str(row["identity_type"]).strip().lower(),
        str(row["downstream_canonical_value"]).strip().lower(),
    )


def _nuisance_class(
    rows: Sequence[Mapping[str, Any]],
    *,
    seller_frequency: int,
    direct_frequency_maximum: int,
) -> str:
    if not rows or seller_frequency < 1:
        raise common.ContractError("Nuisance aggregation has no observed rows")
    flags: list[tuple[int, int, int, int]] = []
    for row in rows:
        risk = int(row["product_data_risk_context"])
        direct = int(row["direct_identity_eligible"])
        support = int(row["support_only"])
        seller_facing = int(row["seller_facing_context"])
        if any(
            value not in {0, 1}
            for value in (risk, direct, support, seller_facing)
        ):
            raise common.ContractError("Parser flags are not binary")
        if risk and direct:
            raise common.ContractError(
                "Risky parser row cannot be direct eligible"
            )
        if support and direct:
            raise common.ContractError(
                "Support parser row cannot be direct eligible"
            )
        if not risk and not support and (not direct or not seller_facing):
            raise common.ContractError(
                "Direct parser row has inconsistent flags"
            )
        flags.append((risk, support, direct, seller_facing))
    if any(risk for risk, _support, _direct, _seller_facing in flags):
        return "risky_product"
    if any(support for _risk, support, _direct, _seller_facing in flags):
        return "public_support"
    if seller_frequency > direct_frequency_maximum:
        return "high_frequency_direct"
    return "direct_or_private"


def project_safe_slots(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots_edit: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Join actual parser rows to the minimal edit table without oracle fields."""

    edit_schema = [
        str(value)
        for value in policy["relational_integrity"][
            "renderer_identity_slots_edit_schema"
        ]
    ]
    safe_schema = [
        str(value) for value in policy["placebo"]["rewire_safe_slot_schema"]
    ]
    _world_uid, _seller_index, item_index, validated_parser_rows = (
        production.validate_parser_artifact(
            policy,
            mode=mode,
            split=split,
            sellers=sellers,
            items=items,
            parsed_rows=parsed_rows,
        )
    )
    parser_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    sellers_by_token: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for source_row in validated_parser_rows:
        row = dict(source_row)
        key = _parser_key(row)
        if key in parser_index:
            raise common.ContractError("Safe-slot parser join key is not unique")
        item = item_index.get(key[0])
        if (
            item is None
            or str(item["seller_uid"]) != str(row["seller_uid"])
            or str(item["world_uid"]) != str(row["world_uid"])
        ):
            raise common.ContractError("Safe-slot parser/item lineage drift")
        parser_index[key] = row
        sellers_by_token[
            (str(row["world_uid"]), key[2], key[3])
        ].add(str(row["seller_uid"]))

    edit_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source_row in identity_slots_edit:
        row = dict(source_row)
        if list(row) != edit_schema:
            raise common.ContractError("Safe-slot edit schema/order drift")
        key = _edit_key(row)
        if key in edit_index:
            raise common.ContractError("Safe-slot edit join key is not unique")
        item = item_index.get(key[0])
        start, end = int(row["start"]), int(row["end"])
        if (
            item is None
            or str(item["seller_uid"]) != str(row["seller_uid"])
            or key[1] != "description"
            or type(row["start"]) is not int
            or type(row["end"]) is not int
            or type(row["time_bucket"]) is not int
            or not 0 <= row["time_bucket"] <= 3
            or row["time_bucket"] != item["time_bucket"]
            or not 0 <= start < end <= len(str(item["description"]))
            or str(item["description"])[start:end] != str(row["raw_surface"])
        ):
            raise common.ContractError("Safe-slot edit offset or lineage drift")
        edit_index[key] = row
    if set(parser_index) != set(edit_index):
        raise common.ContractError(
            "Actual parser and safe edit keysets are not exactly equal"
        )

    direct_maximum = int(
        policy["history_features"]["direct_token_seller_frequency_maximum"]
    )
    parsed_rows_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_token_by_uid: dict[str, tuple[str, str, str]] = {}
    for key, parsed in parser_index.items():
        item = item_index[key[0]]
        identity_uid = _identity_uid(key[2], key[3])
        token = (str(item["world_uid"]), key[2], key[3])
        prior_token = identity_token_by_uid.setdefault(identity_uid, token)
        if prior_token != token:
            raise common.ContractError(
                "One identity UID spans multiple world/token domains"
            )
        parsed_rows_by_identity[identity_uid].append(parsed)
    nuisance_by_identity = {
        identity_uid: _nuisance_class(
            parsed_rows_by_identity[identity_uid],
            seller_frequency=len(sellers_by_token[token]),
            direct_frequency_maximum=direct_maximum,
        )
        for identity_uid, token in identity_token_by_uid.items()
    }
    safe_rows: list[dict[str, Any]] = []
    for key in sorted(
        edit_index,
        key=lambda value: (
            str(item_index[value[0]]["world_uid"]).encode("utf-8"),
            str(item_index[value[0]]["seller_uid"]).encode("utf-8"),
            value[0].encode("utf-8"),
            FIELD_ORDER[value[1]],
            value[2].encode("utf-8"),
            value[3].encode("utf-8"),
        ),
    ):
        edit = edit_index[key]
        parsed = parser_index[key]
        item = item_index[key[0]]
        world_uid = str(item["world_uid"])
        seller_uid = str(item["seller_uid"])
        identity_uid = _identity_uid(key[2], key[3])
        nuisance = nuisance_by_identity[identity_uid]
        row = {
            "slot_uid": str(edit["slot_uid"]),
            "bundle_uid": _bundle_uid(world_uid, seller_uid, identity_uid),
            "world_uid": world_uid,
            "item_uid": key[0],
            "seller_uid": seller_uid,
            "field_name": key[1],
            "start": int(edit["start"]),
            "end": int(edit["end"]),
            "identity_uid": identity_uid,
            "identity_type": key[2],
            "downstream_canonical_value": key[3],
            "raw_surface": str(edit["raw_surface"]),
            "time_bucket": int(edit["time_bucket"]),
            "observed_seller_facing_context": int(
                parsed["seller_facing_context"]
            ),
            "observed_product_data_risk_context": int(
                parsed["product_data_risk_context"]
            ),
            "observed_direct_identity_eligible": int(
                parsed["direct_identity_eligible"]
            ),
            "observed_support_only": int(parsed["support_only"]),
            "observed_nuisance_class": nuisance,
        }
        if list(row) != safe_schema:
            raise common.ContractError("Safe-slot output schema/order drift")
        safe_rows.append(row)

    maximum_per_item_type = int(
        policy["placebo"]["maximum_must_extract_slots_per_item_field_type"]
    )
    item_type_counts = Counter(
        (row["world_uid"], row["item_uid"], row["field_name"], row["identity_type"])
        for row in safe_rows
    )
    if any(value > maximum_per_item_type for value in item_type_counts.values()):
        raise common.ContractError("Safe-slot per-item field/type maximum failed")
    ledger = [
        {
            "identity_uid": identity_uid,
            "nuisance_class": nuisance_by_identity[identity_uid],
        }
        for identity_uid in common.utf8_sort(nuisance_by_identity)
    ]
    audit = {
        "safe_slot_count": len(safe_rows),
        "parser_row_count": len(parsed_rows),
        "edit_row_count": len(identity_slots_edit),
        "identity_count": len(ledger),
        "nuisance_class_counts": dict(
            sorted(Counter(row["observed_nuisance_class"] for row in safe_rows).items())
        ),
        "exact_parser_edit_keyset_equality": True,
        "private_role_or_expectation_read": False,
    }
    return safe_rows, ledger, audit
