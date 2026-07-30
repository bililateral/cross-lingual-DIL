#!/usr/bin/env python3
"""Build the exact 33 parser-observable history features for all 378 pairs."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_history_common as history
import step28_v13_common as common
import step28_v13_production_chain as production


SPLITS = {"train", "development", "audit_a", "audit_b"}
SOURCE_FIELDS = {"title", "description"}
UID_PATTERNS = {
    "world_uid": re.compile(r"^w_[0-9a-f]{64}$"),
    "seller_uid": re.compile(r"^sel_[0-9a-f]{64}$"),
    "item_uid": re.compile(r"^itm_[0-9a-f]{64}$"),
}
FLAG_NAMES = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)


def _exact_schema(
    row: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if list(row) != list(expected):
        raise common.ContractError(
            f"{label} schema/order drift: expected={list(expected)} "
            f"observed={list(row)}"
        )


def _validate_pair_rows(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, set[str]],
]:
    schema = [
        str(value)
        for value in policy["relational_integrity"][
            "pair_projection_contract"
        ]["complete_model_pair_endpoints_schema"]
    ]
    by_world: dict[str, list[dict[str, str]]] = defaultdict(list)
    global_keys: set[tuple[str, str]] = set()
    for source_row in rows:
        _exact_schema(source_row, schema, label="complete pair endpoint")
        row = {name: source_row[name] for name in schema}
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise common.ContractError("Complete pair endpoint contains a non-string")
        world_uid = row["world_uid"]
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        if (
            UID_PATTERNS["world_uid"].fullmatch(world_uid) is None
            or UID_PATTERNS["seller_uid"].fullmatch(left) is None
            or UID_PATTERNS["seller_uid"].fullmatch(right) is None
            or left == right
            or common.utf8_sort((left, right)) != [left, right]
            or row["canonical_pair_uid"]
            != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError("Complete pair endpoint value-domain drift")
        key = (world_uid, row["canonical_pair_uid"])
        if key in global_keys:
            raise common.ContractError("Complete pair endpoint key collision")
        global_keys.add(key)
        by_world[world_uid].append(row)
    if not by_world:
        raise common.ContractError("Complete pair endpoint table is empty")

    sellers_by_world: dict[str, set[str]] = {}
    for world_uid, world_rows in by_world.items():
        sellers = {
            row[name]
            for row in world_rows
            for name in ("seller_uid_left", "seller_uid_right")
        }
        expected_pair_uids = {
            common.canonical_pair_uid(left, right)
            for left, right in itertools.combinations(
                common.utf8_sort(sellers), 2
            )
        }
        observed_pair_uids = {
            row["canonical_pair_uid"] for row in world_rows
        }
        if (
            len(sellers) != 28
            or len(world_rows) != 378
            or observed_pair_uids != expected_pair_uids
        ):
            raise common.ContractError(
                "Complete pair table is not the full 28-choose-2 universe"
            )
        world_rows.sort(
            key=lambda row: row["canonical_pair_uid"].encode("utf-8")
        )
        sellers_by_world[world_uid] = sellers
    return dict(by_world), sellers_by_world


def _validate_history_rows(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    sellers_by_world: Mapping[str, set[str]],
    item_index: Mapping[str, tuple[str, str, int]],
) -> dict[str, list[dict[str, Any]]]:
    schema = [
        str(value)
        for value in policy["relational_integrity"][
            "observed_core_schemas"
        ]["history_safe_occurrences.csv"]
    ]
    identity_types = set(policy["identity_design"]["identity_types"])
    expected_source_dataset = common.source_dataset_name(
        policy,
        mode=mode,
        split=split,
    )
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, str, str, str, str]] = set()
    for source_row in rows:
        _exact_schema(source_row, schema, label="history-safe occurrence")
        row = dict(source_row)
        world_uid = row["world_uid"]
        seller_uid = row["seller_uid"]
        item_uid = row["item_uid"]
        string_fields = (
            "world_uid",
            "seller_uid",
            "item_uid",
            "source_dataset",
            "source_row_number",
            "source_field",
            "contact_type",
            "normalized_value",
        )
        if any(
            not isinstance(row[name], str)
            or not row[name]
            or unicodedata.normalize("NFC", row[name]) != row[name]
            for name in string_fields
        ):
            raise common.ContractError(
                "History-safe occurrence string-domain drift"
            )
        if (
            world_uid not in sellers_by_world
            or seller_uid not in sellers_by_world[world_uid]
            or UID_PATTERNS["world_uid"].fullmatch(world_uid) is None
            or UID_PATTERNS["seller_uid"].fullmatch(seller_uid) is None
            or UID_PATTERNS["item_uid"].fullmatch(item_uid) is None
            or row["source_dataset"] != expected_source_dataset
            or row["source_row_number"] != item_uid
            or row["source_field"] not in SOURCE_FIELDS
            or row["contact_type"] not in identity_types
            or row["contact_type"] != row["contact_type"].strip().lower()
            or row["normalized_value"]
            != row["normalized_value"].strip().lower()
            or type(row["time_bucket"]) is not int
            or not 0 <= row["time_bucket"] <= 3
            or item_uid not in item_index
            or item_index[item_uid]
            != (world_uid, seller_uid, row["time_bucket"])
            or any(
                type(row[name]) is not int or row[name] not in {0, 1}
                for name in FLAG_NAMES
            )
        ):
            raise common.ContractError(
                "History-safe occurrence lineage or value-domain drift"
            )
        direct = row["direct_identity_eligible"]
        if direct and (
            not row["seller_facing_context"]
            or row["product_data_risk_context"]
            or row["support_only"]
        ):
            raise common.ContractError("History-safe direct flags are inconsistent")
        key = (
            world_uid,
            item_uid,
            row["source_field"],
            row["contact_type"],
            row["normalized_value"],
        )
        if key in keys:
            raise common.ContractError("History-safe occurrence key collision")
        keys.add(key)
        by_world[world_uid].append(row)
    unknown_worlds = set(by_world) - set(sellers_by_world)
    if unknown_worlds:
        raise common.ContractError("History-safe occurrence has an unknown world")
    return dict(by_world)


def _validate_history_item_index(
    policy: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    sellers_by_world: Mapping[str, set[str]],
) -> dict[str, tuple[str, str, int]]:
    schema = [
        str(value)
        for value in policy["relational_integrity"][
            "observed_core_schemas"
        ]["history_item_index.csv"]
    ]
    item_index: dict[str, tuple[str, str, int]] = {}
    seller_counts: Counter[tuple[str, str]] = Counter()
    worlds_seen: set[str] = set()
    for source_row in rows:
        _exact_schema(source_row, schema, label="history item index")
        world_uid = source_row["world_uid"]
        seller_uid = source_row["seller_uid"]
        item_uid = source_row["item_uid"]
        time_bucket = source_row["time_bucket"]
        if (
            not isinstance(world_uid, str)
            or not isinstance(seller_uid, str)
            or not isinstance(item_uid, str)
            or UID_PATTERNS["world_uid"].fullmatch(world_uid) is None
            or UID_PATTERNS["seller_uid"].fullmatch(seller_uid) is None
            or UID_PATTERNS["item_uid"].fullmatch(item_uid) is None
            or world_uid not in sellers_by_world
            or seller_uid not in sellers_by_world[world_uid]
            or item_uid in item_index
            or type(time_bucket) is not int
            or not 0 <= time_bucket <= 3
        ):
            raise common.ContractError("History item-index value-domain drift")
        item_index[item_uid] = (world_uid, seller_uid, time_bucket)
        seller_counts[(world_uid, seller_uid)] += 1
        worlds_seen.add(world_uid)
    minimum = int(policy["world_design"]["item_count_minimum"])
    maximum = int(policy["world_design"]["item_count_maximum"])
    expected_sellers = {
        (world_uid, seller_uid)
        for world_uid, sellers in sellers_by_world.items()
        for seller_uid in sellers
    }
    if (
        not item_index
        or worlds_seen != set(sellers_by_world)
        or set(seller_counts) != expected_sellers
        or any(
            not minimum <= count <= maximum
            for count in seller_counts.values()
        )
    ):
        raise common.ContractError("History item-index cardinality drift")
    return item_index


def _validate_projection_attestations(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    attestations: Sequence[Mapping[str, Any]],
    history_rows_by_world: Mapping[str, Sequence[Mapping[str, Any]]],
    history_item_index: Sequence[Mapping[str, Any]],
    expected_worlds: set[str],
) -> None:
    if mode == "formal":
        raise common.ContractError(
            "Formal history projection custody verification is not released"
        )
    fields = [
        "version",
        "mode",
        "split",
        "world_uid",
        "history_safe_occurrence_count",
        "history_safe_occurrences_sha256",
        "history_item_index_count",
        "history_item_index_sha256",
        "parser_artifact_row_count",
        "parser_artifact_sha256",
        "parser_exact_replay",
        "private_plan_exact",
        "projection_producer_path",
        "projection_producer_sha256",
        "step3_parser_code_sha256",
        "custody_status",
        "custody_parent_seal_sha256",
        "canonical_self_hash",
    ]
    item_rows_by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in history_item_index:
        item_rows_by_world[str(row["world_uid"])].append(row)
    by_world: dict[str, Mapping[str, Any]] = {}
    expected_producer_sha = common.sha256_file(
        common.ROOT / "scripts" / "step28_v13_production_chain.py"
    )
    for attestation in attestations:
        _exact_schema(attestation, fields, label="history projection attestation")
        world_uid = attestation["world_uid"]
        without_self = {
            name: attestation[name]
            for name in fields
            if name != "canonical_self_hash"
        }
        if (
            not isinstance(world_uid, str)
            or world_uid in by_world
            or attestation["version"]
            != "2026-07-27-step28-v13-history-projection-attestation-v1"
            or attestation["mode"] != mode
            or attestation["split"] != split
            or attestation["canonical_self_hash"]
            != common.canonical_sha256(without_self)
            or attestation["parser_exact_replay"] is not True
            or attestation["private_plan_exact"] is not True
            or attestation["projection_producer_path"]
            != "scripts/step28_v13_production_chain.py"
            or attestation["projection_producer_sha256"]
            != expected_producer_sha
            or attestation["step3_parser_code_sha256"]
            != policy["frozen_inputs"]["step3_parser_profile_code"]["sha256"]
        ):
            raise common.ContractError("History projection attestation drift")
        if mode == "formal":
            if (
                attestation["custody_status"] != "FORMAL_PARENT_SEALED"
                or not isinstance(
                    attestation["custody_parent_seal_sha256"], str
                )
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    attestation["custody_parent_seal_sha256"],
                )
                is None
            ):
                raise common.ContractError(
                    "Formal history projection lacks a custody seal"
                )
        else:
            expected_nonformal_status = {
                "development_smoke": (
                    "DEVELOPMENT_SMOKE_IN_PROCESS_NOT_FORMAL_SEAL"
                ),
                "training_ready": (
                    "TRAINING_READY_SPLIT_PRIVATE_MATHEMATICAL_REPLAY_"
                    "NOT_OS_CUSTODY_ATTESTED"
                ),
            }.get(mode)
            if (
                expected_nonformal_status is None
                or attestation["custody_status"]
                != expected_nonformal_status
                or attestation["custody_parent_seal_sha256"] is not None
            ):
                raise common.ContractError(
                    "Non-formal history projection custody status drift"
                )
        safe_rows = list(history_rows_by_world.get(world_uid, []))
        item_rows = item_rows_by_world.get(world_uid, [])
        if (
            type(attestation["history_safe_occurrence_count"]) is not int
            or attestation["history_safe_occurrence_count"] != len(safe_rows)
            or attestation["history_safe_occurrences_sha256"]
            != common.canonical_rows_sha256(
                safe_rows,
                order_fields=(
                    "world_uid",
                    "seller_uid",
                    "item_uid",
                    "source_field",
                    "contact_type",
                    "normalized_value",
                ),
            )
            or type(attestation["history_item_index_count"]) is not int
            or attestation["history_item_index_count"] != len(item_rows)
            or attestation["history_item_index_sha256"]
            != common.canonical_rows_sha256(
                item_rows,
                order_fields=("world_uid", "seller_uid", "item_uid"),
            )
            or type(attestation["parser_artifact_row_count"]) is not int
            or attestation["parser_artifact_row_count"]
            != len(safe_rows)
            or not isinstance(attestation["parser_artifact_sha256"], str)
            or re.fullmatch(
                r"[0-9a-f]{64}", attestation["parser_artifact_sha256"]
            )
            is None
        ):
            raise common.ContractError(
                "History projection bytes disagree with their attestation"
            )
        by_world[world_uid] = attestation
    if set(by_world) != expected_worlds:
        raise common.ContractError("History projection attestation world keyset drift")


def _feature_contract(
    policy: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    feature_names = [
        str(value) for value in policy["history_features"]["feature_names"]
    ]
    excluded_names = [
        str(value)
        for value in policy["history_features"][
            "excluded_history_feature_names"
        ]
    ]
    if (
        len(feature_names) != 33
        or len(set(feature_names)) != 33
        or len(excluded_names) != 4
        or set(feature_names) & set(excluded_names)
    ):
        raise common.ContractError(
            "Identity33 selected/excluded feature contract drift"
        )
    return feature_names, excluded_names


def _compute_identity33(
    policy: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    excluded_names: Sequence[str],
    pair_rows_by_world: Mapping[str, Sequence[Mapping[str, str]]],
    history_rows_by_world: Mapping[str, Sequence[Mapping[str, Any]]],
    history_safe_occurrence_count: int,
    history_item_index_count: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Compute features after the caller has independently validated lineage."""

    history_policy = {
        "generation": {
            "direct_token_seller_frequency_maximum": int(
                policy["history_features"][
                    "direct_token_seller_frequency_maximum"
                ]
            ),
            "weak_graph_token_seller_frequency_maximum": int(
                policy["history_features"][
                    "weak_graph_token_seller_frequency_maximum"
                ]
            ),
        }
    }
    rows: list[dict[str, str]] = []
    per_world: list[dict[str, Any]] = []
    zero_feature_pair_count = 0
    output_schema = ["canonical_pair_uid", "world_uid", *feature_names]
    for world_uid in common.utf8_sort(pair_rows_by_world):
        signals = list(history_rows_by_world.get(world_uid, []))
        by_seller, token_df = history.build_signal_index(signals)
        graph = history.build_identity_graph(
            by_seller,
            token_df,
            history_policy,
        )
        world_zero_count = 0
        for pair_row in pair_rows_by_world[world_uid]:
            features, _details = history.history_feature_details(
                pair_row["seller_uid_left"],
                pair_row["seller_uid_right"],
                by_seller,
                token_df,
                graph,
                history_policy,
            )
            if set(features) != set(feature_names) | set(excluded_names):
                raise common.ContractError(
                    "Frozen history helper feature keyset drift"
                )
            values = [float(features[name]) for name in feature_names]
            if any(
                not math.isfinite(value) or value < 0.0
                for value in values
            ):
                raise common.ContractError(
                    "Identity33 feature value-domain drift"
                )
            is_zero = all(value == 0.0 for value in values)
            world_zero_count += int(is_zero)
            zero_feature_pair_count += int(is_zero)
            row = {
                "canonical_pair_uid": pair_row["canonical_pair_uid"],
                "world_uid": world_uid,
                **{
                    name: f"{value:.12f}"
                    for name, value in zip(
                        feature_names, values, strict=True
                    )
                },
            }
            _exact_schema(row, output_schema, label="identity33 output")
            rows.append(row)
        per_world.append(
            {
                "world_uid": world_uid,
                "occurrence_count": len(signals),
                "pair_count": len(pair_rows_by_world[world_uid]),
                "zero_feature_pair_count": world_zero_count,
                "observed_identity_token_count": len(token_df),
            }
        )
    expected_keys = {
        (row["world_uid"], row["canonical_pair_uid"])
        for world_rows in pair_rows_by_world.values()
        for row in world_rows
    }
    observed_keys = {
        (row["world_uid"], row["canonical_pair_uid"]) for row in rows
    }
    if (
        expected_keys != observed_keys
        or len(rows) != 378 * len(pair_rows_by_world)
    ):
        raise common.ContractError("Identity33 all-pair keyset drift")
    audit = {
        "world_count": len(pair_rows_by_world),
        "history_safe_occurrence_count": history_safe_occurrence_count,
        "history_item_index_count": history_item_index_count,
        "pair_count": len(rows),
        "feature_count": len(feature_names),
        "zero_feature_pair_count": zero_feature_pair_count,
        "per_world": per_world,
        "identity33_sha256": common.canonical_sha256(rows),
    }
    return rows, audit


def build_identity33_all_pairs(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    history_safe_occurrences: Sequence[Mapping[str, Any]],
    history_item_index: Sequence[Mapping[str, Any]],
    projection_attestations: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Compute the frozen 33 features without labels, roles, or private plans."""

    common.validate_policy(policy, mode=mode)
    if split not in SPLITS:
        raise common.ContractError(f"Unknown identity33 split: {split}")
    common.verify_file_pin(
        policy["frozen_inputs"]["identity_history_code"],
        label="identity-history feature code",
    )
    feature_names, excluded_names = _feature_contract(policy)
    pair_rows_by_world, sellers_by_world = _validate_pair_rows(
        policy,
        complete_model_pair_endpoints,
    )
    item_index = _validate_history_item_index(
        policy,
        rows=history_item_index,
        sellers_by_world=sellers_by_world,
    )
    history_rows_by_world = _validate_history_rows(
        policy,
        mode=mode,
        split=split,
        rows=history_safe_occurrences,
        sellers_by_world=sellers_by_world,
        item_index=item_index,
    )
    _validate_projection_attestations(
        policy,
        mode=mode,
        split=split,
        attestations=projection_attestations,
        history_rows_by_world=history_rows_by_world,
        history_item_index=history_item_index,
        expected_worlds=set(pair_rows_by_world),
    )
    return _compute_identity33(
        policy,
        feature_names=feature_names,
        excluded_names=excluded_names,
        pair_rows_by_world=pair_rows_by_world,
        history_rows_by_world=history_rows_by_world,
        history_safe_occurrence_count=len(history_safe_occurrences),
        history_item_index_count=len(history_item_index),
    )


def build_identity33_from_rewired_parser(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    rewire_seed_id: str,
    sellers: Sequence[Mapping[str, Any]],
    rewired_items: Sequence[Mapping[str, Any]],
    rewired_parsed_identity_occurrences: Sequence[Mapping[str, Any]],
    rewired_history_safe_occurrences: Sequence[Mapping[str, Any]],
    history_item_index: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Reparse one smoke placebo and build its 33 label-free features.

    This is a separate lineage path from the unrewired projection attestation:
    it reruns the production parser on the supplied rewired observed items and
    requires byte-for-byte row equality before computing any feature.
    """

    common.validate_policy(policy, mode=mode)
    if mode != "development_smoke" or split != "train":
        raise common.ContractError(
            "Rewired identity33 construction is smoke-train only"
        )
    common.verify_file_pin(
        policy["frozen_inputs"]["identity_history_code"],
        label="identity-history feature code",
    )
    registered_seed_ids = {
        "rws_" + hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()
        for seed_hex in policy["randomness"][mode]["rewire_key_hexes"]
    }
    if (
        rewire_seed_id not in registered_seed_ids
        or len(registered_seed_ids)
        != int(policy["placebo"]["replicates"])
    ):
        raise common.ContractError("Rewired identity33 seed ID is not registered")

    feature_names, excluded_names = _feature_contract(policy)
    pair_rows_by_world, sellers_by_world = _validate_pair_rows(
        policy,
        complete_model_pair_endpoints,
    )
    item_index = _validate_history_item_index(
        policy,
        rows=history_item_index,
        sellers_by_world=sellers_by_world,
    )
    schemas = policy["relational_integrity"]["observed_core_schemas"]
    seller_schema = schemas["sellers.csv"]
    item_schema = schemas["items.jsonl"]
    parser_schema = schemas[
        "parsed_identity_occurrences.structural_audit_private.csv"
    ]
    history_schema = schemas["history_safe_occurrences.csv"]
    if any(list(row) != seller_schema for row in sellers):
        raise common.ContractError("Rewired identity33 seller schema drift")
    if any(list(row) != item_schema for row in rewired_items):
        raise common.ContractError("Rewired identity33 item schema drift")
    if any(
        list(row) != parser_schema
        for row in rewired_parsed_identity_occurrences
    ):
        raise common.ContractError("Rewired identity33 parser schema drift")
    if any(
        list(row) != history_schema
        for row in rewired_history_safe_occurrences
    ):
        raise common.ContractError("Rewired identity33 history schema drift")

    sellers_by_world_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    items_by_world_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    parser_by_world_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in sellers:
        sellers_by_world_rows[str(row["world_uid"])].append(row)
    for row in rewired_items:
        items_by_world_rows[str(row["world_uid"])].append(row)
    for row in rewired_parsed_identity_occurrences:
        parser_by_world_rows[str(row["world_uid"])].append(row)
    expected_worlds = set(pair_rows_by_world)
    if (
        set(sellers_by_world_rows) != expected_worlds
        or set(items_by_world_rows) != expected_worlds
        or set(parser_by_world_rows) != expected_worlds
        or any(
            len(sellers_by_world_rows[world_uid]) != 28
            for world_uid in expected_worlds
        )
    ):
        raise common.ContractError(
            "Rewired identity33 complete-world boundary drift"
        )
    observed_item_index = {
        str(row["item_uid"]): (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            int(row["time_bucket"]),
        )
        for row in rewired_items
    }
    if (
        len(observed_item_index) != len(rewired_items)
        or observed_item_index != item_index
    ):
        raise common.ContractError(
            "Rewired identity33 item-index lineage drift"
        )

    reparsed_rows: list[dict[str, Any]] = []
    reprojected_rows: list[dict[str, Any]] = []
    for world_uid in common.utf8_sort(expected_worlds):
        world_parsed = production.parse_observed_world(
            policy,
            mode=mode,
            split=split,
            sellers=sellers_by_world_rows[world_uid],
            items=items_by_world_rows[world_uid],
        )
        if world_parsed != [
            dict(row) for row in parser_by_world_rows[world_uid]
        ]:
            raise common.ContractError(
                "Rewired identity33 production parser replay drift"
            )
        reparsed_rows.extend(world_parsed)
        reprojected_rows.extend(
            production.project_history_safe_occurrences(
                policy,
                mode=mode,
                split=split,
                sellers=sellers_by_world_rows[world_uid],
                items=items_by_world_rows[world_uid],
                parsed_rows=world_parsed,
            )
        )
    if reparsed_rows != [
        dict(row) for row in rewired_parsed_identity_occurrences
    ]:
        raise common.ContractError(
            "Rewired identity33 aggregate parser order drift"
        )
    if reprojected_rows != [
        dict(row) for row in rewired_history_safe_occurrences
    ]:
        raise common.ContractError(
            "Rewired identity33 safe-history projection drift"
        )
    history_rows_by_world = _validate_history_rows(
        policy,
        mode=mode,
        split=split,
        rows=reprojected_rows,
        sellers_by_world=sellers_by_world,
        item_index=item_index,
    )
    rows, audit = _compute_identity33(
        policy,
        feature_names=feature_names,
        excluded_names=excluded_names,
        pair_rows_by_world=pair_rows_by_world,
        history_rows_by_world=history_rows_by_world,
        history_safe_occurrence_count=len(reprojected_rows),
        history_item_index_count=len(history_item_index),
    )
    return rows, {
        **audit,
        "rewire_seed_id": rewire_seed_id,
        "lineage_evidence": (
            "DEVELOPMENT_REPARSED_REWIRED_ITEMS_NOT_FORMAL_CUSTODY"
        ),
        "production_parser_exact_replay": True,
        "safe_history_projection_exact_replay": True,
        "rewired_items_sha256": common.canonical_sha256(
            [dict(row) for row in rewired_items]
        ),
        "rewired_parser_sha256": common.canonical_sha256(reparsed_rows),
        "rewired_history_sha256": common.canonical_sha256(reprojected_rows),
        "labels_or_controller_inputs_read": False,
    }
