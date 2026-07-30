#!/usr/bin/env python3
"""Run the frozen Step3 parser and Step7-v4 redactor on one synthetic world.

The functions in this module deliberately separate the observed-data path from
the private structural audit.  Redaction registries are built only from actual
Step3 rows and label-free seller profiles; planned identity slots are accepted
only by the two validator functions.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_history_common as history
import step28_v13_common as common
import step28_v13_text_renderer as renderer
import step3_build_seller_profiles as step3
import step7_v3_1_source_data as source
import step7_v4_common as redactor


FLAG_NAMES = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)
SOURCE_FIELD_ORDER = {"title": 0, "description": 1}
REGISTRY_NAMES = (
    "global_identity_tokens",
    "contextual_global_alias_tokens",
    "contextual_alias_deletion_tokens",
    "seller_identity_literals",
    "seller_identity_phrase_tokens",
    "seller_contextual_collision_tokens",
    "audited_global_identity_phrase_tokens",
)
UID_PATTERNS = {
    "world_uid": re.compile(r"^w_[0-9a-f]{64}$"),
    "seller_uid": re.compile(r"^sel_[0-9a-f]{64}$"),
    "item_uid": re.compile(r"^itm_[0-9a-f]{64}$"),
}


def _schema(policy: Mapping[str, Any], filename: str) -> list[str]:
    try:
        value = policy["relational_integrity"]["observed_core_schemas"][filename]
    except KeyError as exc:
        raise common.ContractError(f"Missing observed schema for {filename}") from exc
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        raise common.ContractError(f"Invalid observed schema for {filename}")
    return [str(name) for name in value]


def _require_exact_keys(
    row: Mapping[str, Any], fieldnames: Sequence[str], *, label: str
) -> None:
    expected = set(fieldnames)
    observed = set(row)
    if observed != expected:
        raise common.ContractError(
            f"{label} schema mismatch: "
            f"missing={common.utf8_sort(expected - observed)} "
            f"extra={common.utf8_sort(observed - expected)}"
        )


def _world_uid(
    sellers: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]
) -> str:
    values = {
        str(row.get("world_uid", ""))
        for row in (*sellers, *items)
        if str(row.get("world_uid", ""))
    }
    if len(values) != 1:
        raise common.ContractError("Observed world rows do not identify one world")
    return next(iter(values))


def _validate_item_value_domains(
    policy: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    categories: set[str],
) -> None:
    for name in ("world_uid", "seller_uid", "item_uid"):
        value = row[name]
        if not isinstance(value, str) or UID_PATTERNS[name].fullmatch(value) is None:
            raise common.ContractError(f"Invalid observed {name}")
    time_bucket = row["time_bucket"]
    if (
        type(time_bucket) is not int
        or not 0 <= time_bucket <= 3
    ):
        raise common.ContractError("Observed time_bucket must be integer 0..3")
    if not isinstance(row["category"], str) or row["category"] not in categories:
        raise common.ContractError("Observed item category is outside the template")
    for name in ("title", "description"):
        value = row[name]
        if (
            not isinstance(value, str)
            or unicodedata.normalize("NFC", value) != value
        ):
            raise common.ContractError(f"Observed {name} must be an NFC string")


def _validate_item_seller_minimums(
    policy: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> None:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in items:
        key = (str(row["world_uid"]), str(row["seller_uid"]))
        counts[key]["items"] += 1
        counts[key]["title"] += bool(str(row["title"]))
        counts[key]["description"] += bool(str(row["description"]))
    minimum_items = int(policy["world_design"]["item_count_minimum"])
    maximum_items = int(policy["world_design"]["item_count_maximum"])
    minimum_titles = int(
        policy["world_design"][
            "post_v4_redaction_minimum_nonempty_title_per_seller"
        ]
    )
    minimum_descriptions = int(
        policy["world_design"][
            "post_v4_redaction_minimum_nonempty_description_per_seller"
        ]
    )
    if any(
        not minimum_items <= value["items"] <= maximum_items
        or value["title"] < minimum_titles
        or value["description"] < minimum_descriptions
        for value in counts.values()
    ):
        raise common.ContractError("Observed seller item/nonempty minima failed")


def _validated_observed_indexes(
    policy: Mapping[str, Any],
    *,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    seller_schema = _schema(policy, "sellers.csv")
    item_schema = _schema(policy, "items.jsonl")
    if len(sellers) != 28:
        raise common.ContractError("A synthetic world must contain exactly 28 sellers")
    world_uid = _world_uid(sellers, items)
    if UID_PATTERNS["world_uid"].fullmatch(world_uid) is None:
        raise common.ContractError("Observed world UID format drift")
    expected_template = common.load_json(
        common.repo_path(str(policy["template_library"]["path"]))
    )
    categories = set(expected_template["generic_lexicon"]["categories"])
    seller_index: dict[str, dict[str, Any]] = {}
    for source_row in sellers:
        _require_exact_keys(source_row, seller_schema, label="seller")
        row = dict(source_row)
        seller_uid = row["seller_uid"]
        if (
            not isinstance(row["world_uid"], str)
            or row["world_uid"] != world_uid
            or not isinstance(seller_uid, str)
            or UID_PATTERNS["seller_uid"].fullmatch(seller_uid) is None
            or seller_uid in seller_index
            or not isinstance(row["market"], str)
            or row["market"] not in set(policy["world_design"]["markets"])
        ):
            raise common.ContractError("Invalid or duplicate observed seller row")
        seller_index[seller_uid] = row

    item_index: dict[str, dict[str, Any]] = {}
    seller_item_counts: Counter[str] = Counter()
    for source_row in items:
        _require_exact_keys(source_row, item_schema, label="item")
        row = dict(source_row)
        _validate_item_value_domains(policy, row, categories=categories)
        item_uid = row["item_uid"]
        seller_uid = row["seller_uid"]
        if (
            row["world_uid"] != world_uid
            or seller_uid not in seller_index
            or not item_uid
            or item_uid in item_index
        ):
            raise common.ContractError("Invalid or duplicate observed item row")
        item_index[item_uid] = row
        seller_item_counts[seller_uid] += 1
    if set(seller_item_counts) != set(seller_index):
        raise common.ContractError("Every observed seller must own at least one item")
    _validate_item_seller_minimums(policy, list(item_index.values()))
    return world_uid, seller_index, item_index


def registry_profiles_from_sellers(
    policy: Mapping[str, Any],
    *,
    sellers: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the minimal redaction registry projection without raw profiles."""

    seller_schema = _schema(policy, "sellers.csv")
    if len(sellers) != 28:
        raise common.ContractError("Registry projection requires exactly 28 sellers")
    world_uids: set[str] = set()
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_row in sellers:
        _require_exact_keys(
            source_row,
            seller_schema,
            label="registry seller",
        )
        row = dict(source_row)
        seller_uid = row["seller_uid"]
        world_uid = row["world_uid"]
        if (
            not isinstance(world_uid, str)
            or UID_PATTERNS["world_uid"].fullmatch(world_uid) is None
            or not isinstance(seller_uid, str)
            or UID_PATTERNS["seller_uid"].fullmatch(seller_uid) is None
            or seller_uid in seen
            or row["market"] not in set(policy["world_design"]["markets"])
        ):
            raise common.ContractError("Registry seller value-domain drift")
        world_uids.add(world_uid)
        seen.add(seller_uid)
        output.append(
            {
                "seller_uid": seller_uid,
                "source_seller_raw": seller_uid,
                "alias_normalized": seller_uid,
            }
        )
    if len(world_uids) != 1:
        raise common.ContractError("Registry sellers do not belong to one world")
    return sorted(output, key=lambda row: row["seller_uid"].encode("utf-8"))


def build_profile_safe_items(
    policy: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join clean text back to item metadata before Step3 profile aggregation."""

    item_schema = _schema(policy, "items.jsonl")
    clean_schema = _schema(policy, "redacted_items.jsonl")
    raw_index: dict[str, dict[str, Any]] = {}
    for source_row in items:
        _require_exact_keys(source_row, item_schema, label="profile-safe raw item")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        if not item_uid or item_uid in raw_index:
            raise common.ContractError("Profile-safe raw item UID collision")
        raw_index[item_uid] = row
    clean_index: dict[str, dict[str, Any]] = {}
    for source_row in redacted_items:
        _require_exact_keys(
            source_row,
            clean_schema,
            label="profile-safe redacted item",
        )
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        if not item_uid or item_uid in clean_index:
            raise common.ContractError("Profile-safe clean item UID collision")
        clean_index[item_uid] = row
    if not raw_index or set(raw_index) != set(clean_index):
        raise common.ContractError("Profile-safe raw/clean item keyset drift")

    output: list[dict[str, Any]] = []
    for item_uid in common.utf8_sort(raw_index):
        raw = raw_index[item_uid]
        clean = clean_index[item_uid]
        if (
            clean["world_uid"] != raw["world_uid"]
            or clean["seller_uid"] != raw["seller_uid"]
            or clean["item_uid"] != raw["item_uid"]
            or not isinstance(clean["title"], str)
            or not isinstance(clean["description"], str)
            or unicodedata.normalize("NFC", clean["title"]) != clean["title"]
            or unicodedata.normalize("NFC", clean["description"])
            != clean["description"]
        ):
            raise common.ContractError("Profile-safe clean item lineage drift")
        row = {
            **raw,
            "title": clean["title"],
            "description": clean["description"],
        }
        _require_exact_keys(row, item_schema, label="profile-safe item")
        output.append(row)
    _validate_item_seller_minimums(policy, output)
    return output


def _extract_step3_rows(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    world_uid: str,
    seller_index: Mapping[str, Mapping[str, Any]],
    item_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay the frozen Step3 parser from observed rows only."""

    source_dataset = common.source_dataset_name(
        policy,
        mode=mode,
        split=split,
    )
    rows: list[dict[str, Any]] = []
    ordered_items = sorted(
        item_index.values(),
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        ),
    )
    for item in ordered_items:
        item_uid = str(item["item_uid"])
        seller_uid = str(item["seller_uid"])
        seller = seller_index[seller_uid]
        meta = {
            "data_bucket": source_dataset,
            "source_dataset": source_dataset,
            "source_row_number": item_uid,
            "seller_uid": seller_uid,
            "source_market_raw": str(seller["market"]),
            "source_seller_raw": seller_uid,
            "source_seller_id_raw": seller_uid,
            "alias_normalized": seller_uid,
        }
        parsed = step3.extract_item_identity_signals(
            meta,
            title_raw=item["title"],
            description_raw=item["description"],
            structured_snapshot="",
            extra_fields=None,
        )
        rows.extend(
            {"world_uid": world_uid, "item_uid": item_uid, **dict(row)}
            for row in parsed
        )
    return rows


def validate_parser_artifact(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate a Step3 parser artifact using only observed world inputs."""

    if mode not in policy["modes"]:
        raise common.ContractError(f"Unknown parser-artifact mode: {mode}")
    if split not in {"train", "development", "audit_a", "audit_b"}:
        raise common.ContractError(f"Unknown parser-artifact split: {split}")
    world_uid, seller_index, item_index = _validated_observed_indexes(
        policy,
        sellers=sellers,
        items=items,
    )
    common.verify_file_pin(
        policy["frozen_inputs"]["step3_parser_profile_code"],
        label="Step3 parser/profile code",
    )
    source_dataset = common.source_dataset_name(
        policy,
        mode=mode,
        split=split,
    )
    fieldnames = _schema(
        policy, "parsed_identity_occurrences.structural_audit_private.csv"
    )
    type_order = {
        identity_type: index
        for index, identity_type in enumerate(
            policy["identity_design"]["identity_types"]
        )
    }
    string_fields = [name for name in fieldnames if name not in FLAG_NAMES]
    validated: list[dict[str, Any]] = []
    semantic_keys: set[tuple[str, str, str, str]] = set()
    signal_uids: set[str] = set()
    for source_row in parsed_rows:
        _require_exact_keys(
            source_row,
            fieldnames,
            label="parsed identity occurrence",
        )
        row = dict(source_row)
        if any(
            not isinstance(row[name], str)
            or unicodedata.normalize("NFC", row[name]) != row[name]
            for name in string_fields
        ):
            raise common.ContractError(
                "Parser artifact contains a non-NFC or non-string field"
            )
        if any(
            type(row[name]) is not int or row[name] not in {0, 1}
            for name in FLAG_NAMES
        ):
            raise common.ContractError("Parser artifact flags are not binary integers")
        item = item_index.get(row["item_uid"])
        token = history.token_key(row)
        seller_uid = row["seller_uid"]
        seller = seller_index.get(seller_uid)
        if (
            item is None
            or seller is None
            or row["world_uid"] != world_uid
            or row["source_dataset"] != source_dataset
            or row["data_bucket"] != source_dataset
            or row["source_row_number"] != row["item_uid"]
            or seller_uid != str(item["seller_uid"])
            or row["source_market_raw"] != str(seller["market"])
            or row["source_seller_raw"] != seller_uid
            or row["source_seller_id_raw"] != seller_uid
            or row["alias_normalized"] != seller_uid
            or row["source_field"] not in SOURCE_FIELD_ORDER
            or row["contact_type"] not in type_order
            or token is None
            or row["contact_type"] != token[0]
            or row["normalized_value"] != token[1]
        ):
            raise common.ContractError("Production Step3 lineage or type drift")
        expected_seller_facing, expected_risk, expected_support = (
            step3.signal_flags(
                row["contact_type"],
                row["evidence_level"],
                row["context"],
            )
        )
        expected_direct = int(
            row["contact_type"] in step3.DIRECT_ITEM_IDENTITY_TYPES
            and expected_seller_facing
            and not expected_risk
        )
        expected_flags = (
            expected_seller_facing,
            expected_risk,
            expected_direct,
            expected_support,
        )
        if tuple(row[name] for name in FLAG_NAMES) != expected_flags:
            raise common.ContractError(
                "Parser artifact flags disagree with frozen Step3 semantics"
            )
        signal_uid_raw = "|".join(
            (
                row["source_dataset"],
                row["source_row_number"],
                row["contact_type"],
                row["normalized_value"],
                row["source_field"],
            )
        )
        expected_signal_uid = hashlib.sha1(
            signal_uid_raw.encode("utf-8")
        ).hexdigest()
        if row["signal_uid"] != expected_signal_uid:
            raise common.ContractError("Parser artifact signal UID drift")
        semantic_key = (
            row["item_uid"],
            row["source_field"],
            row["contact_type"],
            row["normalized_value"],
        )
        if semantic_key in semantic_keys or row["signal_uid"] in signal_uids:
            raise common.ContractError("Parser artifact row is duplicated")
        semantic_keys.add(semantic_key)
        signal_uids.add(row["signal_uid"])
        validated.append(row)
    validated.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
            SOURCE_FIELD_ORDER[row["source_field"]],
            type_order[row["contact_type"]],
            row["normalized_value"].encode("utf-8"),
        )
    )
    expected = _extract_step3_rows(
        policy=policy,
        mode=mode,
        split=split,
        world_uid=world_uid,
        seller_index=seller_index,
        item_index=item_index,
    )
    expected.sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
            SOURCE_FIELD_ORDER[str(row["source_field"])],
            type_order[str(row["contact_type"])],
            str(row["normalized_value"]).encode("utf-8"),
        )
    )
    if validated != expected:
        raise common.ContractError(
            "Parser artifact is not an exact replay of frozen Step3"
        )
    return world_uid, seller_index, item_index, validated


def parse_observed_world(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the production Step3 parser without reading any private structure."""

    world_uid, seller_index, item_index = _validated_observed_indexes(
        policy, sellers=sellers, items=items
    )
    rows = _extract_step3_rows(
        policy=policy,
        mode=mode,
        split=split,
        world_uid=world_uid,
        seller_index=seller_index,
        item_index=item_index,
    )

    _validated_world_uid, _sellers, _items, rows = validate_parser_artifact(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=items,
        parsed_rows=rows,
    )
    return rows


def _actual_parser_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    token = history.token_key(dict(row))
    if token is None:
        raise common.ContractError("Parsed occurrence lacks a history token")
    return (
        str(row["item_uid"]),
        str(row["source_field"]),
        token[0],
        token[1],
        *(int(row[name]) for name in FLAG_NAMES),
    )


def _planned_parser_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["item_uid"]),
        str(row["field_name"]),
        str(row["identity_type"]).casefold(),
        str(row["downstream_canonical_value"]).strip().casefold(),
        *(int(row[f"expected_{name}"]) for name in FLAG_NAMES),
    )


def _validate_private_slot_universe(
    policy: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Bind private slot tables to the generator AST without vacuous passes."""

    item_index = {str(row["item_uid"]): row for row in items}
    if not items or len(item_index) != len(items):
        raise common.ContractError("Private slot audit item universe is empty or duplicated")
    ast_schema = _schema(policy, "render_asts.jsonl_private")
    ast_index: dict[str, dict[str, Any]] = {}
    ast_identity_uids: set[str] = set()
    ast_noise_uids: set[str] = set()
    noise_sellers: Counter[str] = Counter()
    for source_ast in render_asts:
        _require_exact_keys(source_ast, ast_schema, label="render AST")
        ast = dict(source_ast)
        item_uid = ast["item_uid"]
        item = item_index.get(item_uid)
        if (
            not isinstance(item_uid, str)
            or item is None
            or item_uid in ast_index
            or ast["world_uid"] != item["world_uid"]
            or ast["seller_uid"] != item["seller_uid"]
            or type(ast["time_bucket"]) is not int
            or ast["time_bucket"] != item["time_bucket"]
            or ast["category"] != item["category"]
            or type(ast["title_skeleton_index"]) is not int
            or type(ast["description_skeleton_index"]) is not int
            or type(ast["title_nonempty"]) is not bool
            or type(ast["description_nonempty"]) is not bool
            or not isinstance(ast["identity_slot_uids"], list)
            or not isinstance(ast["noise_slot_uid"], str)
        ):
            raise common.ContractError("Render AST lineage or value-domain drift")
        identity_uids = ast["identity_slot_uids"]
        if (
            any(not isinstance(value, str) or not value for value in identity_uids)
            or len(identity_uids) != len(set(identity_uids))
            or identity_uids != common.utf8_sort(identity_uids)
            or ast_identity_uids.intersection(identity_uids)
        ):
            raise common.ContractError("Render AST identity-slot UID drift")
        ast_identity_uids.update(identity_uids)
        noise_uid = ast["noise_slot_uid"]
        if noise_uid:
            if noise_uid in ast_noise_uids:
                raise common.ContractError("Render AST noise-slot UID collision")
            ast_noise_uids.add(noise_uid)
            noise_sellers[str(ast["seller_uid"])] += 1
        ast_index[item_uid] = ast
    if set(ast_index) != set(item_index):
        raise common.ContractError("Render AST item keyset is not complete")

    identity_schema = [
        str(value)
        for value in policy["relational_integrity"][
            "renderer_identity_slots_audit_schema"
        ]
    ]
    identity_by_uid: dict[str, dict[str, Any]] = {}
    for source_row in identity_slots_audit:
        _require_exact_keys(source_row, identity_schema, label="identity slot audit")
        row = dict(source_row)
        slot_uid = row["slot_uid"]
        item = item_index.get(str(row["item_uid"]))
        start, end = row["start"], row["end"]
        if (
            not isinstance(slot_uid, str)
            or not slot_uid
            or slot_uid in identity_by_uid
            or item is None
            or row["seller_uid"] != item["seller_uid"]
            or row["field_name"] != "description"
            or row["parser_expectation"] != "must_extract"
            or type(start) is not int
            or type(end) is not int
            or type(row["time_bucket"]) is not int
            or row["time_bucket"] != item["time_bucket"]
            or not 0 <= start < end <= len(str(item["description"]))
            or str(item["description"])[start:end] != row["raw_surface"]
            or slot_uid not in ast_index[str(row["item_uid"])][
                "identity_slot_uids"
            ]
        ):
            raise common.ContractError("Identity slot/AST lineage drift")
        identity_by_uid[slot_uid] = row
    if set(identity_by_uid) != ast_identity_uids:
        raise common.ContractError("Identity slot table is not the exact AST universe")

    noise_schema = [
        str(value)
        for value in policy["relational_integrity"][
            "renderer_noise_slots_audit_schema"
        ]
    ]
    noise_by_uid: dict[str, dict[str, Any]] = {}
    for source_row in noise_slots_audit:
        _require_exact_keys(source_row, noise_schema, label="noise slot audit")
        row = dict(source_row)
        noise_uid = row["noise_slot_uid"]
        item = item_index.get(str(row["item_uid"]))
        start, end = row["start"], row["end"]
        if (
            not isinstance(noise_uid, str)
            or not noise_uid
            or noise_uid in noise_by_uid
            or item is None
            or row["seller_uid"] != item["seller_uid"]
            or row["field_name"] != "description"
            or row["parser_expectation"] != "must_ignore"
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(str(item["description"]))
            or str(item["description"])[start:end] != row["raw_surface"]
            or ast_index[str(row["item_uid"])]["noise_slot_uid"] != noise_uid
        ):
            raise common.ContractError("Noise slot/AST lineage drift")
        noise_by_uid[noise_uid] = row
    if set(noise_by_uid) != ast_noise_uids:
        raise common.ContractError("Noise slot table is not the exact AST universe")
    expected_noise_count = int(
        policy["nonidentity_item_dgp"]["must_ignore_noise"]["count_per_seller"]
    )
    seller_uids = {str(row["seller_uid"]) for row in items}
    if (
        len(noise_by_uid) != expected_noise_count * len(seller_uids)
        or set(noise_sellers) != seller_uids
        or any(value != expected_noise_count for value in noise_sellers.values())
    ):
        raise common.ContractError("Noise slot per-seller cardinality drift")
    return {
        "identity_slot_count": len(identity_by_uid),
        "noise_slot_count": len(noise_by_uid),
        "item_count": len(item_index),
    }


def validate_parser_against_private_plan(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare parser output to the independent private plan, exact-set only."""

    _world_uid, _seller_index, _item_index, validated_parser_rows = (
        validate_parser_artifact(
            policy,
            mode=mode,
            split=split,
            sellers=sellers,
            items=items,
            parsed_rows=parsed_rows,
        )
    )
    slot_counts = _validate_private_slot_universe(
        policy,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
    )
    planned = Counter(_planned_parser_tuple(row) for row in identity_slots_audit)
    actual = Counter(_actual_parser_tuple(row) for row in validated_parser_rows)
    if planned != actual:
        raise common.ContractError(
            "Production parser differs from the independent plan: "
            f"missing={list((planned - actual).elements())[:3]} "
            f"extra={list((actual - planned).elements())[:3]}"
        )
    if any(str(row["source_field"]) == "title" for row in validated_parser_rows):
        raise common.ContractError("Synthetic title produced an identity parser row")
    if any(count != 1 for count in planned.values()) or any(
        count != 1 for count in actual.values()
    ):
        raise common.ContractError("Parser plan or output contains duplicate match keys")
    return {
        "planned_must_extract_count": len(identity_slots_audit),
        "actual_parser_row_count": len(validated_parser_rows),
        "must_extract_recall": 1.0,
        "unexpected_parser_row_count": 0,
        "must_ignore_false_positive_count": 0,
        "exact_rows_and_flags": True,
        "ast_identity_slot_count": slot_counts["identity_slot_count"],
        "ast_noise_slot_count": slot_counts["noise_slot_count"],
    }


def _literals_by_seller(
    profiles: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    sellers = {str(row["seller_uid"]) for row in profiles}
    literals: dict[str, set[str]] = {seller_uid: set() for seller_uid in sellers}
    for row in parsed_rows:
        seller_uid = str(row["seller_uid"])
        contact_type = str(row["contact_type"]).casefold()
        if seller_uid not in literals:
            raise common.ContractError("Parser row seller is outside the profile world")
        for field in ("raw_value", "normalized_value"):
            value = source.safe_signal_literal(contact_type, str(row[field]))
            if value is not None:
                literals[seller_uid].add(value)
    return {
        seller_uid: sorted(
            literals[seller_uid],
            key=lambda value: (-len(value), value.casefold(), value),
        )
        for seller_uid in common.utf8_sort(literals)
    }


def _protected_collision_compacts(policy: Mapping[str, Any]) -> set[str]:
    spec = policy["frozen_inputs"]["step7_v3_1_source_policy"]
    path = common.verify_file_pin(spec, label="Step7-v3.1 source policy")
    parent_policy = common.load_json(path)
    quality = parent_policy["clean_text_contract"]["quality_gates"]
    return {
        source.compact_identifier(term)
        for term in (
            *quality["protected_content_words"],
            *quality["protected_identity_collision_terms"],
        )
    }


def _serializable_registry(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _serializable_registry(value[key])
            for key in common.utf8_sort(str(item) for item in value)
        }
    if isinstance(value, (set, frozenset)):
        return common.utf8_sort(str(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serializable_registry(item) for item in value]
    return value


def build_redaction_registry(
    policy: Mapping[str, Any],
    *,
    registry_profiles: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Build the exact per-world v4 registry from observed parser output only."""

    for profile in registry_profiles:
        _require_exact_keys(
            profile,
            ("seller_uid", "source_seller_raw", "alias_normalized"),
            label="redaction registry profile projection",
        )
    parser_schema = _schema(
        policy, "parsed_identity_occurrences.structural_audit_private.csv"
    )
    for row in parsed_rows:
        _require_exact_keys(row, parser_schema, label="redactor parser input")
    seller_uids = {str(row["seller_uid"]) for row in registry_profiles}
    if len(registry_profiles) != 28 or len(seller_uids) != 28:
        raise common.ContractError("Redaction registry requires 28 unique profiles")
    literals = _literals_by_seller(registry_profiles, parsed_rows)
    global_tokens = source.global_identity_tokens(literals, registry_profiles)
    contextual_aliases = source.contextual_global_alias_tokens(
        registry_profiles, literals
    )
    seller_literals = {
        str(profile["seller_uid"]): source.seller_identity_literals(dict(profile))
        for profile in registry_profiles
    }
    seller_phrases = {
        str(profile["seller_uid"]): source.seller_identity_phrase_tokens(
            dict(profile)
        )
        for profile in registry_profiles
    }

    collision_compacts = _protected_collision_compacts(policy)
    global_contextual_collisions = contextual_aliases & collision_compacts
    local_collisions = {
        seller_uid: {
            source.compact_identifier(value)
            for value in seller_literals[seller_uid]
            if source.compact_identifier(value) in collision_compacts
        }
        | (seller_phrases[seller_uid] & collision_compacts)
        for seller_uid in seller_uids
    }
    observed_local_collisions = {
        seller_uid: values
        for seller_uid, values in local_collisions.items()
        if values
    }
    if global_contextual_collisions or observed_local_collisions:
        raise common.ContractError(
            "Synthetic redaction registry collides with protected content; "
            "the allowed collision set is empty"
        )
    contextual_deletions = redactor.v4_contextual_alias_deletion_tokens(
        contextual_aliases
    )
    registry: dict[str, Any] = {
        "global_identity_tokens": set(global_tokens),
        "contextual_global_alias_tokens": set(contextual_aliases),
        "contextual_alias_deletion_tokens": set(contextual_deletions),
        "seller_identity_literals": {
            seller_uid: list(seller_literals[seller_uid])
            for seller_uid in common.utf8_sort(seller_uids)
        },
        "seller_identity_phrase_tokens": {
            seller_uid: set(seller_phrases[seller_uid])
            for seller_uid in common.utf8_sort(seller_uids)
        },
        "seller_contextual_collision_tokens": {
            seller_uid: set() for seller_uid in common.utf8_sort(seller_uids)
        },
        "audited_global_identity_phrase_tokens": set(
            source.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
        ),
    }
    if tuple(registry) != REGISTRY_NAMES:
        raise AssertionError("Redaction registry name order drift")
    hashes = {
        name: common.canonical_sha256(_serializable_registry(registry[name]))
        for name in REGISTRY_NAMES
    }
    collision_audit = {
        "allowed_collision_count": 0,
        "protected_collision_compact_count": len(collision_compacts),
        "protected_collision_compacts_sha256": common.canonical_sha256(
            common.utf8_sort(collision_compacts)
        ),
        "registry_protected_global_contextual_collision_count": 0,
        "registry_protected_seller_local_collision_count": 0,
        "labels_or_private_plan_read": False,
    }
    return registry, hashes, collision_audit


def _actual_identity_allowlist_by_item(
    parsed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    output: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"compact": set(), "global_token": set()}
    )
    for row in parsed_rows:
        item_uid = str(row["item_uid"])
        for field in ("raw_value", "normalized_value"):
            value = str(row[field])
            compact = source.compact_identifier(value)
            if compact:
                output[item_uid]["compact"].add(compact)
            for match in source.IDENTIFIER_TOKEN_RE.finditer(value):
                token = source.canonical_identifier_token(match.group(0))
                if token:
                    output[item_uid]["global_token"].add(token)
    return output


def _pre_redaction_registry_collision_scan(
    *,
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    protected_compacts: set[str],
) -> dict[str, Any]:
    """Reject registry matches in observed text that no actual parser row owns."""

    allowed = _actual_identity_allowlist_by_item(parsed_rows)
    protected_occurrences = 0
    scanned_fields = 0
    for item in items:
        item_uid = str(item["item_uid"])
        seller_uid = str(item["seller_uid"])
        for field_name in ("title", "description"):
            text = str(item[field_name])
            if not text:
                continue
            scanned_fields += 1
            protected_occurrences += len(
                source.unconditional_alias_spans(text, protected_compacts)
            )
            if any(
                pattern.search(text)
                for pattern in (
                    source.identity_literal_pattern(value)
                    for value in registry["seller_identity_literals"][seller_uid]
                )
            ):
                raise common.ContractError(
                    "Seller-local alias collides with observed synthetic content"
                )
            if source.unconditional_alias_spans(
                text, registry["seller_identity_phrase_tokens"][seller_uid]
            ):
                raise common.ContractError(
                    "Seller-local phrase collides with observed synthetic content"
                )
            if source.unconditional_alias_spans(
                text, registry["audited_global_identity_phrase_tokens"]
            ):
                raise common.ContractError(
                    "Audited global phrase collides with synthetic content"
                )
            for match in source.IDENTIFIER_TOKEN_RE.finditer(text):
                if not source.matches_global_identity_token(
                    match.group(0), registry["global_identity_tokens"]
                ):
                    continue
                token = source.canonical_identifier_token(match.group(0))
                if token not in allowed[item_uid]["global_token"]:
                    raise common.ContractError(
                        "Global identity token collides with non-parser content"
                    )
            contextual = source.contextual_alias_spans(
                text,
                registry["contextual_global_alias_tokens"],
                registry["contextual_alias_deletion_tokens"],
            )
            for match in contextual:
                if (
                    match["match_kind"] != "exact"
                    or str(match["compact_alias"])
                    not in allowed[item_uid]["compact"]
                ):
                    raise common.ContractError(
                        "Contextual alias collides with non-parser content"
                    )
    return {
        "observed_item_count": len(items),
        "observed_nonempty_field_count": scanned_fields,
        "protected_content_occurrence_count_before_redaction": (
            protected_occurrences
        ),
        "unexpected_seller_local_literal_match_count": 0,
        "unexpected_seller_local_phrase_match_count": 0,
        "unexpected_audited_global_phrase_match_count": 0,
        "unexpected_global_token_match_count": 0,
        "unexpected_contextual_alias_match_count": 0,
        "one_character_omission_match_count": 0,
    }


def _redact_field(
    value: str,
    *,
    seller_uid: str,
    registry: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    return redactor.redact_raw_field(
        value,
        seller_uid=seller_uid,
        seller_literals=registry["seller_identity_literals"][seller_uid],
        seller_phrase_tokens=registry["seller_identity_phrase_tokens"][seller_uid],
        global_tokens=registry["global_identity_tokens"],
        contextual_aliases=registry["contextual_global_alias_tokens"],
        contextual_alias_deletions=registry[
            "contextual_alias_deletion_tokens"
        ],
        seller_contextual_collision_tokens=registry[
            "seller_contextual_collision_tokens"
        ][seller_uid],
        audited_global_phrases=registry[
            "audited_global_identity_phrase_tokens"
        ],
    )


def _context_guard_boundary(
    value: str,
    *,
    guards: Sequence[str],
) -> tuple[int | None, int]:
    """Locate the earliest registered identity boundary and count all guards."""

    text = str(value)
    if not guards or len(guards) != len(set(guards)):
        raise common.ContractError("Synthetic context guard pool is invalid")
    positions = [
        position
        for guard in guards
        if (position := text.find(guard)) >= 0
    ]
    count = sum(text.count(guard) for guard in guards)
    return (min(positions) if positions else None), count


def _canonicalize_synthetic_description(
    value: str,
    *,
    guards: Sequence[str],
) -> tuple[str, dict[str, int]]:
    """Remove the complete synthetic identity region at its public boundary."""

    text = str(value)
    if not text:
        return "", {
            "precanonical_context_guard_count": 0,
            "synthetic_identity_suffix_character_count_removed": 0,
        }
    boundary, guard_count = _context_guard_boundary(text, guards=guards)
    prefix = text if boundary is None else text[:boundary]
    clean = source.normalize_redacted_text(prefix)
    return clean, {
        "precanonical_context_guard_count": guard_count,
        "synthetic_identity_suffix_character_count_removed": (
            0 if boundary is None else len(text) - len(prefix)
        ),
    }


def redact_observed_world(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    registry_profiles: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Redact a world using only observed, label-free inputs."""

    expected_template = common.load_json(
        common.repo_path(str(policy["template_library"]["path"]))
    )
    if dict(template) != expected_template:
        raise common.ContractError("Redactor received an unregistered template")
    world_uid, seller_index, item_index, validated_parser_rows = (
        validate_parser_artifact(
            policy,
            mode=mode,
            split=split,
            sellers=sellers,
            items=items,
            parsed_rows=parsed_rows,
        )
    )
    profiles = [dict(row) for row in registry_profiles]
    if (
        len(profiles) != 28
        or {str(row.get("seller_uid", "")) for row in profiles}
        != set(seller_index)
    ):
        raise common.ContractError("Redactor registry-profile keyset drift")
    for profile in profiles:
        _require_exact_keys(
            profile,
            ("seller_uid", "source_seller_raw", "alias_normalized"),
            label="redactor registry profile",
        )
        seller_uid = str(profile["seller_uid"])
        if (
            str(profile["source_seller_raw"]) != seller_uid
            or str(profile["alias_normalized"]) != seller_uid
        ):
            raise common.ContractError("Redactor registry aliases are not opaque UIDs")
    registry, registry_hashes, collision_audit = build_redaction_registry(
        policy,
        registry_profiles=profiles,
        parsed_rows=validated_parser_rows,
    )
    protected_compacts = _protected_collision_compacts(policy)
    observed_collision_audit = _pre_redaction_registry_collision_scan(
        items=items,
        parsed_rows=validated_parser_rows,
        registry=registry,
        protected_compacts=protected_compacts,
    )
    output: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    guards = renderer.context_guard_pool(template)
    parsed_item_uids = {
        str(row["item_uid"]) for row in validated_parser_rows
    }
    ordered_items = sorted(
        item_index.values(),
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        ),
    )
    for item in ordered_items:
        item_uid = str(item["item_uid"])
        seller_uid = str(item["seller_uid"])
        raw_description = str(item["description"])
        boundary, _guard_count = _context_guard_boundary(
            raw_description,
            guards=guards,
        )
        has_identity = item_uid in parsed_item_uids
        if has_identity != (boundary is not None):
            raise common.ContractError(
                "Observed identity rows and public suffix boundary disagree"
            )
        if raw_description:
            expected_public_prefix = source.normalize_redacted_text(
                raw_description
                if boundary is None
                else raw_description[:boundary]
            )
        else:
            expected_public_prefix = ""
        clean_title, title_audit = _redact_field(
            str(item["title"]), seller_uid=seller_uid, registry=registry
        )
        v4_description, description_audit = _redact_field(
            raw_description, seller_uid=seller_uid, registry=registry
        )
        clean_description, canonicalization_audit = (
            _canonicalize_synthetic_description(
                v4_description,
                guards=guards,
            )
        )
        description_audit = {
            **description_audit,
            **canonicalization_audit,
        }
        if clean_title != source.normalize_redacted_text(str(item["title"])):
            raise common.ContractError(
                "Identity-free title changed under production redaction"
            )
        if clean_description != expected_public_prefix:
            raise common.ContractError(
                "Production redaction changed public synthetic description content"
            )
        output.append(
            {
                "world_uid": world_uid,
                "seller_uid": seller_uid,
                "item_uid": item_uid,
                "title": clean_title,
                "description": clean_description,
            }
        )
        diagnostics.extend(
            (
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "item_uid": item_uid,
                    "field_name": "title",
                    **title_audit,
                },
                {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "item_uid": item_uid,
                    "field_name": "description",
                    **description_audit,
                },
            )
        )

    redacted_schema = _schema(policy, "redacted_items.jsonl")
    for row in output:
        _require_exact_keys(row, redacted_schema, label="redacted item")
    minimum_titles = int(
        policy["world_design"][
            "post_v4_redaction_minimum_nonempty_title_per_seller"
        ]
    )
    minimum_descriptions = int(
        policy["world_design"][
            "post_v4_redaction_minimum_nonempty_description_per_seller"
        ]
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output:
        counts[str(row["seller_uid"])]["title"] += bool(str(row["title"]))
        counts[str(row["seller_uid"])]["description"] += bool(
            str(row["description"])
        )
    if any(
        counts[seller_uid]["title"] < minimum_titles
        or counts[seller_uid]["description"] < minimum_descriptions
        for seller_uid in seller_index
    ):
        raise common.ContractError("Post-redaction nonempty seller minimum failed")
    protected_after = sum(
        len(
            source.unconditional_alias_spans(
                str(row[field_name]), protected_compacts
            )
        )
        for row in output
        for field_name in ("title", "description")
        if str(row[field_name])
    )
    if (
        protected_after
        != observed_collision_audit[
            "protected_content_occurrence_count_before_redaction"
        ]
    ):
        raise common.ContractError(
            "Protected synthetic content changed under production redaction"
        )
    observed_collision_audit[
        "protected_content_occurrence_count_after_redaction"
    ] = protected_after
    observed_collision_audit["protected_content_exact_retention"] = True
    collision_audit = {**collision_audit, **observed_collision_audit}
    return {
        "redacted_items": output,
        "registry_hashes": registry_hashes,
        "registry_collision_audit": collision_audit,
        "redaction_diagnostics": diagnostics,
    }


def project_history_safe_occurrences(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create the exact 13-column M1/M2 parser projection."""

    world_uid, _seller_index, item_index, validated_parser_rows = (
        validate_parser_artifact(
            policy,
            mode=mode,
            split=split,
            sellers=sellers,
            items=items,
            parsed_rows=parsed_rows,
        )
    )
    fieldnames = _schema(policy, "history_safe_occurrences.csv")
    rows: list[dict[str, Any]] = []
    for parsed in validated_parser_rows:
        item = item_index.get(str(parsed["item_uid"]))
        token = history.token_key(dict(parsed))
        if (
            item is None
            or token is None
            or str(parsed["world_uid"]) != world_uid
        ):
            raise common.ContractError("History-safe projection lineage drift")
        row = {
            "world_uid": world_uid,
            "seller_uid": str(parsed["seller_uid"]),
            "item_uid": str(parsed["item_uid"]),
            "source_dataset": str(parsed["source_dataset"]),
            "source_row_number": str(parsed["source_row_number"]),
            "source_field": str(parsed["source_field"]),
            "contact_type": token[0],
            "normalized_value": token[1],
            "seller_facing_context": int(parsed["seller_facing_context"]),
            "product_data_risk_context": int(
                parsed["product_data_risk_context"]
            ),
            "direct_identity_eligible": int(
                parsed["direct_identity_eligible"]
            ),
            "support_only": int(parsed["support_only"]),
            "time_bucket": int(item["time_bucket"]),
        }
        _require_exact_keys(row, fieldnames, label="history-safe occurrence")
        rows.append(row)
    type_order = {
        value: index
        for index, value in enumerate(policy["identity_design"]["identity_types"])
    }
    rows.sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
            SOURCE_FIELD_ORDER[str(row["source_field"])],
            type_order[str(row["contact_type"])],
            str(row["normalized_value"]).encode("utf-8"),
        )
    )
    if len(
        {
            (
                row["world_uid"],
                row["item_uid"],
                row["source_field"],
                row["contact_type"],
                row["normalized_value"],
            )
            for row in rows
        }
    ) != len(rows):
        raise common.ContractError("History-safe occurrence key collision")
    return rows


def build_history_projection_attestation(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    world_uid: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    history_safe_occurrences: Sequence[Mapping[str, Any]],
    history_item_index: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute and bind a safe projection to its exact parser parent."""

    if mode not in {"development_smoke", "training_ready"}:
        raise common.ContractError(
            "Formal history projection sealing is not released"
        )
    parser_audit = validate_parser_against_private_plan(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=items,
        parsed_rows=parsed_rows,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
    )
    expected_safe = project_history_safe_occurrences(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=items,
        parsed_rows=parsed_rows,
    )
    expected_item_index = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": int(row["time_bucket"]),
        }
        for row in items
    ]
    expected_item_index.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
        )
    )
    if (
        common.canonical_json_bytes(list(history_safe_occurrences))
        != common.canonical_json_bytes(expected_safe)
        or common.canonical_json_bytes(list(history_item_index))
        != common.canonical_json_bytes(expected_item_index)
    ):
        raise common.ContractError(
            "History projection bytes are not the exact trusted projection"
        )
    if (
        parser_audit.get("exact_rows_and_flags") is not True
        or int(parser_audit.get("actual_parser_row_count", -1))
        != len(parsed_rows)
        or int(parser_audit.get("planned_must_extract_count", -1))
        != len(parsed_rows)
        or any(str(row.get("world_uid", "")) != world_uid for row in parsed_rows)
        or any(
            str(row.get("world_uid", "")) != world_uid
            for row in history_safe_occurrences
        )
        or any(
            str(row.get("world_uid", "")) != world_uid
            for row in history_item_index
        )
    ):
        raise common.ContractError("History projection parent audit is incomplete")
    producer_path = common.ROOT / "scripts" / "step28_v13_production_chain.py"
    payload: dict[str, Any] = {
        "version": "2026-07-27-step28-v13-history-projection-attestation-v1",
        "mode": mode,
        "split": split,
        "world_uid": world_uid,
        "history_safe_occurrence_count": len(history_safe_occurrences),
        "history_safe_occurrences_sha256": common.canonical_rows_sha256(
            history_safe_occurrences,
            order_fields=(
                "world_uid",
                "seller_uid",
                "item_uid",
                "source_field",
                "contact_type",
                "normalized_value",
            ),
        ),
        "history_item_index_count": len(history_item_index),
        "history_item_index_sha256": common.canonical_rows_sha256(
            history_item_index,
            order_fields=("world_uid", "seller_uid", "item_uid"),
        ),
        "parser_artifact_row_count": len(parsed_rows),
        "parser_artifact_sha256": common.canonical_rows_sha256(
            parsed_rows,
            order_fields=(
                "world_uid",
                "seller_uid",
                "item_uid",
                "source_field",
                "contact_type",
                "normalized_value",
            ),
        ),
        "parser_exact_replay": True,
        "private_plan_exact": True,
        "projection_producer_path": "scripts/step28_v13_production_chain.py",
        "projection_producer_sha256": common.sha256_file(producer_path),
        "step3_parser_code_sha256": str(
            policy["frozen_inputs"]["step3_parser_profile_code"]["sha256"]
        ),
        "custody_status": (
            "DEVELOPMENT_SMOKE_IN_PROCESS_NOT_FORMAL_SEAL"
            if mode == "development_smoke"
            else (
                "TRAINING_READY_SPLIT_PRIVATE_MATHEMATICAL_REPLAY_"
                "NOT_OS_CUSTODY_ATTESTED"
            )
        ),
        "custody_parent_seal_sha256": None,
    }
    payload["canonical_self_hash"] = common.canonical_sha256(payload)
    return payload


def _base_description(
    *,
    ast: Mapping[str, Any],
    split: str,
    template: Mapping[str, Any],
    styles: Mapping[str, Mapping[str, Any]],
) -> str:
    if not bool(ast["description_nonempty"]):
        return ""
    try:
        style = styles[str(ast["effective_style_uid"])]
        skeletons = template["split_libraries"][split][
            "description_skeletons"
        ]
    except (KeyError, IndexError) as exc:
        raise common.ContractError("Render AST cannot reconstruct its base") from exc
    index = ast["description_skeleton_index"]
    if type(index) is not int or not 0 <= index < len(skeletons):
        raise common.ContractError("Description skeleton index is outside its domain")
    skeleton = skeletons[index]
    return renderer.render_base_description(
        skeleton=skeleton,
        product=str(ast["product"]),
        attribute=str(ast["attribute"]),
        code=str(ast["code"]),
        delivery=str(ast["delivery"]),
        service=str(ast["service"]),
        style=style,
        template=template,
    )


def _base_title(
    *,
    ast: Mapping[str, Any],
    split: str,
    template: Mapping[str, Any],
    styles: Mapping[str, Mapping[str, Any]],
) -> str:
    if not bool(ast["title_nonempty"]):
        return ""
    try:
        style = styles[str(ast["effective_style_uid"])]
        skeletons = template["split_libraries"][split]["title_skeletons"]
    except KeyError as exc:
        raise common.ContractError("Render AST cannot reconstruct its title") from exc
    index = ast["title_skeleton_index"]
    if type(index) is not int or not 0 <= index < len(skeletons):
        raise common.ContractError("Title skeleton index is outside its domain")
    return renderer.render_base_title(
        skeleton=skeletons[index],
        product=str(ast["product"]),
        attribute=str(ast["attribute"]),
        code=str(ast["code"]),
        style=style,
        template=template,
    )


def _validate_observed_raw_against_private_ast(
    policy: Mapping[str, Any],
    *,
    split: str,
    template: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Reconstruct every raw item byte from the independent private AST."""

    item_index = {str(row["item_uid"]): row for row in items}
    ast_index = {str(row["item_uid"]): row for row in render_asts}
    styles = {
        str(row["effective_style_uid"]): row
        for row in renderer.reachable_effective_styles(template)
    }
    lexicon = template["generic_lexicon"]
    attributes = set(lexicon["attributes"])
    deliveries = set(lexicon["delivery"])
    services = set(lexicon["service"])
    category_products = lexicon["category_products"]
    for item_uid, ast in ast_index.items():
        if (
            ast["category"] not in category_products
            or ast["product"] not in category_products[ast["category"]]
            or ast["attribute"] not in attributes
            or ast["delivery"] not in deliveries
            or ast["service"] not in services
            or not isinstance(ast["code"], str)
            or re.fullmatch(r"Q[A-P]{10}", ast["code"]) is None
            or ast["effective_style_uid"] not in styles
        ):
            raise common.ContractError("Render AST text domain drift")
        item = item_index[item_uid]
        if bool(ast["title_nonempty"]) != bool(str(item["title"])):
            raise common.ContractError("Render AST title mask drift")
        if bool(ast["description_nonempty"]) != bool(str(item["description"])):
            raise common.ContractError("Render AST description mask drift")

    expected_titles = {
        item_uid: _base_title(
            ast=ast,
            split=split,
            template=template,
            styles=styles,
        )
        for item_uid, ast in ast_index.items()
    }
    override_schema = {
        "override_kind",
        "asset_index",
        "canonical_pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "item_uid_left",
        "item_uid_right",
    }
    override_rows: list[dict[str, Any]] = []
    for source_row in override_audit:
        _require_exact_keys(
            source_row,
            tuple(override_schema),
            label="registered override audit",
        )
        row = dict(source_row)
        left = item_index.get(str(row["item_uid_left"]))
        right = item_index.get(str(row["item_uid_right"]))
        if (
            row["override_kind"]
            not in {"high_semantic_similarity", "exact_title_clone"}
            or type(row["asset_index"]) is not int
            or left is None
            or right is None
            or row["seller_uid_left"] != left["seller_uid"]
            or row["seller_uid_right"] != right["seller_uid"]
            or row["seller_uid_left"] == row["seller_uid_right"]
            or row["canonical_pair_uid"]
            != common.canonical_pair_uid(
                str(row["seller_uid_left"]),
                str(row["seller_uid_right"]),
            )
        ):
            raise common.ContractError("Registered override lineage drift")
        override_rows.append(row)
    kind_counts = Counter(row["override_kind"] for row in override_rows)
    endpoint_sellers = {
        str(row[name])
        for row in override_rows
        for name in ("seller_uid_left", "seller_uid_right")
    }
    endpoint_items = {
        str(row[name])
        for row in override_rows
        for name in ("item_uid_left", "item_uid_right")
    }
    if (
        kind_counts
        != Counter(
            {"high_semantic_similarity": 4, "exact_title_clone": 2}
        )
        or len(endpoint_sellers) != 12
        or len(endpoint_items) != 12
    ):
        raise common.ContractError("Registered override cardinality drift")
    override_order = lambda value: (
        str(value["override_kind"]).encode("utf-8"),
        int(value["asset_index"]),
        str(value["canonical_pair_uid"]).encode("utf-8"),
    )
    for row in sorted(
        override_rows,
        key=override_order,
    ):
        left_uid = str(row["item_uid_left"])
        right_uid = str(row["item_uid_right"])
        left_ast = ast_index[left_uid]
        right_ast = ast_index[right_uid]
        if row["override_kind"] == "high_semantic_similarity":
            if (
                left_ast["category"] != right_ast["category"]
                or left_ast["product"] != right_ast["product"]
                or left_ast["attribute"] != right_ast["attribute"]
                or left_ast["title_skeleton_index"]
                == right_ast["title_skeleton_index"]
                or left_ast["code"] == right_ast["code"]
                or expected_titles[left_uid] == expected_titles[right_uid]
            ):
                raise common.ContractError("High-semantic override AST drift")
        else:
            if not expected_titles[left_uid]:
                raise common.ContractError("Exact-title source is empty")
            expected_titles[right_uid] = expected_titles[left_uid]

    identity_by_uid = {
        str(row["slot_uid"]): row for row in identity_slots_audit
    }
    noise_by_uid = {
        str(row["noise_slot_uid"]): row for row in noise_slots_audit
    }
    role_to_family = policy["identity_design"]["role_to_template_family"]
    public_prefixes: dict[str, str] = {}
    guards = renderer.context_guard_pool(template)
    for item_uid in common.utf8_sort(item_index):
        item = item_index[item_uid]
        ast = ast_index[item_uid]
        base = _base_description(
            ast=ast,
            split=split,
            template=template,
            styles=styles,
        )
        noise_uid = str(ast["noise_slot_uid"])
        noise = (
            str(noise_by_uid[noise_uid]["raw_surface"]) if noise_uid else ""
        )
        clauses = []
        for slot_uid in ast["identity_slot_uids"]:
            slot = identity_by_uid[str(slot_uid)]
            family = str(role_to_family[str(slot["planned_role"])])
            clauses.append(
                renderer.identity_clause(
                    template_family=family,
                    identity_type=str(slot["identity_type"]),
                    normalized_value=str(slot["raw_surface"]),
                    template=template,
                )
            )
        expected_description = renderer.render_description(
            base_description=base,
            noise_clause=noise,
            identity_clauses=clauses,
            selector_uid=item_uid,
            template=template,
        )
        if str(item["title"]) != expected_titles[item_uid]:
            raise common.ContractError(
                "Observed raw title is not the registered AST realization"
            )
        if str(item["description"]) != expected_description:
            raise common.ContractError(
                "Observed raw description is not the registered AST realization"
            )
        public_prefixes[item_uid] = source.normalize_redacted_text(base + noise)
        if expected_description:
            boundary, _guard_count = _context_guard_boundary(
                expected_description,
                guards=guards,
            )
            if bool(clauses) != (boundary is not None):
                raise common.ContractError(
                    "AST identity clauses and context boundary disagree"
                )
            raw_prefix = source.normalize_redacted_text(
                expected_description
                if boundary is None
                else expected_description[:boundary]
            )
            if raw_prefix != public_prefixes[item_uid]:
                raise common.ContractError("AST public description prefix drift")
        elif public_prefixes[item_uid]:
            raise common.ContractError("Empty description has a public prefix")
    return public_prefixes


def validate_redaction_against_private_plan(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit redaction after the observed-only worker has already completed."""

    _world_uid, _seller_index, _item_index, validated_parser_rows = (
        validate_parser_artifact(
            policy,
            mode=mode,
            split=split,
            sellers=sellers,
            items=items,
            parsed_rows=parsed_rows,
        )
    )
    _validate_private_slot_universe(
        policy,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
    )
    expected_public_prefixes = _validate_observed_raw_against_private_ast(
        policy,
        split=split,
        template=template,
        items=items,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
        override_audit=override_audit,
    )
    item_schema = _schema(policy, "items.jsonl")
    redacted_schema = _schema(policy, "redacted_items.jsonl")
    expected_template = common.load_json(
        common.repo_path(str(policy["template_library"]["path"]))
    )
    if dict(template) != expected_template:
        raise common.ContractError("Redaction audit received an unregistered template")
    categories = set(expected_template["generic_lexicon"]["categories"])
    for row in items:
        _require_exact_keys(row, item_schema, label="redaction audit raw item")
        _validate_item_value_domains(policy, row, categories=categories)
    for row in redacted_items:
        _require_exact_keys(
            row, redacted_schema, label="redaction audit clean item"
        )
        for name in ("world_uid", "seller_uid", "item_uid", "title", "description"):
            if not isinstance(row[name], str):
                raise common.ContractError(
                    f"Redaction audit {name} must be a string"
                )
        if (
            unicodedata.normalize("NFC", row["title"]) != row["title"]
            or unicodedata.normalize("NFC", row["description"])
            != row["description"]
        ):
            raise common.ContractError("Redacted text must be NFC")
    _validate_item_seller_minimums(policy, items)
    item_index = {str(row["item_uid"]): row for row in items}
    clean_index = {str(row["item_uid"]): row for row in redacted_items}
    ast_index = {str(row["item_uid"]): row for row in render_asts}
    if not (
        len(item_index)
        == len(clean_index)
        == len(ast_index)
        == len(items)
        == len(redacted_items)
        == len(render_asts)
    ) or set(item_index) != set(clean_index) or set(item_index) != set(ast_index):
        raise common.ContractError("Redaction structural audit item keyset drift")
    ast_schema = _schema(policy, "render_asts.jsonl_private")
    for ast in render_asts:
        _require_exact_keys(ast, ast_schema, label="render AST")
    styles = {
        row["effective_style_uid"]: row
        for row in renderer.reachable_effective_styles(template)
    }
    guards = renderer.context_guard_pool(template)
    for item_uid in common.utf8_sort(item_index):
        raw = item_index[item_uid]
        clean = clean_index[item_uid]
        ast = ast_index[item_uid]
        if (
            str(raw["world_uid"]) != str(clean["world_uid"])
            or str(raw["world_uid"]) != str(ast["world_uid"])
            or str(raw["item_uid"]) != str(clean["item_uid"])
            or str(raw["item_uid"]) != str(ast["item_uid"])
            or str(raw["seller_uid"]) != str(clean["seller_uid"])
            or str(raw["seller_uid"]) != str(ast["seller_uid"])
        ):
            raise common.ContractError("Redaction seller lineage drift")
        if str(clean["title"]) != source.normalize_redacted_text(
            str(raw["title"])
        ):
            raise common.ContractError("Redacted title differs from raw title")
        base = _base_description(
            ast=ast, split=split, template=template, styles=styles
        )
        noise_rows = [
            row
            for row in noise_slots_audit
            if str(row["item_uid"]) == item_uid
        ]
        if len(noise_rows) > 1:
            raise common.ContractError("An item has multiple must-ignore clauses")
        noise = str(noise_rows[0]["raw_surface"]) if noise_rows else ""
        expected_clean_description = source.normalize_redacted_text(base + noise)
        raw_description = str(raw["description"])
        boundary, _guard_count = _context_guard_boundary(
            raw_description,
            guards=guards,
        )
        if bool(ast["identity_slot_uids"]) != (boundary is not None):
            raise common.ContractError(
                "Private identity slots and raw context boundary disagree"
            )
        raw_public_prefix = (
            source.normalize_redacted_text(
                raw_description
                if boundary is None
                else raw_description[:boundary]
            )
            if raw_description
            else ""
        )
        if (
            raw_public_prefix != expected_clean_description
            or expected_public_prefixes[item_uid]
            != expected_clean_description
        ):
            raise common.ContractError(
                "Raw, AST and clean public description prefixes disagree"
            )
        if str(clean["description"]) != expected_clean_description:
            raise common.ContractError(
                "Final M0 description is not exactly base plus must-ignore text"
            )
        if any(guard in str(clean["description"]) for guard in guards):
            raise common.ContractError(
                "Synthetic identity boundary survived final canonicalization"
            )

    parsed_surfaces_by_item: dict[str, list[str]] = defaultdict(list)
    for row in validated_parser_rows:
        parsed_surfaces_by_item[str(row["item_uid"])].extend(
            (str(row["raw_value"]), str(row["normalized_value"]))
        )
    planned_surfaces_by_item: dict[str, list[str]] = defaultdict(list)
    for row in identity_slots_audit:
        planned_surfaces_by_item[str(row["item_uid"])].append(
            str(row["raw_surface"])
        )
    all_surface_items = set(parsed_surfaces_by_item) | set(
        planned_surfaces_by_item
    )
    for item_uid in all_surface_items:
        surfaces = [
            *parsed_surfaces_by_item[item_uid],
            *planned_surfaces_by_item[item_uid],
        ]
        clean_description = str(clean_index[item_uid]["description"]).casefold()
        for surface in surfaces:
            if surface and surface.casefold() in clean_description:
                raise common.ContractError("Identity surface survived redaction")

    for row in noise_slots_audit:
        item_uid = str(row["item_uid"])
        surface = str(row["raw_surface"])
        before = str(item_index[item_uid]["description"])
        after = str(clean_index[item_uid]["description"])
        if before.count(surface) != 1 or after.count(surface) != 1:
            raise common.ContractError(
                "Must-ignore surface was changed, removed, or duplicated"
            )
    return {
        "planned_identity_surface_residue_count": 0,
        "parsed_identity_surface_residue_count": 0,
        "must_ignore_changed_count": 0,
        "base_description_changed_count": 0,
        "context_guard_residue_count": 0,
        "title_changed_count": 0,
        "identity_slot_count_observable_in_m0_text": False,
        "post_redaction_seller_minimums_pass": True,
    }


def process_world(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    world: Mapping[str, Any],
) -> dict[str, Any]:
    """Non-formal parser/redactor orchestration; formal custody is forbidden."""

    if mode not in {"development_smoke", "training_ready"}:
        raise common.ContractError(
            "Combined parser/redactor/private audit orchestration has no "
            "formal-custody implementation"
        )

    public = world["public"]
    private = world["private"]
    parsed = parse_observed_world(
        policy,
        mode=mode,
        split=split,
        sellers=public["sellers"],
        items=public["items"],
    )
    parser_audit = validate_parser_against_private_plan(
        policy,
        mode=mode,
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    registry_profiles = registry_profiles_from_sellers(
        policy,
        sellers=public["sellers"],
    )
    redaction = redact_observed_world(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=public["sellers"],
        items=public["items"],
        registry_profiles=registry_profiles,
        parsed_rows=parsed,
    )
    history_rows = project_history_safe_occurrences(
        policy,
        mode=mode,
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
    )
    redaction_audit = validate_redaction_against_private_plan(
        policy,
        mode=mode,
        split=split,
        template=template,
        sellers=public["sellers"],
        items=public["items"],
        redacted_items=redaction["redacted_items"],
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
        override_audit=private["override_audit"],
    )
    profile_safe_items = build_profile_safe_items(
        policy,
        items=public["items"],
        redacted_items=redaction["redacted_items"],
    )
    return {
        "public": {
            "redacted_items": redaction["redacted_items"],
            "history_safe_occurrences": history_rows,
            "profile_safe_items": profile_safe_items,
        },
        "private": {
            "parsed_identity_occurrences": parsed,
            "parser_structural_audit": parser_audit,
            "redaction_registry_hashes": redaction["registry_hashes"],
            "redaction_registry_collision_audit": redaction[
                "registry_collision_audit"
            ],
            "redaction_diagnostics": redaction["redaction_diagnostics"],
            "redaction_structural_audit": redaction_audit,
        },
    }
