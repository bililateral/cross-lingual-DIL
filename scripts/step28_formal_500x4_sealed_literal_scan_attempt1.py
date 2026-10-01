#!/usr/bin/env python3
"""Sealed aggregate-only literal scan for V9.4.1 formal-root quality attempt 1.

This process may inspect private strings, including audit A/B custody files.  It
returns counts and commitments only; no private literal, label, controller
relation, qrel, or row-level hit is returned.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "v9_4_1_formal_500x4_attempt1_20260829"
)
PRIVATE_ROOT = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_4_1_formal_500x4_attempt1_20260829"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
VERSION = "2026-08-29-step28-v13-v1-13-v9-4-1-formal-sealed-literal-scan-attempt1-v1"
CONTACT_TOKEN_RE = re.compile(
    r"(?:https?://|www\.)[^\s，。；;]+"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|0x[0-9A-Fa-f]{8,}"
    r"|(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_.-]{4,}(?![A-Za-z0-9])"
    r"|(?<!\d)\+?\d(?:[\s().-]*\d){6,}(?!\d)"
)
INTERNAL_TOKEN_RE = re.compile(
    r"(?:w|sel|itm|qry|controller|ctrl|world|seller|pair)_[A-Za-z0-9_-]+",
    flags=re.IGNORECASE,
)
VISIBLE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9_.:@/+\-]{1,}(?![A-Za-z0-9])"
)
INDEXED_MAPPING_FIELD_RE = re.compile(
    r"(?:registry|register|counts?|by_(?:seller|controller|query|world)|"
    r"(?:seller|controller|query|world).*(?:map|index))",
    flags=re.IGNORECASE,
)
SHORT_ASCII_IDENTIFIERS = frozenset({"bat", "qq", "tg", "wx"})
IDENTITY_TYPES = frozenset(
    {
        "telegram",
        "email",
        "bat",
        "qq",
        "wechat",
        "phone",
        "crypto_wallet",
        "external_url",
    }
)
PARSED_FIELDS = (
    "world_uid",
    "item_uid",
    "signal_uid",
    "data_bucket",
    "source_dataset",
    "source_row_number",
    "seller_uid",
    "source_market_raw",
    "source_seller_raw",
    "source_seller_id_raw",
    "alias_normalized",
    "source_field",
    "contact_type",
    "normalized_value",
    "raw_value",
    "evidence_level",
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
    "context",
    "title_snippet",
    "description_snippet",
)
GENERATION_AUDIT_FIELDS = (
    "world_uid",
    "style_assignments",
    "mechanism_assignments",
    "registered_negative_controls",
    "profile_audit",
    "parser_uid_alias_audit",
    "identity33_active_pair_count",
    "public_projection_exact_replay",
    "truth_projected_after_public_rows",
)
IDENTITY_PLAN_FIELDS = (
    "world_uid",
    "asset_uid",
    "identity_type",
    "value_sha256",
    "role",
    "mechanism",
    "seller_occurrences",
)
CONTROL_FIELDS = (
    "canonical_pair_uid",
    "control_type",
    "source_item_uid",
    "target_item_uid",
)
PARSER_ALIAS_AUDIT_FIELDS = (
    "version",
    "world_uid",
    "alias_world_uid_sha256",
    "seller_alias_mapping_sha256",
    "seller_count",
    "item_count",
    "text_bytes_changed",
    "market_bytes_changed",
    "item_uid_bytes_changed",
    "production_parser_row_count",
)
EXPECTED_CONTROL_COUNTS = Counter(
    {"exact_title_clone_negative": 2, "high_semantic_similarity_negative": 4}
)


class SealedLiteralScanError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SealedLiteralScanError("Private JSONL row is not an object")
                rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SealedLiteralScanError("JSON object required")
    return value


def public_split_indexes(split: str) -> dict[str, Any]:
    observed = PUBLIC_ROOT / split / "observed"
    worlds = read_jsonl(observed / "worlds.jsonl")
    sellers = read_jsonl(observed / "sellers.jsonl")
    items = read_jsonl(observed / "items.jsonl")
    world_order = tuple(str(row.get("world_uid", "")) for row in worlds)
    sellers_by_world: dict[str, set[str]] = defaultdict(set)
    for row in sellers:
        sellers_by_world[str(row.get("world_uid", ""))].add(
            str(row.get("seller_uid", ""))
        )
    item_owner: dict[str, tuple[str, str]] = {}
    item_counts_by_world: Counter[str] = Counter()
    for row in items:
        world_uid = str(row.get("world_uid", ""))
        item_owner[str(row.get("item_uid", ""))] = (
            world_uid,
            str(row.get("seller_uid", "")),
        )
        item_counts_by_world[world_uid] += 1
    pair_sellers: dict[tuple[str, str], tuple[str, str]] = {}
    with (observed / "complete_model_pair_endpoints.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            pair_sellers[(str(row["world_uid"]), str(row["canonical_pair_uid"]))] = (
                str(row["seller_uid_left"]),
                str(row["seller_uid_right"]),
            )
    identity33_row_counts: Counter[str] = Counter()
    identity33_active_pair_counts: Counter[str] = Counter()
    with (observed / "identity33_all_pairs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if (
            len(fieldnames) != 35
            or fieldnames[:2] != ("canonical_pair_uid", "world_uid")
        ):
            raise SealedLiteralScanError("Public identity33 schema drift")
        feature_names = fieldnames[2:]
        for row in reader:
            world_uid = str(row["world_uid"])
            values = tuple(float(row[name]) for name in feature_names)
            if not all(math.isfinite(value) for value in values):
                raise SealedLiteralScanError("Public identity33 finite-value drift")
            identity33_row_counts[world_uid] += 1
            identity33_active_pair_counts[world_uid] += int(
                any(value != 0.0 for value in values)
            )
    return {
        "world_order": world_order,
        "worlds": set(world_order),
        "sellers_by_world": dict(sellers_by_world),
        "item_owner": item_owner,
        "item_counts_by_world": item_counts_by_world,
        "pair_sellers": pair_sellers,
        "identity33_row_counts": identity33_row_counts,
        "identity33_active_pair_counts": identity33_active_pair_counts,
    }


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def compact(value: str) -> str:
    return "".join(
        character
        for character in normalize(value)
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def literal_variants(value: str) -> set[str]:
    base = normalize(value)
    values = {base, compact(base)}
    lowered = base
    for prefix in (
        "https://",
        "http://",
        "www.",
        "telegram:",
        "tg:",
        "qq:",
        "wx:",
        "wechat:",
        "bat:",
    ):
        if lowered.startswith(prefix):
            stripped = lowered[len(prefix) :]
            values.update({stripped, compact(stripped)})
    return {
        candidate
        for candidate in values
        if len(candidate) >= 4 or candidate in SHORT_ASCII_IDENTIFIERS
    }


def collect_private_strings(
    value: Any,
    *,
    parent_field: str = "",
) -> tuple[set[str], set[str], int, int]:
    """Collect string-leaf variants and registered identifier-like mapping keys."""

    variants: set[str] = set()
    indexed_key_variants: set[str] = set()
    string_leaf_count = 0
    indexed_key_count = 0
    stack: list[tuple[str, Any]] = [(parent_field, value)]
    while stack:
        field, current = stack.pop()
        if isinstance(current, dict):
            if INDEXED_MAPPING_FIELD_RE.search(field):
                for key in current:
                    if isinstance(key, str) and key:
                        indexed_key_count += 1
                        key_variants = literal_variants(key)
                        variants.update(key_variants)
                        indexed_key_variants.update(key_variants)
            for key, child in current.items():
                stack.append((str(key), child))
        elif isinstance(current, list):
            stack.extend((field, child) for child in current)
        elif isinstance(current, str) and current:
            string_leaf_count += 1
            variants.update(literal_variants(current))
    return variants, indexed_key_variants, string_leaf_count, indexed_key_count


def visible_literal_variants(value: str) -> set[str]:
    variants = set(literal_variants(value))
    for expression in (CONTACT_TOKEN_RE, VISIBLE_TOKEN_RE, INTERNAL_TOKEN_RE):
        for match in expression.finditer(value):
            variants.update(literal_variants(match.group(0)))
    return variants


def private_registers() -> tuple[set[str], dict[str, set[str]], dict[str, Any]]:
    all_private_variants: set[str] = set()
    forbidden_variants: set[str] = set()
    identity_variants: set[str] = set()
    registers: dict[str, set[str]] = {
        "world": set(),
        "seller": set(),
        "item": set(),
        "pair": set(),
        "mechanism": set(),
        "control_type": set(),
        "style_id": set(),
    }
    identity_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    counts = defaultdict(int)
    manifest = read_json(PUBLIC_ROOT / "root_manifest.json")
    world_counts = manifest.get("world_counts")
    if not isinstance(world_counts, dict) or set(world_counts) != set(SPLITS):
        raise SealedLiteralScanError("Root world-count manifest drift")
    for split in SPLITS:
        public = public_split_indexes(split)
        expected_world_order = public["world_order"]
        expected_world_uids = public["worlds"]
        registers["world"].update(expected_world_uids)
        registers["seller"].update(
            seller_uid
            for seller_uids in public["sellers_by_world"].values()
            for seller_uid in seller_uids
        )
        registers["item"].update(public["item_owner"])
        registers["pair"].update(
            pair_uid for _world_uid, pair_uid in public["pair_sellers"]
        )
        parsed = read_jsonl(PRIVATE_ROOT / split / "parsed_identity_occurrences.jsonl")
        audit = read_jsonl(PRIVATE_ROOT / split / "generation_audit.jsonl")
        identity_plan = read_jsonl(PRIVATE_ROOT / split / "identity_plan.jsonl")
        for payload in (parsed, audit, identity_plan):
            variants, indexed_variants, leaf_count, indexed_key_count = (
                collect_private_strings(payload)
            )
            all_private_variants.update(variants)
            forbidden_variants.update(indexed_variants)
            counts["private_string_leaf_count"] += leaf_count
            counts["private_indexed_mapping_key_count"] += indexed_key_count
        alias_world_to_public: dict[str, str] = {}
        alias_audit_by_public: dict[str, dict[str, Any]] = {}
        alias_audit_contract_valid = True
        for generation_row in audit:
            public_world_uid = str(generation_row.get("world_uid", ""))
            alias_audit = generation_row.get("parser_uid_alias_audit")
            if not isinstance(alias_audit, dict):
                alias_audit_contract_valid = False
                continue
            alias_hash = str(alias_audit.get("alias_world_uid_sha256", ""))
            alias_audit_contract_valid = alias_audit_contract_valid and (
                tuple(alias_audit) == PARSER_ALIAS_AUDIT_FIELDS
                and str(alias_audit.get("world_uid", "")) == public_world_uid
                and re.fullmatch(r"[0-9a-f]{64}", alias_hash) is not None
                and alias_hash not in alias_world_to_public
                and public_world_uid not in alias_audit_by_public
            )
            alias_world_to_public[alias_hash] = public_world_uid
            alias_audit_by_public[public_world_uid] = alias_audit
        parsed_world_uids: set[str] = set()
        parsed_signal_uids: set[str] = set()
        parsed_value_hashes: set[str] = set()
        parsed_value_hashes_by_world: dict[str, set[str]] = defaultdict(set)
        parsed_rows_by_world: Counter[str] = Counter()
        parsed_structure_valid = True
        for row in parsed:
            alias_world_uid = str(row.get("world_uid", ""))
            alias_world_hash = hashlib.sha256(alias_world_uid.encode("utf-8")).hexdigest()
            world_uid = alias_world_to_public.get(alias_world_hash, "")
            seller_uid = str(row.get("seller_uid", ""))
            item_uid = str(row.get("item_uid", ""))
            signal_uid = str(row.get("signal_uid", ""))
            value = str(row.get("normalized_value", ""))
            variants = literal_variants(value)
            identity_variants.update(variants)
            value_hash = hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()
            identity_hashes_by_split[split].add(value_hash)
            parsed_world_uids.add(world_uid)
            parsed_value_hashes.add(value_hash)
            parsed_value_hashes_by_world[world_uid].add(value_hash)
            parsed_rows_by_world[world_uid] += 1
            parsed_structure_valid = parsed_structure_valid and (
                tuple(row) == PARSED_FIELDS
                and world_uid in expected_world_uids
                and public["item_owner"].get(item_uid, (None, None))[0] == world_uid
                and bool(signal_uid)
                and signal_uid not in parsed_signal_uids
                and bool(value)
                and str(row.get("contact_type", "")) in IDENTITY_TYPES
            )
            parsed_signal_uids.add(signal_uid)
            registers["world"].add(alias_world_uid)
            registers["seller"].add(seller_uid)
            registers["item"].add(item_uid)
        audit_world_uids: set[str] = set()
        audit_order: list[str] = []
        audit_structure_valid = True
        all_worlds_identity33_active = all(
            public["identity33_active_pair_counts"][world_uid] > 0
            for world_uid in expected_world_order
        )
        all_worlds_controls_exact = True
        for generation_row in audit:
            world_uid = str(generation_row.get("world_uid", ""))
            audit_world_uids.add(world_uid)
            audit_order.append(world_uid)
            registers["world"].add(world_uid)
            styles = generation_row.get("style_assignments", [])
            controls = generation_row.get("registered_negative_controls", [])
            style_sellers = {
                str(style.get("seller_uid", "")) for style in styles
                if isinstance(style, dict)
            }
            control_counts = Counter(
                str(control.get("control_type", "")) for control in controls
                if isinstance(control, dict)
            )
            control_pair_uids = [
                str(control.get("canonical_pair_uid", ""))
                for control in controls if isinstance(control, dict)
            ]
            controls_valid = (
                isinstance(controls, list)
                and len(controls) == 6
                and control_counts == EXPECTED_CONTROL_COUNTS
                and len(set(control_pair_uids)) == 6
            )
            for control in controls:
                if not isinstance(control, dict) or tuple(control) != CONTROL_FIELDS:
                    controls_valid = False
                    continue
                pair_uid = str(control["canonical_pair_uid"])
                source_uid = str(control["source_item_uid"])
                target_uid = str(control["target_item_uid"])
                pair_sellers = public["pair_sellers"].get((world_uid, pair_uid))
                source_owner = public["item_owner"].get(source_uid)
                target_owner = public["item_owner"].get(target_uid)
                controls_valid = controls_valid and (
                    source_uid != target_uid
                    and source_owner is not None
                    and target_owner is not None
                    and source_owner[0] == world_uid
                    and target_owner[0] == world_uid
                    and source_owner[1] != target_owner[1]
                    and pair_sellers is not None
                    and {source_owner[1], target_owner[1]} == set(pair_sellers)
                )
            audit_structure_valid = audit_structure_valid and (
                tuple(generation_row) == GENERATION_AUDIT_FIELDS
                and world_uid in expected_world_uids
                and isinstance(styles, list)
                and len(styles) == 28
                and style_sellers == public["sellers_by_world"].get(world_uid, set())
                and isinstance(generation_row.get("identity33_active_pair_count"), int)
                and int(generation_row["identity33_active_pair_count"])
                == public["identity33_active_pair_counts"][world_uid]
                and controls_valid
            )
            all_worlds_controls_exact = all_worlds_controls_exact and controls_valid
            for style in styles:
                registers["seller"].add(str(style.get("seller_uid", "")))
                registers["style_id"].add(str(style.get("base_style_id", "")))
            for mechanism in generation_row.get("mechanism_assignments", []):
                registers["mechanism"].add(str(mechanism.get("mechanism", "")))
                for seller_uid in mechanism.get("members", []):
                    registers["seller"].add(str(seller_uid))
                for asset_uid in mechanism.get("identity_asset_uids", []):
                    registers["item"].add(str(asset_uid))
            for control in controls:
                registers["pair"].add(str(control.get("canonical_pair_uid", "")))
                registers["control_type"].add(str(control.get("control_type", "")))
                registers["item"].add(str(control.get("source_item_uid", "")))
                registers["item"].add(str(control.get("target_item_uid", "")))
        plan_world_uids: set[str] = set()
        plan_asset_uids: set[str] = set()
        plan_value_hashes: set[str] = set()
        plan_value_hashes_by_world: dict[str, set[str]] = defaultdict(set)
        identity_types_by_world: dict[str, set[str]] = defaultdict(set)
        plan_structure_valid = True
        for row in identity_plan:
            world_uid = str(row.get("world_uid", ""))
            asset_uid = str(row.get("asset_uid", ""))
            identity_type = str(row.get("identity_type", ""))
            value_hash = str(row.get("value_sha256", ""))
            occurrences = row.get("seller_occurrences")
            plan_structure_valid = plan_structure_valid and (
                tuple(row) == IDENTITY_PLAN_FIELDS
                and world_uid in expected_world_uids
                and bool(asset_uid)
                and asset_uid not in plan_asset_uids
                and identity_type in IDENTITY_TYPES
                and re.fullmatch(r"[0-9a-f]{64}", value_hash) is not None
                and isinstance(occurrences, dict)
                and bool(occurrences)
                and bool(str(row.get("role", "")))
                and bool(str(row.get("mechanism", "")))
            )
            if isinstance(occurrences, dict):
                for seller_uid, count in occurrences.items():
                    plan_structure_valid = plan_structure_valid and (
                        seller_uid
                        in public["sellers_by_world"].get(world_uid, set())
                        and type(count) is int
                        and count > 0
                    )
            plan_world_uids.add(world_uid)
            plan_asset_uids.add(asset_uid)
            plan_value_hashes.add(value_hash)
            plan_value_hashes_by_world[world_uid].add(value_hash)
            identity_types_by_world[world_uid].add(identity_type)

        expected_worlds = int(world_counts[split])
        identity_types_complete = all(
            identity_types_by_world.get(world_uid, set()) == IDENTITY_TYPES
            for world_uid in expected_world_order
        )
        plan_parsed_links_exact = (
            plan_value_hashes == parsed_value_hashes
            and all(
                plan_value_hashes_by_world.get(world_uid, set())
                == parsed_value_hashes_by_world.get(world_uid, set())
                for world_uid in expected_world_order
            )
            and alias_audit_contract_valid
            and set(alias_audit_by_public) == expected_world_uids
            and all(
                int(alias_audit_by_public[world_uid]["production_parser_row_count"])
                == parsed_rows_by_world[world_uid]
                and int(alias_audit_by_public[world_uid]["seller_count"]) == 28
                and int(alias_audit_by_public[world_uid]["item_count"])
                == public["item_counts_by_world"][world_uid]
                and alias_audit_by_public[world_uid]["item_uid_bytes_changed"]
                is False
                for world_uid in expected_world_order
            )
        )
        exact_nontruth_structure = (
            len(expected_world_order) == expected_worlds
            and len(expected_world_uids) == expected_worlds
            and tuple(audit_order) == expected_world_order
            and audit_world_uids == expected_world_uids
            and parsed_world_uids == expected_world_uids
            and plan_world_uids == expected_world_uids
            and parsed_structure_valid
            and audit_structure_valid
            and plan_structure_valid
            and plan_parsed_links_exact
            and identity_types_complete
            and all(
                public["identity33_row_counts"][world_uid] == 378
                for world_uid in expected_world_order
            )
            and all_worlds_identity33_active
        )
        counts[f"{split}_parsed_identity_row_count"] = len(parsed)
        counts[f"{split}_generation_audit_row_count"] = len(audit)
        counts[f"{split}_identity_plan_row_count"] = len(identity_plan)
        counts[f"{split}_nontruth_registry_reconciled"] = (
            len(audit) == expected_worlds
            and len(audit_world_uids) == expected_worlds
            and "" not in audit_world_uids
            and bool(parsed)
            and bool(identity_plan)
        )
        counts[f"{split}_nontruth_structure_exact"] = exact_nontruth_structure
        counts[f"{split}_identity_types_complete_every_world"] = (
            identity_types_complete
        )
        counts[f"{split}_identity33_active_every_world"] = (
            all_worlds_identity33_active
        )
        counts[f"{split}_registered_controls_exact_every_world"] = (
            all_worlds_controls_exact
        )
        counts[f"{split}_plan_parsed_links_exact"] = plan_parsed_links_exact
    for values in registers.values():
        values.discard("")
    identity_intersections = 0
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            identity_intersections += len(
                identity_hashes_by_split[left].intersection(identity_hashes_by_split[right])
            )
    counts["cross_split_identity_hash_intersection_count"] = identity_intersections
    counts["private_identity_variant_count"] = len(identity_variants)
    counts["private_all_string_variant_count"] = len(all_private_variants)
    counts["private_all_string_variant_commitment_sha256"] = canonical_sha256(
        sorted(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in all_private_variants
        )
    )
    counts["world_register_count"] = len(registers["world"])
    counts["seller_register_count"] = len(registers["seller"])
    counts["item_register_count"] = len(registers["item"])
    counts["pair_register_count"] = len(registers["pair"])
    counts["private_register_commitment_sha256"] = canonical_sha256(
        {
            name: sorted(
                hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values
            )
            for name, values in registers.items()
        }
    )
    counts["identity_hash_register_commitment_sha256"] = canonical_sha256(
        {
            split: sorted(values)
            for split, values in identity_hashes_by_split.items()
        }
    )
    counts["all_split_nontruth_registries_reconciled"] = all(
        bool(counts[f"{split}_nontruth_registry_reconciled"])
        for split in SPLITS
    )
    counts["all_split_nontruth_structures_exact"] = all(
        bool(counts[f"{split}_nontruth_structure_exact"]) for split in SPLITS
    )
    for suffix in (
        "identity_types_complete_every_world",
        "identity33_active_every_world",
        "registered_controls_exact_every_world",
        "plan_parsed_links_exact",
    ):
        counts[f"all_split_{suffix}"] = all(
            bool(counts[f"{split}_{suffix}"]) for split in SPLITS
        )
    forbidden_variants.update(identity_variants)
    forbidden_variants.update(
        variant
        for values in registers.values()
        for value in values
        for variant in literal_variants(value)
    )
    counts["private_forbidden_variant_count"] = len(forbidden_variants)
    counts["private_forbidden_variant_commitment_sha256"] = canonical_sha256(
        sorted(
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in forbidden_variants
        )
    )
    return forbidden_variants, registers, dict(counts)


def public_visible_scan(
    forbidden_variants: set[str], registers: dict[str, set[str]]
) -> dict[str, int | str]:
    counts = defaultdict(int)
    for name in (
        "visible_row_count",
        "visible_field_count",
        "visible_contact_candidate_count",
        "internal_token_match_count",
        "exact_private_forbidden_literal_match_count",
        "exact_private_register_literal_match_count",
    ):
        counts[name] = 0
    register_variants = {
        variant
        for values in registers.values()
        for value in values
        for variant in literal_variants(value)
    }
    visible_candidate_hashes: set[str] = set()
    scanned_hash = hashlib.sha256()
    for split in SPLITS:
        paths = (
            PUBLIC_ROOT / split / "observed" / "redacted_items.jsonl",
            PUBLIC_ROOT / split / "observed" / "model_seller_profiles.jsonl",
        )
        for path in paths:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if path.name == "redacted_items.jsonl":
                        values = (str(row["title"]), str(row["description"]))
                    else:
                        values = tuple(
                            str(row[name])
                            for name in (
                                "category_concat_top",
                                "signature_title_concat",
                                "title_concat_top",
                                "signature_description_concat",
                                "description_concat_top",
                            )
                        )
                    counts["visible_row_count"] += 1
                    for value in values:
                        counts["visible_field_count"] += 1
                        scanned_hash.update(normalize(value).encode("utf-8"))
                        scanned_hash.update(b"\0")
                        counts["internal_token_match_count"] += len(
                            INTERNAL_TOKEN_RE.findall(value)
                        )
                        for match in CONTACT_TOKEN_RE.finditer(value):
                            counts["visible_contact_candidate_count"] += 1
                        variants = visible_literal_variants(value)
                        visible_candidate_hashes.update(
                            hashlib.sha256(variant.encode("utf-8")).hexdigest()
                            for variant in variants
                        )
                        counts["exact_private_forbidden_literal_match_count"] += len(
                            variants.intersection(forbidden_variants)
                        )
                        counts["exact_private_register_literal_match_count"] += len(
                            variants.intersection(register_variants)
                        )
    counts["visible_text_stream_sha256"] = scanned_hash.hexdigest()
    counts["visible_candidate_commitment_sha256"] = canonical_sha256(
        sorted(visible_candidate_hashes)
    )
    return dict(counts)


def main() -> None:
    forbidden_variants, registers, private_counts = private_registers()
    public_counts = public_visible_scan(forbidden_variants, registers)
    result: dict[str, Any] = {
        "version": VERSION,
        "status": "SEALED_LITERAL_SCAN_COMPLETE",
        "private_counts_and_commitments": private_counts,
        "public_scan": public_counts,
        "hard_gate_passed": (
            private_counts["cross_split_identity_hash_intersection_count"] == 0
            and private_counts["all_split_nontruth_registries_reconciled"] is True
            and private_counts["all_split_nontruth_structures_exact"] is True
            and private_counts["all_split_plan_parsed_links_exact"] is True
            and private_counts["all_split_identity_types_complete_every_world"]
            is True
            and private_counts["all_split_identity33_active_every_world"] is True
            and private_counts[
                "all_split_registered_controls_exact_every_world"
            ]
            is True
            and public_counts.get("internal_token_match_count", 0) == 0
            and public_counts.get("exact_private_forbidden_literal_match_count", 0)
            == 0
            and public_counts.get("exact_private_register_literal_match_count", 0) == 0
        ),
        "private_values_returned": 0,
        "row_level_hits_returned": 0,
        "pair_labels_parsed": 0,
        "controller_membership_parsed": 0,
        "qrels_parsed": 0,
        "controller_relations_returned": 0,
        "qrels_returned": 0,
    }
    result["canonical_self_hash"] = canonical_sha256(result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
