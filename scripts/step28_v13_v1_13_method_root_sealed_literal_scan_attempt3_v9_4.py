#!/usr/bin/env python3
"""Sealed aggregate-only literal leakage scan for V9.4 quality attempt 3.

This process may inspect private strings, including audit A/B custody files.  It
returns counts and commitments only; no private literal, label, controller
relation, qrel, or row-level hit is returned.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "v9_4_method_root_attempt1_20260828"
)
PRIVATE_ROOT = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_4_method_root_attempt1_20260828"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
VERSION = "2026-08-28-step28-v13-v1-13-v9-4-sealed-literal-scan-attempt3-v1"
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
        "controller": set(),
        "query": set(),
        "item": set(),
        "pair": set(),
        "mechanism": set(),
        "control_type": set(),
        "style_id": set(),
    }
    identity_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    split_registers: dict[str, dict[str, set[str]]] = {
        split: {"controller": set(), "query": set()} for split in SPLITS
    }
    counts = defaultdict(int)
    manifest = read_json(PUBLIC_ROOT / "root_manifest.json")
    world_counts = manifest.get("world_counts")
    if not isinstance(world_counts, dict) or set(world_counts) != set(SPLITS):
        raise SealedLiteralScanError("Root world-count manifest drift")
    for split in SPLITS:
        parsed = read_jsonl(PRIVATE_ROOT / split / "parsed_identity_occurrences.jsonl")
        membership = read_jsonl(PRIVATE_ROOT / split / "controller_membership.jsonl")
        qrels = read_jsonl(PRIVATE_ROOT / split / "qrels.jsonl")
        audit = read_jsonl(PRIVATE_ROOT / split / "generation_audit.jsonl")
        identity_plan = read_jsonl(PRIVATE_ROOT / split / "identity_plan.jsonl")
        for payload in (parsed, membership, qrels, audit, identity_plan):
            variants, indexed_variants, leaf_count, indexed_key_count = (
                collect_private_strings(payload)
            )
            all_private_variants.update(variants)
            forbidden_variants.update(indexed_variants)
            counts["private_string_leaf_count"] += leaf_count
            counts["private_indexed_mapping_key_count"] += indexed_key_count
        for row in parsed:
            value = str(row.get("normalized_value", ""))
            variants = literal_variants(value)
            identity_variants.update(variants)
            identity_hashes_by_split[split].add(
                hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()
            )
            registers["world"].add(str(row.get("world_uid", "")))
            registers["seller"].add(str(row.get("seller_uid", "")))
            registers["item"].add(str(row.get("item_uid", "")))
        for row in membership:
            registers["world"].add(str(row.get("world_uid", "")))
            registers["seller"].add(str(row.get("seller_uid", "")))
            controller = (
                f"{row.get('world_uid', '')}::{row.get('controller_group_index', '')}"
            )
            registers["controller"].add(controller)
            split_registers[split]["controller"].add(controller)
        for row in qrels:
            registers["world"].add(str(row.get("world_uid", "")))
            query = str(row.get("query_seller_uid", ""))
            registers["query"].add(query)
            split_registers[split]["query"].add(query)
            for seller_uid in row.get("relevant_seller_uids", []):
                registers["seller"].add(str(seller_uid))
        for generation_row in audit:
            registers["world"].add(str(generation_row.get("world_uid", "")))
            for style in generation_row.get("style_assignments", []):
                registers["seller"].add(str(style.get("seller_uid", "")))
                registers["style_id"].add(str(style.get("base_style_id", "")))
            for mechanism in generation_row.get("mechanism_assignments", []):
                registers["mechanism"].add(str(mechanism.get("mechanism", "")))
                for seller_uid in mechanism.get("members", []):
                    registers["seller"].add(str(seller_uid))
                for asset_uid in mechanism.get("identity_asset_uids", []):
                    registers["item"].add(str(asset_uid))
            for control in generation_row.get("registered_negative_controls", []):
                registers["pair"].add(str(control.get("canonical_pair_uid", "")))
                registers["control_type"].add(str(control.get("control_type", "")))
                registers["item"].add(str(control.get("source_item_uid", "")))
                registers["item"].add(str(control.get("target_item_uid", "")))
        counts[f"{split}_parsed_identity_row_count"] = len(parsed)
        counts[f"{split}_membership_row_count"] = len(membership)
        counts[f"{split}_qrel_row_count"] = len(qrels)
        counts[f"{split}_identity_plan_row_count"] = len(identity_plan)
        expected_worlds = int(world_counts[split])
        counts[f"{split}_registry_manifest_reconciled"] = (
            len(membership) == expected_worlds * 28
            and len(split_registers[split]["controller"]) == expected_worlds * 12
            and len(qrels) == expected_worlds * 28
            and len(split_registers[split]["query"]) == expected_worlds * 28
        )
    for values in registers.values():
        values.discard("")
    for by_kind in split_registers.values():
        for values in by_kind.values():
            values.discard("")
    identity_intersections = 0
    controller_intersections = 0
    query_intersections = 0
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            identity_intersections += len(
                identity_hashes_by_split[left].intersection(identity_hashes_by_split[right])
            )
            controller_intersections += len(
                split_registers[left]["controller"].intersection(
                    split_registers[right]["controller"]
                )
            )
            query_intersections += len(
                split_registers[left]["query"].intersection(
                    split_registers[right]["query"]
                )
            )
    counts["cross_split_identity_hash_intersection_count"] = identity_intersections
    counts["cross_split_controller_register_intersection_count"] = (
        controller_intersections
    )
    counts["cross_split_query_register_intersection_count"] = query_intersections
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
    counts["controller_register_count"] = len(registers["controller"])
    counts["query_register_count"] = len(registers["query"])
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
    counts["split_register_commitment_sha256"] = canonical_sha256(
        {
            split: {
                name: sorted(
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in values
                )
                for name, values in by_kind.items()
            }
            for split, by_kind in split_registers.items()
        }
    )
    counts["all_split_registries_reconciled"] = all(
        bool(counts[f"{split}_registry_manifest_reconciled"]) for split in SPLITS
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
            and private_counts["cross_split_controller_register_intersection_count"]
            == 0
            and private_counts["cross_split_query_register_intersection_count"] == 0
            and private_counts["all_split_registries_reconciled"] is True
            and public_counts.get("internal_token_match_count", 0) == 0
            and public_counts.get("exact_private_forbidden_literal_match_count", 0)
            == 0
            and public_counts.get("exact_private_register_literal_match_count", 0) == 0
        ),
        "private_values_returned": 0,
        "row_level_hits_returned": 0,
        "pair_labels_parsed": 0,
        "controller_relations_returned": 0,
        "qrels_returned": 0,
    }
    result["canonical_self_hash"] = canonical_sha256(result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
