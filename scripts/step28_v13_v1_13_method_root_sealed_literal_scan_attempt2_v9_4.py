#!/usr/bin/env python3
"""Sealed aggregate-only literal leakage scan for V9.4 quality attempt 2.

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
VERSION = "2026-08-28-step28-v13-v1-13-v9-4-sealed-literal-scan-attempt2-v1"
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
    return {value for value in values if len(value) >= 4}


def private_registers() -> tuple[set[str], dict[str, set[str]], dict[str, int]]:
    identity_variants: set[str] = set()
    registers: dict[str, set[str]] = {
        "world": set(),
        "seller": set(),
        "controller": set(),
        "query": set(),
    }
    identity_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    counts = defaultdict(int)
    for split in SPLITS:
        parsed = read_jsonl(PRIVATE_ROOT / split / "parsed_identity_occurrences.jsonl")
        membership = read_jsonl(PRIVATE_ROOT / split / "controller_membership.jsonl")
        qrels = read_jsonl(PRIVATE_ROOT / split / "qrels.jsonl")
        audit = read_jsonl(PRIVATE_ROOT / split / "generation_audit.jsonl")
        for row in parsed:
            value = str(row.get("normalized_value", ""))
            variants = literal_variants(value)
            identity_variants.update(variants)
            identity_hashes_by_split[split].add(
                hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()
            )
            registers["world"].add(str(row.get("world_uid", "")))
            registers["seller"].add(str(row.get("seller_uid", "")))
        for row in membership:
            registers["world"].add(str(row.get("world_uid", "")))
            registers["seller"].add(str(row.get("seller_uid", "")))
            registers["controller"].add(
                f"{row.get('world_uid', '')}::{row.get('controller_group_index', '')}"
            )
        for row in qrels:
            registers["world"].add(str(row.get("world_uid", "")))
            registers["query"].add(str(row.get("query_seller_uid", "")))
            for seller_uid in row.get("relevant_seller_uids", []):
                registers["seller"].add(str(seller_uid))
        # Read all string leaves in the private generation audit, but retain
        # only a hash count.  Values are not emitted or joined to public rows.
        stack: list[Any] = list(audit)
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str) and value:
                counts["private_generation_audit_string_leaf_count"] += 1
        counts[f"{split}_parsed_identity_row_count"] = len(parsed)
        counts[f"{split}_membership_row_count"] = len(membership)
        counts[f"{split}_qrel_row_count"] = len(qrels)
    for values in registers.values():
        values.discard("")
    intersections = 0
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            intersections += len(
                identity_hashes_by_split[left].intersection(identity_hashes_by_split[right])
            )
    counts["cross_split_identity_hash_intersection_count"] = intersections
    counts["private_identity_variant_count"] = len(identity_variants)
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
    return identity_variants, registers, dict(counts)


def public_visible_scan(
    identity_variants: set[str], registers: dict[str, set[str]]
) -> dict[str, int | str]:
    counts = defaultdict(int)
    for name in (
        "visible_row_count",
        "visible_field_count",
        "visible_contact_candidate_count",
        "internal_token_match_count",
        "exact_private_identity_literal_match_count",
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
                            token = match.group(0)
                            variants = literal_variants(token)
                            visible_candidate_hashes.update(
                                hashlib.sha256(variant.encode("utf-8")).hexdigest()
                                for variant in variants
                            )
                            counts["exact_private_identity_literal_match_count"] += len(
                                variants.intersection(identity_variants)
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
    identity_variants, registers, private_counts = private_registers()
    public_counts = public_visible_scan(identity_variants, registers)
    result: dict[str, Any] = {
        "version": VERSION,
        "status": "SEALED_LITERAL_SCAN_COMPLETE",
        "private_counts_and_commitments": private_counts,
        "public_scan": public_counts,
        "hard_gate_passed": (
            private_counts["cross_split_identity_hash_intersection_count"] == 0
            and public_counts.get("internal_token_match_count", 0) == 0
            and public_counts.get("exact_private_identity_literal_match_count", 0) == 0
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
