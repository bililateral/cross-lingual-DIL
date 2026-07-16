#!/usr/bin/env python3
"""Read-only Step16I audit for split integrity and permanent exclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step16i_data_integrity_policy.json"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Step5 seller components and audit data integrity without modifying inputs."
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument(
        "--run-id",
        default=None,
        help="Immutable output run ID. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional immutable output directory override.",
    )
    parser.add_argument(
        "--v8-readiness-assignment",
        default=None,
        help="Optional override for the V8 readiness assignment archive.",
    )
    parser.add_argument(
        "--skip-v8-readiness-check",
        action="store_true",
        help="Record the optional V8 readiness check as disabled.",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def ensure_within_workspace(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project workspace: {path}") from exc


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_seller_alias(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def portable_seller_alias(value: str) -> str:
    token = normalize_seller_alias(value)
    compact = token.strip("/")
    if not token or re.fullmatch(r"(?:shop/)?\d+", compact):
        return ""
    return token


def alias_from_seller_uid(value: str) -> str:
    token = str(value).strip()
    marker = "seller_raw:"
    position = token.casefold().rfind(marker)
    return portable_seller_alias(token[position + len(marker) :]) if position >= 0 else ""


def read_csv(path: Path, required_columns: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Step16I input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(required_columns - set(fields))
        if missing:
            raise ValueError(f"Required columns are missing from {path}: {missing}")
        rows = list(reader)
    for row_index, row in enumerate(rows, start=2):
        for field in ("pair_uid", "split_name", "seller_uid_left", "seller_uid_right"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Blank {field} in {path} at CSV row {row_index}")
        if row["seller_uid_left"].strip() == row["seller_uid_right"].strip():
            raise ValueError(f"Self-pair in {path} at CSV row {row_index}: {row['pair_uid']}")
    return rows, fields


def render_csv(rows: list[dict[str, object]], fieldnames: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def stable_component_id(prefix: str, sellers: Iterable[str]) -> str:
    members = sorted(set(sellers))
    digest = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def build_components(
    rows: list[dict[str, str]], prefix: str
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    union_find = UnionFind()
    for row in rows:
        union_find.union(row["seller_uid_left"].strip(), row["seller_uid_right"].strip())

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for seller_uid in union_find.parent:
        members_by_root[union_find.find(seller_uid)].add(seller_uid)

    seller_to_component: dict[str, str] = {}
    components: dict[str, dict[str, object]] = {}
    for members in sorted(members_by_root.values(), key=lambda values: sorted(values)):
        component_id = stable_component_id(prefix, members)
        components[component_id] = {
            "sellers": set(members),
            "pair_uids": set(),
            "splits": set(),
            "old_component_ids": set(),
            "review_trace_ids": set(),
        }
        for seller_uid in members:
            seller_to_component[seller_uid] = component_id
    return seller_to_component, components


def canonical_pair_key(row: dict[str, str]) -> str:
    return "||".join(sorted((row["seller_uid_left"].strip(), row["seller_uid_right"].strip())))


def compile_review_patterns(policy: dict) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for item in policy.get("review_assistance_patterns", []):
        pattern_id = str(item.get("id", "")).strip()
        expression = str(item.get("pattern", "")).strip()
        if not pattern_id or not expression:
            raise ValueError("Each review_assistance_patterns record needs id and pattern")
        compiled.append((pattern_id, re.compile(expression, flags=re.IGNORECASE)))
    return compiled


def review_trace_ids(notes: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    return [pattern_id for pattern_id, pattern in patterns if pattern.search(notes or "")]


def partition_status(old_count: int, recomputed_count: int, old_missing: bool) -> str:
    if old_missing:
        return "old_component_missing"
    if old_count == 1 and recomputed_count == 1:
        return "one_to_one"
    if old_count > 1 and recomputed_count == 1:
        return "recomputed_component_split_across_old_ids"
    if old_count == 1 and recomputed_count > 1:
        return "old_component_overmerged_recomputed_components"
    return "many_to_many_partition_mismatch"


def audit_dataset(
    dataset: str,
    rows: list[dict[str, str]],
    primary_splits: set[str],
    patterns: list[tuple[str, re.Pattern[str]]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, str], dict[str, dict[str, object]]]:
    primary_rows = [row for row in rows if row["split_name"].strip() in primary_splits]
    unknown_primary = sorted(
        {row["split_name"].strip() for row in rows if not row["split_name"].strip()}
    )
    if unknown_primary:
        raise ValueError(f"Blank split names are not allowed in {dataset}")

    seller_to_component, components = build_components(primary_rows, f"{dataset}_cc")
    old_to_new: dict[str, set[str]] = defaultdict(set)
    new_to_old: dict[str, set[str]] = defaultdict(set)
    old_pair_counts: Counter[str] = Counter()
    old_stored_sizes: dict[str, set[str]] = defaultdict(set)
    pair_uid_to_splits: dict[str, set[str]] = defaultdict(set)
    canonical_pair_to_splits: dict[str, set[str]] = defaultdict(set)
    seller_to_splits: dict[str, set[str]] = defaultdict(set)
    seller_alias_to_splits: dict[str, set[str]] = defaultdict(set)
    pair_uid_counts: Counter[str] = Counter()

    trace_counts: Counter[str] = Counter()
    trace_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    trace_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    trace_examples: list[dict[str, str]] = []
    trace_ids_by_pair: dict[str, list[str]] = {}

    for row in rows:
        trace_ids = review_trace_ids(row.get("review_notes", ""), patterns)
        trace_ids_by_pair[row["pair_uid"]] = trace_ids
        for trace_id in trace_ids:
            trace_counts[trace_id] += 1
            trace_by_split[row["split_name"]][trace_id] += 1
            trace_by_label[row.get("review_label", "")][trace_id] += 1
        if trace_ids and len(trace_examples) < 50:
            trace_examples.append(
                {
                    "pair_uid": row["pair_uid"],
                    "split_name": row["split_name"],
                    "review_label": row.get("review_label", ""),
                    "trace_ids": "|".join(trace_ids),
                    "review_notes_excerpt": row.get("review_notes", "")[:300],
                }
            )

    for row in primary_rows:
        split = row["split_name"].strip()
        pair_uid = row["pair_uid"].strip()
        left = row["seller_uid_left"].strip()
        right = row["seller_uid_right"].strip()
        component_id = seller_to_component[left]
        if seller_to_component[right] != component_id:
            raise AssertionError(f"Union-find inconsistency for {pair_uid}")
        old_component_id = row.get("split_component_id", "").strip()
        component = components[component_id]
        component["pair_uids"].add(pair_uid)
        component["splits"].add(split)
        component["review_trace_ids"].update(trace_ids_by_pair.get(pair_uid, []))
        if old_component_id:
            component["old_component_ids"].add(old_component_id)
            old_to_new[old_component_id].add(component_id)
            new_to_old[component_id].add(old_component_id)
            old_pair_counts[old_component_id] += 1
            old_stored_sizes[old_component_id].add(row.get("split_component_size", "").strip())
        pair_uid_to_splits[pair_uid].add(split)
        canonical_pair_to_splits[canonical_pair_key(row)].add(split)
        seller_to_splits[left].add(split)
        seller_to_splits[right].add(split)
        for seller_alias in (
            portable_seller_alias(row.get("source_seller_raw_left", "")),
            portable_seller_alias(row.get("source_seller_raw_right", "")),
        ):
            if seller_alias:
                seller_alias_to_splits[seller_alias].add(split)
        pair_uid_counts[pair_uid] += 1

    component_rows: list[dict[str, object]] = []
    for row in sorted(primary_rows, key=lambda item: (item["split_name"], item["pair_uid"])):
        pair_uid = row["pair_uid"].strip()
        old_component_id = row.get("split_component_id", "").strip()
        component_id = seller_to_component[row["seller_uid_left"].strip()]
        component = components[component_id]
        old_count = len(new_to_old.get(component_id, set()))
        recomputed_count = len(old_to_new.get(old_component_id, set())) if old_component_id else 0
        component_rows.append(
            {
                "dataset": dataset,
                "split_name": row["split_name"].strip(),
                "pair_uid": pair_uid,
                "seller_uid_left": row["seller_uid_left"].strip(),
                "seller_uid_right": row["seller_uid_right"].strip(),
                "review_label": row.get("review_label", "").strip(),
                "old_split_component_id": old_component_id,
                "stored_split_component_size": row.get("split_component_size", "").strip(),
                "old_component_observed_pair_count": old_pair_counts.get(old_component_id, 0),
                "old_component_recomputed_component_count": recomputed_count,
                "recomputed_component_id": component_id,
                "recomputed_component_pair_count": len(component["pair_uids"]),
                "recomputed_component_seller_count": len(component["sellers"]),
                "recomputed_component_splits": "|".join(sorted(component["splits"])),
                "recomputed_component_old_id_count": old_count,
                "old_partition_status": partition_status(
                    old_count, recomputed_count, not bool(old_component_id)
                ),
                "cross_split_component_leakage": int(len(component["splits"]) > 1),
                "cross_split_seller_leakage": int(
                    len(seller_to_splits[row["seller_uid_left"].strip()]) > 1
                    or len(seller_to_splits[row["seller_uid_right"].strip()]) > 1
                ),
                "cross_split_pair_uid_leakage": int(len(pair_uid_to_splits[pair_uid]) > 1),
                "review_assistance_trace": "|".join(trace_ids_by_pair.get(pair_uid, [])),
            }
        )

    leaking_pair_uids = sorted(key for key, splits in pair_uid_to_splits.items() if len(splits) > 1)
    leaking_canonical_pairs = sorted(
        key for key, splits in canonical_pair_to_splits.items() if len(splits) > 1
    )
    leaking_sellers = sorted(key for key, splits in seller_to_splits.items() if len(splits) > 1)
    leaking_seller_aliases = sorted(
        key for key, splits in seller_alias_to_splits.items() if len(splits) > 1
    )
    leaking_components = sorted(
        component_id for component_id, item in components.items() if len(item["splits"]) > 1
    )
    duplicate_pair_uids = sorted(key for key, count in pair_uid_counts.items() if count > 1)
    split_counts = Counter(row["split_name"].strip() for row in primary_rows)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in primary_rows:
        split_label_counts[row["split_name"].strip()][row.get("review_label", "").strip()] += 1

    fragmented_new = {
        component_id: sorted(old_ids)
        for component_id, old_ids in new_to_old.items()
        if len(old_ids) > 1
    }
    overmerged_old = {
        old_id: sorted(component_ids)
        for old_id, component_ids in old_to_new.items()
        if len(component_ids) > 1
    }
    inconsistent_old_sizes = []
    for old_id, observed_count in old_pair_counts.items():
        parsed_sizes = set()
        for value in old_stored_sizes[old_id]:
            try:
                parsed_sizes.add(int(float(value)))
            except (TypeError, ValueError):
                parsed_sizes.add(-1)
        if parsed_sizes != {observed_count}:
            inconsistent_old_sizes.append(old_id)

    leakage = {
        "pair_uid_cross_split_count": len(leaking_pair_uids),
        "canonical_seller_pair_cross_split_count": len(leaking_canonical_pairs),
        "seller_cross_split_count": len(leaking_sellers),
        "seller_alias_cross_split_count": len(leaking_seller_aliases),
        "recomputed_component_cross_split_count": len(leaking_components),
        "duplicate_pair_uid_row_count": len(duplicate_pair_uids),
        "pair_uid_cross_split_examples": leaking_pair_uids[:25],
        "canonical_seller_pair_cross_split_examples": leaking_canonical_pairs[:25],
        "seller_cross_split_examples": leaking_sellers[:25],
        "seller_alias_cross_split_examples": leaking_seller_aliases[:25],
        "component_cross_split_examples": leaking_components[:25],
        "duplicate_pair_uid_examples": duplicate_pair_uids[:25],
    }
    leakage["detected"] = any(
        leakage[key]
        for key in (
            "pair_uid_cross_split_count",
            "canonical_seller_pair_cross_split_count",
            "seller_cross_split_count",
            "seller_alias_cross_split_count",
            "recomputed_component_cross_split_count",
            "duplicate_pair_uid_row_count",
        )
    )

    summary = {
        "all_step5_row_count": len(rows),
        "primary_row_count": len(primary_rows),
        "primary_split_counts": dict(sorted(split_counts.items())),
        "primary_split_label_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_label_counts.items())
        },
        "primary_seller_count": len(seller_to_component),
        "recomputed_component_count": len(components),
        "leakage": leakage,
        "old_component_comparison": {
            "old_component_count": len(old_to_new),
            "recomputed_component_count": len(components),
            "recomputed_components_split_across_multiple_old_ids": len(fragmented_new),
            "old_components_overmerging_recomputed_components": len(overmerged_old),
            "old_component_size_inconsistent_count": len(inconsistent_old_sizes),
            "maximum_old_ids_per_recomputed_component": max(
                (len(values) for values in new_to_old.values()), default=0
            ),
            "maximum_recomputed_components_per_old_id": max(
                (len(values) for values in old_to_new.values()), default=0
            ),
            "fragmented_recomputed_component_examples": dict(list(sorted(fragmented_new.items()))[:25]),
            "overmerged_old_component_examples": dict(list(sorted(overmerged_old.items()))[:25]),
            "old_component_size_inconsistent_examples": sorted(inconsistent_old_sizes)[:25],
        },
        "review_assistance_traces": {
            "row_count_with_any_trace": sum(bool(value) for value in trace_ids_by_pair.values()),
            "pattern_counts": dict(sorted(trace_counts.items())),
            "counts_by_split": {
                split: dict(sorted(counts.items())) for split, counts in sorted(trace_by_split.items())
            },
            "counts_by_review_label": {
                label: dict(sorted(counts.items())) for label, counts in sorted(trace_by_label.items())
            },
            "examples": trace_examples,
        },
        "non_primary_split_counts": dict(
            sorted(Counter(row["split_name"] for row in rows if row["split_name"] not in primary_splits).items())
        ),
    }
    return component_rows, summary, seller_to_component, components


def first_present(row: dict[str, str], fields: list[str]) -> tuple[str, str]:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            return field, value
    return "", ""


def normalize_readiness_split(value: str, check_cfg: dict) -> str:
    token = str(value).strip()
    aliases = {
        str(key).strip(): str(target).strip()
        for key, target in check_cfg.get("split_aliases", {}).items()
    }
    return aliases.get(token, token)


def audit_v8_readiness(
    path: Path | None,
    check_cfg: dict,
    primary_splits: set[str],
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, str], dict[str, dict[str, object]]]:
    if path is None:
        return {"status": "disabled", "passed": None}, [], {}, {}
    if not path.is_file():
        return {
            "status": "archive_missing",
            "passed": None,
            "assignment_path": relative_path(path),
            "message": "The configured latest V8 readiness assignment is absent; no older archive was substituted.",
        }, [], {}, {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"pair_uid", "seller_uid_left", "seller_uid_right"}
        missing = sorted(required - fields)
        if missing:
            return {
                "status": "invalid_archive",
                "passed": False,
                "assignment_path": relative_path(path),
                "missing_columns": missing,
            }, [], {}, {}
        rows = list(reader)

    split_precedence = list(check_cfg.get("split_field_precedence", []))
    component_field = str(check_cfg.get("component_field", "split_component_id"))
    normalized_rows: list[dict[str, str]] = []
    missing_split_rows = []
    missing_component_rows = []
    malformed_identity_rows = []
    duplicate_pair_counts: Counter[str] = Counter()
    for row in rows:
        pair_uid = str(row.get("pair_uid", "")).strip()
        left = str(row.get("seller_uid_left", "")).strip()
        right = str(row.get("seller_uid_right", "")).strip()
        if not pair_uid or not left or not right or left == right:
            malformed_identity_rows.append(pair_uid or "<blank_pair_uid>")
        split_field, split_raw = first_present(row, split_precedence)
        split = normalize_readiness_split(split_raw, check_cfg)
        if not split:
            missing_split_rows.append(row.get("pair_uid", ""))
        component_id = str(row.get(component_field, "")).strip()
        if not component_id:
            missing_component_rows.append(row.get("pair_uid", ""))
        normalized = dict(row)
        normalized["_audit_split"] = split
        normalized["_audit_split_field"] = split_field
        normalized_rows.append(normalized)
        duplicate_pair_counts[pair_uid] += 1

    malformed = bool(missing_split_rows or missing_component_rows or malformed_identity_rows)
    if malformed:
        return {
            "status": "invalid_archive",
            "passed": False,
            "assignment_path": relative_path(path),
            "assignment_sha256": file_sha256(path),
            "row_count": len(rows),
            "missing_split_count": len(missing_split_rows),
            "missing_component_count": len(missing_component_rows),
            "malformed_identity_count": len(malformed_identity_rows),
            "missing_split_examples": missing_split_rows[:25],
            "missing_component_examples": missing_component_rows[:25],
            "malformed_identity_examples": malformed_identity_rows[:25],
        }, normalized_rows, {}, {}

    seller_to_component, components = build_components(normalized_rows, "v8ready_cc")
    persisted_to_recomputed: dict[str, set[str]] = defaultdict(set)
    recomputed_to_persisted: dict[str, set[str]] = defaultdict(set)
    seller_splits: dict[str, set[str]] = defaultdict(set)
    seller_alias_splits: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    endpoint_mismatch_count = 0
    for row in normalized_rows:
        left = row["seller_uid_left"].strip()
        right = row["seller_uid_right"].strip()
        recomputed = seller_to_component[left]
        if seller_to_component[right] != recomputed:
            endpoint_mismatch_count += 1
        persisted = row[component_field].strip()
        split = row["_audit_split"]
        persisted_to_recomputed[persisted].add(recomputed)
        recomputed_to_persisted[recomputed].add(persisted)
        components[recomputed]["pair_uids"].add(row["pair_uid"])
        components[recomputed]["splits"].add(split)
        components[recomputed]["old_component_ids"].add(persisted)
        seller_splits[left].add(split)
        seller_splits[right].add(split)
        for seller_alias in (alias_from_seller_uid(left), alias_from_seller_uid(right)):
            if seller_alias:
                seller_alias_splits[seller_alias].add(split)
        pair_splits[row["pair_uid"]].add(split)

    persisted_overmerged = {
        key: sorted(values) for key, values in persisted_to_recomputed.items() if len(values) > 1
    }
    recomputed_fragmented = {
        key: sorted(values) for key, values in recomputed_to_persisted.items() if len(values) > 1
    }
    leaking_components = [key for key, value in components.items() if len(value["splits"]) > 1]
    leaking_sellers = [key for key, value in seller_splits.items() if len(value) > 1]
    leaking_seller_aliases = [
        key for key, value in seller_alias_splits.items() if len(value) > 1
    ]
    leaking_pairs = [key for key, value in pair_splits.items() if len(value) > 1]
    duplicate_pairs = [key for key, value in duplicate_pair_counts.items() if value > 1]
    non_primary_splits = sorted({row["_audit_split"] for row in normalized_rows} - primary_splits)
    partition_equivalent = not persisted_overmerged and not recomputed_fragmented
    conservative_coarsening = bool(persisted_overmerged) and not recomputed_fragmented
    passed = not any(
        (
            endpoint_mismatch_count,
            len(recomputed_fragmented),
            len(leaking_components),
            len(leaking_sellers),
            len(leaking_seller_aliases),
            len(leaking_pairs),
            len(duplicate_pairs),
            len(non_primary_splits),
        )
    )
    status = "warning" if passed and conservative_coarsening else ("pass" if passed else "fail")
    summary = {
        "status": status,
        "passed": passed,
        "assignment_path": relative_path(path),
        "assignment_sha256": file_sha256(path),
        "row_count": len(normalized_rows),
        "seller_count": len(seller_to_component),
        "recomputed_component_count": len(components),
        "persisted_component_count": len(persisted_to_recomputed),
        "partition_equivalent_to_recomputed_components": partition_equivalent,
        "partition_is_safe_conservative_coarsening": conservative_coarsening,
        "partition_safety_rule": (
            "Persisted components may merge disconnected recomputed components within one split, "
            "but a recomputed seller-connected component may not be fragmented across persisted IDs."
        ),
        "persisted_components_overmerging_recomputed_count": len(persisted_overmerged),
        "recomputed_components_split_across_persisted_count": len(recomputed_fragmented),
        "cross_split_component_count": len(leaking_components),
        "cross_split_seller_count": len(leaking_sellers),
        "cross_split_seller_alias_count": len(leaking_seller_aliases),
        "cross_split_pair_count": len(leaking_pairs),
        "duplicate_pair_uid_count": len(duplicate_pairs),
        "endpoint_component_mismatch_count": endpoint_mismatch_count,
        "non_primary_splits": non_primary_splits,
        "persisted_overmerge_examples": dict(list(sorted(persisted_overmerged.items()))[:25]),
        "recomputed_fragmentation_examples": dict(list(sorted(recomputed_fragmented.items()))[:25]),
        "cross_split_component_examples": sorted(leaking_components)[:25],
        "cross_split_seller_examples": sorted(leaking_sellers)[:25],
        "cross_split_seller_alias_examples": sorted(leaking_seller_aliases)[:25],
        "cross_split_pair_examples": sorted(leaking_pairs)[:25],
        "duplicate_pair_examples": sorted(duplicate_pairs)[:25],
    }
    return summary, normalized_rows, seller_to_component, components


def add_exclusion(
    records: dict[tuple[str, str, str], dict[str, set[str]]],
    *,
    dataset: str,
    entity_type: str,
    entity_id: str,
    split: str,
    pair_uid: str,
    component_id: str,
    reason: str,
    trace_ids: Iterable[str] = (),
) -> None:
    if not entity_id:
        return
    key = (dataset, entity_type, entity_id)
    record = records.setdefault(
        key,
        {
            "source_splits": set(),
            "source_pair_uids": set(),
            "recomputed_component_ids": set(),
            "exclusion_reasons": set(),
            "review_assistance_traces": set(),
        },
    )
    if split:
        record["source_splits"].add(split)
    if pair_uid:
        record["source_pair_uids"].add(pair_uid)
    if component_id:
        record["recomputed_component_ids"].add(component_id)
    record["exclusion_reasons"].add(reason)
    record["review_assistance_traces"].update(trace_ids)


def build_exclusion_manifest(
    datasets: dict[str, list[dict[str, str]]],
    dataset_seller_components: dict[str, dict[str, str]],
    dataset_components: dict[str, dict[str, dict[str, object]]],
    primary_splits: set[str],
    patterns: list[tuple[str, re.Pattern[str]]],
    readiness_rows: list[dict[str, str]],
    readiness_seller_components: dict[str, str],
    readiness_components: dict[str, dict[str, object]],
    input_manifest_sha256: str,
    manifest_version: str,
) -> list[dict[str, object]]:
    records: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for dataset, rows in datasets.items():
        seller_components = dataset_seller_components[dataset]
        for row in rows:
            split = row["split_name"].strip()
            pair_uid = row["pair_uid"].strip()
            left = row["seller_uid_left"].strip()
            right = row["seller_uid_right"].strip()
            component_id = seller_components.get(left, "") if split in primary_splits else ""
            traces = review_trace_ids(row.get("review_notes", ""), patterns)
            add_exclusion(
                records,
                dataset=dataset,
                entity_type="pair_uid",
                entity_id=pair_uid,
                split=split,
                pair_uid=pair_uid,
                component_id=component_id,
                reason="step5_historically_reviewed_pair",
                trace_ids=traces,
            )
            for seller_uid in (left, right):
                add_exclusion(
                    records,
                    dataset=dataset,
                    entity_type="seller_uid",
                    entity_id=seller_uid,
                    split=split,
                    pair_uid=pair_uid,
                    component_id=component_id,
                    reason="step5_historically_observed_seller",
                    trace_ids=traces,
                )
            for seller_alias in (
                portable_seller_alias(row.get("source_seller_raw_left", "")),
                portable_seller_alias(row.get("source_seller_raw_right", "")),
            ):
                add_exclusion(
                    records,
                    dataset=dataset,
                    entity_type="seller_alias",
                    entity_id=seller_alias,
                    split=split,
                    pair_uid=pair_uid,
                    component_id=component_id,
                    reason="step5_historically_observed_seller_alias",
                    trace_ids=traces,
                )
        for component_id, component in dataset_components[dataset].items():
            for pair_uid in component["pair_uids"]:
                add_exclusion(
                    records,
                    dataset=dataset,
                    entity_type="seller_component",
                    entity_id=component_id,
                    split="|".join(sorted(component["splits"])),
                    pair_uid=pair_uid,
                    component_id=component_id,
                    reason="step5_primary_recomputed_component",
                    trace_ids=component["review_trace_ids"],
                )

    for row in readiness_rows:
        split = row.get("_audit_split", "")
        pair_uid = row.get("pair_uid", "").strip()
        left = row.get("seller_uid_left", "").strip()
        right = row.get("seller_uid_right", "").strip()
        component_id = readiness_seller_components.get(left, "")
        for entity_type, entity_id in (
            ("pair_uid", pair_uid),
            ("seller_uid", left),
            ("seller_uid", right),
            ("seller_alias", alias_from_seller_uid(left)),
            ("seller_alias", alias_from_seller_uid(right)),
        ):
            add_exclusion(
                records,
                dataset="zh_target_strict",
                entity_type=entity_type,
                entity_id=entity_id,
                split=f"v8_readiness:{split}",
                pair_uid=pair_uid,
                component_id=component_id,
                reason="v8_readiness_assignment",
            )
    for component_id, component in readiness_components.items():
        for pair_uid in component["pair_uids"]:
            add_exclusion(
                records,
                dataset="zh_target_strict",
                entity_type="seller_component",
                entity_id=component_id,
                split="|".join(f"v8_readiness:{value}" for value in sorted(component["splits"])),
                pair_uid=pair_uid,
                component_id=component_id,
                reason="v8_readiness_recomputed_component",
            )

    output = []
    for (dataset, entity_type, entity_id), record in sorted(records.items()):
        output.append(
            {
                "manifest_version": manifest_version,
                "input_manifest_sha256": input_manifest_sha256,
                "dataset": dataset,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "source_splits": "|".join(sorted(record["source_splits"])),
                "source_pair_count": len(record["source_pair_uids"]),
                "recomputed_component_ids": "|".join(sorted(record["recomputed_component_ids"])),
                "exclusion_reasons": "|".join(sorted(record["exclusion_reasons"])),
                "review_assistance_traces": "|".join(sorted(record["review_assistance_traces"])),
            }
        )
    return output


def validate_output_name(name: str) -> str:
    if not name or Path(name).name != name:
        raise ValueError(f"Step16I output must be a plain filename: {name!r}")
    return name


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = load_json(policy_path)
    primary_splits = {str(value).strip() for value in policy.get("primary_splits", [])}
    if primary_splits != {"train", "valid", "test"}:
        raise ValueError("Step16I primary_splits must be exactly train, valid, and test")
    required_columns = set(policy.get("required_columns", []))
    patterns = compile_review_patterns(policy)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Step16I run-id may contain only letters, digits, underscore, and hyphen")
    outputs_cfg = policy["outputs"]
    final_root = (
        resolve(args.output_root)
        if args.output_root
        else resolve(str(outputs_cfg["root_template"]).format(run_id=run_id))
    )
    ensure_within_workspace(final_root, "Step16I output root")
    staging_root = final_root.with_name(f".{final_root.name}.tmp-{os.getpid()}")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step16I output: {final_root} / {staging_root}"
        )

    datasets: dict[str, list[dict[str, str]]] = {}
    input_records = []
    for dataset, configured_path in policy["inputs"].items():
        path = resolve(configured_path)
        rows, _ = read_csv(path, required_columns)
        datasets[dataset] = rows
        input_records.append(
            {
                "role": f"step5_labels:{dataset}",
                "path": relative_path(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "row_count": len(rows),
                "sha256": file_sha256(path),
            }
        )

    policy_record = {
        "role": "policy",
        "path": relative_path(policy_path),
        "exists": True,
        "size_bytes": policy_path.stat().st_size,
        "sha256": file_sha256(policy_path),
    }
    input_records.append(policy_record)

    readiness_cfg = policy.get("v8_readiness_assignment_check", {})
    readiness_path: Path | None
    if args.skip_v8_readiness_check or not readiness_cfg.get("enabled", False):
        readiness_path = None
    else:
        configured_readiness = args.v8_readiness_assignment or readiness_cfg.get("assignment_path")
        readiness_path = resolve(configured_readiness) if configured_readiness else None
    if readiness_path is not None:
        input_records.append(
            {
                "role": "v8_readiness_assignment",
                "path": relative_path(readiness_path),
                "exists": readiness_path.is_file(),
                "size_bytes": readiness_path.stat().st_size if readiness_path.is_file() else None,
                "sha256": file_sha256(readiness_path) if readiness_path.is_file() else None,
            }
        )

    input_records = sorted(input_records, key=lambda item: (item["role"], item["path"]))
    input_manifest_sha256 = canonical_hash(input_records)

    all_component_rows: list[dict[str, object]] = []
    dataset_summaries: dict[str, object] = {}
    dataset_seller_components: dict[str, dict[str, str]] = {}
    dataset_components: dict[str, dict[str, dict[str, object]]] = {}
    for dataset, rows in datasets.items():
        component_rows, summary, seller_components, components = audit_dataset(
            dataset, rows, primary_splits, patterns
        )
        all_component_rows.extend(component_rows)
        dataset_summaries[dataset] = summary
        dataset_seller_components[dataset] = seller_components
        dataset_components[dataset] = components

    readiness_summary, readiness_rows, readiness_seller_components, readiness_components = (
        audit_v8_readiness(readiness_path, readiness_cfg, primary_splits)
    )

    primary_pair_sets = {
        dataset: {
            row["pair_uid"]
            for row in rows
            if row["split_name"].strip() in primary_splits
        }
        for dataset, rows in datasets.items()
    }
    primary_seller_sets = {
        dataset: {
            seller
            for row in rows
            if row["split_name"].strip() in primary_splits
            for seller in (row["seller_uid_left"].strip(), row["seller_uid_right"].strip())
        }
        for dataset, rows in datasets.items()
    }
    primary_seller_alias_sets = {
        dataset: {
            seller_alias
            for row in rows
            if row["split_name"].strip() in primary_splits
            for seller_alias in (
                portable_seller_alias(row.get("source_seller_raw_left", "")),
                portable_seller_alias(row.get("source_seller_raw_right", "")),
            )
            if seller_alias
        }
        for dataset, rows in datasets.items()
    }
    dataset_names = sorted(datasets)
    cross_dataset = []
    for index, left_name in enumerate(dataset_names):
        for right_name in dataset_names[index + 1 :]:
            pair_overlap = sorted(primary_pair_sets[left_name] & primary_pair_sets[right_name])
            seller_overlap = sorted(primary_seller_sets[left_name] & primary_seller_sets[right_name])
            seller_alias_overlap = sorted(
                primary_seller_alias_sets[left_name] & primary_seller_alias_sets[right_name]
            )
            cross_dataset.append(
                {
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "pair_uid_overlap_count": len(pair_overlap),
                    "seller_uid_overlap_count": len(seller_overlap),
                    "seller_alias_overlap_count": len(seller_alias_overlap),
                    "pair_uid_overlap_examples": pair_overlap[:25],
                    "seller_uid_overlap_examples": seller_overlap[:25],
                    "seller_alias_overlap_examples": seller_alias_overlap[:25],
                }
            )

    exclusion_rows = build_exclusion_manifest(
        datasets,
        dataset_seller_components,
        dataset_components,
        primary_splits,
        patterns,
        readiness_rows,
        readiness_seller_components,
        readiness_components,
        input_manifest_sha256,
        str(policy.get("version", "")),
    )

    component_fields = [
        "dataset",
        "split_name",
        "pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "review_label",
        "old_split_component_id",
        "stored_split_component_size",
        "old_component_observed_pair_count",
        "old_component_recomputed_component_count",
        "recomputed_component_id",
        "recomputed_component_pair_count",
        "recomputed_component_seller_count",
        "recomputed_component_splits",
        "recomputed_component_old_id_count",
        "old_partition_status",
        "cross_split_component_leakage",
        "cross_split_seller_leakage",
        "cross_split_pair_uid_leakage",
        "review_assistance_trace",
    ]
    exclusion_fields = [
        "manifest_version",
        "input_manifest_sha256",
        "dataset",
        "entity_type",
        "entity_id",
        "source_splits",
        "source_pair_count",
        "recomputed_component_ids",
        "exclusion_reasons",
        "review_assistance_traces",
    ]
    component_payload = render_csv(all_component_rows, component_fields)
    exclusion_payload = render_csv(exclusion_rows, exclusion_fields)

    primary_leakage_detected = any(
        bool(summary["leakage"]["detected"]) for summary in dataset_summaries.values()
    ) or any(
        item["pair_uid_overlap_count"]
        or item["seller_uid_overlap_count"]
        or item["seller_alias_overlap_count"]
        for item in cross_dataset
    )
    old_partition_warning = any(
        summary["old_component_comparison"]["recomputed_components_split_across_multiple_old_ids"]
        or summary["old_component_comparison"]["old_components_overmerging_recomputed_components"]
        for summary in dataset_summaries.values()
    )
    review_trace_warning = any(
        summary["review_assistance_traces"]["row_count_with_any_trace"]
        for summary in dataset_summaries.values()
    )
    readiness_failed = readiness_summary.get("passed") is False
    if primary_leakage_detected or readiness_failed:
        status = "fail"
    elif old_partition_warning or review_trace_warning or readiness_summary.get("status") == "archive_missing":
        status = "warning"
    else:
        status = "pass"

    component_name = validate_output_name(outputs_cfg["component_assignments"])
    exclusion_name = validate_output_name(outputs_cfg["permanent_exclusion_manifest"])
    summary_name = validate_output_name(outputs_cfg["summary"])
    summary = {
        "step": "step16i_audit_data_integrity",
        "version": policy.get("version"),
        "run_id": run_id,
        "status": status,
        "read_only_inputs": True,
        "primary_splits": sorted(primary_splits),
        "inputs": input_records,
        "input_manifest_sha256": input_manifest_sha256,
        "datasets": dataset_summaries,
        "cross_dataset_integrity": cross_dataset,
        "v8_readiness_assignment_check": readiness_summary,
        "permanent_exclusion_manifest": {
            "row_count": len(exclusion_rows),
            "entity_type_counts": dict(
                sorted(Counter(str(row["entity_type"]) for row in exclusion_rows).items())
            ),
            "includes_all_step5_splits": True,
            "includes_v8_readiness_when_available": bool(readiness_rows),
            "input_manifest_sha256_embedded_in_each_row": True,
        },
        "findings": {
            "primary_split_leakage_detected": primary_leakage_detected,
            "old_component_partition_warning": old_partition_warning,
            "review_assistance_trace_warning": review_trace_warning,
            "v8_readiness_archive_status": readiness_summary.get("status"),
        },
        "outputs": {
            "component_assignments": {
                "path": f"{relative_path(final_root)}/{component_name}",
                "row_count": len(all_component_rows),
                "sha256": hashlib.sha256(component_payload).hexdigest(),
            },
            "permanent_exclusion_manifest": {
                "path": f"{relative_path(final_root)}/{exclusion_name}",
                "row_count": len(exclusion_rows),
                "sha256": hashlib.sha256(exclusion_payload).hexdigest(),
            },
        },
        "producer": {
            "path": relative_path(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    summary["summary_sha256"] = canonical_hash(summary)
    summary_payload = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()
    try:
        (staging_root / component_name).write_bytes(component_payload)
        (staging_root / exclusion_name).write_bytes(exclusion_payload)
        (staging_root / summary_name).write_bytes(summary_payload)
        staging_root.replace(final_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    print(
        json.dumps(
            {
                "status": status,
                "output_root": relative_path(final_root),
                "component_assignment_rows": len(all_component_rows),
                "permanent_exclusion_rows": len(exclusion_rows),
                "v8_readiness_assignment_status": readiness_summary.get("status"),
                "input_manifest_sha256": input_manifest_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
