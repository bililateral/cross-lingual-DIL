#!/usr/bin/env python3
"""Build score-blind Step15-v8 cross-snapshot identity-control review queues.

The generated candidates are evidence-expert-only controls. They are never
eligible for the primary alias benchmark, and raw platform seller IDs must not
enter clean semantic text or clean pair features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRICT_PROFILES = ROOT / "reports" / "step3_seller_profiles.zh_target_strict.jsonl"
DEFAULT_AUX_PROFILES = ROOT / "reports" / "step3_seller_profiles.zh_target_aux.jsonl"
DEFAULT_PRODUCTS = ROOT / "products_data.csv"
DEFAULT_ASSIGNMENTS = (
    ROOT
    / "reports"
    / "step15_v7"
    / "v2_identifier_redacted_20260714"
    / "splits"
    / "representative_validation_assignments.csv"
)
DEFAULT_STRICT_LABELS = ROOT / "reports" / "step5_zh_target_strict_frozen_silver_labels.csv"
DEFAULT_AUX_LABELS = ROOT / "reports" / "step5_zh_target_aux_frozen_silver_labels.csv"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PAIR_SEPARATOR = "||"
SPLIT_SEED = "step15-v8-identity-control-split-v1-20260715"
COHORT_SEED = "step15-v8-component-cohort-v1-20260715"

DIRECT_RESERVE_MINIMUMS = {"valid": 30, "train": 45}
COMPONENT_RESERVE_MINIMUMS = {"valid": 18, "train": 12}
COMPONENT_FORMAL_MINIMUMS = {"valid": 15, "train": 10}

PRODUCT_REQUIRED_FIELDS = {
    "序号",
    "卖家ID",
    "交易编号",
    "标题",
    "类别",
    "商品描述",
}

SOURCE_EVIDENCE_FIELDS = [
    "candidate_uid",
    "seller_uid_left",
    "seller_uid_right",
    "platform_vendor_id",
    "strict_profile_seller_uid",
    "aux_profile_seller_uid",
    "strict_source_dataset",
    "strict_source_market",
    "aux_source_dataset",
    "aux_source_market",
    "strict_profile_item_count",
    "aux_profile_item_count",
    "exact_shared_title_count",
    "exact_shared_titles",
    "exact_shared_description_count",
    "exact_shared_descriptions",
    "strict_title_preview",
    "strict_description_preview",
    "aux_title_preview",
    "aux_description_preview",
    "cohort_a_item_count",
    "cohort_b_item_count",
    "cohort_a_item_preview",
    "cohort_b_item_preview",
    "same_vendor_path_evidence",
    "strict_profile_jsonl_line_number",
    "aux_profile_jsonl_line_number",
    "strict_profile_record_sha256",
    "aux_profile_record_sha256",
    "source_evidence_sha256",
]

ANSWER_FIELDS = [
    "review_label",
    "evidence_type",
    "review_confidence",
    "review_reason",
    "reviewer_id",
]

REVIEW_PACKET_FIELDS = SOURCE_EVIDENCE_FIELDS + ANSWER_FIELDS

MASTER_FIELDS = [
    *SOURCE_EVIDENCE_FIELDS,
    "candidate_kind",
    "candidate_rule",
    "assigned_split",
    "split_assignment_basis",
    "split_assignment_sha256_rank",
    "existing_split_membership",
    "existing_component_ids",
    "frozen_pair_duplicate_count",
    "evidence_expert_only",
    "primary_alias_benchmark_eligible",
    "clean_semantic_exact_vendor_id_must_be_excluded",
    "clean_feature_exact_vendor_id_must_be_excluded",
    "candidate_record_sha256",
]

COHORT_MANIFEST_FIELDS = [
    "candidate_uid",
    "platform_vendor_id",
    "cohort_id",
    "cohort_seller_uid",
    "item_uid",
    "source_dataset",
    "source_record_index",
    "source_csv_line_number_end",
    "source_sequence",
    "transaction_id",
    "title",
    "category",
    "description",
    "item_record_sha256",
    "manifest_row_sha256",
]

OUTPUT_FILENAMES = {
    "candidate_master": "identity_control_candidate_master.csv",
    "reviewer_a": "reviewer_a_blind_packet.template.csv",
    "reviewer_b": "reviewer_b_blind_packet.template.csv",
    "adjudicator": "reviewer_adjudicator_blind_packet.template.csv",
    "cohort_manifest": "component_cohort_item_manifest.csv",
    "summary": "identity_control_review_summary.json",
}


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def normalize_vendor_id(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def normalize_exact_text(value: object) -> str:
    # Exact overlap permits Unicode canonical composition and outer whitespace
    # removal only. Case, internal whitespace, punctuation, and wording remain.
    return unicodedata.normalize("NFC", str(value or "")).strip()


def clipped(value: object, limit: int = 360) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def joined_preview(values: Iterable[object], *, limit: int = 3, item_limit: int = 220) -> str:
    rendered = []
    for value in values:
        text = clipped(value, item_limit)
        if text and text not in rendered:
            rendered.append(text)
        if len(rendered) >= limit:
            break
    return " || ".join(rendered)


def profile_values(profile: dict, field: str) -> list[str]:
    result = []
    for item in profile.get(field, []):
        if not isinstance(item, dict):
            raise ValueError(f"Profile {field} must contain objects")
        value = normalize_exact_text(item.get("value"))
        if value and value not in result:
            result.append(value)
    return result


def load_jsonl_profiles(path: Path) -> tuple[list[dict], dict]:
    records = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                profile = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL record {path}:{line_number}: {exc}") from exc
            if not isinstance(profile, dict):
                raise ValueError(f"JSONL record is not an object: {path}:{line_number}")
            seller_uid = str(profile.get("seller_uid", "")).strip()
            if not seller_uid:
                raise ValueError(f"Missing seller_uid: {path}:{line_number}")
            records.append(
                {
                    "profile": profile,
                    "line_number": line_number,
                    "raw_record_sha256": sha256_bytes(raw_line.rstrip(b"\r\n")),
                }
            )
    seller_uids = [row["profile"]["seller_uid"] for row in records]
    duplicates = [key for key, count in Counter(seller_uids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate profile seller_uid in {path}: first={duplicates[0]}")
    return records, {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "row_count": len(records),
        "seller_uid_sha256": canonical_hash(sorted(seller_uids)),
    }


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    return rows, fields


def load_products(path: Path) -> tuple[dict[str, list[dict]], dict]:
    products_by_vendor: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = PRODUCT_REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"products_data.csv missing required fields: {sorted(missing)}")
        for record_index, row in enumerate(reader, start=1):
            vendor_id = normalize_vendor_id(row.get("卖家ID"))
            if not vendor_id:
                raise ValueError(f"Empty 卖家ID at products record {record_index}")
            canonical_row = {field: str(row.get(field, "")) for field in reader.fieldnames}
            row_hash = canonical_hash(canonical_row)
            products_by_vendor[vendor_id].append(
                {
                    "row": canonical_row,
                    "record_index": record_index,
                    "csv_line_number_end": reader.line_num,
                    "row_sha256": row_hash,
                }
            )
    return products_by_vendor, {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "row_count": sum(len(value) for value in products_by_vendor.values()),
        "vendor_count": len(products_by_vendor),
        "vendor_item_counts_sha256": canonical_hash(
            {key: len(value) for key, value in sorted(products_by_vendor.items())}
        ),
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def load_split_state(assignments_path: Path, strict_labels_path: Path) -> tuple[dict, dict]:
    assignments, assignment_fields = load_csv(assignments_path)
    required_assignment_fields = {
        "pair_uid",
        "v7_component_id",
        "seller_uid_left",
        "seller_uid_right",
        "v7_split_name",
    }
    if not required_assignment_fields.issubset(assignment_fields):
        raise ValueError(
            f"Representative assignments missing fields: "
            f"{sorted(required_assignment_fields - set(assignment_fields))}"
        )
    allowed_splits = {"train", "valid", "internal_development_test"}
    unknown_splits = sorted({row["v7_split_name"] for row in assignments} - allowed_splits)
    if unknown_splits:
        raise ValueError(f"Unexpected representative split names: {unknown_splits}")

    strict_labels, strict_label_fields = load_csv(strict_labels_path)
    required_label_fields = {
        "pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "review_label",
        "usable_for_supervision",
    }
    if not required_label_fields.issubset(strict_label_fields):
        raise ValueError(
            f"Strict frozen labels missing fields: "
            f"{sorted(required_label_fields - set(strict_label_fields))}"
        )

    union_find = UnionFind()
    for row in assignments:
        union_find.union(row["seller_uid_left"], row["seller_uid_right"])
    for row in strict_labels:
        if row.get("usable_for_supervision") == "1" and row.get("review_label") in {
            "positive",
            "negative",
        }:
            union_find.union(row["seller_uid_left"], row["seller_uid_right"])

    component_splits: dict[str, set[str]] = defaultdict(set)
    component_ids: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        for seller_uid in (row["seller_uid_left"], row["seller_uid_right"]):
            root = union_find.find(seller_uid)
            component_splits[root].add(row["v7_split_name"])
            component_ids[root].add(row["v7_component_id"])
    conflicts = {
        root: sorted(splits) for root, splits in component_splits.items() if len(splits) > 1
    }
    if conflicts:
        first = next(iter(conflicts.items()))
        raise ValueError(
            "Existing supervision has a seller component across representative splits; "
            f"count={len(conflicts)} first={first}"
        )

    seller_state = {}
    for seller_uid in sorted(union_find.parent):
        root = union_find.find(seller_uid)
        seller_state[seller_uid] = {
            "component_root": root,
            "splits": set(component_splits.get(root, set())),
            "component_ids": set(component_ids.get(root, set())),
        }
    frozen_pair_counts = Counter(
        canonical_pair_uid(row["seller_uid_left"], row["seller_uid_right"])
        for row in strict_labels
        if row.get("seller_uid_left") and row.get("seller_uid_right")
    )
    return seller_state, {
        "assignments": assignments,
        "strict_labels": strict_labels,
        "frozen_pair_counts": frozen_pair_counts,
        "assignment_input": {
            "path": relative_path(assignments_path),
            "sha256": sha256_file(assignments_path),
            "row_count": len(assignments),
            "pair_uid_sha256": canonical_hash(sorted(row["pair_uid"] for row in assignments)),
        },
        "strict_labels_input": {
            "path": relative_path(strict_labels_path),
            "sha256": sha256_file(strict_labels_path),
            "row_count": len(strict_labels),
            "pair_uid_sha256": canonical_hash(sorted(row["pair_uid"] for row in strict_labels)),
        },
    }


def canonical_pair_uid(left: str, right: str) -> str:
    first, second = sorted((str(left), str(right)))
    return f"{first}{PAIR_SEPARATOR}{second}"


def unique_vendor_index(
    profile_records: list[dict],
    vendor_field: str,
    source_name: str,
) -> tuple[dict[str, dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    missing_count = 0
    for record in profile_records:
        vendor_id = normalize_vendor_id(record["profile"].get(vendor_field))
        if not vendor_id:
            missing_count += 1
            continue
        grouped[vendor_id].append(record)
    ambiguous = {key: value for key, value in grouped.items() if len(value) != 1}
    unique = {key: value[0] for key, value in grouped.items() if len(value) == 1}
    return unique, {
        "source": source_name,
        "vendor_field": vendor_field,
        "unique_vendor_count": len(unique),
        "ambiguous_vendor_count": len(ambiguous),
        "ambiguous_vendor_ids_sha256": canonical_hash(sorted(ambiguous)),
        "missing_vendor_id_profile_count": missing_count,
    }


def exact_overlap(strict_profile: dict, aux_profile: dict) -> tuple[list[str], list[str]]:
    strict_titles = set(profile_values(strict_profile, "top_titles"))
    aux_titles = set(profile_values(aux_profile, "top_titles"))
    strict_descriptions = set(profile_values(strict_profile, "top_description_snippets"))
    aux_descriptions = set(profile_values(aux_profile, "top_description_snippets"))
    return sorted(strict_titles & aux_titles), sorted(strict_descriptions & aux_descriptions)


def seller_membership(candidate_sellers: Iterable[str], seller_state: dict) -> tuple[set[str], set[str]]:
    splits: set[str] = set()
    components: set[str] = set()
    for seller_uid in candidate_sellers:
        state = seller_state.get(seller_uid)
        if not state:
            continue
        splits.update(state["splits"])
        components.update(state["component_ids"])
    return splits, components


def split_rank(namespace: str, vendor_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}|{namespace}|{vendor_id}".encode("utf-8")).hexdigest()


def assign_vendor_splits(candidates: list[dict], minimums: dict[str, int], namespace: str) -> dict:
    assigned_counts = Counter()
    unseen = []
    for row in candidates:
        membership = set(row["_existing_split_membership"])
        if membership == {"valid"}:
            split = "valid"
            basis = "existing_valid_only"
        elif membership == {"train"}:
            split = "train"
            basis = "existing_train_only"
        elif not membership:
            unseen.append(row)
            continue
        else:
            raise ValueError(
                f"Candidate has an invalid existing split membership: "
                f"vendor={row['platform_vendor_id']} membership={sorted(membership)}"
            )
        row["assigned_split"] = split
        row["split_assignment_basis"] = basis
        row["split_assignment_sha256_rank"] = split_rank(namespace, row["platform_vendor_id"])
        assigned_counts[split] += 1

    unseen.sort(key=lambda row: split_rank(namespace, row["platform_vendor_id"]))
    cursor = 0
    for split in ("valid", "train"):
        deficit = max(0, int(minimums[split]) - assigned_counts[split])
        if cursor + deficit > len(unseen):
            raise ValueError(
                f"Insufficient unseen vendors for {namespace} {split} reserve: "
                f"required={minimums[split]} forced={assigned_counts[split]} "
                f"remaining_unseen={len(unseen) - cursor}"
            )
        for row in unseen[cursor : cursor + deficit]:
            row["assigned_split"] = split
            row["split_assignment_basis"] = f"unseen_sha256_quota_{split}"
            row["split_assignment_sha256_rank"] = split_rank(
                namespace, row["platform_vendor_id"]
            )
            assigned_counts[split] += 1
        cursor += deficit

    # Publish the whole evidence reserve. Remaining unseen vendors use a fixed
    # 1:4 valid/train hash bucket after the explicit minimums are secured.
    for row in unseen[cursor:]:
        rank = split_rank(namespace, row["platform_vendor_id"])
        split = "valid" if int(rank[:8], 16) % 5 == 0 else "train"
        row["assigned_split"] = split
        row["split_assignment_basis"] = "unseen_sha256_bucket_1_of_5_valid"
        row["split_assignment_sha256_rank"] = rank
        assigned_counts[split] += 1

    unmet = {
        split: {"required": target, "observed": assigned_counts[split]}
        for split, target in minimums.items()
        if assigned_counts[split] < target
    }
    if unmet:
        raise ValueError(f"Candidate split reserves were not met for {namespace}: {unmet}")
    return {
        "minimums": dict(minimums),
        "observed": dict(sorted(assigned_counts.items())),
        "unseen_vendor_count": len(unseen),
        "assignment_seed": SPLIT_SEED,
        "assignment_method": "forced_existing_membership_then_sha256_quota_then_fixed_1_of_5_valid_bucket",
    }


def profile_preview(profile: dict, field: str) -> str:
    return joined_preview((item.get("value", "") for item in profile.get(field, [])))


def cohort_partition(vendor_id: str, product_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(product_rows) < 4:
        raise ValueError(f"Component cohort vendor has fewer than four items: {vendor_id}")
    ranked = sorted(
        product_rows,
        key=lambda item: hashlib.sha256(
            f"{COHORT_SEED}|{vendor_id}|{item['record_index']}|{item['row_sha256']}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )
    cohort_a = ranked[::2]
    cohort_b = ranked[1::2]
    if len(cohort_a) < 2 or len(cohort_b) < 2:
        raise ValueError(f"Deterministic cohort partition is undersized: {vendor_id}")
    ids_a = {item["record_index"] for item in cohort_a}
    ids_b = {item["record_index"] for item in cohort_b}
    if ids_a & ids_b or ids_a | ids_b != {item["record_index"] for item in product_rows}:
        raise AssertionError(f"Cohort partition is not a disjoint complete cover: {vendor_id}")
    return cohort_a, cohort_b


def cohort_item_preview(items: list[dict]) -> str:
    return " || ".join(
        f"row={item['record_index']}; transaction={clipped(item['row'].get('交易编号'), 40)}; "
        f"title={clipped(item['row'].get('标题'), 120)}; "
        f"description={clipped(item['row'].get('商品描述'), 180)}"
        for item in items[:3]
    )


def build_source_evidence(
    *,
    candidate_uid: str,
    seller_uid_left: str,
    seller_uid_right: str,
    vendor_id: str,
    strict_record: dict,
    aux_record: dict,
    shared_titles: list[str],
    shared_descriptions: list[str],
    cohort_a: list[dict] | None,
    cohort_b: list[dict] | None,
) -> dict:
    strict_profile = strict_record["profile"]
    aux_profile = aux_record["profile"]
    is_cohort = cohort_a is not None and cohort_b is not None
    evidence = {
        "candidate_uid": candidate_uid,
        "seller_uid_left": seller_uid_left,
        "seller_uid_right": seller_uid_right,
        "platform_vendor_id": vendor_id,
        "strict_profile_seller_uid": strict_profile["seller_uid"],
        "aux_profile_seller_uid": aux_profile["seller_uid"],
        "strict_source_dataset": strict_profile.get("source_dataset", ""),
        "strict_source_market": strict_profile.get("source_market_raw", ""),
        "aux_source_dataset": aux_profile.get("source_dataset", ""),
        "aux_source_market": aux_profile.get("source_market_raw", ""),
        "strict_profile_item_count": str(strict_profile.get("item_count", "")),
        "aux_profile_item_count": str(aux_profile.get("item_count", "")),
        "exact_shared_title_count": str(len(shared_titles)),
        "exact_shared_titles": " || ".join(shared_titles),
        "exact_shared_description_count": str(len(shared_descriptions)),
        "exact_shared_descriptions": " || ".join(shared_descriptions),
        "strict_title_preview": profile_preview(strict_profile, "top_titles"),
        "strict_description_preview": profile_preview(
            strict_profile, "top_description_snippets"
        ),
        "aux_title_preview": profile_preview(aux_profile, "top_titles"),
        "aux_description_preview": profile_preview(aux_profile, "top_description_snippets"),
        "cohort_a_item_count": str(len(cohort_a or [])),
        "cohort_b_item_count": str(len(cohort_b or [])),
        "cohort_a_item_preview": cohort_item_preview(cohort_a or []),
        "cohort_b_item_preview": cohort_item_preview(cohort_b or []),
        "same_vendor_path_evidence": (
            f"strict_profile_vendor_id={vendor_id} -> aux_profile_vendor_id={vendor_id} -> "
            "deterministic_nonoverlapping_products_cohort_A/B"
            if is_cohort
            else f"strict_profile_vendor_id={vendor_id} -> aux_profile_vendor_id={vendor_id}"
        ),
        "strict_profile_jsonl_line_number": str(strict_record["line_number"]),
        "aux_profile_jsonl_line_number": str(aux_record["line_number"]),
        "strict_profile_record_sha256": strict_record["raw_record_sha256"],
        "aux_profile_record_sha256": aux_record["raw_record_sha256"],
    }
    evidence["source_evidence_sha256"] = canonical_hash(evidence)
    if set(evidence) != set(SOURCE_EVIDENCE_FIELDS):
        raise AssertionError("Source evidence fields differ from the fixed schema")
    return evidence


def component_candidate_uids(vendor_id: str) -> tuple[str, str]:
    return (
        f"products_data.csv|identity_control|seller_id:{vendor_id}|cohort:A",
        f"products_data.csv|identity_control|seller_id:{vendor_id}|cohort:B",
    )


def candidate_uid(kind: str, vendor_id: str, strict_hash: str, aux_hash: str) -> str:
    digest = canonical_hash(
        {
            "kind": kind,
            "vendor_id": vendor_id,
            "strict_profile_record_sha256": strict_hash,
            "aux_profile_record_sha256": aux_hash,
        }
    )
    return f"v8_identity_control_{kind}_{digest[:24]}"


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            missing = sorted(set(fields) - set(row))
            extra = sorted(set(row) - set(fields))
            raise ValueError(f"CSV row violates fixed fields: missing={missing} extra={extra}")
        writer.writerow(row)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def validate_review_packet_schema() -> None:
    forbidden_exact = {
        "candidate_kind",
        "candidate_rule",
        "assigned_split",
        "split_name",
        "old_label",
        "model_score",
        "prediction_score",
        "test_membership",
    }
    overlap = forbidden_exact & set(REVIEW_PACKET_FIELDS)
    if overlap:
        raise AssertionError(f"Blind packet leaks forbidden fields: {sorted(overlap)}")
    for field in REVIEW_PACKET_FIELDS:
        lowered = field.lower()
        if any(token in lowered for token in ("model_score", "prediction", "test_split")):
            raise AssertionError(f"Blind packet field leaks model/test information: {field}")


def packet_row(master_row: dict) -> dict:
    packet = {field: master_row[field] for field in SOURCE_EVIDENCE_FIELDS}
    packet.update({field: "" for field in ANSWER_FIELDS})
    if set(packet) != set(REVIEW_PACKET_FIELDS):
        raise AssertionError("Reviewer packet fields differ from the fixed schema")
    return packet


def write_atomic_file(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite staged artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary artifact: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def count_by(rows: list[dict], *keys: str) -> dict:
    counter = Counter(tuple(row[key] for key in keys) for row in rows)
    return {"|".join(key): value for key, value in sorted(counter.items())}


def build(args: argparse.Namespace) -> tuple[dict[str, bytes], dict]:
    input_paths = {
        "strict_profiles": resolve(args.strict_profiles),
        "aux_profiles": resolve(args.aux_profiles),
        "products_data": resolve(args.products_data),
        "representative_assignments": resolve(args.assignments),
        "strict_frozen_labels": resolve(args.strict_frozen_labels),
        "aux_frozen_labels": resolve(args.aux_frozen_labels),
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing identity-control inputs: {missing_inputs}")

    strict_records, strict_input = load_jsonl_profiles(input_paths["strict_profiles"])
    aux_records, aux_input = load_jsonl_profiles(input_paths["aux_profiles"])
    products_by_vendor, products_input = load_products(input_paths["products_data"])
    seller_state, split_inputs = load_split_state(
        input_paths["representative_assignments"], input_paths["strict_frozen_labels"]
    )
    aux_labels, _ = load_csv(input_paths["aux_frozen_labels"])
    aux_labels_input = {
        "path": relative_path(input_paths["aux_frozen_labels"]),
        "sha256": sha256_file(input_paths["aux_frozen_labels"]),
        "row_count": len(aux_labels),
        "pair_uid_sha256": canonical_hash(sorted(row.get("pair_uid", "") for row in aux_labels)),
    }

    strict_by_vendor, strict_vendor_diagnostics = unique_vendor_index(
        strict_records, "source_seller_raw", "zh_target_strict"
    )
    aux_by_vendor, aux_vendor_diagnostics = unique_vendor_index(
        aux_records, "source_seller_id_raw", "zh_target_aux"
    )

    profile_item_mismatches = []
    for vendor_id, aux_record in aux_by_vendor.items():
        expected = int(aux_record["profile"].get("item_count", 0))
        observed = len(products_by_vendor.get(vendor_id, []))
        if expected != observed:
            profile_item_mismatches.append((vendor_id, expected, observed))
    if profile_item_mismatches:
        raise ValueError(
            "Aux profile item_count does not match products_data.csv; "
            f"first={profile_item_mismatches[0]}"
        )

    candidate_pool = []
    exclusion_counts = Counter()
    exclusion_vendor_ids: dict[str, list[str]] = defaultdict(list)
    for vendor_id in sorted(set(strict_by_vendor) & set(aux_by_vendor)):
        strict_record = strict_by_vendor[vendor_id]
        aux_record = aux_by_vendor[vendor_id]
        strict_profile = strict_record["profile"]
        aux_profile = aux_record["profile"]
        shared_titles, shared_descriptions = exact_overlap(strict_profile, aux_profile)
        if not shared_titles and not shared_descriptions:
            exclusion_counts["no_exact_title_or_description_overlap"] += 1
            exclusion_vendor_ids["no_exact_title_or_description_overlap"].append(vendor_id)
            continue
        strict_uid = strict_profile["seller_uid"]
        aux_uid = aux_profile["seller_uid"]
        memberships, component_ids = seller_membership((strict_uid, aux_uid), seller_state)
        if "internal_development_test" in memberships:
            exclusion_counts["touches_fixed_internal_development_test_component"] += 1
            exclusion_vendor_ids["touches_fixed_internal_development_test_component"].append(
                vendor_id
            )
            continue
        if len(memberships) > 1:
            exclusion_counts["existing_train_valid_component_conflict"] += 1
            exclusion_vendor_ids["existing_train_valid_component_conflict"].append(vendor_id)
            continue
        candidate_pool.append(
            {
                "platform_vendor_id": vendor_id,
                "strict_record": strict_record,
                "aux_record": aux_record,
                "shared_titles": shared_titles,
                "shared_descriptions": shared_descriptions,
                "_existing_split_membership": memberships,
                "_existing_component_ids": component_ids,
                "_product_rows": products_by_vendor.get(vendor_id, []),
            }
        )

    component_candidates = [row for row in candidate_pool if len(row["_product_rows"]) >= 4]
    component_vendor_ids = {row["platform_vendor_id"] for row in component_candidates}
    direct_candidates = [
        row for row in candidate_pool if row["platform_vendor_id"] not in component_vendor_ids
    ]
    if component_vendor_ids & {row["platform_vendor_id"] for row in direct_candidates}:
        raise AssertionError("Direct and component raw vendor IDs overlap")

    component_assignment = assign_vendor_splits(
        component_candidates, COMPONENT_RESERVE_MINIMUMS, "component_closure_control"
    )
    direct_assignment = assign_vendor_splits(
        direct_candidates, DIRECT_RESERVE_MINIMUMS, "direct_persistence_control"
    )

    frozen_pair_counts = split_inputs["frozen_pair_counts"]
    master_rows = []
    cohort_manifest_rows = []
    for row in sorted(
        direct_candidates + component_candidates,
        key=lambda item: (
            item["platform_vendor_id"],
            0 if item in direct_candidates else 1,
        ),
    ):
        vendor_id = row["platform_vendor_id"]
        strict_record = row["strict_record"]
        aux_record = row["aux_record"]
        strict_uid = strict_record["profile"]["seller_uid"]
        aux_uid = aux_record["profile"]["seller_uid"]
        is_component = vendor_id in component_vendor_ids
        kind = "component_closure" if is_component else "direct_persistence"
        uid = candidate_uid(
            kind,
            vendor_id,
            strict_record["raw_record_sha256"],
            aux_record["raw_record_sha256"],
        )
        cohort_a = cohort_b = None
        if is_component:
            seller_left, seller_right = component_candidate_uids(vendor_id)
            cohort_a, cohort_b = cohort_partition(vendor_id, row["_product_rows"])
        else:
            seller_left, seller_right = strict_uid, aux_uid
        evidence = build_source_evidence(
            candidate_uid=uid,
            seller_uid_left=seller_left,
            seller_uid_right=seller_right,
            vendor_id=vendor_id,
            strict_record=strict_record,
            aux_record=aux_record,
            shared_titles=row["shared_titles"],
            shared_descriptions=row["shared_descriptions"],
            cohort_a=cohort_a,
            cohort_b=cohort_b,
        )
        pair_uid = canonical_pair_uid(seller_left, seller_right)
        master = {
            **evidence,
            "candidate_kind": (
                "evidence_expert_component_closure_control"
                if is_component
                else "evidence_expert_direct_persistence_control"
            ),
            "candidate_rule": (
                "same_raw_vendor_id_exact_profile_overlap_products_item_count_ge4_"
                "deterministic_nonoverlapping_cohorts"
                if is_component
                else "same_raw_vendor_id_and_exact_shared_profile_title_or_description"
            ),
            "assigned_split": row["assigned_split"],
            "split_assignment_basis": row["split_assignment_basis"],
            "split_assignment_sha256_rank": row["split_assignment_sha256_rank"],
            "existing_split_membership": "|".join(sorted(row["_existing_split_membership"])),
            "existing_component_ids": "|".join(sorted(row["_existing_component_ids"])),
            "frozen_pair_duplicate_count": str(frozen_pair_counts.get(pair_uid, 0)),
            "evidence_expert_only": "1",
            "primary_alias_benchmark_eligible": "0",
            "clean_semantic_exact_vendor_id_must_be_excluded": "1",
            "clean_feature_exact_vendor_id_must_be_excluded": "1",
        }
        if master["frozen_pair_duplicate_count"] != "0":
            raise ValueError(f"Identity-control candidate duplicates a frozen pair: {pair_uid}")
        master["candidate_record_sha256"] = canonical_hash(master)
        if set(master) != set(MASTER_FIELDS):
            raise AssertionError("Candidate master fields differ from the fixed schema")
        master_rows.append(master)

        if is_component:
            assert cohort_a is not None and cohort_b is not None
            for cohort_id, cohort_uid, items in (
                ("A", seller_left, cohort_a),
                ("B", seller_right, cohort_b),
            ):
                for item in items:
                    source_row = item["row"]
                    item_uid = (
                        f"products_data.csv|record:{item['record_index']}|"
                        f"sha256:{item['row_sha256'][:20]}"
                    )
                    manifest_row = {
                        "candidate_uid": uid,
                        "platform_vendor_id": vendor_id,
                        "cohort_id": cohort_id,
                        "cohort_seller_uid": cohort_uid,
                        "item_uid": item_uid,
                        "source_dataset": "products_data.csv",
                        "source_record_index": str(item["record_index"]),
                        "source_csv_line_number_end": str(item["csv_line_number_end"]),
                        "source_sequence": source_row.get("序号", ""),
                        "transaction_id": source_row.get("交易编号", ""),
                        "title": source_row.get("标题", ""),
                        "category": source_row.get("类别", ""),
                        "description": source_row.get("商品描述", ""),
                        "item_record_sha256": item["row_sha256"],
                    }
                    manifest_row["manifest_row_sha256"] = canonical_hash(manifest_row)
                    if set(manifest_row) != set(COHORT_MANIFEST_FIELDS):
                        raise AssertionError("Cohort manifest fields differ from the fixed schema")
                    cohort_manifest_rows.append(manifest_row)

    candidate_uids = [row["candidate_uid"] for row in master_rows]
    if len(candidate_uids) != len(set(candidate_uids)):
        raise ValueError("Candidate UID collision detected")
    vendor_rows: dict[str, list[dict]] = defaultdict(list)
    for row in master_rows:
        vendor_rows[row["platform_vendor_id"]].append(row)
    split_violations = {
        vendor: sorted({row["assigned_split"] for row in rows})
        for vendor, rows in vendor_rows.items()
        if len({row["assigned_split"] for row in rows}) != 1
    }
    if split_violations:
        raise ValueError(f"Derived nodes from one vendor cross splits: {split_violations}")
    if any(
        "internal_development_test" in row["existing_split_membership"] for row in master_rows
    ):
        raise AssertionError("Published candidate touches the fixed internal development test")

    component_counts = Counter(
        row["assigned_split"]
        for row in master_rows
        if row["candidate_kind"] == "evidence_expert_component_closure_control"
    )
    formal_component_unmet = {
        split: {"required": target, "observed": component_counts[split]}
        for split, target in COMPONENT_FORMAL_MINIMUMS.items()
        if component_counts[split] < target
    }
    if formal_component_unmet:
        raise ValueError(f"Formal component-control minimums were not met: {formal_component_unmet}")

    master_rows.sort(key=lambda row: (row["candidate_kind"], row["candidate_uid"]))
    packet_rows = [packet_row(row) for row in master_rows]
    cohort_manifest_rows.sort(
        key=lambda row: (
            row["candidate_uid"],
            row["cohort_id"],
            int(row["source_record_index"]),
        )
    )
    validate_review_packet_schema()

    payloads = {
        "candidate_master": render_csv(master_rows, MASTER_FIELDS),
        "reviewer_a": render_csv(packet_rows, REVIEW_PACKET_FIELDS),
        "reviewer_b": render_csv(packet_rows, REVIEW_PACKET_FIELDS),
        "adjudicator": render_csv(packet_rows, REVIEW_PACKET_FIELDS),
        "cohort_manifest": render_csv(cohort_manifest_rows, COHORT_MANIFEST_FIELDS),
    }

    producer_path = Path(__file__).resolve()
    summary = {
        "step": "step16_build_v8_identity_control_queues",
        "version": "2026-07-15-step15-v8-cross-snapshot-identity-controls-v1",
        "run_id": args.run_id,
        "status": "pass",
        "selection_is_model_score_blind": True,
        "model_score_inputs_read": False,
        "fixed_internal_development_test_changed": False,
        "fixed_internal_development_test_candidate_count": 0,
        "seller_component_split_leakage_count": 0,
        "candidate_purpose": "evidence_expert_only_controls",
        "primary_alias_benchmark_eligible": False,
        "review_labels_materialized": False,
        "review_packet_answer_fields_are_blank": True,
        "clean_semantic_contract": {
            "exact_platform_vendor_id_allowed_in_review_evidence": True,
            "exact_platform_vendor_id_allowed_in_clean_semantic_text": False,
            "exact_platform_vendor_id_allowed_in_clean_pair_features": False,
            "raw_profile_text_exported_to_clean_features": False,
            "candidate_controls_may_enter_primary_alias_benchmark": False,
        },
        "candidate_definition": {
            "direct_persistence": (
                "same exact platform vendor ID across strict and auxiliary snapshots plus at least "
                "one exact profile-level title or description overlap"
            ),
            "component_closure": (
                "same exact platform vendor ID path plus deterministic disjoint products_data item "
                "cohorts; auxiliary item_count >= 4 and each cohort has at least two source rows"
            ),
            "exact_text_normalization": "Unicode NFC plus outer-strip only",
            "direct_component_raw_vendor_ids_are_disjoint": True,
        },
        "split_contract": {
            "fixed_seed": SPLIT_SEED,
            "test_components_excluded": True,
            "existing_valid_sellers_may_only_generate_valid_candidates": True,
            "existing_train_sellers_may_only_generate_train_candidates": True,
            "unseen_vendor_assignment_is_sha256_only": True,
            "all_nodes_derived_from_one_raw_vendor_share_one_split": True,
        },
        "cohort_contract": {
            "fixed_seed": COHORT_SEED,
            "minimum_aux_item_count": 4,
            "minimum_items_per_side": 2,
            "cohorts_are_source_row_disjoint": True,
            "cohorts_form_complete_vendor_item_cover": True,
        },
        "reserve_thresholds": {
            "direct_persistence": DIRECT_RESERVE_MINIMUMS,
            "component_closure_output": COMPONENT_RESERVE_MINIMUMS,
            "component_closure_formal_readiness_floor": COMPONENT_FORMAL_MINIMUMS,
        },
        "reserve_results": {
            "direct_persistence": direct_assignment,
            "component_closure": component_assignment,
            "component_formal_minimums_met": not formal_component_unmet,
        },
        "counts": {
            "candidate_total": len(master_rows),
            "by_kind_and_split": count_by(master_rows, "candidate_kind", "assigned_split"),
            "by_kind_membership_and_split": count_by(
                master_rows,
                "candidate_kind",
                "existing_split_membership",
                "assigned_split",
            ),
            "direct_vendor_count": len({row["platform_vendor_id"] for row in direct_candidates}),
            "component_vendor_count": len(component_vendor_ids),
            "cohort_manifest_item_count": len(cohort_manifest_rows),
            "excluded_vendor_counts": dict(sorted(exclusion_counts.items())),
        },
        "excluded_vendor_id_hashes": {
            reason: canonical_hash(sorted(vendors))
            for reason, vendors in sorted(exclusion_vendor_ids.items())
        },
        "profile_vendor_index_diagnostics": {
            "strict": strict_vendor_diagnostics,
            "aux": aux_vendor_diagnostics,
        },
        "provenance": {
            "producer": relative_path(producer_path),
            "producer_sha256": sha256_file(producer_path),
            "inputs": {
                "strict_profiles": strict_input,
                "aux_profiles": aux_input,
                "products_data": products_input,
                "representative_assignments": split_inputs["assignment_input"],
                "strict_frozen_labels": split_inputs["strict_labels_input"],
                "aux_frozen_labels": aux_labels_input,
            },
            "candidate_uid_sha256": canonical_hash(sorted(candidate_uids)),
            "raw_vendor_assignment_sha256": canonical_hash(
                sorted(
                    (row["platform_vendor_id"], row["candidate_kind"], row["assigned_split"])
                    for row in master_rows
                )
            ),
            "cohort_partition_sha256": canonical_hash(
                sorted(
                    (
                        row["platform_vendor_id"],
                        row["cohort_id"],
                        row["item_uid"],
                        row["item_record_sha256"],
                    )
                    for row in cohort_manifest_rows
                )
            ),
        },
        "fixed_schemas": {
            "candidate_master_fields": MASTER_FIELDS,
            "review_packet_fields": REVIEW_PACKET_FIELDS,
            "review_packet_source_evidence_fields": SOURCE_EVIDENCE_FIELDS,
            "review_packet_answer_fields": ANSWER_FIELDS,
            "cohort_manifest_fields": COHORT_MANIFEST_FIELDS,
        },
        "artifacts": {
            key: {
                "filename": OUTPUT_FILENAMES[key],
                "sha256": sha256_bytes(payload),
                "byte_count": len(payload),
            }
            for key, payload in sorted(payloads.items())
        },
    }
    summary["summary_self_sha256"] = canonical_hash(summary)
    summary_payload = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    payloads["summary"] = summary_payload
    return payloads, summary


def publish(payloads: dict[str, bytes], final_root: Path) -> None:
    staging_root = final_root.with_name(f".{final_root.name}.incomplete")
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite identity-control review publication: "
            f"{final_root} / {staging_root}"
        )
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(exist_ok=False)
    try:
        for key, payload in payloads.items():
            write_atomic_file(staging_root / OUTPUT_FILENAMES[key], payload)
        for key, payload in payloads.items():
            staged_path = staging_root / OUTPUT_FILENAMES[key]
            observed = sha256_file(staged_path)
            expected = sha256_bytes(payload)
            if observed != expected:
                raise ValueError(
                    f"Staged artifact hash mismatch: {staged_path} "
                    f"expected={expected} observed={observed}"
                )
        staging_root.replace(final_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strict-profiles", default=str(DEFAULT_STRICT_PROFILES))
    parser.add_argument("--aux-profiles", default=str(DEFAULT_AUX_PROFILES))
    parser.add_argument("--products-data", default=str(DEFAULT_PRODUCTS))
    parser.add_argument("--assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--strict-frozen-labels", default=str(DEFAULT_STRICT_LABELS))
    parser.add_argument("--aux-frozen-labels", default=str(DEFAULT_AUX_LABELS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError(
            "run-id must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ and cannot contain paths"
        )
    return args


def main() -> None:
    args = parse_args()
    payloads, summary = build(args)
    final_root = ROOT / "reports" / "step15_v8" / args.run_id / "identity_control_review"
    if not args.validate_only:
        publish(payloads, final_root)
    result = {
        "status": "pass",
        "mode": "validate_only" if args.validate_only else "published",
        "output_root": relative_path(final_root),
        "candidate_total": summary["counts"]["candidate_total"],
        "by_kind_and_split": summary["counts"]["by_kind_and_split"],
        "reserve_results": summary["reserve_results"],
        "summary_self_sha256": summary["summary_self_sha256"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
