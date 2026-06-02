from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_en_source_expansion_policy.json"

TARGET_EXTRA_FIELDS = [
    "en_source_expansion_rank",
    "target_bucket",
    "target_label_hint",
    "target_action",
    "source_existing_active_bool",
    "source_existing_step4_bool",
    "seller_uid_left",
    "seller_uid_right",
    "shared_contact_count",
    "shared_title_count",
    "shared_description_count",
    "shared_pgp_fingerprint_count",
    "review_rubric_hint",
]

TRUST_SUFFIX_RE = re.compile(r"\s*\(\d+%\)\s*$", re.I)
NON_ALIAS_CHARS_RE = re.compile(r"[^0-9a-z\u3400-\u9fff/]+", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an English source-domain Step 5 expansion queue from pending high-evidence candidates."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the English source expansion policy JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def normalize_label(value: object) -> str:
    return normalize_text(value).lower()


def normalize_soft_alias(value: object) -> str:
    text = normalize_text(value).casefold()
    text = TRUST_SUFFIX_RE.sub("", text)
    text = NON_ALIAS_CHARS_RE.sub("", text)
    return text


def is_soft_same_alias(row: dict) -> bool:
    left = normalize_soft_alias(row.get("source_seller_raw_left"))
    right = normalize_soft_alias(row.get("source_seller_raw_right"))
    return bool(left and right and left == right)


def to_float(value: object, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def to_int(value: object) -> int:
    return int(round(to_float(value, 0.0)))


def is_pending(row: dict) -> bool:
    return normalize_text(row.get("review_status")).lower() in {"", "pending"} and not normalize_label(
        row.get("review_label")
    )


def queue_sort_key(row: dict) -> tuple:
    return (
        -to_float(row.get("candidate_rank_score")),
        -to_float(row.get("lexical_similarity")),
        -to_float(row.get("structural_support_score")),
        int(row.get("balanced_review_rank", "0") or 0),
        row.get("pair_uid", ""),
    )


def row_with_candidate_counts(row: dict, candidate: dict | None) -> dict:
    enriched = dict(row)
    if candidate:
        for key in ("seller_uid_left", "seller_uid_right", "candidate_scope"):
            enriched[key] = candidate.get(key, enriched.get(key, ""))
        for key in (
            "shared_contact_count",
            "shared_title_count",
            "shared_description_count",
            "shared_pgp_fingerprint_count",
        ):
            enriched[key] = candidate.get(key, enriched.get(key, "0"))
    else:
        for key in (
            "shared_contact_count",
            "shared_title_count",
            "shared_description_count",
            "shared_pgp_fingerprint_count",
        ):
            enriched.setdefault(key, "0")
    return enriched


def bucket_candidates(rows: list[dict], bucket_cfg: dict) -> list[dict]:
    strata = set(bucket_cfg["source_strata"])
    candidates = [row for row in rows if row.get("review_stratum") in strata]
    bucket_id = bucket_cfg["bucket_id"]

    if bucket_id in {"seller_facing_identifier_plus_text", "seller_facing_identifier_primary"}:
        candidates = [
            row
            for row in candidates
            if to_int(row.get("shared_contact_count")) > 0 or to_int(row.get("shared_pgp_fingerprint_count")) > 0
        ]
    elif bucket_id == "strong_text_clone_positive_probe":
        candidates = [
            row
            for row in candidates
            if to_int(row.get("shared_title_count")) > 0 or to_int(row.get("shared_description_count")) > 0
        ]
    elif bucket_id == "english_hard_negative_template_probe":
        candidates = [
            row
            for row in candidates
            if to_int(row.get("shared_contact_count")) == 0
            and to_int(row.get("shared_pgp_fingerprint_count")) == 0
            and row.get("candidate_scope") == "sockpuppet_primary"
        ]

    return sorted(candidates, key=queue_sort_key)


def select_with_seller_cap(rows: list[dict], target_count: int, max_per_seller: int, selected_pair_uids: set[str]) -> list[dict]:
    selected: list[dict] = []
    seller_counts: Counter = Counter()
    for row in rows:
        pair_uid = row["pair_uid"]
        if pair_uid in selected_pair_uids:
            continue
        left = row.get("seller_uid_left", "")
        right = row.get("seller_uid_right", "")
        if seller_counts[left] >= max_per_seller or seller_counts[right] >= max_per_seller:
            continue
        selected.append(row)
        selected_pair_uids.add(pair_uid)
        seller_counts[left] += 1
        seller_counts[right] += 1
        if len(selected) >= target_count:
            break
    return selected


def target_row(row: dict, rank: int, bucket_cfg: dict) -> dict:
    result = dict(row)
    result.update(
        {
            "en_source_expansion_rank": rank,
            "target_bucket": bucket_cfg["bucket_id"],
            "target_label_hint": bucket_cfg["target_label_hint"],
            "target_action": "review_existing_pending_queue_row",
            "source_existing_active_bool": "1",
            "source_existing_step4_bool": "1",
            "review_rubric_hint": bucket_cfg["description"],
            "review_status": "pending",
            "review_label": "",
            "reviewer_id": "",
            "review_notes": "",
        }
    )
    return result


def summarize_frozen(rows: list[dict]) -> dict:
    supervision = [row for row in rows if normalize_text(row.get("usable_for_supervision")) == "1"]
    split_counts = Counter(row.get("split_name", "") for row in supervision)
    split_label_counts: dict[str, Counter] = {}
    for row in supervision:
        split_label_counts.setdefault(row.get("split_name", ""), Counter())[row.get("review_label", "")] += 1
    return {
        "reviewed_row_count": len(rows),
        "supervision_row_count": len(supervision),
        "label_counts": dict(Counter(row.get("review_label", "") for row in rows)),
        "split_counts": dict(split_counts),
        "split_label_counts": {split: dict(counter) for split, counter in split_label_counts.items()},
    }


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]

    active_path = ROOT / inputs["active_review_queue"]
    frozen_path = ROOT / inputs["active_frozen_labels"]
    step4_path = ROOT / inputs["step4_candidates"]
    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["queue_summary"]

    active_rows, active_fieldnames = load_csv(active_path)
    candidate_rows, _candidate_fields = load_csv(step4_path)
    frozen_rows, _frozen_fields = load_csv(frozen_path)
    candidate_index = {row["pair_uid"]: row for row in candidate_rows}

    eligible_rows: list[dict] = []
    skipped_counts = Counter()
    for row in active_rows:
        if not is_pending(row):
            skipped_counts["not_pending"] += 1
            continue
        if row.get("candidate_scope") != "sockpuppet_primary":
            skipped_counts["non_primary_scope"] += 1
            continue
        if is_soft_same_alias(row):
            skipped_counts["soft_same_alias"] += 1
            continue
        candidate = candidate_index.get(row["pair_uid"])
        if candidate is None:
            skipped_counts["missing_step4_candidate"] += 1
            continue
        eligible_rows.append(row_with_candidate_counts(row, candidate))

    selected_rows: list[dict] = []
    selected_pair_uids: set[str] = set()
    bucket_summaries = {}
    rank = 1
    for bucket_cfg in policy["selection_buckets"]:
        bucket_pool = bucket_candidates(eligible_rows, bucket_cfg)
        selected = select_with_seller_cap(
            bucket_pool,
            int(bucket_cfg["target_count"]),
            int(bucket_cfg["max_per_seller"]),
            selected_pair_uids,
        )
        for row in selected:
            selected_rows.append(target_row(row, rank, bucket_cfg))
            rank += 1
        bucket_summaries[bucket_cfg["bucket_id"]] = {
            "eligible_count": len(bucket_pool),
            "selected_count": len(selected),
            "target_count": int(bucket_cfg["target_count"]),
            "max_per_seller": int(bucket_cfg["max_per_seller"]),
        }

    output_fields = list(active_fieldnames)
    for field in TARGET_EXTRA_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    write_csv(queue_path, selected_rows, output_fields)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "active_review_queue": str(active_path.relative_to(ROOT)),
        "active_frozen_labels": str(frozen_path.relative_to(ROOT)),
        "step4_candidates": str(step4_path.relative_to(ROOT)),
        "targeted_review_queue": str(queue_path.relative_to(ROOT)),
        "current_frozen_summary": summarize_frozen(frozen_rows),
        "active_queue_row_count": len(active_rows),
        "pending_primary_eligible_count": len(eligible_rows),
        "selected_row_count": len(selected_rows),
        "selected_bucket_counts": dict(Counter(row["target_bucket"] for row in selected_rows)),
        "selected_hint_counts": dict(Counter(row["target_label_hint"] for row in selected_rows)),
        "skipped_counts": dict(skipped_counts),
        "bucket_summaries": bucket_summaries,
        "supervision_targets": policy["supervision_targets"],
        "hard_rules": policy["hard_rules"],
    }
    write_json(summary_path, summary)
    print(f"Wrote English source expansion queue: {queue_path}")
    print(f"Wrote queue summary: {summary_path}")
    print(f"selected_row_count={len(selected_rows)} bucket_counts={summary['selected_bucket_counts']}")


if __name__ == "__main__":
    main()
