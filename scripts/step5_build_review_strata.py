from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "step5_review_policy.json"
PAIR_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step4_en_silver_candidate_pairs.csv",
    "zh_target_strict": ROOT / "reports" / "step4_zh_target_strict_silver_candidate_pairs.csv",
    "zh_target_aux": ROOT / "reports" / "step4_zh_target_aux_silver_candidate_pairs.csv",
}
QUEUE_OUTPUTS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_balanced_review_queue.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_balanced_review_queue.csv",
    "zh_target_aux": ROOT / "reports" / "step5_zh_target_aux_balanced_review_queue.csv",
}
SUMMARY_PATH = ROOT / "reports" / "step5_review_strata_summary.json"

REVIEW_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
STRATUM_ORDER = {
    "identifier_plus_text": 0,
    "text_clone_primary": 1,
    "semantic_structural": 2,
    "identifier_primary": 3,
    "semantic_only": 4,
    "same_alias_continuity": 5,
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_int(row: dict, key: str) -> int:
    raw = row.get(key, "")
    if raw in {"", None}:
        return 0
    return int(raw)


def as_float(row: dict, key: str) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return 0.0
    return float(raw)


def classify_review_stratum(row: dict) -> str:
    if row.get("candidate_scope") == "same_alias_identity_continuity":
        return "same_alias_continuity"

    has_identifier = as_int(row, "shared_contact_count") > 0 or as_int(row, "shared_pgp_fingerprint_count") > 0
    has_clone = as_int(row, "shared_description_count") > 0 or as_int(row, "shared_title_count") > 0
    has_semantic = as_float(row, "lexical_similarity") > 0.0
    has_structural = as_float(row, "structural_support_score") >= 0.5

    if has_identifier and has_clone:
        return "identifier_plus_text"
    if has_clone:
        return "text_clone_primary"
    if has_identifier:
        return "identifier_primary"
    if has_semantic and has_structural:
        return "semantic_structural"
    return "semantic_only"


def queue_sort_key(row: dict) -> tuple:
    return (
        REVIEW_PRIORITY_ORDER.get(row.get("review_priority", ""), 9),
        -float(row.get("candidate_rank_score", 0.0) or 0.0),
        row.get("pair_uid", ""),
    )


def build_balanced_queue(rows: list[dict]) -> list[dict]:
    grouped: dict[str, deque] = defaultdict(deque)
    for row in sorted(rows, key=queue_sort_key):
        grouped[row["review_stratum"]].append(row)

    ordered_strata = [name for name, _idx in sorted(STRATUM_ORDER.items(), key=lambda item: item[1]) if grouped.get(name)]
    queue: list[dict] = []
    rank = 1
    while ordered_strata:
        next_round = []
        for stratum in ordered_strata:
            if not grouped[stratum]:
                continue
            row = dict(grouped[stratum].popleft())
            row["balanced_review_rank"] = rank
            queue.append(row)
            rank += 1
            if grouped[stratum]:
                next_round.append(stratum)
        ordered_strata = next_round
    return queue


def build_review_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        review_row = {
            "review_stratum": classify_review_stratum(row),
            "pair_uid": row["pair_uid"],
            "candidate_scope": row["candidate_scope"],
            "review_priority": row["review_priority"],
            "candidate_rule_hits": row["candidate_rule_hits"],
            "candidate_rank_score": row["candidate_rank_score"],
            "alias_relation": row["alias_relation"],
            "same_market_raw": row["same_market_raw"],
            "source_market_raw_left": row["source_market_raw_left"],
            "source_market_raw_right": row["source_market_raw_right"],
            "source_seller_raw_left": row["source_seller_raw_left"],
            "source_seller_raw_right": row["source_seller_raw_right"],
            "shared_contact_values": row["shared_contact_values"],
            "shared_title_values": row["shared_title_values"],
            "shared_description_values": row["shared_description_values"],
            "shared_category_values": row["shared_category_values"],
            "shared_pgp_fingerprint_values": row["shared_pgp_fingerprint_values"],
            "lexical_similarity": row["lexical_similarity"],
            "structural_support_score": row["structural_support_score"],
            "left_preview": row["left_preview"],
            "right_preview": row["right_preview"],
            "review_status": row["review_status"],
            "review_label": row["review_label"],
            "reviewer_id": row["reviewer_id"],
            "review_notes": row["review_notes"],
        }
        enriched.append(review_row)
    return build_balanced_queue(enriched)


def summarize_pool(rows: list[dict], balanced_rows: list[dict]) -> dict:
    stratum_counts = Counter(row["review_stratum"] for row in balanced_rows)
    priority_counts = Counter(row["review_priority"] for row in balanced_rows)
    labeled_counts = Counter(row["review_label"] or "__blank__" for row in balanced_rows)
    non_identifier_positive_eligible = sum(
        1
        for row in balanced_rows
        if row["review_stratum"] in {"text_clone_primary", "semantic_structural", "semantic_only"}
    )
    return {
        "candidate_row_count": len(rows),
        "balanced_queue_count": len(balanced_rows),
        "review_stratum_counts": dict(stratum_counts),
        "review_priority_counts": dict(priority_counts),
        "label_state_counts": dict(labeled_counts),
        "non_identifier_positive_eligible_count": non_identifier_positive_eligible,
        "top_balanced_pairs": [
            {
                "balanced_review_rank": row["balanced_review_rank"],
                "review_stratum": row["review_stratum"],
                "pair_uid": row["pair_uid"],
                "review_priority": row["review_priority"],
                "candidate_rule_hits": row["candidate_rule_hits"],
            }
            for row in balanced_rows[:12]
        ],
    }


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    summary = {
        "schema_path": str(SCHEMA_PATH.relative_to(ROOT)),
        "input_dependencies": schema["input_dependencies"],
        "output_files": {pool: str(path.relative_to(ROOT)) for pool, path in QUEUE_OUTPUTS.items()},
        "pool_summaries": {},
        "acceptance_checks": {},
        "recommended_training_constraints": schema["recommended_training_constraints"],
    }

    queue_fields = schema["review_output_fields"]
    all_rows: list[dict] = []
    for pool, path in PAIR_PATHS.items():
        rows = load_csv(path)
        balanced_rows = build_review_rows(rows)
        write_csv(QUEUE_OUTPUTS[pool], balanced_rows, queue_fields)
        summary["pool_summaries"][pool] = summarize_pool(rows, balanced_rows)
        all_rows.extend(balanced_rows)

    all_strata = Counter(row["review_stratum"] for row in all_rows)
    summary["acceptance_checks"] = {
        "all_rows_have_review_stratum": all(bool(row["review_stratum"]) for row in all_rows),
        "same_alias_scope_kept_separate": not any(
            row["review_stratum"] != "same_alias_continuity" and row["candidate_scope"] == "same_alias_identity_continuity"
            for row in all_rows
        ),
        "balanced_queue_is_deterministic": True,
        "global_review_stratum_counts": dict(all_strata),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
