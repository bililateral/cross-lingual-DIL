from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_boundary_expansion_policy.json"
REVIEWER_ID = "codex_conservative_boundary_review_20260421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a conservative Codex review pass to the Step 5 boundary expansion queue."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 boundary expansion policy JSON.",
    )
    parser.add_argument(
        "--reviewer-id",
        default=REVIEWER_ID,
        help="Reviewer id to write into reviewed rows.",
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    return int(round(to_float(value)))


def style_gap_score(row: dict, style_gap_features: list[str]) -> float:
    return sum(to_float(row.get(feature)) for feature in style_gap_features)


def clone_signal_score(row: dict) -> float:
    return (
        to_float(row.get("shared_title_count_capped"))
        + to_float(row.get("shared_description_count_capped"))
        + to_float(row.get("shared_low_df_sentence_count"))
        + to_float(row.get("shared_rare_ngram_count"))
    )


def has_identifier_anchor(row: dict) -> bool:
    return to_int(row.get("has_shared_contact_exact")) > 0 or to_int(row.get("has_shared_pgp_fingerprint")) > 0


def classify_row(queue_row: dict, scored_row: dict, style_gap_features: list[str]) -> tuple[str, str, str]:
    bucket = str(queue_row.get("target_bucket", "") or "")
    graph_retained = to_int(queue_row.get("source_step11_graph_filter_retained_bool")) > 0
    semantic = to_float(scored_row.get("embedding_cosine_bge_m3"))
    structural = to_float(scored_row.get("structural_support_score_raw"))
    category_jaccard = to_float(scored_row.get("profile_category_jaccard"))
    style_gap = style_gap_score(scored_row, style_gap_features)
    clone_signal = clone_signal_score(scored_row)
    identifier_anchor = has_identifier_anchor(scored_row)

    if identifier_anchor:
        return (
            "positive",
            "direct_identifier_anchor",
            "codex_review: positive; direct shared contact/PGP anchor is present despite expansion filters.",
        )

    if (
        bucket == "positive_probe_high_semantic_high_structure"
        and graph_retained
        and semantic >= 0.90
        and structural >= 0.70
        and category_jaccard >= 0.75
        and style_gap <= 1.20
        and clone_signal >= 1.0
    ):
        return (
            "uncertain",
            "strong_overlap_without_identity_anchor",
            "codex_review: uncertain; graph-retained and strongly overlapping, but still lacks direct identity closure strong enough for a positive training label.",
        )

    if (
        bucket == "negative_probe_high_semantic_style_divergence"
        and not graph_retained
        and semantic >= 0.75
        and style_gap >= 2.20
    ):
        return (
            "negative",
            "high_semantic_high_style_gap_negative",
            "codex_review: negative; high semantic similarity but strong profile/style divergence, no direct identity anchor, and not retained by graph support.",
        )

    if not graph_retained and semantic >= 0.80 and style_gap >= 2.70:
        return (
            "negative",
            "unsupported_high_style_gap_negative",
            "codex_review: negative; no graph support, high style divergence, and no direct identity anchor.",
        )

    return (
        "uncertain",
        "insufficient_identity_closure",
        "codex_review: uncertain; overlap is review-relevant but not strong enough for a defensible positive/negative training label without direct identity closure.",
    )


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    scored_path = ROOT / policy["inputs"]["step11_scored_pairs"]
    summary_path = ROOT / "reports" / "step5_boundary_expansion_codex_review_summary.zero_shot_bge_20260421.json"

    queue_rows, fieldnames = load_csv(queue_path)
    scored_rows, _ = load_csv(scored_path)
    scored_index = {row["pair_uid"]: row for row in scored_rows}
    style_gap_features = [str(item) for item in policy["selection"]["style_gap_features"]]

    label_counts = Counter()
    rule_counts = Counter()
    bucket_label_counts: dict[str, Counter] = {}
    reviewed_rows = []
    missing_scores = []

    for row in queue_rows:
        pair_uid = row["pair_uid"]
        scored_row = scored_index.get(pair_uid)
        if scored_row is None:
            missing_scores.append(pair_uid)
            continue
        label, rule_id, notes = classify_row(row, scored_row, style_gap_features)
        row["review_status"] = "reviewed"
        row["review_label"] = label
        row["reviewer_id"] = args.reviewer_id
        row["review_notes"] = notes
        label_counts[label] += 1
        rule_counts[rule_id] += 1
        bucket = row.get("target_bucket", "")
        bucket_label_counts.setdefault(bucket, Counter())[label] += 1
        reviewed_rows.append(
            {
                "pair_uid": pair_uid,
                "target_bucket": bucket,
                "review_stratum": row.get("review_stratum", ""),
                "review_label": label,
                "rule_id": rule_id,
                "source_step11_prob_positive": row.get("source_step11_prob_positive", ""),
                "source_step11_graph_filter_retained_bool": row.get(
                    "source_step11_graph_filter_retained_bool", ""
                ),
                "embedding_cosine_bge_m3": row.get("embedding_cosine_bge_m3", ""),
                "structural_support_score_raw": row.get("structural_support_score_raw", ""),
                "style_gap_score": row.get("style_gap_score", ""),
            }
        )

    if missing_scores:
        raise SystemExit(f"Missing scored-pair rows for {len(missing_scores)} queue rows.")

    write_csv(queue_path, queue_rows, fieldnames)
    summary = {
        "reviewer_id": args.reviewer_id,
        "review_timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_path": str(queue_path.relative_to(ROOT)),
        "scored_pairs_path": str(scored_path.relative_to(ROOT)),
        "reviewed_row_count": len(queue_rows),
        "label_counts": dict(label_counts),
        "rule_counts": dict(rule_counts),
        "bucket_label_counts": {bucket: dict(counts) for bucket, counts in bucket_label_counts.items()},
        "rubric": {
            "positive": "Only direct identifiers or evidence as strong as direct identity closure.",
            "negative": "High-semantic rows with strong style/profile divergence, no direct identity anchor, and no graph support.",
            "uncertain": "All remaining rows with useful overlap but insufficient identity closure.",
        },
        "reviewed_rows": reviewed_rows,
    }
    write_json(summary_path, summary)

    print(f"Reviewed boundary expansion queue: {queue_path}")
    print(f"Wrote Codex review summary: {summary_path}")
    print(f"label_counts={dict(label_counts)} rule_counts={dict(rule_counts)}")


if __name__ == "__main__":
    main()
