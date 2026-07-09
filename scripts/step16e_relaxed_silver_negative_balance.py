from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step16e_relaxed_silver_negative_balance_policy.json"
REVIEWER_ID = "step16e_relaxed_silver_negative_balance_20260709"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add low-weight train-only weak negatives until zh train is class-balanced."
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dry_run_path(path: Path, dry_run: bool) -> Path:
    if not dry_run:
        return path
    return path.with_name(f"{path.stem}.dry_run{path.suffix}")


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(round(float(value or default)))
    except (TypeError, ValueError):
        return default


def current_eval_sellers(rows: list[dict[str, str]]) -> set[str]:
    sellers: set[str] = set()
    for row in rows:
        if row.get("split_name") in {"valid", "test"} and row.get("review_label") in {"positive", "negative"}:
            sellers.add(row.get("seller_uid_left", ""))
            sellers.add(row.get("seller_uid_right", ""))
    sellers.discard("")
    return sellers


def train_label_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("split_name") == "train" and row.get("review_label") in {"positive", "negative"}:
            counts[row["review_label"]] += 1
    return counts


def is_positive_like(row: dict[str, str], cfg: dict) -> bool:
    title = as_int(row.get("shared_title_count"))
    desc = as_int(row.get("shared_description_count"))
    category = as_int(row.get("shared_category_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))
    rank = as_float(row.get("candidate_rank_score"))
    high_similarity = (
        lexical >= float(cfg.get("high_similarity_min_lexical_similarity", 0.78))
        and structural >= float(cfg.get("high_similarity_min_structural_support_score", 0.2))
        and title + desc + category >= 1
    )
    template_like = (
        title >= int(cfg.get("template_min_shared_title_count", 1))
        and desc >= int(cfg.get("template_min_shared_description_count", 1))
        and lexical >= float(cfg.get("template_min_lexical_similarity", 0.28))
        and structural >= float(cfg.get("template_min_structural_support_score", 0.2))
    )
    rank_like = (
        rank >= float(cfg.get("rank_min_candidate_rank_score", 8.0))
        and lexical >= float(cfg.get("rank_min_lexical_similarity", 0.28))
        and structural >= float(cfg.get("rank_min_structural_support_score", 0.3))
    )
    return bool(high_similarity or template_like or rank_like)


def relaxed_negative_tags(row: dict[str, str], rules: dict) -> list[tuple[str, float]]:
    title = as_int(row.get("shared_title_count"))
    desc = as_int(row.get("shared_description_count"))
    contact = as_int(row.get("shared_contact_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))

    if contact != 0:
        return []

    matched: list[tuple[str, float]] = []

    cfg = rules.get("silver_ordinary_negative_relaxed", {})
    if cfg.get("enabled", False):
        if (
            title <= int(cfg.get("max_shared_title_count", 0))
            and desc <= int(cfg.get("max_shared_description_count", 0))
            and lexical < float(cfg.get("max_lexical_similarity", 0.5))
            and structural < float(cfg.get("max_structural_support_score", 0.24))
        ):
            matched.append(("silver_ordinary_negative_relaxed", float(cfg.get("training_sample_weight", 0.2))))

    cfg = rules.get("silver_semantic_low_structure_negative_relaxed", {})
    if cfg.get("enabled", False):
        if (
            title + desc <= int(cfg.get("max_shared_text_overlap", 1))
            and lexical < float(cfg.get("max_lexical_similarity", 0.6))
            and structural < float(cfg.get("max_structural_support_score", 0.25))
        ):
            matched.append(
                ("silver_semantic_low_structure_negative_relaxed", float(cfg.get("training_sample_weight", 0.18)))
            )

    cfg = rules.get("silver_template_noise_negative_relaxed", {})
    if cfg.get("enabled", False):
        if (
            title >= int(cfg.get("min_shared_title_count", 1))
            and desc >= int(cfg.get("min_shared_description_count", 1))
            and lexical < float(cfg.get("max_lexical_similarity", 0.6))
            and structural < float(cfg.get("max_structural_support_score", 0.28))
        ):
            matched.append(("silver_template_noise_negative_relaxed", float(cfg.get("training_sample_weight", 0.16))))

    return matched


def relaxed_score(row: dict[str, str], tags: list[tuple[str, float]]) -> float:
    # Higher score means a more informative hard-ish negative, but still with low
    # structural reliability and no direct shared contact.
    score = as_float(row.get("candidate_rank_score"))
    score += min(as_int(row.get("shared_title_count")), 3) * 2.0
    score += min(as_int(row.get("shared_description_count")), 4) * 2.5
    score += (1.0 - min(as_float(row.get("structural_support_score")), 1.0)) * 8.0
    score += len(tags) * 5.0
    return score


def build_silver_row(
    candidate: dict[str, str],
    fieldnames: list[str],
    balanced_rank: int,
    label_tier: str,
    sample_weight: float,
    reasons: list[str],
) -> dict[str, str]:
    row = {field: "" for field in fieldnames}
    row.update(
        {
            "balanced_review_rank": str(balanced_rank),
            "pair_uid": candidate["pair_uid"],
            "data_bucket": candidate.get("data_bucket", "zh_target_strict"),
            "candidate_language": candidate.get("candidate_language", "zh"),
            "candidate_scope": candidate.get("candidate_scope", "sockpuppet_primary"),
            "review_stratum": label_tier,
            "review_priority": "silver_train_only",
            "review_status": "reviewed",
            "review_label": "negative",
            "reviewer_id": REVIEWER_ID,
            "review_notes": (
                "Step16E relaxed weak-supervision balance: train-only silver negative. "
                f"label_tier={label_tier}; rules={'|'.join(reasons)}; "
                "not a gold benchmark label."
            ),
            "soft_same_alias_continuity_bool": "0",
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "split_name": "train",
            "split_component_id": f"silver_negative_train_comp_{balanced_rank:05d}",
            "split_component_size": "2",
            "seller_uid_left": candidate.get("seller_uid_left", ""),
            "seller_uid_right": candidate.get("seller_uid_right", ""),
            "source_market_raw_left": candidate.get("source_market_raw_left", ""),
            "source_market_raw_right": candidate.get("source_market_raw_right", ""),
            "source_seller_raw_left": candidate.get("source_seller_raw_left", ""),
            "source_seller_raw_right": candidate.get("source_seller_raw_right", ""),
            "alias_relation": candidate.get("alias_relation", ""),
            "same_market_raw": candidate.get("same_market_raw", ""),
            "candidate_rule_hits": candidate.get("candidate_rule_hits", ""),
            "candidate_rank_score": candidate.get("candidate_rank_score", ""),
            "lexical_similarity": candidate.get("lexical_similarity", ""),
            "structural_support_score": candidate.get("structural_support_score", ""),
            "shared_contact_count": candidate.get("shared_contact_count", ""),
            "shared_contact_values": candidate.get("shared_contact_values", ""),
            "shared_title_count": candidate.get("shared_title_count", ""),
            "shared_title_values": candidate.get("shared_title_values", ""),
            "shared_description_count": candidate.get("shared_description_count", ""),
            "shared_description_values": candidate.get("shared_description_values", ""),
            "shared_category_count": candidate.get("shared_category_count", ""),
            "shared_category_values": candidate.get("shared_category_values", ""),
            "shared_pgp_fingerprint_count": candidate.get("shared_pgp_fingerprint_count", ""),
            "shared_pgp_fingerprint_values": candidate.get("shared_pgp_fingerprint_values", ""),
            "left_preview": candidate.get("left_preview", ""),
            "right_preview": candidate.get("right_preview", ""),
            "label_tier": label_tier,
            "benchmark_eligible": "0",
            "silver_train_only": "1",
            "training_sample_weight": f"{sample_weight:.6f}",
            "silver_positive_reasons": "|".join(reasons),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = load_json(policy_path)
    inputs = policy["inputs"]
    outputs = policy["outputs"]

    frozen_path = resolve(inputs["frozen_labels"])
    candidate_path = resolve(inputs["candidate_pairs"])
    feature_path = resolve(inputs["pair_features"])

    frozen_rows = load_csv(frozen_path)
    candidate_rows = load_csv(candidate_path)
    feature_uids = {row["pair_uid"] for row in load_csv(feature_path)}
    existing_by_uid = {row["pair_uid"]: row for row in frozen_rows}
    candidate_by_uid = {row["pair_uid"]: row for row in candidate_rows}
    eval_sellers = current_eval_sellers(frozen_rows)

    counts_before = train_label_counts(frozen_rows)
    negative_gap = max(0, counts_before["positive"] - counts_before["negative"])
    max_additional = int(policy["selection"].get("max_additional_negative_train_rows", negative_gap))
    target_additional = min(negative_gap, max_additional)

    rules = policy["selection"]["relaxed_negative_rules"]
    positive_like_cfg = policy["selection"].get("positive_like_exclusion", {})
    candidates: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if candidate.get("candidate_scope") != policy["selection"]["candidate_scope"]:
            continue
        if candidate.get("candidate_language") != policy["selection"]["candidate_language"]:
            continue
        if candidate["pair_uid"] not in feature_uids:
            continue
        existing = existing_by_uid.get(candidate["pair_uid"])
        if existing and existing.get("review_label") in {"positive", "negative"}:
            continue
        if candidate.get("seller_uid_left", "") in eval_sellers:
            continue
        if candidate.get("seller_uid_right", "") in eval_sellers:
            continue
        if is_positive_like(candidate, positive_like_cfg):
            continue
        tags = relaxed_negative_tags(candidate, rules)
        if not tags:
            continue
        label_tier, sample_weight = max(tags, key=lambda item: item[1])
        candidates.append(
            {
                "pair_uid": candidate["pair_uid"],
                "silver_score": round(relaxed_score(candidate, tags), 6),
                "label_tier": label_tier,
                "training_sample_weight": round(sample_weight, 6),
                "silver_negative_reasons": "|".join(tag for tag, _weight in tags),
                "existing_review_label": existing.get("review_label", "") if existing else "",
                "seller_uid_left": candidate.get("seller_uid_left", ""),
                "seller_uid_right": candidate.get("seller_uid_right", ""),
                "shared_contact_count": candidate.get("shared_contact_count", ""),
                "shared_title_count": candidate.get("shared_title_count", ""),
                "shared_description_count": candidate.get("shared_description_count", ""),
                "lexical_similarity": candidate.get("lexical_similarity", ""),
                "structural_support_score": candidate.get("structural_support_score", ""),
                "candidate_rank_score": candidate.get("candidate_rank_score", ""),
            }
        )

    candidates.sort(key=lambda row: (-float(row["silver_score"]), str(row["pair_uid"])))
    selected = candidates[:target_additional]

    fieldnames = list(frozen_rows[0].keys())
    for extra in (
        "label_tier",
        "benchmark_eligible",
        "silver_train_only",
        "training_sample_weight",
        "silver_positive_reasons",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)

    max_rank = max(as_int(row.get("balanced_review_rank")) for row in frozen_rows) if frozen_rows else 0
    expanded_by_uid = {row["pair_uid"]: dict(row) for row in frozen_rows}
    added_rows: list[dict[str, str]] = []
    for offset, record in enumerate(selected, start=1):
        candidate = candidate_by_uid[str(record["pair_uid"])]
        reasons = str(record["silver_negative_reasons"]).split("|")
        added = build_silver_row(
            candidate,
            fieldnames,
            max_rank + offset,
            str(record["label_tier"]),
            float(record["training_sample_weight"]),
            reasons,
        )
        expanded_by_uid[added["pair_uid"]] = added
        added_rows.append(added)

    selected_uids = {str(row["pair_uid"]) for row in selected}
    expanded_rows: list[dict[str, str]] = []
    for row in frozen_rows:
        if row["pair_uid"] in selected_uids:
            expanded_rows.append(expanded_by_uid[row["pair_uid"]])
        else:
            out = dict(row)
            for extra in fieldnames:
                out.setdefault(extra, "")
            if out.get("review_label") in {"positive", "negative"} and not out.get("training_sample_weight"):
                out["training_sample_weight"] = "1.000000"
            expanded_rows.append(out)
    existing_uids = {row["pair_uid"] for row in frozen_rows}
    for row in added_rows:
        if row["pair_uid"] not in existing_uids:
            expanded_rows.append(row)

    counts_after = train_label_counts(expanded_rows)
    candidate_path_out = dry_run_path(resolve(outputs["candidate_audit_csv"]), args.dry_run)
    applied_path_out = dry_run_path(resolve(outputs["applied_training_pairs_csv"]), args.dry_run)
    summary_path_out = dry_run_path(resolve(outputs["summary_json"]), args.dry_run)
    expanded_path = resolve(outputs["expanded_frozen_labels"])

    write_csv(candidate_path_out, candidates, list(candidates[0].keys()) if candidates else [])
    write_csv(applied_path_out, selected, list(selected[0].keys()) if selected else [])

    summary = {
        "step": "step16e_relaxed_silver_negative_balance",
        "policy": str(policy_path.relative_to(ROOT)),
        "dry_run": bool(args.dry_run),
        "input_counts": {
            "frozen_rows": len(frozen_rows),
            "candidate_pairs": len(candidate_rows),
            "pair_feature_rows": len(feature_uids),
            "current_eval_seller_count": len(eval_sellers),
        },
        "train_label_counts_before": dict(counts_before),
        "train_label_counts_after": dict(counts_after),
        "negative_gap_before": negative_gap,
        "target_additional_negative_train_rows": target_additional,
        "candidate_count": len(candidates),
        "selected_total_silver_count": len(selected),
        "selected_label_tier_counts": dict(Counter(str(row["label_tier"]) for row in selected)),
        "selected_reason_counts": dict(
            Counter(reason for row in selected for reason in str(row["silver_negative_reasons"]).split("|") if reason)
        ),
        "selected_existing_uncertain_count": sum(1 for row in selected if row.get("existing_review_label") == "uncertain"),
        "safety_checks": {
            "valid_test_rows_modified": False,
            "selected_pair_feature_coverage": all(str(row["pair_uid"]) in feature_uids for row in selected),
            "selected_eval_seller_overlap_count": sum(
                1
                for row in selected
                if row.get("seller_uid_left") in eval_sellers or row.get("seller_uid_right") in eval_sellers
            ),
            "existing_positive_converted_count": sum(
                1 for row in selected if existing_by_uid.get(str(row["pair_uid"]), {}).get("review_label") == "positive"
            ),
        },
        "outputs": {
            "candidate_audit_csv": str(candidate_path_out.relative_to(ROOT)),
            "applied_training_pairs_csv": str(applied_path_out.relative_to(ROOT)),
            "expanded_frozen_labels": str(expanded_path.relative_to(ROOT)) if not args.dry_run else None,
            "summary_json": str(summary_path_out.relative_to(ROOT)),
        },
    }
    if summary["safety_checks"]["selected_eval_seller_overlap_count"] != 0:
        raise ValueError("Step16E selected rows overlap current validation/test sellers")
    if summary["safety_checks"]["existing_positive_converted_count"] != 0:
        raise ValueError("Step16E attempted to convert reviewed positive rows")

    if not args.dry_run:
        write_csv(expanded_path, expanded_rows, fieldnames)
    write_json(summary_path_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
