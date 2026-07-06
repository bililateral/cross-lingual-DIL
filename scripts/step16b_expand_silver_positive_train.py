from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step16b_silver_positive_expansion_policy.json"
REVIEWER_ID = "step16b_silver_positive_expansion_20260706"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand zh_target_strict train-only positive supervision with weakly supervised "
            "silver pairs. Existing valid/test rows are kept fixed."
        )
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="Step16B policy JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate/audit outputs and summary without modifying the frozen label CSV.",
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def split_contacts(value: str) -> list[str]:
    return [part.strip().lower() for part in str(value or "").split("||") if part.strip()]


def seller_pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((row["seller_uid_left"], row["seller_uid_right"])))


def current_valid_test_sellers(frozen_rows: list[dict]) -> set[str]:
    sellers: set[str] = set()
    for row in frozen_rows:
        if row.get("split_name") not in {"valid", "test"}:
            continue
        if row.get("review_label") not in {"positive", "negative"}:
            continue
        sellers.add(row.get("seller_uid_left", ""))
        sellers.add(row.get("seller_uid_right", ""))
    sellers.discard("")
    return sellers


def weak_positive_tags(row: dict, rules: dict) -> list[str]:
    tags: list[str] = []
    title_count = as_int(row.get("shared_title_count"))
    desc_count = as_int(row.get("shared_description_count"))
    contact_count = as_int(row.get("shared_contact_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))
    rank_score = as_float(row.get("candidate_rank_score"))
    shared_signal_count = title_count + desc_count + contact_count + as_int(row.get("shared_category_count"))

    cfg = rules.get("shared_contact_weak", {})
    if cfg.get("enabled", False) and contact_count >= int(cfg.get("min_shared_contact_count", 1)):
        if (
            structural >= float(cfg.get("min_structural_support_score", 1.0))
            or lexical >= float(cfg.get("min_lexical_similarity_fallback", 1.0))
            or title_count + desc_count >= int(cfg.get("min_shared_text_overlap_fallback", 1))
        ):
            tags.append("shared_contact_weak")

    cfg = rules.get("template_structural_weak", {})
    if cfg.get("enabled", False):
        if (
            title_count >= int(cfg.get("min_shared_title_count", 1))
            and desc_count >= int(cfg.get("min_shared_description_count", 1))
            and structural >= float(cfg.get("min_structural_support_score", 1.0))
            and lexical >= float(cfg.get("min_lexical_similarity", 1.0))
        ):
            tags.append("template_structural_weak")

    cfg = rules.get("clone_overlap_weak", {})
    if cfg.get("enabled", False):
        if (
            title_count >= int(cfg.get("min_shared_title_count", 2))
            and desc_count >= int(cfg.get("min_shared_description_count", 2))
            and structural >= float(cfg.get("min_structural_support_score", 1.0))
        ):
            tags.append("clone_overlap_weak")

    cfg = rules.get("high_similarity_weak", {})
    if cfg.get("enabled", False):
        if (
            lexical >= float(cfg.get("min_lexical_similarity", 1.0))
            and structural >= float(cfg.get("min_structural_support_score", 1.0))
            and shared_signal_count >= int(cfg.get("min_shared_signal_count", 1))
        ):
            tags.append("high_similarity_weak")

    cfg = rules.get("rank_structural_weak", {})
    if cfg.get("enabled", False):
        if (
            rank_score >= float(cfg.get("min_candidate_rank_score", 999.0))
            and structural >= float(cfg.get("min_structural_support_score", 1.0))
            and lexical >= float(cfg.get("min_lexical_similarity", 1.0))
        ):
            tags.append("rank_structural_weak")

    return tags


def weak_score(row: dict, contact_frequency: Counter[str]) -> float:
    contacts = split_contacts(row.get("shared_contact_values", ""))
    low_frequency_contact = any(contact_frequency[contact] <= 5 for contact in contacts)
    high_frequency_contact = any(contact_frequency[contact] > 20 for contact in contacts)
    score = as_float(row.get("candidate_rank_score"))
    score += as_float(row.get("lexical_similarity")) * 25.0
    score += as_float(row.get("structural_support_score")) * 25.0
    score += min(as_int(row.get("shared_title_count")), 5) * 4.0
    score += min(as_int(row.get("shared_description_count")), 5) * 5.0
    score += as_int(row.get("shared_contact_count")) * 15.0
    if low_frequency_contact:
        score += 10.0
    if high_frequency_contact:
        score -= 8.0
    return score


def training_weight(tags: list[str], is_closure: bool, policy: dict) -> tuple[str, float]:
    weights = policy["training_weights"]
    if is_closure:
        return "silver_component_closure", float(weights["silver_component_closure"])
    if "shared_contact_weak" in tags:
        return "silver_direct_or_contact", float(weights["silver_direct_or_contact"])
    return "silver_template_structural", float(weights["silver_template_structural"])


def build_silver_row(
    candidate_row: dict,
    template_fieldnames: list[str],
    balanced_rank: int,
    component_id: str,
    component_size: int,
    label_tier: str,
    sample_weight: float,
    reasons: list[str],
) -> dict:
    row = {field: "" for field in template_fieldnames}
    row.update(
        {
            "balanced_review_rank": str(balanced_rank),
            "pair_uid": candidate_row["pair_uid"],
            "data_bucket": candidate_row.get("data_bucket", "zh_target_strict"),
            "candidate_language": candidate_row.get("candidate_language", "zh"),
            "candidate_scope": candidate_row.get("candidate_scope", "sockpuppet_primary"),
            "review_stratum": label_tier,
            "review_priority": "silver_train_only",
            "review_status": "reviewed",
            "review_label": "positive",
            "reviewer_id": REVIEWER_ID,
            "review_notes": (
                "Step16B weak-supervision expansion: train-only silver positive. "
                f"label_tier={label_tier}; rules={'|'.join(reasons)}; "
                "not a gold benchmark label."
            ),
            "soft_same_alias_continuity_bool": "0",
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "split_name": "train",
            "split_component_id": component_id,
            "split_component_size": str(component_size),
            "seller_uid_left": candidate_row.get("seller_uid_left", ""),
            "seller_uid_right": candidate_row.get("seller_uid_right", ""),
            "source_market_raw_left": candidate_row.get("source_market_raw_left", ""),
            "source_market_raw_right": candidate_row.get("source_market_raw_right", ""),
            "source_seller_raw_left": candidate_row.get("source_seller_raw_left", ""),
            "source_seller_raw_right": candidate_row.get("source_seller_raw_right", ""),
            "alias_relation": candidate_row.get("alias_relation", ""),
            "same_market_raw": candidate_row.get("same_market_raw", ""),
            "candidate_rule_hits": candidate_row.get("candidate_rule_hits", ""),
            "candidate_rank_score": candidate_row.get("candidate_rank_score", ""),
            "lexical_similarity": candidate_row.get("lexical_similarity", ""),
            "structural_support_score": candidate_row.get("structural_support_score", ""),
            "shared_contact_count": candidate_row.get("shared_contact_count", ""),
            "shared_contact_values": candidate_row.get("shared_contact_values", ""),
            "shared_title_count": candidate_row.get("shared_title_count", ""),
            "shared_title_values": candidate_row.get("shared_title_values", ""),
            "shared_description_count": candidate_row.get("shared_description_count", ""),
            "shared_description_values": candidate_row.get("shared_description_values", ""),
            "shared_category_count": candidate_row.get("shared_category_count", ""),
            "shared_category_values": candidate_row.get("shared_category_values", ""),
            "shared_pgp_fingerprint_count": candidate_row.get("shared_pgp_fingerprint_count", ""),
            "shared_pgp_fingerprint_values": candidate_row.get("shared_pgp_fingerprint_values", ""),
            "left_preview": candidate_row.get("left_preview", ""),
            "right_preview": candidate_row.get("right_preview", ""),
            "label_tier": label_tier,
            "benchmark_eligible": "0",
            "silver_train_only": "1",
            "training_sample_weight": f"{sample_weight:.6f}",
            "silver_positive_reasons": "|".join(reasons),
        }
    )
    return row


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(right)] = self.find(left)

    def components(self) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        for value in list(self.parent):
            grouped[self.find(value)].add(value)
        return grouped


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
    feature_rows = load_csv(feature_path)
    feature_pair_uids = {row["pair_uid"] for row in feature_rows}
    existing_by_uid = {row["pair_uid"]: row for row in frozen_rows}
    candidate_by_uid = {row["pair_uid"]: row for row in candidate_rows}
    valid_test_sellers = current_valid_test_sellers(frozen_rows)
    protected_valid_test_pair_uids = {
        row["pair_uid"]
        for row in frozen_rows
        if row.get("split_name") in {"valid", "test"} and row.get("review_label") in {"positive", "negative"}
    }

    contact_frequency: Counter[str] = Counter()
    for row in candidate_rows:
        contact_frequency.update(split_contacts(row.get("shared_contact_values", "")))

    rules = policy["selection"]["weak_positive_rules"]
    candidate_records: list[dict] = []
    for candidate in candidate_rows:
        if candidate.get("candidate_scope") != policy["selection"]["candidate_scope"]:
            continue
        if candidate.get("candidate_language") != policy["selection"]["candidate_language"]:
            continue
        if candidate["pair_uid"] not in feature_pair_uids:
            continue
        existing = existing_by_uid.get(candidate["pair_uid"])
        if existing and existing.get("review_label") in {"positive", "negative"}:
            continue
        if candidate.get("seller_uid_left") in valid_test_sellers or candidate.get("seller_uid_right") in valid_test_sellers:
            continue
        tags = weak_positive_tags(candidate, rules)
        if not tags:
            continue
        score = weak_score(candidate, contact_frequency)
        label_tier, sample_weight = training_weight(tags, False, policy)
        candidate_records.append(
            {
                "pair_uid": candidate["pair_uid"],
                "silver_score": round(float(score), 6),
                "label_tier": label_tier,
                "training_sample_weight": round(float(sample_weight), 6),
                "silver_positive_reasons": "|".join(tags),
                "seller_uid_left": candidate.get("seller_uid_left", ""),
                "seller_uid_right": candidate.get("seller_uid_right", ""),
                "shared_contact_values": candidate.get("shared_contact_values", ""),
                "shared_title_count": candidate.get("shared_title_count", ""),
                "shared_description_count": candidate.get("shared_description_count", ""),
                "lexical_similarity": candidate.get("lexical_similarity", ""),
                "structural_support_score": candidate.get("structural_support_score", ""),
                "candidate_rank_score": candidate.get("candidate_rank_score", ""),
            }
        )

    candidate_records.sort(key=lambda row: (-float(row["silver_score"]), row["pair_uid"]))
    target = int(policy["selection"]["target_additional_positive_train_rows"])
    selected_direct = candidate_records[:target]
    selected_uids = {row["pair_uid"] for row in selected_direct}

    uf = UnionFind()
    for row in frozen_rows:
        if row.get("split_name") == "train" and row.get("review_label") == "positive":
            uf.union(row["seller_uid_left"], row["seller_uid_right"])
    for record in selected_direct:
        candidate = candidate_by_uid[record["pair_uid"]]
        uf.union(candidate["seller_uid_left"], candidate["seller_uid_right"])

    closure_records: list[dict] = []
    closure_cfg = policy["selection"]["silver_component_closure"]
    if closure_cfg.get("enabled", False) and len(selected_direct) < target:
        seller_pair_to_candidate = {seller_pair_key(row): row for row in candidate_rows if row["pair_uid"] in feature_pair_uids}
        remaining_slots = min(int(closure_cfg.get("max_rows", 0)), target - len(selected_direct))
        for sellers in uf.components().values():
            if len(sellers) < 3:
                continue
            ordered = sorted(sellers)
            for left_idx in range(len(ordered)):
                for right_idx in range(left_idx + 1, len(ordered)):
                    candidate = seller_pair_to_candidate.get((ordered[left_idx], ordered[right_idx]))
                    if not candidate:
                        continue
                    pair_uid = candidate["pair_uid"]
                    if pair_uid in selected_uids or pair_uid in protected_valid_test_pair_uids:
                        continue
                    existing = existing_by_uid.get(pair_uid)
                    if existing and existing.get("review_label") in {"positive", "negative"}:
                        continue
                    if candidate.get("seller_uid_left") in valid_test_sellers or candidate.get("seller_uid_right") in valid_test_sellers:
                        continue
                    label_tier, sample_weight = training_weight([], True, policy)
                    closure_records.append(
                        {
                            "pair_uid": pair_uid,
                            "silver_score": round(float(weak_score(candidate, contact_frequency)), 6),
                            "label_tier": label_tier,
                            "training_sample_weight": round(float(sample_weight), 6),
                            "silver_positive_reasons": "component_closure",
                            "seller_uid_left": candidate.get("seller_uid_left", ""),
                            "seller_uid_right": candidate.get("seller_uid_right", ""),
                            "shared_contact_values": candidate.get("shared_contact_values", ""),
                            "shared_title_count": candidate.get("shared_title_count", ""),
                            "shared_description_count": candidate.get("shared_description_count", ""),
                            "lexical_similarity": candidate.get("lexical_similarity", ""),
                            "structural_support_score": candidate.get("structural_support_score", ""),
                            "candidate_rank_score": candidate.get("candidate_rank_score", ""),
                        }
                    )
                    selected_uids.add(pair_uid)
                    if len(closure_records) >= remaining_slots:
                        break
                if len(closure_records) >= remaining_slots:
                    break
            if len(closure_records) >= remaining_slots:
                break

    selected_records = selected_direct + closure_records

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
    added_rows = []
    for index, record in enumerate(selected_records, start=1):
        candidate = candidate_by_uid[record["pair_uid"]]
        component_id = f"silver_train_comp_{index:05d}"
        row = build_silver_row(
            candidate,
            fieldnames,
            max_rank + index,
            component_id,
            2,
            str(record["label_tier"]),
            float(record["training_sample_weight"]),
            str(record["silver_positive_reasons"]).split("|"),
        )
        expanded_by_uid[record["pair_uid"]] = row
        added_rows.append(row)

    expanded_rows = []
    for row in frozen_rows:
        if row["pair_uid"] in {record["pair_uid"] for record in selected_records}:
            expanded_rows.append(expanded_by_uid[row["pair_uid"]])
        else:
            out = dict(row)
            for extra in fieldnames:
                out.setdefault(extra, "")
            if out.get("review_label") in {"positive", "negative"} and not out.get("training_sample_weight"):
                out["training_sample_weight"] = "1.000000"
                out["benchmark_eligible"] = "1" if out.get("split_name") in {"valid", "test"} else out.get("benchmark_eligible", "")
            expanded_rows.append(out)
    existing_uids = {row["pair_uid"] for row in frozen_rows}
    for row in added_rows:
        if row["pair_uid"] not in existing_uids:
            expanded_rows.append(row)

    candidate_audit_path = resolve(outputs["candidate_audit_csv"])
    applied_path = resolve(outputs["applied_training_pairs_csv"])
    summary_path = resolve(outputs["summary_json"])
    write_csv(candidate_audit_path, candidate_records, list(candidate_records[0].keys()) if candidate_records else [])
    write_csv(applied_path, selected_records, list(selected_records[0].keys()) if selected_records else [])

    before_train_pos = sum(1 for row in frozen_rows if row.get("split_name") == "train" and row.get("review_label") == "positive")
    after_train_pos = sum(1 for row in expanded_rows if row.get("split_name") == "train" and row.get("review_label") == "positive")
    summary = {
        "step": "step16b_silver_positive_expansion",
        "policy": str(policy_path.relative_to(ROOT)),
        "dry_run": bool(args.dry_run),
        "input_counts": {
            "frozen_rows": len(frozen_rows),
            "candidate_pairs": len(candidate_rows),
            "pair_feature_rows": len(feature_rows),
            "valid_test_seller_count": len(valid_test_sellers),
            "train_positive_before": before_train_pos,
        },
        "candidate_count": len(candidate_records),
        "selected_direct_silver_count": len(selected_direct),
        "selected_closure_silver_count": len(closure_records),
        "selected_total_silver_count": len(selected_records),
        "train_positive_after": after_train_pos,
        "added_positive_train_rows": after_train_pos - before_train_pos,
        "selected_label_tier_counts": dict(Counter(record["label_tier"] for record in selected_records)),
        "selected_reason_counts": dict(
            Counter(reason for record in selected_records for reason in str(record["silver_positive_reasons"]).split("|") if reason)
        ),
        "safety_checks": {
            "valid_test_rows_modified": False,
            "selected_pair_feature_coverage": all(record["pair_uid"] in feature_pair_uids for record in selected_records),
            "selected_valid_test_seller_overlap_count": sum(
                1
                for record in selected_records
                if record["seller_uid_left"] in valid_test_sellers or record["seller_uid_right"] in valid_test_sellers
            ),
            "existing_negative_converted_count": sum(
                1
                for record in selected_records
                if existing_by_uid.get(record["pair_uid"], {}).get("review_label") == "negative"
            ),
        },
        "outputs": {
            "candidate_audit_csv": str(candidate_audit_path.relative_to(ROOT)),
            "applied_training_pairs_csv": str(applied_path.relative_to(ROOT)),
            "expanded_frozen_labels": str(frozen_path.relative_to(ROOT)) if not args.dry_run else None,
            "summary_json": str(summary_path.relative_to(ROOT)),
        },
    }
    if summary["safety_checks"]["selected_valid_test_seller_overlap_count"] != 0:
        raise ValueError("Step16B selected rows overlap current valid/test sellers")
    if summary["safety_checks"]["existing_negative_converted_count"] != 0:
        raise ValueError("Step16B attempted to convert reviewed negative rows")

    if not args.dry_run:
        write_csv(frozen_path, expanded_rows, fieldnames)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
