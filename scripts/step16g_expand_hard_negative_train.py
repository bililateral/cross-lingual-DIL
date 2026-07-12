from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step16g_hard_negative_imbalance_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add evidence-filtered, low-weight, train-only Chinese hard negatives to a configured "
            "negative:positive ratio without changing validation or test."
        )
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value or default)))
    except (TypeError, ValueError):
        return default


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        if not left or not right:
            return
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def connected(self, left: str, right: str) -> bool:
        return bool(left and right and left in self.parent and right in self.parent and self.find(left) == self.find(right))


def supervised_split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = row.get("review_label", "")
        if label in {"positive", "negative"}:
            counts[row.get("split_name", "")][label] += 1
    return {split: dict(labels) for split, labels in sorted(counts.items())}


def split_sellers(rows: list[dict[str, str]], split_name: str) -> set[str]:
    sellers: set[str] = set()
    for row in rows:
        if row.get("split_name") != split_name or row.get("review_label") not in {"positive", "negative"}:
            continue
        sellers.update({row.get("seller_uid_left", ""), row.get("seller_uid_right", "")})
    sellers.discard("")
    return sellers


def seller_overlap_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    train = split_sellers(rows, "train")
    valid = split_sellers(rows, "valid")
    test = split_sellers(rows, "test")
    return {
        "train__valid": len(train & valid),
        "train__test": len(train & test),
        "valid__test": len(valid & test),
    }


def build_positive_components(rows: list[dict[str, str]]) -> UnionFind:
    components = UnionFind()
    for row in rows:
        if row.get("review_label") == "positive" and row.get("usable_for_supervision") == "1":
            components.union(row.get("seller_uid_left", ""), row.get("seller_uid_right", ""))
    return components


def is_positive_like(row: dict[str, str], cfg: dict) -> bool:
    title = as_int(row.get("shared_title_count"))
    description = as_int(row.get("shared_description_count"))
    category = as_int(row.get("shared_category_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))
    rank = as_float(row.get("candidate_rank_score"))
    high_similarity = (
        lexical >= float(cfg.get("high_similarity_min_lexical_similarity", 0.78))
        and structural >= float(cfg.get("high_similarity_min_structural_support_score", 0.2))
        and title + description + category >= 1
    )
    template_like = (
        title >= int(cfg.get("template_min_shared_title_count", 1))
        and description >= int(cfg.get("template_min_shared_description_count", 1))
        and lexical >= float(cfg.get("template_min_lexical_similarity", 0.28))
        and structural >= float(cfg.get("template_min_structural_support_score", 0.2))
    )
    rank_like = (
        rank >= float(cfg.get("rank_min_candidate_rank_score", 8.0))
        and lexical >= float(cfg.get("rank_min_lexical_similarity", 0.28))
        and structural >= float(cfg.get("rank_min_structural_support_score", 0.3))
    )
    return bool(high_similarity or template_like or rank_like)


def negative_tier(row: dict[str, str], rules: dict) -> tuple[str, float] | None:
    if as_int(row.get("shared_contact_count")) != 0 or as_int(row.get("shared_pgp_fingerprint_count")) != 0:
        return None
    title = as_int(row.get("shared_title_count"))
    description = as_int(row.get("shared_description_count"))
    lexical = as_float(row.get("lexical_similarity"))
    structural = as_float(row.get("structural_support_score"))

    boundary_cfg = rules["semantic_low_structure"]
    if (
        title + description <= int(boundary_cfg["max_shared_text_overlap"])
        and lexical < float(boundary_cfg["max_lexical_similarity"])
        and structural < float(boundary_cfg["max_structural_support_score"])
    ):
        ordinary_cfg = rules["ordinary_low_overlap"]
        if (
            title == 0
            and description == 0
            and lexical < float(ordinary_cfg["max_lexical_similarity"])
            and structural < float(ordinary_cfg["max_structural_support_score"])
        ):
            return "silver_ordinary_negative_imbalance", float(ordinary_cfg["training_sample_weight"])
        return "silver_semantic_low_structure_negative_imbalance", float(boundary_cfg["training_sample_weight"])
    return None


def candidate_score(row: dict[str, str]) -> float:
    # Rank close-to-boundary negatives first while all hard safety constraints remain active.
    return (
        as_float(row.get("lexical_similarity")) * 10.0
        + as_float(row.get("structural_support_score")) * 6.0
        + min(as_int(row.get("shared_title_count")) + as_int(row.get("shared_description_count")), 1) * 4.0
        + min(as_float(row.get("candidate_rank_score")), 20.0) * 0.25
    )


def build_training_row(
    candidate: dict[str, str],
    fieldnames: list[str],
    balanced_rank: int,
    label_tier: str,
    sample_weight: float,
    reviewer_id: str,
) -> dict[str, str]:
    row = {field: "" for field in fieldnames}
    row.update(candidate)
    row.update(
        {
            "balanced_review_rank": str(balanced_rank),
            "review_stratum": label_tier,
            "review_priority": "silver_train_only",
            "review_status": "reviewed",
            "review_label": "negative",
            "reviewer_id": reviewer_id,
            "review_notes": (
                "Step16G evidence-filtered hard-negative expansion: low-weight train-only weak negative; "
                f"label_tier={label_tier}; not a gold benchmark label."
            ),
            "soft_same_alias_continuity_bool": "0",
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "split_name": "train",
            "split_component_id": f"step16g_negative_train_comp_{balanced_rank:05d}",
            "split_component_size": "2",
            "label_tier": label_tier,
            "benchmark_eligible": "0",
            "silver_train_only": "1",
            "training_sample_weight": f"{sample_weight:.6f}",
            "silver_negative_reasons": label_tier,
        }
    )
    return {field: str(row.get(field, "")) for field in fieldnames}


def main() -> None:
    args = parse_args()
    policy_path = resolve(args.policy)
    policy = load_json(policy_path)
    inputs = policy["inputs"]
    outputs = policy["outputs"]
    selection = policy["selection"]

    frozen_path = resolve(inputs["frozen_labels"])
    candidate_path = resolve(inputs["candidate_pairs"])
    feature_path = resolve(inputs["pair_features"])
    frozen_rows = load_csv(frozen_path)
    candidate_rows = load_csv(candidate_path)
    feature_uids = {row["pair_uid"] for row in load_csv(feature_path)}
    existing_by_uid = {row["pair_uid"]: row for row in frozen_rows}

    reviewer_id = str(policy["metadata"]["reviewer_id"])
    existing_step_rows = [row for row in frozen_rows if row.get("reviewer_id") == reviewer_id]
    if existing_step_rows and not args.dry_run:
        raise SystemExit(
            f"Step16G is already present in the frozen labels ({len(existing_step_rows)} rows); "
            "refusing to apply it twice."
        )

    before_counts = supervised_split_counts(frozen_rows)
    train_before = before_counts.get("train", {})
    positive_count = int(train_before.get("positive", 0))
    negative_count = int(train_before.get("negative", 0))
    target_ratio = float(selection["target_negative_to_positive_ratio"])
    desired_negative_count = int(math.ceil(positive_count * target_ratio))
    additional_needed = max(0, desired_negative_count - negative_count)
    target_additional = min(additional_needed, int(selection["max_additional_negative_train_rows"]))

    eval_sellers = split_sellers(frozen_rows, "valid") | split_sellers(frozen_rows, "test")
    positive_components = build_positive_components(frozen_rows)
    positive_like_cfg = selection["positive_like_exclusion"]
    rules = selection["negative_rules"]
    candidates: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()

    for candidate in candidate_rows:
        if candidate.get("candidate_scope") != selection["candidate_scope"]:
            rejection_counts["scope"] += 1
            continue
        if candidate.get("candidate_language") != selection["candidate_language"]:
            rejection_counts["language"] += 1
            continue
        pair_uid = candidate.get("pair_uid", "")
        if pair_uid not in feature_uids:
            rejection_counts["missing_pair_features"] += 1
            continue
        existing = existing_by_uid.get(pair_uid)
        if existing and existing.get("review_label") in {"positive", "negative"}:
            rejection_counts["already_supervised"] += 1
            continue
        if selection.get("exclude_existing_uncertain", True) and existing and existing.get("review_label") == "uncertain":
            rejection_counts["existing_uncertain"] += 1
            continue
        left = candidate.get("seller_uid_left", "")
        right = candidate.get("seller_uid_right", "")
        if left in eval_sellers or right in eval_sellers:
            rejection_counts["evaluation_seller_overlap"] += 1
            continue
        if positive_components.connected(left, right):
            rejection_counts["known_positive_component_conflict"] += 1
            continue
        if is_positive_like(candidate, positive_like_cfg):
            rejection_counts["positive_like"] += 1
            continue
        tier = negative_tier(candidate, rules)
        if tier is None:
            rejection_counts["negative_rule_not_met"] += 1
            continue
        label_tier, sample_weight = tier
        candidates.append(
            {
                "pair_uid": pair_uid,
                "candidate_score": round(candidate_score(candidate), 6),
                "label_tier": label_tier,
                "training_sample_weight": round(sample_weight, 6),
                "existing_review_label": existing.get("review_label", "") if existing else "",
                "seller_uid_left": left,
                "seller_uid_right": right,
                "shared_contact_count": candidate.get("shared_contact_count", ""),
                "shared_pgp_fingerprint_count": candidate.get("shared_pgp_fingerprint_count", ""),
                "shared_title_count": candidate.get("shared_title_count", ""),
                "shared_description_count": candidate.get("shared_description_count", ""),
                "lexical_similarity": candidate.get("lexical_similarity", ""),
                "structural_support_score": candidate.get("structural_support_score", ""),
                "candidate_rank_score": candidate.get("candidate_rank_score", ""),
            }
        )

    candidates.sort(key=lambda row: (-float(row["candidate_score"]), str(row["pair_uid"])))
    seller_limit = int(selection["max_selected_pairs_per_seller"])
    seller_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    for candidate in candidates:
        left = str(candidate["seller_uid_left"])
        right = str(candidate["seller_uid_right"])
        if seller_counts[left] >= seller_limit or seller_counts[right] >= seller_limit:
            continue
        selected.append(candidate)
        seller_counts[left] += 1
        seller_counts[right] += 1
        if len(selected) >= target_additional:
            break

    if len(selected) < target_additional:
        raise ValueError(
            f"Step16G found only {len(selected)} seller-diverse candidates, below the required {target_additional}."
        )

    original_fieldnames = list(frozen_rows[0].keys())
    fieldnames = list(original_fieldnames)
    for extra in ("label_tier", "benchmark_eligible", "silver_train_only", "training_sample_weight", "silver_negative_reasons"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    max_rank = max((as_int(row.get("balanced_review_rank")) for row in frozen_rows), default=0)
    selected_by_uid = {str(row["pair_uid"]): row for row in selected}
    expanded_rows: list[dict[str, str]] = []
    added_rows: list[dict[str, str]] = []

    for row in frozen_rows:
        record = selected_by_uid.get(row["pair_uid"])
        if record is None:
            expanded_rows.append({field: row.get(field, "") for field in fieldnames})
            continue
        candidate = next(item for item in candidate_rows if item["pair_uid"] == row["pair_uid"])
        added = build_training_row(
            candidate,
            fieldnames,
            max_rank + len(added_rows) + 1,
            str(record["label_tier"]),
            float(record["training_sample_weight"]),
            reviewer_id,
        )
        expanded_rows.append(added)
        added_rows.append(added)

    existing_uids = {row["pair_uid"] for row in frozen_rows}
    candidate_lookup = {row["pair_uid"]: row for row in candidate_rows}
    for record in selected:
        pair_uid = str(record["pair_uid"])
        if pair_uid in existing_uids:
            continue
        added = build_training_row(
            candidate_lookup[pair_uid],
            fieldnames,
            max_rank + len(added_rows) + 1,
            str(record["label_tier"]),
            float(record["training_sample_weight"]),
            reviewer_id,
        )
        expanded_rows.append(added)
        added_rows.append(added)

    after_counts = supervised_split_counts(expanded_rows)
    overlap_after = seller_overlap_counts(expanded_rows)
    before_eval = {
        row["pair_uid"]: {field: row.get(field, "") for field in original_fieldnames}
        for row in frozen_rows
        if row.get("split_name") in {"valid", "test"}
    }
    after_eval = {
        row["pair_uid"]: {field: row.get(field, "") for field in original_fieldnames}
        for row in expanded_rows
        if row.get("split_name") in {"valid", "test"}
    }
    safety_checks = {
        "valid_test_rows_byte_equivalent": before_eval == after_eval,
        "selected_pair_feature_coverage": all(str(row["pair_uid"]) in feature_uids for row in selected),
        "selected_eval_seller_overlap_count": sum(
            1
            for row in selected
            if str(row["seller_uid_left"]) in eval_sellers or str(row["seller_uid_right"]) in eval_sellers
        ),
        "selected_known_positive_component_conflict_count": sum(
            1
            for row in selected
            if positive_components.connected(str(row["seller_uid_left"]), str(row["seller_uid_right"]))
        ),
        "existing_positive_or_negative_converted_count": sum(
            1
            for row in selected
            if existing_by_uid.get(str(row["pair_uid"]), {}).get("review_label") in {"positive", "negative"}
        ),
        "seller_overlap_counts_after": overlap_after,
        "maximum_selected_pairs_per_seller": max(seller_counts.values(), default=0),
        "target_negative_count_reached": after_counts.get("train", {}).get("negative", 0) >= desired_negative_count,
    }
    if not safety_checks["valid_test_rows_byte_equivalent"]:
        raise ValueError("Step16G would modify validation/test rows")
    if safety_checks["selected_eval_seller_overlap_count"] != 0 or any(overlap_after.values()):
        raise ValueError("Step16G would create train/validation/test seller leakage")
    if safety_checks["selected_known_positive_component_conflict_count"] != 0:
        raise ValueError("Step16G selected a pair inside an existing positive seller component")
    if safety_checks["existing_positive_or_negative_converted_count"] != 0:
        raise ValueError("Step16G attempted to convert an existing supervised label")
    if any(row.get("existing_review_label") == "uncertain" for row in selected):
        raise ValueError("Step16G attempted to convert an existing uncertain label")
    if not safety_checks["target_negative_count_reached"]:
        raise ValueError("Step16G did not reach the configured train negative target")

    output_suffix = ".dry_run" if args.dry_run else ""
    candidate_out = resolve(outputs["candidate_audit_csv"])
    selected_out = resolve(outputs["applied_training_pairs_csv"])
    summary_out = resolve(outputs["summary_json"])
    if output_suffix:
        candidate_out = candidate_out.with_name(candidate_out.stem + output_suffix + candidate_out.suffix)
        selected_out = selected_out.with_name(selected_out.stem + output_suffix + selected_out.suffix)
        summary_out = summary_out.with_name(summary_out.stem + output_suffix + summary_out.suffix)

    write_csv(candidate_out, candidates, list(candidates[0].keys()) if candidates else [])
    write_csv(selected_out, selected, list(selected[0].keys()) if selected else [])
    summary: dict[str, object] = {
        "step": "step16g_hard_negative_imbalance",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy.get("version"),
        "dry_run": bool(args.dry_run),
        "input_hashes": {
            "frozen_labels_sha256": sha256(frozen_path),
            "candidate_pairs_sha256": sha256(candidate_path),
            "pair_features_sha256": sha256(feature_path),
        },
        "split_counts_before": before_counts,
        "split_counts_after": after_counts,
        "target_negative_to_positive_ratio": target_ratio,
        "desired_train_negative_count": desired_negative_count,
        "target_additional_negative_train_rows": target_additional,
        "candidate_count": len(candidates),
        "selected_total_silver_count": len(selected),
        "selected_label_tier_counts": dict(Counter(str(row["label_tier"]) for row in selected)),
        "selected_existing_uncertain_count": sum(1 for row in selected if row.get("existing_review_label") == "uncertain"),
        "candidate_rejection_counts": dict(rejection_counts),
        "safety_checks": safety_checks,
        "scientific_scope": {
            "training_only_weak_supervision": True,
            "benchmark_eligible": False,
            "validation_test_unchanged": True,
            "purpose": "hard_negative_training_support_and_positive_pair_mixup_factorial_control",
            "not_a_claim_of_gold_negative_truth": True,
        },
        "outputs": {
            "candidate_audit_csv": str(candidate_out.relative_to(ROOT)),
            "applied_training_pairs_csv": str(selected_out.relative_to(ROOT)),
            "expanded_frozen_labels": None if args.dry_run else outputs["expanded_frozen_labels"],
            "summary_json": str(summary_out.relative_to(ROOT)),
        },
    }

    if not args.dry_run:
        expanded_path = resolve(outputs["expanded_frozen_labels"])
        backup_path = resolve(outputs["backup_frozen_labels"])
        if backup_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing Step16G backup: {backup_path}")
        shutil.copy2(expanded_path, backup_path)
        write_csv(expanded_path, expanded_rows, fieldnames)
        summary["outputs"]["backup_frozen_labels"] = str(backup_path.relative_to(ROOT))
        summary["output_hashes"] = {"expanded_frozen_labels_sha256": sha256(expanded_path)}

    write_json(summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
