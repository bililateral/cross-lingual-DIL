from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_paper_targeted_expansion_policy.json"

EXTRA_FIELDS = [
    "paper_queue_rank",
    "target_bucket",
    "target_reason",
    "suggested_label",
    "review_priority",
    "review_instruction",
    "source_buckets",
    "score_tokens_present",
    "lr_l2_seed_count",
    "lr_l2_prob_min",
    "lr_l2_prob_mean",
    "lr_l2_prob_max",
    "zero_shot_bge_m3_prob",
    "identifier_prob_mean",
    "identifier_prob_max",
    "any_current_prob_max",
    "touches_strict_direct_proof_seller",
    "strict_direct_proof_seller_uids",
    "existing_active_bool",
    "existing_active_review_status",
    "existing_active_review_label",
    "existing_frozen_bool",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-targeted Step 5 expansion queue from current Step 11 candidate outputs."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the paper-targeted expansion policy JSON.",
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


def to_float(value: object) -> float | None:
    if value in {"", None}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def rounded(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def token_family(token: str) -> str:
    if token.startswith("core_few_shot_bge_m3_lr_l2_ratio_10pct_seed_"):
        return "lr_l2"
    if token == "core_zero_shot_bge_m3":
        return "zero_shot_bge_m3"
    if token.startswith("identifier_augmented_few_shot_default_"):
        return "identifier_augmented"
    return "other"


def reviewed_pair_uids(active_rows: list[dict], frozen_rows: list[dict]) -> set[str]:
    reviewed = {
        row["pair_uid"]
        for row in active_rows
        if normalize_label(row.get("review_label")) in {"positive", "negative", "uncertain"}
    }
    reviewed.update(
        row["pair_uid"]
        for row in frozen_rows
        if normalize_label(row.get("review_label")) in {"positive", "negative", "uncertain"}
    )
    return reviewed


def proof_sellers_from_strict_review(rows: list[dict]) -> tuple[set[str], set[str]]:
    proof_pair_uids = set()
    proof_sellers = set()
    for row in rows:
        if row.get("strict_direct_edge_class") != "proof_seller_facing_direct_contact_edge":
            continue
        pair_uid = row.get("pair_uid", "")
        parts = pair_uid.split("||")
        if len(parts) != 2:
            continue
        proof_pair_uids.add(pair_uid)
        proof_sellers.update(parts)
    return proof_pair_uids, proof_sellers


def load_step11_scores(summary_paths: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    score_paths = {}
    for summary_path_text in summary_paths:
        summary_path = ROOT / summary_path_text
        summary = load_json(summary_path)
        token = summary["selected_scorer"]["scorer_token"]
        scored_path = ROOT / summary["output_paths"]["scored_pairs"]
        score_paths[token] = str(scored_path.relative_to(ROOT))
        rows, _ = load_csv(scored_path)
        for row in rows:
            prob = to_float(row.get("prob_positive"))
            if prob is None:
                continue
            scores[row["pair_uid"]][token] = prob
    return scores, score_paths


def pair_score_summary(pair_scores: dict[str, float]) -> dict:
    by_family: dict[str, list[float]] = defaultdict(list)
    zero_prob = None
    for token, prob in pair_scores.items():
        family = token_family(token)
        by_family[family].append(prob)
        if family == "zero_shot_bge_m3":
            zero_prob = prob
    lr_probs = by_family.get("lr_l2", [])
    identifier_probs = by_family.get("identifier_augmented", [])
    all_probs = list(pair_scores.values())
    return {
        "lr_l2_seed_count": len(lr_probs),
        "lr_l2_prob_min": min(lr_probs) if lr_probs else None,
        "lr_l2_prob_mean": mean(lr_probs) if lr_probs else None,
        "lr_l2_prob_max": max(lr_probs) if lr_probs else None,
        "zero_shot_bge_m3_prob": zero_prob,
        "identifier_prob_mean": mean(identifier_probs) if identifier_probs else None,
        "identifier_prob_max": max(identifier_probs) if identifier_probs else None,
        "any_current_prob_max": max(all_probs) if all_probs else None,
        "score_tokens_present": " || ".join(sorted(pair_scores)),
    }


def candidate_row(
    pair_uid: str,
    *,
    bucket_id: str,
    bucket_cfg: dict,
    source_buckets: list[str],
    candidate: dict,
    active_row: dict | None,
    score_info: dict,
    proof_sellers: set[str],
) -> dict:
    base = dict(candidate)
    if active_row:
        # Keep the active queue's reviewer-facing fields where available, but
        # preserve candidate-level structural columns from Step 4.
        for key in (
            "balanced_review_rank",
            "review_stratum",
            "review_priority",
            "review_status",
            "review_label",
            "reviewer_id",
            "review_notes",
            "left_preview",
            "right_preview",
        ):
            if key in active_row:
                base[key] = active_row[key]
    left = candidate.get("seller_uid_left", "")
    right = candidate.get("seller_uid_right", "")
    touching = sorted({seller for seller in (left, right) if seller in proof_sellers})
    base.update(
        {
            "target_bucket": bucket_id,
            "target_reason": bucket_cfg["target_reason"],
            "suggested_label": bucket_cfg["suggested_label"],
            "review_priority": bucket_cfg["review_priority"],
            "review_instruction": bucket_cfg["review_instruction"],
            "source_buckets": " || ".join(source_buckets),
            "score_tokens_present": score_info["score_tokens_present"],
            "lr_l2_seed_count": score_info["lr_l2_seed_count"],
            "lr_l2_prob_min": rounded(score_info["lr_l2_prob_min"]),
            "lr_l2_prob_mean": rounded(score_info["lr_l2_prob_mean"]),
            "lr_l2_prob_max": rounded(score_info["lr_l2_prob_max"]),
            "zero_shot_bge_m3_prob": rounded(score_info["zero_shot_bge_m3_prob"]),
            "identifier_prob_mean": rounded(score_info["identifier_prob_mean"]),
            "identifier_prob_max": rounded(score_info["identifier_prob_max"]),
            "any_current_prob_max": rounded(score_info["any_current_prob_max"]),
            "touches_strict_direct_proof_seller": int(bool(touching)),
            "strict_direct_proof_seller_uids": " || ".join(touching),
            "existing_active_bool": int(active_row is not None),
            "existing_active_review_status": active_row.get("review_status", "") if active_row else "",
            "existing_active_review_label": active_row.get("review_label", "") if active_row else "",
            "existing_frozen_bool": 0,
            "review_status": "pending",
            "review_label": "",
            "reviewer_id": "",
            "review_notes": "",
        }
    )
    return base


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]
    selection = policy["selection"]

    candidates, candidate_fields = load_csv(ROOT / inputs["step4_candidates"])
    active_rows, active_fields = load_csv(ROOT / inputs["active_review_queue"])
    frozen_rows, _ = load_csv(ROOT / inputs["active_frozen_labels"])
    strict_rows, _ = load_csv(ROOT / inputs["strict_direct_edge_review"])

    candidate_index = {row["pair_uid"]: row for row in candidates}
    active_index = {row["pair_uid"]: row for row in active_rows}
    frozen_uids = {row["pair_uid"] for row in frozen_rows}
    reviewed_uids = reviewed_pair_uids(active_rows, frozen_rows)
    proof_pair_uids, proof_sellers = proof_sellers_from_strict_review(strict_rows)
    step11_scores, score_paths = load_step11_scores(inputs["step11_summaries"])

    direct_shared_non_external_unreviewed = [
        row["pair_uid"]
        for row in candidates
        if row["pair_uid"] not in reviewed_uids
        and normalize_text(row.get("shared_contact_values"))
        and "external_url" not in normalize_text(row.get("shared_contact_values")).lower()
    ]

    bucket_cfgs = selection["buckets"]
    buckets: dict[str, list[tuple[tuple, str, dict, list[str], dict]]] = defaultdict(list)
    bucket_hits_by_pair: dict[str, list[str]] = defaultdict(list)

    for pair_uid, pair_scores in step11_scores.items():
        if pair_uid not in candidate_index:
            continue
        if selection.get("exclude_reviewed_rows", True) and pair_uid in reviewed_uids:
            continue
        if selection.get("exclude_frozen_rows", True) and pair_uid in frozen_uids:
            continue

        candidate = candidate_index[pair_uid]
        score_info = pair_score_summary(pair_scores)
        left = candidate.get("seller_uid_left", "")
        right = candidate.get("seller_uid_right", "")
        touches_proof = left in proof_sellers or right in proof_sellers

        robust_cfg = bucket_cfgs["robust_lr_l2_high_score_unreviewed"]
        if (
            score_info["lr_l2_seed_count"] >= int(robust_cfg["min_lr_l2_seed_count"])
            and (score_info["lr_l2_prob_min"] or 0.0) >= float(robust_cfg["min_lr_l2_prob_min"])
        ):
            bucket_hits_by_pair[pair_uid].append("robust_lr_l2_high_score_unreviewed")
            buckets["robust_lr_l2_high_score_unreviewed"].append(
                (
                    (
                        -(score_info["lr_l2_prob_min"] or 0.0),
                        -(score_info["lr_l2_prob_mean"] or 0.0),
                        pair_uid,
                    ),
                    pair_uid,
                    score_info,
                    ["robust_lr_l2_high_score_unreviewed"],
                    candidate,
                )
            )

        identifier_cfg = bucket_cfgs["identifier_control_high_score_unreviewed"]
        if (score_info["identifier_prob_max"] or 0.0) >= float(identifier_cfg["min_identifier_prob_max"]):
            bucket_hits_by_pair[pair_uid].append("identifier_control_high_score_unreviewed")
            buckets["identifier_control_high_score_unreviewed"].append(
                (
                    (
                        -(score_info["identifier_prob_max"] or 0.0),
                        -(score_info["lr_l2_prob_mean"] or 0.0),
                        pair_uid,
                    ),
                    pair_uid,
                    score_info,
                    ["identifier_control_high_score_unreviewed"],
                    candidate,
                )
            )

        proof_cfg = bucket_cfgs["direct_proof_anchor_neighbor_unreviewed"]
        if touches_proof and (score_info["any_current_prob_max"] or 0.0) >= float(proof_cfg["min_any_current_prob"]):
            bucket_hits_by_pair[pair_uid].append("direct_proof_anchor_neighbor_unreviewed")
            buckets["direct_proof_anchor_neighbor_unreviewed"].append(
                (
                    (
                        -(score_info["any_current_prob_max"] or 0.0),
                        -(score_info["lr_l2_prob_mean"] or 0.0),
                        pair_uid,
                    ),
                    pair_uid,
                    score_info,
                    ["direct_proof_anchor_neighbor_unreviewed"],
                    candidate,
                )
            )

    selected_rows = []
    selected_uids = set()
    bucket_selected_counts = Counter()
    bucket_order = [
        "robust_lr_l2_high_score_unreviewed",
        "identifier_control_high_score_unreviewed",
        "direct_proof_anchor_neighbor_unreviewed",
    ]
    for bucket_id in bucket_order:
        cfg = bucket_cfgs[bucket_id]
        max_rows = int(cfg["max_rows"])
        for _sort_key, pair_uid, score_info, _source, candidate in sorted(buckets[bucket_id], key=lambda item: item[0]):
            if pair_uid in selected_uids:
                continue
            if bucket_selected_counts[bucket_id] >= max_rows:
                break
            selected_uids.add(pair_uid)
            bucket_selected_counts[bucket_id] += 1
            source_buckets = bucket_hits_by_pair[pair_uid]
            selected_rows.append(
                candidate_row(
                    pair_uid,
                    bucket_id=bucket_id,
                    bucket_cfg=cfg,
                    source_buckets=source_buckets,
                    candidate=candidate,
                    active_row=active_index.get(pair_uid),
                    score_info=score_info,
                    proof_sellers=proof_sellers,
                )
            )
            if len(selected_rows) >= int(selection["max_queue_rows"]):
                break
        if len(selected_rows) >= int(selection["max_queue_rows"]):
            break

    for idx, row in enumerate(selected_rows, start=1):
        row["paper_queue_rank"] = idx

    output_fields = []
    for field in EXTRA_FIELDS + active_fields + candidate_fields:
        if field not in output_fields:
            output_fields.append(field)

    queue_path = ROOT / policy["outputs"]["targeted_review_queue"]
    summary_path = ROOT / policy["outputs"]["summary"]
    write_csv(queue_path, selected_rows, output_fields)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "queue_version": policy["version"],
        "targeted_review_queue": str(queue_path.relative_to(ROOT)),
        "selected_row_count": len(selected_rows),
        "selected_bucket_counts": dict(Counter(row["target_bucket"] for row in selected_rows)),
        "source_bucket_counts": dict(
            Counter(
                bucket
                for row in selected_rows
                for bucket in normalize_text(row.get("source_buckets")).split(" || ")
                if bucket
            )
        ),
        "proof_pair_count": len(proof_pair_uids),
        "proof_seller_count": len(proof_sellers),
        "proof_pair_uids": sorted(proof_pair_uids),
        "step11_scored_pair_files": score_paths,
        "step11_unique_pair_count": len(step11_scores),
        "step4_candidate_count": len(candidates),
        "active_review_row_count": len(active_rows),
        "reviewed_or_frozen_exclusion_count": len(reviewed_uids | frozen_uids),
        "direct_shared_non_external_unreviewed_count": len(direct_shared_non_external_unreviewed),
        "direct_shared_non_external_unreviewed_examples": direct_shared_non_external_unreviewed[:20],
        "candidate_bucket_pool_counts_before_caps": {
            bucket_id: len(rows)
            for bucket_id, rows in buckets.items()
        },
        "acceptance_checks": {
            "no_duplicate_pair_uid": len(selected_uids) == len(selected_rows),
            "all_selected_unreviewed": all(row["pair_uid"] not in reviewed_uids for row in selected_rows),
            "all_selected_not_frozen": all(row["pair_uid"] not in frozen_uids for row in selected_rows),
            "all_selected_from_step4": all(row["pair_uid"] in candidate_index for row in selected_rows),
        },
        "top_rows": [
            {
                "paper_queue_rank": row["paper_queue_rank"],
                "target_bucket": row["target_bucket"],
                "pair_uid": row["pair_uid"],
                "lr_l2_prob_min": row.get("lr_l2_prob_min", ""),
                "identifier_prob_max": row.get("identifier_prob_max", ""),
                "any_current_prob_max": row.get("any_current_prob_max", ""),
                "touches_strict_direct_proof_seller": row.get("touches_strict_direct_proof_seller", ""),
                "review_stratum": row.get("review_stratum", ""),
                "shared_contact_values": row.get("shared_contact_values", ""),
            }
            for row in selected_rows[:25]
        ],
    }
    write_json(summary_path, summary)

    print(f"Wrote paper-targeted review queue: {queue_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"selected_row_count={len(selected_rows)} bucket_counts={summary['selected_bucket_counts']}")
    print(f"direct_shared_non_external_unreviewed_count={len(direct_shared_non_external_unreviewed)}")


if __name__ == "__main__":
    main()
