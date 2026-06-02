from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import step11_cluster_chinese_graph as step11


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_boundary_expansion_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a targeted Step 5 boundary-expansion review queue from current zero-shot BGE scores."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 boundary expansion policy JSON.",
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


def is_review_pending(row: dict) -> bool:
    status = str(row.get("review_status", "") or "").strip().lower()
    label = str(row.get("review_label", "") or "").strip().lower()
    return status in {"", "pending"} and label == ""


def style_gap_score(row: dict, style_gap_features: list[str]) -> float:
    return sum(to_float(row.get(feature)) for feature in style_gap_features)


def style_gap_hit_count(row: dict, style_gap_features: list[str], threshold: float = 0.25) -> int:
    return sum(1 for feature in style_gap_features if to_float(row.get(feature)) >= threshold)


def has_identifier_anchor(row: dict) -> bool:
    return to_int(row.get("has_shared_contact_exact")) > 0 or to_int(row.get("has_shared_pgp_fingerprint")) > 0


def passes_bucket_filters(row: dict, bucket_cfg: dict, style_gap_features: list[str]) -> bool:
    filters = bucket_cfg.get("filters", {}) or {}
    if filters.get("exclude_identifier_edges", False) and has_identifier_anchor(row):
        return False
    numeric_checks = (
        ("min_embedding_cosine_bge_m3", "embedding_cosine_bge_m3", ">="),
        ("min_structural_support_score_raw", "structural_support_score_raw", ">="),
        ("min_prob_positive", "prob_positive", ">="),
        ("max_prob_positive", "prob_positive", "<="),
    )
    for filter_key, row_key, op in numeric_checks:
        if filter_key not in filters:
            continue
        row_value = to_float(row.get(row_key))
        filter_value = float(filters[filter_key])
        if op == ">=" and row_value < filter_value:
            return False
        if op == "<=" and row_value > filter_value:
            return False
    if "min_style_gap_score" in filters:
        if style_gap_score(row, style_gap_features) < float(filters["min_style_gap_score"]):
            return False
    return True


def boundary_distance(row: dict, graph_primary_threshold: float) -> float:
    return abs(to_float(row.get("prob_positive")) - graph_primary_threshold)


def bucket_sort_key(
    row: dict,
    bucket_cfg: dict,
    style_gap_features: list[str],
    graph_primary_threshold: float,
    retained_pair_uids: set[str],
    active_queue_index: dict[str, dict],
) -> tuple:
    ranking = str(bucket_cfg.get("ranking", "") or "")
    pair_uid = row["pair_uid"]
    active_rank = int(to_float(active_queue_index[pair_uid].get("balanced_review_rank"), 10**9))
    graph_retained = int(pair_uid in retained_pair_uids)
    semantic = to_float(row.get("embedding_cosine_bge_m3"))
    structural = to_float(row.get("structural_support_score_raw"))
    prob = to_float(row.get("prob_positive"))
    shared_clone_count = to_int(row.get("shared_title_count_capped")) + to_int(row.get("shared_description_count_capped"))
    category_jaccard = to_float(row.get("profile_category_jaccard"))
    style_gap = style_gap_score(row, style_gap_features)
    style_hits = style_gap_hit_count(row, style_gap_features)
    distance = boundary_distance(row, graph_primary_threshold)

    if ranking == "prefer_graph_retained_then_high_score_structure_semantic":
        return (
            -graph_retained,
            -prob,
            -structural,
            -semantic,
            -shared_clone_count,
            -category_jaccard,
            active_rank,
            pair_uid,
        )
    if ranking == "prefer_not_graph_retained_then_style_gap_and_boundary_score":
        return (
            graph_retained,
            -style_gap,
            distance,
            -semantic,
            -prob,
            -style_hits,
            active_rank,
            pair_uid,
        )
    return (-prob, active_rank, pair_uid)


def build_output_row(
    *,
    rank: int,
    active_row: dict,
    scored_row: dict,
    bucket_cfg: dict,
    style_gap_features: list[str],
    graph_primary_threshold: float,
    scorer_token: str,
    retained_pair_uids: set[str],
) -> dict:
    result = dict(active_row)
    result["balanced_review_rank"] = rank
    result["review_priority"] = bucket_cfg.get("review_priority", active_row.get("review_priority", "high"))
    result["candidate_rule_hits"] = f"{active_row.get('candidate_rule_hits', '')}|{bucket_cfg['target_reason']}"
    result["candidate_rank_score"] = scored_row.get("prob_positive", active_row.get("candidate_rank_score", ""))
    result["review_status"] = "pending"
    result["review_label"] = ""
    result["reviewer_id"] = ""
    result["review_notes"] = bucket_cfg.get("review_notes", "")
    result.update(
        {
            "target_bucket": bucket_cfg["bucket_id"],
            "target_reason": bucket_cfg["target_reason"],
            "target_action": "review_existing_pending_queue_row",
            "suggested_label": bucket_cfg.get("suggested_label", ""),
            "suggested_confidence": bucket_cfg.get("suggested_confidence", ""),
            "source_step11_scorer_token": scorer_token,
            "source_step11_graph_primary_threshold": round(graph_primary_threshold, 6),
            "source_step11_prob_positive": scored_row.get("prob_positive", ""),
            "source_step11_score_rank_desc": scored_row.get("score_rank_desc", ""),
            "source_step11_threshold_pass_bool": int(to_float(scored_row.get("prob_positive")) >= graph_primary_threshold),
            "source_step11_graph_filter_retained_bool": int(scored_row["pair_uid"] in retained_pair_uids),
            "embedding_cosine_bge_m3": scored_row.get("embedding_cosine_bge_m3", ""),
            "structural_support_score_raw": scored_row.get("structural_support_score_raw", ""),
            "profile_category_jaccard": scored_row.get("profile_category_jaccard", ""),
            "style_gap_score": round(style_gap_score(scored_row, style_gap_features), 6),
            "style_gap_feature_hit_count": style_gap_hit_count(scored_row, style_gap_features),
            "boundary_distance_to_graph_threshold": round(boundary_distance(scored_row, graph_primary_threshold), 6),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)

    step11_summary = load_json(ROOT / policy["inputs"]["step11_summary"])
    step11_policy = load_json(ROOT / policy["inputs"]["step11_policy"])
    scored_rows, _ = load_csv(ROOT / policy["inputs"]["step11_scored_pairs"])
    active_rows, active_fieldnames = load_csv(ROOT / policy["inputs"]["active_review_queue"])
    frozen_rows, _ = load_csv(ROOT / policy["inputs"]["active_frozen_labels"])

    selected_scorer = step11_summary["selected_scorer"]
    graph_primary_threshold = float(selected_scorer["graph_primary_threshold"])
    scorer_token = str(selected_scorer["scorer_token"])
    pair_score_lookup = {row["pair_uid"]: to_float(row.get("prob_positive")) for row in scored_rows}
    threshold_edges = [row for row in scored_rows if pair_score_lookup[row["pair_uid"]] >= graph_primary_threshold]
    kept_edges, filter_diagnostics = step11.apply_graph_edge_filters(
        threshold_edges,
        pair_score_lookup,
        scorer_token,
        step11_policy,
    )
    retained_pair_uids = {row["pair_uid"] for row in kept_edges}

    active_queue_index = {row["pair_uid"]: row for row in active_rows}
    frozen_pair_uids = {row["pair_uid"] for row in frozen_rows}
    pending_pair_uids = {
        pair_uid
        for pair_uid, row in active_queue_index.items()
        if pair_uid not in frozen_pair_uids and is_review_pending(row)
    }
    candidate_rows = [row for row in scored_rows if row["pair_uid"] in pending_pair_uids]

    style_gap_features = [str(item) for item in policy["selection"]["style_gap_features"]]
    selected_rows: list[dict] = []
    selected_pair_uids: set[str] = set()
    bucket_summaries = []

    for bucket_cfg in policy["selection"]["buckets"]:
        available = [
            row
            for row in candidate_rows
            if row["pair_uid"] not in selected_pair_uids
            and passes_bucket_filters(row, bucket_cfg, style_gap_features)
        ]
        available.sort(
            key=lambda row: bucket_sort_key(
                row,
                bucket_cfg,
                style_gap_features,
                graph_primary_threshold,
                retained_pair_uids,
                active_queue_index,
            )
        )
        target_count = int(bucket_cfg["target_count"])
        chosen = available[:target_count]
        for row in chosen:
            selected_pair_uids.add(row["pair_uid"])
            selected_rows.append(
                build_output_row(
                    rank=len(selected_rows) + 1,
                    active_row=active_queue_index[row["pair_uid"]],
                    scored_row=row,
                    bucket_cfg=bucket_cfg,
                    style_gap_features=style_gap_features,
                    graph_primary_threshold=graph_primary_threshold,
                    scorer_token=scorer_token,
                    retained_pair_uids=retained_pair_uids,
                )
            )
        bucket_summaries.append(
            {
                "bucket_id": bucket_cfg["bucket_id"],
                "target_count": target_count,
                "available_count_after_prior_bucket_dedupe": len(available),
                "selected_count": len(chosen),
                "selected_review_stratum_counts": dict(Counter(row["review_stratum"] for row in chosen)),
                "selected_graph_filter_retained_count": sum(1 for row in chosen if row["pair_uid"] in retained_pair_uids),
                "selected_threshold_pass_count": sum(
                    1 for row in chosen if to_float(row.get("prob_positive")) >= graph_primary_threshold
                ),
            }
        )

    output_fieldnames = list(active_fieldnames)
    extra_fields = [
        "target_bucket",
        "target_reason",
        "target_action",
        "suggested_label",
        "suggested_confidence",
        "source_step11_scorer_token",
        "source_step11_graph_primary_threshold",
        "source_step11_prob_positive",
        "source_step11_score_rank_desc",
        "source_step11_threshold_pass_bool",
        "source_step11_graph_filter_retained_bool",
        "embedding_cosine_bge_m3",
        "structural_support_score_raw",
        "profile_category_jaccard",
        "style_gap_score",
        "style_gap_feature_hit_count",
        "boundary_distance_to_graph_threshold",
    ]
    for field in extra_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    output_csv = ROOT / policy["outputs"]["targeted_review_queue"]
    output_summary = ROOT / policy["outputs"]["summary"]
    write_csv(output_csv, selected_rows, output_fieldnames)

    selected_strata = Counter(row["review_stratum"] for row in selected_rows)
    selected_buckets = Counter(row["target_bucket"] for row in selected_rows)
    summary = {
        "queue_version": policy["queue_version"],
        "scope": policy["scope"],
        "policy_path": str(policy_path.relative_to(ROOT)),
        "active_boundary_kept_fixed": True,
        "source_step11_summary": policy["inputs"]["step11_summary"],
        "source_step11_scorer_token": scorer_token,
        "graph_primary_threshold": round(graph_primary_threshold, 6),
        "graph_edge_filtering": filter_diagnostics,
        "active_review_queue_row_count": len(active_rows),
        "active_pending_row_count": len(pending_pair_uids),
        "active_frozen_row_count": len(frozen_pair_uids),
        "candidate_row_count": len(candidate_rows),
        "selected_row_count": len(selected_rows),
        "selected_bucket_counts": dict(selected_buckets),
        "selected_review_stratum_counts": dict(selected_strata),
        "selected_graph_filter_retained_count": sum(
            to_int(row.get("source_step11_graph_filter_retained_bool")) for row in selected_rows
        ),
        "selected_threshold_pass_count": sum(
            to_int(row.get("source_step11_threshold_pass_bool")) for row in selected_rows
        ),
        "bucket_summaries": bucket_summaries,
        "review_guidelines": policy["review_guidelines"],
        "target_after_review": policy["target_after_review"],
        "outputs": policy["outputs"],
        "acceptance_checks": {
            "no_auto_labels": all(row["review_label"] == "" and row["review_status"] == "pending" for row in selected_rows),
            "no_duplicate_pair_uid": len(selected_pair_uids) == len(selected_rows),
            "all_selected_rows_pending_in_active_queue": all(row["pair_uid"] in pending_pair_uids for row in selected_rows),
            "no_selected_rows_in_active_frozen_labels": not any(row["pair_uid"] in frozen_pair_uids for row in selected_rows),
            "all_selected_rows_have_source_scores": all(row["pair_uid"] in pair_score_lookup for row in selected_rows),
        },
    }
    write_json(output_summary, summary)

    print(f"Wrote Step 5 boundary expansion queue: {output_csv}")
    print(f"Wrote Step 5 boundary expansion summary: {output_summary}")
    print(
        f"selected_rows={len(selected_rows)} buckets={dict(selected_buckets)} "
        f"strata={dict(selected_strata)}"
    )


if __name__ == "__main__":
    main()
