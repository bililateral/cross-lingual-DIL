from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import step11_cluster_chinese_graph as step11
import step5_build_targeted_review_queue as base


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_v3_targeted_review_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the corrected Step 5 v3 targeted review queue from a current Step 11 graph."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 v3 targeted review policy JSON.",
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
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_pair_uid(left: str, right: str) -> str:
    return base.canonical_pair_uid(left, right)


def threshold_token(threshold: float) -> str:
    return f"{int(round(threshold * 1_000_000)):07d}"


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def numeric_bool(row: dict, field: str) -> bool:
    return int(float(row.get(field, 0) or 0)) > 0


def cluster_member_sets(cluster_rows: list[dict]) -> dict[int, dict]:
    clusters: dict[int, dict] = {}
    for row in cluster_rows:
        rank = int(row["cluster_rank"])
        cluster = clusters.setdefault(
            rank,
            {
                "cluster_id": row["cluster_id"],
                "members": set(),
            },
        )
        cluster["members"].add(row["seller_uid"])
    return clusters


def cluster_edges(kept_edges: list[dict], members: set[str]) -> list[dict]:
    edges = [
        row
        for row in kept_edges
        if row["seller_uid_left"] in members and row["seller_uid_right"] in members
    ]
    edges.sort(
        key=lambda row: (
            -float(row["prob_positive"]),
            canonical_pair_uid(row["seller_uid_left"], row["seller_uid_right"]),
        )
    )
    return edges


def edge_bucket(row: dict) -> str:
    if numeric_bool(row, "has_shared_contact_exact") or numeric_bool(row, "has_shared_pgp_fingerprint"):
        return "identifier_anchor"
    if row.get("review_stratum") == "text_clone_primary":
        return "template_clone"
    return "semantic_or_structural"


def policy_for_edge(cluster_cfg: dict, row: dict, cluster_rank: int) -> dict:
    bucket = edge_bucket(row)
    bucket_cfg = dict(cluster_cfg.get("bucket_rules", {}).get(bucket, {}))
    default_label = "positive_candidate" if bucket == "identifier_anchor" else "uncertain_candidate"
    return {
        "target_bucket": bucket_cfg.get(
            "target_bucket",
            f"{cluster_cfg.get('target_bucket_prefix', 'step11_cluster')}_{cluster_rank:04d}_{bucket}",
        ),
        "target_reason": bucket_cfg.get(
            "target_reason",
            f"{cluster_cfg.get('target_reason_prefix', 'step11_v3')}|cluster_{cluster_rank:04d}|{bucket}",
        ),
        "suggested_label": bucket_cfg.get("suggested_label", default_label),
        "suggested_confidence": bucket_cfg.get("suggested_confidence", "medium"),
        "review_notes": bucket_cfg.get("review_notes", cluster_cfg["review_notes"]),
    }


def find_threshold_view(step11_summary: dict, threshold: float) -> dict:
    token = threshold_token(threshold)
    view = step11_summary.get("threshold_views", {}).get(token)
    if view is not None:
        return view
    for candidate in step11_summary.get("threshold_views", {}).values():
        if round(float(candidate.get("threshold", -1)), 6) == round(threshold, 6):
            return candidate
    raise SystemExit(f"Missing Step 11 threshold view for threshold={threshold}")


def assert_source_graph_usable(step11_summary: dict, threshold_view: dict, graph_primary_threshold: float) -> None:
    graph_diagnostics = step11_summary.get("graph_threshold_diagnostics", {}) or {}
    if graph_diagnostics.get("graph_primary_threshold_exceeds_score_ceiling"):
        raise SystemExit(
            "Step 5 v3 queue builder refuses to consume a Step 11 graph whose primary threshold exceeds "
            f"the scorer score ceiling. threshold={graph_primary_threshold}, "
            f"score_max={graph_diagnostics.get('score_max')}. Choose a current non-empty Step 11 graph "
            "or set an explicit Step 11 graph threshold override first."
        )
    if int(threshold_view.get("threshold_pass_edge_count", 0) or 0) <= 0:
        raise SystemExit(
            "Step 5 v3 queue builder refuses to consume an empty Step 11 graph: "
            f"threshold={graph_primary_threshold} has zero threshold-pass edges."
        )
    if int(threshold_view.get("edge_count", 0) or 0) <= 0:
        raise SystemExit(
            "Step 5 v3 queue builder refuses to consume an empty post-filter Step 11 graph: "
            f"threshold={graph_primary_threshold} has zero retained edges after graph filters."
        )


def assert_graph_consistency(
    *,
    threshold_edges: list[dict],
    kept_edges: list[dict],
    filter_diagnostics: dict,
    threshold_view: dict,
    selected_edge_count: int,
) -> dict:
    expected_filter = threshold_view.get("graph_edge_filtering", {})
    checks = {
        "threshold_pass_edge_count_matches_summary": len(threshold_edges)
        == int(threshold_view.get("threshold_pass_edge_count", -1)),
        "pre_filter_edge_count_matches_summary": int(filter_diagnostics.get("pre_filter_edge_count", -1))
        == int(expected_filter.get("pre_filter_edge_count", -2)),
        "post_filter_edge_count_matches_summary": int(filter_diagnostics.get("post_filter_edge_count", -1))
        == int(expected_filter.get("post_filter_edge_count", -2)),
        "selected_edge_count_matches_summary_edge_count": selected_edge_count
        == int(threshold_view.get("edge_count", -1)),
        "selected_edge_count_matches_kept_edges": selected_edge_count == len(kept_edges),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Step 5 v3 graph consistency check failed: {failed}")
    return checks


def queue_or_rereview(
    *,
    row: dict,
    cluster_cfg: dict,
    cluster_rank: int,
    cluster_id: str,
    graph_primary_threshold: float,
    scorer_token: str,
    seller_index: dict[str, dict],
    frozen_index: dict[str, dict],
    active_queue_index: dict[str, dict],
    queued_rows: list[dict],
    rereview_rows: list[dict],
    skipped_existing: list[dict],
) -> None:
    pair_uid = row["pair_uid"]
    left_profile = seller_index[row["seller_uid_left"]]
    right_profile = seller_index[row["seller_uid_right"]]
    payload = base.build_row(
        rank=0,
        row=row,
        left_profile=left_profile,
        right_profile=right_profile,
        target_reason=cluster_cfg["target_reason"],
        target_bucket=cluster_cfg["target_bucket"],
        cluster_rank=cluster_rank,
        cluster_id=cluster_id,
        graph_primary_threshold=graph_primary_threshold,
        scorer_token=scorer_token,
        suggested_label=cluster_cfg["suggested_label"],
        suggested_confidence=cluster_cfg["suggested_confidence"],
        review_notes=cluster_cfg["review_notes"],
    )

    if pair_uid in frozen_index:
        existing = frozen_index[pair_uid]
        payload["existing_boundary_source"] = "active_frozen_labels"
        payload["existing_review_status"] = existing.get("review_status", "")
        payload["existing_review_label"] = existing.get("review_label", "")
        payload["existing_reviewer_id"] = existing.get("reviewer_id", "")
        payload["existing_review_notes"] = existing.get("review_notes", "")
        payload["target_action"] = "rereview_existing_frozen_label"
        rereview_rows.append(payload)
        skipped_existing.append({"pair_uid": pair_uid, "reason": "already_in_active_boundary::frozen"})
        return

    if pair_uid in active_queue_index:
        existing = active_queue_index[pair_uid]
        payload["existing_boundary_source"] = "active_review_queue"
        payload["existing_review_status"] = existing.get("review_status", "")
        payload["existing_review_label"] = existing.get("review_label", "")
        payload["existing_reviewer_id"] = existing.get("reviewer_id", "")
        payload["existing_review_notes"] = existing.get("review_notes", "")
        payload["target_action"] = (
            "rereview_existing_reviewed_queue_row"
            if str(existing.get("review_label", "")).strip()
            else "review_existing_pending_queue_row"
        )
        rereview_rows.append(payload)
        skipped_existing.append({"pair_uid": pair_uid, "reason": "already_in_active_boundary::queue"})
        return

    payload["target_action"] = "new_targeted_review"
    queued_rows.append(payload)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)

    step11_summary = load_json(ROOT / policy["inputs"]["step11_summary"])
    step11_policy = load_json(ROOT / policy["inputs"]["step11_policy"])
    seller_index = step11.load_seller_index(ROOT / policy["inputs"]["seller_profiles"])
    scored_rows, _ = load_csv(ROOT / policy["inputs"]["step11_scored_pairs"])
    cluster_rows, _ = load_csv(ROOT / policy["inputs"]["step11_clusters"])
    frozen_rows, _ = load_csv(ROOT / policy["inputs"]["active_frozen_labels"])
    active_queue_rows, _ = load_csv(ROOT / policy["inputs"]["active_review_queue"])

    selected_scorer = step11_summary["selected_scorer"]
    graph_primary_threshold = float(selected_scorer["graph_primary_threshold"])
    scorer_token = str(selected_scorer["scorer_token"])

    pair_score_lookup = {row["pair_uid"]: float(row["prob_positive"]) for row in scored_rows}
    threshold_edges = [row for row in scored_rows if pair_score_lookup[row["pair_uid"]] >= graph_primary_threshold]
    kept_edges, filter_diagnostics = step11.apply_graph_edge_filters(
        threshold_edges,
        pair_score_lookup,
        scorer_token,
        step11_policy,
    )
    threshold_view = find_threshold_view(step11_summary, graph_primary_threshold)
    assert_source_graph_usable(step11_summary, threshold_view, graph_primary_threshold)

    kept_index = {
        canonical_pair_uid(row["seller_uid_left"], row["seller_uid_right"]): row
        for row in kept_edges
    }
    frozen_index = {row["pair_uid"]: row for row in frozen_rows}
    active_queue_index = {row["pair_uid"]: row for row in active_queue_rows}

    queued_rows: list[dict] = []
    rereview_rows: list[dict] = []
    skipped_existing: list[dict] = []
    missing_pairs: list[dict] = []

    cluster_summary = {}
    for cluster_key, cluster_cfg in policy["selection"]["clusters"].items():
        selection_mode = cluster_cfg.get("selection_mode", "explicit_pairs")
        cluster_items = []
        if selection_mode == "all_retained_cluster_edges":
            clusters = cluster_member_sets(cluster_rows)
            for cluster_rank in sorted(clusters):
                cluster = clusters[cluster_rank]
                for row in cluster_edges(kept_edges, cluster["members"]):
                    cluster_items.append(
                        {
                            "cluster_rank": cluster_rank,
                            "cluster_id": cluster["cluster_id"],
                            "row": row,
                            "edge_policy": policy_for_edge(cluster_cfg, row, cluster_rank),
                        }
                    )
        elif selection_mode == "explicit_pairs":
            cluster_rank = int(cluster_cfg["cluster_rank"])
            cluster_id = base.cluster_id_for_rank(cluster_rows, cluster_rank)
            for explicit_pair in cluster_cfg.get("explicit_pairs", []):
                row = base.lookup_pair(
                    kept_index,
                    explicit_pair["seller_uid_left"],
                    explicit_pair["seller_uid_right"],
                )
                if row is None:
                    missing_pairs.append(
                        {
                            "cluster_key": cluster_key,
                            "cluster_rank": cluster_rank,
                            "seller_uid_left": explicit_pair["seller_uid_left"],
                            "seller_uid_right": explicit_pair["seller_uid_right"],
                            "reason": "not_retained_in_current_primary_graph",
                        }
                    )
                    continue
                cluster_items.append(
                    {
                        "cluster_rank": cluster_rank,
                        "cluster_id": cluster_id,
                        "row": row,
                        "edge_policy": cluster_cfg,
                    }
                )
        else:
            raise SystemExit(f"Unsupported Step 5 v3 selection_mode={selection_mode!r}")

        selected_by_rank: dict[int, dict] = {}
        for item in cluster_items:
            edge_cfg = item["edge_policy"]
            cluster_rank = item["cluster_rank"]
            cluster_id = item["cluster_id"]
            before_queue = len(queued_rows)
            before_rereview = len(rereview_rows)
            queue_or_rereview(
                row=item["row"],
                cluster_cfg=edge_cfg,
                cluster_rank=cluster_rank,
                cluster_id=cluster_id,
                graph_primary_threshold=graph_primary_threshold,
                scorer_token=scorer_token,
                seller_index=seller_index,
                frozen_index=frozen_index,
                active_queue_index=active_queue_index,
                queued_rows=queued_rows,
                rereview_rows=rereview_rows,
                skipped_existing=skipped_existing,
            )
            rank_summary = selected_by_rank.setdefault(
                cluster_rank,
                {
                    "cluster_rank": cluster_rank,
                    "cluster_id": cluster_id,
                    "selected_pair_uids": [],
                    "selected_rereview_pair_uids": [],
                    "requested_pair_count": 0,
                },
            )
            rank_summary["requested_pair_count"] += 1
            pair_uid = item["row"]["pair_uid"]
            if len(queued_rows) > before_queue:
                rank_summary["selected_pair_uids"].append(pair_uid)
            if len(rereview_rows) > before_rereview:
                rank_summary["selected_rereview_pair_uids"].append(pair_uid)

        cluster_summary[cluster_key] = {
            "selection_mode": selection_mode,
            "cluster_count": len(selected_by_rank),
            "selected_edge_count": sum(item["requested_pair_count"] for item in selected_by_rank.values()),
            "clusters": [selected_by_rank[rank] for rank in sorted(selected_by_rank)],
        }

    selected_edge_count = sum(item["selected_edge_count"] for item in cluster_summary.values())
    consistency_checks = assert_graph_consistency(
        threshold_edges=threshold_edges,
        kept_edges=kept_edges,
        filter_diagnostics=filter_diagnostics,
        threshold_view=threshold_view,
        selected_edge_count=selected_edge_count,
    )

    for idx, row in enumerate(queued_rows, start=1):
        row["balanced_review_rank"] = idx
    for idx, row in enumerate(rereview_rows, start=1):
        row["balanced_review_rank"] = idx

    fieldnames = list((queued_rows or rereview_rows)[0].keys()) if (queued_rows or rereview_rows) else [
        "balanced_review_rank",
        "review_stratum",
        "pair_uid",
        "candidate_scope",
        "review_priority",
        "candidate_rule_hits",
        "candidate_rank_score",
        "alias_relation",
        "same_market_raw",
        "source_market_raw_left",
        "source_market_raw_right",
        "source_seller_raw_left",
        "source_seller_raw_right",
        "shared_contact_values",
        "shared_title_values",
        "shared_description_values",
        "shared_category_values",
        "shared_pgp_fingerprint_values",
        "lexical_similarity",
        "structural_support_score",
        "left_preview",
        "right_preview",
        "review_status",
        "review_label",
        "reviewer_id",
        "review_notes",
        "target_bucket",
        "target_reason",
        "source_step11_cluster_rank",
        "source_step11_cluster_id",
        "source_step11_prob_positive",
        "source_step11_graph_primary_threshold",
        "source_step11_scorer_token",
        "suggested_label",
        "suggested_confidence",
        "existing_boundary_source",
        "existing_review_status",
        "existing_review_label",
        "existing_reviewer_id",
        "existing_review_notes",
        "target_action",
    ]

    output_csv = ROOT / policy["outputs"]["targeted_review_queue"]
    rereview_csv = ROOT / policy["outputs"]["targeted_rereview_queue"]
    output_summary = ROOT / policy["outputs"]["summary"]
    write_csv(output_csv, queued_rows, fieldnames)
    write_csv(rereview_csv, rereview_rows, fieldnames)

    summary = {
        "queue_version": policy["queue_version"],
        "scope": policy["scope"],
        "milestone_prerequisite": policy["milestone_prerequisite"],
        "active_boundary_kept_fixed": True,
        "source_step11_summary": policy["inputs"]["step11_summary"],
        "source_step11_scorer_token": scorer_token,
        "source_step11_pair_score_distribution": step11_summary.get("pair_score_distribution", {}),
        "source_step11_threshold_view": {
            "threshold": threshold_view.get("threshold"),
            "threshold_token": threshold_view.get("threshold_token"),
            "threshold_pass_edge_count": threshold_view.get("threshold_pass_edge_count"),
            "edge_count": threshold_view.get("edge_count"),
            "cluster_count": threshold_view.get("cluster_count"),
            "cluster_member_count": threshold_view.get("cluster_member_count"),
            "largest_cluster_size": threshold_view.get("largest_cluster_size"),
        },
        "graph_primary_threshold": round(graph_primary_threshold, 6),
        "graph_edge_filtering": filter_diagnostics,
        "consistency_checks": consistency_checks,
        "targeted_review_queue_path": policy["outputs"]["targeted_review_queue"],
        "targeted_rereview_queue_path": policy["outputs"]["targeted_rereview_queue"],
        "targeted_review_queue_row_count": len(queued_rows),
        "targeted_rereview_queue_row_count": len(rereview_rows),
        "cluster_selection": cluster_summary,
        "review_guidelines": policy["review_guidelines"],
        "anchor_controls_not_queued": policy.get("anchor_controls_not_queued", []),
        "missing_or_not_retained_pairs": missing_pairs,
        "skipped_existing": skipped_existing,
    }
    write_json(output_summary, summary)

    print(f"Wrote Step 5 v3 targeted review queue: {output_csv}")
    print(f"Wrote Step 5 v3 targeted rereview queue: {rereview_csv}")
    print(f"Wrote Step 5 v3 review summary: {output_summary}")
    print(
        f"queued_rows={len(queued_rows)} rereview_rows={len(rereview_rows)} "
        f"missing_pairs={len(missing_pairs)} skipped_existing={len(skipped_existing)}"
    )


if __name__ == "__main__":
    main()
