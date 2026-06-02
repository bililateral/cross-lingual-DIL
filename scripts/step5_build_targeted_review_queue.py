from __future__ import annotations

import csv
import json
from pathlib import Path

import step11_cluster_chinese_graph as step11


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_targeted_review_policy.json"


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
    return f"{left}||{right}" if left <= right else f"{right}||{left}"


def canonical_edge_tuple(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def format_review_preview(profile: dict) -> str:
    if "signature_title_preview" not in profile:
        profile = step11.seller_preview_fields(profile)
    title = str(profile.get("signature_title_preview", "") or "").strip()
    description = str(profile.get("signature_description_preview", "") or "").strip()
    category = str(profile.get("top_category_preview", "") or "").strip()
    contact = str(profile.get("contact_preview", "") or "").strip()
    parts = []
    if title:
        parts.append(f"T: {title}")
    if description:
        parts.append(f"D: {description}")
    if category:
        parts.append(f"C: {category}")
    if contact:
        parts.append(f"Contact: {contact}")
    return " | ".join(parts)


def extract_exact_shared_contacts(left_profile: dict, right_profile: dict) -> str:
    left_signals = left_profile.get("contact_signals") or {}
    right_signals = right_profile.get("contact_signals") or {}
    shared = []
    for contact_type in ("email", "telegram", "wickr", "wechat", "qq", "phone"):
        left_values = {
            str(item.get("value", "")).strip()
            for item in left_signals.get(contact_type, []) or []
            if isinstance(item, dict) and str(item.get("value", "")).strip()
        }
        right_values = {
            str(item.get("value", "")).strip()
            for item in right_signals.get(contact_type, []) or []
            if isinstance(item, dict) and str(item.get("value", "")).strip()
        }
        for value in sorted(left_values & right_values):
            shared.append(f"{contact_type}:{value}")
    return " || ".join(shared)


def cluster_members(cluster_rows: list[dict], cluster_rank: int) -> set[str]:
    rank_token = str(cluster_rank)
    return {row["seller_uid"] for row in cluster_rows if row.get("cluster_rank") == rank_token}


def cluster_id_for_rank(cluster_rows: list[dict], cluster_rank: int) -> str:
    rank_token = str(cluster_rank)
    for row in cluster_rows:
        if row.get("cluster_rank") == rank_token:
            return row["cluster_id"]
    raise SystemExit(f"Missing cluster_id for rank={cluster_rank}")


def select_cluster_one_template_edges(kept_edges: list[dict], members: set[str], limit_top: int, limit_tail: int) -> list[dict]:
    cluster_edges = [
        row
        for row in kept_edges
        if row["seller_uid_left"] in members and row["seller_uid_right"] in members
    ]
    cross_market_template_edges = [
        row
        for row in cluster_edges
        if row["review_stratum"] == "text_clone_primary"
        and not normalize_bool(row.get("same_market_raw_bool"))
        and int(float(row.get("has_shared_contact_exact", 0) or 0)) == 0
    ]
    cross_market_template_edges.sort(
        key=lambda row: (
            -float(row["prob_positive"]),
            canonical_edge_tuple(row["seller_uid_left"], row["seller_uid_right"]),
        )
    )
    selected = cross_market_template_edges[:limit_top]
    if limit_tail > 0 and cross_market_template_edges:
        tail_candidates = sorted(
            cross_market_template_edges,
            key=lambda row: (
                float(row["prob_positive"]),
                canonical_edge_tuple(row["seller_uid_left"], row["seller_uid_right"]),
            ),
        )
        for row in tail_candidates:
            pair_key = canonical_edge_tuple(row["seller_uid_left"], row["seller_uid_right"])
            if all(canonical_edge_tuple(item["seller_uid_left"], item["seller_uid_right"]) != pair_key for item in selected):
                selected.append(row)
                if len(selected) >= limit_top + limit_tail:
                    break
    return selected


def lookup_pair(pair_index: dict[str, dict], left: str, right: str) -> dict | None:
    return pair_index.get(canonical_pair_uid(left, right))


def build_row(
    *,
    rank: int,
    row: dict,
    left_profile: dict,
    right_profile: dict,
    target_reason: str,
    target_bucket: str,
    cluster_rank: int,
    cluster_id: str,
    graph_primary_threshold: float,
    scorer_token: str,
    suggested_label: str,
    suggested_confidence: str,
    review_notes: str,
) -> dict:
    same_market = normalize_bool(row.get("same_market_raw_bool"))
    shared_contacts = extract_exact_shared_contacts(left_profile, right_profile)
    shared_title_values = (
        f"shared_title_count_capped={row.get('shared_title_count_capped', '')}"
        if int(float(row.get("shared_title_count_capped", 0) or 0)) > 0
        else ""
    )
    shared_description_values = (
        f"shared_description_count_capped={row.get('shared_description_count_capped', '')}"
        if int(float(row.get("shared_description_count_capped", 0) or 0)) > 0
        else ""
    )
    shared_category_values = (
        f"profile_category_jaccard={row.get('profile_category_jaccard', '')}"
        if float(row.get("profile_category_jaccard", 0.0) or 0.0) > 0.0
        else ""
    )
    return {
        "balanced_review_rank": rank,
        "review_stratum": row["review_stratum"],
        "pair_uid": row["pair_uid"],
        "candidate_scope": row.get("candidate_scope", "sockpuppet_primary"),
        "review_priority": "high",
        "candidate_rule_hits": target_reason,
        "candidate_rank_score": row["prob_positive"],
        "alias_relation": "different_alias",
        "same_market_raw": str(same_market).lower(),
        "source_market_raw_left": row["source_market_raw_left"],
        "source_market_raw_right": row["source_market_raw_right"],
        "source_seller_raw_left": row["source_seller_raw_left"],
        "source_seller_raw_right": row["source_seller_raw_right"],
        "shared_contact_values": shared_contacts,
        "shared_title_values": shared_title_values,
        "shared_description_values": shared_description_values,
        "shared_category_values": shared_category_values,
        "shared_pgp_fingerprint_values": "",
        "lexical_similarity": row.get("sparse_lexical_similarity_raw", ""),
        "structural_support_score": row.get("structural_support_score_raw", ""),
        "left_preview": format_review_preview(left_profile),
        "right_preview": format_review_preview(right_profile),
        "review_status": "pending",
        "review_label": "",
        "reviewer_id": "",
        "review_notes": review_notes,
        "target_bucket": target_bucket,
        "target_reason": target_reason,
        "source_step11_cluster_rank": cluster_rank,
        "source_step11_cluster_id": cluster_id,
        "source_step11_prob_positive": row["prob_positive"],
        "source_step11_graph_primary_threshold": graph_primary_threshold,
        "source_step11_scorer_token": scorer_token,
        "suggested_label": suggested_label,
        "suggested_confidence": suggested_confidence,
    }


def main() -> None:
    policy = load_json(POLICY_PATH)

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
    kept_index = {
        canonical_pair_uid(row["seller_uid_left"], row["seller_uid_right"]): row
        for row in kept_edges
    }
    frozen_index = {row["pair_uid"]: row for row in frozen_rows}
    active_queue_index = {row["pair_uid"]: row for row in active_queue_rows}
    frozen_pair_uids = set(frozen_index)
    active_queue_pair_uids = set(active_queue_index)

    queued_rows: list[dict] = []
    rereview_rows: list[dict] = []
    skipped_existing = []

    def queue_or_rereview(
        *,
        row: dict,
        target_reason: str,
        target_bucket: str,
        cluster_rank: int,
        cluster_id: str,
        suggested_label: str,
        suggested_confidence: str,
        review_notes: str,
    ) -> None:
        pair_uid = row["pair_uid"]
        left_profile = seller_index[row["seller_uid_left"]]
        right_profile = seller_index[row["seller_uid_right"]]
        base = build_row(
            rank=0,
            row=row,
            left_profile=left_profile,
            right_profile=right_profile,
            target_reason=target_reason,
            target_bucket=target_bucket,
            cluster_rank=cluster_rank,
            cluster_id=cluster_id,
            graph_primary_threshold=graph_primary_threshold,
            scorer_token=scorer_token,
            suggested_label=suggested_label,
            suggested_confidence=suggested_confidence,
            review_notes=review_notes,
        )
        if pair_uid in frozen_pair_uids:
            existing = frozen_index[pair_uid]
            base["existing_boundary_source"] = "active_frozen_labels"
            base["existing_review_status"] = existing.get("review_status", "")
            base["existing_review_label"] = existing.get("review_label", "")
            base["existing_reviewer_id"] = existing.get("reviewer_id", "")
            base["existing_review_notes"] = existing.get("review_notes", "")
            base["target_action"] = "rereview_existing_frozen_label"
            rereview_rows.append(base)
            skipped_existing.append({"pair_uid": pair_uid, "reason": "already_in_active_boundary::frozen"})
            return
        if pair_uid in active_queue_pair_uids:
            existing = active_queue_index[pair_uid]
            base["existing_boundary_source"] = "active_review_queue"
            base["existing_review_status"] = existing.get("review_status", "")
            base["existing_review_label"] = existing.get("review_label", "")
            base["existing_reviewer_id"] = existing.get("reviewer_id", "")
            base["existing_review_notes"] = existing.get("review_notes", "")
            base["target_action"] = (
                "rereview_existing_reviewed_queue_row"
                if str(existing.get("review_label", "")).strip()
                else "review_existing_pending_queue_row"
            )
            rereview_rows.append(base)
            skipped_existing.append({"pair_uid": pair_uid, "reason": "already_in_active_boundary::queue"})
            return
        base["target_action"] = "new_targeted_review"
        queued_rows.append(base)

    cluster_one_cfg = policy["selection"]["cluster_one_template_review"]
    cluster_one_rank = int(cluster_one_cfg["cluster_rank"])
    cluster_one_members = cluster_members(cluster_rows, cluster_one_rank)
    cluster_one_id = cluster_id_for_rank(cluster_rows, cluster_one_rank)
    cluster_one_selected = select_cluster_one_template_edges(
        kept_edges,
        cluster_one_members,
        limit_top=int(cluster_one_cfg.get("top_cross_market_limit", 0) or 0),
        limit_tail=int(cluster_one_cfg.get("tail_cross_market_limit", 0) or 0),
    )
    for row in cluster_one_selected:
        note = (
            "Step 11 retained inside the 14-node four-piece template clique. "
            "Cross-market, no shared contact anchor, pure text_clone_primary; review as potential hard negative "
            "unless business evidence supports the same controller."
        )
        queue_or_rereview(
            row=row,
            target_reason="step11_cluster1_template_clique|cross_market_text_clone",
            target_bucket="cluster1_template_copy",
            cluster_rank=cluster_one_rank,
            cluster_id=cluster_one_id,
            suggested_label="negative_candidate",
            suggested_confidence="medium",
            review_notes=note,
        )

    pair_index = {
        canonical_pair_uid(row["seller_uid_left"], row["seller_uid_right"]): row
        for row in kept_edges
    }
    cluster_two_cfg = policy["selection"]["cluster_two_mixed_review"]
    cluster_two_rank = int(cluster_two_cfg["cluster_rank"])
    cluster_two_id = cluster_id_for_rank(cluster_rows, cluster_two_rank)
    missing_pairs = []
    for item in cluster_two_cfg.get("explicit_pairs", []):
        left = item["seller_uid_left"]
        right = item["seller_uid_right"]
        row = lookup_pair(pair_index, left, right)
        if row is None:
            missing_pairs.append(
                {
                    "seller_uid_left": left,
                    "seller_uid_right": right,
                    "reason": "not_retained_in_current_primary_graph",
                    "suggested_label": item.get("suggested_label"),
                }
            )
            continue
        note = item["review_notes"]
        queue_or_rereview(
            row=row,
            target_reason=item["target_reason"],
            target_bucket="cluster2_mixed_expansion",
            cluster_rank=cluster_two_rank,
            cluster_id=cluster_two_id,
            suggested_label=item.get("suggested_label", "negative_candidate"),
            suggested_confidence=item.get("suggested_confidence", "medium"),
            review_notes=note,
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
        "active_boundary_kept_fixed": True,
        "source_step11_summary": policy["inputs"]["step11_summary"],
        "source_step11_scorer_token": scorer_token,
        "graph_primary_threshold": round(graph_primary_threshold, 6),
        "graph_edge_filtering": filter_diagnostics,
        "targeted_review_queue_path": policy["outputs"]["targeted_review_queue"],
        "targeted_rereview_queue_path": policy["outputs"]["targeted_rereview_queue"],
        "targeted_review_queue_row_count": len(queued_rows),
        "targeted_rereview_queue_row_count": len(rereview_rows),
        "cluster_one_template_review": {
            "cluster_rank": cluster_one_rank,
            "cluster_id": cluster_one_id,
            "selected_pair_uids": [row["pair_uid"] for row in queued_rows if row["target_bucket"] == "cluster1_template_copy"],
            "selected_rereview_pair_uids": [row["pair_uid"] for row in rereview_rows if row["target_bucket"] == "cluster1_template_copy"],
        },
        "cluster_two_mixed_review": {
            "cluster_rank": cluster_two_rank,
            "cluster_id": cluster_two_id,
            "selected_pair_uids": [row["pair_uid"] for row in queued_rows if row["target_bucket"] == "cluster2_mixed_expansion"],
            "selected_rereview_pair_uids": [row["pair_uid"] for row in rereview_rows if row["target_bucket"] == "cluster2_mixed_expansion"],
            "missing_or_not_retained_pairs": missing_pairs,
        },
        "anchor_controls_not_queued": policy.get("anchor_controls_not_queued", []),
        "skipped_existing": skipped_existing,
    }
    write_json(output_summary, summary)

    print(f"Wrote targeted review queue: {output_csv}")
    print(f"Wrote targeted rereview queue: {rereview_csv}")
    print(f"Wrote targeted review summary: {output_summary}")
    print(
        f"queued_rows={len(queued_rows)} rereview_rows={len(rereview_rows)} "
        f"missing_pairs={len(missing_pairs)} skipped_existing={len(skipped_existing)}"
    )


if __name__ == "__main__":
    main()
