#!/usr/bin/env python3
"""Materialize the reviewed Step15-v8 readiness expansion as an isolated freeze."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import step3_build_seller_profiles as step3
import step4_build_silver_candidates as step4
import step7_build_pair_feature_preview as preview
import step15_v8_common as common
import step16_apply_v8_context_reviews as context_apply
import step16_reconcile_v8_identity_control_reviews as identity_review


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_POLICY = ROOT / "schema" / "step15_v8_contextual_evidence_policy.json"
DEFAULT_CONTEXT_SUMMARY = (
    ROOT
    / "reports"
    / "step15_v8"
    / "validation_expansion_queue_v2_20260714"
    / "context_review"
    / "step16_v8_context_review_summary.json"
)
DEFAULT_IDENTITY_SUMMARY = (
    ROOT
    / "reports"
    / "step15_v8"
    / "identity_control_review_20260715"
    / "identity_control_review"
    / "identity_control_review_summary.json"
)

READINESS_REQUIREMENTS = {
    "valid": {
        "state_backed_public_noise_negative": 20,
        "state_backed_verified_direct_positive": 20,
        "same_controller_component_anchor_positive": 15,
    },
    "train": {
        "state_backed_public_noise_negative": 20,
        "state_backed_verified_direct_positive": 30,
        "same_controller_component_anchor_positive": 10,
    },
}

NEW_LABEL_FIELDS = [
    "primary_identity_model_eligible",
    "evidence_expert_eligible",
    "evidence_expert_validation_eligible",
    "identity_control_role",
]
NEW_EVIDENCE_FIELDS = list(NEW_LABEL_FIELDS)


def readiness_row_eligible(row: dict) -> bool:
    if row.get("review_label") not in {"positive", "negative"}:
        return False
    primary = (
        row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
        and str(row.get("primary_identity_model_eligible", "1")).strip() != "0"
    )
    evidence_control = (
        str(row.get("primary_identity_model_eligible", "1")).strip() == "0"
        and str(row.get("evidence_expert_eligible", "0")).strip() == "1"
    )
    return primary or evidence_control


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def render_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")


def pair_uid(left: str, right: str) -> str:
    return "||".join(sorted((left, right)))


def deterministic_rank(run_id: str, slice_name: str, uid: str) -> str:
    return hashlib.sha256(f"{run_id}|{slice_name}|{uid}".encode("utf-8")).hexdigest()


def select_quota(
    rows: list[dict], run_id: str, slice_name: str, count: int
) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            deterministic_rank(run_id, slice_name, row["selection_uid"]),
            row["selection_uid"],
        ),
    )
    if len(ordered) < count:
        raise ValueError(
            f"Insufficient reviewed rows for {slice_name}: required={count} observed={len(ordered)}"
        )
    return ordered[:count]


def select_quota_component_safe(
    rows: list[dict],
    run_id: str,
    slice_name: str,
    count: int,
    reserved_split: dict[str, str],
    seller_keys,
) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            deterministic_rank(run_id, slice_name, row["selection_uid"]),
            row["selection_uid"],
        ),
    )
    selected = []
    for row in ordered:
        split = row["assigned_split"]
        sellers = {seller for seller in seller_keys(row) if seller}
        if any(
            seller in reserved_split and reserved_split[seller] != split
            for seller in sellers
        ):
            continue
        selected.append(row)
        for seller in sellers:
            reserved_split[seller] = split
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"Insufficient component-safe reviewed rows for {slice_name}: "
            f"required={count} observed={len(selected)} available={len(rows)}"
        )
    return selected


def validate_context_summary(summary_path: Path, policy: dict) -> tuple[dict, dict[str, dict]]:
    summary = common.load_json(summary_path)
    expected = summary.get("summary_sha256")
    unsigned = dict(summary)
    unsigned.pop("summary_sha256", None)
    if expected != common.canonical_hash(unsigned):
        raise ValueError("Context review summary self-hash is invalid")
    producer = ROOT / "scripts" / "step16_build_v8_context_review_queues.py"
    if summary.get("producer_sha256") != common.sha256(producer):
        raise ValueError("Context review producer changed after queue generation")
    pool = policy["pools"]["zh_target_strict"]
    inputs = {
        "item_identity_signals_sha256": resolve(pool["item_identity_signals"]),
        "frozen_labels_sha256": resolve(pool["frozen_labels"]),
        "step4_candidates_sha256": resolve(pool["step4_candidates"]),
        "v7_pair_features_sha256": resolve(pool["v7_pair_features"]),
        "v7_clean_e5_metadata_sha256": resolve(pool["v7_clean_e5_metadata"]),
    }
    for key, path in inputs.items():
        if summary["inputs"].get(key) != common.sha256(path):
            raise ValueError(f"Context review input changed: {path}")
    candidates = {}
    for kind, record in summary["outputs"].items():
        path = resolve(record["path"])
        if common.sha256(path) != record["sha256"]:
            raise ValueError(f"Context candidate queue changed: {path}")
        for row in common.load_csv(path):
            uid = row["review_candidate_uid"]
            if uid in candidates or row["queue_kind"] != kind:
                raise ValueError(f"Invalid context candidate UID/kind: {uid}")
            candidates[uid] = row
    return summary, candidates


def resolve_context_reviews(
    summary: dict,
    candidates: dict[str, dict],
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    adjudication_path: Path,
) -> list[dict]:
    template_record = summary["blind_review_templates"]["reviewer_a"]
    template_path = resolve(template_record["path"])
    if common.sha256(template_path) != template_record["sha256"]:
        raise ValueError("Context blind template changed")
    template_rows = common.load_csv(template_path)
    universe = {row["review_candidate_uid"] for row in template_rows}
    candidate_index = {uid: candidates[uid] for uid in universe}
    reviewer_a = context_apply.load_completed_blind_packet(
        reviewer_a_path, candidate_index, "a", True
    )
    reviewer_b = context_apply.load_completed_blind_packet(
        reviewer_b_path, candidate_index, "b", True
    )
    adjudication = context_apply.load_completed_blind_packet(
        adjudication_path, candidate_index, "adjudicator", False
    )
    protocol = {
        "require_distinct_reviewer_ids": True,
        "allowed_identity_labels": ["positive", "negative", "uncertain"],
        "allowed_evidence_types": [
            "same_controller_direct_identifier",
            "public_contact_or_url_noise",
            "uncertain_insufficient_evidence",
        ],
        "adjudicator_must_differ_from_both_reviewers": True,
        "require_high_confidence_for_supervision": True,
        "accepted_queue_decisions": {
            kind: {
                "positive": ["same_controller_direct_identifier"],
                "negative": ["public_contact_or_url_noise"],
                "uncertain": ["uncertain_insufficient_evidence"],
            }
            for kind in (
                "risky_only_public_noise",
                "mixed_context_identifier",
                "verified_direct_both_sides",
            )
        },
    }
    resolved = []
    for uid in sorted(universe):
        a = reviewer_a[uid]
        b = reviewer_b[uid]
        c = adjudication.get(uid, {})
        combined = {
            "review_candidate_uid": uid,
            "queue_kind": candidate_index[uid]["queue_kind"],
            "reviewer_a_id": a["reviewer_id"],
            "reviewer_a_identity_label": a["identity_label"],
            "reviewer_a_evidence_type": a["evidence_type"],
            "reviewer_a_confidence": a["confidence"],
            "reviewer_a_notes": a["notes"],
            "reviewer_b_id": b["reviewer_id"],
            "reviewer_b_identity_label": b["identity_label"],
            "reviewer_b_evidence_type": b["evidence_type"],
            "reviewer_b_confidence": b["confidence"],
            "reviewer_b_notes": b["notes"],
            "adjudicator_id": c.get("reviewer_id", ""),
            "adjudicated_identity_label": c.get("identity_label", ""),
            "adjudicated_evidence_type": c.get("evidence_type", ""),
            "adjudication_confidence": c.get("confidence", ""),
            "adjudication_notes": c.get("notes", ""),
        }
        decision = context_apply.resolve_review_decision(combined, protocol)
        resolved.append({**candidate_index[uid], **decision})
    unresolved = [row for row in resolved if row["status"] == "requires_adjudication"]
    if unresolved:
        raise ValueError(f"Context review still requires adjudication: {unresolved[0]['review_candidate_uid']}")
    return resolved


def validate_identity_summary(summary_path: Path) -> tuple[dict, list[dict]]:
    summary = common.load_json(summary_path)
    expected = summary.get("summary_self_sha256")
    unsigned = dict(summary)
    unsigned.pop("summary_self_sha256", None)
    if expected != canonical_hash(unsigned):
        raise ValueError("Identity-control summary self-hash is invalid")
    producer = resolve(summary["provenance"]["producer"])
    if sha256(producer) != summary["provenance"]["producer_sha256"]:
        raise ValueError("Identity-control producer changed after generation")
    for record in summary["provenance"]["inputs"].values():
        path = resolve(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Identity-control input changed: {path}")
    master_record = summary["artifacts"]["candidate_master"]
    master_path = summary_path.parent / master_record["filename"]
    if sha256(master_path) != master_record["sha256"]:
        raise ValueError("Identity-control candidate master changed")
    return summary, common.load_csv(master_path)


def resolve_identity_reviews(
    summary: dict,
    master_rows: list[dict],
    reviewer_a_path: Path,
    reviewer_b_path: Path,
) -> list[dict]:
    root = reviewer_a_path.parent
    template_record = summary["artifacts"]["reviewer_a"]
    template_path = root / template_record["filename"]
    if sha256(template_path) != template_record["sha256"]:
        raise ValueError("Identity-control blind template changed")
    template_rows = common.load_csv(template_path)
    template_index = {row["candidate_uid"]: row for row in template_rows}
    reviewer_a = identity_review.load_completed(
        reviewer_a_path, template_index, "a"
    )
    reviewer_b = identity_review.load_completed(
        reviewer_b_path, template_index, "b"
    )
    if {
        row["reviewer_id"].casefold() for row in reviewer_a.values()
    } & {row["reviewer_id"].casefold() for row in reviewer_b.values()}:
        raise ValueError("Identity-control reviewer ids are not independent")
    master = {row["candidate_uid"]: row for row in master_rows}
    if set(master) != set(template_index):
        raise ValueError("Identity-control master/review universe differs")
    resolved = []
    for uid in sorted(template_index):
        a = reviewer_a[uid]
        b = reviewer_b[uid]
        if identity_review.decision(a) != identity_review.decision(b):
            raise ValueError(f"Identity-control review disagreement remains: {uid}")
        label, evidence, confidence = identity_review.decision(a)
        kind = master[uid]["candidate_kind"]
        expected_evidence = (
            "same_controller_component_anchor"
            if kind == "evidence_expert_component_closure_control"
            else "same_controller_direct_identifier"
        )
        if label != "positive" or evidence != expected_evidence or confidence != "high":
            raise ValueError(f"Identity-control candidate is not high-confidence positive: {uid}")
        resolved.append(
            {
                **master[uid],
                "review_label": label,
                "evidence_type": evidence,
                "review_confidence": confidence,
                "reviewer_ids": f"{a['reviewer_id']}+{b['reviewer_id']}",
                "review_reason": f"{a['review_reason']} | {b['review_reason']}",
                "selection_uid": uid,
            }
        )
    return resolved


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def profile_values(profile: dict, key: str) -> set[str]:
    return {
        str(item.get("value", "")).strip()
        for item in profile.get(key, [])
        if isinstance(item, dict) and str(item.get("value", "")).strip()
    }


def build_component_cohort_profiles(
    selected_component_uids: set[str],
    manifest_rows: list[dict],
) -> list[dict]:
    selected_rows = [
        row for row in manifest_rows if row["candidate_uid"] in selected_component_uids
    ]
    by_seller: dict[str, list[dict]] = defaultdict(list)
    for row in selected_rows:
        by_seller[row["cohort_seller_uid"]].append(row)
    expected_sellers = 2 * len(selected_component_uids)
    if len(by_seller) != expected_sellers:
        raise ValueError(
            "Component cohort manifest lacks two sellers per selected candidate: "
            f"expected={expected_sellers} observed={len(by_seller)}"
        )
    raw_profiles: dict[str, dict] = {}
    for seller_uid, rows in sorted(by_seller.items()):
        rows.sort(key=lambda row: int(row["source_record_index"]))
        if len(rows) < 2:
            raise ValueError(f"Component cohort has fewer than two items: {seller_uid}")
        candidate_ids = {row["candidate_uid"] for row in rows}
        vendor_ids = {row["platform_vendor_id"] for row in rows}
        if len(candidate_ids) != 1 or len(vendor_ids) != 1:
            raise ValueError(f"Component cohort provenance is inconsistent: {seller_uid}")
        for position, row in enumerate(rows, start=1):
            meta = {
                "seller_uid": seller_uid,
                "data_bucket": "zh_target_strict_identity_control",
                "source_dataset": "products_data.csv",
                "source_market_raw": "__cross_snapshot_identity_control__",
                "source_seller_raw": "",
                "source_seller_id_raw": "",
                "alias_normalized": "",
                "source_row_number": int(row["source_record_index"]),
            }
            profile = step3.ensure_profile(raw_profiles, meta)
            step3.update_profile(
                profile,
                meta,
                title_raw=row["title"],
                description_raw=row["description"],
                category_raw=row["category"],
                price_raw="",
                structured_snapshot="",
            )
    specificity = step3.build_specificity_catalog(raw_profiles)
    schema = json.loads(
        (ROOT / "schema" / "step3_seller_profile_schema.json").read_text(
            encoding="utf-8"
        )
    )
    compression = schema["compression_policy"]
    return [
        step3.finalize_profile(raw_profiles[uid], compression, specificity)
        for uid in sorted(raw_profiles)
    ]


def sanitized_aux_profile(profile: dict) -> dict:
    output = copy.deepcopy(profile)
    output["source_seller_raw"] = ""
    output["source_seller_id_raw"] = ""
    output["alias_normalized"] = ""
    output["profile_text"] = step3.build_profile_text(output)
    return output


def build_platform_signal(
    seller_uid: str,
    vendor_id: str,
    candidate_uid: str,
    source_dataset: str,
    source_market: str,
) -> dict:
    signal_uid = canonical_hash(
        ["step16_v8_platform_vendor_id", candidate_uid, seller_uid, vendor_id]
    )
    return {
        "signal_uid": signal_uid,
        "data_bucket": "zh_target_strict_identity_control",
        "source_dataset": source_dataset,
        "source_row_number": "",
        "seller_uid": seller_uid,
        "source_market_raw": source_market,
        "source_seller_raw": "",
        "source_seller_id_raw": "",
        "alias_normalized": "",
        "source_field": "reviewed_cross_snapshot_platform_vendor_id",
        "contact_type": "platform_vendor_id",
        "normalized_value": vendor_id,
        "raw_value": vendor_id,
        "evidence_level": "reviewed_direct_platform_identity",
        "seller_facing_context": "1",
        "product_data_risk_context": "0",
        "direct_identity_eligible": "1",
        "support_only": "0",
        "context": (
            "Independently reviewed cross-snapshot platform identity continuity; "
            f"candidate={candidate_uid}"
        ),
        "title_snippet": "",
        "description_snippet": "",
    }


def lexical_cosine(left: step4.SellerProfile, right: step4.SellerProfile) -> float:
    if not left.retrieval_norm or not right.retrieval_norm:
        return 0.0
    if len(left.retrieval_weights) > len(right.retrieval_weights):
        left, right = right, left
    dot = sum(
        weight * right.retrieval_weights.get(term, 0.0)
        for term, weight in left.retrieval_weights.items()
    )
    return float(dot / (left.retrieval_norm * right.retrieval_norm))


def build_step4_profile_index(profiles: list[dict]) -> dict[str, step4.SellerProfile]:
    schema = json.loads(
        (ROOT / "schema" / "step4_silver_candidate_schema.json").read_text(
            encoding="utf-8"
        )
    )
    stopwords = set(schema["filtering_policy"]["contact_noise_stopwords"])
    minimums = schema["filtering_policy"]["content_minimums"]
    converted = step4.build_seller_profiles(
        profiles,
        "zh_target_strict",
        "zh",
        stopwords,
        minimums,
        step4.load_pgp_alias_map(),
    )
    step4.compute_retrieval_weights(
        converted, schema["retrieval_policy"]["zh_target_strict"]
    )
    return {profile.seller_uid: profile for profile in converted}


def make_step4_candidate(
    left: step4.SellerProfile,
    right: step4.SellerProfile,
    *,
    review_label: str,
    reviewer_id: str,
    review_notes: str,
    role: str,
    public_shared_types: str = "",
    public_shared_values: str = "",
) -> dict:
    if left.seller_uid > right.seller_uid:
        left, right = right, left
    shared_titles = sorted(set(left.title_norm_to_raw) & set(right.title_norm_to_raw))
    shared_descriptions = sorted(set(left.desc_norm_to_raw) & set(right.desc_norm_to_raw))
    shared_categories = sorted(set(left.category_norm_to_raw) & set(right.category_norm_to_raw))
    shared_contacts = []
    for contact_type in sorted(set(left.contact_values_by_type) & set(right.contact_values_by_type)):
        for value in sorted(
            left.contact_values_by_type[contact_type]
            & right.contact_values_by_type[contact_type]
        ):
            shared_contacts.append(f"{contact_type}:{value}")
    if role == "public_noise" and public_shared_values:
        shared_contacts = [
            token.strip()
            for token in public_shared_values.replace(" || ", "|").split("|")
            if token.strip()
        ]
    support, item_ratio, price_ratio, style_l1 = step4.structural_support(left, right)
    lexical = lexical_cosine(left, right)
    rule_hits = set()
    if shared_titles:
        rule_hits.add("shared_title_clone")
    if shared_descriptions:
        rule_hits.add("shared_description_clone")
    if shared_contacts:
        rule_hits.add("shared_contact_exact")
    if lexical > 0.0:
        rule_hits.add("profile_lexical_neighbor")
    if support >= 0.55:
        rule_hits.add("structural_support")
    if role == "direct_control":
        rule_hits.add("cross_snapshot_platform_vendor_id_control")
    elif role == "component_control":
        rule_hits.add("reviewed_component_closure_control")
    allowlist = {
        "shared_title_clone",
        "shared_description_clone",
        "profile_lexical_neighbor",
        "structural_support",
    }
    title_values = [left.title_norm_to_raw[value] for value in shared_titles[:5]]
    description_values = [left.desc_norm_to_raw[value] for value in shared_descriptions[:3]]
    category_values = [left.category_norm_to_raw[value] for value in shared_categories[:5]]
    source_left = "" if role != "public_noise" else left.source_seller_raw
    source_right = "" if role != "public_noise" else right.source_seller_raw
    return {
        "pair_uid": pair_uid(left.seller_uid, right.seller_uid),
        "candidate_language": "zh",
        "data_bucket": "zh_target_strict",
        "candidate_scope": (
            "sockpuppet_primary" if role == "public_noise" else "evidence_expert_control"
        ),
        "seller_uid_left": left.seller_uid,
        "seller_uid_right": right.seller_uid,
        "source_market_raw_left": left.source_market_raw,
        "source_market_raw_right": right.source_market_raw,
        "source_seller_raw_left": source_left,
        "source_seller_raw_right": source_right,
        "alias_normalized_left": "" if role != "public_noise" else left.alias_normalized,
        "alias_normalized_right": "" if role != "public_noise" else right.alias_normalized,
        "alias_relation": "alias_missing",
        "same_market_raw": str(left.source_market_raw == right.source_market_raw).lower(),
        "item_count_left": str(left.item_count),
        "item_count_right": str(right.item_count),
        "shared_contact_count": str(len(shared_contacts)),
        "shared_contact_types": public_shared_types
        if role == "public_noise" and public_shared_types
        else "|".join(sorted({value.split(":", 1)[0] for value in shared_contacts})),
        "shared_contact_values": " || ".join(shared_contacts[:5]),
        "shared_title_count": str(len(shared_titles)),
        "shared_title_values": " || ".join(title_values),
        "shared_description_count": str(len(shared_descriptions)),
        "shared_description_values": " || ".join(value[:180] for value in description_values),
        "shared_category_count": str(len(category_values)),
        "shared_category_values": " || ".join(category_values),
        "lexical_similarity": f"{lexical:.6f}",
        "structural_support_score": f"{support:.6f}",
        "item_count_ratio": "" if item_ratio is None else f"{item_ratio:.6f}",
        "price_median_ratio": "" if price_ratio is None else f"{price_ratio:.6f}",
        "style_distance_l1": f"{style_l1:.6f}",
        "shared_pgp_fingerprint_count": "0",
        "shared_pgp_fingerprint_values": "",
        "pgp_alias_hit_count_left": "0",
        "pgp_alias_hit_count_right": "0",
        "candidate_rule_hits": "|".join(sorted(rule_hits)),
        "candidate_rule_count": str(len(rule_hits)),
        "candidate_rule_count_non_identifier": str(len(rule_hits & allowlist)),
        "candidate_rank_score": "0.000000",
        "review_priority": "high",
        "left_preview": left.preview,
        "right_preview": right.preview,
        "review_status": "reviewed",
        "review_label": review_label,
        "reviewer_id": reviewer_id,
        "review_notes": review_notes,
    }


def label_row_from_candidate(
    candidate: dict,
    *,
    split: str,
    review_label: str,
    evidence_type: str,
    reviewer_id: str,
    review_notes: str,
    role: str,
    fields: list[str],
) -> dict:
    positive = review_label == "positive"
    control = role in {"public_noise", "direct_control", "component_control"}
    row = {field: "" for field in fields}
    row.update(
        {
            "balanced_review_rank": f"step16v8_{canonical_hash(candidate['pair_uid'])[:16]}",
            "pair_uid": candidate["pair_uid"],
            "data_bucket": "zh_target_strict",
            "candidate_language": "zh",
            "candidate_scope": candidate["candidate_scope"],
            "review_stratum": (
                "identifier_primary"
                if role == "direct_control"
                else (
                    "component_anchor_control"
                    if role == "component_control"
                    else "public_contact_or_url_noise"
                )
            ),
            "review_priority": "high",
            "review_status": "reviewed",
            "review_label": review_label,
            "reviewer_id": reviewer_id,
            "review_notes": review_notes,
            "soft_same_alias_continuity_bool": "0",
            "usable_for_supervision": "0" if control else "1",
            "usable_for_core_transfer": "0" if control else "1",
            "split_name": split,
            "split_component_id": "pending_step16_v8_refreeze",
            "split_component_size": "",
            "seller_uid_left": candidate["seller_uid_left"],
            "seller_uid_right": candidate["seller_uid_right"],
            "source_market_raw_left": candidate["source_market_raw_left"],
            "source_market_raw_right": candidate["source_market_raw_right"],
            "source_seller_raw_left": candidate["source_seller_raw_left"],
            "source_seller_raw_right": candidate["source_seller_raw_right"],
            "alias_relation": candidate["alias_relation"],
            "same_market_raw": candidate["same_market_raw"],
            "candidate_rule_hits": candidate["candidate_rule_hits"],
            "candidate_rank_score": candidate["candidate_rank_score"],
            "lexical_similarity": candidate["lexical_similarity"],
            "structural_support_score": candidate["structural_support_score"],
            "shared_contact_count": candidate["shared_contact_count"],
            "shared_contact_values": candidate["shared_contact_values"],
            "shared_title_count": candidate["shared_title_count"],
            "shared_title_values": candidate["shared_title_values"],
            "shared_description_count": candidate["shared_description_count"],
            "shared_description_values": candidate["shared_description_values"],
            "shared_category_count": candidate["shared_category_count"],
            "shared_category_values": candidate["shared_category_values"],
            "shared_pgp_fingerprint_count": "0",
            "shared_pgp_fingerprint_values": "",
            "left_preview": candidate["left_preview"],
            "right_preview": candidate["right_preview"],
            "label_tier": (
                "gold_cross_snapshot_direct_control"
                if role == "direct_control"
                else (
                    "gold_cross_snapshot_component_control"
                    if role == "component_control"
                    else "high_confidence_silver_agent_reviewed_public_noise_control"
                )
            ),
            "benchmark_eligible": "0" if control else "1",
            "silver_train_only": "0",
            "training_sample_weight": "1.0",
            "silver_positive_reasons": evidence_type if positive else "",
            "silver_negative_reasons": evidence_type if not positive else "",
            "primary_identity_model_eligible": "0" if control else "1",
            "evidence_expert_eligible": "1",
            "evidence_expert_validation_eligible": "1" if control else "0",
            "identity_control_role": (
                "public_noise_control" if role == "public_noise" else role
            )
            if control
            else "",
        }
    )
    return row


def evidence_row_from_label(
    label: dict,
    candidate: dict,
    evidence_type: str,
    fields: list[str],
) -> dict:
    positive = label["review_label"] == "positive"
    row = {field: "" for field in fields}
    row.update(
        {
            "pair_uid": label["pair_uid"],
            "data_bucket": label["data_bucket"],
            "candidate_language": "zh",
            "split_name": label["split_name"],
            "split_component_id": label["split_component_id"],
            "review_label": label["review_label"],
            "review_stratum": label["review_stratum"],
            "usable_for_supervision": label["usable_for_supervision"],
            "usable_for_core_transfer": label["usable_for_core_transfer"],
            "candidate_rule_hits": candidate["candidate_rule_hits"],
            "shared_contact_count": candidate["shared_contact_count"],
            "shared_pgp_fingerprint_count": "0",
            "shared_title_count": candidate["shared_title_count"],
            "shared_description_count": candidate["shared_description_count"],
            "structural_support_score_raw": candidate["structural_support_score"],
            "source_seller_raw_left": candidate["source_seller_raw_left"],
            "source_seller_raw_right": candidate["source_seller_raw_right"],
            "identity_label": "same_controller" if positive else "different_controller",
            "evidence_type": evidence_type,
            "evidence_type_confident": "1",
            "identity_training_eligible": label["primary_identity_model_eligible"],
            "has_direct_identifier_signal": "1"
            if evidence_type == "same_controller_direct_identifier"
            else "0",
            "has_template_clone_signal": "0",
            "has_semantic_topic_signal": "0",
            "has_public_contact_or_url_noise_signal": "1"
            if evidence_type == "public_contact_or_url_noise"
            else "0",
            "evidence_type_reasons": "step16_v8_independent_reviewed_readiness_expansion",
            "primary_identity_model_eligible": label[
                "primary_identity_model_eligible"
            ],
            "evidence_expert_eligible": label["evidence_expert_eligible"],
            "evidence_expert_validation_eligible": label[
                "evidence_expert_validation_eligible"
            ],
            "identity_control_role": label["identity_control_role"],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-v8-policy", default=str(DEFAULT_BASE_POLICY))
    parser.add_argument("--context-summary", default=str(DEFAULT_CONTEXT_SUMMARY))
    parser.add_argument("--context-reviewer-a", default=None)
    parser.add_argument("--context-reviewer-b", default=None)
    parser.add_argument("--context-adjudication", default=None)
    parser.add_argument("--identity-summary", default=str(DEFAULT_IDENTITY_SUMMARY))
    parser.add_argument("--identity-reviewer-a", default=None)
    parser.add_argument("--identity-reviewer-b", default=None)
    parser.add_argument("--run-id", default="readiness_expansion_20260715")
    parser.add_argument(
        "--output-root",
        default="reports/step16_v8_validation_refreeze/readiness_expansion_20260715",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    base_policy_path = resolve(args.base_v8_policy)
    policy = common.load_json(base_policy_path)
    base_v7_path = resolve(policy["frozen_dependencies"]["v7_policy"])
    base_v7 = common.load_json(base_v7_path)
    pre_assignment_path = resolve(
        policy["frozen_dependencies"]["representative_validation_assignments"]
    )
    pre_assignment_rows = common.load_csv(pre_assignment_path)
    reserved_split: dict[str, str] = {}
    for row in pre_assignment_rows:
        split = row["v7_split_name"]
        for seller in (row["seller_uid_left"], row["seller_uid_right"]):
            prior = reserved_split.get(seller)
            if prior is not None and prior != split:
                raise ValueError(
                    f"Frozen assignment already places one seller across splits: {seller}"
                )
            reserved_split[seller] = split
    context_summary_path = resolve(args.context_summary)
    context_root = context_summary_path.parent
    context_reviewer_a = resolve(
        args.context_reviewer_a
        or context_root / "reviewer_a_blind_packet.completed.csv"
    )
    context_reviewer_b = resolve(
        args.context_reviewer_b
        or context_root / "reviewer_b_blind_packet.completed.csv"
    )
    context_adjudication = resolve(
        args.context_adjudication
        or context_root / "reviewer_adjudicator_blind_packet.completed.csv"
    )
    identity_summary_path = resolve(args.identity_summary)
    identity_root = identity_summary_path.parent
    identity_reviewer_a = resolve(
        args.identity_reviewer_a
        or identity_root / "reviewer_a_blind_packet.completed.csv"
    )
    identity_reviewer_b = resolve(
        args.identity_reviewer_b
        or identity_root / "reviewer_b_blind_packet.completed.csv"
    )

    context_summary, context_candidates = validate_context_summary(
        context_summary_path, policy
    )
    context_resolved = resolve_context_reviews(
        context_summary,
        context_candidates,
        context_reviewer_a,
        context_reviewer_b,
        context_adjudication,
    )
    allowed_public_splits = {
        "valid": {"valid_only", "valid_candidate"},
        "train": {"train_only", "train_candidate"},
    }
    public_candidate_pools = {}
    public_available = {}
    for split, eligibility in allowed_public_splits.items():
        rows = [
            {
                **row,
                "assigned_split": split,
                "selection_uid": row["review_candidate_uid"],
            }
            for row in context_resolved
            if row["status"] == "resolved_high_confidence"
            and row.get("identity_label") == "negative"
            and row.get("evidence_type") == "public_contact_or_url_noise"
            and row["split_eligibility"] in eligibility
        ]
        public_available[split] = len(rows)
        public_candidate_pools[split] = rows

    identity_summary, identity_master = validate_identity_summary(
        identity_summary_path
    )
    identity_resolved = resolve_identity_reviews(
        identity_summary,
        identity_master,
        identity_reviewer_a,
        identity_reviewer_b,
    )
    identity_available = {}
    identity_specs = {
        "component": (
            "evidence_expert_component_closure_control",
            "same_controller_component_anchor_positive",
        ),
        "direct": (
            "evidence_expert_direct_persistence_control",
            "state_backed_verified_direct_positive",
        ),
    }
    identity_candidate_pools = {}
    for short_name, (kind, readiness_key) in identity_specs.items():
        for split in ("valid", "train"):
            rows = [
                row
                for row in identity_resolved
                if row["candidate_kind"] == kind and row["assigned_split"] == split
            ]
            identity_available[f"{short_name}:{split}"] = len(rows)
            identity_candidate_pools[(short_name, split)] = (
                rows,
                readiness_key,
            )
    selected_public = []
    for split in ("valid", "train"):
        selected_public.extend(
            select_quota_component_safe(
                public_candidate_pools[split],
                args.run_id,
                f"public_noise_{split}",
                READINESS_REQUIREMENTS[split][
                    "state_backed_public_noise_negative"
                ],
                reserved_split,
                lambda item: (item["seller_uid_left"], item["seller_uid_right"]),
            )
        )
    selected_identity = []
    for short_name in ("component", "direct"):
        for split in ("valid", "train"):
            rows, readiness_key = identity_candidate_pools[(short_name, split)]
            selected_identity.extend(
                select_quota_component_safe(
                    rows,
                    args.run_id,
                    f"{short_name}_{split}",
                    READINESS_REQUIREMENTS[split][readiness_key],
                    reserved_split,
                    lambda item: (
                        item["seller_uid_left"],
                        item["seller_uid_right"],
                        item["strict_profile_seller_uid"],
                        item["aux_profile_seller_uid"],
                    ),
                )
            )
    selected_direct = [
        row
        for row in selected_identity
        if row["candidate_kind"]
        == "evidence_expert_direct_persistence_control"
    ]
    selected_component = [
        row
        for row in selected_identity
        if row["candidate_kind"]
        == "evidence_expert_component_closure_control"
    ]
    if {row["platform_vendor_id"] for row in selected_direct} & {
        row["platform_vendor_id"] for row in selected_component
    }:
        raise ValueError("Selected direct/component controls reuse a raw vendor id")

    zh_pool = policy["pools"]["zh_target_strict"]
    label_path = resolve(zh_pool["frozen_labels"])
    evidence_path = resolve(zh_pool["evidence_labels"])
    profile_path = resolve(zh_pool["seller_profiles"])
    signal_path = resolve(zh_pool["item_identity_signals"])
    step4_path = resolve(zh_pool["step4_candidates"])
    canonical_path = resolve(base_v7["pools"]["zh_target_strict"]["canonical_pair_features"])
    assignment_path = pre_assignment_path
    assignment_manifest_path = resolve(
        policy["frozen_dependencies"]["representative_validation_manifest"]
    )
    labels_original = common.load_csv(label_path)
    evidence_original = common.load_csv(evidence_path)
    profiles_original = step4.load_jsonl(profile_path)
    signals_original = common.load_csv(signal_path)
    step4_original = common.load_csv(step4_path)
    canonical_original = common.load_csv(canonical_path)
    assignments_original = common.load_csv(assignment_path)
    label_fields = list(labels_original[0]) + [
        field for field in NEW_LABEL_FIELDS if field not in labels_original[0]
    ]
    evidence_fields = list(evidence_original[0]) + [
        field for field in NEW_EVIDENCE_FIELDS if field not in evidence_original[0]
    ]
    step4_schema = json.loads(
        (ROOT / "schema" / "step4_silver_candidate_schema.json").read_text(
            encoding="utf-8"
        )
    )
    step4_fields = list(step4_original[0]) + [
        field
        for field in step4_schema["candidate_output_fields"]
        if field not in step4_original[0]
    ]
    feature_schema = json.loads(
        (ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_fields = list(canonical_original[0]) + [
        field
        for field in feature_schema["pair_output_fields"]
        if field not in canonical_original[0]
    ]
    schema_input_paths = [
        ROOT / "schema" / "step4_silver_candidate_schema.json",
        ROOT / "schema" / "step7_transfer_safe_pair_feature_schema.json",
    ]
    identity_transitive_input_paths = [
        resolve(record["path"])
        for record in identity_summary["provenance"]["inputs"].values()
    ]

    aux_profiles_path = resolve(
        identity_summary["provenance"]["inputs"]["aux_profiles"]["path"]
    )
    aux_profiles = {
        row["seller_uid"]: row for row in step4.load_jsonl(aux_profiles_path)
    }
    selected_aux_uids = {row["aux_profile_seller_uid"] for row in selected_direct}
    missing_aux = selected_aux_uids - set(aux_profiles)
    if missing_aux:
        raise ValueError(f"Selected direct control lacks aux profile: {sorted(missing_aux)[0]}")
    cohort_manifest_record = identity_summary["artifacts"]["cohort_manifest"]
    cohort_manifest_path = identity_root / cohort_manifest_record["filename"]
    if sha256(cohort_manifest_path) != cohort_manifest_record["sha256"]:
        raise ValueError("Identity-control cohort manifest changed")
    cohort_profiles = build_component_cohort_profiles(
        {row["candidate_uid"] for row in selected_component},
        common.load_csv(cohort_manifest_path),
    )
    augmented_profiles = list(profiles_original)
    augmented_profiles.extend(
        sanitized_aux_profile(aux_profiles[uid]) for uid in sorted(selected_aux_uids)
    )
    augmented_profiles.extend(cohort_profiles)
    profile_uids = [row["seller_uid"] for row in augmented_profiles]
    if len(profile_uids) != len(set(profile_uids)):
        raise ValueError("Augmented Step3 profile universe contains duplicate sellers")
    profile_index_raw = {row["seller_uid"]: row for row in augmented_profiles}

    signal_fields = list(signals_original[0])
    augmented_signals = [dict(row) for row in signals_original]
    for row in selected_direct:
        for seller_uid, dataset, market in (
            (
                row["strict_profile_seller_uid"],
                row["strict_source_dataset"],
                row["strict_source_market"],
            ),
            (
                row["aux_profile_seller_uid"],
                row["aux_source_dataset"],
                row["aux_source_market"],
            ),
        ):
            augmented_signals.append(
                build_platform_signal(
                    seller_uid,
                    row["platform_vendor_id"],
                    row["candidate_uid"],
                    dataset,
                    market,
                )
            )
    signal_uids = [row["signal_uid"] for row in augmented_signals]
    if len(signal_uids) != len(set(signal_uids)):
        raise ValueError("Augmented item signals contain duplicate signal_uid")

    step4_profiles = build_step4_profile_index(augmented_profiles)
    existing_step4 = {row["pair_uid"]: dict(row) for row in step4_original}
    original_labels_by_uid = {row["pair_uid"]: row for row in labels_original}
    selected_candidate_rows: dict[str, dict] = {}
    new_candidate_rows: dict[str, dict] = {}
    selected_records = []

    def register_candidate(
        candidate: dict,
        *,
        split: str,
        label: str,
        evidence_type: str,
        reviewers: str,
        notes: str,
        role: str,
        selection_uid: str,
    ) -> None:
        uid = candidate["pair_uid"]
        existing_label = original_labels_by_uid.get(uid)
        if existing_label and (
            existing_label.get("review_label") in {"positive", "negative"}
            and existing_label.get("usable_for_supervision") == "1"
        ):
            raise ValueError(f"Readiness expansion attempted to replace supervision: {uid}")
        if uid in selected_candidate_rows:
            raise ValueError(f"Readiness expansion duplicated a new pair: {uid}")
        selected_candidate_rows[uid] = candidate
        if uid not in existing_step4:
            new_candidate_rows[uid] = candidate
        selected_records.append(
            {
                "selection_uid": selection_uid,
                "pair_uid": uid,
                "assigned_split": split,
                "review_label": label,
                "evidence_type": evidence_type,
                "reviewer_ids": reviewers,
                "review_notes": notes,
                "role": role,
                "supersedes_non_supervision_row": bool(existing_label),
            }
        )

    for row in selected_public:
        left_uid, right_uid = row["seller_uid_left"], row["seller_uid_right"]
        candidate = make_step4_candidate(
            step4_profiles[left_uid],
            step4_profiles[right_uid],
            review_label="negative",
            reviewer_id="+".join(row["reviewer_ids"]),
            review_notes=row["review_notes"],
            role="public_noise",
            public_shared_types=row["shared_identifier_types"],
            public_shared_values=row["shared_identifier_values"],
        )
        register_candidate(
            candidate,
            split=row["assigned_split"],
            label="negative",
            evidence_type="public_contact_or_url_noise",
            reviewers="+".join(row["reviewer_ids"]),
            notes=row["review_notes"],
            role="public_noise",
            selection_uid=row["review_candidate_uid"],
        )
    for row in selected_direct:
        candidate = make_step4_candidate(
            step4_profiles[row["seller_uid_left"]],
            step4_profiles[row["seller_uid_right"]],
            review_label="positive",
            reviewer_id=row["reviewer_ids"],
            review_notes=row["review_reason"],
            role="direct_control",
        )
        register_candidate(
            candidate,
            split=row["assigned_split"],
            label="positive",
            evidence_type="same_controller_direct_identifier",
            reviewers=row["reviewer_ids"],
            notes=row["review_reason"],
            role="direct_control",
            selection_uid=row["candidate_uid"],
        )
    for row in selected_component:
        candidate = make_step4_candidate(
            step4_profiles[row["seller_uid_left"]],
            step4_profiles[row["seller_uid_right"]],
            review_label="positive",
            reviewer_id=row["reviewer_ids"],
            review_notes=row["review_reason"],
            role="component_control",
        )
        register_candidate(
            candidate,
            split=row["assigned_split"],
            label="positive",
            evidence_type="same_controller_component_anchor",
            reviewers=row["reviewer_ids"],
            notes=row["review_reason"],
            role="component_control",
            selection_uid=row["candidate_uid"],
        )

    augmented_step4_index = dict(existing_step4)
    augmented_step4_index.update(new_candidate_rows)
    if len(augmented_step4_index) != len(existing_step4) + len(new_candidate_rows):
        raise ValueError("Augmented Step4 candidate universe did not grow exactly once")
    preview_profile_index, preview_groups = preview.prepare_profiles(
        augmented_profiles,
        feature_schema["market_relative_numeric_fields"],
        feature_schema["en_only_auxiliary_fields"],
    )
    new_preview_rows = preview.build_pair_rows(
        list(new_candidate_rows.values()),
        preview_profile_index,
        preview_groups,
        feature_schema,
    )
    canonical_index = {row["pair_uid"]: dict(row) for row in canonical_original}
    for row in new_preview_rows:
        if row["pair_uid"] in canonical_index:
            raise ValueError(f"New canonical Step7 row already exists: {row['pair_uid']}")
        canonical_index[row["pair_uid"]] = row
    if set(canonical_index) != set(augmented_step4_index):
        missing = sorted(set(augmented_step4_index) - set(canonical_index))
        extra = sorted(set(canonical_index) - set(augmented_step4_index))
        raise ValueError(
            f"Augmented Step4/canonical Step7 universes differ: missing={missing[:1]} extra={extra[:1]}"
        )

    labels_index = {}
    for row in labels_original:
        item = dict(row)
        item.setdefault("primary_identity_model_eligible", "1")
        item.setdefault("evidence_expert_eligible", "1")
        item.setdefault("evidence_expert_validation_eligible", "0")
        item.setdefault("identity_control_role", "")
        labels_index[item["pair_uid"]] = item
    evidence_index = {}
    for row in evidence_original:
        item = dict(row)
        item.setdefault("primary_identity_model_eligible", "1")
        item.setdefault("evidence_expert_eligible", "1")
        item.setdefault("evidence_expert_validation_eligible", "0")
        item.setdefault("identity_control_role", "")
        evidence_index[item["pair_uid"]] = item
    for record in selected_records:
        candidate = selected_candidate_rows[record["pair_uid"]]
        label = label_row_from_candidate(
            candidate,
            split=record["assigned_split"],
            review_label=record["review_label"],
            evidence_type=record["evidence_type"],
            reviewer_id=record["reviewer_ids"],
            review_notes=record["review_notes"],
            role=record["role"],
            fields=label_fields,
        )
        evidence = evidence_row_from_label(
            label, candidate, record["evidence_type"], evidence_fields
        )
        labels_index[record["pair_uid"]] = label
        evidence_index[record["pair_uid"]] = evidence

    eligible_labels = [row for row in labels_index.values() if readiness_row_eligible(row)]
    old_assignments = {row["pair_uid"]: row for row in assignments_original}
    new_split_by_uid = {
        record["pair_uid"]: record["assigned_split"] for record in selected_records
    }
    missing_assignments = [
        row["pair_uid"]
        for row in eligible_labels
        if row["pair_uid"] not in old_assignments
        and row["pair_uid"] not in new_split_by_uid
    ]
    if missing_assignments:
        raise ValueError(f"Eligible label lacks a split assignment: {missing_assignments[0]}")
    uf = UnionFind()
    for row in eligible_labels:
        uf.union(row["seller_uid_left"], row["seller_uid_right"])
    selected_identity_by_uid = {
        row["candidate_uid"]: row for row in selected_identity
    }
    for record in selected_records:
        if record["role"] not in {"direct_control", "component_control"}:
            continue
        source = selected_identity_by_uid[record["selection_uid"]]
        strict_uid = source["strict_profile_seller_uid"]
        aux_uid = source["aux_profile_seller_uid"]
        uf.union(strict_uid, aux_uid)
        uf.union(strict_uid, source["seller_uid_left"])
        uf.union(strict_uid, source["seller_uid_right"])
    desired_split_by_uid = {}
    split_reason_by_uid = {}
    for row in eligible_labels:
        uid = row["pair_uid"]
        if uid in old_assignments:
            desired_split_by_uid[uid] = old_assignments[uid]["v7_split_name"]
            split_reason_by_uid[uid] = "retained_frozen_v7_assignment"
        else:
            desired_split_by_uid[uid] = new_split_by_uid[uid]
            split_reason_by_uid[uid] = "step16_v8_score_blind_reviewed_expansion"
    splits_by_component: dict[str, set[str]] = defaultdict(set)
    members_by_component: dict[str, set[str]] = defaultdict(set)
    for seller in list(uf.parent):
        members_by_component[uf.find(seller)].add(seller)
    for row in eligible_labels:
        root = uf.find(row["seller_uid_left"])
        if uf.find(row["seller_uid_right"]) != root:
            raise AssertionError("Seller component closure failed")
        splits_by_component[root].add(desired_split_by_uid[row["pair_uid"]])
    leakage = {
        root: sorted(splits)
        for root, splits in splits_by_component.items()
        if len(splits) > 1
    }
    if leakage:
        first = next(iter(leakage.items()))
        raise ValueError(
            "Step16-v8 reviewed expansion would leak seller components across splits: "
            f"count={len(leakage)} first={first}"
        )
    component_record = {}
    for root, members in members_by_component.items():
        ordered = sorted(members)
        component_record[root] = {
            "component_id": f"v8ready_{canonical_hash(ordered)[:16]}",
            "size": len(ordered),
            "members": ordered,
        }
    assignment_rows = []
    for row in sorted(eligible_labels, key=lambda item: item["pair_uid"]):
        uid = row["pair_uid"]
        component = component_record[uf.find(row["seller_uid_left"])]
        row["split_component_id"] = component["component_id"]
        row["split_component_size"] = str(component["size"])
        evidence_index[uid]["split_component_id"] = component["component_id"]
        assignment_rows.append(
            {
                "pair_uid": uid,
                "split_component_id": component["component_id"],
                "v7_component_id": component["component_id"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "review_label": row["review_label"],
                "evidence_type": evidence_index[uid]["evidence_type"],
                "original_split_name": row["split_name"],
                "v7_split_name": desired_split_by_uid[uid],
                "assignment_reason": split_reason_by_uid[uid],
            }
        )
    old_test = {
        row["pair_uid"]
        for row in assignments_original
        if row["v7_split_name"] == "internal_development_test"
    }
    new_test = {
        row["pair_uid"]
        for row in assignment_rows
        if row["v7_split_name"] == "internal_development_test"
    }
    if old_test != new_test or len(new_test) != 200:
        raise ValueError(
            "Fixed internal development test changed during readiness refreeze: "
            f"old={len(old_test)} new={len(new_test)}"
        )

    assignment_by_uid = {row["pair_uid"]: row for row in assignment_rows}
    train_sellers = {
        seller
        for row in assignment_rows
        if row["v7_split_name"] == "train"
        for seller in (row["seller_uid_left"], row["seller_uid_right"])
    }
    occurrence_by_seller: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sellers_by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in augmented_signals:
        seller = str(row.get("seller_uid", "")).strip()
        token = (
            str(row.get("contact_type", "")).strip().lower(),
            str(row.get("normalized_value", "")).strip().lower(),
        )
        if seller and all(token):
            occurrence_by_seller[seller][token].append(row)
            sellers_by_token[token].add(seller)
    token_df = Counter(
        {token: len(sellers & train_sellers) for token, sellers in sellers_by_token.items()}
    )
    frequency_threshold = int(
        policy["occurrence_evidence_expert"][
            "public_identifier_train_seller_frequency_threshold"
        ]
    )
    readiness = {}
    readiness_states = {}
    for split in ("valid", "train"):
        rows = []
        states = []
        for label in eligible_labels:
            uid = label["pair_uid"]
            if assignment_by_uid[uid]["v7_split_name"] != split:
                continue
            joined = {
                **label,
                "evidence_type": evidence_index[uid]["evidence_type"],
                "domain": "zh",
            }
            state = common.occurrence_evidence(
                joined,
                occurrence_by_seller,
                token_df,
                frequency_threshold,
            )["evidence_state"]
            rows.append(joined)
            states.append(state)
        masks = common.validation_slice_masks(rows, states)
        readiness[split] = {
            key: {
                "observed": int(sum(masks[key])),
                "required": int(required),
                "met": int(sum(masks[key])) >= int(required),
            }
            for key, required in READINESS_REQUIREMENTS[split].items()
        }
        readiness_states[split] = dict(sorted(Counter(states).items()))
    if not all(item["met"] for split in readiness.values() for item in split.values()):
        raise ValueError(
            "Reviewed expansion did not satisfy the preregistered readiness gates: "
            + json.dumps(readiness, ensure_ascii=False, sort_keys=True)
        )

    output_root = resolve(args.output_root)
    output_names = {
        "profiles": "step3_seller_profiles.zh_target_strict.v8_readiness.jsonl",
        "signals": "step3_item_identity_signals.zh_target_strict.v8_readiness.csv",
        "step4": "step4_zh_target_strict_candidates.v8_readiness.csv",
        "labels": "step5_zh_target_strict_labels.v8_readiness.csv",
        "evidence": "step15_evidence_type_labels.zh_target_strict.v8_readiness.csv",
        "canonical": "step7_pair_features.zh_target_strict.canonical.v8_readiness.csv",
        "assignments": "representative_validation_assignments.v8_readiness.csv",
        "assignment_manifest": "representative_validation_manifest.v8_readiness.json",
        "generated_v7_policy": "step15_v7_readiness_policy.json",
        "generated_v8_policy": "step15_v8_readiness_policy.json",
        "readiness_summary": "step16_v8_readiness_expansion_summary.json",
        "freeze_manifest": "step16_v8_readiness_freeze_manifest.json",
    }
    final_paths = {key: output_root / name for key, name in output_names.items()}
    linux_root = output_root / "linux_generated"
    clean_e5_root = linux_root / "clean_embeddings"
    final_v7_feature_paths = {
        "en_content_train_pool": (
            linux_root
            / "features"
            / "step7_pair_features.en_content_train_pool.v8_readiness.csv"
        ),
        "zh_target_strict": (
            linux_root
            / "features"
            / "step7_pair_features.zh_target_strict.v8_readiness.csv"
        ),
    }
    clean_e5_metadata_path = (
        clean_e5_root
        / "multilingual_e5_large_identifier_redacted.zh_target_strict.json"
    )
    clean_e5_matrix_path = (
        clean_e5_root
        / "multilingual_e5_large_identifier_redacted.zh_target_strict.npy"
    )

    ordered_labels = [labels_index[row["pair_uid"]] for row in labels_original]
    ordered_labels.extend(
        labels_index[record["pair_uid"]]
        for record in sorted(selected_records, key=lambda item: item["pair_uid"])
        if record["pair_uid"] not in original_labels_by_uid
    )
    old_evidence_order = [row["pair_uid"] for row in evidence_original]
    ordered_evidence = [evidence_index[uid] for uid in old_evidence_order]
    ordered_evidence.extend(
        evidence_index[record["pair_uid"]]
        for record in sorted(selected_records, key=lambda item: item["pair_uid"])
        if record["pair_uid"] not in set(old_evidence_order)
    )
    ordered_step4 = list(step4_original) + [
        new_candidate_rows[uid] for uid in sorted(new_candidate_rows)
    ]
    ordered_canonical = list(canonical_original) + [
        canonical_index[uid] for uid in sorted(new_candidate_rows)
    ]
    payloads: dict[str, bytes] = {
        "profiles": render_jsonl(augmented_profiles),
        "signals": render_csv(augmented_signals, signal_fields),
        "step4": render_csv(ordered_step4, step4_fields),
        "labels": render_csv(ordered_labels, label_fields),
        "evidence": render_csv(ordered_evidence, evidence_fields),
        "canonical": render_csv(ordered_canonical, canonical_fields),
        "assignments": render_csv(
            assignment_rows,
            [
                "pair_uid",
                "split_component_id",
                "v7_component_id",
                "seller_uid_left",
                "seller_uid_right",
                "review_label",
                "evidence_type",
                "original_split_name",
                "v7_split_name",
                "assignment_reason",
            ],
        ),
    }

    generated_v7 = copy.deepcopy(base_v7)
    generated_v7["version"] = f"{base_v7['version']}-step16-v8-readiness-{args.run_id}"
    generated_v7["readiness_execution_contract"] = {
        "clean_embedding_rebuild_pools": ["zh_target_strict"],
        "inductive_feature_rebuild_pools": sorted(final_v7_feature_paths),
        "reason": (
            "Chinese profiles/pairs changed; the frozen English E5 cache is reused, "
            "while both v7 feature tables are atomically republished under one directory"
        ),
    }
    for pool_name, feature_path in final_v7_feature_paths.items():
        generated_v7["pools"][pool_name]["v7_pair_features"] = rel(feature_path)
    generated_v7_pool = generated_v7["pools"]["zh_target_strict"]
    generated_v7_pool.update(
        {
            "frozen_labels": rel(final_paths["labels"]),
            "evidence_labels": rel(final_paths["evidence"]),
            "seller_profiles": rel(final_paths["profiles"]),
            "item_identity_signals": rel(final_paths["signals"]),
            "canonical_pair_features": rel(final_paths["canonical"]),
            "step4_candidates": rel(final_paths["step4"]),
            "clean_e5_cache_metadata": rel(clean_e5_metadata_path),
            "clean_e5_cache_matrix": rel(clean_e5_matrix_path),
            "v7_pair_features": rel(final_v7_feature_paths["zh_target_strict"]),
        }
    )
    generated_v7["clean_semantic_encoder"]["outputs_root"] = rel(clean_e5_root)
    generated_v7["clean_semantic_encoder"]["manifest_output"] = rel(
        clean_e5_root / "clean_embedding_manifest.json"
    )
    generated_v7["representative_validation"]["split_assignment_output"] = rel(
        final_paths["assignments"]
    )
    generated_v7["representative_validation"]["manifest_output"] = rel(
        final_paths["assignment_manifest"]
    )
    generated_v7["inductive_features"]["reference_bundle_output"] = rel(
        linux_root / "features" / "v7_train_only_corpus_reference.json"
    )
    generated_v7["inductive_features"]["manifest_output"] = rel(
        linux_root / "features" / "step15_v7_inductive_feature_manifest.json"
    )
    generated_v7_payload = (
        json.dumps(generated_v7, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    payloads["generated_v7_policy"] = generated_v7_payload

    generated_v8 = copy.deepcopy(policy)
    generated_v8["version"] = f"{policy['version']}-step16-readiness-{args.run_id}"
    if not args.run_id.startswith("readiness_expansion_"):
        raise ValueError(
            "Step16-v8 readiness run ids must start with readiness_expansion_: "
            f"{args.run_id}"
        )
    generated_v8["default_run_id"] = (
        "bridge_v8_readiness_" + args.run_id.removeprefix("readiness_expansion_")
    )
    generated_v8["frozen_dependencies"]["v7_policy"] = rel(
        final_paths["generated_v7_policy"]
    )
    generated_v8["frozen_dependencies"][
        "representative_validation_assignments"
    ] = rel(final_paths["assignments"])
    generated_v8["frozen_dependencies"]["representative_validation_manifest"] = rel(
        final_paths["assignment_manifest"]
    )
    generated_v8_pool = generated_v8["pools"]["zh_target_strict"]
    generated_v8_pool.update(
        {
            "frozen_labels": rel(final_paths["labels"]),
            "evidence_labels": rel(final_paths["evidence"]),
            "seller_profiles": rel(final_paths["profiles"]),
            "item_identity_signals": rel(final_paths["signals"]),
            "step4_candidates": rel(final_paths["step4"]),
            "v7_pair_features": rel(final_v7_feature_paths["zh_target_strict"]),
            "v7_clean_e5_metadata": rel(clean_e5_metadata_path),
            "v7_clean_e5_matrix": rel(clean_e5_matrix_path),
        }
    )
    generated_v8["pools"]["en_content_train_pool"]["v7_pair_features"] = rel(
        final_v7_feature_paths["en_content_train_pool"]
    )
    generated_v8["validation_context_refreeze"] = {
        "protocol": "docs/STEP16_V8_READINESS_EXPANSION_PROTOCOL_20260715.zh.md",
        "readiness_summary": rel(final_paths["readiness_summary"]),
        "freeze_manifest": rel(final_paths["freeze_manifest"]),
        "primary_alias_benchmark_excludes_identity_controls": True,
        "identity_controls_are_evidence_expert_only": True,
        "internal_development_test_unchanged": True,
        "thresholds_lowered": False,
    }
    generated_v8_payload = (
        json.dumps(generated_v8, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    payloads["generated_v8_policy"] = generated_v8_payload

    assignment_hash = hashlib.sha256(payloads["assignments"]).hexdigest()
    en_pool = base_v7["pools"]["en_content_train_pool"]
    en_train_labels = [
        row
        for row in common.load_csv(resolve(en_pool["frozen_labels"]))
        if row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
        and row.get("split_name") == "train"
    ]
    primary_zh = [
        row
        for row in eligible_labels
        if str(row.get("primary_identity_model_eligible", "1")).strip() != "0"
    ]
    controls_zh = [
        row
        for row in eligible_labels
        if str(row.get("primary_identity_model_eligible", "1")).strip() == "0"
    ]
    row_counts = {
        "train": len(en_train_labels)
        + sum(
            assignment_by_uid[row["pair_uid"]]["v7_split_name"] == "train"
            for row in primary_zh
        ),
        "valid": sum(
            assignment_by_uid[row["pair_uid"]]["v7_split_name"] == "valid"
            for row in primary_zh
        ),
        "internal_development_test": len(new_test),
        "evidence_expert_train_controls": sum(
            assignment_by_uid[row["pair_uid"]]["v7_split_name"] == "train"
            for row in controls_zh
        ),
        "evidence_expert_valid_controls": sum(
            assignment_by_uid[row["pair_uid"]]["v7_split_name"] == "valid"
            for row in controls_zh
        ),
    }
    assignment_manifest = {
        "step": "step16_v8_readiness_representative_validation_freeze",
        "version": args.run_id,
        "selection_is_model_score_blind": True,
        "current_test_used_for_selection": False,
        "current_test_role": "internal_development_test_only",
        "component_disjoint": True,
        "seller_disjoint": True,
        "identity_controls_are_evidence_expert_only": True,
        "primary_alias_benchmark_excludes_identity_controls": True,
        "row_counts": row_counts,
        "assignment_csv_sha256": assignment_hash,
        "policy": rel(final_paths["generated_v7_policy"]),
        "policy_sha256": hashlib.sha256(generated_v7_payload).hexdigest(),
        "internal_development_test_pair_uid_sha256": canonical_hash(sorted(new_test)),
        "inputs": {
            rel(path): sha256(path)
            for path in dict.fromkeys(
                [
                    label_path,
                    evidence_path,
                    assignment_path,
                    context_summary_path,
                    identity_summary_path,
                    *schema_input_paths,
                    *identity_transitive_input_paths,
                ]
            )
        },
        "effective_inputs": {
            rel(final_paths["labels"]): hashlib.sha256(payloads["labels"]).hexdigest(),
            rel(final_paths["evidence"]): hashlib.sha256(payloads["evidence"]).hexdigest(),
        },
        "manifest_hash_scope": "all_fields_except_manifest_sha256",
    }
    assignment_manifest["manifest_sha256"] = canonical_hash(assignment_manifest)
    payloads["assignment_manifest"] = (
        json.dumps(assignment_manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    readiness_summary = {
        "step": "step16_materialize_v8_reviewed_readiness_freeze",
        "run_id": args.run_id,
        "status": "ready",
        "readiness": readiness,
        "readiness_state_counts": readiness_states,
        "thresholds_lowered": False,
        "internal_development_test_unchanged": True,
        "internal_development_test_count": len(new_test),
        "internal_development_test_pair_uid_sha256": canonical_hash(sorted(new_test)),
        "seller_component_split_leakage_count": 0,
        "selected_counts": {
            "public_noise": dict(
                sorted(Counter(row["assigned_split"] for row in selected_public).items())
            ),
            "direct_control": dict(
                sorted(Counter(row["assigned_split"] for row in selected_direct).items())
            ),
            "component_control": dict(
                sorted(Counter(row["assigned_split"] for row in selected_component).items())
            ),
        },
        "available_reviewed_counts": {
            "public": public_available,
            "identity": identity_available,
        },
        "selected_records": selected_records,
        "identity_control_scope": "evidence_expert_only_not_primary_alias_benchmark",
        "model_scores_read": False,
        "outputs_pending_linux_generation": {
            "clean_e5_metadata": rel(clean_e5_metadata_path),
            "clean_e5_matrix": rel(clean_e5_matrix_path),
            "v7_pair_features": {
                pool_name: rel(path)
                for pool_name, path in sorted(final_v7_feature_paths.items())
            },
        },
    }
    readiness_summary["summary_sha256"] = canonical_hash(readiness_summary)
    payloads["readiness_summary"] = (
        json.dumps(readiness_summary, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    freeze_manifest = {
        "step": "step16_v8_readiness_freeze_manifest",
        "run_id": args.run_id,
        "producer": rel(Path(__file__).resolve()),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "model_scores_read": False,
        "fixed_internal_test_unchanged": True,
        "selected_candidate_uid_sha256": canonical_hash(
            sorted(record["selection_uid"] for record in selected_records)
        ),
        "inputs": {
            rel(path): sha256(path)
            for path in dict.fromkeys(
                [
                base_policy_path,
                base_v7_path,
                context_summary_path,
                context_reviewer_a,
                context_reviewer_b,
                context_adjudication,
                identity_summary_path,
                identity_reviewer_a,
                identity_reviewer_b,
                label_path,
                evidence_path,
                profile_path,
                signal_path,
                step4_path,
                canonical_path,
                assignment_path,
                assignment_manifest_path,
                cohort_manifest_path,
                *schema_input_paths,
                *identity_transitive_input_paths,
                ]
            )
        },
        "outputs": {
            key: {
                "path": rel(final_paths[key]),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for key, payload in payloads.items()
        },
        "primary_alias_benchmark_excludes_identity_controls": True,
        "identity_controls_are_evidence_expert_only": True,
    }
    freeze_manifest["manifest_sha256"] = canonical_hash(freeze_manifest)
    payloads["freeze_manifest"] = (
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    diagnostics = {
        "status": "ready",
        "mode": "check_only" if args.check_only else "published",
        "run_id": args.run_id,
        "readiness": readiness,
        "selected_counts": readiness_summary["selected_counts"],
        "row_counts": row_counts,
        "profile_count": len(augmented_profiles),
        "signal_count": len(augmented_signals),
        "step4_pair_count": len(ordered_step4),
        "canonical_step7_pair_count": len(ordered_canonical),
        "internal_development_test_count": len(new_test),
        "component_leakage_count": 0,
        "thresholds_lowered": False,
        "model_scores_read": False,
    }
    if args.check_only:
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        return
    staging_root = output_root.with_name(f".{output_root.name}.incomplete")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Step16-v8 readiness freeze: {output_root} / {staging_root}"
        )
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        for key, payload in payloads.items():
            staged = staging_root / output_names[key]
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
        staging_root.replace(output_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    diagnostics["output_root"] = rel(output_root)
    diagnostics["generated_v7_policy"] = rel(final_paths["generated_v7_policy"])
    diagnostics["generated_v8_policy"] = rel(final_paths["generated_v8_policy"])
    diagnostics["freeze_manifest"] = rel(final_paths["freeze_manifest"])
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
