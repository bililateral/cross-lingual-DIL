#!/usr/bin/env python3
"""Apply blind Step16-v8 context reviews to an isolated, component-safe freeze."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step16_v8_validation_refreeze_policy.json"

PUBLIC_STATES = {"risky_only_shared", "support_only_shared", "high_frequency_public"}
BLIND_PACKET_FIELDS = {
    "review_candidate_uid",
    "seller_uid_left",
    "seller_uid_right",
    "shared_identifier_types",
    "shared_identifier_values",
    "left_context_preview",
    "right_context_preview",
    "reviewer_id",
    "identity_label",
    "evidence_type",
    "confidence",
    "notes",
}
IMMUTABLE_BLIND_FIELDS = (
    "seller_uid_left",
    "seller_uid_right",
    "shared_identifier_types",
    "shared_identifier_values",
    "left_context_preview",
    "right_context_preview",
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def normalized(value: object) -> str:
    return str(value or "").strip()


def load_completed_blind_packet(
    path: Path,
    candidate_index: dict[str, dict],
    role: str,
    require_complete_universe: bool,
) -> dict[str, dict]:
    if not path.is_file():
        if require_complete_universe:
            raise FileNotFoundError(f"Missing completed reviewer-{role} packet: {path}")
        return {}
    rows = common.load_csv(path)
    if not rows:
        if require_complete_universe:
            raise ValueError(f"Completed reviewer-{role} packet has no rows: {path}")
        return {}
    observed_fields = set(rows[0])
    if observed_fields != BLIND_PACKET_FIELDS:
        raise ValueError(
            f"Reviewer-{role} packet schema changed; "
            f"missing={sorted(BLIND_PACKET_FIELDS - observed_fields)} "
            f"unknown={sorted(observed_fields - BLIND_PACKET_FIELDS)}"
        )
    indexed = {}
    for row in rows:
        uid = normalized(row.get("review_candidate_uid"))
        if uid not in candidate_index:
            raise ValueError(f"Reviewer-{role} packet references unknown candidate: {uid}")
        if uid in indexed:
            raise ValueError(f"Reviewer-{role} packet duplicates candidate: {uid}")
        candidate = candidate_index[uid]
        for field in IMMUTABLE_BLIND_FIELDS:
            if normalized(row.get(field)) != normalized(candidate.get(field)):
                raise ValueError(
                    f"Reviewer-{role} packet changed immutable evidence for {uid}:{field}"
                )
        indexed[uid] = row
    if require_complete_universe and set(indexed) != set(candidate_index):
        missing = sorted(set(candidate_index) - set(indexed))
        extra = sorted(set(indexed) - set(candidate_index))
        raise ValueError(
            f"Reviewer-{role} packet universe changed; missing={missing[:1]} extra={extra[:1]}"
        )
    return indexed


def resolve_review_decision(row: dict, cfg: dict) -> dict:
    """Resolve two independent reviews without inferring a label from queue rules."""
    uid = normalized(row.get("review_candidate_uid"))
    queue_kind = normalized(row.get("queue_kind"))
    if not uid or not queue_kind:
        raise ValueError("Review decision lacks review_candidate_uid or queue_kind")
    reviewer_a = normalized(row.get("reviewer_a_id"))
    reviewer_b = normalized(row.get("reviewer_b_id"))
    a_identity = normalized(row.get("reviewer_a_identity_label")).lower()
    b_identity = normalized(row.get("reviewer_b_identity_label")).lower()
    a_evidence = normalized(row.get("reviewer_a_evidence_type"))
    b_evidence = normalized(row.get("reviewer_b_evidence_type"))
    a_confidence = normalized(row.get("reviewer_a_confidence")).lower()
    b_confidence = normalized(row.get("reviewer_b_confidence")).lower()
    review_started = any(
        (reviewer_a, reviewer_b, a_identity, b_identity, a_evidence, b_evidence)
    )
    if not review_started:
        return {"status": "not_reviewed", "review_candidate_uid": uid}
    if not all((reviewer_a, reviewer_b, a_identity, b_identity, a_evidence, b_evidence)):
        return {"status": "incomplete_dual_review", "review_candidate_uid": uid}
    if cfg["require_distinct_reviewer_ids"] and reviewer_a.casefold() == reviewer_b.casefold():
        raise ValueError(f"Dual reviewers must differ for {uid}")
    allowed_identity = set(cfg["allowed_identity_labels"])
    allowed_evidence = set(cfg["allowed_evidence_types"])
    for identity, evidence, reviewer in (
        (a_identity, a_evidence, reviewer_a),
        (b_identity, b_evidence, reviewer_b),
    ):
        if identity not in allowed_identity:
            raise ValueError(f"Invalid identity label from {reviewer} for {uid}: {identity}")
        if evidence not in allowed_evidence:
            raise ValueError(f"Invalid evidence type from {reviewer} for {uid}: {evidence}")
    matching = a_identity == b_identity and a_evidence == b_evidence
    high_matching = matching and a_confidence == "high" and b_confidence == "high"
    if high_matching:
        identity = a_identity
        evidence = a_evidence
        confidence = "high"
        source = "matching_independent_reviews"
        reviewer_ids = [reviewer_a, reviewer_b]
    else:
        adjudicator = normalized(row.get("adjudicator_id"))
        identity = normalized(row.get("adjudicated_identity_label")).lower()
        evidence = normalized(row.get("adjudicated_evidence_type"))
        confidence = normalized(row.get("adjudication_confidence")).lower()
        if not all((adjudicator, identity, evidence, confidence)):
            return {"status": "requires_adjudication", "review_candidate_uid": uid}
        if cfg["adjudicator_must_differ_from_both_reviewers"] and adjudicator.casefold() in {
            reviewer_a.casefold(),
            reviewer_b.casefold(),
        }:
            raise ValueError(f"Adjudicator must differ from both reviewers for {uid}")
        if identity not in allowed_identity or evidence not in allowed_evidence:
            raise ValueError(f"Invalid adjudicated decision for {uid}: {identity}/{evidence}")
        source = "independent_adjudication"
        reviewer_ids = [reviewer_a, reviewer_b, adjudicator]
    if cfg["require_high_confidence_for_supervision"] and confidence != "high":
        return {
            "status": "resolved_but_not_high_confidence",
            "review_candidate_uid": uid,
            "identity_label": identity,
            "evidence_type": evidence,
        }
    allowed_for_queue = cfg["accepted_queue_decisions"].get(queue_kind, {}).get(identity, [])
    if evidence not in allowed_for_queue:
        raise ValueError(
            f"Queue/decision contract rejected {uid}: {queue_kind}/{identity}/{evidence}"
        )
    return {
        "status": "resolved_high_confidence",
        "review_candidate_uid": uid,
        "identity_label": identity,
        "evidence_type": evidence,
        "confidence": confidence,
        "decision_source": source,
        "reviewer_ids": reviewer_ids,
        "review_notes": " | ".join(
            value
            for value in (
                normalized(row.get("reviewer_a_notes")),
                normalized(row.get("reviewer_b_notes")),
                normalized(row.get("adjudication_notes")),
            )
            if value
        ),
    }


def attach_components(rows: list[dict]) -> dict[str, dict]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for row in rows:
        union(row["seller_uid_left"], row["seller_uid_right"])
    members = defaultdict(list)
    for seller in sorted(parent):
        members[find(seller)].append(seller)
    component_by_seller = {}
    for sellers in members.values():
        component_id = f"v8ctx_{common.canonical_hash(sellers)[:16]}"
        item = {"component_id": component_id, "members": tuple(sellers), "size": len(sellers)}
        for seller in sellers:
            component_by_seller[seller] = item
    return component_by_seller


def build_occurrence_index(signals: list[dict], train_sellers: set[str]) -> tuple[dict, Counter]:
    by_seller = defaultdict(lambda: defaultdict(list))
    sellers_by_token = defaultdict(set)
    for row in signals:
        seller = normalized(row.get("seller_uid"))
        contact_type = normalized(row.get("contact_type")).lower()
        value = normalized(row.get("normalized_value")).lower()
        if not seller or not contact_type or not value:
            continue
        token = (contact_type, value)
        by_seller[seller][token].append(row)
        if seller in train_sellers:
            sellers_by_token[token].add(seller)
    return by_seller, Counter({token: len(sellers) for token, sellers in sellers_by_token.items()})


def readiness_counts(
    rows: list[dict],
    split_by_uid: dict[str, str],
    evidence_index: dict[str, dict],
    signals: list[dict],
    frequency_threshold: int,
) -> dict[str, Counter]:
    train_sellers = {
        row[seller_key]
        for row in rows
        if split_by_uid[row["pair_uid"]] == "train"
        for seller_key in ("seller_uid_left", "seller_uid_right")
    }
    by_seller, token_df = build_occurrence_index(signals, train_sellers)
    result = {"train": Counter(), "valid": Counter(), "internal_development_test": Counter()}
    for row in rows:
        split = split_by_uid[row["pair_uid"]]
        evidence = common.occurrence_evidence(row, by_seller, token_df, frequency_threshold)
        label = row["review_label"]
        evidence_type = evidence_index[row["pair_uid"]]["evidence_type"]
        benchmark_ok = (
            row.get("benchmark_eligible", "1") == "1"
            and row.get("silver_train_only", "0") != "1"
        )
        if benchmark_ok and label == "negative" and evidence["evidence_state"] in PUBLIC_STATES:
            result[split]["state_backed_public_noise_negative"] += 1
        if (
            benchmark_ok
            and label == "positive"
            and evidence["evidence_state"] == "verified_direct_both_sides"
        ):
            result[split]["state_backed_verified_direct_positive"] += 1
        if (
            benchmark_ok
            and label == "positive"
            and evidence_type == "same_controller_component_anchor"
        ):
            result[split]["same_controller_component_anchor_positive"] += 1
    return result


def make_label_row(
    candidate: dict,
    step4: dict,
    decision: dict,
    target_split: str,
    fields: list[str],
) -> dict:
    row = {field: "" for field in fields}
    for field in fields:
        if field in step4:
            row[field] = step4[field]
    label = decision["identity_label"]
    evidence_type = decision["evidence_type"]
    row.update(
        {
            "balanced_review_rank": f"step16v8_{candidate['review_candidate_uid']}",
            "pair_uid": step4["pair_uid"],
            "data_bucket": "zh_target_strict",
            "candidate_language": "zh",
            "candidate_scope": step4.get("candidate_scope", "sockpuppet_primary"),
            "review_stratum": (
                "identifier_plus_text"
                if evidence_type == "same_controller_direct_identifier"
                else "public_contact_or_url_noise"
            ),
            "review_priority": "step16_v8_blind_context_review",
            "review_status": "reviewed",
            "review_label": label,
            "reviewer_id": "+".join(decision["reviewer_ids"]),
            "review_notes": (
                "Step16-v8 independent occurrence-context review; "
                f"decision_source={decision['decision_source']}; "
                f"evidence_type={evidence_type}; notes={decision['review_notes']}"
            ),
            "soft_same_alias_continuity_bool": "0",
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "split_name": target_split,
            "split_component_id": "pending_v8_context_component",
            "split_component_size": "",
            "label_tier": "gold_context_reviewed",
            "benchmark_eligible": "1",
            "silver_train_only": "0",
            "training_sample_weight": "1.000000",
            "silver_positive_reasons": "",
            "silver_negative_reasons": "",
        }
    )
    return row


def make_evidence_row(
    label: dict,
    step4: dict,
    features: dict,
    evidence_type: str,
    fields: list[str],
) -> dict:
    row = {field: "" for field in fields}
    for field in fields:
        if field in features:
            row[field] = features[field]
        elif field in step4:
            row[field] = step4[field]
        elif field in label:
            row[field] = label[field]
    positive = label["review_label"] == "positive"
    direct = evidence_type == "same_controller_direct_identifier"
    public = evidence_type == "public_contact_or_url_noise"
    row.update(
        {
            "pair_uid": label["pair_uid"],
            "data_bucket": "zh_target_strict",
            "candidate_language": "zh",
            "split_name": label["split_name"],
            "split_component_id": label["split_component_id"],
            "review_label": label["review_label"],
            "review_stratum": label["review_stratum"],
            "usable_for_supervision": "1",
            "usable_for_core_transfer": "1",
            "identity_label": "same_controller" if positive else "different_controller",
            "evidence_type": evidence_type,
            "evidence_type_confident": "1",
            "identity_training_eligible": "1",
            "has_direct_identifier_signal": "1" if direct else "0",
            "has_template_clone_signal": "0",
            "has_semantic_topic_signal": "0",
            "has_public_contact_or_url_noise_signal": "1" if public else "0",
            "evidence_type_reasons": "step16_v8_independent_occurrence_context_review",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--reviewer-a-file", default=None)
    parser.add_argument("--reviewer-b-file", default=None)
    parser.add_argument("--adjudication-file", default=None)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    refreeze_path = common.resolve(args.policy)
    cfg = common.load_json(refreeze_path)
    base_v8_path = common.resolve(cfg["base_v8_policy"])
    _, base_v8, effective_v7 = common.load_policy(base_v8_path)
    common.validate_policy_contract(base_v8, effective_v7)
    required_readiness = cfg["data_readiness"]["minimum_valid_slice_counts"]
    if required_readiness != base_v8["promotion_gates"]["minimum_valid_slice_counts"]:
        raise ValueError("Step16-v8 and Step15-v8 readiness thresholds differ")
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "version": cfg["version"],
                    "minimum_valid_slice_counts": required_readiness,
                    "thresholds_lowered": False,
                },
                indent=2,
            )
        )
        return

    summary_path = common.resolve(cfg["queue_summary"])
    summary = common.load_json(summary_path)
    expected_summary_hash = summary.get("summary_sha256")
    unsigned_summary = dict(summary)
    unsigned_summary.pop("summary_sha256", None)
    if expected_summary_hash != common.canonical_hash(unsigned_summary):
        raise ValueError("Step16-v8 queue summary self-hash is invalid")
    if summary["run_id"] != cfg["review_run_id"]:
        raise ValueError("Step16-v8 queue run-id differs from refreeze policy")
    if common.sha256(base_v8_path) != summary["policy_sha256"]:
        raise ValueError("Step16-v8 queue was generated from a different v8 policy")
    summary_pool = base_v8["pools"]["zh_target_strict"]
    queue_input_contract = {
        "item_identity_signals_sha256": common.resolve(
            summary_pool["item_identity_signals"]
        ),
        "frozen_labels_sha256": common.resolve(summary_pool["frozen_labels"]),
        "step4_candidates_sha256": common.resolve(summary_pool["step4_candidates"]),
        "v7_pair_features_sha256": common.resolve(summary_pool["v7_pair_features"]),
        "v7_clean_e5_metadata_sha256": common.resolve(
            summary_pool["v7_clean_e5_metadata"]
        ),
    }
    for key, path in queue_input_contract.items():
        if summary["inputs"].get(key) != common.sha256(path):
            raise ValueError(f"Step16-v8 queue source changed after generation: {path}")
    candidate_index = {}
    for queue_kind, record in summary["outputs"].items():
        path = common.resolve(record["path"])
        if common.sha256(path) != record["sha256"]:
            raise ValueError(f"Immutable context queue changed after generation: {path}")
        for row in common.load_csv(path):
            uid = row["review_candidate_uid"]
            if uid in candidate_index:
                raise ValueError(f"Duplicate review candidate UID across queues: {uid}")
            if row["queue_kind"] != queue_kind:
                raise ValueError(f"Queue-kind mismatch for review candidate {uid}")
            candidate_index[uid] = row
    blind_template_universes = {}
    for role, record in summary["blind_review_templates"].items():
        template_path = common.resolve(record["path"])
        if common.sha256(template_path) != record["sha256"]:
            raise ValueError(f"Immutable reviewer template changed: {template_path}")
        template_rows = common.load_csv(template_path)
        if template_rows and set(template_rows[0]) != BLIND_PACKET_FIELDS:
            raise ValueError(f"Reviewer template schema changed: {template_path}")
        template_uids = {row["review_candidate_uid"] for row in template_rows}
        if len(template_uids) != len(template_rows):
            raise ValueError(f"Reviewer template duplicates candidates: {template_path}")
        unknown = template_uids - set(candidate_index)
        if unknown:
            raise ValueError(f"Reviewer template contains unknown candidate: {sorted(unknown)[0]}")
        blind_template_universes[role] = template_uids
    if len({frozenset(value) for value in blind_template_universes.values()}) != 1:
        raise ValueError("Reviewer A/B/adjudicator template universes differ")
    review_universe = next(iter(blind_template_universes.values()), set())
    if not review_universe:
        raise ValueError(
            "Step16-v8 produced no feature-ready, split-eligible candidates for blind review"
        )
    candidate_index = {uid: candidate_index[uid] for uid in sorted(review_universe)}

    completed_paths = cfg["completed_review_files"]
    reviewer_a_path = common.resolve(args.reviewer_a_file or completed_paths["reviewer_a"])
    reviewer_b_path = common.resolve(args.reviewer_b_file or completed_paths["reviewer_b"])
    adjudication_path = common.resolve(
        args.adjudication_file or completed_paths["adjudication"]
    )
    reviewer_a = load_completed_blind_packet(
        reviewer_a_path, candidate_index, "a", require_complete_universe=True
    )
    reviewer_b = load_completed_blind_packet(
        reviewer_b_path, candidate_index, "b", require_complete_universe=True
    )
    adjudication = load_completed_blind_packet(
        adjudication_path,
        candidate_index,
        "adjudicator",
        require_complete_universe=False,
    )
    resolved = []
    for uid, candidate in sorted(candidate_index.items()):
        row_a = reviewer_a[uid]
        row_b = reviewer_b[uid]
        adjudicated = adjudication.get(uid, {})
        combined = {
            "review_candidate_uid": uid,
            "queue_kind": candidate["queue_kind"],
            "reviewer_a_id": row_a["reviewer_id"],
            "reviewer_a_identity_label": row_a["identity_label"],
            "reviewer_a_evidence_type": row_a["evidence_type"],
            "reviewer_a_confidence": row_a["confidence"],
            "reviewer_a_notes": row_a["notes"],
            "reviewer_b_id": row_b["reviewer_id"],
            "reviewer_b_identity_label": row_b["identity_label"],
            "reviewer_b_evidence_type": row_b["evidence_type"],
            "reviewer_b_confidence": row_b["confidence"],
            "reviewer_b_notes": row_b["notes"],
            "adjudicator_id": adjudicated.get("reviewer_id", ""),
            "adjudicated_identity_label": adjudicated.get("identity_label", ""),
            "adjudicated_evidence_type": adjudicated.get("evidence_type", ""),
            "adjudication_confidence": adjudicated.get("confidence", ""),
            "adjudication_notes": adjudicated.get("notes", ""),
        }
        resolved.append(resolve_review_decision(combined, cfg["review_protocol"]))
    resolution_counts = Counter(item["status"] for item in resolved)
    high_confidence = {
        item["review_candidate_uid"]: item
        for item in resolved
        if item["status"] == "resolved_high_confidence"
        and item.get("identity_label") in {"positive", "negative"}
    }

    zh_pool = base_v8["pools"]["zh_target_strict"]
    label_path = common.resolve(zh_pool["frozen_labels"])
    evidence_path = common.resolve(zh_pool["evidence_labels"])
    step4_path = common.resolve(zh_pool["step4_candidates"])
    feature_path = common.resolve(zh_pool["v7_pair_features"])
    signal_path = common.resolve(zh_pool["item_identity_signals"])
    assignment_path = common.resolve(
        base_v8["frozen_dependencies"]["representative_validation_assignments"]
    )
    label_rows = common.load_csv(label_path)
    evidence_rows = common.load_csv(evidence_path)
    label_fields = list(label_rows[0])
    evidence_fields = list(evidence_rows[0])
    labels = {row["pair_uid"]: dict(row) for row in label_rows}
    evidence = {row["pair_uid"]: dict(row) for row in evidence_rows}
    step4 = {row["pair_uid"]: row for row in common.load_csv(step4_path)}
    features = {row["pair_uid"]: row for row in common.load_csv(feature_path)}
    clean_e5_metadata_path = common.resolve(zh_pool["v7_clean_e5_metadata"])
    clean_e5_metadata = common.load_json(clean_e5_metadata_path)
    clean_e5_sellers = set(clean_e5_metadata.get("seller_uids", []))
    old_assignments = {
        row["pair_uid"]: row for row in common.load_csv(assignment_path)
    }
    accepted_new = []
    not_materialized = []
    desired_new_split = {}
    for uid, decision in sorted(high_confidence.items()):
        candidate = candidate_index[uid]
        pair_uid = candidate["pair_uid_if_in_step4"]
        eligibility = candidate["split_eligibility"]
        if decision["identity_label"] == "uncertain":
            continue
        state_label_conflict = (
            candidate["queue_kind"] == "risky_only_public_noise"
            and decision["identity_label"] == "positive"
        ) or (
            candidate["queue_kind"] == "verified_direct_both_sides"
            and decision["identity_label"] == "negative"
        )
        if state_label_conflict:
            not_materialized.append(
                {
                    "review_candidate_uid": uid,
                    "reason": "reviewed_identity_conflicts_with_occurrence_state_requires_parser_or_external_anchor_update",
                }
            )
            continue
        target_split = cfg["split_policy"]["accepted_split_eligibility"].get(eligibility)
        if target_split is None:
            not_materialized.append(
                {"review_candidate_uid": uid, "reason": f"split_not_eligible:{eligibility}"}
            )
            continue
        if (
            candidate["existing_v7_pair_feature_ready"] != "1"
            or not pair_uid
            or pair_uid not in step4
            or pair_uid not in features
        ):
            not_materialized.append(
                {"review_candidate_uid": uid, "reason": "requires_step4_and_v7_feature_expansion"}
            )
            continue
        step4_sellers = {
            step4[pair_uid]["seller_uid_left"],
            step4[pair_uid]["seller_uid_right"],
        }
        candidate_sellers = {
            candidate["seller_uid_left"],
            candidate["seller_uid_right"],
        }
        if step4_sellers != candidate_sellers:
            raise ValueError(f"Context queue/Step4 seller mismatch for {pair_uid}")
        if not candidate_sellers.issubset(clean_e5_sellers):
            not_materialized.append(
                {
                    "review_candidate_uid": uid,
                    "reason": "requires_identifier_redacted_e5_cache_expansion",
                }
            )
            continue
        if pair_uid in old_assignments:
            raise ValueError(f"Context queue attempted to replace existing supervision: {pair_uid}")
        label = make_label_row(candidate, step4[pair_uid], decision, target_split, label_fields)
        evidence_row = make_evidence_row(
            label, step4[pair_uid], features[pair_uid], decision["evidence_type"], evidence_fields
        )
        labels[pair_uid] = label
        evidence[pair_uid] = evidence_row
        desired_new_split[pair_uid] = target_split
        accepted_new.append(
            {
                "review_candidate_uid": uid,
                "pair_uid": pair_uid,
                "review_label": decision["identity_label"],
                "evidence_type": decision["evidence_type"],
                "target_split": target_split,
                "decision_source": decision["decision_source"],
            }
        )

    eligible_labels = [
        row
        for row in labels.values()
        if row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    ]
    missing_evidence = [row["pair_uid"] for row in eligible_labels if row["pair_uid"] not in evidence]
    if missing_evidence:
        raise ValueError(f"Overlay supervision lacks evidence rows: {missing_evidence[0]}")
    component_by_seller = attach_components(eligible_labels)
    split_by_uid = {}
    split_reason = {}
    split_by_component = defaultdict(set)
    for row in eligible_labels:
        pair_uid = row["pair_uid"]
        if pair_uid in old_assignments:
            split = old_assignments[pair_uid]["v7_split_name"]
            reason = "retained_frozen_v7_assignment"
        else:
            split = desired_new_split[pair_uid]
            reason = "step16_v8_reviewed_context_assignment"
        component = component_by_seller[row["seller_uid_left"]]
        if component_by_seller[row["seller_uid_right"]]["component_id"] != component["component_id"]:
            raise AssertionError("Context-refreeze component closure failed")
        split_by_component[component["component_id"]].add(split)
        split_by_uid[pair_uid] = split
        split_reason[pair_uid] = reason
    leakage = {
        component: sorted(splits)
        for component, splits in split_by_component.items()
        if len(splits) > 1
    }
    if leakage:
        first = next(iter(leakage.items()))
        raise ValueError(
            "Reviewed context pairs would leak seller components across splits; "
            f"count={len(leakage)} first={first}"
        )
    old_test = {
        uid
        for uid, row in old_assignments.items()
        if row["v7_split_name"] == "internal_development_test"
    }
    new_test = {uid for uid, split in split_by_uid.items() if split == "internal_development_test"}
    if old_test != new_test:
        raise ValueError("Step16-v8 attempted to change the frozen internal development test")

    for row in eligible_labels:
        component = component_by_seller[row["seller_uid_left"]]
        row["split_component_id"] = component["component_id"]
        row["split_component_size"] = str(component["size"])
        evidence[row["pair_uid"]]["split_component_id"] = component["component_id"]
    assignment_rows = []
    for row in sorted(eligible_labels, key=lambda item: item["pair_uid"]):
        pair_uid = row["pair_uid"]
        component = component_by_seller[row["seller_uid_left"]]
        assignment_rows.append(
            {
                "pair_uid": pair_uid,
                "split_component_id": component["component_id"],
                "v7_component_id": component["component_id"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "review_label": row["review_label"],
                "evidence_type": evidence[pair_uid]["evidence_type"],
                "original_split_name": row["split_name"],
                "v7_split_name": split_by_uid[pair_uid],
                "assignment_reason": split_reason[pair_uid],
            }
        )
    signals = common.load_csv(signal_path)
    readiness = readiness_counts(
        eligible_labels,
        split_by_uid,
        evidence,
        signals,
        int(base_v8["occurrence_evidence_expert"]["public_identifier_train_seller_frequency_threshold"]),
    )
    valid_requirements = cfg["data_readiness"]["minimum_valid_slice_counts"]
    train_requirements = cfg["data_readiness"]["minimum_remaining_train_counts"]
    readiness_report = {
        "valid": {
            key: {
                "observed": int(readiness["valid"][key]),
                "required": int(required),
                "met": int(readiness["valid"][key]) >= int(required),
            }
            for key, required in valid_requirements.items()
        },
        "train": {
            key: {
                "observed": int(readiness["train"][key]),
                "required": int(required),
                "met": int(readiness["train"][key]) >= int(required),
            }
            for key, required in train_requirements.items()
        },
    }
    ready = all(item["met"] for split in readiness_report.values() for item in split.values())
    diagnostics = {
        "status": "ready" if ready else "blocked_insufficient_reviewed_context_evidence",
        "review_resolution_counts": dict(sorted(resolution_counts.items())),
        "requires_adjudication_candidate_uids": [
            item["review_candidate_uid"]
            for item in resolved
            if item["status"] == "requires_adjudication"
        ],
        "incomplete_dual_review_candidate_uids": [
            item["review_candidate_uid"]
            for item in resolved
            if item["status"] == "incomplete_dual_review"
        ],
        "resolved_binary_high_confidence_count": len(high_confidence),
        "materialized_reviewed_pair_count": len(accepted_new),
        "not_materialized_count": len(not_materialized),
        "not_materialized": not_materialized,
        "readiness": readiness_report,
        "thresholds_lowered": False,
        "internal_development_test_unchanged": True,
        "internal_development_test_count": len(new_test),
    }
    if args.check_only:
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        return
    if not ready:
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        raise SystemExit(3)

    output_root = common.resolve(cfg["output_root"])
    staging_root = output_root.with_name(f".{output_root.name}.incomplete")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(f"Refusing to overwrite Step16-v8 refreeze: {output_root}")
    staging_root.mkdir(parents=True, exist_ok=False)
    output_names = cfg["outputs"]
    final_paths = {key: output_root / name for key, name in output_names.items()}
    staged_paths = {key: staging_root / name for key, name in output_names.items()}
    ordered_labels = sorted(labels.values(), key=lambda row: row["pair_uid"])
    ordered_evidence = sorted(evidence.values(), key=lambda row: row["pair_uid"])
    label_payload = common.render_csv(ordered_labels, label_fields)
    evidence_payload = common.render_csv(ordered_evidence, evidence_fields)
    assignment_fields = list(assignment_rows[0])
    assignment_payload = common.render_csv(assignment_rows, assignment_fields)
    staged_paths["frozen_labels_overlay"].write_bytes(label_payload)
    staged_paths["evidence_labels_overlay"].write_bytes(evidence_payload)
    staged_paths["representative_validation_assignments"].write_bytes(assignment_payload)

    by_split = defaultdict(list)
    for row in assignment_rows:
        by_split[row["v7_split_name"]].append(row)
    base_v7_path = common.resolve(base_v8["frozen_dependencies"]["v7_policy"])
    manifest = {
        "step": "step16_apply_v8_context_reviews",
        "version": cfg["version"],
        "selection_is_model_score_blind": True,
        "current_test_used_for_selection": False,
        "current_test_role": "internal_development_test_only",
        "component_disjoint": True,
        "seller_disjoint": True,
        "review_protocol": cfg["review_protocol"],
        "row_counts": {key: len(value) for key, value in sorted(by_split.items())},
        "label_counts": {
            key: dict(sorted(Counter(row["review_label"] for row in value).items()))
            for key, value in sorted(by_split.items())
        },
        "evidence_counts": {
            key: dict(sorted(Counter(row["evidence_type"] for row in value).items()))
            for key, value in sorted(by_split.items())
        },
        "occurrence_state_readiness": readiness_report,
        "accepted_reviewed_pairs": accepted_new,
        "policy": rel(base_v7_path),
        "policy_sha256": common.sha256(base_v7_path),
        "pair_uid_sha256": common.canonical_hash(sorted(split_by_uid)),
        "internal_development_test_pair_uid_sha256": common.canonical_hash(sorted(new_test)),
        "assignment_csv_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "manifest_hash_scope": "all_fields_except_manifest_sha256",
        "inputs": {
            rel(label_path): common.sha256(label_path),
            rel(evidence_path): common.sha256(evidence_path),
            rel(assignment_path): common.sha256(assignment_path),
            rel(summary_path): common.sha256(summary_path),
            rel(reviewer_a_path): common.sha256(reviewer_a_path),
            rel(reviewer_b_path): common.sha256(reviewer_b_path),
            rel(step4_path): common.sha256(step4_path),
            rel(feature_path): common.sha256(feature_path),
            rel(signal_path): common.sha256(signal_path),
            rel(clean_e5_metadata_path): common.sha256(clean_e5_metadata_path),
        },
        "effective_inputs": {
            rel(final_paths["frozen_labels_overlay"]): hashlib.sha256(label_payload).hexdigest(),
            rel(final_paths["evidence_labels_overlay"]): hashlib.sha256(evidence_payload).hexdigest(),
        },
    }
    if adjudication_path.is_file():
        manifest["inputs"][rel(adjudication_path)] = common.sha256(adjudication_path)
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    staged_paths["representative_validation_manifest"].write_bytes(manifest_payload)

    generated_v8 = copy.deepcopy(base_v8)
    generated_v8["version"] = f"{base_v8['version']}-context-reviewed-freeze-v1"
    generated_v8["default_run_id"] = cfg["generated_v8_default_run_id"]
    generated_v8["pools"]["zh_target_strict"]["frozen_labels"] = rel(
        final_paths["frozen_labels_overlay"]
    )
    generated_v8["pools"]["zh_target_strict"]["evidence_labels"] = rel(
        final_paths["evidence_labels_overlay"]
    )
    generated_v8["frozen_dependencies"]["representative_validation_assignments"] = rel(
        final_paths["representative_validation_assignments"]
    )
    generated_v8["frozen_dependencies"]["representative_validation_manifest"] = rel(
        final_paths["representative_validation_manifest"]
    )
    generated_v8["validation_context_refreeze"] = {
        "review_application_summary": rel(final_paths["review_application_summary"]),
        "representative_validation_manifest": rel(
            final_paths["representative_validation_manifest"]
        ),
        "data_readiness_definition": cfg["data_readiness"]["definition"],
        "legacy_evidence_type_only_counts_forbidden": True,
        "internal_development_test_unchanged": True,
    }
    generated_policy_payload = (
        json.dumps(generated_v8, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    staged_paths["generated_v8_policy"].write_bytes(generated_policy_payload)
    application_summary = {
        "step": "step16_apply_v8_context_reviews",
        "version": cfg["version"],
        **diagnostics,
        "accepted_reviewed_pairs": accepted_new,
        "outputs": {key: rel(path) for key, path in final_paths.items()},
        "hashes": {
            "frozen_labels_overlay": hashlib.sha256(label_payload).hexdigest(),
            "evidence_labels_overlay": hashlib.sha256(evidence_payload).hexdigest(),
            "representative_validation_assignments": hashlib.sha256(
                assignment_payload
            ).hexdigest(),
            "representative_validation_manifest": hashlib.sha256(
                manifest_payload
            ).hexdigest(),
            "generated_v8_policy": hashlib.sha256(generated_policy_payload).hexdigest(),
        },
    }
    application_summary["summary_sha256"] = common.canonical_hash(application_summary)
    staged_paths["review_application_summary"].write_text(
        json.dumps(application_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    staging_root.replace(output_root)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_root": rel(output_root),
                "generated_v8_policy": rel(final_paths["generated_v8_policy"]),
                "readiness": readiness_report,
                "internal_development_test_unchanged": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
