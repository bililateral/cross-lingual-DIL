#!/usr/bin/env python3
"""Aggregate dual reviews and fail-closed freeze a prospective holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import step15_v7_common as common
from step20_prepare_prospective_holdout import (
    RESPONSE_FIELDS,
    evidence_hash,
    parse_utc_timestamp,
    render_csv,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"
V7_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prospective holdout freeze artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_decision(decision: str, evidence_type: str, policy: dict) -> None:
    if decision not in policy["review_decisions"]:
        raise ValueError(f"Unsupported prospective review decision: {decision}")
    if evidence_type not in policy["review_evidence_types"]:
        raise ValueError(f"Unsupported prospective evidence type: {evidence_type}")
    positive_types = {
        "same_controller_direct_identifier",
        "same_controller_component_anchor",
    }
    negative_types = {
        "template_clone_not_controller",
        "semantic_topic_not_controller",
        "public_contact_or_url_noise",
        "ordinary_negative",
    }
    if decision == "positive" and evidence_type not in positive_types:
        raise ValueError("Positive prospective decision lacks direct/component evidence")
    if decision == "negative" and evidence_type not in negative_types:
        raise ValueError("Negative prospective decision has an incompatible evidence type")
    if decision == "uncertain" and evidence_type != "uncertain_insufficient_evidence":
        raise ValueError("Uncertain prospective decision must use uncertain_insufficient_evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--v7-policy", default=str(V7_POLICY))
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    v7_policy_path = common.resolve(args.v7_policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    outputs = {key: common.resolve(value) for key, value in policy["outputs"].items()}
    preparation = json.loads(outputs["preparation_manifest"].read_text(encoding="utf-8"))
    if preparation.get("holdout_freeze_ready") is not True:
        raise ValueError(
            "Prospective preparation manifest is not freeze-ready; at least one post-v7, "
            "seller-disjoint candidate must be present"
        )
    model_freeze_path = common.resolve(policy["model_freeze_manifest"])
    if not model_freeze_path.is_file():
        raise FileNotFoundError(
            "The preregistered v7 model/threshold freeze manifest must exist before holdout labels freeze"
        )
    if preparation.get("model_freeze_manifest_sha256") != common.sha256(model_freeze_path):
        raise ValueError("Prospective preparation was not performed against this model freeze")
    model_freeze = json.loads(model_freeze_path.read_text(encoding="utf-8"))
    model_frozen_at = parse_utc_timestamp(model_freeze.get("frozen_at_utc", ""), "frozen_at_utc")
    for record in preparation.get("prospective_discovery_inputs", {}).values():
        path = common.resolve(record["path"])
        if not path.is_file() or common.sha256(path) != record["sha256"]:
            raise ValueError(f"Prospective discovery evidence changed after blind review: {path}")
    for record in preparation.get("candidate_sources", []):
        if not record.get("present"):
            continue
        path = common.resolve(record["path"])
        if not path.is_file() or common.sha256(path) != record["sha256"]:
            raise ValueError(f"Prospective candidate source changed after blind review: {path}")
    mapping_rows = common.load_csv(outputs["blind_mapping"])
    if common.sha256(outputs["blind_mapping"]) != preparation["blind_mapping_sha256"]:
        raise ValueError("Prospective blind mapping changed after score-blind preparation")
    mapping = {row["blind_id"]: row for row in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("Prospective blind mapping contains duplicate blind IDs")
    mapped_sellers = [
        row[key]
        for row in mapping_rows
        for key in ("seller_uid_left", "seller_uid_right")
        if row.get(key)
    ]
    if len(mapped_sellers) != 2 * len(mapping_rows) or len(set(mapped_sellers)) != len(mapped_sellers):
        raise ValueError("Prospective mapped pairs are not mutually seller-disjoint")
    for row in mapping_rows:
        collected_at = parse_utc_timestamp(
            row.get("collection_timestamp_utc", ""), "collection_timestamp_utc"
        )
        if collected_at <= model_frozen_at:
            raise ValueError(f"Prospective mapped pair predates model freeze: {row['pair_uid']}")
    prior_rows = common.load_csv(common.resolve(policy["existing_supervision"]))
    prior_sellers = {
        row[key]
        for row in prior_rows
        for key in ("seller_uid_left", "seller_uid_right")
        if row.get(key)
    }
    overlap = sorted(set(mapped_sellers) & prior_sellers)
    if overlap:
        raise ValueError(f"Prospective sellers overlap prior supervision; first={overlap[0]}")
    queues = {
        "reviewer_a": common.load_csv(outputs["reviewer_a_queue"]),
        "reviewer_b": common.load_csv(outputs["reviewer_b_queue"]),
    }
    if set(row["blind_id"] for row in queues["reviewer_a"]) != set(mapping) or set(
        row["blind_id"] for row in queues["reviewer_b"]
    ) != set(mapping):
        raise ValueError("Prospective reviewer queues and blind mapping have different universes")
    expected_evidence_hash = preparation["prepared_canonical_evidence_sha256"]
    for reviewer, rows in queues.items():
        if evidence_hash(rows) != expected_evidence_hash:
            raise ValueError(f"Prepared evidence changed in {reviewer} queue")
    queue_indices = {
        reviewer: {row["blind_id"]: row for row in rows} for reviewer, rows in queues.items()
    }
    incomplete = []
    for reviewer, index in queue_indices.items():
        for blind_id, row in index.items():
            if not all(str(row.get(field, "")).strip() for field in RESPONSE_FIELDS):
                incomplete.append(f"{reviewer}:{blind_id}")
                continue
            validate_decision(row["independent_decision"], row["review_evidence_type"], policy)
            if row["reviewer_id"] != reviewer:
                raise ValueError(f"Reviewer identity mismatch for {reviewer}:{blind_id}")
    if incomplete:
        raise ValueError(
            f"Prospective dual review is incomplete ({len(incomplete)} fields/rows); first={incomplete[0]}"
        )

    disagreements = []
    agreed = {}
    for blind_id in sorted(mapping):
        left = queue_indices["reviewer_a"][blind_id]
        right = queue_indices["reviewer_b"][blind_id]
        if (
            left["independent_decision"] == right["independent_decision"]
            and left["review_evidence_type"] == right["review_evidence_type"]
        ):
            agreed[blind_id] = {
                "final_decision": left["independent_decision"],
                "final_evidence_type": left["review_evidence_type"],
                "adjudication_rationale": "independent_reviewer_agreement",
                "adjudicator_id": "not_required",
            }
        else:
            disagreements.append(
                {
                    "blind_id": blind_id,
                    "reviewer_a_decision": left["independent_decision"],
                    "reviewer_a_evidence_type": left["review_evidence_type"],
                    "reviewer_b_decision": right["independent_decision"],
                    "reviewer_b_evidence_type": right["review_evidence_type"],
                    "final_decision": "",
                    "final_evidence_type": "",
                    "adjudication_rationale": "",
                    "adjudicator_id": "",
                }
            )
    if disagreements:
        if not outputs["adjudication_queue"].exists():
            write_new(outputs["adjudication_queue"], render_csv(disagreements, list(disagreements[0])))
            raise ValueError(
                f"Prospective review has {len(disagreements)} disagreements; adjudication queue created"
            )
        adjudication = {row["blind_id"]: row for row in common.load_csv(outputs["adjudication_queue"])}
        if set(adjudication) != {row["blind_id"] for row in disagreements}:
            raise ValueError("Prospective adjudication universe changed")
        for blind_id, row in adjudication.items():
            required = ("final_decision", "final_evidence_type", "adjudication_rationale", "adjudicator_id")
            if not all(str(row.get(field, "")).strip() for field in required):
                raise ValueError(f"Prospective adjudication incomplete: {blind_id}")
            validate_decision(row["final_decision"], row["final_evidence_type"], policy)
            if row["adjudicator_id"] in policy["reviewers"]:
                raise ValueError("Prospective adjudicator must be independent from both reviewers")
            agreed[blind_id] = row

    final_rows = []
    for blind_id in sorted(mapping):
        source = mapping[blind_id]
        decision = agreed[blind_id]
        final_rows.append(
            {
                "pair_uid": source["pair_uid"],
                "seller_uid_left": source["seller_uid_left"],
                "seller_uid_right": source["seller_uid_right"],
                "review_label": decision["final_decision"],
                "evidence_type": decision["final_evidence_type"],
                "prospective_candidate_category": source["prospective_candidate_category"],
                "candidate_source_path": source.get("candidate_source_path", ""),
                "prospective_final_eligible": source.get("prospective_final_eligible", "0"),
                "collection_timestamp_utc": source.get("collection_timestamp_utc", ""),
                "language_code": source.get("language_code", ""),
                "source_domain": source.get("source_domain", ""),
                "source_collection_id": source.get("source_collection_id", ""),
                "source_record_id_left": source.get("source_record_id_left", ""),
                "source_record_id_right": source.get("source_record_id_right", ""),
                "source_provenance_ref_left": source.get("source_provenance_ref_left", ""),
                "source_provenance_ref_right": source.get("source_provenance_ref_right", ""),
                "source_content_sha256_left": source.get("source_content_sha256_left", ""),
                "source_content_sha256_right": source.get("source_content_sha256_right", ""),
                "split_name": "prospective_final_holdout",
                "adjudicator_id": decision["adjudicator_id"],
            }
        )
    binary_rows = [row for row in final_rows if row["review_label"] in {"positive", "negative"}]
    if not binary_rows:
        raise ValueError("Prospective review produced no binary evaluation rows")
    ineligible = [row for row in binary_rows if row["prospective_final_eligible"] != "1"]
    if ineligible:
        raise ValueError(
            "Prospective final holdout contains candidates that predate the frozen v7 method; "
            f"count={len(ineligible)}"
        )
    evidence_counts = Counter(row["evidence_type"] for row in binary_rows)
    label_counts = Counter(row["review_label"] for row in binary_rows)
    observed = {
        "positive_total": label_counts["positive"],
        "same_controller_direct_or_component": evidence_counts["same_controller_direct_identifier"]
        + evidence_counts["same_controller_component_anchor"],
        "public_contact_or_url_noise": evidence_counts["public_contact_or_url_noise"],
        "template_clone_not_controller": evidence_counts["template_clone_not_controller"],
        "semantic_topic_not_controller": evidence_counts["semantic_topic_not_controller"],
        "ordinary_negative": evidence_counts["ordinary_negative"],
    }
    unmet = {
        key: {"required": int(required), "observed": observed.get(key, 0)}
        for key, required in policy["final_holdout_minimums"].items()
        if observed.get(key, 0) < int(required)
    }
    if unmet:
        raise ValueError(f"Prospective final holdout evidence quotas are unmet: {unmet}")

    if model_freeze.get("current_internal_test_used_for_model_selection") is not False:
        raise ValueError("V7 model freeze is not test-independent")
    if model_freeze.get("prospective_holdout_required") is not True:
        raise ValueError("V7 model freeze does not require prospective evaluation")
    step15_run_id = v7_policy["two_stage_method"]["default_run_id"]
    model_summary = (
        common.resolve(v7_policy["outputs"]["two_stage_outputs_root"])
        / step15_run_id
        / "step15_v7_two_stage_summary.json"
    )
    if not model_summary.is_file():
        raise FileNotFoundError("Step15-v7 model must be frozen before prospective holdout freeze")
    fields = list(binary_rows[0])
    label_payload = render_csv(binary_rows, fields)
    pair_universe_rows = [
        {
            "pair_uid": row["pair_uid"],
            "seller_uid_left": row["seller_uid_left"],
            "seller_uid_right": row["seller_uid_right"],
            "collection_timestamp_utc": row["collection_timestamp_utc"],
        }
        for row in binary_rows
    ]
    pair_universe_payload = render_csv(pair_universe_rows, list(pair_universe_rows[0]))
    freeze_manifest = {
        "step": "step20_freeze_prospective_holdout",
        "version": policy["version"],
        "row_count": len(binary_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "minimums_met": True,
        "seller_overlap_with_prior_supervision": len(overlap),
        "selected_pairs_are_mutually_seller_disjoint": True,
        "model_frozen_at_utc": model_frozen_at.isoformat(),
        "all_rows_originate_after_v7_freeze": all(
            parse_utc_timestamp(row["collection_timestamp_utc"], "collection_timestamp_utc")
            > model_frozen_at
            for row in binary_rows
        ),
        "model_frozen_before_holdout_evaluation": True,
        "model_summary": str(model_summary.relative_to(ROOT)).replace("\\", "/"),
        "model_summary_sha256": common.sha256(model_summary),
        "model_freeze_manifest": str(model_freeze_path.relative_to(ROOT)).replace("\\", "/"),
        "model_freeze_manifest_sha256": common.sha256(model_freeze_path),
        "frozen_labels_file_sha256": hashlib.sha256(label_payload).hexdigest(),
        "frozen_labels_canonical_sha256": common.canonical_hash(binary_rows),
        "frozen_pair_universe_file_sha256": hashlib.sha256(pair_universe_payload).hexdigest(),
        "frozen_pair_universe_canonical_sha256": common.canonical_hash(pair_universe_rows),
        "preparation_manifest_sha256": common.sha256(outputs["preparation_manifest"]),
        "policy_sha256": common.sha256(policy_path),
    }
    freeze_manifest["manifest_sha256"] = common.canonical_hash(freeze_manifest)
    freeze_root = outputs["freeze_manifest"].parent
    managed_outputs = [
        outputs["frozen_pair_universe"],
        outputs["frozen_labels"],
        outputs["freeze_manifest"],
    ]
    if any(path.parent != freeze_root for path in managed_outputs):
        raise ValueError("Prospective freeze outputs must share one publication directory")
    staging_root = freeze_root.with_name(f".{freeze_root.name}.incomplete")
    if freeze_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Prospective freeze final or incomplete directory exists: {freeze_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(freeze_root)

    write_new(staged(outputs["frozen_pair_universe"]), pair_universe_payload)
    write_new(staged(outputs["frozen_labels"]), label_payload)
    write_new(
        staged(outputs["freeze_manifest"]),
        (json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    staging_root.replace(freeze_root)
    print(json.dumps({"status": "frozen", "manifest": freeze_manifest}, indent=2))


if __name__ == "__main__":
    main()
