#!/usr/bin/env python3
"""Prepare score-blind, seller-disjoint prospective holdout review queues."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import step15_v7_common as common


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step20_prospective_holdout_policy.json"
V7_POLICY = ROOT / "schema" / "step15_v7_two_stage_policy.json"
RESPONSE_FIELDS = ["independent_decision", "review_evidence_type", "review_rationale", "reviewer_id"]


def parse_utc_timestamp(value: str, field_name: str) -> dt.datetime:
    token = str(value).strip().replace("Z", "+00:00")
    if not token:
        raise ValueError(f"Missing prospective collection timestamp: {field_name}")
    parsed = dt.datetime.fromisoformat(token)
    if parsed.tzinfo is None:
        raise ValueError(f"Prospective collection timestamp must include a UTC offset: {value}")
    return parsed.astimezone(dt.timezone.utc)


def render_csv(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prospective holdout preparation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def evidence_hash(rows: list[dict]) -> str:
    payload = [
        {key: value for key, value in row.items() if key not in RESPONSE_FIELDS}
        for row in sorted(rows, key=lambda item: item["blind_id"])
    ]
    return common.canonical_hash(payload)


def raw_occurrences_for_pair(row: dict, by_seller: dict) -> str:
    left = by_seller.get(row["seller_uid_left"], {})
    right = by_seller.get(row["seller_uid_right"], {})
    evidence = []
    for token in sorted(set(left) & set(right)):
        evidence.append(
            {
                "contact_type": token[0],
                "normalized_value": token[1],
                "left": [
                    {
                        key: item.get(key, "")
                        for key in (
                            "raw_value",
                            "seller_facing_context",
                            "product_data_risk_context",
                            "direct_identity_eligible",
                            "support_only",
                            "context",
                        )
                    }
                    for item in left[token]
                ],
                "right": [
                    {
                        key: item.get(key, "")
                        for key in (
                            "raw_value",
                            "seller_facing_context",
                            "product_data_risk_context",
                            "direct_identity_eligible",
                            "support_only",
                            "context",
                        )
                    }
                    for item in right[token]
                ],
            }
        )
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def candidate_category(row: dict, reliability: dict) -> str:
    decision = reliability["decision"]
    if decision == "verified_seller_facing_direct":
        return "direct_identifier_candidate"
    if decision == "public_or_product_contact_veto":
        return "public_contact_noise_candidate"
    stratum = str(row.get("review_stratum", "")).lower()
    if "clone" in stratum or "template" in stratum:
        return "template_clone_candidate"
    if stratum == "semantic_only":
        return "semantic_topic_candidate"
    return "ordinary_candidate"


def preparation_ready(
    selected_pair_count: int,
    selected_all_prospective_final_eligible: bool,
    queue_size_met: bool,
    quota_status: dict[str, dict],
) -> bool:
    return (
        selected_pair_count > 0
        and selected_all_prospective_final_eligible
        and queue_size_met
        and bool(quota_status)
        and all(item.get("met") is True for item in quota_status.values())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--v7-policy", default=str(V7_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    v7_policy_path = common.resolve(args.v7_policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
    outputs = {key: common.resolve(value) for key, value in policy["outputs"].items()}
    candidate_schema_path = common.resolve(policy["candidate_schema"])
    candidate_schema = json.loads(candidate_schema_path.read_text(encoding="utf-8"))
    if args.validate_config_only:
        eligible_sources = [
            source for source in policy["candidate_sources"] if source.get("prospective_final_eligible")
        ]
        if not eligible_sources or any(
            not source.get("requires_collection_after_model_freeze") for source in eligible_sources
        ):
            raise ValueError("Every eligible prospective source must be collected after model freeze")
        if "collection_timestamp_utc" not in candidate_schema["required_non_empty_fields"]:
            raise ValueError("Prospective candidate schema lacks the required collection timestamp")
        discovery_keys = policy["prospective_upstream"].get(
            "discovery_required_before_blind_review", []
        )
        if set(discovery_keys) != {"seller_profiles", "item_identity_signals", "step4_candidates"}:
            raise ValueError("Prospective pre-review discovery bundle is incomplete")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "eligible_candidate_sources": eligible_sources,
                    "candidate_schema": str(candidate_schema_path.relative_to(ROOT)).replace("\\", "/"),
                },
                indent=2,
            )
        )
        return
    model_freeze_path = common.resolve(policy["model_freeze_manifest"])
    if not model_freeze_path.is_file():
        raise FileNotFoundError(
            "The v7 model/threshold freeze must exist before prospective candidate preparation"
        )
    model_freeze = json.loads(model_freeze_path.read_text(encoding="utf-8"))
    model_frozen_at = parse_utc_timestamp(model_freeze.get("frozen_at_utc", ""), "frozen_at_utc")
    supervision = common.load_csv(common.resolve(policy["existing_supervision"]))
    seen_sellers = {
        row[key]
        for row in supervision
        for key in ("seller_uid_left", "seller_uid_right")
        if row.get(key)
    }
    upstream_cfg = policy["prospective_upstream"]
    discovery_paths = {
        key: common.resolve(upstream_cfg[key])
        for key in upstream_cfg["discovery_required_before_blind_review"]
    }
    for key, path in discovery_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing prospective pre-review discovery artifact {key}: {path}")
    historical_step4_rows = common.load_csv(common.resolve(policy["step4_candidates"]))
    prospective_step4_rows = common.load_csv(discovery_paths["step4_candidates"])
    historical_step4 = {row["pair_uid"]: row for row in historical_step4_rows}
    prospective_step4 = {row["pair_uid"]: row for row in prospective_step4_rows}
    if len(prospective_step4) != len(prospective_step4_rows):
        raise ValueError("Prospective Step4 candidate file contains duplicate pair UIDs")
    overlap_pair_uids = set(historical_step4) & set(prospective_step4)
    if overlap_pair_uids:
        raise ValueError(
            "Prospective Step4 pair UIDs overlap the historical candidate universe; "
            f"first={sorted(overlap_pair_uids)[0]}"
        )
    step4 = {**historical_step4, **prospective_step4}
    prospective_profiles = {}
    with discovery_paths["seller_profiles"].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profile = json.loads(line)
            seller_uid = str(profile.get("seller_uid", "")).strip()
            if not seller_uid or seller_uid in prospective_profiles:
                raise ValueError("Prospective seller profiles contain a missing or duplicate seller_uid")
            prospective_profiles[seller_uid] = profile
    source_rows = []
    source_records = []
    for source in policy["candidate_sources"]:
        path = common.resolve(source["path"])
        if not path.exists():
            if source.get("required"):
                raise FileNotFoundError(f"Missing required prospective candidate source: {path}")
            source_records.append({**source, "present": False})
            continue
        rows = common.load_csv(path)
        timestamp_field = source.get("collection_timestamp_field", "collection_timestamp_utc")
        parsed_timestamps = []
        if source.get("prospective_final_eligible") and source.get(
            "requires_collection_after_model_freeze"
        ):
            for row in rows:
                missing_present = [
                    field
                    for field in candidate_schema["required_present_fields"]
                    if field not in row
                ]
                missing_non_empty = [
                    field
                    for field in candidate_schema["required_non_empty_fields"]
                    if not str(row.get(field, "")).strip()
                ]
                forbidden_present = [
                    field
                    for field in candidate_schema["forbidden_fields"]
                    if str(row.get(field, "")).strip()
                ]
                if missing_present or missing_non_empty or forbidden_present:
                    raise ValueError(
                        "Prospective candidate schema violation for "
                        f"{row.get('pair_uid', '<missing>')}: "
                        f"missing_present={missing_present} missing_non_empty={missing_non_empty} "
                        f"forbidden_present={forbidden_present}"
                    )
                source_contract = candidate_schema["source_contract"]
                if row["language_code"] not in source_contract["language_code_allowed"]:
                    raise ValueError(f"Prospective candidate is not Chinese: {row['pair_uid']}")
                if row["source_domain"] not in source_contract["source_domain_allowed"]:
                    raise ValueError(
                        f"Prospective candidate is outside the underground-market domain: {row['pair_uid']}"
                    )
                sha_pattern = re.compile(source_contract["content_sha256_regex"])
                for field in source_contract["content_sha256_fields"]:
                    if sha_pattern.fullmatch(str(row[field]).strip().lower()) is None:
                        raise ValueError(
                            f"Prospective source content hash is invalid: {row['pair_uid']}:{field}"
                        )
                collected_at = parse_utc_timestamp(row.get(timestamp_field, ""), timestamp_field)
                if collected_at <= model_frozen_at:
                    raise ValueError(
                        "Prospective candidate predates the v7 model freeze: "
                        f"{row.get('pair_uid', '<missing>')}"
                    )
                parsed_timestamps.append(collected_at)
        source_rows.extend(
            {
                **row,
                "candidate_source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "prospective_final_eligible": "1" if source.get("prospective_final_eligible") else "0",
                "collection_timestamp_utc": str(row.get(timestamp_field, "")).strip(),
            }
            for row in rows
        )
        source_records.append(
            {
                **source,
                "present": True,
                "row_count": len(rows),
                "sha256": common.sha256(path),
                "collection_timestamp_min_utc": None
                if not parsed_timestamps
                else min(parsed_timestamps).isoformat(),
                "collection_timestamp_max_utc": None
                if not parsed_timestamps
                else max(parsed_timestamps).isoformat(),
            }
        )

    signal_paths = [
        common.resolve(policy["item_identity_signals"]),
        discovery_paths["item_identity_signals"],
    ]
    by_seller, token_df = common.item_signal_index_many(signal_paths)
    stage_b_cfg = v7_policy["two_stage_method"]["stage_b"]
    candidates = []
    skip_counts = Counter()
    seen_eligible_pair_uids = set()
    for source_row in source_rows:
        pair_uid = str(source_row.get("pair_uid", "")).strip()
        if not pair_uid:
            continue
        if str(source_row.get("review_status", "")).strip().lower() == "reviewed":
            skip_counts["already_reviewed"] += 1
            continue
        canonical = step4.get(pair_uid, {})
        merged = {**canonical, **source_row}
        if merged.get("prospective_final_eligible") != "1":
            skip_counts["pre_v7_candidate_not_prospective_final"] += 1
            continue
        if pair_uid in seen_eligible_pair_uids:
            raise ValueError(f"Duplicate eligible prospective pair_uid: {pair_uid}")
        seen_eligible_pair_uids.add(pair_uid)
        if pair_uid not in prospective_step4:
            raise ValueError(f"Eligible prospective candidate is absent from prospective Step4: {pair_uid}")
        left_uid = str(merged.get("seller_uid_left", "")).strip()
        right_uid = str(merged.get("seller_uid_right", "")).strip()
        if not left_uid or not right_uid:
            skip_counts["missing_seller_uid"] += 1
            continue
        canonical_prospective = prospective_step4[pair_uid]
        if (
            str(canonical_prospective.get("seller_uid_left", "")).strip() != left_uid
            or str(canonical_prospective.get("seller_uid_right", "")).strip() != right_uid
        ):
            raise ValueError(f"Prospective candidate endpoints disagree with Step4: {pair_uid}")
        if left_uid not in prospective_profiles or right_uid not in prospective_profiles:
            raise ValueError(f"Prospective candidate seller profile is missing: {pair_uid}")
        merged["seller_uid_left"] = left_uid
        merged["seller_uid_right"] = right_uid
        if left_uid in seen_sellers or right_uid in seen_sellers:
            skip_counts["seller_seen_in_existing_supervision"] += 1
            continue
        reliability = common.relation_reliability(merged, by_seller, token_df, stage_b_cfg)
        category = candidate_category(merged, reliability)
        candidates.append(
            {
                **merged,
                "prospective_candidate_category": category,
                "raw_contact_occurrences_json": raw_occurrences_for_pair(merged, by_seller),
                "reliability_discovery_decision": reliability["decision"],
            }
        )

    grouped = defaultdict(list)
    seed = int(policy["selection_seed"])
    for row in candidates:
        grouped[row["prospective_candidate_category"]].append(row)
    selected = []
    selected_sellers = set()
    observed_before_disjoint = {key: len(value) for key, value in sorted(grouped.items())}
    for category, quota in policy["candidate_quotas"].items():
        ordered = sorted(
            grouped.get(category, []),
            key=lambda row: common.deterministic_rank(row["pair_uid"], seed),
        )
        count = 0
        for row in ordered:
            if row["seller_uid_left"] in selected_sellers or row["seller_uid_right"] in selected_sellers:
                continue
            selected.append(row)
            selected_sellers.update([row["seller_uid_left"], row["seller_uid_right"]])
            count += 1
            if count >= int(quota):
                break
    selected_counts = Counter(row["prospective_candidate_category"] for row in selected)
    quota_status = {
        category: {
            "required": int(quota),
            "selected": selected_counts.get(category, 0),
            "met": selected_counts.get(category, 0) >= int(quota),
        }
        for category, quota in policy["candidate_quotas"].items()
    }
    queue_size_met = len(selected) >= int(policy["queue_size"])
    selected_all_prospective_final_eligible = bool(selected) and all(
        row.get("prospective_final_eligible") == "1" for row in selected
    )
    candidate_quota_coverage_ready = queue_size_met and all(
        item["met"] for item in quota_status.values()
    )
    holdout_freeze_ready = preparation_ready(
        len(selected),
        selected_all_prospective_final_eligible,
        queue_size_met,
        quota_status,
    )

    mapping_rows = []
    blind_rows = []
    for row in selected:
        blind_id = f"prospective_{common.deterministic_rank(row['pair_uid'], seed)[:20]}"
        mapping_rows.append(
            {
                "blind_id": blind_id,
                "pair_uid": row["pair_uid"],
                "seller_uid_left": row["seller_uid_left"],
                "seller_uid_right": row["seller_uid_right"],
                "prospective_candidate_category": row["prospective_candidate_category"],
                "candidate_source_path": row.get("candidate_source_path", ""),
                "prospective_final_eligible": row.get("prospective_final_eligible", "0"),
                "collection_timestamp_utc": row.get("collection_timestamp_utc", ""),
                "language_code": row.get("language_code", ""),
                "source_domain": row.get("source_domain", ""),
                "source_collection_id": row.get("source_collection_id", ""),
                "source_record_id_left": row.get("source_record_id_left", ""),
                "source_record_id_right": row.get("source_record_id_right", ""),
                "source_provenance_ref_left": row.get("source_provenance_ref_left", ""),
                "source_provenance_ref_right": row.get("source_provenance_ref_right", ""),
                "source_content_sha256_left": row.get("source_content_sha256_left", ""),
                "source_content_sha256_right": row.get("source_content_sha256_right", ""),
            }
        )
        blind_rows.append(
            {
                "blind_id": blind_id,
                "source_market_raw_left": row.get("source_market_raw_left", ""),
                "source_market_raw_right": row.get("source_market_raw_right", ""),
                "source_seller_raw_left": row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": row.get("source_seller_raw_right", ""),
                "shared_contact_values": row.get("shared_contact_values", ""),
                "shared_pgp_fingerprint_values": row.get("shared_pgp_fingerprint_values", ""),
                "shared_title_values": row.get("shared_title_values", ""),
                "shared_description_values": row.get("shared_description_values", ""),
                "left_preview": row.get("left_preview", ""),
                "right_preview": row.get("right_preview", ""),
                "raw_contact_occurrences_json": row["raw_contact_occurrences_json"],
                "independent_decision": "",
                "review_evidence_type": "",
                "review_rationale": "",
                "reviewer_id": "",
            }
        )
    mapping_fields = list(mapping_rows[0]) if mapping_rows else [
        "blind_id", "pair_uid", "seller_uid_left", "seller_uid_right", "prospective_candidate_category"
    ]
    queue_fields = list(blind_rows[0]) if blind_rows else ["blind_id", *RESPONSE_FIELDS]
    mapping_payload = render_csv(mapping_rows, mapping_fields)
    queue_payloads = {}
    for reviewer, offset in (("reviewer_a", 1), ("reviewer_b", 2)):
        ordered = sorted(
            blind_rows,
            key=lambda row: common.deterministic_rank(row["blind_id"], seed + offset),
        )
        queue_payloads[reviewer] = render_csv(ordered, queue_fields)
    manifest = {
        "step": "step20_prepare_prospective_holdout",
        "version": policy["version"],
        "score_blind": True,
        "prior_label_blind": True,
        "existing_supervision_seller_count": len(seen_sellers),
        "eligible_candidate_count_before_seller_disjoint_selection": len(candidates),
        "candidate_category_counts_before_seller_disjoint_selection": observed_before_disjoint,
        "selected_pair_count": len(selected),
        "selected_seller_count": len(selected_sellers),
        "selected_pairs_are_seller_disjoint": len(selected_sellers) == 2 * len(selected),
        "selected_prospective_final_eligible_count": sum(
            row.get("prospective_final_eligible") == "1" for row in mapping_rows
        ),
        "selected_category_counts": dict(sorted(selected_counts.items())),
        "candidate_quota_status": quota_status,
        "all_candidate_quotas_met": all(item["met"] for item in quota_status.values()),
        "queue_size_met": queue_size_met,
        "selected_all_prospective_final_eligible": selected_all_prospective_final_eligible,
        "candidate_quota_coverage_ready": candidate_quota_coverage_ready,
        "holdout_freeze_ready": holdout_freeze_ready,
        "skip_counts": dict(sorted(skip_counts.items())),
        "candidate_sources": source_records,
        "blind_mapping_sha256": hashlib.sha256(mapping_payload).hexdigest(),
        "queue_evidence_sha256": {
            reviewer: hashlib.sha256(payload).hexdigest() for reviewer, payload in queue_payloads.items()
        },
        "prepared_canonical_evidence_sha256": evidence_hash(blind_rows),
        "model_frozen_at_utc": model_frozen_at.isoformat(),
        "model_freeze_manifest": str(model_freeze_path.relative_to(ROOT)).replace("\\", "/"),
        "model_freeze_manifest_sha256": common.sha256(model_freeze_path),
        "candidate_schema": str(candidate_schema_path.relative_to(ROOT)).replace("\\", "/"),
        "candidate_schema_sha256": common.sha256(candidate_schema_path),
        "prospective_discovery_inputs": {
            key: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": common.sha256(path),
            }
            for key, path in sorted(discovery_paths.items())
        },
        "selected_collection_timestamp_min_utc": None
        if not mapping_rows
        else min(
            parse_utc_timestamp(row["collection_timestamp_utc"], "collection_timestamp_utc")
            for row in mapping_rows
        ).isoformat(),
        "selected_collection_timestamp_max_utc": None
        if not mapping_rows
        else max(
            parse_utc_timestamp(row["collection_timestamp_utc"], "collection_timestamp_utc")
            for row in mapping_rows
        ).isoformat(),
        "policy": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": common.sha256(policy_path),
    }
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    if args.validate_only:
        print(json.dumps({"status": "pass", "manifest": manifest}, indent=2, ensure_ascii=False))
        return
    if not holdout_freeze_ready:
        raise ValueError(
            "Post-freeze prospective candidates do not satisfy the preregistered, "
            "seller-disjoint review-queue size and category quotas; no incomplete "
            f"formal review queue was published. quota_status={quota_status} "
            f"selected_pair_count={len(selected)}"
        )
    preparation_root = outputs["preparation_manifest"].parent
    managed_outputs = [
        outputs["blind_mapping"],
        outputs["reviewer_a_queue"],
        outputs["reviewer_b_queue"],
        outputs["preparation_manifest"],
        outputs["adjudication_queue"],
    ]
    if any(path.parent != preparation_root for path in managed_outputs):
        raise ValueError("Prospective preparation outputs must share one publication directory")
    staging_root = preparation_root.with_name(f".{preparation_root.name}.incomplete")
    if preparation_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Prospective preparation final or incomplete directory exists: "
            f"{preparation_root} / {staging_root}"
        )

    def staged(final_path: Path) -> Path:
        return staging_root / final_path.relative_to(preparation_root)

    write_new(staged(outputs["blind_mapping"]), mapping_payload)
    write_new(staged(outputs["reviewer_a_queue"]), queue_payloads["reviewer_a"])
    write_new(staged(outputs["reviewer_b_queue"]), queue_payloads["reviewer_b"])
    write_new(
        staged(outputs["preparation_manifest"]),
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    staging_root.replace(preparation_root)
    print(
        json.dumps(
            {
                "status": "prepared_not_frozen",
                "selected_pair_count": len(selected),
                "holdout_freeze_ready": manifest["holdout_freeze_ready"],
                "quota_status": quota_status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
