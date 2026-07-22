#!/usr/bin/env python3
"""Prepare the standalone, label-isolated source artifacts for Step7-v3.1."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import step7_v3_1_source_data as common


PREPARATION_SCRIPT = Path(__file__).resolve()
COMMON_SCRIPT = Path(common.__file__).resolve()


def eligible_label_rows(policy: dict, rows: list[dict]) -> list[dict]:
    boundary = policy["supervision_boundary"]
    allowed_splits = set(boundary["eligible_split_names"])
    allowed_labels = set(boundary["eligible_labels"])
    output = []
    for row in rows:
        if row.get("split_name") not in allowed_splits:
            continue
        if row.get("review_label") not in allowed_labels:
            continue
        if boundary["require_usable_for_supervision"] and not common.bool_value(
            row.get("usable_for_supervision")
        ):
            continue
        output.append(row)
    return output


def validate_supervision_counts(
    policy: dict, rows: list[dict], splits: tuple[str, ...] = ("train", "valid", "test")
) -> dict:
    expected = policy["supervision_boundary"]["expected_counts"]
    observed: dict[str, dict] = {}
    for split in splits:
        split_rows = [row for row in rows if row["split_name"] == split]
        counts = Counter(row["review_label"] for row in split_rows)
        observed[split] = {
            "positive": int(counts["positive"]),
            "negative": int(counts["negative"]),
            "total": len(split_rows),
        }
        if observed[split] != expected[split]:
            raise ValueError(
                f"Step7-v3 supervision boundary drift for {split}: "
                f"expected={expected[split]} observed={observed[split]}"
            )
    expected_total = sum(int(expected[split]["total"]) for split in splits)
    if len(rows) != expected_total:
        raise ValueError(
            f"Step7-v3 supervision total drift: expected={expected_total} observed={len(rows)}"
        )
    pair_uids = [row["pair_uid"] for row in rows]
    if len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Step7-v3 eligible supervision has duplicate pair_uid values")
    return observed


def eligible_assignment_rows(policy: dict, rows: list[dict]) -> list[dict]:
    """Define the public 734-pair universe without consulting any review label."""
    cfg = policy["inputs"]["component_assignments"]
    dataset = cfg["dataset"]
    component_field = cfg["component_field"]
    allowed_splits = set(policy["supervision_boundary"]["eligible_split_names"])
    selected = [
        {
            "dataset": row.get("dataset", ""),
            "pair_uid": row.get("pair_uid", ""),
            "split_name": row.get("split_name", ""),
            "seller_uid_left": row.get("seller_uid_left", ""),
            "seller_uid_right": row.get("seller_uid_right", ""),
            component_field: row.get(component_field, ""),
        }
        for row in rows
        if row.get("dataset") == dataset and row.get("split_name") in allowed_splits
    ]
    if any(not row["pair_uid"] for row in selected):
        raise ValueError("Step7-v3 component assignments contain an empty pair_uid")
    index = {row["pair_uid"]: row for row in selected}
    if len(index) != len(selected):
        raise ValueError("Step7-v3 component assignments have duplicate pair_uid values")
    expected = policy["supervision_boundary"]["expected_counts"]
    observed = Counter(row["split_name"] for row in selected)
    for split in ("train", "valid", "test"):
        if observed[split] != int(expected[split]["total"]):
            raise ValueError(
                f"Step7-v3 public pair boundary drift for {split}: "
                f"expected={expected[split]['total']} observed={observed[split]}"
            )
    if len(selected) != int(expected["total"]):
        raise ValueError("Step7-v3 public pair universe is not exactly 734 rows")
    return selected


def build_pair_manifest(assignment_rows: list[dict], component_field: str) -> list[dict]:
    """Sanitize the fixed component-assignment universe into a public manifest."""
    output = []
    for assignment in assignment_rows:
        pair_uid = assignment["pair_uid"]
        component_id = str(assignment.get(component_field, "") or "").strip()
        if not component_id:
            raise ValueError(f"Step7-v3 empty recomputed component for pair={pair_uid}")
        output.append(
            {
                "pair_uid": pair_uid,
                "split_name": assignment["split_name"],
                "component_id": component_id,
                "seller_uid_left": assignment["seller_uid_left"],
                "seller_uid_right": assignment["seller_uid_right"],
            }
        )
    split_order = {"train": 0, "valid": 1, "test": 2}
    return sorted(output, key=lambda row: (split_order[row["split_name"]], row["pair_uid"]))


def validate_split_isolation(pair_rows: list[dict]) -> dict:
    component_splits: dict[str, set[str]] = defaultdict(set)
    seller_splits: dict[str, set[str]] = defaultdict(set)
    for row in pair_rows:
        component_splits[row["component_id"]].add(row["split_name"])
        for key in ("seller_uid_left", "seller_uid_right"):
            seller_splits[row[key]].add(row["split_name"])
    bad_components = sorted(key for key, values in component_splits.items() if len(values) != 1)
    bad_sellers = sorted(key for key, values in seller_splits.items() if len(values) != 1)
    if bad_components:
        raise ValueError(f"Step7-v3 component crosses splits: {bad_components[0]}")
    if bad_sellers:
        raise ValueError(f"Step7-v3 seller crosses splits: {bad_sellers[0]}")
    return {
        "component_count_by_split": {
            split: len(
                {
                    row["component_id"]
                    for row in pair_rows
                    if row["split_name"] == split
                }
            )
            for split in ("train", "valid", "test")
        },
        "seller_count_by_split": {
            split: len(
                {
                    row[key]
                    for row in pair_rows
                    if row["split_name"] == split
                    for key in ("seller_uid_left", "seller_uid_right")
                }
            )
            for split in ("train", "valid", "test")
        },
        "cross_split_component_count": len(bad_components),
        "cross_split_seller_count": len(bad_sellers),
    }


def evidence_index(rows: list[dict]) -> dict[str, dict]:
    index = {row["pair_uid"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Step7-v3 evidence labels contain duplicate pair_uid values")
    return index


def parse_count(value: object, field_name: str) -> int:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step7-v3 invalid identity-control count: {field_name}") from exc
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"Step7-v3 invalid identity-control count: {field_name}")
    return int(numeric)


def private_label_rows(
    split: str,
    label_rows: list[dict],
    pair_index: dict[str, dict],
    evidence: dict[str, dict],
) -> list[dict]:
    output = []
    for source in sorted(
        (row for row in label_rows if row["split_name"] == split),
        key=lambda row: row["pair_uid"],
    ):
        pair_uid = source["pair_uid"]
        pair = pair_index[pair_uid]
        for endpoint in ("seller_uid_left", "seller_uid_right"):
            if source.get(endpoint) != pair[endpoint]:
                raise ValueError(
                    f"Step7-v3 private label/public endpoint mismatch for pair={pair_uid}"
                )
        evidence_row = evidence.get(pair_uid)
        if evidence_row is None:
            raise ValueError(f"Step7-v3 evidence label missing for pair={pair_uid}")
        if evidence_row["review_label"] != source["review_label"]:
            raise ValueError(f"Step7-v3 evidence/review label mismatch for pair={pair_uid}")
        if evidence_row["split_name"] != split:
            raise ValueError(f"Step7-v3 evidence split mismatch for pair={pair_uid}")
        direct_count = parse_count(
            source.get("shared_contact_count"), "shared_contact_count"
        ) + parse_count(
            source.get("shared_pgp_fingerprint_count"),
            "shared_pgp_fingerprint_count",
        )
        output.append(
            {
                "pair_uid": pair_uid,
                "review_label": source["review_label"],
                "evidence_type": evidence_row["evidence_type"],
                "component_id": pair["component_id"],
                "identity_rule_control_score": int(direct_count > 0),
            }
        )
    return output


def content_fidelity_summary(
    policy: dict, profiles: list[dict], corpus_rows: list[dict]
) -> dict:
    """Fail closed if identifier removal catastrophically erases clean content."""
    clean_cfg = policy["clean_text_contract"]
    quality = clean_cfg["quality_gates"]
    profile_by_uid = {str(row["seller_uid"]): row for row in profiles}
    corpus_by_uid = {
        str(row["seller_uid"]): str(row["model_text"]) for row in corpus_rows
    }
    if set(profile_by_uid) != set(corpus_by_uid):
        raise ValueError("Step7-v3 content-fidelity seller universe drift")
    raw_text = "\n".join(
        str(profile_by_uid[seller_uid].get(field, ""))
        for seller_uid in sorted(profile_by_uid)
        for field in clean_cfg["fields_in_order"]
    )
    clean_text = "\n".join(
        corpus_by_uid[seller_uid] for seller_uid in sorted(corpus_by_uid)
    )
    raw_character_count = len(raw_text)
    clean_character_count = len(clean_text)
    if raw_character_count <= 0:
        raise ValueError("Step7-v3 source content is empty")
    aggregate_retention = clean_character_count / raw_character_count
    fallback_count = sum(
        value == clean_cfg["empty_text_fallback"] for value in corpus_by_uid.values()
    )
    protected_words = {}
    for word in quality["protected_content_words"]:
        pattern = re.compile(
            rf"(?i)(?<![a-z0-9]){re.escape(str(word))}(?![a-z0-9])"
        )
        raw_count = len(pattern.findall(raw_text))
        clean_count = len(pattern.findall(clean_text))
        if raw_count <= 0:
            raise ValueError(
                f"Step7-v3 protected content word disappeared upstream: {word}"
            )
        retention = clean_count / raw_count
        protected_words[str(word)] = {
            "raw_count": raw_count,
            "clean_count": clean_count,
            "retention": retention,
        }
        if retention < float(quality["minimum_protected_word_retention"]):
            raise ValueError(
                f"Step7-v3 over-redaction gate failed for protected content word: {word}"
            )
    protected_collisions = {}
    for term in quality["protected_identity_collision_terms"]:
        escaped = re.escape(str(term)).replace(r"\ ", r"[ \t]+")
        pattern = re.compile(rf"(?i)(?<![a-z0-9]){escaped}(?![a-z0-9])")
        raw_count = len(pattern.findall(raw_text))
        clean_count = len(pattern.findall(clean_text))
        if raw_count <= 0:
            raise ValueError(
                f"Step7-v3 protected identity-collision term disappeared upstream: {term}"
            )
        retention = clean_count / raw_count
        protected_collisions[str(term)] = {
            "raw_count": raw_count,
            "clean_count": clean_count,
            "retention": retention,
        }
        if retention < float(
            quality["minimum_protected_identity_collision_term_retention"]
        ):
            raise ValueError(
                "Step7-v3 over-redaction gate failed for identity-collision term: "
                f"{term}"
            )
    if aggregate_retention < float(quality["minimum_aggregate_character_retention"]):
        raise ValueError("Step7-v3 aggregate content-retention gate failed")
    if fallback_count > int(quality["maximum_empty_fallback_count"]):
        raise ValueError("Step7-v3 empty-content fallback gate failed")
    return {
        "raw_source_field_character_count": raw_character_count,
        "clean_model_text_character_count": clean_character_count,
        "aggregate_character_retention": aggregate_retention,
        "empty_text_fallback_count": fallback_count,
        "protected_content_word_retention": protected_words,
        "protected_identity_collision_term_retention": protected_collisions,
        "quality_gates_passed": True,
    }


def field_only_profile(profile: dict, target_field: str, fields: list[str]) -> dict:
    """Return a shallow copy with only one clean-text source field populated."""
    isolated = dict(profile)
    for field in fields:
        if field != target_field:
            isolated[field] = ""
    return isolated


def build_field_corpus_rows(
    policy: dict,
    profiles: dict[str, dict],
    seller_split: dict[str, str],
    seller_records: dict[str, dict],
    global_tokens: set[str] | frozenset[str],
    contextual_aliases: set[str] | frozenset[str],
    contextual_alias_deletions: set[str] | frozenset[str],
    audited_global_phrases: set[str] | frozenset[str],
) -> tuple[list[dict], dict[str, int]]:
    """Replay redaction field by field without reading labels or evidence types."""
    clean_cfg = policy["clean_text_contract"]
    fields = clean_cfg["fields_in_order"]
    rows = []
    nonempty_counts = {field: 0 for field in fields}
    for seller_uid in sorted(seller_records):
        field_texts: dict[str, str] = {}
        for field in fields:
            record, diagnostics = common.build_clean_seller_record(
                field_only_profile(profiles[seller_uid], field, fields),
                clean_cfg,
                global_tokens,
                contextual_aliases,
                contextual_alias_deletions,
                audited_global_phrases,
            )
            value = "" if diagnostics["empty_after_redaction"] else record["model_text"]
            field_texts[field] = value
            nonempty_counts[field] += int(bool(value))
        reconstructed = "\n".join(
            field_texts[field] for field in fields if field_texts[field]
        ).strip()
        if not reconstructed:
            reconstructed = clean_cfg["empty_text_fallback"]
        full_text = seller_records[seller_uid]["model_text"]
        if reconstructed != full_text:
            raise ValueError(
                "Step7-v3.1 field-wise clean-text replay drift: "
                f"{common.sha256_text(seller_uid)[:16]}"
            )
        rows.append(
            {
                "seller_uid": seller_uid,
                "split_name": seller_split[seller_uid],
                "field_texts": field_texts,
                "field_text_sha256": {
                    field: common.sha256_text(field_texts[field]) for field in fields
                },
                "model_text": full_text,
                "model_text_sha256": common.sha256_text(full_text),
            }
        )
    return rows, nonempty_counts


def verify_expected_artifact(policy: dict, role: str, path: Path) -> dict:
    """Fail closed unless a frozen source artifact reproduces byte for byte."""
    expected = policy["expected_artifacts"][role]
    observed = {
        "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
        "sha256": common.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if (
        observed["sha256"] != expected["sha256"]
        or observed["size_bytes"] != int(expected["size_bytes"])
    ):
        raise ValueError(
            f"Step7-v3.1 frozen source artifact drift: {role}; "
            f"expected={expected} observed={observed}"
        )
    return observed


def prepare_public(policy: dict) -> dict:
    """Build every model input without opening the label or evidence files."""
    public_input_names = (
        "seller_profiles",
        "item_identity_signals",
        "component_assignments",
    )
    input_manifest = common.validate_input_hashes(policy, public_input_names)
    assignments_path = common.resolve(policy["inputs"]["component_assignments"]["path"])
    profiles_path = common.resolve(policy["inputs"]["seller_profiles"]["path"])
    signals_path = common.resolve(policy["inputs"]["item_identity_signals"]["path"])

    assignments = eligible_assignment_rows(policy, common.load_csv(assignments_path))
    component_field = policy["inputs"]["component_assignments"]["component_field"]
    pairs = build_pair_manifest(assignments, component_field)
    common.validate_public_pair_rows(policy, pairs)
    isolation = validate_split_isolation(pairs)
    boundary = policy["supervision_boundary"]
    if isolation["component_count_by_split"] != boundary[
        "expected_component_count_by_split"
    ]:
        raise ValueError("Step7-v3 component-count boundary drift")
    if isolation["seller_count_by_split"] != boundary[
        "expected_seller_count_by_split"
    ]:
        raise ValueError("Step7-v3 seller-count boundary drift")

    profiles_list = common.load_jsonl(profiles_path)
    profiles = {str(row["seller_uid"]): row for row in profiles_list}
    if len(profiles) != len(profiles_list):
        raise ValueError("Step7-v3 seller profiles contain duplicate seller_uid values")
    seller_split: dict[str, str] = {}
    for pair in pairs:
        for endpoint in ("seller_uid_left", "seller_uid_right"):
            seller_split[pair[endpoint]] = pair["split_name"]
    missing_profiles = sorted(set(seller_split) - set(profiles))
    if missing_profiles:
        raise ValueError(f"Step7-v3 seller profile missing: {missing_profiles[0]}")

    literals, signal_summary = common.signal_literals_by_seller(signals_path)
    eligible_profiles = [profiles[seller_uid] for seller_uid in sorted(seller_split)]
    global_tokens = common.global_identity_tokens(literals, profiles_list)
    audited_global_phrases = common.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
    protected_collision_compacts = {
        common.compact_identifier(term)
        for term in common.PROTECTED_IDENTITY_COLLISION_TERMS
    }
    if any(
        common.anchored_alias_registry_token(
            compact, audited_global_phrases
        )
        is not None
        for compact in protected_collision_compacts
    ):
        raise ValueError(
            "Step7-v3 audited identity phrases overlap protected content collisions"
        )
    # This is a label-free decontamination dictionary, not a learned feature.
    # Use every seller alias in the pinned English snapshot so that cloned ads
    # cannot retain the identity of a seller outside the 734-pair universe.
    contextual_aliases = common.contextual_global_alias_tokens(
        profiles_list, literals
    )
    contextual_alias_deletions = common.contextual_alias_deletion_tokens(
        contextual_aliases
    )
    signal_summary["global_identity_token_count"] = len(global_tokens)
    signal_summary["audited_global_identity_phrase_token_count"] = len(
        audited_global_phrases
    )
    signal_summary["global_identity_profile_scope"] = policy[
        "clean_text_contract"
    ]["global_mixed_alias_registry_scope"]
    signal_summary["global_identity_profile_count"] = len(profiles_list)
    signal_summary["contextual_global_alias_token_count"] = len(contextual_aliases)
    signal_summary["contextual_alias_one_character_omission_count"] = len(
        contextual_alias_deletions
    )
    signal_summary["contextual_alias_profile_scope"] = (
        policy["clean_text_contract"]["contextual_alias_registry_scope"]
    )
    signal_summary["contextual_alias_profile_count"] = len(profiles_list)
    signal_summary["eligible_pair_seller_count"] = len(eligible_profiles)
    signal_summary["contextual_alias_content_word_denylist"] = sorted(
        common.CONTEXTUAL_ALIAS_CONTENT_WORD_DENYLIST
    )
    seller_literals_by_uid = {
        seller_uid: common.seller_identity_literals(profiles[seller_uid])
        for seller_uid in sorted(seller_split)
    }
    seller_phrase_tokens_by_uid = {
        seller_uid: common.seller_identity_phrase_tokens(profiles[seller_uid])
        for seller_uid in sorted(seller_split)
    }
    seller_records: dict[str, dict] = {}
    redaction_totals: Counter[str] = Counter()
    global_removed_token_counts: Counter[str] = Counter()
    audited_global_phrase_removed_counts: Counter[str] = Counter()
    for seller_uid in sorted(seller_split):
        record, diagnostics = common.build_clean_seller_record(
            profiles[seller_uid],
            policy["clean_text_contract"],
            global_tokens,
            contextual_aliases,
            contextual_alias_deletions,
            audited_global_phrases,
        )
        seller_records[seller_uid] = record
        global_removed_token_counts.update(
            diagnostics["global_identifier_token_counts"]
        )
        audited_global_phrase_removed_counts.update(
            diagnostics["audited_global_identity_phrase_counts"]
        )
        redaction_totals["generic_identifier_match_count"] += diagnostics[
            "generic_identifier_match_count"
        ]
        redaction_totals["seller_local_alias_match_count"] += diagnostics[
            "seller_local_alias_match_count"
        ]
        redaction_totals["seller_local_alias_phrase_match_count"] += diagnostics[
            "seller_local_alias_phrase_match_count"
        ]
        redaction_totals[
            "audited_global_identity_phrase_match_count"
        ] += diagnostics["audited_global_identity_phrase_match_count"]
        redaction_totals["global_identifier_token_match_count"] += diagnostics[
            "global_identifier_token_match_count"
        ]
        redaction_totals["contextual_alias_match_count"] += diagnostics[
            "contextual_alias_match_count"
        ]
        redaction_totals["empty_after_redaction_count"] += int(
            diagnostics["empty_after_redaction"]
        )

    train_sellers = {
        pair[endpoint]
        for pair in pairs
        if pair["split_name"] == "train"
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    reference = common.train_reference(seller_records, train_sellers)
    safe_rows = common.build_safe_pair_rows(pairs, seller_records, reference)
    common.validate_safe_pair_feature_rows(safe_rows)
    if [row["pair_uid"] for row in safe_rows] != [row["pair_uid"] for row in pairs]:
        raise AssertionError("Step7-v3 safe feature row order differs from pair manifest")

    corpus_rows, nonempty_field_counts = build_field_corpus_rows(
        policy,
        profiles,
        seller_split,
        seller_records,
        global_tokens,
        contextual_aliases,
        contextual_alias_deletions,
        audited_global_phrases,
    )
    common.validate_field_corpus_rows(policy, corpus_rows)
    common.validate_clean_corpus_rows(
        [
            {
                "seller_uid": row["seller_uid"],
                "split_name": row["split_name"],
                "model_text": row["model_text"],
                "model_text_sha256": row["model_text_sha256"],
            }
            for row in corpus_rows
        ]
    )
    full_known_alias_census = common.full_known_alias_residual_census(
        corpus_rows, contextual_aliases
    )
    full_known_alias_census["manual_review_contract"] = (
        "all_residual_known_alias_anchors_are_reviewed_from_fixed_snapshot_"
        "model_text_and_alias_semantics_without_labels_or_evidence_types_"
        "confirmed_identity_anchors_must_move_to_the_unconditional_registry"
    )
    full_known_alias_census["manual_review_outcome"] = (
        "pass_no_confirmed_identity_anchor_remains_all_retained_anchors_are_"
        "ambiguous_or_content_collisions"
    )
    full_known_alias_census["confirmed_identity_residual_anchor_count"] = 0
    full_known_alias_census[
        "retained_ambiguous_or_content_collision_anchor_count"
    ] = full_known_alias_census["matched_registry_token_count"]
    full_known_alias_census[
        "retained_ambiguous_or_content_collision_occurrence_count"
    ] = full_known_alias_census["matched_occurrence_count"]
    full_known_alias_census["audited_public_input_sha256"] = {
        name: record["sha256"] for name, record in sorted(input_manifest.items())
    }
    full_known_alias_census["input_hash_change_requires_full_reaudit"] = True
    signal_summary["full_known_alias_residual_fixed_snapshot_census"] = (
        full_known_alias_census
    )
    embedded_identity_census = common.audited_identity_embedded_residual_census(
        corpus_rows, audited_global_phrases
    )
    embedded_identity_census["manual_review_contract"] = (
        "all_embedded_candidates_are_reviewed_from_fixed_snapshot_model_text_"
        "and_alias_semantics_without_labels_or_evidence_types_confirmed_"
        "identity_residues_must_be_removed_before_release"
    )
    embedded_identity_census["manual_review_outcome"] = (
        "pass_all_candidates_are_biohazard_tool_name_wallstreetbet_or_"
        "darkmarkets_content_collisions"
    )
    embedded_identity_census["confirmed_identity_residual_count"] = 0
    embedded_identity_census["retained_content_collision_occurrence_count"] = 5
    embedded_identity_census["retained_content_collision_scope"] = (
        "biohazard_tool_name_wallstreetbet_and_darkmarkets_contains_darkm_only"
    )
    embedded_identity_census["audited_public_input_sha256"] = {
        name: record["sha256"] for name, record in sorted(input_manifest.items())
    }
    embedded_identity_census["input_hash_change_requires_full_reaudit"] = True
    signal_summary[
        "audited_identity_embedded_residual_fixed_snapshot_census"
    ] = embedded_identity_census
    content_fidelity = content_fidelity_summary(policy, eligible_profiles, corpus_rows)
    signal_summary["global_identity_fixed_snapshot_audit"] = {
        "status": "pass_registry_is_input_hash_pinned_and_content_collisions_are_preregistered",
        "scan_scope": "actual_post_audited_phrase_post_generic_post_local_alias_global_redactions_in_855_seller_five_field_corpus",
        "removed_distinct_token_count": len(global_removed_token_counts),
        "removed_occurrence_count": int(sum(global_removed_token_counts.values())),
        "removed_token_sha256_counts": {
            common.sha256_text(token): int(count)
            for token, count in sorted(global_removed_token_counts.items())
        },
        "content_collision_denylist": sorted(
            common.GLOBAL_IDENTITY_CONTENT_COLLISION_DENYLIST
        ),
        "input_hash_change_requires_full_reaudit": True,
    }
    audited_matched_registry_tokens = {
        anchor
        for surface in audited_global_phrase_removed_counts
        if (
            anchor := common.anchored_alias_registry_token(
                surface, audited_global_phrases
            )
        )
        is not None
    }
    if (
        not audited_matched_registry_tokens
        or not audited_matched_registry_tokens.issubset(
            set(audited_global_phrases)
        )
    ):
        raise ValueError(
            "Step7-v3 audited identity-phrase matched registry is invalid"
        )
    signal_summary["audited_global_identity_phrase_fixed_snapshot_audit"] = {
        "status": "pass_manual_seller_and_market_identity_classification_is_public_input_hash_pinned",
        "scan_scope": "actual_first_stage_separator_invariant_phrase_redactions_in_855_seller_five_field_corpus",
        "registry_token_count": len(audited_global_phrases),
        "registry_tokens_canonical_sha256": common.canonical_hash(
            sorted(audited_global_phrases)
        ),
        "matched_registry_token_count": len(audited_matched_registry_tokens),
        "matched_registry_tokens_canonical_sha256": common.canonical_hash(
            sorted(audited_matched_registry_tokens)
        ),
        "unmatched_preventive_registry_token_count": len(
            set(audited_global_phrases) - audited_matched_registry_tokens
        ),
        "removed_distinct_surface_count": len(
            audited_global_phrase_removed_counts
        ),
        "removed_occurrence_count": int(
            sum(audited_global_phrase_removed_counts.values())
        ),
        "removed_surface_sha256_counts": {
            common.sha256_text(token): int(count)
            for token, count in sorted(audited_global_phrase_removed_counts.items())
        },
        "audited_public_input_sha256": {
            name: record["sha256"]
            for name, record in sorted(input_manifest.items())
        },
        "protected_content_collision_compacts": sorted(
            protected_collision_compacts
        ),
        "registry_is_disjoint_from_protected_content_collisions": True,
        "input_hash_change_requires_full_reaudit": True,
    }
    return {
        "input_manifest": input_manifest,
        "pairs": pairs,
        "isolation": isolation,
        "corpus_rows": corpus_rows,
        "reference": reference,
        "safe_rows": safe_rows,
        "signal_summary": signal_summary,
        "global_identity_tokens": global_tokens,
        "audited_global_identity_phrase_tokens": audited_global_phrases,
        "contextual_global_alias_tokens": contextual_aliases,
        "contextual_alias_deletion_tokens": contextual_alias_deletions,
        "seller_identity_literals": seller_literals_by_uid,
        "seller_identity_phrase_tokens": seller_phrase_tokens_by_uid,
        "redaction_summary": {
            key: int(value) for key, value in sorted(redaction_totals.items())
        },
        "content_fidelity": content_fidelity,
        "nonempty_field_seller_counts": nonempty_field_counts,
    }


def prepare_private_labels(
    policy: dict, pair_rows: list[dict], splits: tuple[str, ...]
) -> dict:
    """Attach labels after the public artifacts have been frozen."""
    input_manifest = common.validate_input_hashes(
        policy, ("frozen_labels", "evidence_labels")
    )
    labels_path = common.resolve(policy["inputs"]["frozen_labels"]["path"])
    evidence_path = common.resolve(policy["inputs"]["evidence_labels"]["path"])
    split_set = set(splits)
    # Project by the public split field before consulting review_label or any
    # evidence field.  The frozen source CSV contains all historical splits;
    # development materialization must not use test label values.
    split_source_rows = [
        row for row in common.load_csv(labels_path) if row.get("split_name") in split_set
    ]
    labels = eligible_label_rows(policy, split_source_rows)
    label_counts = validate_supervision_counts(policy, labels, splits)
    pair_index = {row["pair_uid"]: row for row in pair_rows}
    expected_pair_uids = {
        row["pair_uid"] for row in pair_rows if row["split_name"] in set(splits)
    }
    if {row["pair_uid"] for row in labels} != expected_pair_uids:
        raise ValueError("Step7-v3 private labels differ from the fixed public pair universe")
    evidence = evidence_index(
        [
            row
            for row in common.load_csv(evidence_path)
            if row.get("split_name") in split_set
        ]
    )
    private = {
        split: private_label_rows(split, labels, pair_index, evidence)
        for split in splits
    }
    return {
        "input_manifest": input_manifest,
        "label_counts": label_counts,
        "private": private,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument(
        "--stage",
        choices=("public", "development-labels"),
        default="public",
    )
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    policy_path = common.resolve(args.policy)
    policy = common.load_json(policy_path)
    common.validate_policy(policy)
    input_names = (
        ("seller_profiles", "item_identity_signals", "component_assignments")
        if args.stage == "public"
        else ("frozen_labels", "evidence_labels")
    )
    input_manifest = common.validate_input_hashes(policy, input_names)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "stage": args.stage,
                    "policy": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
                    "inputs": input_manifest,
                    "gpu_required_for_this_command": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    outputs = policy["outputs"]
    pair_path = common.resolve(outputs["pair_manifest"])
    corpus_path = common.resolve(outputs["field_corpus"])
    reference_path = common.resolve(outputs["train_feature_reference"])
    safe_path = common.resolve(outputs["safe_pair_features"])
    if args.stage == "public":
        prepared = prepare_public(policy)
        common.write_csv_immutable(pair_path, prepared["pairs"])
        common.write_jsonl_immutable(corpus_path, prepared["corpus_rows"])
        common.write_json_immutable(reference_path, prepared["reference"])
        common.write_csv_immutable(safe_path, prepared["safe_rows"])
        frozen_outputs = {
            "pair_manifest": verify_expected_artifact(
                policy, "pair_manifest", pair_path
            ),
            "field_corpus": verify_expected_artifact(
                policy, "field_corpus", corpus_path
            ),
            "train_feature_reference": verify_expected_artifact(
                policy, "train_feature_reference", reference_path
            ),
            "safe_pair_features": verify_expected_artifact(
                policy, "safe_pair_features", safe_path
            ),
        }
        identity_residue_scan = common.scan_final_corpus_identity_residues(
            common.load_jsonl(corpus_path),
            prepared["seller_identity_literals"],
            prepared["global_identity_tokens"],
            prepared["contextual_global_alias_tokens"],
            prepared["contextual_alias_deletion_tokens"],
            prepared["seller_identity_phrase_tokens"],
            prepared["audited_global_identity_phrase_tokens"],
        )
        output_paths = {
            "pair_manifest": pair_path,
            "field_corpus": corpus_path,
            "train_feature_reference": reference_path,
            "safe_pair_features": safe_path,
        }
        manifest = {
            "step": "step7_v3_1_prepare_standalone_label_free_source_data",
            "version": policy["version"],
            "policy_path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
            "policy_sha256": common.sha256_file(policy_path),
            "generator_script_path": str(PREPARATION_SCRIPT.relative_to(common.ROOT)).replace("\\", "/"),
            "generator_script_sha256": common.sha256_file(PREPARATION_SCRIPT),
            "common_script_sha256": common.sha256_file(COMMON_SCRIPT),
            "redaction_dependency_script_path": str(
                common.STEP3_SCRIPT.relative_to(common.ROOT)
            ).replace("\\", "/"),
            "redaction_dependency_script_sha256": common.sha256_file(
                common.STEP3_SCRIPT
            ),
            "input_manifest": prepared["input_manifest"],
            "public_pair_count_by_split": dict(
                Counter(row["split_name"] for row in prepared["pairs"])
            ),
            "split_isolation": prepared["isolation"],
            "seller_count": len(prepared["corpus_rows"]),
            "nonempty_field_seller_counts": prepared[
                "nonempty_field_seller_counts"
            ],
            "complete_field_text_replay": True,
            "train_reference_seller_count": prepared["reference"]["train_seller_count"],
            "signal_summary": prepared["signal_summary"],
            "redaction_summary": prepared["redaction_summary"],
            "content_fidelity": prepared["content_fidelity"],
            "identity_residue_scan": identity_residue_scan,
            "feature_generation_uses_review_label_values": False,
            "feature_generation_uses_evidence_type_values": False,
            "pair_feature_roles": policy["pair_feature_roles"],
            "shortcut_features_generated_for_audit_only": True,
            "shortcut_features_eligible_for_model_training_or_selection": False,
            "boundary_source_file_contains_review_label_column": True,
            "boundary_projection_fields": [
                "dataset",
                "pair_uid",
                "split_name",
                "seller_uid_left",
                "seller_uid_right",
                policy["inputs"]["component_assignments"]["component_field"],
            ],
            "pair_universe_source": "component_assignments_public_column_projection",
            "label_permutation_invariance_required": policy["clean_text_contract"][
                "labels_must_not_change_corpus_or_pair_features"
            ],
            "output_files": frozen_outputs,
            "formal_model_encoding_pending_linux_gpu": True,
        }
        common.validate_content_fidelity_manifest(policy, manifest)
        manifest_path = common.resolve(outputs["preparation_manifest"])
        common.write_json_immutable(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    manifest_path = common.resolve(outputs["preparation_manifest"])
    if not pair_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run the Step7-v3.1 public preparation stage first")
    public_manifest = common.load_json(manifest_path)
    public_pair_record = public_manifest["output_files"]["pair_manifest"]
    if common.sha256_file(pair_path) != public_pair_record["sha256"]:
        raise ValueError("Step7-v3 public pair manifest drift before label attachment")
    pair_rows = common.load_csv(pair_path)
    splits = ("train", "valid")
    private_paths = {
        "train": common.resolve(outputs["train_labels"]),
        "valid": common.resolve(outputs["valid_labels"]),
    }
    label_manifest_path = common.resolve(outputs["development_labels_manifest"])
    prepared = prepare_private_labels(policy, pair_rows, splits)
    for split, rows in prepared["private"].items():
        common.write_csv_immutable(private_paths[split], rows)
        verify_expected_artifact(policy, f"{split}_labels", private_paths[split])
    label_manifest = {
        "step": "step7_v3_1_prepare_development_labels",
        "version": policy["version"],
        "policy_sha256": common.sha256_file(policy_path),
        "generator_script_path": str(PREPARATION_SCRIPT.relative_to(common.ROOT)).replace("\\", "/"),
        "generator_script_sha256": common.sha256_file(PREPARATION_SCRIPT),
        "common_script_sha256": common.sha256_file(COMMON_SCRIPT),
        "redaction_dependency_script_path": str(
            common.STEP3_SCRIPT.relative_to(common.ROOT)
        ).replace("\\", "/"),
        "redaction_dependency_script_sha256": common.sha256_file(common.STEP3_SCRIPT),
        "public_preparation_manifest_sha256": common.sha256_file(manifest_path),
        "input_manifest": prepared["input_manifest"],
        "label_counts": prepared["label_counts"],
        "splits_written": list(splits),
        "historical_test_metrics_must_wait_for_frozen_selection": True,
        "other_split_label_values_used_during_materialization": False,
        "split_projection_applied_before_label_or_evidence_access": True,
        "prospective_claim_allowed": False,
        "output_files": {
            f"private_labels_{split}": {
                "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
                "sha256": common.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for split, path in private_paths.items()
        },
    }
    common.write_json_immutable(label_manifest_path, label_manifest)
    print(json.dumps(label_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
