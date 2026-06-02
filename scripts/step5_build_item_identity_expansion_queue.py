from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import step4_build_silver_candidates as step4
import step5_build_positive_anchor_expansion_queue as positive_anchor
import step5_build_review_strata as step5_strata


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_item_identity_expansion_policy.json"
TRUST_SUFFIX_RE = re.compile(r"\s*\(\d+%\)\s*$", re.I)
NON_ALIAS_CHARS_RE = re.compile(r"[^0-9a-z\u3400-\u9fff/]+", re.I)

EXTRA_FIELDS = [
    "item_identity_queue_rank",
    "target_bucket",
    "target_reason",
    "target_action",
    "suggested_label",
    "suggested_confidence",
    "anchor_token",
    "anchor_type",
    "anchor_seller_frequency",
    "anchor_evidence_levels",
    "left_item_signal_count",
    "right_item_signal_count",
    "left_item_source_rows",
    "right_item_source_rows",
    "left_item_contexts",
    "right_item_contexts",
    "both_seller_facing_context",
    "product_data_risk_any",
    "source_existing_active_bool",
    "source_existing_step4_bool",
    "existing_active_review_status",
    "existing_active_review_label",
    "existing_frozen_bool",
    "review_instruction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Step 5 positive-candidate queue from Step 3 item-level identity signals."
    )
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="Path to the Step 5 item-identity expansion policy JSON.",
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


def is_reviewed(row: dict) -> bool:
    return normalize_label(row.get("review_label")) in {"positive", "negative", "uncertain"}


def is_pending(row: dict) -> bool:
    return normalize_text(row.get("review_status")).lower() in {"", "pending"} and normalize_label(row.get("review_label")) == ""


def pair_uid_from_sellers(left_uid: str, right_uid: str) -> str:
    left, right = sorted([left_uid, right_uid])
    return left + "||" + right


def normalize_soft_alias(value: object) -> str:
    text = normalize_text(value).casefold()
    text = TRUST_SUFFIX_RE.sub("", text)
    text = NON_ALIAS_CHARS_RE.sub("", text)
    return text


def is_soft_same_alias_candidate(row: dict) -> bool:
    left = normalize_soft_alias(row.get("source_seller_raw_left"))
    right = normalize_soft_alias(row.get("source_seller_raw_right"))
    return bool(left and right and left == right)


def load_profiles(policy: dict) -> dict[str, object]:
    inputs = policy["inputs"]
    data_bucket = policy.get("scope") or inputs.get("data_bucket", "zh_target_strict")
    language = inputs.get("language") or ("en" if data_bucket == "en_content_train_pool" else "zh")
    step4_schema = load_json(ROOT / inputs["step4_schema"])
    raw_profiles = step4.load_jsonl(ROOT / inputs["seller_profiles"])
    profiles = step4.build_seller_profiles(
        rows=raw_profiles,
        data_bucket=data_bucket,
        language=language,
        stopwords={value.lower() for value in step4_schema["filtering_policy"]["contact_noise_stopwords"]},
        min_config=step4_schema["filtering_policy"]["content_minimums"],
        pgp_alias_map=step4.load_pgp_alias_map(),
    )
    step4.compute_retrieval_weights(profiles, step4_schema["retrieval_policy"][data_bucket])
    return {profile.seller_uid: profile for profile in profiles}


def compact_context(rows: list[dict], limit: int = 3) -> str:
    chunks = []
    for row in sorted(rows, key=lambda item: (int(item.get("source_row_number", 0) or 0), item.get("source_field", "")))[:limit]:
        source = f"{row.get('source_dataset')}#{row.get('source_row_number')}:{row.get('source_field')}"
        chunks.append(f"{source}: {normalize_text(row.get('context'))}")
    return " || ".join(chunks)


def compact_source_rows(rows: list[dict], limit: int = 8) -> str:
    values = []
    for row in sorted(rows, key=lambda item: (item.get("source_dataset", ""), int(item.get("source_row_number", 0) or 0)))[:limit]:
        values.append(f"{row.get('source_dataset')}#{row.get('source_row_number')}")
    return " || ".join(values)


def review_instruction(anchor_type: str) -> str:
    if anchor_type == "email":
        return "Review as positive only if the email is seller-facing contact on both sides, not product/victim/sample data."
    if anchor_type == "external_url":
        return "Support-only URL evidence; do not mark positive without independent seller-facing direct contact."
    return "Review as positive only if both item-level contexts present the shared identifier as seller-facing identity/contact evidence."


def make_output_row(
    *,
    rank: int,
    pair_uid: str,
    candidate_row: dict,
    active_row: dict | None,
    anchor_type: str,
    token: str,
    seller_frequency: int,
    left_signals: list[dict],
    right_signals: list[dict],
    existing_step4: bool,
) -> dict:
    base = dict(candidate_row)
    if active_row:
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
    else:
        base["review_stratum"] = step5_strata.classify_review_stratum(candidate_row)

    evidence_levels = sorted({row.get("evidence_level", "") for row in left_signals + right_signals if row.get("evidence_level")})
    product_risk_any = any(to_int(row.get("product_data_risk_context")) > 0 for row in left_signals + right_signals)
    both_seller_facing = all(
        any(to_int(row.get("seller_facing_context")) > 0 for row in side_rows)
        for side_rows in (left_signals, right_signals)
    )
    base.update(
        {
            "pair_uid": pair_uid,
            "review_stratum": "identifier_plus_text"
            if to_int(base.get("shared_title_count")) > 0 or to_int(base.get("shared_description_count")) > 0
            else "identifier_primary",
            "review_priority": "high",
            "shared_contact_count": max(1, to_int(base.get("shared_contact_count"))),
            "shared_contact_types": anchor_type
            if not normalize_text(base.get("shared_contact_types"))
            else normalize_text(base.get("shared_contact_types")),
            "shared_contact_values": f"{anchor_type}:{token}"
            if not normalize_text(base.get("shared_contact_values"))
            else normalize_text(base.get("shared_contact_values")),
            "candidate_rule_hits": "|".join(
                sorted(
                    set(filter(None, normalize_text(base.get("candidate_rule_hits")).split("|")))
                    | {"item_level_shared_identity_signal", "shared_contact_exact"}
                )
            ),
            "item_identity_queue_rank": rank,
            "target_bucket": "item_level_shared_direct_identity",
            "target_reason": "item_identity_expansion|shared_seller_facing_direct_identifier",
            "target_action": "review_existing_pending_queue_row" if active_row else "append_candidate_then_review",
            "suggested_label": "positive_candidate",
            "suggested_confidence": "high" if anchor_type != "email" else "medium",
            "anchor_token": f"{anchor_type}:{token}",
            "anchor_type": anchor_type,
            "anchor_seller_frequency": seller_frequency,
            "anchor_evidence_levels": "|".join(evidence_levels),
            "left_item_signal_count": len(left_signals),
            "right_item_signal_count": len(right_signals),
            "left_item_source_rows": compact_source_rows(left_signals),
            "right_item_source_rows": compact_source_rows(right_signals),
            "left_item_contexts": compact_context(left_signals),
            "right_item_contexts": compact_context(right_signals),
            "both_seller_facing_context": int(both_seller_facing),
            "product_data_risk_any": int(product_risk_any),
            "source_existing_active_bool": int(active_row is not None),
            "source_existing_step4_bool": int(existing_step4),
            "existing_active_review_status": active_row.get("review_status", "") if active_row else "",
            "existing_active_review_label": active_row.get("review_label", "") if active_row else "",
            "existing_frozen_bool": 0,
            "review_instruction": review_instruction(anchor_type),
            "review_status": "pending",
            "review_label": "",
            "reviewer_id": "",
            "review_notes": "",
        }
    )
    return base


def eligible_signal(row: dict, selection: dict) -> bool:
    contact_type = row.get("contact_type")
    if contact_type not in set(selection["eligible_contact_types"]):
        return False
    normalized_value = normalize_text(row.get("normalized_value")).casefold()
    excluded_by_type = selection.get("excluded_normalized_values_by_type", {})
    excluded_values = set(excluded_by_type.get("*", [])) | set(excluded_by_type.get(contact_type, []))
    if normalized_value in {normalize_text(value).casefold() for value in excluded_values}:
        return False
    if bool(selection.get("require_direct_identity_eligible", True)) and to_int(row.get("direct_identity_eligible")) <= 0:
        return False
    if bool(selection.get("exclude_product_data_risk", True)) and to_int(row.get("product_data_risk_context")) > 0:
        return False
    if bool(selection.get("exclude_support_only", True)) and to_int(row.get("support_only")) > 0:
        return False
    return True


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = load_json(policy_path)
    inputs = policy["inputs"]
    selection = policy["selection"]

    item_signals, _ = load_csv(ROOT / inputs["item_identity_signals"])
    candidate_rows, candidate_fields = load_csv(ROOT / inputs["step4_candidates"])
    active_rows, active_fields = load_csv(ROOT / inputs["active_review_queue"])
    frozen_rows, _ = load_csv(ROOT / inputs["active_frozen_labels"])
    profile_by_uid = load_profiles(policy)

    candidate_index = {row["pair_uid"]: row for row in candidate_rows}
    active_index = {row["pair_uid"]: row for row in active_rows}
    frozen_pair_uids = {row["pair_uid"] for row in frozen_rows}
    reviewed_pair_uids = {
        row["pair_uid"] for row in active_rows if is_reviewed(row)
    }
    reviewed_pair_uids.update(row["pair_uid"] for row in frozen_rows if is_reviewed(row))
    max_active_rank = max((to_int(row.get("balanced_review_rank")) for row in active_rows), default=0)
    next_rank = max_active_rank

    signal_index: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    filter_counts = Counter()
    for row in item_signals:
        if not eligible_signal(row, selection):
            filter_counts["ineligible_item_signal"] += 1
            continue
        signal_index[(row["contact_type"], row["normalized_value"])][row["seller_uid"]].append(row)

    type_priority = {contact_type: idx for idx, contact_type in enumerate(selection["type_priority"])}
    max_freq_default = int(selection["max_token_seller_frequency_default"])
    max_freq_by_type = {
        str(key): int(value)
        for key, value in selection.get("max_token_seller_frequency_by_type", {}).items()
    }
    min_freq = int(selection["min_token_seller_frequency"])
    target_queue_size = selection.get("target_queue_size", {})
    selection_limit = int(selection.get("selection_limit") or target_queue_size.get("goal_max") or 0)
    require_primary_scope = bool(selection.get("require_primary_scope", True))
    exclude_soft_same_alias = bool(selection.get("exclude_soft_same_alias", False))
    max_pairs_per_anchor_token = int(selection.get("max_pairs_per_anchor_token") or 0)

    selected_rows: list[dict] = []
    selected_pair_uids: set[str] = set()
    selected_token_pair_counts: Counter = Counter()
    token_group_counts = Counter()
    skipped_pair_counts = Counter()
    candidates = []
    shared_token_samples = []
    skipped_pair_examples = []
    for (contact_type, token), seller_rows in signal_index.items():
        seller_uids = sorted(seller_rows)
        seller_frequency = len(seller_uids)
        max_freq = max_freq_by_type.get(contact_type, max_freq_default)
        if seller_frequency < min_freq or seller_frequency > max_freq:
            token_group_counts["frequency_outside_bounds"] += 1
            continue
        token_group_counts["shared_token_group"] += 1
        if len(shared_token_samples) < 25:
            shared_token_samples.append(
                {
                    "anchor_token": f"{contact_type}:{token}",
                    "seller_frequency": seller_frequency,
                    "seller_uids": " || ".join(seller_uids[:10]),
                }
            )
        for left_uid, right_uid in combinations(seller_uids, 2):
            pair_uid = pair_uid_from_sellers(left_uid, right_uid)
            if pair_uid in selected_pair_uids:
                skipped_pair_counts["duplicate_pair"] += 1
                continue
            if pair_uid in frozen_pair_uids:
                skipped_pair_counts["frozen_pair"] += 1
                if len(skipped_pair_examples) < 25:
                    skipped_pair_examples.append(
                        {
                            "pair_uid": pair_uid,
                            "anchor_token": f"{contact_type}:{token}",
                            "skip_reason": "frozen_pair",
                        }
                    )
                continue
            if pair_uid in reviewed_pair_uids:
                skipped_pair_counts["already_reviewed"] += 1
                if len(skipped_pair_examples) < 25:
                    skipped_pair_examples.append(
                        {
                            "pair_uid": pair_uid,
                            "anchor_token": f"{contact_type}:{token}",
                            "skip_reason": "already_reviewed",
                        }
                    )
                continue
            active_row = active_index.get(pair_uid)
            if active_row and not is_pending(active_row):
                skipped_pair_counts["active_not_pending"] += 1
                if len(skipped_pair_examples) < 25:
                    skipped_pair_examples.append(
                        {
                            "pair_uid": pair_uid,
                            "anchor_token": f"{contact_type}:{token}",
                            "skip_reason": "active_not_pending",
                        }
                    )
                continue
            if left_uid not in profile_by_uid or right_uid not in profile_by_uid:
                skipped_pair_counts["profile_missing"] += 1
                continue
            existing_step4 = pair_uid in candidate_index
            candidate_row = candidate_index.get(pair_uid)
            if not candidate_row:
                candidate_row = positive_anchor.make_candidate_row(
                    profile_by_uid[left_uid],
                    profile_by_uid[right_uid],
                    shared_contacts=[f"{contact_type}:{token}"],
                    extra_rule_hits=["item_level_shared_identity_signal"],
                )
            if require_primary_scope and candidate_row.get("candidate_scope") != "sockpuppet_primary":
                skipped_pair_counts["non_primary_scope"] += 1
                continue
            if exclude_soft_same_alias and is_soft_same_alias_candidate(candidate_row):
                skipped_pair_counts["soft_same_alias"] += 1
                continue
            score = to_float(candidate_row.get("candidate_rank_score"))
            candidates.append(
                (
                    type_priority.get(contact_type, 999),
                    seller_frequency,
                    -score,
                    pair_uid,
                    contact_type,
                    token,
                    seller_rows[left_uid],
                    seller_rows[right_uid],
                    candidate_row,
                    active_row,
                    existing_step4,
                )
            )

    for (
        _type_priority,
        seller_frequency,
        _neg_score,
        pair_uid,
        contact_type,
        token,
        left_signals,
        right_signals,
        candidate_row,
        active_row,
        existing_step4,
    ) in sorted(candidates):
        if pair_uid in selected_pair_uids:
            skipped_pair_counts["duplicate_pair_after_sort"] += 1
            continue
        token_key = f"{contact_type}:{token}"
        if max_pairs_per_anchor_token > 0 and selected_token_pair_counts[token_key] >= max_pairs_per_anchor_token:
            skipped_pair_counts["anchor_token_pair_cap"] += 1
            continue
        selected_pair_uids.add(pair_uid)
        selected_token_pair_counts[token_key] += 1
        next_rank += 1
        selected_rows.append(
            make_output_row(
                rank=len(selected_rows) + 1,
                pair_uid=pair_uid,
                candidate_row=candidate_row,
                active_row=active_row,
                anchor_type=contact_type,
                token=token,
                seller_frequency=seller_frequency,
                left_signals=left_signals,
                right_signals=right_signals,
                existing_step4=existing_step4,
            )
        )
        selected_rows[-1]["balanced_review_rank"] = selected_rows[-1].get("balanced_review_rank") or next_rank
        if selection_limit > 0 and len(selected_rows) >= selection_limit:
            break

    output_paths = policy["outputs"]
    fieldnames = []
    for field in active_fields + candidate_fields + EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in selected_rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    write_csv(ROOT / output_paths["targeted_review_queue"], selected_rows, fieldnames)

    summary = {
        "policy_path": str(policy_path.relative_to(ROOT)),
        "input_item_signal_count": len(item_signals),
        "eligible_item_signal_count": sum(len(rows) for seller_rows in signal_index.values() for rows in seller_rows.values()),
        "eligible_token_group_count": len(signal_index),
        "token_group_counts": dict(token_group_counts),
        "selected_row_count": len(selected_rows),
        "selected_counts_by_anchor_type": dict(Counter(row["anchor_type"] for row in selected_rows)),
        "selected_counts_by_existing_step4": dict(Counter(str(row["source_existing_step4_bool"]) for row in selected_rows)),
        "selection_limit": selection_limit,
        "max_pairs_per_anchor_token": max_pairs_per_anchor_token,
        "filter_counts": dict(filter_counts),
        "skipped_pair_counts": dict(skipped_pair_counts),
        "shared_token_samples": shared_token_samples,
        "skipped_pair_examples": skipped_pair_examples,
        "top_rows": [
            {
                "pair_uid": row["pair_uid"],
                "anchor_token": row["anchor_token"],
                "source_existing_step4_bool": row["source_existing_step4_bool"],
                "candidate_rank_score": row.get("candidate_rank_score", ""),
                "left_item_contexts": row["left_item_contexts"],
                "right_item_contexts": row["right_item_contexts"],
            }
            for row in selected_rows[:10]
        ],
        "outputs": {
            "targeted_review_queue": output_paths["targeted_review_queue"],
            "summary": output_paths["summary"],
            "codex_review_summary": output_paths.get("codex_review_summary", ""),
        },
        "hard_rule": "Rows are positive candidates only. They must not enter Step 5 supervision until reviewed and frozen.",
    }
    write_json(ROOT / output_paths["summary"], summary)
    if output_paths.get("codex_review_summary"):
        review_summary = {
            "policy_path": str(policy_path.relative_to(ROOT)),
            "queue_path": output_paths["targeted_review_queue"],
            "review_status": "not_applicable_empty_queue" if not selected_rows else "pending_manual_or_codex_review",
            "reviewed_row_count": 0,
            "label_counts": {},
            "reason": (
                "No unreviewed item-level shared direct-identity pairs survived frozen/reviewed-pair exclusion."
                if not selected_rows
                else "Queue has rows and must be conservatively reviewed before any Step 5 application."
            ),
        }
        write_json(ROOT / output_paths["codex_review_summary"], review_summary)
    print(f"Wrote {ROOT / output_paths['targeted_review_queue']}")
    print(f"Wrote {ROOT / output_paths['summary']}")


if __name__ == "__main__":
    main()
