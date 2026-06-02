from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "schema" / "step15_evidence_type_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build conservative Step 15 auxiliary evidence_type labels from frozen Step 5 labels "
            "and Step 7 pair features. This script does not modify Step 5 files."
        )
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH), help="Path to Step 15 policy JSON.")
    parser.add_argument(
        "--pool",
        action="append",
        dest="pools",
        help="Pool to process. Repeat to process multiple pools. Defaults to all policy pools.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def as_int(row: dict, key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def text_contains_any(value: str, needles: list[str]) -> bool:
    haystack = str(value or "").lower()
    return any(str(needle).lower() in haystack for needle in needles)


def rule_hits(row: dict) -> set[str]:
    raw = row.get("candidate_rule_hits", "")
    return {part.strip() for part in str(raw).split("|") if part.strip()}


def bool_feature(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def has_direct_identifier(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    rules = cfg["direct_identifier_positive"]
    reasons: list[str] = []
    if as_int(row, "shared_contact_count") >= int(rules.get("shared_contact_count_min", 1)):
        reasons.append("step5_shared_contact_count")
    if as_int(row, "shared_pgp_fingerprint_count") >= int(rules.get("shared_pgp_fingerprint_count_min", 1)):
        reasons.append("step5_shared_pgp_fingerprint_count")
    if as_int(row, "shared_contact_count_capped") >= int(rules.get("shared_contact_count_min", 1)):
        reasons.append("feature_shared_contact_count")
    if as_int(row, "shared_pgp_fingerprint_count_capped") >= int(rules.get("shared_pgp_fingerprint_count_min", 1)):
        reasons.append("feature_shared_pgp_fingerprint_count")
    if bool_feature(row, "has_shared_contact_exact"):
        reasons.append("feature_has_shared_contact_exact")
    if bool_feature(row, "has_shared_pgp_fingerprint"):
        reasons.append("feature_has_shared_pgp_fingerprint")
    review_stratum = str(row.get("review_stratum", "")).lower()
    for token in rules.get("review_stratum_contains", []):
        if token.lower() in review_stratum:
            reasons.append(f"review_stratum:{token}")
    hits = rule_hits(row)
    for token in rules.get("candidate_rule_hits_any", []):
        if token in hits:
            reasons.append(f"candidate_rule_hit:{token}")
    return bool(reasons), sorted(set(reasons))


def has_component_anchor(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    rules = cfg["component_anchor_positive"]
    if text_contains_any(row.get("review_notes", ""), rules.get("review_notes_keywords", [])):
        return True, ["review_notes_component_anchor_keyword"]
    return False, []


def has_template_clone(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    rules = cfg["template_clone_negative"]
    reasons: list[str] = []
    if as_int(row, "shared_title_count") >= int(rules.get("shared_title_count_min", 1)):
        reasons.append("step5_shared_title_count")
    if as_int(row, "shared_description_count") >= int(rules.get("shared_description_count_min", 1)):
        reasons.append("step5_shared_description_count")
    if as_int(row, "shared_title_count_capped") >= int(rules.get("shared_title_count_min", 1)):
        reasons.append("feature_shared_title_count")
    if as_int(row, "shared_description_count_capped") >= int(rules.get("shared_description_count_min", 1)):
        reasons.append("feature_shared_description_count")
    if bool_feature(row, "has_shared_title_clone"):
        reasons.append("feature_has_shared_title_clone")
    if bool_feature(row, "has_shared_description_clone"):
        reasons.append("feature_has_shared_description_clone")
    if as_int(row, "shared_low_df_sentence_count") >= int(rules.get("shared_low_df_sentence_count_min", 1)):
        reasons.append("feature_shared_low_df_sentence_count")
    if as_int(row, "shared_rare_ngram_count") >= int(rules.get("shared_rare_ngram_count_min", 1)):
        reasons.append("feature_shared_rare_ngram_count")
    review_stratum = str(row.get("review_stratum", "")).lower()
    for token in rules.get("review_stratum_contains", []):
        if token.lower() in review_stratum:
            reasons.append(f"review_stratum:{token}")
    hits = rule_hits(row)
    for token in rules.get("candidate_rule_hits_any", []):
        if token in hits:
            reasons.append(f"candidate_rule_hit:{token}")
    return bool(reasons), sorted(set(reasons))


def has_semantic_topic(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    rules = cfg["semantic_topic_negative"]
    reasons: list[str] = []
    for feature in (
        "embedding_cosine_multilingual_e5_large",
        "embedding_cosine_bge_m3",
        "embedding_cosine_labse",
        "embedding_cosine_gte_multilingual_base",
    ):
        threshold_key = f"{feature}_min"
        threshold = float(rules.get(threshold_key, 1.0))
        if as_float(row, feature, 0.0) >= threshold:
            reasons.append(f"{feature}>={threshold:g}")
    structural_max = float(rules.get("structural_support_score_raw_max", 1.0))
    if as_float(row, "structural_support_score_raw", 0.0) > structural_max:
        return False, []
    if as_int(row, "shared_title_count_capped") > int(rules.get("shared_title_count_max", 0)):
        return False, []
    if as_int(row, "shared_description_count_capped") > int(rules.get("shared_description_count_max", 0)):
        return False, []
    return bool(reasons), sorted(set(reasons))


def has_public_contact_or_url_noise(row: dict, cfg: dict) -> tuple[bool, list[str]]:
    rules = cfg["public_contact_or_url_noise"]
    reasons: list[str] = []
    contact_like, contact_reasons = has_direct_identifier(row, cfg)
    if contact_like:
        reasons.extend(f"contact_like:{reason}" for reason in contact_reasons)
    if text_contains_any(row.get("review_notes", ""), rules.get("review_notes_keywords", [])):
        reasons.append("review_notes_public_or_non_seller_keyword")
    # Negative contact-like overlap is treated as identifier noise even without an explicit note,
    # because the row has already been reviewed as different-controller supervision.
    if str(row.get("review_label", "")).strip() == "negative" and contact_like:
        reasons.append("negative_label_with_contact_like_overlap")
    return bool(reasons), sorted(set(reasons))


def classify_row(row: dict, policy: dict) -> dict:
    cfg = policy["auxiliary_label_rules"]
    review_label = str(row.get("review_label", "")).strip()
    identity_label = policy["identity_label_mapping"].get(review_label, "uncertain")

    direct, direct_reasons = has_direct_identifier(row, cfg)
    component, component_reasons = has_component_anchor(row, cfg)
    template, template_reasons = has_template_clone(row, cfg)
    semantic, semantic_reasons = has_semantic_topic(row, cfg)
    public_noise, public_noise_reasons = has_public_contact_or_url_noise(row, cfg)

    evidence_type = "uncertain_insufficient_evidence"
    evidence_reasons: list[str] = []
    confident = False

    if review_label == "positive":
        confident = True
        if direct:
            evidence_type = "same_controller_direct_identifier"
            evidence_reasons = direct_reasons
        elif component:
            evidence_type = "same_controller_component_anchor"
            evidence_reasons = component_reasons
        else:
            evidence_type = "same_controller_style_structural_soft"
            evidence_reasons = ["positive_without_direct_identifier"]
    elif review_label == "negative":
        confident = True
        if public_noise:
            evidence_type = "public_contact_or_url_noise"
            evidence_reasons = public_noise_reasons
        elif template:
            evidence_type = "template_clone_not_controller"
            evidence_reasons = template_reasons
        elif semantic:
            evidence_type = "semantic_topic_not_controller"
            evidence_reasons = semantic_reasons
        else:
            evidence_type = "ordinary_negative"
            evidence_reasons = ["negative_without_hard_ambiguity"]
    else:
        evidence_reasons = ["uncertain_or_not_supervision_label"]

    identity_training_eligible = (
        review_label in {"positive", "negative"}
        and str(row.get("usable_for_supervision", "")) == "1"
        and str(row.get("usable_for_core_transfer", "")) == "1"
    )

    return {
        "identity_label": identity_label,
        "evidence_type": evidence_type,
        "evidence_type_confident": "1" if confident else "0",
        "identity_training_eligible": "1" if identity_training_eligible else "0",
        "has_direct_identifier_signal": "1" if direct else "0",
        "has_template_clone_signal": "1" if template else "0",
        "has_semantic_topic_signal": "1" if semantic else "0",
        "has_public_contact_or_url_noise_signal": "1" if public_noise else "0",
        "evidence_type_reasons": "|".join(evidence_reasons),
    }


def output_row(row: dict, classification: dict) -> dict:
    selected = {
        "pair_uid": row.get("pair_uid", ""),
        "data_bucket": row.get("data_bucket", ""),
        "candidate_language": row.get("candidate_language", ""),
        "split_name": row.get("split_name", ""),
        "split_component_id": row.get("split_component_id", ""),
        "review_label": row.get("review_label", ""),
        "review_stratum": row.get("review_stratum", ""),
        "usable_for_supervision": row.get("usable_for_supervision", ""),
        "usable_for_core_transfer": row.get("usable_for_core_transfer", ""),
        "candidate_rule_hits": row.get("candidate_rule_hits", ""),
        "shared_contact_count": row.get("shared_contact_count", ""),
        "shared_pgp_fingerprint_count": row.get("shared_pgp_fingerprint_count", ""),
        "shared_title_count": row.get("shared_title_count", ""),
        "shared_description_count": row.get("shared_description_count", ""),
        "embedding_cosine_multilingual_e5_large": row.get("embedding_cosine_multilingual_e5_large", ""),
        "embedding_cosine_bge_m3": row.get("embedding_cosine_bge_m3", ""),
        "embedding_cosine_labse": row.get("embedding_cosine_labse", ""),
        "structural_support_score_raw": row.get("structural_support_score_raw", ""),
        "source_seller_raw_left": row.get("source_seller_raw_left", ""),
        "source_seller_raw_right": row.get("source_seller_raw_right", ""),
    }
    selected.update(classification)
    return selected


def summarize_rows(rows: list[dict]) -> dict:
    by_split = defaultdict(Counter)
    by_label = Counter()
    by_evidence = Counter()
    eligible_by_split_evidence = defaultdict(Counter)
    for row in rows:
        split = row.get("split_name", "")
        label = row.get("review_label", "")
        evidence_type = row.get("evidence_type", "")
        by_split[split][evidence_type] += 1
        by_label[label] += 1
        by_evidence[evidence_type] += 1
        if row.get("identity_training_eligible") == "1":
            eligible_by_split_evidence[split][evidence_type] += 1
    return {
        "row_count": len(rows),
        "review_label_counts": dict(sorted(by_label.items())),
        "evidence_type_counts": dict(sorted(by_evidence.items())),
        "evidence_type_by_split": {split: dict(sorted(counter.items())) for split, counter in sorted(by_split.items())},
        "identity_training_eligible_by_split_evidence": {
            split: dict(sorted(counter.items())) for split, counter in sorted(eligible_by_split_evidence.items())
        },
    }


def process_pool(pool_name: str, pool_cfg: dict, policy: dict) -> dict:
    frozen_path = resolve_path(pool_cfg["frozen_labels"])
    feature_path = resolve_path(pool_cfg["pair_features"])
    output_path = resolve_path(pool_cfg["label_output"])

    frozen_rows = step7.load_csv(frozen_path)
    feature_rows = step7.load_csv(feature_path)
    joined_rows = step7.join_frozen_with_features(frozen_rows, feature_rows)

    output_rows = []
    for row in joined_rows:
        classification = classify_row(row, policy)
        output_rows.append(output_row(row, classification))

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    step7.write_csv(output_path, output_rows, fieldnames)

    summary = summarize_rows(output_rows)
    summary.update(
        {
            "pool": pool_name,
            "frozen_labels": str(frozen_path.relative_to(ROOT)),
            "pair_features": str(feature_path.relative_to(ROOT)),
            "output": str(output_path.relative_to(ROOT)),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    policy_path = resolve_path(args.policy)
    policy = step7.load_json(policy_path)
    pool_names = args.pools or list(policy["pools"].keys())

    summaries = {}
    for pool_name in pool_names:
        if pool_name not in policy["pools"]:
            raise SystemExit(f"Unknown Step15 pool: {pool_name}")
        summaries[pool_name] = process_pool(pool_name, policy["pools"][pool_name], policy)

    summary = {
        "step": "step15_build_evidence_type_labels",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy.get("version"),
        "pools": summaries,
        "hard_rule_status": {
            "step5_files_modified": False,
            "auxiliary_labels_only": True,
            "uncertain_rows_binary_training_eligible": False,
        },
    }
    summary_path = resolve_path(policy["label_summary_output"])
    step7.write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
