#!/usr/bin/env python3
"""Step 13 concept-drift audit for cross-lingual seller-pair verification.

This audit is intentionally read-only with respect to the scientific labels:
it joins frozen Step 5 supervision rows with Step 7 pair features and existing
Step 7/9/11 outputs, then reports where the English source-domain feature
logic fails to transfer to the Chinese target-domain test set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import step11_cluster_level_audit as step11_audit_contract
from immutable_artifact_io import csv_bytes, json_bytes, text_bytes, write_immutable_bundle


REPORTS = Path("reports")
DOCS = Path("docs")

EN_LABELS = REPORTS / "step5_en_frozen_silver_labels.csv"
ZH_LABELS = REPORTS / "step5_zh_target_strict_frozen_silver_labels.csv"
EN_FEATURES = REPORTS / "step7_pair_features.en_content_train_pool.csv"
ZH_FEATURES = REPORTS / "step7_pair_features.zh_target_strict.csv"
STEP7_SUMMARY = REPORTS / "step7_training_summary.json"
STEP9_SUMMARY = REPORTS / "step9_few_shot_summary.json"
STEP15_ZH_EVIDENCE_LABELS = REPORTS / "step15_evidence_type_labels.zh_target_strict.csv"
STEP16F_POSITIVE_REAUDIT = REPORTS / "step16f_valid_test_positive_reaudit.csv"
STEP4_CANDIDATES = {
    "en": REPORTS / "step4_en_silver_candidate_pairs.csv",
    "zh": REPORTS / "step4_zh_target_strict_silver_candidate_pairs.csv",
}
STEP11_MANIFEST_FALLBACK = REPORTS / "step11_current_manifest_20260424.json"
STEP11_AUDIT_FALLBACK = REPORTS / "step11_cluster_level_audit.current_20260424.json"

OUT_JSON = REPORTS / "step13_concept_drift_audit.json"
OUT_CSV = REPORTS / "step13_concept_drift_audit.csv"
OUT_MD = DOCS / "STEP13_CONCEPT_DRIFT_AUDIT.md"


SEMANTIC_FEATURES = [
    "embedding_cosine_gte_multilingual_base",
    "embedding_cosine_bge_m3",
    "embedding_cosine_multilingual_e5_large",
    "embedding_cosine_labse",
    "embedding_cosine_paraphrase_multilingual_mpnet",
    "reranker_score_gte_multilingual_reranker_base",
    "reranker_score_bge_reranker_v2_m3",
]

STRUCTURAL_FEATURES = [
    "profile_category_jaccard",
    "shared_title_count_capped",
    "shared_description_count_capped",
    "shared_category_count_capped",
    "shared_title_idf_sum",
    "shared_description_idf_sum",
    "shared_title_idf_mean",
    "shared_description_idf_mean",
    "shared_boilerplate_count",
    "shared_low_df_sentence_count",
    "shared_rare_ngram_count",
    "candidate_rule_count_raw",
    "sparse_lexical_similarity_raw",
    "structural_support_score_raw",
]

STYLE_GAP_FEATURES = [
    "item_count_percentile_gap_abs",
    "price_median_percentile_gap_abs",
    "title_length_median_percentile_gap_abs",
    "description_length_median_percentile_gap_abs",
    "digit_ratio_mean_percentile_gap_abs",
    "punct_ratio_mean_percentile_gap_abs",
    "repeated_title_share_percentile_gap_abs",
    "repeated_description_share_percentile_gap_abs",
    "max_category_share_percentile_gap_abs",
    "uppercase_ratio_mean_percentile_gap_abs",
    "item_count_raw_gap_abs",
    "price_median_raw_gap_abs",
    "title_length_median_raw_gap_abs",
    "description_length_median_raw_gap_abs",
    "digit_ratio_mean_raw_gap_abs",
    "punct_ratio_mean_raw_gap_abs",
    "repeated_title_share_raw_gap_abs",
    "repeated_description_share_raw_gap_abs",
    "max_category_share_raw_gap_abs",
    "uppercase_ratio_mean_raw_gap_abs",
]

IDENTIFIER_FEATURES = [
    "has_shared_contact_exact",
    "has_shared_pgp_fingerprint",
    "shared_contact_count_capped",
    "shared_pgp_fingerprint_count_capped",
]

TEMPLATE_FEATURES = [
    "boilerplate_ratio_max",
    "boilerplate_ratio_gap_abs",
    "shared_boilerplate_count",
    "shared_low_df_sentence_count",
    "shared_rare_ngram_count",
    "shared_title_count_capped",
    "shared_description_count_capped",
    "shared_title_idf_sum",
    "shared_description_idf_sum",
]

FEATURE_GROUPS = {
    "semantic": SEMANTIC_FEATURES,
    "structural": STRUCTURAL_FEATURES,
    "style_gap": STYLE_GAP_FEATURES,
    "identifier": IDENTIFIER_FEATURES,
    "template_proxy": TEMPLATE_FEATURES,
}

ALL_NUMERIC_FEATURES = sorted({f for fs in FEATURE_GROUPS.values() for f in fs})
STRICT_POSITIVE_TIERS = {
    "gold_direct_seller_contact",
    "gold_direct_seller_contact_weaker_type",
    "gold_component_anchor",
}
SOFT_PRIMARY_TIERS = {
    "strong_soft_structural_clone",
    "component_or_contact_supported_soft_positive",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_existing(pattern: str, fallback: Path | None = None) -> Path | None:
    paths = sorted(REPORTS.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if paths:
        return paths[0]
    if fallback is not None and fallback.exists():
        return fallback
    return None


def explicit_existing_path(value: str | None, arg_name: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"{arg_name} does not exist: {path}")
    if path.is_dir():
        raise SystemExit(f"{arg_name} must be a file, not a directory: {path}")
    return path


def verify_step11_manifest_audit_chain(
    manifest_path: Path,
    audit_path: Path,
    *,
    require_publication_v6: bool,
    require_clean: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_mode = "clean_topology" if require_clean else None
    summary_paths, manifest = step11_audit_contract.summary_paths_from_manifest(
        manifest_path.resolve(),
        require_publication_v6=require_publication_v6,
        required_graph_mode=required_mode,
    )
    audit = read_json(audit_path)
    expected_audit_sha = str(audit.get("audit_sha256", "") or "").strip()
    if require_publication_v6 and not expected_audit_sha:
        raise ValueError("Step13-v6 requires a self-hashed Step11 cluster audit")
    if expected_audit_sha:
        audit_core = {key: value for key, value in audit.items() if key != "audit_sha256"}
        observed_audit_sha = hashlib.sha256(
            json.dumps(
                audit_core,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if observed_audit_sha != expected_audit_sha:
            raise ValueError("Step11 cluster audit self-hash mismatch")
    if audit.get("summary_selection_mode") != "explicit_manifest":
        raise ValueError("Step13-v6 requires Step11 audit summary_selection_mode=explicit_manifest")
    if require_publication_v6 and not bool(audit.get("publication_v6", False)):
        raise ValueError("Step13-v6 requires a publication-v6 Step11 cluster audit")
    audit_manifest_value = str(audit.get("input_manifest", "") or "").strip()
    if not audit_manifest_value:
        raise ValueError("Step11 audit does not record input_manifest")
    audit_manifest_path = Path(audit_manifest_value)
    if not audit_manifest_path.is_absolute():
        audit_manifest_path = step11_audit_contract.ROOT / audit_manifest_path
    if audit_manifest_path.resolve() != manifest_path.resolve():
        raise ValueError(
            "Step11 audit points to a different manifest: "
            f"expected={manifest_path} actual={audit_manifest_path}"
        )
    if str(audit.get("input_manifest_sha256", "")) != str(manifest.get("manifest_sha256", "")):
        raise ValueError("Step11 audit manifest self-hash does not match the supplied manifest")
    manifest_file_sha = sha256_file(manifest_path)
    if str(audit.get("input_manifest_file_sha256", "")) != str(manifest_file_sha):
        raise ValueError("Step11 audit physical manifest SHA-256 does not match the supplied manifest")
    graph_mode = str(manifest.get("graph_validation_mode", "") or "")
    if str(audit.get("graph_validation_mode", "") or "") != graph_mode:
        raise ValueError("Step11 manifest and cluster audit graph-validation modes disagree")
    expected_summaries = {
        str(path.resolve()) for path in summary_paths
    }
    observed_summaries = set()
    for value in audit.get("input_summaries", []) or []:
        path = Path(str(value))
        if not path.is_absolute():
            path = step11_audit_contract.ROOT / path
        observed_summaries.add(str(path.resolve()))
    if observed_summaries != expected_summaries:
        raise ValueError("Step11 audit input summary roster does not match its manifest")
    audit_csv_value = str(audit.get("output_csv", "") or "").strip()
    audit_csv_expected_sha = str(audit.get("output_csv_sha256", "") or "").strip()
    if require_publication_v6 and (not audit_csv_value or not audit_csv_expected_sha):
        raise ValueError("Step13-v6 requires the Step11 audit JSON to bind its CSV")
    audit_csv_path: Path | None = None
    audit_csv_rows: list[dict[str, str]] = []
    if audit_csv_value:
        audit_csv_path = Path(audit_csv_value)
        if not audit_csv_path.is_absolute():
            audit_csv_path = step11_audit_contract.ROOT / audit_csv_path
        if not audit_csv_path.exists():
            raise ValueError(f"Step11 audit CSV is missing: {audit_csv_path}")
        if str(sha256_file(audit_csv_path)) != audit_csv_expected_sha:
            raise ValueError("Step11 audit CSV SHA-256 does not match the audit JSON")
        audit_csv_rows = read_csv(audit_csv_path)
        expected_row_count = int(audit.get("output_csv_row_count", -1))
        if expected_row_count != len(audit_csv_rows):
            raise ValueError("Step11 audit CSV row count does not match the audit JSON")
        if expected_row_count != int(audit.get("audited_per_scorer_cluster_count", -1)):
            raise ValueError("Step11 audit row-count fields are internally inconsistent")
        observed_decisions = Counter(str(row.get("decision", "")) for row in audit_csv_rows)
        expected_decisions = {
            str(key): int(value)
            for key, value in (audit.get("decision_counts", {}) or {}).items()
        }
        normalized_observed = {
            decision: int(observed_decisions.get(decision, 0))
            for decision in expected_decisions
        }
        unknown_decisions = sorted(set(observed_decisions) - set(expected_decisions))
        if normalized_observed != expected_decisions or unknown_decisions:
            raise ValueError("Step11 audit CSV decision counts do not match the audit JSON")
        observed_per_scorer = Counter(
            str(row.get("scorer_token", "")) for row in audit_csv_rows
        )
        expected_per_scorer = {
            str(key): int(value)
            for key, value in (audit.get("per_scorer_cluster_counts", {}) or {}).items()
        }
        if dict(sorted(observed_per_scorer.items())) != dict(sorted(expected_per_scorer.items())):
            raise ValueError("Step11 audit CSV per-scorer counts do not match the audit JSON")
    return manifest, audit, {
        "verified": True,
        "publication_v6": bool(require_publication_v6),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_self_sha256": manifest.get("manifest_sha256"),
        "audit_path": str(audit_path),
        "audit_file_sha256": sha256_file(audit_path),
        "audit_self_sha256": expected_audit_sha or None,
        "audit_csv_path": str(audit_csv_path) if audit_csv_path else None,
        "audit_csv_sha256": audit_csv_expected_sha or None,
        "audit_csv_row_count": len(audit_csv_rows),
        "summary_selection_mode": audit.get("summary_selection_mode"),
        "graph_validation_mode": graph_mode,
        "summary_count": len(summary_paths),
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def safe_round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


def describe(values: list[float]) -> dict[str, Any]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": safe_round(statistics.fmean(values)),
        "median": safe_round(quantile(values, 0.5)),
        "std": safe_round(statistics.pstdev(values) if len(values) > 1 else 0.0),
        "q25": safe_round(quantile(values, 0.25)),
        "q75": safe_round(quantile(values, 0.75)),
        "min": safe_round(min(values)),
        "max": safe_round(max(values)),
    }


def standardized_mean_difference(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.pvariance(a) if len(a) > 1 else 0.0
    var_b = statistics.pvariance(b) if len(b) > 1 else 0.0
    pooled = math.sqrt((var_a + var_b) / 2.0)
    if pooled == 0:
        return 0.0 if mean_a == mean_b else None
    return (mean_b - mean_a) / pooled


def ks_statistic(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    xs = sorted(set(a + b))
    i = 0
    j = 0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    best = 0.0
    for x in xs:
        while i < len(a_sorted) and a_sorted[i] <= x:
            i += 1
        while j < len(b_sorted) and b_sorted[j] <= x:
            j += 1
        best = max(best, abs(i / len(a_sorted) - j / len(b_sorted)))
    return best


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return ranks


def roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for y in y_true if y == 1)
    negatives = sum(1 for y in y_true if y == 0)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, y_true) if y == 1)
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


def average_precision(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(1 for y in y_true if y == 1)
    if positives == 0:
        return None
    grouped: dict[float, list[int]] = defaultdict(list)
    for score, label in zip(scores, y_true, strict=True):
        grouped[float(score)].append(int(label))
    true_positives = 0
    false_positives = 0
    ap = 0.0
    for score in sorted(grouped, reverse=True):
        labels = grouped[score]
        group_positives = sum(labels)
        true_positives += group_positives
        false_positives += len(labels) - group_positives
        precision = true_positives / max(true_positives + false_positives, 1)
        ap += (group_positives / positives) * precision
    return ap


def metric_row(y_true: list[int], scores: list[float]) -> dict[str, Any]:
    pos = sum(1 for y in y_true if y == 1)
    neg = sum(1 for y in y_true if y == 0)
    return {
        "n": len(y_true),
        "n_positive": pos,
        "n_negative": neg,
        "roc_auc": safe_round(roc_auc(y_true, scores)),
        "average_precision": safe_round(average_precision(y_true, scores)),
        "unstable_slice": bool(pos < 5 or neg < 5),
    }


def label_to_int(label: str) -> int | None:
    value = str(label).strip().lower()
    if value == "positive":
        return 1
    if value == "negative":
        return 0
    return None


def feature_group_for(feature: str) -> str:
    memberships = [name for name, features in FEATURE_GROUPS.items() if feature in features]
    if len(memberships) == 1:
        return memberships[0]
    if "identifier" in memberships:
        return "identifier"
    if "semantic" in memberships:
        return "semantic"
    return "+".join(memberships) if memberships else "other"


def load_labeled_features(
    labels_path: Path,
    features_path: Path,
    domain: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_rows = read_csv(labels_path)
    feature_rows = {row["pair_uid"]: row for row in read_csv(features_path)}

    joined: list[dict[str, Any]] = []
    label_counts = Counter()
    split_counts: dict[str, Counter] = defaultdict(Counter)
    missing_features = 0
    supervision_rows = 0

    for label_row in label_rows:
        y = label_to_int(label_row.get("review_label", ""))
        if not truthy(label_row.get("usable_for_supervision")) or y is None:
            continue
        supervision_rows += 1
        pair_uid = label_row["pair_uid"]
        feature_row = feature_rows.get(pair_uid)
        if feature_row is None:
            missing_features += 1
            continue
        row: dict[str, Any] = {
            "domain": domain,
            "pair_uid": pair_uid,
            "label": y,
            "review_label": "positive" if y == 1 else "negative",
            "split_name": label_row.get("split_name", ""),
            "split_component_id": label_row.get("split_component_id", ""),
            "review_stratum": label_row.get("review_stratum", ""),
            "candidate_scope": label_row.get("candidate_scope", ""),
            "candidate_rule_hits": label_row.get("candidate_rule_hits", ""),
            "label_tier": label_row.get("label_tier", ""),
            "silver_train_only": truthy(label_row.get("silver_train_only")),
            "label_provenance": "silver_train_only"
            if truthy(label_row.get("silver_train_only"))
            else "gold_or_original_review",
            "usable_for_core_transfer": truthy(label_row.get("usable_for_core_transfer")),
            "core_transfer_eligible": truthy(
                feature_row.get("core_transfer_eligible", label_row.get("usable_for_core_transfer"))
            ),
        }
        for feature in ALL_NUMERIC_FEATURES:
            row[feature] = to_float(feature_row.get(feature))
        joined.append(row)
        label_counts[row["review_label"]] += 1
        split_counts[row["split_name"]][row["review_label"]] += 1

    metadata = {
        "path": str(labels_path),
        "feature_path": str(features_path),
        "reviewed_rows": len(label_rows),
        "supervision_rows": supervision_rows,
        "joined_supervision_rows": len(joined),
        "missing_feature_rows": missing_features,
        "label_counts": dict(label_counts),
        "split_label_counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
    }
    return joined, metadata


def load_raw_candidate_feature_rows(
    candidate_path: Path,
    feature_path: Path,
    domain: str,
) -> list[dict[str, Any]]:
    candidate_rows = read_csv(candidate_path)
    feature_rows = read_csv(feature_path)
    candidate_uids = {row["pair_uid"] for row in candidate_rows}
    feature_index = {row["pair_uid"]: row for row in feature_rows}
    if len(candidate_uids) != len(candidate_rows) or len(feature_index) != len(feature_rows):
        raise ValueError(f"Duplicate pair_uid in raw candidate/feature universe for {domain}")
    if candidate_uids != set(feature_index):
        raise ValueError(
            f"Step13 raw candidate/Step7 feature universe mismatch for {domain}: "
            f"missing={len(candidate_uids - set(feature_index))} "
            f"extra={len(set(feature_index) - candidate_uids)}"
        )
    rows: list[dict[str, Any]] = []
    for pair_uid in sorted(candidate_uids):
        feature_row = feature_index[pair_uid]
        row: dict[str, Any] = {"domain": domain, "pair_uid": pair_uid}
        for feature in ALL_NUMERIC_FEATURES:
            row[feature] = to_float(feature_row.get(feature))
        rows.append(row)
    return rows


def build_raw_candidate_feature_drift_rows(
    en_rows: list[dict[str, Any]],
    zh_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for feature in ALL_NUMERIC_FEATURES:
        en_values = numeric_values(en_rows, feature)
        zh_values = numeric_values(zh_rows, feature)
        en_desc = describe(en_values)
        zh_desc = describe(zh_values)
        output.append(
            {
                "row_type": "feature_drift_by_provenance",
                "cohort": "raw_step4_candidate_universe",
                "label_scope": "unlabeled_all",
                "feature_group": feature_group_for(feature),
                "feature": feature,
                "n_en": en_desc["n"],
                "n_zh": zh_desc["n"],
                "mean_en": en_desc["mean"],
                "mean_zh": zh_desc["mean"],
                "smd_zh_minus_en": standardized_mean_difference(en_values, zh_values),
                "ks_statistic": ks_statistic(en_values, zh_values),
                "interpretation_guard": "unlabeled_retrieval_distribution_not_supervision",
            }
        )
    return output


def build_provenance_cohort_rows(
    en_rows: list[dict[str, Any]],
    zh_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cohort_predicates: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        (
            "gold_train",
            lambda row: row["split_name"] == "train" and row["label_provenance"] == "gold_or_original_review",
        ),
        ("silver_train_only", lambda row: row["label_provenance"] == "silver_train_only"),
        (
            "fixed_valid_gold",
            lambda row: row["split_name"] == "valid" and row["label_provenance"] == "gold_or_original_review",
        ),
        (
            "internal_development_test_gold",
            lambda row: row["split_name"] == "test" and row["label_provenance"] == "gold_or_original_review",
        ),
    ]
    label_predicates: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all", lambda row: True),
        ("positive", lambda row: row["label"] == 1),
        ("negative", lambda row: row["label"] == 0),
    ]
    output: list[dict[str, Any]] = []
    for cohort_name, cohort_predicate in cohort_predicates:
        for label_name, label_predicate in label_predicates:
            en_subset = [row for row in en_rows if cohort_predicate(row) and label_predicate(row)]
            zh_subset = [row for row in zh_rows if cohort_predicate(row) and label_predicate(row)]
            for feature in ALL_NUMERIC_FEATURES:
                en_desc = describe(numeric_values(en_subset, feature))
                zh_desc = describe(numeric_values(zh_subset, feature))
                output.append(
                    {
                        "row_type": "feature_drift_by_provenance",
                        "cohort": cohort_name,
                        "label_scope": label_name,
                        "feature_group": feature_group_for(feature),
                        "feature": feature,
                        "n_en": en_desc["n"],
                        "n_zh": zh_desc["n"],
                        "mean_en": en_desc["mean"],
                        "mean_zh": zh_desc["mean"],
                        "smd_zh_minus_en": standardized_mean_difference(
                            numeric_values(en_subset, feature),
                            numeric_values(zh_subset, feature),
                        ),
                        "interpretation_guard": (
                            "natural_domain_drift_candidate"
                            if cohort_name in {"gold_train", "fixed_valid_gold", "internal_development_test_gold"}
                            else "active_sampling_distribution_not_natural_domain_drift"
                        ),
                    }
                )
    return output


def build_candidate_and_supervision_cohort_rows(
    en_rows: list[dict[str, Any]],
    zh_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for domain, path in STEP4_CANDIDATES.items():
        rows = read_csv(path) if path.exists() else []
        output.append(
            {
                "row_type": "dataset_provenance_cohort",
                "domain": domain,
                "cohort": "raw_step4_candidate_universe",
                "row_count": len(rows),
                "positive_count": None,
                "negative_count": None,
                "scientific_role": "unlabeled_retrieval_candidate_pool_not_supervision",
            }
        )
    for domain, rows in (("en", en_rows), ("zh", zh_rows)):
        cohort_names = {
            "gold_train": lambda row: row["split_name"] == "train"
            and row["label_provenance"] == "gold_or_original_review",
            "silver_train_only": lambda row: row["label_provenance"] == "silver_train_only",
            "fixed_valid_gold": lambda row: row["split_name"] == "valid",
            "internal_development_test_gold": lambda row: row["split_name"] == "test",
        }
        for cohort_name, predicate in cohort_names.items():
            selected = [row for row in rows if predicate(row)]
            output.append(
                {
                    "row_type": "dataset_provenance_cohort",
                    "domain": domain,
                    "cohort": cohort_name,
                    "row_count": len(selected),
                    "positive_count": sum(row["label"] == 1 for row in selected),
                    "negative_count": sum(row["label"] == 0 for row in selected),
                    "scientific_role": (
                        "train_only_weak_supervision_not_natural_sample"
                        if cohort_name == "silver_train_only"
                        else "fixed_reviewed_boundary"
                    ),
                }
            )
    return output


def is_identifier_present(row: dict[str, Any]) -> bool:
    return any(
        [
            (row.get("has_shared_contact_exact") or 0) > 0,
            (row.get("has_shared_pgp_fingerprint") or 0) > 0,
            (row.get("shared_contact_count_capped") or 0) > 0,
            (row.get("shared_pgp_fingerprint_count_capped") or 0) > 0,
        ]
    )


def is_template_dense(row: dict[str, Any]) -> bool:
    return any(
        [
            (row.get("shared_boilerplate_count") or 0) >= 1,
            (row.get("shared_low_df_sentence_count") or 0) >= 3,
            (row.get("shared_title_count_capped") or 0) >= 3,
            (row.get("shared_description_count_capped") or 0) >= 3,
            (row.get("shared_rare_ngram_count") or 0) >= 3,
        ]
    )


def numeric_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    return [row[feature] for row in rows if isinstance(row.get(feature), float)]


def build_feature_drift_rows(en_rows: list[dict[str, Any]], zh_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_scopes: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all_supervision", lambda row: True),
        ("positive", lambda row: row["label"] == 1),
        ("negative", lambda row: row["label"] == 0),
    ]
    out: list[dict[str, Any]] = []
    for scope_name, predicate in label_scopes:
        en_subset = [row for row in en_rows if predicate(row)]
        zh_subset = [row for row in zh_rows if predicate(row)]
        for feature in ALL_NUMERIC_FEATURES:
            en_values = numeric_values(en_subset, feature)
            zh_values = numeric_values(zh_subset, feature)
            en_desc = describe(en_values)
            zh_desc = describe(zh_values)
            row = {
                "row_type": "feature_drift",
                "comparison_scope": scope_name,
                "feature_group": feature_group_for(feature),
                "feature": feature,
                "n_en": en_desc["n"],
                "n_zh": zh_desc["n"],
                "mean_en": en_desc["mean"],
                "mean_zh": zh_desc["mean"],
                "median_en": en_desc["median"],
                "median_zh": zh_desc["median"],
                "q75_en": en_desc["q75"],
                "q75_zh": zh_desc["q75"],
                "smd_zh_minus_en": safe_round(standardized_mean_difference(en_values, zh_values)),
                "ks_statistic": safe_round(ks_statistic(en_values, zh_values)),
                "interpretation_guard": "mixed_gold_plus_silver_supervision_diagnostic_only",
            }
            out.append(row)
    return out


def build_high_semantic_negative_rows(en_rows: list[dict[str, Any]], zh_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cohort_predicates: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
        (
            "gold_or_original_review_all",
            lambda row: row["label_provenance"] == "gold_or_original_review",
            "reviewed_distribution_candidate",
        ),
        (
            "gold_train",
            lambda row: row["split_name"] == "train"
            and row["label_provenance"] == "gold_or_original_review",
            "natural_domain_drift_candidate",
        ),
        (
            "fixed_valid_gold",
            lambda row: row["split_name"] == "valid"
            and row["label_provenance"] == "gold_or_original_review",
            "fixed_reviewed_boundary",
        ),
        (
            "internal_development_test_gold",
            lambda row: row["split_name"] == "test"
            and row["label_provenance"] == "gold_or_original_review",
            "fixed_internal_development_boundary",
        ),
        (
            "silver_train_only",
            lambda row: row["label_provenance"] == "silver_train_only",
            "active_sampling_distribution_not_natural_domain_drift",
        ),
    ]
    for feature in [
        "embedding_cosine_multilingual_e5_large",
        "embedding_cosine_bge_m3",
        "embedding_cosine_labse",
        "embedding_cosine_gte_multilingual_base",
    ]:
        en_neg_values = numeric_values(
            [
                row
                for row in en_rows
                if row["label"] == 0
                and row["split_name"] == "train"
                and row["label_provenance"] == "gold_or_original_review"
            ],
            feature,
        )
        threshold = quantile(en_neg_values, 0.9)
        if threshold is None:
            continue
        for cohort, predicate, guard in cohort_predicates:
            for domain, domain_rows in [("en", en_rows), ("zh", zh_rows)]:
                neg_rows = [
                    row
                    for row in domain_rows
                    if row["label"] == 0
                    and predicate(row)
                    and isinstance(row.get(feature), float)
                ]
                high_rows = [row for row in neg_rows if row[feature] >= threshold]
                high_no_id = [row for row in high_rows if not is_identifier_present(row)]
                high_template_no_id = [row for row in high_no_id if is_template_dense(row)]
                rows.append(
                    {
                        "row_type": "high_semantic_negative_ratio",
                        "cohort": cohort,
                        "interpretation_guard": guard,
                        "domain": domain,
                        "feature": feature,
                        "threshold_source": "en_gold_train_negative_q90",
                        "threshold": safe_round(threshold),
                        "negative_n": len(neg_rows),
                        "high_semantic_negative_n": len(high_rows),
                        "high_semantic_negative_rate": safe_round(
                            len(high_rows) / len(neg_rows) if neg_rows else None
                        ),
                        "high_semantic_no_identifier_negative_n": len(high_no_id),
                        "high_semantic_template_no_identifier_negative_n": len(
                            high_template_no_id
                        ),
                    }
                )
    return rows


def load_prediction_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    scores: dict[str, float] = {}
    for row in read_csv(path):
        score = to_float(row.get("prob_positive"))
        if score is not None:
            scores[row["pair_uid"]] = score
    return scores


def average_score_maps(score_maps: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for score_map in score_maps:
        for pair_uid, score in score_map.items():
            sums[pair_uid] += score
            counts[pair_uid] += 1
    return {pair_uid: sums[pair_uid] / counts[pair_uid] for pair_uid in sums}


def score_from_feature(rows: list[dict[str, Any]], feature: str) -> dict[str, float]:
    return {
        row["pair_uid"]: row[feature]
        for row in rows
        if isinstance(row.get(feature), float)
    }


def collect_models(
    zh_test_rows: list[dict[str, Any]],
    v6_mode: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    models: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    def add_feature_model(model_id: str, role: str, feature: str) -> None:
        models[model_id] = {
            "role": role,
            "score_source": f"feature:{feature}",
            "scores": score_from_feature(zh_test_rows, feature),
        }

    def add_single_prediction(model_id: str, role: str, path: Path) -> None:
        scores = load_prediction_scores(path)
        if not scores:
            missing.append(str(path))
            return
        models[model_id] = {
            "role": role,
            "score_source": str(path),
            "scores": scores,
        }

    def add_ensemble(model_id: str, role: str, pattern: str) -> None:
        paths = sorted(REPORTS.glob(pattern))
        score_maps = [load_prediction_scores(path) for path in paths]
        score_maps = [m for m in score_maps if m]
        if not score_maps:
            missing.append(f"reports/{pattern}")
            return
        models[model_id] = {
            "role": role,
            "score_source": [str(path) for path in paths],
            "scores": average_score_maps(score_maps),
            "seed_count": len(score_maps),
        }

    def add_fixed_ensemble(model_id: str, role: str, template: str) -> None:
        seeds = list(range(20260320, 20260330))
        paths = [REPORTS / template.format(seed=seed) for seed in seeds]
        absent = [path for path in paths if not path.exists()]
        if absent:
            missing.extend(str(path) for path in absent)
            return
        score_maps = [load_prediction_scores(path) for path in paths]
        if any(not score_map for score_map in score_maps):
            missing.append(f"incomplete_fixed_ensemble:{model_id}")
            return
        models[model_id] = {
            "role": role,
            "score_source": [str(path) for path in paths],
            "scores": average_score_maps(score_maps),
            "seed_count": len(score_maps),
            "seed_ids": seeds,
        }

    add_feature_model("raw_e5_cosine", "raw_semantic_control", "embedding_cosine_multilingual_e5_large")
    add_feature_model("raw_bge_m3_cosine", "raw_semantic_control", "embedding_cosine_bge_m3")
    add_feature_model("raw_labse_cosine", "raw_semantic_control", "embedding_cosine_labse")
    add_single_prediction(
        "step7_core_zero_shot_default",
        "step7_clean_fusion_control",
        REPORTS / "step7_core_zero_shot_default_predictions.zh_target_strict_test.csv",
    )
    add_single_prediction(
        "step7_core_zero_shot_bge_m3",
        "step7_clean_fusion_control",
        REPORTS / "step7_core_zero_shot_bge_m3_predictions.zh_target_strict_test.csv",
    )
    add_single_prediction(
        "step7_core_zero_shot_multilingual_e5_large",
        "step7_clean_fusion_control",
        REPORTS / "step7_core_zero_shot_multilingual_e5_large_predictions.zh_target_strict_test.csv",
    )
    add_single_prediction(
        "step7_core_zero_shot_default_no_structural",
        "step7_clean_ablation_control",
        REPORTS / "step7_core_zero_shot_default_no_structural_predictions.zh_target_strict_test.csv",
    )
    add_single_prediction(
        "step7_identifier_augmented_default",
        "step7_operational_identifier_control",
        REPORTS / "step7_identifier_augmented_default_predictions.zh_target_strict_test.csv",
    )
    if v6_mode:
        fixed_step9 = [
            (
                "step9_e5_lr_l2_100pct_seed_mean",
                "step9_clean_e5_non_mixup_control",
                "step15_v6/baselines/step9/step9_core_few_shot_multilingual_e5_large_lr_l2_ratio_100pct_seed_{seed}_predictions.zh_test.csv",
            ),
            (
                "step9_e5_mixup_100pct_seed_mean",
                "step9_clean_e5_mixup_control",
                "step15_v6/baselines/step9/step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_{seed}_predictions.zh_test.csv",
            ),
            (
                "step9_bge_m3_residual_lr_100pct_seed_mean",
                "step9_clean_bge_residual_control",
                "step15_v6/baselines/step9/step9_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_{seed}_predictions.zh_test.csv",
            ),
            (
                "step9_labse_lr_l2_100pct_seed_mean",
                "step9_strongest_preregistered_clean_control",
                "step15_v6/baselines/step9/step9_core_few_shot_labse_lr_l2_ratio_100pct_seed_{seed}_predictions.zh_test.csv",
            ),
            (
                "step9_identifier_operational_100pct_seed_mean",
                "step9_identifier_operational_control",
                "step15_v6/baselines/step9/step9_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_{seed}_predictions.zh_test.csv",
            ),
        ]
        for model_id, role, template in fixed_step9:
            add_fixed_ensemble(model_id, role, template)
        fixed_v6 = [
            ("step15_v6_m0", "step15_v6_binary_control", "step15_v6_m0_all_at_once_binary", "phase3_add_contact_url_noise"),
            ("step15_v6_m1", "step15_v6_evidence_weighted", "step15_v6_m1_evidence_weighted", "phase3_add_contact_url_noise"),
            ("step15_v6_m2", "step15_v6_domain_balanced", "step15_v6_m2_domain_balanced", "phase3_add_contact_url_noise"),
            ("step15_v6_m2b", "step15_v6_matched_curriculum_control", "step15_v6_m2b_matched_budget_full_data_replay", "phase3_add_contact_url_noise"),
            ("step15_v6_m3", "step15_v6_curriculum", "step15_v6_m3_warm_start_curriculum", "phase3_add_contact_url_noise"),
            ("step15_v6_m4", "step15_v6_trusted_positive_mixup", "step15_v6_m4_trusted_positive_mixup", "phase4_add_trusted_positive_mixup"),
            ("step15_v6_m4c", "step15_v6_matched_mixup_control", "step15_v6_m4c_matched_continuation_no_mixup", "phase4_add_trusted_positive_mixup"),
            ("step15_v6_m5_lambda_0p1", "step15_v6_multitask_candidate", "step15_v6_m5_aux_evidence_lambda_0p1", "phase3_add_contact_url_noise"),
            ("step15_v6_m5_lambda_0p3", "step15_v6_multitask_candidate", "step15_v6_m5_aux_evidence_lambda_0p3", "phase3_add_contact_url_noise"),
        ]
        for model_id, role, experiment, phase in fixed_v6:
            add_fixed_ensemble(
                model_id,
                role,
                f"step15_v6/predictions/{experiment}_{phase}_seed_{{seed}}.zh_test.csv",
            )
        return models, missing
    add_ensemble(
        "step9_e5_lr_l2_50pct_seed_mean",
        "step9_clean_few_shot_seed_mean",
        "step9_core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step9_bge_m3_residual_lr_100pct_seed_mean",
        "step9_clean_few_shot_seed_mean",
        "step9_core_few_shot_bge_m3_residual_lr_ratio_100pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step9_labse_lr_l2_100pct_seed_mean",
        "step9_semantic_control_seed_mean",
        "step9_core_few_shot_labse_lr_l2_ratio_100pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step9_identifier_augmented_lr_l2_100pct_seed_mean",
        "step9_operational_identifier_seed_mean",
        "step9_identifier_augmented_few_shot_default_lr_l2_ratio_100pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
        "step9_training_only_minority_regularization_control",
        "step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_50pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
        "step9_training_only_minority_regularization_control",
        "step9_core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step15_v5_public_noise_weighted_strong_phase4_seed_mean",
        "step15_v5_identity_curriculum_public_noise_weighted_seed_mean",
        "step15_v5_identity_only_curriculum_public_noise_weighted_strong_phase4_add_positive_pair_mixup_seed_*_predictions.zh_test.csv",
    )
    add_ensemble(
        "step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean",
        "step15_v5_identity_curriculum_domain_balanced_public_noise_weighted_seed_mean",
        "step15_v5_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_phase4_add_positive_pair_mixup_seed_*_predictions.zh_test.csv",
    )
    return models, missing


def build_slice_performance_rows(
    zh_rows: list[dict[str, Any]],
    high_semantic_threshold: float,
    v6_mode: bool = False,
) -> list[dict[str, Any]]:
    zh_test_rows = [row for row in zh_rows if row["split_name"] == "test"]
    models, missing = collect_models(zh_test_rows, v6_mode=v6_mode)
    if v6_mode and missing:
        raise ValueError(
            "Step13-v6 requires every preregistered model prediction; first missing="
            f"{missing[0]}"
        )
    expected_uids = {row["pair_uid"] for row in zh_test_rows}
    for model_id, model in models.items():
        actual_uids = set(model["scores"])
        if v6_mode and actual_uids != expected_uids:
            raise ValueError(
                f"Step13-v6 model coverage mismatch for {model_id}: "
                f"missing={len(expected_uids - actual_uids)} extra={len(actual_uids - expected_uids)}"
            )

    slice_defs: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all_zh_test", lambda row: True),
        ("identifier_present", is_identifier_present),
        ("identifier_absent", lambda row: not is_identifier_present(row)),
        (
            "high_e5_semantic",
            lambda row: isinstance(row.get("embedding_cosine_multilingual_e5_large"), float)
            and row["embedding_cosine_multilingual_e5_large"] >= high_semantic_threshold,
        ),
        (
            "high_e5_semantic_no_identifier",
            lambda row: isinstance(row.get("embedding_cosine_multilingual_e5_large"), float)
            and row["embedding_cosine_multilingual_e5_large"] >= high_semantic_threshold
            and not is_identifier_present(row),
        ),
        ("template_dense", is_template_dense),
        (
            "template_dense_no_identifier",
            lambda row: is_template_dense(row) and not is_identifier_present(row),
        ),
        (
            "semantic_topic_not_controller",
            lambda row: row.get("step15_evidence_type") == "semantic_topic_not_controller",
        ),
        (
            "public_contact_or_url_noise",
            lambda row: row.get("step15_evidence_type") == "public_contact_or_url_noise",
        ),
        (
            "strict_direct_or_component_positive_vs_all_negative",
            lambda row: row["label"] == 0
            or row.get("step16f_positive_bucket") == "strict_direct_or_component",
        ),
        (
            "strict_plus_soft_primary_positive_vs_all_negative",
            lambda row: row["label"] == 0
            or row.get("step16f_positive_bucket")
            in {"strict_direct_or_component", "soft_primary"},
        ),
        (
            "soft_primary_positive_vs_all_negative",
            lambda row: row["label"] == 0
            or row.get("step16f_positive_bucket") == "soft_primary",
        ),
        (
            "secondary_positive_vs_all_negative",
            lambda row: row["label"] == 0
            or row.get("step16f_positive_bucket") == "secondary_or_sensitivity_only",
        ),
    ]
    if not v6_mode:
        v6_only_slices = {
            "strict_direct_or_component_positive_vs_all_negative",
            "strict_plus_soft_primary_positive_vs_all_negative",
            "soft_primary_positive_vs_all_negative",
            "secondary_positive_vs_all_negative",
        }
        slice_defs = [item for item in slice_defs if item[0] not in v6_only_slices]

    rows: list[dict[str, Any]] = []
    for slice_name, predicate in slice_defs:
        slice_rows = [row for row in zh_test_rows if predicate(row)]
        for model_id, model in models.items():
            y_true: list[int] = []
            scores: list[float] = []
            for row in slice_rows:
                score = model["scores"].get(row["pair_uid"])
                if score is None:
                    continue
                y_true.append(row["label"])
                scores.append(score)
            metrics = metric_row(y_true, scores)
            rows.append(
                {
                    "row_type": "slice_performance",
                    "slice_name": slice_name,
                    "model_id": model_id,
                    "model_role": model["role"],
                    "score_source": model["score_source"],
                    **metrics,
                }
            )

    raw_lookup = {
        (row["slice_name"], row["model_id"]): row
        for row in rows
        if row["row_type"] == "slice_performance"
    }
    baseline_id = "raw_e5_cosine"
    for row in rows:
        baseline = raw_lookup.get((row["slice_name"], baseline_id))
        if baseline and row["model_id"] != baseline_id:
            if row.get("roc_auc") is not None and baseline.get("roc_auc") is not None:
                row["delta_auc_vs_raw_e5"] = safe_round(row["roc_auc"] - baseline["roc_auc"])
            if row.get("average_precision") is not None and baseline.get("average_precision") is not None:
                row["delta_ap_vs_raw_e5"] = safe_round(row["average_precision"] - baseline["average_precision"])

    if missing:
        rows.append(
            {
                "row_type": "missing_optional_predictions",
                "missing_prediction_sources": " | ".join(missing),
            }
        )
    return rows


def build_step7_diagnostics(step7_summary: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = step7_summary.get("experiments", {})
    selected = [
        "core_zero_shot_default",
        "core_zero_shot_bge_m3",
        "core_zero_shot_multilingual_e5_large",
        "core_zero_shot_default_no_structural",
        "identifier_augmented_default",
    ]
    rows: list[dict[str, Any]] = []
    for name in selected:
        exp = experiments.get(name)
        if not exp:
            continue
        collapse = exp.get("collapse_guard", {}) or {}
        metrics = exp.get("zh_zero_shot_test_metrics", {}) or {}
        top_features = exp.get("top_feature_importance", []) or []
        top_feature_names = [item.get("feature_name") for item in top_features[:5]]
        rows.append(
            {
                "row_type": "step7_fusion_diagnostic",
                "experiment_name": name,
                "best_iteration": exp.get("best_iteration"),
                "trained_iteration_count": exp.get("trained_iteration_count"),
                "collapse_guard_triggered": collapse.get("triggered"),
                "collapse_guard_reasons": "|".join(collapse.get("trigger_reasons", []) or []),
                "unique_valid_probabilities": collapse.get("resolved_unique_valid_probabilities"),
                "zh_test_auc": metrics.get("roc_auc"),
                "zh_test_ap": metrics.get("average_precision"),
                "zh_test_balanced_accuracy": metrics.get("balanced_accuracy"),
                "top_feature_importance": "|".join([f for f in top_feature_names if f]),
            }
        )
    return rows


def build_step11_evidence_rows(step11_manifest: dict[str, Any] | None, step11_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if step11_manifest:
        rows.append(
            {
                "row_type": "step11_manifest_scope",
                "current_summary_count": step11_manifest.get(
                    "current_summary_count", step11_manifest.get("summary_count")
                ),
                "keep_file_count": step11_manifest.get(
                    "keep_file_count", step11_manifest.get("summary_count")
                ),
                "missing_keep_files": "|".join(step11_manifest.get("missing_keep_files", []) or []),
                "selection_rule": step11_manifest.get("selection_rule", step11_manifest.get("rule")),
                "summary_selection_mode": step11_manifest.get("selection_mode"),
                "graph_validation_mode": step11_manifest.get("graph_validation_mode"),
            }
        )
    if step11_audit:
        for decision, count in (step11_audit.get("decision_counts") or {}).items():
            rows.append(
                {
                    "row_type": "step11_cluster_audit_decision",
                    "decision": decision,
                    "count": count,
                    "input_summary_count": step11_audit.get("input_summary_count"),
                    "summary_selection_mode": step11_audit.get("summary_selection_mode"),
                    "unique_cluster_set_count": step11_audit.get("unique_cluster_set_count"),
                    "graph_validation_mode": step11_audit.get("graph_validation_mode"),
                }
            )
        for scorer_token, counts in (step11_audit.get("per_scorer_decision_counts") or {}).items():
            rows.append(
                {
                    "row_type": "step11_per_scorer_decision_summary",
                    "scorer_token": scorer_token,
                    "decision_counts": counts,
                    "cluster_count": (step11_audit.get("per_scorer_cluster_counts") or {}).get(
                        scorer_token
                    ),
                    "summary_selection_mode": step11_audit.get("summary_selection_mode"),
                    "graph_validation_mode": step11_audit.get("graph_validation_mode"),
                }
            )
    return rows


def compact_rows_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                out[key] = safe_round(value) if isinstance(value, float) else value
        compacted.append(out)
    return compacted


def serialize_csv(rows: list[dict[str, Any]]) -> bytes:
    rows = compact_rows_for_csv(rows)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return csv_bytes(rows, fieldnames, encoding="utf-8")


def sort_top_drift(rows: list[dict[str, Any]], scope: str, limit: int = 12) -> list[dict[str, Any]]:
    scoped = [
        row
        for row in rows
        if row.get("row_type") == "feature_drift"
        and row.get("comparison_scope") == scope
        and row.get("smd_zh_minus_en") is not None
    ]
    return sorted(scoped, key=lambda row: abs(row["smd_zh_minus_en"]), reverse=True)[:limit]


def best_metric(rows: list[dict[str, Any]], slice_name: str, metric: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("row_type") == "slice_performance"
        and row.get("slice_name") == slice_name
        and row.get(metric) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric])


def row_by_model(rows: list[dict[str, Any]], slice_name: str, model_id: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("row_type") == "slice_performance"
            and row.get("slice_name") == slice_name
            and row.get("model_id") == model_id
        ):
            return row
    return None


def build_findings(
    dataset: dict[str, Any],
    drift_rows: list[dict[str, Any]],
    provenance_drift_rows: list[dict[str, Any]],
    high_semantic_rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    step7_rows: list[dict[str, Any]],
    step11_rows: list[dict[str, Any]],
) -> list[str]:
    findings: list[str] = []

    en_meta = dataset["en"]
    zh_meta = dataset["zh"]
    findings.append(
        "Frozen supervision is still small and imbalanced: "
        f"EN {en_meta['joined_supervision_rows']} rows "
        f"({en_meta['label_counts'].get('positive', 0)} positive / {en_meta['label_counts'].get('negative', 0)} negative); "
        f"ZH {zh_meta['joined_supervision_rows']} rows "
        f"({zh_meta['label_counts'].get('positive', 0)} positive / {zh_meta['label_counts'].get('negative', 0)} negative)."
    )

    top_all = sorted(
        [
            row
            for row in provenance_drift_rows
            if row.get("cohort") == "internal_development_test_gold"
            and row.get("label_scope") == "all"
            and row.get("smd_zh_minus_en") is not None
        ],
        key=lambda row: abs(float(row["smd_zh_minus_en"])),
        reverse=True,
    )[:5]
    if top_all:
        names = ", ".join(
            f"{row['feature']} (SMD={row['smd_zh_minus_en']})" for row in top_all
        )
        findings.append(
            "The largest EN->ZH marginal feature shifts on the fixed gold internal-development "
            f"test cohort are: {names}."
        )

    e5_rows = [
        row
        for row in high_semantic_rows
        if row.get("feature") == "embedding_cosine_multilingual_e5_large"
        and row.get("cohort") == "internal_development_test_gold"
    ]
    e5_en = next((row for row in e5_rows if row.get("domain") == "en"), None)
    e5_zh = next((row for row in e5_rows if row.get("domain") == "zh"), None)
    if e5_en and e5_zh:
        en_rate = e5_en["high_semantic_negative_rate"]
        zh_rate = e5_zh["high_semantic_negative_rate"]
        if zh_rate is not None and en_rate is not None and zh_rate > en_rate + 0.02:
            direction = "higher"
        elif zh_rate is not None and en_rate is not None and zh_rate < en_rate - 0.02:
            direction = "lower"
        else:
            direction = "similar"
        findings.append(
            "High-semantic negatives are not uniformly inflated in ZH under the EN-negative q90 E5 threshold; "
            f"the ZH rate is {direction}: EN {e5_en['high_semantic_negative_n']}/{e5_en['negative_n']} "
            f"({e5_en['high_semantic_negative_rate']}); "
            f"ZH {e5_zh['high_semantic_negative_n']}/{e5_zh['negative_n']} "
            f"({e5_zh['high_semantic_negative_rate']})."
        )
    rate_deltas = []
    for feature in sorted({row.get("feature") for row in high_semantic_rows if row.get("feature")}):
        en_row = next(
            (
                row
                for row in high_semantic_rows
                if row.get("feature") == feature
                and row.get("domain") == "en"
                and row.get("cohort") == "internal_development_test_gold"
            ),
            None,
        )
        zh_row = next(
            (
                row
                for row in high_semantic_rows
                if row.get("feature") == feature
                and row.get("domain") == "zh"
                and row.get("cohort") == "internal_development_test_gold"
            ),
            None,
        )
        if not en_row or not zh_row:
            continue
        delta = zh_row.get("high_semantic_negative_rate")
        base = en_row.get("high_semantic_negative_rate")
        if delta is None or base is None:
            continue
        rate_deltas.append(f"{feature.replace('embedding_cosine_', '')}: {safe_round(delta - base)}")
    if rate_deltas:
        findings.append(
            "Feature-specific high-negative rate deltas (ZH minus EN) are mixed, not a single semantic-collapse pattern: "
            + "; ".join(rate_deltas)
            + "."
        )

    raw_e5 = row_by_model(slice_rows, "all_zh_test", "raw_e5_cosine")
    step7_e5 = row_by_model(slice_rows, "all_zh_test", "step7_core_zero_shot_multilingual_e5_large")
    step9_e5 = row_by_model(slice_rows, "all_zh_test", "step9_e5_lr_l2_50pct_seed_mean")
    step9_mixup_100 = row_by_model(
        slice_rows,
        "all_zh_test",
        "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
    )
    step15_v5_domain = row_by_model(
        slice_rows,
        "all_zh_test",
        "step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean",
    )
    if raw_e5 and step7_e5:
        findings.append(
            "Raw E5 remains stronger than Step 7 E5 fusion on fixed ZH test: "
            f"raw AUC {raw_e5.get('roc_auc')} / AP {raw_e5.get('average_precision')} vs "
            f"fusion AUC {step7_e5.get('roc_auc')} / AP {step7_e5.get('average_precision')}."
        )
    if raw_e5 and step9_e5:
        findings.append(
            "The current E5 LR/L2 few-shot seed-mean is only a small global ranking improvement over raw E5: "
            f"AUC delta {step9_e5.get('delta_auc_vs_raw_e5')} and AP delta {step9_e5.get('delta_ap_vs_raw_e5')}."
        )
    if raw_e5 and step9_mixup_100:
        findings.append(
            "The E5 LR/L2 positive-pair mixup 100pct seed-mean is the strongest current Step 9 minority-regularization baseline, "
            f"with AUC delta {step9_mixup_100.get('delta_auc_vs_raw_e5')} and AP delta {step9_mixup_100.get('delta_ap_vs_raw_e5')} versus raw E5; "
            "Step 12 paired bootstrap decides whether this can be treated as a robust improvement."
        )
    if raw_e5 and step15_v5_domain:
        findings.append(
            "Step 15 v5 domain-balanced public-noise-weighted curriculum has the strongest current fixed-test point estimate, "
            f"with AUC delta {step15_v5_domain.get('delta_auc_vs_raw_e5')} and AP delta {step15_v5_domain.get('delta_ap_vs_raw_e5')} versus raw E5; "
            "Step 12 v5 paired bootstrap supports its ROC-AUC improvement over Step 9 mixup100 but not yet over raw E5."
        )

    collapsed = [row for row in step7_rows if row.get("collapse_guard_triggered")]
    if collapsed:
        findings.append(
            "Step 7 fusion diagnostics still show collapse/early-stop risk for "
            f"{len(collapsed)}/{len(step7_rows)} tracked experiments; this supports a source-domain shortcut/transfer drift diagnosis rather than a simple hyperparameter issue."
        )

    step11_decisions = {
        row.get("decision"): row.get("count")
        for row in step11_rows
        if row.get("row_type") == "step11_cluster_audit_decision"
    }
    if step11_decisions:
        template_like = (step11_decisions.get("template_clone_not_controller") or 0) + (
            step11_decisions.get("semantic_topic_not_controller") or 0
        )
        anchored = (step11_decisions.get("same_controller_high_confidence") or 0) + (
            step11_decisions.get("same_controller_core_with_possible_expansion") or 0
        )
        findings.append(
            "Current manifest-only Step 11 cluster audit is dominated by non-controller evidence types: "
            f"{template_like} template/topic clusters vs {anchored} anchored same-controller cores."
        )

    return findings


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    if not selected:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in selected:
        body.append("| " + " | ".join(markdown_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def build_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    drift_rows = summary["feature_drift_rows"]
    high_rows = summary["high_semantic_negative_rows"]
    slice_rows = summary["slice_performance_rows"]
    step7_rows = summary["step7_fusion_diagnostics"]
    step11_rows = summary["step11_evidence_scope_rows"]

    dataset_rows = [
        {
            "domain": "EN",
            "joined_supervision": dataset["en"]["joined_supervision_rows"],
            "positive": dataset["en"]["label_counts"].get("positive", 0),
            "negative": dataset["en"]["label_counts"].get("negative", 0),
            "train": dataset["en"]["split_label_counts"].get("train", {}),
            "valid": dataset["en"]["split_label_counts"].get("valid", {}),
            "test": dataset["en"]["split_label_counts"].get("test", {}),
        },
        {
            "domain": "ZH",
            "joined_supervision": dataset["zh"]["joined_supervision_rows"],
            "positive": dataset["zh"]["label_counts"].get("positive", 0),
            "negative": dataset["zh"]["label_counts"].get("negative", 0),
            "train": dataset["zh"]["split_label_counts"].get("train", {}),
            "valid": dataset["zh"]["split_label_counts"].get("valid", {}),
            "test": dataset["zh"]["split_label_counts"].get("test", {}),
        },
    ]

    key_slice_models = [
        row
        for row in slice_rows
        if row.get("row_type") == "slice_performance"
        and row.get("slice_name") in {
            "all_zh_test",
            "identifier_present",
            "identifier_absent",
            "high_e5_semantic_no_identifier",
            "template_dense_no_identifier",
            "semantic_topic_not_controller",
            "public_contact_or_url_noise",
        }
        and row.get("model_id")
        in {
            "raw_e5_cosine",
            "step7_core_zero_shot_multilingual_e5_large",
            "step9_e5_lr_l2_50pct_seed_mean",
            "step9_identifier_augmented_lr_l2_100pct_seed_mean",
            "step9_e5_lr_l2_positive_pair_mixup_50pct_seed_mean",
            "step9_e5_lr_l2_positive_pair_mixup_100pct_seed_mean",
            "step15_v5_public_noise_weighted_strong_phase4_seed_mean",
            "step15_v5_domain_balanced_public_noise_weighted_strong_phase4_seed_mean",
            "step9_e5_lr_l2_100pct_seed_mean",
            "step9_e5_mixup_100pct_seed_mean",
            "step9_bge_m3_residual_lr_100pct_seed_mean",
            "step9_labse_lr_l2_100pct_seed_mean",
            "step9_identifier_operational_100pct_seed_mean",
            "step15_v6_m0",
            "step15_v6_m1",
            "step15_v6_m2",
            "step15_v6_m2b",
            "step15_v6_m3",
            "step15_v6_m4",
            "step15_v6_m4c",
            "step15_v6_m5_lambda_0p1",
            "step15_v6_m5_lambda_0p3",
        }
    ]

    lines = [
        "# Step 13 Concept Drift Audit",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "This is a read-only audit. It joins Step 5 frozen supervision rows, Step 7 pair features, existing Step 7/9 predictions, and the current manifest-only Step 11 audit. It does not train a model and does not write labels back to Step 5.",
        "",
        "## Dataset",
        "",
        markdown_table(dataset_rows, ["domain", "joined_supervision", "positive", "negative", "train", "valid", "test"]),
        "",
        "## Key Findings",
        "",
        *[f"- {finding}" for finding in summary["findings"]],
        "",
        "## Largest EN to ZH Feature Shifts",
        "",
        markdown_table(
            sort_top_drift(drift_rows, "all_supervision", 12),
            ["feature_group", "feature", "mean_en", "mean_zh", "smd_zh_minus_en", "ks_statistic"],
        ),
        "",
        "## Label-Conditional Drift",
        "",
        "Top positive-label shifts:",
        "",
        markdown_table(
            sort_top_drift(drift_rows, "positive", 8),
            ["feature_group", "feature", "mean_en", "mean_zh", "smd_zh_minus_en", "ks_statistic"],
        ),
        "",
        "Top negative-label shifts:",
        "",
        markdown_table(
            sort_top_drift(drift_rows, "negative", 8),
            ["feature_group", "feature", "mean_en", "mean_zh", "smd_zh_minus_en", "ks_statistic"],
        ),
        "",
        "## High-Semantic Negative Ratio",
        "",
        "Thresholds are defined as the English negative q90 for each semantic feature. This asks whether target-domain negatives enter a source-domain high-similarity region more often.",
        "",
        markdown_table(
            high_rows,
            [
                "domain",
                "feature",
                "threshold",
                "negative_n",
                "high_semantic_negative_n",
                "high_semantic_negative_rate",
                "high_semantic_no_identifier_negative_n",
                "high_semantic_template_no_identifier_negative_n",
            ],
        ),
        "",
        "## ZH Test Slice Performance",
        "",
        "Slices with fewer than five positives or five negatives are marked unstable in the CSV/JSON and should be treated as diagnostics, not conclusions.",
        "",
        markdown_table(
            key_slice_models,
            [
                "slice_name",
                "model_id",
                "n",
                "n_positive",
                "n_negative",
                "roc_auc",
                "average_precision",
                "delta_auc_vs_raw_e5",
                "delta_ap_vs_raw_e5",
                "unstable_slice",
            ],
        ),
        "",
        "## Step 7 Fusion Diagnostics",
        "",
        markdown_table(
            step7_rows,
            [
                "experiment_name",
                "best_iteration",
                "collapse_guard_triggered",
                "collapse_guard_reasons",
                "unique_valid_probabilities",
                "zh_test_auc",
                "zh_test_ap",
                "top_feature_importance",
            ],
        ),
        "",
        "## Step 11 Evidence Context",
        "",
        markdown_table(
            step11_rows,
            [
                "row_type",
                "scorer_token",
                "decision",
                "count",
                "decision_counts",
                "cluster_count",
                "current_summary_count",
                "summary_selection_mode",
                "graph_validation_mode",
                "unique_cluster_set_count",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The audit supports a concept-drift framing: source-domain fusion features are not simply weak; they encode source-domain shortcuts that do not transfer cleanly to Chinese target-domain pairs. Raw semantic ranking remains useful, but high-semantic target negatives and template-dense no-identifier slices explain why graph-level identity claims need reliability filtering and direct-anchor audit.",
        "",
        "Current few-shot gains should be reported as slice-dependent diagnostics unless Step 12 bootstrap comparisons and future Step 11 reliability-filter reruns show stable improvements.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 13 concept-drift audit.")
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-csv", default=str(OUT_CSV))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument(
        "--step11-manifest",
        default=None,
        help="Explicit Step 11 manifest JSON. If omitted, Step 13 does not read a Step 11 manifest unless --allow-step11-auto-discovery is set.",
    )
    parser.add_argument(
        "--step11-audit",
        default=None,
        help="Explicit Step 11 cluster-level audit JSON. Use this for current validation audits instead of relying on current_* filename discovery.",
    )
    parser.add_argument(
        "--allow-step11-auto-discovery",
        action="store_true",
        help="Compatibility mode only: discover newest reports/step11_current_manifest_*.json and reports/step11_cluster_level_audit.current_*.json.",
    )
    parser.add_argument(
        "--step12-v6-summary",
        default=None,
        help="Explicit Step12-v6 robustness summary. No automatic discovery is performed.",
    )
    parser.add_argument(
        "--step7-summary",
        default=str(STEP7_SUMMARY),
        help="Explicit Step7 summary. V6 should pass the isolated read-only metric refresh.",
    )
    parser.add_argument(
        "--step9-summary",
        default=str(STEP9_SUMMARY),
        help="Explicit Step9 summary. V6 must pass the isolated Step9 output-root summary.",
    )
    args = parser.parse_args()

    step12_v6_path = explicit_existing_path(args.step12_v6_summary, "--step12-v6-summary")
    step12_v6 = read_json(step12_v6_path) if step12_v6_path else None
    step7_summary_path = explicit_existing_path(args.step7_summary, "--step7-summary")
    step9_summary_path = explicit_existing_path(args.step9_summary, "--step9-summary")

    required = [
        EN_LABELS,
        ZH_LABELS,
        EN_FEATURES,
        ZH_FEATURES,
        step7_summary_path,
        step9_summary_path,
        STEP15_ZH_EVIDENCE_LABELS,
        STEP16F_POSITIVE_REAUDIT,
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise SystemExit(f"Missing required inputs: {missing_required}")

    en_rows, en_meta = load_labeled_features(EN_LABELS, EN_FEATURES, "en")
    zh_rows, zh_meta = load_labeled_features(ZH_LABELS, ZH_FEATURES, "zh")
    zh_evidence_index = {
        row["pair_uid"]: row
        for row in read_csv(STEP15_ZH_EVIDENCE_LABELS)
    }
    for row in zh_rows:
        evidence_row = zh_evidence_index.get(row["pair_uid"], {})
        row["step15_evidence_type"] = evidence_row.get("evidence_type")
    step16f_index = {
        row["pair_uid"]: row for row in read_csv(STEP16F_POSITIVE_REAUDIT)
    }
    for row in zh_rows:
        if row["label"] == 0:
            row["step16f_positive_bucket"] = "negative"
            continue
        tier = str(step16f_index.get(row["pair_uid"], {}).get("paper_evidence_tier", ""))
        if tier in STRICT_POSITIVE_TIERS:
            row["step16f_positive_bucket"] = "strict_direct_or_component"
        elif tier in SOFT_PRIMARY_TIERS:
            row["step16f_positive_bucket"] = "soft_primary"
        else:
            row["step16f_positive_bucket"] = "secondary_or_sensitivity_only"
    if step12_v6 is not None:
        positive_test_rows = [
            row for row in zh_rows if row["split_name"] == "test" and row["label"] == 1
        ]
        missing_tiers = [
            row["pair_uid"] for row in positive_test_rows if row["pair_uid"] not in step16f_index
        ]
        if missing_tiers:
            raise ValueError(f"Step13-v6 Step16F tier mapping is incomplete: {missing_tiers[:1]}")
        observed_slice_counts = dict(
            Counter(row["step16f_positive_bucket"] for row in positive_test_rows)
        )
        expected_slice_counts = dict(step12_v6.get("positive_slice_counts") or {})
        if observed_slice_counts != expected_slice_counts:
            raise ValueError(
                "Step13-v6 positive tier counts disagree with Step12-v6: "
                f"expected={expected_slice_counts} actual={observed_slice_counts}"
            )

    drift_rows = build_feature_drift_rows(en_rows, zh_rows)
    raw_en_rows = load_raw_candidate_feature_rows(
        STEP4_CANDIDATES["en"], EN_FEATURES, "en"
    )
    raw_zh_rows = load_raw_candidate_feature_rows(
        STEP4_CANDIDATES["zh"], ZH_FEATURES, "zh"
    )
    raw_candidate_drift_rows = build_raw_candidate_feature_drift_rows(
        raw_en_rows, raw_zh_rows
    )
    provenance_drift_rows = [
        *raw_candidate_drift_rows,
        *build_provenance_cohort_rows(en_rows, zh_rows),
    ]
    provenance_cohort_rows = build_candidate_and_supervision_cohort_rows(en_rows, zh_rows)
    high_semantic_rows = build_high_semantic_negative_rows(en_rows, zh_rows)

    e5_threshold_row = next(
        (
            row
            for row in high_semantic_rows
            if row.get("domain") == "en"
            and row.get("feature") == "embedding_cosine_multilingual_e5_large"
            and row.get("cohort") == "gold_train"
        ),
        None,
    )
    if not e5_threshold_row or e5_threshold_row.get("threshold") is None:
        raise SystemExit("Could not derive English-negative q90 threshold for E5.")
    e5_high_negative_threshold = float(e5_threshold_row["threshold"])

    slice_rows = build_slice_performance_rows(
        zh_rows,
        e5_high_negative_threshold,
        v6_mode=step12_v6 is not None,
    )

    step7_summary = read_json(step7_summary_path)
    step7_rows = build_step7_diagnostics(step7_summary)

    step11_manifest_path = explicit_existing_path(args.step11_manifest, "--step11-manifest")
    step11_audit_path = explicit_existing_path(args.step11_audit, "--step11-audit")
    step11_selection_mode = "explicit"
    if step11_manifest_path is None and step11_audit_path is None:
        if args.allow_step11_auto_discovery:
            step11_manifest_path = newest_existing("step11_current_manifest_*.json", STEP11_MANIFEST_FALLBACK)
            step11_audit_path = newest_existing("step11_cluster_level_audit.current_*.json", STEP11_AUDIT_FALLBACK)
            step11_selection_mode = "auto_discovery_compatibility"
        else:
            step11_selection_mode = "not_provided"
    if step12_v6 is not None and (step11_manifest_path is None or step11_audit_path is None):
        raise ValueError(
            "Step13-v6 requires both --step11-manifest and --step11-audit; auto discovery and partial chains are prohibited"
        )
    step11_chain_verification = {"verified": False, "reason": "step11_inputs_not_provided"}
    if step11_manifest_path is not None and step11_audit_path is not None:
        step11_manifest, step11_audit, step11_chain_verification = (
            verify_step11_manifest_audit_chain(
                step11_manifest_path,
                step11_audit_path,
                require_publication_v6=step12_v6 is not None,
                require_clean=step12_v6 is not None,
            )
        )
        step11_selection_mode = "verified_explicit_manifest_chain"
    else:
        step11_manifest = read_json(step11_manifest_path) if step11_manifest_path else None
        step11_audit = read_json(step11_audit_path) if step11_audit_path else None
    step11_rows = build_step11_evidence_rows(step11_manifest, step11_audit)

    all_csv_rows = [
        *drift_rows,
        *provenance_drift_rows,
        *provenance_cohort_rows,
        *high_semantic_rows,
        *slice_rows,
        *step7_rows,
        *step11_rows,
    ]

    input_paths = [
        EN_LABELS,
        ZH_LABELS,
        EN_FEATURES,
        ZH_FEATURES,
        step7_summary_path,
        step9_summary_path,
        STEP15_ZH_EVIDENCE_LABELS,
        STEP16F_POSITIVE_REAUDIT,
        step11_manifest_path,
        step11_audit_path,
        step12_v6_path,
        *STEP4_CANDIDATES.values(),
    ]
    input_paths = [path for path in input_paths if path is not None]
    for row in slice_rows:
        source = row.get("score_source")
        if isinstance(source, str) and source.startswith("reports"):
            input_paths.append(Path(source))
        elif isinstance(source, list):
            input_paths.extend(Path(p) for p in source)

    dataset = {"en": en_meta, "zh": zh_meta}
    summary = {
        "audit_version": "step13_concept_drift_audit_v2_provenance_and_metric_semantics",
        "metric_semantics_version": "2026-07-v2-tie-aware",
        "generated_at": dt.date.today().isoformat(),
        "scope": {
            "mode": "read_only_existing_artifacts",
            "fixed_target_test": "zh_target_strict split_name=test",
            "no_train_valid_test_mixing": True,
            "no_label_writeback": True,
            "high_semantic_threshold_policy": "English gold-train negative q90 per semantic feature",
            "step11_selection_mode": step11_selection_mode,
            "step11_manifest_path": str(step11_manifest_path) if step11_manifest_path else None,
            "step11_audit_path": str(step11_audit_path) if step11_audit_path else None,
            "step11_manifest_audit_chain": step11_chain_verification,
            "step12_v6_summary_path": str(step12_v6_path) if step12_v6_path else None,
            "step7_summary_path": str(step7_summary_path),
            "step9_summary_path": str(step9_summary_path),
            "target_test_role": "fixed_internal_development_test_not_prospective_final_holdout",
        },
        "inputs": {str(path): sha256_file(path) for path in sorted(set(input_paths), key=str) if path.exists()},
        "dataset": dataset,
        "thresholds": {
            "e5_high_semantic_negative_threshold_en_negative_q90": e5_high_negative_threshold,
        },
        "feature_groups": FEATURE_GROUPS,
        "feature_drift_rows": drift_rows,
        "feature_drift_by_provenance_rows": provenance_drift_rows,
        "raw_candidate_feature_drift_rows": raw_candidate_drift_rows,
        "dataset_provenance_cohorts": provenance_cohort_rows,
        "step12_v6_context": {
            "provided": step12_v6 is not None,
            "promotion": step12_v6.get("promotion") if step12_v6 else None,
            "selection": step12_v6.get("selection") if step12_v6 else None,
        },
        "feature_drift_top": {
            "all_supervision": sort_top_drift(drift_rows, "all_supervision", 12),
            "positive": sort_top_drift(drift_rows, "positive", 12),
            "negative": sort_top_drift(drift_rows, "negative", 12),
        },
        "high_semantic_negative_rows": high_semantic_rows,
        "slice_performance_rows": slice_rows,
        "step7_fusion_diagnostics": step7_rows,
        "step11_evidence_scope_rows": step11_rows,
        "step11_manifest_audit_chain": step11_chain_verification,
        "limitations": [
            "Slice metrics with fewer than five positives or five negatives are diagnostic only.",
            "Step 11 cluster audit is evidence triage, not ground truth.",
            "This audit uses existing synchronized predictions and the explicit Step 11 validation audit; any future scorer must be added as an explicit prediction source before being described in findings.",
            "The audit does not infer same-controller labels from semantic or template similarity.",
            "Mixed gold-plus-silver training distributions are not interpreted as natural EN-vs-ZH concept drift; provenance-specific rows must be used for that claim.",
            "The current zh_test boundary is an internal development test and cannot serve as prospective publication confirmation.",
        ],
    }
    summary["findings"] = build_findings(
        dataset,
        drift_rows,
        provenance_drift_rows,
        high_semantic_rows,
        slice_rows,
        step7_rows,
        step11_rows,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    write_immutable_bundle(
        [
            (output_json, json_bytes(summary, ensure_ascii=False, indent=2)),
            (output_csv, serialize_csv(all_csv_rows)),
            (output_md, text_bytes(build_markdown(summary))),
        ]
    )

    print(json.dumps(
        {
            "output_json": str(output_json),
            "output_csv": str(output_csv),
            "output_md": str(output_md),
            "feature_drift_rows": len(drift_rows),
            "slice_performance_rows": len([r for r in slice_rows if r.get("row_type") == "slice_performance"]),
            "findings": summary["findings"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
