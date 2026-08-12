#!/usr/bin/env python3
"""Frozen design-scale quality gates for the Step28-v13 v1.13 builder."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

import step28_v13_common as common
import step28_v13_history_features as history_features
import step28_v13_production_chain as production
import step28_v13_v1_13_counterfactual_text as counterfactual_text
import step28_v13_v1_13_blind_literal_scan as blind_literal_scan
import step28_v13_v1_13_candidate_parent as candidate_parent
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common as scientific
import step28_v13_v1_13_scientific_dataset_builder as dataset_builder
import step28_v13_v1_13_scientific_world as world_module
import step28_v13_v1_13_style_derangement as style_derangement


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_scientific_quality_audit_policy.json"
)
DEFAULT_LAUNCH_ANCHOR_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_scientific_quality_audit_launch_anchor.json"
)
DEFAULT_RELEASE_MANIFEST_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_scientific_quality_audit_release_manifest.json"
)
DEFAULT_EXTERNAL_REVIEW_ATTESTATION_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_scientific_quality_audit_external_review_attestation.json"
)
DEFAULT_LAUNCH_GUARD_PATH = (
    ROOT / "scripts" / "run_step28_v13_v1_13_scientific_quality_audit_guarded.py"
)
VERSION = "2026-08-12-step28-v13-v1-13-scientific-quality-audit-v7"
STATUS = "DESIGN_PREFLIGHT_AUDIT_ONLY"
VISIBLE_PROFILE_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
SURFACE_METRICS = (
    "codepoint_length_absdiff",
    "codepoint_length_sum",
    "newline_count_absdiff",
    "newline_count_sum",
    "unicode_punctuation_count_absdiff",
    "unicode_punctuation_count_sum",
    "ascii_whitespace_count_absdiff",
    "ascii_whitespace_count_sum",
    "unicode_decimal_digit_count_absdiff",
    "unicode_decimal_digit_count_sum",
    "empty_both",
    "empty_xor",
)
FIXED_SUPPORT_SURFACE_METRICS = (
    "codepoint_length_absdiff",
    "codepoint_length_sum",
    "newline_count_absdiff",
    "newline_count_sum",
    "unicode_punctuation_count_absdiff",
    "unicode_punctuation_count_sum",
    "ascii_whitespace_count_absdiff",
    "ascii_whitespace_count_sum",
    "unicode_decimal_digit_count_absdiff",
    "unicode_decimal_digit_count_sum",
    "empty_rate_absdiff",
    "empty_rate_sum",
)
PUNCTUATION_CATEGORIES = frozenset({"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"})
ASCII_WHITESPACE = frozenset("\t\n\v\f\r ")
COMBINED_SEPARATOR = "\n␞\n"
FIXED_SUPPORT_FIELDS = ("title", "description")
PRODUCTION_NUMERIC_FIELDS = (
    "item_count",
    "title_length_median",
    "description_length_median",
    "digit_ratio_mean",
    "punct_ratio_mean",
    "repeated_title_share",
    "repeated_description_share",
    "max_category_share",
)
BLIND_DESCRIPTION_REPEAT_SNIPPET_LIMIT = 280
BLIND_COUNTER_KEYS = (
    "private_payload_open_requests",
    "truth_read_requests",
    "world_reconstruction_requests",
    "build_private_truth_calls",
    "controller_relation_reads",
    "qrels_generations",
    "sealed_literal_scan_calls",
    "sealed_registry_isolation_calls",
)
EXPECTED_FINAL_BLIND_COUNTERS = {
    key: int(key in {"sealed_literal_scan_calls", "sealed_registry_isolation_calls"})
    for key in BLIND_COUNTER_KEYS
}
VISIBLE_IDENTITY_PATTERNS = (
    ("email", re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?!\w)")),
    ("telegram_handle", re.compile(r"(?i)(?<!\w)@[a-z][a-z0-9_]{4,}(?!\w)")),
    (
        "external_url",
        re.compile(
            r"(?i)(?<![\w@])(?:(?:https?://|www\.)[^\s]+|"
            r"(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?)"
        ),
    ),
    ("phone_like", re.compile(r"(?<!\w)\+?\d[\d\s()\-]{6,}\d(?!\w)")),
    ("hex_wallet", re.compile(r"(?i)(?<![0-9a-f])0x[0-9a-f]{16,}(?![0-9a-f])")),
    ("base58_wallet", re.compile(r"(?<![A-Za-z0-9])[13bc][A-Za-z0-9]{25,}(?![A-Za-z0-9])")),
    ("identity_keyword", re.compile(r"(?i)(?:telegram|wechat|\bqq\b|微信|电报|蝙蝠)")),
)
class ScientificQualityAuditError(common.ContractError):
    """Fail-closed quality-audit error."""


class AuditLaunchPreflightError(ScientificQualityAuditError):
    """Launch failed before any dataset row could be opened."""


class DatasetInvalidationError(ScientificQualityAuditError):
    """A frozen row or statistical quality gate invalidated the design data."""


def _failure_classification(error: Exception) -> str:
    return (
        "DATASET_INVALIDATED"
        if isinstance(error, DatasetInvalidationError)
        else "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
    )


def _verify_launch_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    caller_path: Path,
) -> dict[str, Any]:
    if caller_path.resolve() != DEFAULT_LAUNCH_GUARD_PATH.resolve():
        raise ScientificQualityAuditError(
            "Direct quality execution is forbidden; use the reviewed launch guard"
        )
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "release_manifest",
        "anchor",
        "guard",
        "external_review_attestation",
        "entry",
    }:
        raise ScientificQualityAuditError(
            "Direct quality execution is forbidden; use the reviewed launch guard"
        )
    release_evidence = evidence["release_manifest"]
    anchor_evidence = evidence["anchor"]
    guard_evidence = evidence["guard"]
    attestation_evidence = evidence["external_review_attestation"]
    entry_evidence = evidence["entry"]
    if (
        not isinstance(release_evidence, Mapping)
        or set(release_evidence)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not isinstance(anchor_evidence, Mapping)
        or set(anchor_evidence)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not isinstance(guard_evidence, Mapping)
        or set(guard_evidence) != {"path", "size_bytes", "sha256"}
        or not isinstance(attestation_evidence, Mapping)
        or set(attestation_evidence)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not isinstance(entry_evidence, Mapping)
        or set(entry_evidence) != {"main", "argv0", "quality_module"}
        or any(
            not isinstance(entry_evidence[name], Mapping)
            or set(entry_evidence[name]) != {"path", "size_bytes", "sha256"}
            for name in ("main", "argv0", "quality_module")
        )
    ):
        raise ScientificQualityAuditError("Launch evidence schema drift")
    release_path = common.repo_path(str(release_evidence["path"]))
    anchor_path = common.repo_path(str(anchor_evidence["path"]))
    guard_path = common.repo_path(str(guard_evidence["path"]))
    attestation_path = common.repo_path(str(attestation_evidence["path"]))
    if (
        release_path.resolve() != DEFAULT_RELEASE_MANIFEST_PATH.resolve()
        or anchor_path.resolve() != DEFAULT_LAUNCH_ANCHOR_PATH.resolve()
        or guard_path.resolve() != DEFAULT_LAUNCH_GUARD_PATH.resolve()
        or attestation_path.resolve()
        != DEFAULT_EXTERNAL_REVIEW_ATTESTATION_PATH.resolve()
    ):
        raise ScientificQualityAuditError("Launch path drift")
    if (
        not release_path.is_file()
        or release_path.stat().st_size != release_evidence["size_bytes"]
        or common.sha256_file(release_path) != release_evidence["sha256"]
        or not anchor_path.is_file()
        or anchor_path.stat().st_size != anchor_evidence["size_bytes"]
        or common.sha256_file(anchor_path) != anchor_evidence["sha256"]
        or not guard_path.is_file()
        or guard_path.stat().st_size != guard_evidence["size_bytes"]
        or common.sha256_file(guard_path) != guard_evidence["sha256"]
        or not attestation_path.is_file()
        or attestation_path.stat().st_size != attestation_evidence["size_bytes"]
        or common.sha256_file(attestation_path) != attestation_evidence["sha256"]
    ):
        raise ScientificQualityAuditError("Launch evidence byte drift")
    release = common.load_json(release_path)
    if (
        not isinstance(release, dict)
        or set(release)
        != {"version", "status", "pins", "canonical_self_hash"}
        or release.get("canonical_self_hash") != _canonical_self_hash(release)
        or release.get("canonical_self_hash")
        != release_evidence["canonical_self_hash"]
        or release.get("status")
        != "EXTERNAL_REVIEW_CANDIDATE_NOT_FORMAL_AUTHORIZATION"
        or tuple(release["pins"])
        != (
            "launch_anchor",
            "quality_audit_c_amendment",
            "quality_policy",
            "quality_audit",
            "counterfactual_text",
            "blind_literal_scan",
            "sealed_literal_registry_builder",
            "sealed_literal_registry_receipt",
            "quality_tests",
            "dataset_root_manifest",
            "launch_guard",
        )
    ):
        raise ScientificQualityAuditError("Release manifest contract drift")
    for name, spec in release["pins"].items():
        _verify_pin(spec, label=f"release_manifest/{name}")
    if (
        dict(release["pins"]["launch_anchor"])
        != {
            key: anchor_evidence[key]
            for key in ("path", "size_bytes", "sha256")
        }
        or dict(release["pins"]["launch_guard"]) != dict(guard_evidence)
    ):
        raise ScientificQualityAuditError("Release manifest evidence drift")
    attestation = common.load_json(attestation_path)
    if (
        not isinstance(attestation, dict)
        or attestation.get("canonical_self_hash") != _canonical_self_hash(attestation)
        or attestation.get("canonical_self_hash")
        != attestation_evidence["canonical_self_hash"]
        or attestation.get("status")
        != "EXTERNAL_REVIEW_GO_DESIGN_QUALITY_AUDIT_ONLY"
        or attestation.get("verdict_last_line")
        != "允许清洁运行104-world质量审计"
        or attestation.get("release_manifest") != dict(release_evidence)
        or attestation.get("review_scope")
        != {
            "design_dataset_root": "design_preflight_v2_20260811",
            "world_count": 104,
            "quality_audit_run_authorized": True,
            "formal_generation_authorized": False,
            "model_training_authorized": False,
            "audit_truth_release_authorized": False,
        }
    ):
        raise ScientificQualityAuditError("External review attestation drift")
    anchor = common.load_json(anchor_path)
    if (
        not isinstance(anchor, dict)
        or set(anchor)
        != {
            "version",
            "status",
            "pins",
            "runtime",
            "quality_policy_canonical_self_hash",
            "dataset_root_manifest_canonical_self_hash",
            "canonical_self_hash",
        }
        or anchor.get("canonical_self_hash") != _canonical_self_hash(anchor)
        or anchor.get("canonical_self_hash")
        != anchor_evidence["canonical_self_hash"]
        or anchor.get("status")
        != "EXTERNAL_REVIEW_ATTESTATION_REQUIRED_BEFORE_EXECUTION"
    ):
        raise ScientificQualityAuditError("Launch anchor contract drift")
    if tuple(anchor["pins"]) != (
        "quality_audit_c_amendment",
        "quality_policy",
        "quality_audit",
        "counterfactual_text",
        "blind_literal_scan",
        "sealed_literal_registry_builder",
        "sealed_literal_registry_receipt",
        "quality_tests",
        "dataset_root_manifest",
        "launch_guard",
    ):
        raise ScientificQualityAuditError("Launch anchor pin order drift")
    for name, spec in anchor["pins"].items():
        _verify_pin(spec, label=f"launch_anchor/{name}")
    if dict(anchor["pins"]["launch_guard"]) != dict(guard_evidence):
        raise ScientificQualityAuditError("Launch guard is not bound by the anchor")
    actual_main = Path(
        str(getattr(sys.modules.get("__main__"), "__file__", ""))
    ).resolve()
    actual_argv0 = Path(sys.argv[0]).resolve()
    quality_path = Path(__file__).resolve()
    if (
        actual_main != DEFAULT_LAUNCH_GUARD_PATH.resolve()
        or actual_argv0 != DEFAULT_LAUNCH_GUARD_PATH.resolve()
        or caller_path.resolve() != DEFAULT_LAUNCH_GUARD_PATH.resolve()
        or common.repo_path(str(entry_evidence["main"]["path"])).resolve()
        != actual_main
        or common.repo_path(str(entry_evidence["argv0"]["path"])).resolve()
        != actual_argv0
        or common.repo_path(
            str(entry_evidence["quality_module"]["path"])
        ).resolve()
        != quality_path
        or dict(entry_evidence["main"]) != dict(guard_evidence)
        or dict(entry_evidence["argv0"]) != dict(guard_evidence)
        or dict(entry_evidence["quality_module"])
        != dict(release["pins"]["quality_audit"])
    ):
        raise ScientificQualityAuditError("Official launch process evidence drift")
    policy = common.load_json(DEFAULT_POLICY_PATH)
    root_manifest = common.load_json(
        common.repo_path(str(anchor["pins"]["dataset_root_manifest"]["path"]))
    )
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "unicode_data": unicodedata.unidata_version,
    }
    if (
        anchor["runtime"] != observed_runtime
        or anchor["quality_policy_canonical_self_hash"]
        != policy.get("canonical_self_hash")
        or anchor["dataset_root_manifest_canonical_self_hash"]
        != root_manifest.get("canonical_self_hash")
    ):
        raise ScientificQualityAuditError("Launch anchor semantic binding drift")
    return copy.deepcopy(dict(evidence))


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    if set(spec) != {"path", "size_bytes", "sha256"}:
        raise ScientificQualityAuditError(f"{label} pin schema drift")
    path = common.repo_path(str(spec["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != spec["size_bytes"]
        or common.sha256_file(path) != spec["sha256"]
    ):
        raise ScientificQualityAuditError(f"{label} bytes drift")
    return path


def _validate_policy_contract(policy: Mapping[str, Any]) -> None:
    if policy.get("version") != VERSION or policy.get("status") != STATUS:
        raise ScientificQualityAuditError("Quality policy version/status drift")
    if policy.get("runtime") != {
        "python": "3.10.11",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "scikit_learn": "1.7.2",
        "unicode_data": "13.0.0",
    }:
        raise ScientificQualityAuditError("Quality policy runtime contract drift")
    if tuple(policy["pins"]) != (
        "scientific_contract",
        "quality_audit_c_amendment",
        "dataset_builder_policy",
        "dataset_root_manifest",
        "style_derangement",
        "counterfactual_text",
        "blind_literal_scan",
        "sealed_literal_registry_builder",
        "sealed_literal_registry",
        "sealed_literal_registry_receipt",
        "quality_audit",
        "quality_tests",
    ):
        raise ScientificQualityAuditError("Quality pin universe/order drift")
    if policy["claim_boundary"] != {
        "design_scale_only": True,
        "formal_dataset_quality_pass": False,
        "training_qualified": False,
        "absence_of_all_unknown_shortcuts_proven": False,
    }:
        raise ScientificQualityAuditError("Quality claim boundary drift")
    if policy["row_audit"] != {
        "visible_profile_fields_in_order": list(VISIBLE_PROFILE_FIELDS),
        "forbidden_markers": [
            "train",
            "development",
            "audit_a",
            "audit_b",
            "candidate",
            "controller",
            "mechanism",
            "world_uid",
            "seller_uid",
            "canonical_pair_uid",
            "split_ordinal",
            "positive_targets",
            "negative_flags",
            "override_audit",
        ],
        "private_uid_literal_scan": True,
        "exact_production_replay": True,
        "train_development_private_truth_exact_replay": [
            "controller_membership",
            "pair_labels",
            "qrels",
        ],
        "audit_truth_read_before_prediction_freeze": False,
        "audit_split_private_semantic_open_allowed": False,
        "audit_split_private_byte_integrity_required": True,
        "audit_split_world_reconstruction_allowed": False,
        "audit_split_sealed_literal_scan_required": True,
        "audit_split_sealed_literal_scan_private_values_returned": False,
        "audit_split_all_private_string_leaves_and_uid_mapping_keys_scanned": True,
        "private_literal_symmetric_normalization": [
            "NFKC_casefold",
            "digit_separator_collapse",
            "url_and_platform_equivalents",
        ],
        "private_mapping_key_field_patterns": [
            "^occurrence_counts$",
            ".*_(?:registry|registries|counts)$",
            ".*_by_(?:uid|seller|controller|query)$",
        ],
        "short_private_markers_boundary_aware": ["bat", "qq", "tg", "wx"],
        "counterfactual_visible_text_scan": True,
        "counterfactual_scan_failure_classification": "auditor_execution_failed_dataset_retained",
        "train_development_private_identity_literal_scan": True,
        "identity_literal_equivalents": [
            "raw",
            "strip",
            "lower",
            "upper",
            "casefold",
            "NFC",
            "NFKC",
            "type_specific_url_handle_phone",
            "scheme_stripped_url",
            "www_stripped_url",
            "platform_prefix_stripped",
        ],
        "noise_slot_uid_scan": True,
        "noise_raw_surface_is_expected_model_visible_nuisance": True,
        "query_uid_scan": True,
        "controller_blind_visible_identity_pattern_scan": [
            name for name, _pattern in VISIBLE_IDENTITY_PATTERNS
        ],
        "train_development_privileged_registry_replay": [
            "controller_uid",
            "query_uid",
            "identity_value_hash",
        ],
        "audit_split_sealed_registry_isolation_required": [
            "controller_uid",
            "query_uid",
            "identity_value_hash",
        ],
        "audit_split_registry_values_returned_to_main_before_prediction_freeze": False,
        "blind_visible_numeric_replay_from_redacted_items": [
            "item_count",
            "title_length_median",
            "description_length_median",
            "digit_ratio_mean",
            "punct_ratio_mean",
            "repeated_title_share",
            "repeated_description_share",
        ],
    }:
        raise ScientificQualityAuditError("Quality row-audit boundary drift")
    receipt_path = common.repo_path(
        str(policy["pins"]["sealed_literal_registry_receipt"]["path"])
    )
    if not receipt_path.is_file():
        raise ScientificQualityAuditError("Sealed registry receipt is missing")
    receipt = common.load_json(receipt_path)
    if policy["input"] != {
        "execution_mode": "design_preflight",
        "dataset_root": "reports/step28_v13_v1_13_scientific_builder/design_preflight_v2_20260811",
        "root_manifest_canonical_self_hash": "9baa90828cf459bcee3cc6101c166f6c1084353dd2997e40e5d3d85d29f49d48",
        "builder_policy_canonical_self_hash": "8f40cc0b008e6447e5ace55b59159545346c6527a91d2129677e59ce087a7a47",
        "sealed_registry_source_closure_sha256": common.canonical_sha256(
            receipt.get("source_closure")
        ),
        "world_counts": {"train": 50, "development": 50, "audit_a": 2, "audit_b": 2},
        "output_root": "reports/step28_v13_v1_13_scientific_builder/quality_audit_design_v7_20260812",
    }:
        raise ScientificQualityAuditError("Quality input/output boundary drift")
    metadata = policy["metadata_probe"]
    if set(metadata) != {
        "feature_names_in_order",
        "average_precision_baseline",
        "feature_source_contract",
        "models",
        "hard_gates",
    }:
        raise ScientificQualityAuditError("Metadata policy keyset drift")
    seller_names = (
        "item_count",
        "title_missing_rate",
        "description_missing_rate",
        "time_bucket_probability_00",
        "time_bucket_probability_01",
        "time_bucket_probability_02",
        "time_bucket_probability_03",
        "seller_output_ordinal",
        "seller_uid_digest_00",
        "seller_uid_digest_01",
        "seller_uid_digest_02",
        "seller_uid_digest_03",
        *tuple(
            f"item_uid_digest_{stat}_{index:02d}"
            for stat in ("mean", "min", "max")
            for index in range(4)
        ),
        "item_output_ordinal_mean",
        "item_output_ordinal_min",
        "item_output_ordinal_max",
        "global_seller_output_ordinal",
    )
    expected_metadata_names = tuple(
        [f"absdiff__{name}" for name in seller_names]
        + [f"sum__{name}" for name in seller_names]
        + [
            "pair_output_ordinal",
            "global_pair_output_ordinal",
            "world_output_ordinal",
        ]
        + [f"pair_uid_digest_{index:02d}" for index in range(4)]
        + [f"world_uid_digest_{index:02d}" for index in range(4)]
    )
    if (
        tuple(metadata["feature_names_in_order"]) != expected_metadata_names
        or not math.isclose(float(metadata["average_precision_baseline"]), 20 / 378)
        or metadata.get("feature_source_contract")
        != {
            "item_count": "mounted_model_profile_value_equal_to_joined_observed_item_rows",
            "title_description_missingness": "observed_redacted_items_exact_model_projection",
            "time_bucket_probabilities": "joined_pre_redaction_source_items_structural_independence_only_not_model_visible",
            "item_join_key": "item_uid",
            "source_join_invariants": ["seller_uid", "world_uid"],
            "ordering_and_identifier_channels": "seller_item_pair_world_local_global_ordinals_and_sha256_digest_chunks",
        }
    ):
        raise ScientificQualityAuditError("Metadata feature/baseline contract drift")
    expected_gates = {
        "maximum_single_feature_symmetric_auc": 0.52,
        "maximum_family_symmetric_auc": 0.53,
        "maximum_family_average_precision_uplift": 0.01,
        "bootstrap_95_upper_symmetric_auc": 0.53,
        "bootstrap_95_upper_average_precision_uplift": 0.015,
    }
    if metadata["hard_gates"] != expected_gates:
        raise ScientificQualityAuditError("Metadata hard-gate drift")
    text = policy["text_counterfactual"]
    if set(text) != {
        "derangement_domain_ascii",
        "seller_count",
        "fixed_point_count",
        "intervention_dose",
        "controller_or_label_input_allowed",
        "only_intervention",
        "excluded_negative_flags",
        "eligible_pairs_per_world",
        "eligible_positives_per_world",
        "average_precision_baseline",
        "profile_fields_in_order",
        "fixed_support_slot_contract",
        "production_provenance_delta_contract",
        "production_numeric_fields_in_order",
        "vectorizers",
        "template_mask",
        "surface_metrics_in_order",
        "fixed_support_surface_metrics_in_order",
        "view_path_order",
        "views",
        "models",
        "hard_gates",
    }:
        raise ScientificQualityAuditError("Text policy keyset drift")
    expected_views = {
        "p_full": (
            "production_step3",
            75,
            "1c08e76c0f74ff126a0d3f722afa652c36393d3e30f200077c4c13c91820ec8b",
            "five_fields_char3_word12_plus_combined_char3_word12_plus_five_fields_surface_plus_max_mean_top2",
        ),
        "p_topic": (
            "production_step3",
            14,
            "6a201d2afbc4b1579cef20e53ae81703924b69ceba53df2b2913002447d0e891",
            "category_char3_word12_plus_category_surface",
        ),
        "p_template_surface": (
            "production_step3",
            56,
            "57fd57edf108f3ad09f50e88c9b3ee23644dcc1f3abff53ddc86a2d051cd4156",
            "four_text_fields_masked_char3_plus_combined_masked_char3_plus_four_fields_surface_plus_max_mean_top2",
        ),
        "fs_full": (
            "fixed_support",
            33,
            "e7e929d856423d03951612884bbffd57649190ecf5c414b76819e4129265957b",
            "per_item_title_description_joint_char3_word12_order_invariant_sum_plus_title_description_surface_plus_max_mean_top2",
        ),
        "fs_title": (
            "fixed_support",
            14,
            "71e72e4c3cf6ea36d78477acb0617f5436d2813c201998b13ae051b55fe9afe8",
            "per_item_title_char3_word12_order_invariant_sum_plus_title_surface",
        ),
        "fs_template_surface": (
            "fixed_support",
            30,
            "4ef95fb703e708e59f5334636c1bae539ed44dc35293832d23a908fea9252606",
            "per_item_title_description_joint_masked_char3_order_invariant_sum_plus_title_description_surface_plus_max_mean_top2",
        ),
        "u_joint_full": (
            "joint_visible_input",
            124,
            "420333af4f991424cd7d65ebeeaeb0aafd43ea612eba8398852c14c25525a745",
            "p_full_75_plus_fs_full_33_plus_eight_model_visible_numeric_absdiff_sum_16",
        ),
    }
    if tuple(text["views"]) != tuple(expected_views):
        raise ScientificQualityAuditError("Text view order drift")
    for name, (path_name, width, digest, recipe) in expected_views.items():
        if text["views"][name] != {
            "path": path_name,
            "recipe": recipe,
            "expected_width": width,
            "feature_names_canonical_sha256": digest,
        }:
            raise ScientificQualityAuditError("Text view recipe drift")
    if (
        text["derangement_domain_ascii"]
        != "step28-v13-v1.13-scientific-style-derangement-v1"
        or text["seller_count"] != 28
        or text["fixed_point_count"] != 0
        or text.get("intervention_dose")
        != {
            "required_source_seller_changed_count": 28,
            "minimum_effective_style_uid_changed_count": 20,
            "minimum_effective_style_factor_tuple_changed_count": 20,
            "minimum_seller_profile_text_changed_count": 20,
            "minimum_visible_seller_changed_count": 20,
            "maximum_zero_dose_seller_count": 8,
            "maximum_zero_visible_dose_seller_count": 8,
            "mapping_redraw_on_insufficient_dose_forbidden": True,
        }
        or text["controller_or_label_input_allowed"] is not False
        or text["only_intervention"]
        != "effective_style_uid_from_mapped_source_seller"
        or text["excluded_negative_flags"]
        != ["exact_title_clone_target", "high_semantic_similarity_target"]
        or text["eligible_pairs_per_world"] != 372
        or text["eligible_positives_per_world"] != 20
        or not math.isclose(float(text["average_precision_baseline"]), 20 / 372)
        or tuple(text["profile_fields_in_order"]) != VISIBLE_PROFILE_FIELDS
        or text.get("fixed_support_slot_contract")
        != {
            "source": "observed/redacted_items.jsonl_exact_production_replay",
            "key_fields_in_order": ["world_uid", "seller_uid", "item_uid", "field"],
            "allowed_fields_in_order": ["title", "description"],
            "original_counterfactual_keyset_equal": True,
            "original_counterfactual_empty_pattern_equal": True,
            "item_uid_use": "join_and_receipt_only_never_feature",
            "slot_vectorization": "each_item_field_independently_before_seller_aggregation",
            "seller_aggregation": "canonical_text_byte_order_sparse_sum_then_l2_normalize",
            "cross_item_ngram_allowed": False,
            "top_k_or_truncation_allowed": False,
            "surface_aggregation": "per_field_slot_mean_including_empty_slots",
            "empty_slot_rate_pair_features": "absolute_difference_and_sum",
        }
        or text.get("production_provenance_delta_contract")
        != {
            "original_equals_counterfactual_required": False,
            "counterfactual_equals_independent_replay_required": True,
            "source_item_must_belong_to_same_seller": True,
            "exact_wrapper_and_row_schema_required": True,
            "row_counts_digests_roles_ranks_and_ranges_recomputed": True,
            "delta_receipt_required": True,
            "delta_may_select_mapping_candidate_world_or_seller": False,
        }
        or tuple(text.get("production_numeric_fields_in_order", ()))
        != PRODUCTION_NUMERIC_FIELDS
        or text.get("view_path_order")
        != ["production_step3", "fixed_support", "joint_visible_input"]
        or tuple(text["surface_metrics_in_order"]) != SURFACE_METRICS
        or tuple(text["fixed_support_surface_metrics_in_order"])
        != FIXED_SUPPORT_SURFACE_METRICS
        or text["vectorizers"]
        != {
            "char3": {
                "analyzer": "char",
                "ngram_range": [3, 3],
                "n_features": 65536,
                "lowercase": False,
                "alternate_sign": False,
                "norm": "l2",
                "dtype": "float64",
            },
            "word12": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "n_features": 65536,
                "lowercase": False,
                "alternate_sign": False,
                "norm": "l2",
                "dtype": "float64",
                "tokenizer": "ascii_alnum_runs_lowercase_plus_individual_han_codepoints",
            },
        }
        or text["template_mask"]
        != "unicode_letters_to_字_decimal_digits_to_数_preserve_other_codepoints"
    ):
        raise ScientificQualityAuditError("Text intervention/feature boundary drift")
    expected_text_gates = {
        "maximum_single_feature_symmetric_auc": 0.52,
        "maximum_fourteen_model_symmetric_auc": 0.53,
        "maximum_fourteen_model_average_precision_uplift": 0.01,
        "bootstrap_95_upper_symmetric_auc": 0.53,
        "bootstrap_95_upper_average_precision_uplift": 0.015,
    }
    if text["hard_gates"] != expected_text_gates or text["models"] != metadata["models"]:
        raise ScientificQualityAuditError("Text hard-gate/model family drift")
    expected_probe_models = {
        "logistic_l2": {
            "C": 1.0,
            "class_weight": None,
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "l1_ratio": None,
            "max_iter": 10000,
            "n_jobs": None,
            "penalty": "l2",
            "random_state": 793820367,
            "solver": "lbfgs",
            "tol": 1e-10,
            "verbose": 0,
            "warm_start": False,
        },
        "shallow_tree": {
            "categorical_features": "from_dtype",
            "class_weight": None,
            "early_stopping": False,
            "interaction_cst": None,
            "l2_regularization": 1.0,
            "learning_rate": 0.03,
            "loss": "log_loss",
            "max_bins": 255,
            "max_depth": 2,
            "max_features": 1.0,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "monotonic_cst": None,
            "n_iter_no_change": 10,
            "random_state": 793820367,
            "scoring": "loss",
            "tol": 1e-7,
            "validation_fraction": 0.1,
            "verbose": 0,
            "warm_start": False,
        },
    }
    if metadata["models"] != expected_probe_models:
        raise ScientificQualityAuditError("Probe hyperparameter drift")
    if policy["bootstrap"] != {
        "replicates": 9999,
        "generator": "numpy.random.Generator(numpy.random.PCG64)",
        "development_world_count": 50,
        "world_order": "persisted_worlds_jsonl_split_ordinal_ascending_first_occurrence",
        "sampling": "world_indices_with_replacement",
        "quantile": 0.95,
        "quantile_method": "linear",
        "metadata_design_seed": 281320260810,
        "text_design_seed": 281320260810,
        "probe_model_seed_derivation": {
            "domain_ascii": "step28-v13-v1.13-quality-probe-model-v1",
            "source_uint64": 281320260810,
            "method": "sha256(domain || 0x1f || source_uint64_be8).first_uint32_be",
            "derived_uint32": 793820367,
        },
        "refit_models_inside_bootstrap": False,
    }:
        raise ScientificQualityAuditError("Quality bootstrap contract drift")
    if policy["launch_and_failure"] != {
        "external_review_attestation_required": True,
        "external_review_scope": "104_world_design_quality_audit_only",
        "formal_generation_authorized_by_attestation": False,
        "model_training_authorized_by_attestation": False,
        "official_entry_evidence": [
            "main_file",
            "argv0",
            "caller_file",
            "quality_module_file",
        ],
        "launch_failure_receipt_before_dataset_rows": True,
        "failure_receipt_raw_exception_message_forbidden": True,
        "cleanup_intent_committed_before_dataset_deletion": True,
        "cleanup_recovery_when_target_absent": True,
    }:
        raise ScientificQualityAuditError("Quality launch/failure contract drift")
    if policy["authorizations"] != {
        "formal_seed": False,
        "formal_generation": False,
        "model_training": False,
        "audit_a_truth_read": False,
        "audit_b_truth_read": False,
    }:
        raise ScientificQualityAuditError("Quality authorization drift")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise ScientificQualityAuditError("Only the canonical quality policy may pass")
    policy = common.load_json(path)
    expected = {
        "version",
        "status",
        "claim_boundary",
        "pins",
        "runtime",
        "input",
        "row_audit",
        "metadata_probe",
        "text_counterfactual",
        "bootstrap",
        "launch_and_failure",
        "authorizations",
        "canonical_self_hash",
    }
    if not isinstance(policy, dict) or set(policy) != expected:
        raise ScientificQualityAuditError("Quality policy keyset drift")
    if (
        policy["version"] != VERSION
        or policy["status"] != STATUS
        or policy.get("canonical_self_hash") != _canonical_self_hash(policy)
    ):
        raise ScientificQualityAuditError("Quality policy version/self-hash drift")
    for label, spec in policy["pins"].items():
        _verify_pin(spec, label=label)
    _validate_policy_contract(policy)
    runtime = policy["runtime"]
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "unicode_data": unicodedata.unidata_version,
    }
    if runtime != observed_runtime:
        raise ScientificQualityAuditError(
            f"Quality runtime drift: expected={runtime} observed={observed_runtime}"
        )
    authorizations = policy["authorizations"]
    if authorizations != {
        "formal_seed": False,
        "formal_generation": False,
        "model_training": False,
        "audit_a_truth_read": False,
        "audit_b_truth_read": False,
    }:
        raise ScientificQualityAuditError("Quality-audit authorization drift")
    return policy


def load_policy_for_cleanup_recovery() -> dict[str, Any]:
    """Validate policy without requiring the intentionally deleted dataset pin."""

    policy = common.load_json(DEFAULT_POLICY_PATH)
    expected = {
        "version",
        "status",
        "claim_boundary",
        "pins",
        "runtime",
        "input",
        "row_audit",
        "metadata_probe",
        "text_counterfactual",
        "bootstrap",
        "launch_and_failure",
        "authorizations",
        "canonical_self_hash",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != expected
        or policy.get("canonical_self_hash") != _canonical_self_hash(policy)
    ):
        raise ScientificQualityAuditError("Cleanup-recovery policy drift")
    _validate_policy_contract(policy)
    dataset_root = common.repo_path(str(policy["input"]["dataset_root"])).resolve()
    manifest_spec = policy["pins"]["dataset_root_manifest"]
    manifest_path = common.repo_path(str(manifest_spec["path"])).resolve()
    if (
        dataset_root.exists()
        or manifest_path != dataset_root / "root_manifest.json"
        or set(manifest_spec) != {"path", "size_bytes", "sha256"}
    ):
        raise ScientificQualityAuditError(
            "Cleanup recovery requires the exact dataset target to be absent"
        )
    for label, spec in policy["pins"].items():
        if label != "dataset_root_manifest":
            _verify_pin(spec, label=label)
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "unicode_data": unicodedata.unidata_version,
    }
    if policy["runtime"] != observed_runtime:
        raise ScientificQualityAuditError("Cleanup-recovery runtime drift")
    return policy


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetInvalidationError(
                    f"Malformed JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetInvalidationError("JSONL row must be an object")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetInvalidationError(f"CSV header missing: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise DatasetInvalidationError(f"CSV duplicate header: {path}")
        return [dict(row) for row in reader]


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[field])].append(dict(row))
    return dict(output)


def _new_blind_boundary_counters() -> dict[str, dict[str, int]]:
    return {
        split: {key: 0 for key in BLIND_COUNTER_KEYS}
        for split in ("audit_a", "audit_b")
    }


def _validate_final_blind_boundary_counters(
    counters_by_split: Mapping[str, Mapping[str, int]],
) -> None:
    if set(counters_by_split) != {"audit_a", "audit_b"} or any(
        tuple(counters) != BLIND_COUNTER_KEYS
        or dict(counters) != EXPECTED_FINAL_BLIND_COUNTERS
        for counters in counters_by_split.values()
    ):
        raise ScientificQualityAuditError("Quality blind-boundary counter drift")


def _sealed_literal_registry_contract(
    policy: Mapping[str, Any], *, split: str
) -> tuple[Path, dict[str, Any]]:
    if split not in {"audit_a", "audit_b"}:
        raise ScientificQualityAuditError("Sealed registry requested outside audit split")
    registry_path = _verify_pin(
        policy["pins"]["sealed_literal_registry"],
        label="sealed literal registry",
    )
    receipt_path = _verify_pin(
        policy["pins"]["sealed_literal_registry_receipt"],
        label="sealed literal registry receipt",
    )
    receipt = common.load_json(receipt_path)
    try:
        blind_literal_scan.validate_sealed_registry_public_receipt_structure(
            receipt
        )
    except blind_literal_scan.BlindLiteralInputError as exc:
        raise ScientificQualityAuditError(
            "Sealed registry public receipt structure drift"
        ) from exc
    expected_receipt_keys = set(
        blind_literal_scan.SEALED_REGISTRY_PUBLIC_RECEIPT_KEYS
    )
    registry_pin = policy["pins"]["sealed_literal_registry"]
    dataset_manifest_pin = policy["pins"]["dataset_root_manifest"]
    builder_pin = policy["pins"]["sealed_literal_registry_builder"]
    expected_transaction_id = common.canonical_sha256(
        {
            "version": (
                "2026-08-12-step28-v13-v1-13-"
                "sealed-literal-registry-transaction-v1"
            ),
            "dataset_root_manifest": receipt.get("dataset_root_manifest"),
            "builder_policy": receipt.get("builder_policy"),
            "source_closure": receipt.get("source_closure"),
            "targets": {
                "private_sidecar": registry_pin["path"],
                "public_receipt": policy["pins"][
                    "sealed_literal_registry_receipt"
                ]["path"],
            },
        }
    )
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt.get("version") != blind_literal_scan.SEALED_REGISTRY_VERSION
        or receipt.get("status")
        != "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO"
        or not isinstance(receipt.get("transaction_id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt.get("transaction_id", ""))
        is None
        or receipt.get("transaction_id") != expected_transaction_id
        or receipt.get("canonical_self_hash") != _canonical_self_hash(receipt)
        or receipt.get("worlds_replayed") != 104
        or receipt.get("audit_worlds_projected") != 4
        or receipt.get("labels_opened_for_exact_replay") is not True
        or receipt.get("labels_used_for_candidate_selection") is not False
        or receipt.get("labels_used_for_literal_selection") is not False
        or receipt.get("pair_label_rows_replayed") != 39_312
        or tuple(receipt.get("private_input_files_semantically_replayed", ()))
        != blind_literal_scan.SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
        or receipt.get("private_input_file_count")
        != len(blind_literal_scan.SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES)
        or any(
            receipt.get(name) != expected
            for name, expected in (
                ("private_values_returned", 0),
                ("private_relations_returned", 0),
                ("labels_returned", 0),
                ("qrels_returned", 0),
                ("observed_rows_modified", 0),
            )
        )
        or receipt.get("candidate_selection_changed") is not False
        or receipt.get("derangement_changed") is not False
        or receipt.get("quality_probe_run") is not False
        or receipt.get("formal_generation_authorized") is not False
        or receipt.get("model_training_authorized") is not False
        or any(
            receipt.get(name) is not False
            for name in (
                "formal_seed_authorized",
                "audit_truth_release_authorized",
                "quality_audit_run_authorized",
                "formal_500x4_generation_authorized",
                "design_dataset_training_qualified",
            )
        )
        or receipt.get("dataset_root_manifest")
        != {
            **dataset_manifest_pin,
            "canonical_self_hash": policy["input"][
                "root_manifest_canonical_self_hash"
            ],
        }
        or receipt.get("literal_authority_source")
        != policy["pins"]["blind_literal_scan"]
        or receipt.get("builder_policy")
        != {
            **policy["pins"]["dataset_builder_policy"],
            "canonical_self_hash": policy["input"][
                "builder_policy_canonical_self_hash"
            ],
        }
        or not isinstance(receipt.get("source_closure"), dict)
        or common.canonical_sha256(receipt.get("source_closure"))
        != policy["input"]["sealed_registry_source_closure_sha256"]
        or receipt.get("sealed_registry")
        != {
            **registry_pin,
            "canonical_self_hash": receipt.get("sealed_registry", {}).get(
                "canonical_self_hash"
            ),
        }
        or receipt.get("builder_source") != builder_pin
    ):
        raise ScientificQualityAuditError("Sealed registry public receipt drift")
    for name, spec in receipt["source_closure"].items():
        _verify_pin(spec, label=f"sealed_registry_source_closure/{name}")
    sealed_receipt = receipt["sealed_registry"]
    if (
        set(sealed_receipt) != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not isinstance(sealed_receipt["canonical_self_hash"], str)
        or len(sealed_receipt["canonical_self_hash"]) != 64
    ):
        raise ScientificQualityAuditError("Sealed registry public binding drift")
    split_commitments = receipt.get("split_commitments")
    if not isinstance(split_commitments, dict) or set(split_commitments) != {
        "audit_a",
        "audit_b",
    }:
        raise ScientificQualityAuditError("Sealed registry receipt split drift")
    split_value = split_commitments[split]
    if not isinstance(split_value, dict) or set(split_value) != {
        "world_count",
        "forbidden_literal_count",
        "category_commitments",
        "allowed_noise_raw_surface_commitment",
    }:
        raise ScientificQualityAuditError("Sealed registry receipt category drift")
    return registry_path, copy.deepcopy(dict(split_value))


def _run_blind_literal_scan(
    *,
    policy: Mapping[str, Any],
    dataset_root: Path,
    split: str,
    split_manifest: Mapping[str, Any],
    blind_counters: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if split not in {"audit_a", "audit_b"}:
        raise ScientificQualityAuditError("Sealed scan requested outside audit split")
    scanner = common.repo_path(str(policy["pins"]["blind_literal_scan"]["path"]))
    sealed_registry, sealed_contract = _sealed_literal_registry_contract(
        policy, split=split
    )
    blind_counters[split]["sealed_literal_scan_calls"] += 1
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(scanner),
            "--dataset-root",
            str(dataset_root),
            "--split",
            split,
            "--sealed-registry",
            str(sealed_registry),
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=300,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0:
        sealed_input_failure: Any = None
        if len(lines) == 1:
            try:
                sealed_input_failure = json.loads(lines[0])
            except json.JSONDecodeError:
                sealed_input_failure = None
        if (
            completed.returncode == 3
            and isinstance(sealed_input_failure, dict)
            and set(sealed_input_failure)
            == {
                "version",
                "status",
                "exception_type",
                "private_values_returned",
                "private_relation_rows_returned",
                "canonical_self_hash",
            }
            and sealed_input_failure.get("status") == "FAIL_SEALED_INPUT_INVALID"
            and sealed_input_failure.get("private_values_returned") == 0
            and sealed_input_failure.get("private_relation_rows_returned") == 0
            and sealed_input_failure.get("canonical_self_hash")
            == _canonical_self_hash(sealed_input_failure)
        ):
            raise DatasetInvalidationError(
                f"Sealed literal scanner found invalid dataset input: {split}"
            )
        raise ScientificQualityAuditError(
            f"Sealed literal scanner execution failed for {split}"
        )
    if len(lines) != 1:
        raise ScientificQualityAuditError("Sealed scanner output cardinality drift")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ScientificQualityAuditError("Sealed scanner output is malformed") from exc
    expected_keys = {
        "version",
        "status",
        "split",
        "private_relation_rows_returned",
        "private_values_returned",
        "labels_opened",
        "world_reconstructed",
        "private_literal_file_count",
        "private_row_counts",
        "sealed_registry_binding",
        "literal_category_counts",
        "visible_text_count",
        "hit_count",
        "hit_category_counts",
        "input_binding",
        "canonical_self_hash",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("split") != split
        or receipt.get("private_relation_rows_returned") != 0
        or receipt.get("private_values_returned") != 0
        or receipt.get("labels_opened") is not False
        or receipt.get("world_reconstructed") is not False
        or receipt.get("private_literal_file_count") != len(
            blind_literal_scan.PRIVATE_LITERAL_FILES
        )
        or receipt.get("canonical_self_hash") != _canonical_self_hash(receipt)
    ):
        raise ScientificQualityAuditError("Sealed scanner receipt contract drift")
    manifest_records = {
        str(record["path"]): record for record in split_manifest["files"]
    }
    expected_inputs = {
        relative: {
            "size_bytes": int(manifest_records[relative]["size_bytes"]),
            "sha256": str(manifest_records[relative]["sha256"]),
        }
        for relative in (
            *blind_literal_scan.PRIVATE_LITERAL_FILES,
            "observed/model_seller_profiles.jsonl",
            "observed/redacted_items.jsonl",
        )
    }
    sealed_public_receipt = common.load_json(
        common.repo_path(
            str(policy["pins"]["sealed_literal_registry_receipt"]["path"])
        )
    )
    sealed_public = sealed_public_receipt["sealed_registry"]
    expected_sealed_binding = {
        "transaction_id": sealed_public_receipt["transaction_id"],
        "size_bytes": sealed_public["size_bytes"],
        "sha256": sealed_public["sha256"],
        "canonical_self_hash": sealed_public["canonical_self_hash"],
        "world_count": sealed_contract["world_count"],
        "forbidden_literal_count": sealed_contract["forbidden_literal_count"],
        "category_counts": {
            category: sealed_contract["category_commitments"][category]["count"]
            for category in blind_literal_scan.SEALED_REGISTRY_CATEGORIES
        },
        "allowed_noise_raw_surface_count": sealed_contract[
            "allowed_noise_raw_surface_commitment"
        ]["count"],
    }
    if (
        receipt.get("input_binding") != expected_inputs
        or receipt.get("sealed_registry_binding") != expected_sealed_binding
    ):
        raise ScientificQualityAuditError("Sealed scanner input binding drift")
    if receipt.get("status") != "PASS_NO_PRIVATE_LITERAL_HIT" or receipt.get(
        "hit_count"
    ) != 0:
        raise DatasetInvalidationError(
            f"Sealed private literal leaked into model-visible text: {split}"
        )
    return receipt


def _run_blind_registry_isolation_scan(
    *,
    policy: Mapping[str, Any],
    dataset_root: Path,
    root_manifest: Mapping[str, Any],
    split_manifests: Mapping[str, Mapping[str, Any]],
    blind_counters: dict[str, dict[str, int]],
) -> dict[str, Any]:
    scanner = common.repo_path(str(policy["pins"]["blind_literal_scan"]["path"]))
    for split in ("audit_a", "audit_b"):
        blind_counters[split]["sealed_registry_isolation_calls"] += 1
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(scanner),
            "--dataset-root",
            str(dataset_root),
            "--mode",
            "registry-isolation",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=300,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0:
        sealed_input_failure: Any = None
        if len(lines) == 1:
            try:
                sealed_input_failure = json.loads(lines[0])
            except json.JSONDecodeError:
                sealed_input_failure = None
        if (
            completed.returncode == 3
            and isinstance(sealed_input_failure, dict)
            and sealed_input_failure.get("status") == "FAIL_SEALED_INPUT_INVALID"
            and sealed_input_failure.get("private_values_returned") == 0
            and sealed_input_failure.get("private_relation_rows_returned") == 0
            and sealed_input_failure.get("canonical_self_hash")
            == _canonical_self_hash(sealed_input_failure)
        ):
            raise DatasetInvalidationError(
                "Sealed private registry isolation found invalid dataset input"
            )
        raise ScientificQualityAuditError(
            "Sealed private registry isolation execution failed"
        )
    if len(lines) != 1:
        raise ScientificQualityAuditError(
            "Sealed registry-isolation output cardinality drift"
        )
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ScientificQualityAuditError(
            "Sealed registry-isolation output is malformed"
        ) from exc
    expected_keys = {
        "version",
        "status",
        "split_order",
        "private_relation_rows_returned",
        "private_values_returned",
        "labels_opened",
        "world_reconstructed",
        "split_commitments",
        "union_commitments",
        "cross_split_overlap_counts",
        "input_binding",
        "canonical_self_hash",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("status") != "PASS_PRIVATE_REGISTRY_SPLIT_ISOLATION"
        or tuple(receipt.get("split_order", ())) != tuple(scientific.SPLITS)
        or receipt.get("private_relation_rows_returned") != 0
        or receipt.get("private_values_returned") != 0
        or receipt.get("labels_opened") is not False
        or receipt.get("world_reconstructed") is not False
        or receipt.get("cross_split_overlap_counts")
        != {"controller_uid": 0, "identity_value": 0, "query_uid": 0}
        or receipt.get("canonical_self_hash") != _canonical_self_hash(receipt)
    ):
        raise ScientificQualityAuditError(
            "Sealed registry-isolation receipt contract drift"
        )
    expected_split_commitments: dict[str, dict[str, dict[str, Any]]] = {}
    expected_inputs = {
        "root_manifest.json": {
            "size_bytes": (dataset_root / "root_manifest.json").stat().st_size,
            "sha256": common.sha256_file(dataset_root / "root_manifest.json"),
        }
    }
    for split in scientific.SPLITS:
        manifest = split_manifests[split]
        records = {str(row["path"]): row for row in manifest["files"]}
        expected_split_commitments[split] = {
            "identity_value": {
                "count": manifest["identity_value_registry_count"],
                "sha256": manifest["identity_value_registry_sha256"],
            },
            "controller_uid": dict(manifest["uid_registries"]["controller"]),
            "query_uid": dict(manifest["uid_registries"]["query"]),
        }
        manifest_path = dataset_root / split / "split_manifest.json"
        expected_inputs[f"{split}/split_manifest.json"] = {
            "size_bytes": manifest_path.stat().st_size,
            "sha256": common.sha256_file(manifest_path),
        }
        for relative in blind_literal_scan.PRIVATE_LITERAL_FILES:
            expected_inputs[f"{split}/{relative}"] = {
                "size_bytes": int(records[relative]["size_bytes"]),
                "sha256": str(records[relative]["sha256"]),
            }
    expected_union = {
        "identity_value": {
            "count": root_manifest["identity_value_registry_count"],
            "sha256": root_manifest["identity_value_registry_sha256"],
        },
        "controller_uid": dict(root_manifest["uid_registries"]["controller"]),
        "query_uid": dict(root_manifest["uid_registries"]["query"]),
    }
    if (
        receipt.get("split_commitments") != expected_split_commitments
        or receipt.get("union_commitments") != expected_union
        or receipt.get("input_binding") != expected_inputs
    ):
        raise ScientificQualityAuditError(
            "Sealed registry-isolation commitment binding drift"
        )
    return receipt


def _reject_blind_privileged_replay(
    split: str, counters: dict[str, dict[str, int]]
) -> None:
    if split in counters:
        counters[split]["world_reconstruction_requests"] += 1
        counters[split]["build_private_truth_calls"] += 1
        counters[split]["controller_relation_reads"] += 1
        counters[split]["qrels_generations"] += 1
        raise ScientificQualityAuditError(
            "Audit split cannot enter privileged world reconstruction"
        )


def _verify_quality_input_tree(
    root: Path, root_manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Verify bytes without opening sealed audit-split private payloads."""

    if common.load_json(root / "root_manifest.json") != root_manifest:
        raise DatasetInvalidationError("Root manifest replay drift")
    if _canonical_self_hash(root_manifest) != root_manifest.get(
        "canonical_self_hash"
    ):
        raise DatasetInvalidationError("Root manifest self-hash drift")
    manifests: dict[str, dict[str, Any]] = {}
    expected_relative_files = {"root_manifest.json"}
    for split in scientific.SPLITS:
        split_root = root / split
        manifest = common.load_json(split_root / "split_manifest.json")
        if _canonical_self_hash(manifest) != manifest.get("canonical_self_hash"):
            raise DatasetInvalidationError(
                f"Split manifest self-hash drift: {split}"
            )
        if (
            root_manifest["split_manifest_self_hashes"].get(split)
            != manifest["canonical_self_hash"]
        ):
            raise DatasetInvalidationError(
                f"Root/split manifest binding drift: {split}"
            )
        records = manifest.get("files")
        if not isinstance(records, list) or {
            str(record.get("path"))
            for record in records
            if isinstance(record, Mapping)
        } != set(dataset_builder.EXPECTED_SPLIT_DATA_PATHS):
            raise DatasetInvalidationError(f"Split file universe drift: {split}")
        for record in records:
            if set(record) != {"path", "size_bytes", "sha256", "row_count"}:
                raise DatasetInvalidationError(
                    f"Split file-record schema drift: {split}"
                )
            relative = str(record["path"])
            path = (split_root / relative).resolve()
            if split_root.resolve() not in path.parents or not path.is_file():
                raise DatasetInvalidationError(
                    f"Unsafe or missing split file: {split}/{relative}"
                )
            # Byte/line verification deliberately avoids the semantic readers,
            # so audit-private payloads remain sealed while every manifest
            # record is still reverified.
            if (
                path.stat().st_size != record["size_bytes"]
                or common.sha256_file(path) != record["sha256"]
                or dataset_builder._count_file_rows(path) != record["row_count"]
            ):
                raise DatasetInvalidationError(
                    f"Split file byte/line replay drift: {split}/{relative}"
                )
        manifests[split] = manifest
        expected_relative_files.add(f"{split}/split_manifest.json")
        expected_relative_files.update(
            f"{split}/{relative}"
            for relative in dataset_builder.EXPECTED_SPLIT_DATA_PATHS
        )
    _assert_exact_file_universe(root, expected_relative_files)
    return manifests


def _assert_exact_file_universe(root: Path, expected_relative_files: set[str]) -> None:
    actual_relative_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_relative_files != expected_relative_files:
        raise DatasetInvalidationError("Dataset input file universe drift")


def symmetric_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ScientificQualityAuditError("Metric vector shape/nonfinite drift")
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def _rank_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return symmetric_auc(labels, scores), float(average_precision_score(labels, scores))


def _bootstrap_family_upper(
    *,
    labels: np.ndarray,
    row_world_uids: np.ndarray,
    ordered_world_uids: Sequence[str],
    score_family: Sequence[np.ndarray],
    replicates: int,
    seed: int,
    baseline: float,
) -> dict[str, Any]:
    worlds = [str(value) for value in ordered_world_uids]
    if (
        not worlds
        or len(worlds) != len(set(worlds))
        or set(worlds) != {str(value) for value in row_world_uids.tolist()}
        or not score_family
        or any(scores.shape != labels.shape for scores in score_family)
        or any(not np.isfinite(scores).all() for scores in score_family)
        or any(
            int(np.sum(row_world_uids == world_uid)) <= 0
            for world_uid in worlds
        )
    ):
        raise ScientificQualityAuditError("Bootstrap world universe drift")
    world_index_map = {world_uid: index for index, world_uid in enumerate(worlds)}
    row_world_indices = np.fromiter(
        (world_index_map[str(value)] for value in row_world_uids),
        dtype=np.int64,
        count=len(row_world_uids),
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(
        0,
        len(worlds),
        size=(replicates, len(worlds)),
        endpoint=False,
        dtype=np.int64,
    )
    multiplicities = np.zeros((replicates, len(worlds)), dtype=np.float64)
    for replicate, draw in enumerate(draws):
        multiplicities[replicate] = np.bincount(draw, minlength=len(worlds))
    auc_family: list[np.ndarray] = []
    ap_family: list[np.ndarray] = []
    for scores in score_family:
        unique_scores, group_index = np.unique(scores, return_inverse=True)
        group_positive = np.zeros((len(unique_scores), len(worlds)), dtype=np.float64)
        group_negative = np.zeros_like(group_positive)
        np.add.at(group_positive, (group_index, row_world_indices), labels)
        np.add.at(group_negative, (group_index, row_world_indices), 1 - labels)
        auc_values = np.empty(replicates, dtype=np.float64)
        ap_values = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 64):
            stop = min(replicates, start + 64)
            weights = multiplicities[start:stop]
            positive = weights @ group_positive.T
            negative = weights @ group_negative.T
            total_positive = positive.sum(axis=1)
            total_negative = negative.sum(axis=1)
            if np.any(total_positive <= 0) or np.any(total_negative <= 0):
                raise ScientificQualityAuditError("Bootstrap replicate lost a class")
            negative_before = np.cumsum(negative, axis=1) - negative
            auc = np.sum(
                positive * (negative_before + 0.5 * negative), axis=1
            ) / (total_positive * total_negative)
            auc_values[start:stop] = np.maximum(auc, 1.0 - auc)
            positive_desc = positive[:, ::-1]
            total_desc = (positive + negative)[:, ::-1]
            cumulative_positive = np.cumsum(positive_desc, axis=1)
            cumulative_total = np.cumsum(total_desc, axis=1)
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive),
                where=cumulative_total > 0,
            )
            ap_values[start:stop] = (
                np.sum(positive_desc * precision, axis=1) / total_positive
            )
        auc_family.append(auc_values)
        ap_family.append(ap_values - baseline)
    auc_max = np.maximum.reduce(auc_family)
    ap_max = np.maximum.reduce(ap_family)
    draw_binding = hashlib.sha256()
    draw_binding.update(common.canonical_json_bytes(worlds))
    draw_binding.update(len(worlds).to_bytes(8, "big", signed=False))
    draw_binding.update(replicates.to_bytes(8, "big", signed=False))
    draw_binding.update(draws.astype(">u8", copy=False).tobytes(order="C"))
    return {
        "replicates": replicates,
        "seed": seed,
        "score_family_size": len(score_family),
        "world_count": len(worlds),
        "ordered_world_uids_sha256": common.canonical_sha256(worlds),
        "ordinal_sequence_sha256": common.canonical_sha256(
            list(range(len(worlds)))
        ),
        "draws_and_world_order_sha256": draw_binding.hexdigest(),
        "auc_95_upper": float(np.quantile(auc_max, 0.95, method="linear")),
        "ap_uplift_95_upper": float(np.quantile(ap_max, 0.95, method="linear")),
    }


def _uid_chunks(value: str) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    denominator = float((1 << 64) - 1)
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "big") / denominator
        for offset in range(0, 32, 8)
    )


def _seller_metadata(
    *,
    profiles: Sequence[Mapping[str, Any]],
    observed_items: Sequence[Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    item_rows: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed_items:
        item_rows[str(row["seller_uid"])].append(row)
    source_by_uid = {str(row["item_uid"]): row for row in source_items}
    if len(source_by_uid) != len(source_items):
        raise DatasetInvalidationError("Metadata source item UID collision")
    if set(source_by_uid) != {str(row["item_uid"]) for row in observed_items}:
        raise DatasetInvalidationError("Metadata observed/source item keyset drift")
    for row in observed_items:
        source = source_by_uid[str(row["item_uid"])]
        if any(
            str(source[field]) != str(row[field])
            for field in ("seller_uid", "world_uid")
        ):
            raise DatasetInvalidationError("Metadata observed/source item join drift")
    profile_uids = [str(row["seller_uid"]) for row in profiles]
    if len(profile_uids) != len(set(profile_uids)) or set(profile_uids) != set(item_rows):
        raise DatasetInvalidationError("Metadata seller/item join drift")
    profile_by_uid = {str(row["seller_uid"]): row for row in profiles}
    seller_world = {
        seller_uid: {str(row["world_uid"]) for row in rows}
        for seller_uid, rows in item_rows.items()
    }
    if any(len(worlds) != 1 for worlds in seller_world.values()):
        raise DatasetInvalidationError("Seller crosses world boundary")
    ordered_by_world: defaultdict[str, list[str]] = defaultdict(list)
    for seller_uid in profile_uids:
        ordered_by_world[next(iter(seller_world[seller_uid]))].append(seller_uid)
    seller_ordinal = {
        seller_uid: ordinal / max(1, len(world_sellers) - 1)
        for world_sellers in ordered_by_world.values()
        for ordinal, seller_uid in enumerate(world_sellers)
    }
    global_seller_ordinal = {
        seller_uid: ordinal / max(1, len(profile_uids) - 1)
        for ordinal, seller_uid in enumerate(profile_uids)
    }
    item_ordinal = {
        str(row["item_uid"]): ordinal / max(1, len(observed_items) - 1)
        for ordinal, row in enumerate(observed_items)
    }
    names = (
        "item_count",
        "title_missing_rate",
        "description_missing_rate",
        "time_bucket_probability_00",
        "time_bucket_probability_01",
        "time_bucket_probability_02",
        "time_bucket_probability_03",
        "seller_output_ordinal",
        "seller_uid_digest_00",
        "seller_uid_digest_01",
        "seller_uid_digest_02",
        "seller_uid_digest_03",
        "item_uid_digest_mean_00",
        "item_uid_digest_mean_01",
        "item_uid_digest_mean_02",
        "item_uid_digest_mean_03",
        "item_uid_digest_min_00",
        "item_uid_digest_min_01",
        "item_uid_digest_min_02",
        "item_uid_digest_min_03",
        "item_uid_digest_max_00",
        "item_uid_digest_max_01",
        "item_uid_digest_max_02",
        "item_uid_digest_max_03",
        "item_output_ordinal_mean",
        "item_output_ordinal_min",
        "item_output_ordinal_max",
        "global_seller_output_ordinal",
    )
    result: dict[str, np.ndarray] = {}
    for seller_uid in profile_uids:
        rows = item_rows[seller_uid]
        count = len(rows)
        mounted_count = profile_by_uid[seller_uid].get("item_count")
        if (
            isinstance(mounted_count, bool)
            or not isinstance(mounted_count, int)
            or mounted_count != count
        ):
            raise DatasetInvalidationError("Mounted seller item_count drift")
        source_rows = [source_by_uid[str(row["item_uid"])] for row in rows]
        item_uid_matrix = np.asarray(
            [_uid_chunks(str(row["item_uid"])) for row in rows], dtype=np.float64
        )
        item_ordinals = np.asarray(
            [item_ordinal[str(row["item_uid"])] for row in rows], dtype=np.float64
        )
        values = [
            float(mounted_count),
            sum(not str(row["title"]) for row in rows) / count,
            sum(not str(row["description"]) for row in rows) / count,
            *(
                sum(int(row.get("time_bucket", -1)) == bucket for row in source_rows)
                / count
                for bucket in range(4)
            ),
            seller_ordinal[seller_uid],
            *_uid_chunks(seller_uid),
            *item_uid_matrix.mean(axis=0).tolist(),
            *item_uid_matrix.min(axis=0).tolist(),
            *item_uid_matrix.max(axis=0).tolist(),
            float(item_ordinals.mean()),
            float(item_ordinals.min()),
            float(item_ordinals.max()),
            global_seller_ordinal[seller_uid],
        ]
        result[seller_uid] = np.asarray(values, dtype=np.float64)
    return result, names


def build_metadata_matrix(
    *,
    profiles: Sequence[Mapping[str, Any]],
    observed_items: Sequence[Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    seller, seller_names = _seller_metadata(
        profiles=profiles,
        observed_items=observed_items,
        source_items=source_items,
    )
    names = tuple(
        [f"absdiff__{name}" for name in seller_names]
        + [f"sum__{name}" for name in seller_names]
        + [
            "pair_output_ordinal",
            "global_pair_output_ordinal",
            "world_output_ordinal",
        ]
        + [f"pair_uid_digest_{index:02d}" for index in range(4)]
        + [f"world_uid_digest_{index:02d}" for index in range(4)]
    )
    matrix = np.empty((len(endpoints), len(names)), dtype=np.float64)
    pair_uids: list[str] = []
    world_uids: list[str] = []
    denominator_by_world = Counter(str(row["world_uid"]) for row in endpoints)
    ordinal_by_world: Counter[str] = Counter()
    ordered_worlds = list(
        dict.fromkeys(str(row["world_uid"]) for row in endpoints)
    )
    world_ordinal = {
        world_uid: ordinal / max(1, len(ordered_worlds) - 1)
        for ordinal, world_uid in enumerate(ordered_worlds)
    }
    for row_index, row in enumerate(endpoints):
        left = seller[str(row["seller_uid_left"])]
        right = seller[str(row["seller_uid_right"])]
        world_uid = str(row["world_uid"])
        ordinal = ordinal_by_world[world_uid]
        ordinal_by_world[world_uid] += 1
        pair_uid = str(row["canonical_pair_uid"])
        values = np.concatenate(
            (
                np.abs(left - right),
                left + right,
                np.asarray(
                    [
                        ordinal / max(1, denominator_by_world[world_uid] - 1),
                        row_index / max(1, len(endpoints) - 1),
                        world_ordinal[world_uid],
                    ]
                ),
                np.asarray(_uid_chunks(pair_uid)),
                np.asarray(_uid_chunks(world_uid)),
            )
        )
        matrix[row_index] = values
        pair_uids.append(pair_uid)
        world_uids.append(world_uid)
    if not np.isfinite(matrix).all():
        raise ScientificQualityAuditError("Metadata matrix is nonfinite")
    return matrix, names, np.asarray(pair_uids, dtype=object), np.asarray(world_uids, dtype=object)


def _fit_probe_family(
    train_x: np.ndarray,
    train_y: np.ndarray,
    development_x: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    logistic = LogisticRegression(**dict(config["logistic_l2"])).fit(
        scaler.transform(train_x), train_y
    )
    if int(logistic.n_iter_[0]) >= int(config["logistic_l2"]["max_iter"]):
        raise ScientificQualityAuditError("Metadata/text logistic did not converge")
    tree = HistGradientBoostingClassifier(**dict(config["shallow_tree"])).fit(
        train_x, train_y
    )
    return {
        "logistic_l2": logistic.predict_proba(scaler.transform(development_x))[:, 1],
        "shallow_tree": tree.predict_proba(development_x)[:, 1],
    }


def word12_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    ascii_run: list[str] = []

    def flush_ascii() -> None:
        if ascii_run:
            tokens.append("".join(ascii_run).lower())
            ascii_run.clear()

    for character in text:
        codepoint = ord(character)
        if (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
            or "0" <= character <= "9"
        ):
            ascii_run.append(character)
        elif (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            flush_ascii()
            tokens.append(character)
        else:
            flush_ascii()
    flush_ascii()
    return tokens


def template_mask(text: str) -> str:
    output = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L"):
            output.append("字")
        elif category == "Nd":
            output.append("数")
        else:
            output.append(character)
    return "".join(output)


def _char_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=None,
        ngram_range=(3, 3),
        analyzer="char",
        n_features=65536,
        binary=False,
        norm="l2",
        alternate_sign=False,
        dtype=np.float64,
    )


def _word_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=word12_tokens,
        stop_words=None,
        token_pattern=None,
        ngram_range=(1, 2),
        analyzer="word",
        n_features=65536,
        binary=False,
        norm="l2",
        alternate_sign=False,
        dtype=np.float64,
    )


def _pair_cosines(
    matrix: sparse.csr_matrix,
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    left = np.fromiter(
        (seller_row[str(row["seller_uid_left"])] for row in endpoints),
        dtype=np.int64,
        count=len(endpoints),
    )
    right = np.fromiter(
        (seller_row[str(row["seller_uid_right"])] for row in endpoints),
        dtype=np.int64,
        count=len(endpoints),
    )
    values = np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
    if not np.isfinite(values).all():
        raise ScientificQualityAuditError("Text cosine is nonfinite")
    return values.astype(np.float64, copy=False)


def _surface_counts(text: str) -> tuple[int, int, int, int, int, int]:
    return (
        len(text),
        text.count("\n"),
        sum(unicodedata.category(character) in PUNCTUATION_CATEGORIES for character in text),
        sum(character in ASCII_WHITESPACE for character in text),
        sum(unicodedata.category(character) == "Nd" for character in text),
        int(not text),
    )


def _surface_pair_features(
    texts: Sequence[str],
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    counts = [_surface_counts(text) for text in texts]
    return _surface_pair_features_from_counts(
        counts, seller_row=seller_row, endpoints=endpoints
    )


def _surface_pair_features_from_counts(
    counts: Sequence[Sequence[float]],
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if len(counts) != len(seller_row) or any(len(row) != 6 for row in counts):
        raise ScientificQualityAuditError("Surface-count matrix shape drift")
    output = np.empty((len(endpoints), len(SURFACE_METRICS)), dtype=np.float64)
    for index, row in enumerate(endpoints):
        left = counts[seller_row[str(row["seller_uid_left"])]]
        right = counts[seller_row[str(row["seller_uid_right"])]]
        values: list[float] = []
        for offset in range(5):
            values.extend((float(abs(left[offset] - right[offset])), float(left[offset] + right[offset])))
        values.extend((float(left[5] and right[5]), float(bool(left[5]) ^ bool(right[5]))))
        output[index] = values
    return output


def _fixed_support_surface_pair_features_from_counts(
    counts: Sequence[Sequence[float]],
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Build symmetric pair features from fixed-slot seller means.

    The sixth seller value is an empty-slot *rate* in [0, 1], not the binary
    empty indicator used by the production-profile text path.
    """

    if len(counts) != len(seller_row) or any(len(row) != 6 for row in counts):
        raise ScientificQualityAuditError(
            "Fixed-support surface-count matrix shape drift"
        )
    output = np.empty(
        (len(endpoints), len(FIXED_SUPPORT_SURFACE_METRICS)), dtype=np.float64
    )
    for index, row in enumerate(endpoints):
        left = counts[seller_row[str(row["seller_uid_left"])]]
        right = counts[seller_row[str(row["seller_uid_right"])]]
        if any(
            not math.isfinite(float(value)) for value in (*left, *right)
        ) or not (
            0.0 <= float(left[5]) <= 1.0 and 0.0 <= float(right[5]) <= 1.0
        ):
            raise ScientificQualityAuditError(
                "Fixed-support surface count/rate is invalid"
            )
        values: list[float] = []
        for offset in range(6):
            values.extend(
                (
                    float(abs(float(left[offset]) - float(right[offset]))),
                    float(float(left[offset]) + float(right[offset])),
                )
            )
        output[index] = values
    return output


def _seller_slot_matrix(
    *,
    texts_by_seller: Mapping[str, Sequence[str]],
    seller_uids: Sequence[str],
    vectorizer: HashingVectorizer,
    mask: bool = False,
) -> sparse.csr_matrix:
    """Vectorize each item slot independently, then aggregate without row order."""

    rows: list[sparse.csr_matrix] = []
    for seller_uid in seller_uids:
        values = list(texts_by_seller[seller_uid])
        if not values:
            raise ScientificQualityAuditError("Fixed-support seller has no slots")
        if mask:
            values = [template_mask(value) for value in values]
        # Canonical text-byte ordering makes sparse floating-point addition
        # independent of persisted item order and of item-UID renaming.
        values.sort(key=lambda value: value.encode("utf-8"))
        transformed = vectorizer.transform(values).tocsr()
        aggregate = sparse.csr_matrix(transformed.sum(axis=0), dtype=np.float64)
        rows.append(aggregate)
    return normalize(sparse.vstack(rows, format="csr"), norm="l2", axis=1)


def _fixed_support_slot_contract(
    *,
    split: str,
    world_uid: str,
    original_items: Sequence[Mapping[str, Any]],
    counterfactual_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_fields = tuple(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)

    def index_rows(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> dict[tuple[str, str, str], dict[str, Any]]:
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for source_row in rows:
            if tuple(source_row) != expected_fields:
                raise ScientificQualityAuditError(
                    f"{label} fixed-support item schema/order drift"
                )
            row = dict(source_row)
            item_uid = str(row["item_uid"])
            key = (str(row["world_uid"]), str(row["seller_uid"]), item_uid)
            if not all(key) or key in index:
                raise ScientificQualityAuditError(
                    f"{label} fixed-support item key collision"
                )
            index[key] = row
        return index

    original = index_rows(original_items, label="Original")
    changed = index_rows(counterfactual_items, label="Counterfactual")
    if split not in {"train", "development"} or not world_uid:
        raise ScientificQualityAuditError("Fixed-support receipt boundary drift")
    observed_worlds = {key[0] for key in original}
    if observed_worlds != {world_uid}:
        raise ScientificQualityAuditError("Fixed-support world binding drift")
    if set(original) != set(changed):
        raise ScientificQualityAuditError("Fixed-support item keyset drift")
    slot_keys: list[list[str]] = []
    title_changed = 0
    description_changed = 0
    seller_changed: set[str] = set()
    for key in sorted(
        original,
        key=lambda value: tuple(part.encode("utf-8") for part in value),
    ):
        before = original[key]
        after = changed[key]
        for field in FIXED_SUPPORT_FIELDS:
            slot_keys.append([*key, field])
            if bool(str(before[field])) != bool(str(after[field])):
                raise ScientificQualityAuditError(
                    "Fixed-support original/counterfactual empty pattern drift"
                )
            if str(before[field]) != str(after[field]):
                seller_changed.add(key[1])
                if field == "title":
                    title_changed += 1
                else:
                    description_changed += 1
    receipt = {
        "version": "2026-08-11-step28-v13-v1-13-fixed-support-receipt-v2",
        "split": split,
        "world_uid": world_uid,
        "item_count": len(original),
        "slot_count": len(slot_keys),
        "slot_keyset_sha256": common.canonical_sha256(slot_keys),
        "title_slot_changed_count": title_changed,
        "description_slot_changed_count": description_changed,
        "visible_seller_changed_count": len(seller_changed),
        "item_uid_used_as_feature": False,
        "cross_item_ngram_allowed": False,
        "row_order_sensitive": False,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def build_fixed_support_text_views(
    *,
    items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], np.ndarray, np.ndarray]:
    item_uids: set[str] = set()
    by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_fields = tuple(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)
    for source_row in items:
        if tuple(source_row) != expected_fields:
            raise ScientificQualityAuditError("Fixed-support item schema/order drift")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        if not item_uid or item_uid in item_uids:
            raise ScientificQualityAuditError("Fixed-support item UID collision")
        item_uids.add(item_uid)
        by_seller[str(row["seller_uid"])].append(row)
    endpoint_sellers = {
        str(row[field])
        for row in endpoints
        for field in ("seller_uid_left", "seller_uid_right")
    }
    if set(by_seller) != endpoint_sellers:
        raise ScientificQualityAuditError("Fixed-support seller/endpoint join drift")
    seller_uids = sorted(endpoint_sellers, key=lambda value: value.encode("utf-8"))
    seller_row = {seller_uid: index for index, seller_uid in enumerate(seller_uids)}
    texts: dict[str, dict[str, list[str]]] = {
        field: {} for field in (*FIXED_SUPPORT_FIELDS, "item_joint")
    }
    surface_means: dict[str, list[tuple[float, ...]]] = {
        field: [] for field in FIXED_SUPPORT_FIELDS
    }
    for seller_uid in seller_uids:
        rows = by_seller[seller_uid]
        for field in FIXED_SUPPORT_FIELDS:
            values = [str(row[field]) for row in rows]
            texts[field][seller_uid] = values
            counts = np.asarray([_surface_counts(value) for value in values], dtype=np.float64)
            surface_means[field].append(tuple(counts.mean(axis=0).tolist()))
        texts["item_joint"][seller_uid] = [
            str(row["title"]) + COMBINED_SEPARATOR + str(row["description"])
            for row in rows
        ]
    char = _char_vectorizer()
    word = _word_vectorizer()
    char_cos: dict[str, np.ndarray] = {}
    word_cos: dict[str, np.ndarray] = {}
    masked_cos: dict[str, np.ndarray] = {}
    for field in (*FIXED_SUPPORT_FIELDS, "item_joint"):
        char_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=char,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        word_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=word,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        masked_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=char,
                mask=True,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
    surfaces = {
        field: _fixed_support_surface_pair_features_from_counts(
            surface_means[field], seller_row=seller_row, endpoints=endpoints
        )
        for field in FIXED_SUPPORT_FIELDS
    }

    full_similarity = np.column_stack(
        [
            char_cos["title"],
            word_cos["title"],
            char_cos["description"],
            word_cos["description"],
            char_cos["item_joint"],
            word_cos["item_joint"],
        ]
    )
    full = np.column_stack(
        [
            full_similarity,
            surfaces["title"],
            surfaces["description"],
            np.max(full_similarity, axis=1),
            np.mean(full_similarity, axis=1),
            np.mean(np.sort(full_similarity, axis=1)[:, -2:], axis=1),
        ]
    )
    full_names = tuple(
        [
            "char3_cosine__slot_title",
            "word12_cosine__slot_title",
            "char3_cosine__slot_description",
            "word12_cosine__slot_description",
            "char3_cosine__slot_item_joint",
            "word12_cosine__slot_item_joint",
        ]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
        + [
            f"{name}__slot_description_mean"
            for name in FIXED_SUPPORT_SURFACE_METRICS
        ]
        + [
            "similarity_max__fixed_slots",
            "similarity_mean__fixed_slots",
            "similarity_top2_mean__fixed_slots",
        ]
    )
    title = np.column_stack(
        [char_cos["title"], word_cos["title"], surfaces["title"]]
    )
    title_names = tuple(
        ["char3_cosine__slot_title", "word12_cosine__slot_title"]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
    )
    masked_similarity = np.column_stack(
        [
            masked_cos["title"],
            masked_cos["description"],
            masked_cos["item_joint"],
        ]
    )
    template = np.column_stack(
        [
            masked_similarity,
            surfaces["title"],
            surfaces["description"],
            np.max(masked_similarity, axis=1),
            np.mean(masked_similarity, axis=1),
            np.mean(np.sort(masked_similarity, axis=1)[:, -2:], axis=1),
        ]
    )
    template_names = tuple(
        [
            "masked_char3_cosine__slot_title",
            "masked_char3_cosine__slot_description",
            "masked_char3_cosine__slot_item_joint",
        ]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
        + [
            f"{name}__slot_description_mean"
            for name in FIXED_SUPPORT_SURFACE_METRICS
        ]
        + [
            "masked_similarity_max__fixed_slots",
            "masked_similarity_mean__fixed_slots",
            "masked_similarity_top2_mean__fixed_slots",
        ]
    )
    views = {"fs_full": full, "fs_title": title, "fs_template_surface": template}
    names = {
        "fs_full": full_names,
        "fs_title": title_names,
        "fs_template_surface": template_names,
    }
    if {name: matrix.shape[1] for name, matrix in views.items()} != {
        "fs_full": 33,
        "fs_title": 14,
        "fs_template_surface": 30,
    }:
        raise ScientificQualityAuditError("Fixed-support view width drift")
    pair_uids = np.asarray(
        [str(row["canonical_pair_uid"]) for row in endpoints], dtype=object
    )
    world_uids = np.asarray([str(row["world_uid"]) for row in endpoints], dtype=object)
    return views, names, pair_uids, world_uids


def build_production_numeric_matrix(
    *,
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    expected_fields = tuple(dataset_builder.MODEL_PROFILE_FIELDS)
    for row in profiles:
        if tuple(row) != expected_fields:
            raise ScientificQualityAuditError("Production numeric profile schema drift")
        seller_uid = str(row["seller_uid"])
        if not seller_uid or seller_uid in values:
            raise ScientificQualityAuditError("Production numeric seller collision")
        title_stats = row["title_length_stats"]
        description_stats = row["description_length_stats"]
        style_stats = row["style_stats"]
        if (
            not isinstance(title_stats, Mapping)
            or set(title_stats) != {"median"}
            or not isinstance(description_stats, Mapping)
            or set(description_stats) != {"median"}
            or not isinstance(style_stats, Mapping)
            or tuple(style_stats) != dataset_builder.MODEL_PROFILE_STYLE_FIELDS
        ):
            raise ScientificQualityAuditError("Production numeric nested schema drift")
        vector = np.asarray(
            [
                row["item_count"],
                title_stats["median"],
                description_stats["median"],
                *(style_stats[name] for name in dataset_builder.MODEL_PROFILE_STYLE_FIELDS),
            ],
            dtype=np.float64,
        )
        if vector.shape != (8,) or not np.isfinite(vector).all():
            raise ScientificQualityAuditError("Production numeric profile value drift")
        values[seller_uid] = vector
    endpoint_sellers = {
        str(row[field])
        for row in endpoints
        for field in ("seller_uid_left", "seller_uid_right")
    }
    if endpoint_sellers != set(values):
        raise ScientificQualityAuditError("Production numeric endpoint join drift")
    matrix = np.empty((len(endpoints), 16), dtype=np.float64)
    for index, row in enumerate(endpoints):
        left = values[str(row["seller_uid_left"])]
        right = values[str(row["seller_uid_right"])]
        matrix[index] = np.concatenate((np.abs(left - right), left + right))
    names = tuple(
        [f"absdiff__model_visible_{name}" for name in PRODUCTION_NUMERIC_FIELDS]
        + [f"sum__model_visible_{name}" for name in PRODUCTION_NUMERIC_FIELDS]
    )
    return (
        matrix,
        names,
        np.asarray(
            [str(row["canonical_pair_uid"]) for row in endpoints], dtype=object
        ),
        np.asarray([str(row["world_uid"]) for row in endpoints], dtype=object),
    )


def build_text_views(
    *,
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], np.ndarray, np.ndarray]:
    seller_uids = [str(row["seller_uid"]) for row in profiles]
    if len(seller_uids) != len(set(seller_uids)):
        raise ScientificQualityAuditError("Text profile seller UID collision")
    seller_row = {seller_uid: index for index, seller_uid in enumerate(seller_uids)}
    endpoint_sellers = {
        str(row[field])
        for row in endpoints
        for field in ("seller_uid_left", "seller_uid_right")
    }
    if endpoint_sellers != set(seller_uids):
        raise ScientificQualityAuditError("Text profile/endpoint seller join drift")
    texts = {
        field: [str(row[field]) for row in profiles] for field in VISIBLE_PROFILE_FIELDS
    }
    texts["all_fields"] = [
        COMBINED_SEPARATOR.join(texts[field][index] for field in VISIBLE_PROFILE_FIELDS)
        for index in range(len(profiles))
    ]
    text_only_fields = VISIBLE_PROFILE_FIELDS[1:]
    texts["all_text_fields"] = [
        COMBINED_SEPARATOR.join(texts[field][index] for field in text_only_fields)
        for index in range(len(profiles))
    ]
    char = _char_vectorizer()
    word = _word_vectorizer()
    char_cos: dict[str, np.ndarray] = {}
    word_cos: dict[str, np.ndarray] = {}
    for field in (*VISIBLE_PROFILE_FIELDS, "all_fields"):
        char_cos[field] = _pair_cosines(
            char.transform(texts[field]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        word_cos[field] = _pair_cosines(
            word.transform(texts[field]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
    masked_cos: dict[str, np.ndarray] = {}
    for field in (*text_only_fields, "all_text_fields"):
        masked_cos[field] = _pair_cosines(
            char.transform([template_mask(value) for value in texts[field]]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
    surfaces = {
        field: _surface_pair_features(
            texts[field], seller_row=seller_row, endpoints=endpoints
        )
        for field in VISIBLE_PROFILE_FIELDS
    }

    full_columns: list[np.ndarray] = []
    full_names: list[str] = []
    per_field_similarities: list[np.ndarray] = []
    for field in VISIBLE_PROFILE_FIELDS:
        full_columns.extend((char_cos[field], word_cos[field]))
        full_names.extend((f"char3_cosine__{field}", f"word12_cosine__{field}"))
        per_field_similarities.extend((char_cos[field], word_cos[field]))
    full_columns.extend((char_cos["all_fields"], word_cos["all_fields"]))
    full_names.extend(("char3_cosine__all_fields", "word12_cosine__all_fields"))
    for field in VISIBLE_PROFILE_FIELDS:
        full_columns.extend(surfaces[field][:, index] for index in range(len(SURFACE_METRICS)))
        full_names.extend(f"{name}__{field}" for name in SURFACE_METRICS)
    similarity_matrix = np.column_stack(per_field_similarities)
    full_columns.extend(
        (
            np.max(similarity_matrix, axis=1),
            np.mean(similarity_matrix, axis=1),
            np.mean(np.sort(similarity_matrix, axis=1)[:, -2:], axis=1),
        )
    )
    full_names.extend(
        ("similarity_max__field_char_word", "similarity_mean__field_char_word", "similarity_top2_mean__field_char_word")
    )

    topic_columns = [char_cos["category_concat_top"], word_cos["category_concat_top"]]
    topic_names = ["char3_cosine__category_concat_top", "word12_cosine__category_concat_top"]
    topic_columns.extend(
        surfaces["category_concat_top"][:, index] for index in range(len(SURFACE_METRICS))
    )
    topic_names.extend(f"{name}__category_concat_top" for name in SURFACE_METRICS)

    template_columns = [masked_cos[field] for field in (*text_only_fields, "all_text_fields")]
    template_names = [f"masked_char3_cosine__{field}" for field in (*text_only_fields, "all_text_fields")]
    for field in text_only_fields:
        template_columns.extend(surfaces[field][:, index] for index in range(len(SURFACE_METRICS)))
        template_names.extend(f"{name}__{field}" for name in SURFACE_METRICS)
    template_similarity = np.column_stack([masked_cos[field] for field in text_only_fields])
    template_columns.extend(
        (
            np.max(template_similarity, axis=1),
            np.mean(template_similarity, axis=1),
            np.mean(np.sort(template_similarity, axis=1)[:, -2:], axis=1),
        )
    )
    template_names.extend(
        ("masked_similarity_max__text_fields", "masked_similarity_mean__text_fields", "masked_similarity_top2_mean__text_fields")
    )
    views = {
        "p_full": np.column_stack(full_columns),
        "p_topic": np.column_stack(topic_columns),
        "p_template_surface": np.column_stack(template_columns),
    }
    names = {
        "p_full": tuple(full_names),
        "p_topic": tuple(topic_names),
        "p_template_surface": tuple(template_names),
    }
    if {name: matrix.shape[1] for name, matrix in views.items()} != {
        "p_full": 75,
        "p_topic": 14,
        "p_template_surface": 56,
    }:
        raise ScientificQualityAuditError("Frozen text-view width drift")
    pair_uids = np.asarray([str(row["canonical_pair_uid"]) for row in endpoints], dtype=object)
    world_uids = np.asarray([str(row["world_uid"]) for row in endpoints], dtype=object)
    return views, names, pair_uids, world_uids


def _load_persisted_split(
    root: Path,
    split: str,
    *,
    read_truth: bool,
    blind_counters: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    split_root = root / split
    if read_truth and split in {"audit_a", "audit_b"}:
        if blind_counters is not None:
            blind_counters[split]["truth_read_requests"] += 1
            blind_counters[split]["private_payload_open_requests"] += 1
            blind_counters[split]["controller_relation_reads"] += 1
        raise ScientificQualityAuditError(
            "Audit split truth/private payload requested before prediction freeze"
        )
    value = {
        "worlds": _read_jsonl(split_root / "observed" / "worlds.jsonl"),
        "profiles": _read_jsonl(
            split_root / "observed" / "model_seller_profiles.jsonl"
        ),
        "items": _read_jsonl(split_root / "observed" / "redacted_items.jsonl"),
        "endpoints": _read_csv(
            split_root / "observed" / "complete_model_pair_endpoints.csv"
        ),
        "identity33": _read_csv(
            split_root / "observed" / "identity33_all_pairs.csv"
        ),
        "truth_read": read_truth,
    }
    if read_truth:
        value["collision_attempts"] = _read_jsonl(
            split_root / "private" / "document_collision_attempts.jsonl"
        )
        value["identity_allocation"] = _read_jsonl(
            split_root / "private" / "identity_allocation_receipts.jsonl"
        )
        value["world_audit"] = _read_jsonl(
            split_root / "private" / "world_generation_audit.jsonl"
        )
        value["labels"] = _read_csv(split_root / "private" / "pair_labels.csv")
        value["membership"] = _read_jsonl(
            split_root / "private" / "controller_membership.jsonl"
        )
        value["qrels"] = _read_jsonl(split_root / "private" / "qrels.jsonl")
    return value


def _validate_observed_split_isolation(
    *,
    persisted: Mapping[str, Mapping[str, Any]],
    split_manifests: Mapping[str, Mapping[str, Any]],
    root_manifest: Mapping[str, Any],
) -> None:
    """Close all UID classes available in the public model-facing tree."""

    selectors = {
        "world": ("worlds", "world_uid"),
        "seller": ("profiles", "seller_uid"),
        "item": ("items", "item_uid"),
        "pair": ("endpoints", "canonical_pair_uid"),
    }
    global_sets: dict[str, set[str]] = {kind: set() for kind in selectors}
    for split in scientific.SPLITS:
        for kind, (group, field) in selectors.items():
            rows = persisted[split][group]
            values = [str(row[field]) for row in rows]
            value_set = set(values)
            if len(values) != len(value_set):
                raise DatasetInvalidationError(
                    f"Within-split observed {kind} UID reuse: {split}"
                )
            expected = split_manifests[split]["uid_registries"][kind]
            if (
                len(value_set) != int(expected["count"])
                or common.canonical_sha256(sorted(value_set))
                != str(expected["sha256"])
                or global_sets[kind] & value_set
            ):
                raise DatasetInvalidationError(
                    f"Observed split-isolation drift: {split}/{kind}"
                )
            global_sets[kind].update(value_set)
    for kind, values in global_sets.items():
        expected = root_manifest["uid_registries"][kind]
        if (
            len(values) != int(expected["count"])
            or common.canonical_sha256(sorted(values)) != str(expected["sha256"])
        ):
            raise DatasetInvalidationError(
                f"Observed root UID registry drift: {kind}"
            )


def _validate_observed_document_registries(
    *,
    persisted: Mapping[str, Mapping[str, Any]],
    split_manifests: Mapping[str, Mapping[str, Any]],
    root_manifest: Mapping[str, Any],
    historical: collision.HistoricalExclusionRegistries,
) -> None:
    global_items: set[str] = set()
    global_sellers: set[str] = set()
    for split in scientific.SPLITS:
        item_hashes = [
            collision.item_document_hash(
                title=str(row["title"]), description=str(row["description"])
            )
            for row in persisted[split]["items"]
        ]
        seller_hashes = [
            collision.seller_document_hash(row)
            for row in persisted[split]["profiles"]
        ]
        item_set = set(item_hashes)
        seller_set = set(seller_hashes)
        manifest = split_manifests[split]
        if (
            len(item_hashes) != len(item_set)
            or len(seller_hashes) != len(seller_set)
            or item_set & global_items
            or seller_set & global_sellers
            or item_set & historical.item_document_hashes
            or seller_set & historical.seller_document_hashes
            or len(item_set) != int(manifest["item_document_registry_count"])
            or common.canonical_sha256(sorted(item_set))
            != str(manifest["item_document_registry_sha256"])
            or len(seller_set) != int(manifest["seller_document_registry_count"])
            or common.canonical_sha256(sorted(seller_set))
            != str(manifest["seller_document_registry_sha256"])
        ):
            raise DatasetInvalidationError(
                f"Observed split document registry drift: {split}"
            )
        global_items.update(item_set)
        global_sellers.update(seller_set)
    if (
        len(global_items) != int(root_manifest["item_document_registry_count"])
        or common.canonical_sha256(sorted(global_items))
        != str(root_manifest["item_document_registry_sha256"])
        or len(global_sellers)
        != int(root_manifest["seller_document_registry_count"])
        or common.canonical_sha256(sorted(global_sellers))
        != str(root_manifest["seller_document_registry_sha256"])
    ):
        raise DatasetInvalidationError("Observed root document registry drift")


def _assert_json_rows_equal(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if common.canonical_json_bytes(list(expected)) != common.canonical_json_bytes(
        list(observed)
    ):
        raise DatasetInvalidationError(f"Persisted replay drift: {label}")


def _csv_string_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, str]]:
    return [{field: str(row[field]) for field in fields} for row in rows]


def _validate_graph_and_truth(
    *,
    accepted: world_module.AcceptedScientificWorld,
    persisted_labels: Sequence[Mapping[str, str]],
    persisted_membership: Sequence[Mapping[str, Any]],
) -> None:
    endpoints = list(accepted.world["public"]["complete_model_pair_endpoints"])
    sellers = {str(row["seller_uid"]) for row in accepted.world["public"]["sellers"]}
    pairs = {
        tuple(sorted((str(row["seller_uid_left"]), str(row["seller_uid_right"]))))
        for row in endpoints
    }
    if (
        len(sellers) != 28
        or len(endpoints) != 378
        or len(pairs) != 378
        or any(left == right for left, right in pairs)
    ):
        raise DatasetInvalidationError("Complete 28-node graph closure failed")
    membership = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in persisted_membership
    }
    labels = {str(row["canonical_pair_uid"]): int(row["label"]) for row in persisted_labels}
    if set(membership) != sellers or len(labels) != 378:
        raise DatasetInvalidationError("Truth keyset drift")
    positives = 0
    for row in endpoints:
        expected = int(
            membership[str(row["seller_uid_left"])]
            == membership[str(row["seller_uid_right"])]
        )
        if labels[str(row["canonical_pair_uid"])] != expected:
            raise DatasetInvalidationError("Label formula drift")
        positives += expected
    if positives != 20:
        raise DatasetInvalidationError("Per-world positive count drift")


def _mechanism_neutral_exclusions(
    negative_flags: Sequence[Mapping[str, Any]],
    label_index: Mapping[str, int],
) -> set[str]:
    rows = [
        row
        for row in negative_flags
        if str(row["flag"])
        in {"exact_title_clone_target", "high_semantic_similarity_target"}
    ]
    counts = Counter(str(row["flag"]) for row in rows)
    pair_uids = [str(row["canonical_pair_uid"]) for row in rows]
    excluded = set(pair_uids)
    if (
        counts
        != Counter(
            {
                "exact_title_clone_target": 2,
                "high_semantic_similarity_target": 4,
            }
        )
        or len(pair_uids) != 6
        or len(excluded) != 6
        or any(pair_uid not in label_index or label_index[pair_uid] != 0 for pair_uid in excluded)
    ):
        raise DatasetInvalidationError("Mechanism-neutral exclusion drift")
    return excluded


def _private_leak_literals(
    accepted: world_module.AcceptedScientificWorld,
) -> set[str]:
    """Collect forbidden values through the scanner's single shared authority."""

    return blind_literal_scan.collect_complete_world_forbidden_literals(
        world_uid=accepted.world_uid,
        public_sellers=accepted.world["public"]["sellers"],
        public_items=accepted.world["public"]["items"],
        public_pair_endpoints=accepted.world["public"][
            "complete_model_pair_endpoints"
        ],
        qrels=accepted.qrels,
        private_world=accepted.world["private"],
        persisted_private_world_audit=dataset_builder._private_world_audit_row(
            accepted
        ),
    )


def _scan_visible_text(
    *,
    profiles: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
    forbidden_markers: Sequence[str],
    forbidden_literals: Sequence[str],
    failure_domain: str = "persisted_dataset",
) -> int:
    if failure_domain not in {"persisted_dataset", "auditor_counterfactual"}:
        raise ScientificQualityAuditError("Visible-text scan failure domain drift")
    error_type = (
        DatasetInvalidationError
        if failure_domain == "persisted_dataset"
        else ScientificQualityAuditError
    )
    texts = [
        str(row[field]) for row in profiles for field in VISIBLE_PROFILE_FIELDS
    ] + [
        str(row[field]) for row in redacted_items for field in ("title", "description")
    ]
    pattern = re.compile("|".join(re.escape(value) for value in forbidden_markers), re.IGNORECASE)
    for text in texts:
        if blind_literal_scan.contains_private_literal(text, forbidden_literals):
            raise error_type("Private identifier leaked into visible text")
        if pattern.search(text):
            raise error_type("Internal generation marker leaked into text")
        for name, identity_pattern in VISIBLE_IDENTITY_PATTERNS:
            if identity_pattern.search(text):
                raise error_type(
                    f"Visible identity pattern survived redaction: {name}"
                )
    return len(texts)


def _validate_counterfactual_dose(
    *,
    dose: Mapping[str, int],
    counterfactual_audit: Mapping[str, Any],
    dose_contract: Mapping[str, int],
) -> None:
    """Separate renderer invariants from preregistered scientific dose gates."""

    if (
        int(dose["source_seller_changed_count"])
        != int(dose_contract["required_source_seller_changed_count"])
        or counterfactual_audit["original_style_uid_multiset_sha256"]
        != counterfactual_audit["mapped_style_uid_multiset_sha256"]
        or counterfactual_audit["original_style_factor_multiset_sha256"]
        != counterfactual_audit["mapped_style_factor_multiset_sha256"]
    ):
        raise ScientificQualityAuditError(
            "Counterfactual intervention invariant drift"
        )
    if (
        int(dose["effective_style_uid_changed_count"])
        < int(dose_contract["minimum_effective_style_uid_changed_count"])
        or int(dose["effective_style_factor_tuple_changed_count"])
        < int(dose_contract["minimum_effective_style_factor_tuple_changed_count"])
        or int(dose["seller_profile_text_changed_count"])
        < int(dose_contract["minimum_seller_profile_text_changed_count"])
        or int(dose["visible_seller_changed_count"])
        < int(dose_contract["minimum_visible_seller_changed_count"])
        or int(dose["zero_dose_seller_count"])
        > int(dose_contract["maximum_zero_dose_seller_count"])
        or int(dose["zero_visible_dose_seller_count"])
        > int(dose_contract["maximum_zero_visible_dose_seller_count"])
    ):
        raise DatasetInvalidationError(
            "Counterfactual intervention dose is insufficient"
        )


def _blind_visible_numeric_profiles(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Independently derive model-visible statistics from redacted items only."""

    by_seller: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in items:
        by_seller[str(row["seller_uid"])].append(row)

    def clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value)).strip()

    def median(values: Sequence[int]) -> float | int:
        ordered = sorted(values)
        size = len(ordered)
        if size % 2:
            return ordered[size // 2]
        return (ordered[size // 2 - 1] + ordered[size // 2]) / 2

    def rounded_ratio(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else round(numerator / denominator, 6)

    output: dict[str, dict[str, float | int]] = {}
    for seller_uid, seller_items in by_seller.items():
        title_values = [clean_text(row["title"]) for row in seller_items]
        description_values = [
            clean_text(row["description"]) for row in seller_items
        ]
        # The production Step3 profile counts distinct description snippets,
        # not distinct full descriptions.  Reproduce that frozen 280-character
        # projection independently so the blind audit cannot reject a valid
        # profile merely because two descriptions diverge after the snippet.
        description_repeat_values = [
            value
            if len(value) <= BLIND_DESCRIPTION_REPEAT_SNIPPET_LIMIT
            else value[:BLIND_DESCRIPTION_REPEAT_SNIPPET_LIMIT].rstrip()
            for value in description_values
        ]
        digit_ratios: list[float] = []
        punctuation_ratios: list[float] = []
        for row in seller_items:
            combined = str(row["title"]) + " " + str(row["description"])
            visible = [character for character in combined if not character.isspace()]
            digit_ratios.append(
                rounded_ratio(
                    sum(character.isdigit() for character in visible), len(visible)
                )
            )
            punctuation_ratios.append(
                rounded_ratio(
                    sum(not character.isalnum() for character in visible),
                    len(visible),
                )
            )
        item_count = len(seller_items)
        output[seller_uid] = {
            "item_count": item_count,
            "title_length_median": median([len(value) for value in title_values]),
            "description_length_median": median(
                [len(value) for value in description_values]
            ),
            "digit_ratio_mean": round(sum(digit_ratios) / item_count, 6),
            "punct_ratio_mean": round(sum(punctuation_ratios) / item_count, 6),
            "repeated_title_share": round(
                1 - (len({value for value in title_values if value}) / item_count), 6
            ),
            "repeated_description_share": round(
                1
                - (
                    len({value for value in description_repeat_values if value})
                    / item_count
                ),
                6,
            ),
        }
    return output


def _audit_blind_observed_world(
    *,
    policy: Mapping[str, Any],
    record: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    identity33: Sequence[Mapping[str, Any]],
    endpoint_fields: Sequence[str],
    identity_fields: Sequence[str],
    historical: collision.HistoricalExclusionRegistries,
    current_item_hashes: set[str],
    current_seller_hashes: set[str],
) -> int:
    """Audit A/B using observed rows only; never reconstruct private truth."""

    split = str(record["split"])
    world_uid = str(record["world_uid"])
    if split not in {"audit_a", "audit_b"}:
        raise ScientificQualityAuditError("Blind observed audit used outside audit split")
    if len(profiles) != 28 or len(endpoints) != 378 or len(identity33) != 378:
        raise DatasetInvalidationError("Blind audit cardinality drift")
    if any(set(row) != set(dataset_builder.MODEL_PROFILE_FIELDS) for row in profiles):
        raise DatasetInvalidationError("Blind seller-profile schema drift")
    if any(
        common.canonical_json_bytes(
            dataset_builder._project_model_seller_profile(row)
        )
        != common.canonical_json_bytes(row)
        for row in profiles
    ):
        raise DatasetInvalidationError("Blind seller-profile projection drift")
    if any(
        set(row) != set(dataset_builder.MODEL_REDACTED_ITEM_FIELDS) for row in items
    ):
        raise DatasetInvalidationError("Blind redacted-item schema drift")
    if any(
        common.canonical_json_bytes(dataset_builder._project_model_redacted_item(row))
        != common.canonical_json_bytes(row)
        for row in items
    ):
        raise DatasetInvalidationError("Blind redacted-item projection drift")
    if any(tuple(row) != tuple(endpoint_fields) for row in endpoints):
        raise DatasetInvalidationError("Blind endpoint schema/order drift")
    if any(tuple(row) != tuple(identity_fields) for row in identity33):
        raise DatasetInvalidationError("Blind identity33 schema/order drift")

    seller_uids = [str(row["seller_uid"]) for row in profiles]
    item_uids = [str(row["item_uid"]) for row in items]
    if (
        seller_uids != sorted(seller_uids, key=lambda value: value.encode("utf-8"))
        or len(set(seller_uids)) != 28
        or item_uids != sorted(item_uids, key=lambda value: value.encode("utf-8"))
        or len(set(item_uids)) != len(item_uids)
        or {str(row["seller_uid"]) for row in items} != set(seller_uids)
        or any(str(row["world_uid"]) != world_uid for row in items)
    ):
        raise DatasetInvalidationError("Blind observed seller/item closure drift")

    recomputed_numeric = _blind_visible_numeric_profiles(items)
    if set(recomputed_numeric) != set(seller_uids):
        raise DatasetInvalidationError("Blind visible numeric seller closure drift")
    for row in profiles:
        seller_uid = str(row["seller_uid"])
        expected = recomputed_numeric[seller_uid]
        observed = {
            "item_count": row["item_count"],
            "title_length_median": row["title_length_stats"]["median"],
            "description_length_median": row["description_length_stats"]["median"],
            **{
                name: row["style_stats"][name]
                for name in (
                    "digit_ratio_mean",
                    "punct_ratio_mean",
                    "repeated_title_share",
                    "repeated_description_share",
                )
            },
        }
        if observed != expected:
            raise DatasetInvalidationError(
                "Blind model-visible numeric profile replay drift"
            )

    observed_pairs: set[tuple[str, str]] = set()
    pair_uids: list[str] = []
    for row in endpoints:
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        ordered = sorted((left, right), key=lambda value: value.encode("utf-8"))
        pair_uid = str(row["canonical_pair_uid"])
        if (
            left == right
            or [left, right] != ordered
            or left not in seller_uids
            or right not in seller_uids
            or str(row["world_uid"]) != world_uid
            or pair_uid != f"{left}||{right}"
        ):
            raise DatasetInvalidationError("Blind endpoint relation drift")
        observed_pairs.add((left, right))
        pair_uids.append(pair_uid)
    expected_pairs = {
        (left, right)
        for index, left in enumerate(seller_uids)
        for right in seller_uids[index + 1 :]
    }
    if (
        observed_pairs != expected_pairs
        or len(pair_uids) != len(set(pair_uids))
        or pair_uids
        != sorted(pair_uids, key=lambda value: value.encode("utf-8"))
    ):
        raise DatasetInvalidationError("Blind complete-graph closure drift")

    identity_pair_uids: list[str] = []
    for row in identity33:
        if str(row["world_uid"]) != world_uid:
            raise DatasetInvalidationError("Blind identity33 world join drift")
        pair_uid = str(row["canonical_pair_uid"])
        identity_pair_uids.append(pair_uid)
        try:
            values = [float(row[name]) for name in identity_fields[2:]]
        except (TypeError, ValueError) as exc:
            raise DatasetInvalidationError("Blind identity33 numeric drift") from exc
        if not all(math.isfinite(value) for value in values):
            raise DatasetInvalidationError("Blind identity33 nonfinite value")
    if identity_pair_uids != pair_uids:
        raise DatasetInvalidationError("Blind endpoint/identity33 join drift")

    item_hashes = [
        collision.item_document_hash(
            title=str(row["title"]), description=str(row["description"])
        )
        for row in items
    ]
    seller_hashes = [collision.seller_document_hash(row) for row in profiles]
    if (
        len(item_hashes) != len(set(item_hashes))
        or len(seller_hashes) != len(set(seller_hashes))
        or set(item_hashes) & historical.item_document_hashes
        or set(seller_hashes) & historical.seller_document_hashes
        or set(item_hashes) & current_item_hashes
        or set(seller_hashes) & current_seller_hashes
    ):
        raise DatasetInvalidationError("Blind exact-document isolation drift")
    current_item_hashes.update(item_hashes)
    current_seller_hashes.update(seller_hashes)
    forbidden_literals = {
        world_uid,
        *seller_uids,
        *item_uids,
        *pair_uids,
    }
    return _scan_visible_text(
        profiles=profiles,
        redacted_items=items,
        forbidden_markers=policy["row_audit"]["forbidden_markers"],
        forbidden_literals=tuple(forbidden_literals),
    )


def _profile_provenance_delta_receipt(
    *,
    world_uid: str,
    original: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    support_by_seller: Mapping[str, set[str]],
    original_items: Sequence[Mapping[str, Any]],
    counterfactual_items: Sequence[Mapping[str, Any]],
    expected_original_sha256: str,
    expected_counterfactual_sha256: str,
) -> dict[str, Any]:
    wrapper_fields = (
        "contribution_row_count",
        "private_audit_only",
        "profile_count",
        "raw_contribution_values_persisted",
        "rows",
        "rows_sha256",
        "seller_count",
        "version",
        "world_uid",
    )
    role_contract = {
        output_field: (role, source_kind)
        for output_field, _list_field, role, source_kind in candidate_parent.PROFILE_ROLES
    }
    empty_support_digest = candidate_parent._support_digest(())

    def raw_signature_context(
        items: Sequence[Mapping[str, Any]], *, label: str
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, set[str]],
        dict[str, set[str]],
    ]:
        by_uid: dict[str, dict[str, Any]] = {}
        title_sellers: defaultdict[str, set[str]] = defaultdict(set)
        segment_sellers: defaultdict[str, set[str]] = defaultdict(set)
        for source in items:
            row = dataset_builder._project_model_redacted_item(dict(source))
            if tuple(row) != tuple(dataset_builder.MODEL_REDACTED_ITEM_FIELDS):
                raise ScientificQualityAuditError(
                    f"{label} provenance raw-item schema drift"
                )
            item_uid = str(row["item_uid"])
            seller_uid = str(row["seller_uid"])
            if item_uid in by_uid or seller_uid not in support_by_seller:
                raise ScientificQualityAuditError(
                    f"{label} provenance raw-item support drift"
                )
            by_uid[item_uid] = row
            title_norm = candidate_parent.step3.normalize_signature_text(
                candidate_parent.step3.clean_text(str(row["title"]))
            )
            if title_norm:
                title_sellers[title_norm].add(seller_uid)
            for segment in candidate_parent.step3.extract_description_segments(
                str(row["description"])
            ):
                norm = candidate_parent.step3.normalize_signature_text(segment)
                if norm:
                    segment_sellers[norm].add(seller_uid)
        if set(by_uid) != set().union(*support_by_seller.values()):
            raise ScientificQualityAuditError(
                f"{label} provenance raw-item keyset drift"
            )
        return by_uid, dict(title_sellers), dict(segment_sellers)

    raw_context = {
        "Original": raw_signature_context(original_items, label="Original"),
        "Counterfactual": raw_signature_context(
            counterfactual_items,
            label="Counterfactual",
        ),
    }

    def provenance_error(label: str, message: str) -> None:
        error_type = (
            DatasetInvalidationError
            if label == "Original"
            else ScientificQualityAuditError
        )
        raise error_type(f"{label} {message}")

    def exact_int(value: Any, *, minimum: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            owner, detail = label.split(" ", 1)
            provenance_error(owner, f"{detail} integer/range drift")
        return value

    def validate_rows(
        value: Mapping[str, Any], *, label: str
    ) -> list[dict[str, Any]]:
        rows = value.get("rows")
        if (
            tuple(value) != wrapper_fields
            or not isinstance(rows, list)
            or value.get("version")
            != "2026-08-09-step28-v13-v1-13-profile-provenance-v1"
            or str(value.get("world_uid", "")) != world_uid
            or value.get("raw_contribution_values_persisted") is not False
            or value.get("private_audit_only") is not True
            or value.get("seller_count") != len(support_by_seller)
            or value.get("profile_count") != len(support_by_seller)
            or value.get("contribution_row_count") != len(rows)
            or str(value.get("rows_sha256", "")) != common.canonical_sha256(rows)
        ):
            provenance_error(label, "profile provenance wrapper drift")
        output: list[dict[str, Any]] = []
        for source_row in rows:
            if (
                not isinstance(source_row, Mapping)
                or tuple(source_row)
                != tuple(sorted(candidate_parent.PROVENANCE_FIELDS))
            ):
                provenance_error(label, "profile provenance row schema/order drift")
            row = dict(source_row)
            if str(row["world_uid"]) != world_uid:
                provenance_error(label, "profile provenance world binding drift")
            seller_uid = str(row.get("seller_uid", ""))
            output_field = str(row.get("output_field", ""))
            if output_field not in role_contract:
                provenance_error(label, "profile provenance output-field drift")
            expected_role, source_kind = role_contract[output_field]
            if str(row.get("aggregation_role", "")) != expected_role:
                provenance_error(label, "profile provenance aggregation-role drift")
            source_value = row.get("source_item_uids")
            if not isinstance(source_value, list) or any(
                not isinstance(item_uid, str) or not item_uid
                for item_uid in source_value
            ):
                provenance_error(label, "profile provenance source support type drift")
            source_uids = list(source_value)
            output_rank = exact_int(
                row.get("output_rank"), minimum=1, label=f"{label} output_rank"
            )
            source_count = exact_int(
                row.get("source_item_count"),
                minimum=1,
                label=f"{label} source_item_count",
            )
            first_seen = exact_int(
                row.get("first_seen_position"),
                minimum=1,
                label=f"{label} first_seen_position",
            )
            seller_df = exact_int(
                row.get("seller_df"), minimum=0, label=f"{label} seller_df"
            )
            seller_df_count = exact_int(
                row.get("seller_df_seller_count"),
                minimum=0,
                label=f"{label} seller_df_seller_count",
            )
            item_uid = row.get("item_uid")
            ordinal = row.get("extracted_segment_ordinal")
            if (
                seller_uid not in support_by_seller
                or not source_uids
                or len(source_uids) != len(set(source_uids))
                or source_uids != common.utf8_sort(source_uids)
                or not set(source_uids) <= support_by_seller[seller_uid]
                or source_count != len(source_uids)
                or row.get("source_item_uids_sha256")
                != candidate_parent._support_digest(source_uids)
                or not isinstance(item_uid, str)
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or seller_df != seller_df_count
                or seller_df_count > len(support_by_seller)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("seller_df_seller_uids_sha256", ""))
                )
            ):
                provenance_error(label, "profile provenance escaped seller support")
            if source_kind == "description_segment":
                if (
                    not item_uid
                    or item_uid not in source_uids
                    or ordinal < 1
                    or first_seen < 101
                    or first_seen % 100 != ordinal
                ):
                    provenance_error(label, "description-fragment provenance drift")
            elif item_uid or ordinal != -1 or first_seen > len(
                support_by_seller[seller_uid]
            ):
                provenance_error(label, "ordinary contribution provenance drift")
            is_signature = output_field.startswith("signature_")
            if is_signature:
                if seller_df < 1:
                    provenance_error(label, "signature seller-df drift")
                item_by_uid, title_sellers, segment_sellers = raw_context[label]
                if source_kind == "title":
                    norms = {
                        candidate_parent.step3.normalize_signature_text(
                            candidate_parent.step3.clean_text(
                                str(item_by_uid[source_uid]["title"])
                            )
                        )
                        for source_uid in source_uids
                    }
                    norms.discard("")
                    expected_df_sellers = (
                        title_sellers[next(iter(norms))]
                        if len(norms) == 1
                        else set()
                    )
                else:
                    source_item = item_by_uid.get(str(item_uid))
                    segments = (
                        candidate_parent.step3.extract_description_segments(
                            str(source_item["description"])
                        )
                        if source_item is not None
                        else []
                    )
                    expected_norm = (
                        candidate_parent.step3.normalize_signature_text(
                            segments[ordinal - 1]
                        )
                        if 1 <= ordinal <= len(segments)
                        else ""
                    )
                    expected_df_sellers = (
                        segment_sellers.get(expected_norm, set())
                        if expected_norm
                        else set()
                    )
                if (
                    seller_df != len(expected_df_sellers)
                    or seller_df_count != len(expected_df_sellers)
                    or str(row["seller_df_seller_uids_sha256"])
                    != candidate_parent._support_digest(
                        common.utf8_sort(expected_df_sellers)
                    )
                ):
                    provenance_error(
                        label,
                        "signature seller-df independent raw-item replay drift",
                    )
            elif (
                seller_df != 0
                or str(row["seller_df_seller_uids_sha256"])
                != empty_support_digest
            ):
                provenance_error(label, "nonsignature seller-df drift")
            # Force use of the validated integer to make accidental coercion
            # impossible before rank-continuity checks below.
            row["output_rank"] = output_rank
            output.append(row)
        expected_order = sorted(
            output,
            key=lambda row: (
                str(row["seller_uid"]).encode("utf-8"),
                str(row["output_field"]).encode("utf-8"),
                int(row["output_rank"]),
            ),
        )
        if output != expected_order:
            provenance_error(label, "profile provenance row-order drift")
        by_key: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        for row in output:
            by_key[(str(row["seller_uid"]), str(row["output_field"]))].append(
                int(row["output_rank"])
            )
        if set(seller_uid for seller_uid, _field in by_key) != set(
            support_by_seller
        ) or any(ranks != list(range(1, len(ranks) + 1)) for ranks in by_key.values()):
            provenance_error(label, "profile provenance rank continuity drift")
        return output

    if common.canonical_sha256(original) != expected_original_sha256:
        raise DatasetInvalidationError(
            "Original profile provenance independent-replay binding drift"
        )
    if common.canonical_sha256(counterfactual) != expected_counterfactual_sha256:
        raise ScientificQualityAuditError(
            "Counterfactual profile provenance independent-replay binding drift"
        )
    old_rows = validate_rows(original, label="Original")
    new_rows = validate_rows(counterfactual, label="Counterfactual")
    rank_fields = (
        "source_item_uids",
        "source_item_count",
        "first_seen_position",
        "item_uid",
        "extracted_segment_ordinal",
        "seller_df",
        "seller_df_seller_count",
        "seller_df_seller_uids_sha256",
    )

    def rank_index(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> dict[tuple[str, str, int], dict[str, Any]]:
        output: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            key = (
                str(row["seller_uid"]),
                str(row["output_field"]),
                int(row["output_rank"]),
            )
            if key in output:
                provenance_error(label, "profile provenance rank collision")
            output[key] = dict(row)
        return output

    old_rank = rank_index(old_rows, label="Original")
    new_rank = rank_index(new_rows, label="Counterfactual")
    all_fields = sorted(
        {str(row["output_field"]) for row in (*old_rows, *new_rows)},
        key=lambda value: value.encode("utf-8"),
    )
    rows: list[dict[str, Any]] = []
    for output_field in all_fields:
        old_field = [row for row in old_rows if str(row["output_field"]) == output_field]
        new_field = [row for row in new_rows if str(row["output_field"]) == output_field]
        rank_keys = sorted(
            {
                key
                for key in (*old_rank.keys(), *new_rank.keys())
                if key[1] == output_field
            },
            key=lambda value: (value[0].encode("utf-8"), value[2]),
        )
        aligned = [key for key in rank_keys if key in old_rank and key in new_rank]
        changed_aligned = sum(
            any(old_rank[key][name] != new_rank[key][name] for name in rank_fields)
            for key in aligned
        )
        old_support = {
            str(item_uid) for row in old_field for item_uid in row["source_item_uids"]
        }
        new_support = {
            str(item_uid) for row in new_field for item_uid in row["source_item_uids"]
        }
        union = old_support | new_support

        def support_signature(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
            return (
                str(row["seller_uid"]),
                str(row["source_item_uids_sha256"]),
                str(row["item_uid"]),
                int(row["extracted_segment_ordinal"]),
                int(row["source_item_count"]),
            )

        old_support_ranks: defaultdict[tuple[str, str, str, int, int], list[int]] = defaultdict(list)
        new_support_ranks: defaultdict[tuple[str, str, str, int, int], list[int]] = defaultdict(list)
        for row in old_field:
            old_support_ranks[support_signature(row)].append(int(row["output_rank"]))
        for row in new_field:
            new_support_ranks[support_signature(row)].append(int(row["output_rank"]))
        support_aligned = 0
        output_rank_changed = 0
        for signature in set(old_support_ranks) & set(new_support_ranks):
            old_ranks = sorted(old_support_ranks[signature])
            new_ranks = sorted(new_support_ranks[signature])
            count = min(len(old_ranks), len(new_ranks))
            support_aligned += count
            output_rank_changed += sum(
                left != right
                for left, right in zip(old_ranks[:count], new_ranks[:count], strict=True)
            )
        rows.append(
            {
                "output_field": output_field,
                "original_contribution_row_count": len(old_field),
                "counterfactual_contribution_row_count": len(new_field),
                "rank_aligned_slot_count": len(aligned),
                "changed_rank_aligned_slot_count": changed_aligned,
                "source_item_uid_changed_count": sum(
                    old_rank[key]["source_item_uids"]
                    != new_rank[key]["source_item_uids"]
                    for key in aligned
                ),
                "fragment_ordinal_changed_count": sum(
                    old_rank[key]["extracted_segment_ordinal"]
                    != new_rank[key]["extracted_segment_ordinal"]
                    for key in aligned
                ),
                "first_position_changed_count": sum(
                    old_rank[key]["first_seen_position"]
                    != new_rank[key]["first_seen_position"]
                    for key in aligned
                ),
                "seller_df_changed_count": sum(
                    any(
                        old_rank[key][name] != new_rank[key][name]
                        for name in (
                            "seller_df",
                            "seller_df_seller_count",
                            "seller_df_seller_uids_sha256",
                        )
                    )
                    for key in aligned
                ),
                "support_aligned_slot_count": support_aligned,
                "output_rank_changed_count": output_rank_changed,
                "source_item_support_jaccard": (
                    len(old_support & new_support) / len(union) if union else 1.0
                ),
            }
        )
    receipt = {
        "version": "2026-08-11-step28-v13-v1-13-profile-provenance-delta-v1",
        "world_uid": world_uid,
        "original_provenance_sha256": common.canonical_sha256(original),
        "counterfactual_provenance_sha256": common.canonical_sha256(counterfactual),
        "independent_replay_binding_verified": True,
        "original_contribution_row_count": len(old_rows),
        "counterfactual_contribution_row_count": len(new_rows),
        "rows": rows,
        "rows_sha256": common.canonical_sha256(rows),
        "used_for_mapping_candidate_or_world_selection": False,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def _validate_three_path_alignment_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != {
        "version",
        "common_binding",
        "fixed_support",
        "production_step3",
        "joint_visible_input",
        "numeric_projection",
        "dose",
        "canonical_self_hash",
    } or receipt.get("canonical_self_hash") != _canonical_self_hash(receipt):
        raise ScientificQualityAuditError("Three-path receipt schema/self-hash drift")
    common_binding = receipt["common_binding"]
    paths = (
        receipt["fixed_support"],
        receipt["production_step3"],
        receipt["joint_visible_input"],
    )
    if not isinstance(common_binding, Mapping) or any(
        not isinstance(path, Mapping) for path in paths
    ):
        raise ScientificQualityAuditError("Three-path receipt object drift")
    for name, value in common_binding.items():
        if any(path.get(name) != value for path in paths):
            raise ScientificQualityAuditError(
                f"Three-path receipt alignment drift: {name}"
            )
    for name in (
        "pair_order_sha256",
        "world_order_sha256",
        "eligible_mask_sha256",
        "eligible_pair_uids_sha256",
        "dose_sha256",
    ):
        observed = [path.get(name) for path in paths]
        if any(value is None for value in observed) or len(set(observed)) != 1:
            raise ScientificQualityAuditError(
                f"Three-path independently derived alignment drift: {name}"
            )
    if (
        receipt["joint_visible_input"].get("production_binding_sha256")
        != common.canonical_sha256(receipt["production_step3"])
        or receipt["joint_visible_input"].get("fixed_support_binding_sha256")
        != common.canonical_sha256(receipt["fixed_support"])
        or receipt["joint_visible_input"].get("numeric_projection_sha256")
        != common.canonical_sha256(receipt["numeric_projection"])
        or receipt["dose"].get("world_uid") != common_binding.get("world_uid")
        or any(
            path.get("dose_sha256") != common.canonical_sha256(receipt["dose"])
            for path in paths
        )
    ):
        raise ScientificQualityAuditError("Three-path receipt hash/dose drift")


def _world_three_path_alignment_receipt(
    *,
    split: str,
    split_ordinal: int,
    world_uid: str,
    target_source_pairs: Sequence[Sequence[str]],
    expected_mapping_sha256: str,
    fixed_support: Mapping[str, Any],
    provenance_delta: Mapping[str, Any],
    original_profiles: Sequence[Mapping[str, Any]],
    counterfactual_profiles: Sequence[Mapping[str, Any]],
    original_items: Sequence[Mapping[str, Any]],
    counterfactual_items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    excluded_pair_uids: set[str],
    dose: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = tuple((str(row[0]), str(row[1])) for row in target_source_pairs)
    sellers = {str(row["seller_uid"]) for row in original_profiles}
    if (
        len(pairs) != 28
        or {target for target, _source in pairs} != sellers
        or {source for _target, source in pairs} != sellers
        or any(target == source for target, source in pairs)
    ):
        raise ScientificQualityAuditError("Three-path derangement closure drift")
    mapping_sha256 = style_derangement._mapping_digest(pairs)
    if mapping_sha256 != expected_mapping_sha256:
        raise ScientificQualityAuditError("Three-path mapping digest drift")
    _p_original, p_names, p_pairs, p_worlds = build_text_views(
        profiles=original_profiles,
        endpoints=endpoints,
    )
    _p_counterfactual, cf_p_names, cf_p_pairs, cf_p_worlds = build_text_views(
        profiles=counterfactual_profiles,
        endpoints=endpoints,
    )
    _f_original, f_names, f_pairs, f_worlds = build_fixed_support_text_views(
        items=original_items,
        endpoints=endpoints,
    )
    _f_counterfactual, cf_f_names, cf_f_pairs, cf_f_worlds = (
        build_fixed_support_text_views(
            items=counterfactual_items,
            endpoints=endpoints,
        )
    )
    original_numeric, numeric_names, numeric_pairs, numeric_worlds = (
        build_production_numeric_matrix(
            profiles=original_profiles, endpoints=endpoints
        )
    )
    counterfactual_numeric, cf_numeric_names, cf_pairs, cf_worlds = (
        build_production_numeric_matrix(
            profiles=counterfactual_profiles, endpoints=endpoints
        )
    )
    if (
        p_names != cf_p_names
        or f_names != cf_f_names
        or numeric_names != cf_numeric_names
        or not np.array_equal(p_pairs, cf_p_pairs)
        or not np.array_equal(p_worlds, cf_p_worlds)
        or not np.array_equal(f_pairs, cf_f_pairs)
        or not np.array_equal(f_worlds, cf_f_worlds)
        or not np.array_equal(numeric_pairs, cf_pairs)
        or not np.array_equal(numeric_worlds, cf_worlds)
    ):
        raise ScientificQualityAuditError("Three-path numeric projection join drift")

    def path_order_binding(
        pair_values: np.ndarray,
        world_values: np.ndarray,
    ) -> dict[str, str]:
        path_pairs = [str(value) for value in pair_values.tolist()]
        path_worlds = [str(value) for value in world_values.tolist()]
        if (
            len(path_pairs) != 378
            or len(set(path_pairs)) != 378
            or set(path_worlds) != {world_uid}
            or len(excluded_pair_uids) != 6
        ):
            raise ScientificQualityAuditError(
                "Three-path pair/world boundary drift"
            )
        path_mask = [
            pair_uid not in excluded_pair_uids for pair_uid in path_pairs
        ]
        if sum(path_mask) != 372:
            raise ScientificQualityAuditError(
                "Three-path mask cardinality drift"
            )
        return {
            "pair_order_sha256": common.canonical_sha256(path_pairs),
            "world_order_sha256": common.canonical_sha256(path_worlds),
            "eligible_mask_sha256": common.canonical_sha256(path_mask),
            "eligible_pair_uids_sha256": common.canonical_sha256(
                [
                    pair_uid
                    for pair_uid, keep in zip(
                        path_pairs,
                        path_mask,
                        strict=True,
                    )
                    if keep
                ]
            ),
        }

    production_order = path_order_binding(p_pairs, p_worlds)
    fixed_order = path_order_binding(f_pairs, f_worlds)
    numeric_order = path_order_binding(numeric_pairs, numeric_worlds)
    if not (
        np.array_equal(p_pairs, f_pairs)
        and np.array_equal(p_worlds, f_worlds)
        and np.array_equal(p_pairs, numeric_pairs)
        and np.array_equal(p_worlds, numeric_worlds)
    ):
        raise ScientificQualityAuditError(
            "Three-path independently generated pair/world arrays drift"
        )
    common_binding = {
        "split": split,
        "split_ordinal": split_ordinal,
        "world_uid": world_uid,
        "mapping_sha256": mapping_sha256,
        "target_source_pairs_sha256": common.canonical_sha256(
            [list(row) for row in pairs]
        ),
    }
    fixed_binding = {
        **common_binding,
        **fixed_order,
        "dose_sha256": common.canonical_sha256(dict(dose)),
        "slot_keyset_sha256": str(fixed_support["slot_keyset_sha256"]),
        "original_items_sha256": common.canonical_sha256(list(original_items)),
        "counterfactual_items_sha256": common.canonical_sha256(
            list(counterfactual_items)
        ),
    }
    production_binding = {
        **common_binding,
        **production_order,
        "dose_sha256": common.canonical_sha256(dict(dose)),
        "original_provenance_sha256": str(
            provenance_delta["original_provenance_sha256"]
        ),
        "counterfactual_provenance_sha256": str(
            provenance_delta["counterfactual_provenance_sha256"]
        ),
        "original_profiles_sha256": common.canonical_sha256(
            list(original_profiles)
        ),
        "counterfactual_profiles_sha256": common.canonical_sha256(
            list(counterfactual_profiles)
        ),
    }
    numeric_projection = {
        "feature_names_sha256": common.canonical_sha256(list(numeric_names)),
        "original_matrix_sha256": common.canonical_sha256(
            original_numeric.tolist()
        ),
        "counterfactual_matrix_sha256": common.canonical_sha256(
            counterfactual_numeric.tolist()
        ),
    }
    joint_binding = {
        **common_binding,
        **numeric_order,
        "dose_sha256": common.canonical_sha256(dict(dose)),
        "production_binding_sha256": common.canonical_sha256(production_binding),
        "fixed_support_binding_sha256": common.canonical_sha256(fixed_binding),
        "numeric_projection_sha256": common.canonical_sha256(numeric_projection),
    }
    for name in common_binding:
        if not (
            fixed_binding[name]
            == production_binding[name]
            == joint_binding[name]
            == common_binding[name]
        ):
            raise ScientificQualityAuditError(
                f"Three-path common binding mismatch: {name}"
            )
    receipt = {
        "version": "2026-08-11-step28-v13-v1-13-three-path-alignment-v1",
        "common_binding": common_binding,
        "fixed_support": fixed_binding,
        "production_step3": production_binding,
        "joint_visible_input": joint_binding,
        "numeric_projection": numeric_projection,
        "dose": copy.deepcopy(dict(dose)),
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
    _validate_three_path_alignment_receipt(receipt)
    return receipt


def _recompute_counterfactual_identity33(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    template: Mapping[str, Any],
    accepted: world_module.AcceptedScientificWorld,
    counterfactual: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    world = copy.deepcopy(accepted.world)

    def invariant_projection(value: Mapping[str, Any]) -> dict[str, Any]:
        public = value["public"]
        private = value["private"]
        return {
            "sellers": copy.deepcopy(public["sellers"]),
            "complete_model_pair_endpoints": copy.deepcopy(
                public["complete_model_pair_endpoints"]
            ),
            "item_nontext": [
                {
                    name: field_value
                    for name, field_value in row.items()
                    if name not in {"title", "description"}
                }
                for row in public["items"]
            ],
            "private_nonintervened": {
                name: copy.deepcopy(field_value)
                for name, field_value in private.items()
                if name
                not in {
                    "render_asts",
                    "identity_slots_audit",
                    "noise_slots_audit",
                }
            },
        }

    original_invariant = invariant_projection(accepted.world)
    world["public"]["items"] = copy.deepcopy(counterfactual["public"]["raw_items"])
    world["private"]["render_asts"] = copy.deepcopy(
        counterfactual["private"]["render_asts"]
    )
    world["private"]["identity_slots_audit"] = copy.deepcopy(
        counterfactual["private"]["identity_slots_audit"]
    )
    world["private"]["noise_slots_audit"] = copy.deepcopy(
        counterfactual["private"]["noise_slots_audit"]
    )
    if common.canonical_json_bytes(original_invariant) != common.canonical_json_bytes(
        invariant_projection(world)
    ):
        raise ScientificQualityAuditError(
            "Counterfactual changed a non-style world invariant"
        )
    original_profiles, original_provenance, original_identity33, original_redacted = (
        world_module._build_profiles_and_identity33(
            policy=policy,
            mode=mode,
            split=split,
            template=template,
            world=accepted.world,
        )
    )
    if (
        common.canonical_json_bytes(original_profiles)
        != common.canonical_json_bytes(accepted.seller_profiles)
        or common.canonical_json_bytes(original_redacted)
        != common.canonical_json_bytes(accepted.redacted_items)
        or common.canonical_json_bytes(original_identity33)
        != common.canonical_json_bytes(accepted.identity33)
        or common.canonical_sha256(original_provenance)
        != accepted.profile_provenance_sha256
    ):
        raise ScientificQualityAuditError("Original production replay drift")
    profiles, provenance, identity33, redacted = (
        world_module._build_profiles_and_identity33(
            policy=policy,
            mode=mode,
            split=split,
            template=template,
            world=world,
        )
    )
    repeated_profiles, repeated_provenance, repeated_identity33, repeated_redacted = (
        world_module._build_profiles_and_identity33(
            policy=policy,
            mode=mode,
            split=split,
            template=template,
            world=copy.deepcopy(world),
        )
    )
    if any(
        common.canonical_json_bytes(left) != common.canonical_json_bytes(right)
        for left, right in (
            (profiles, repeated_profiles),
            (provenance, repeated_provenance),
            (identity33, repeated_identity33),
            (redacted, repeated_redacted),
        )
    ):
        raise ScientificQualityAuditError(
            "Counterfactual production replay is nondeterministic"
        )
    if common.canonical_json_bytes(identity33) != common.canonical_json_bytes(
        accepted.identity33
    ):
        raise ScientificQualityAuditError("Counterfactual changed identity33")
    counterfactual_profiles = sorted(
        counterfactual["public"]["seller_profiles"],
        key=lambda row: str(row["seller_uid"]).encode("utf-8"),
    )
    counterfactual_redacted = sorted(
        counterfactual["public"]["redacted_items"],
        key=lambda row: str(row["item_uid"]).encode("utf-8"),
    )
    if common.canonical_json_bytes(profiles) != common.canonical_json_bytes(
        counterfactual_profiles
    ) or common.canonical_json_bytes(redacted) != common.canonical_json_bytes(
        counterfactual_redacted
    ):
        raise ScientificQualityAuditError("Counterfactual production replay drift")
    support_by_seller: defaultdict[str, set[str]] = defaultdict(set)
    for row in accepted.world["public"]["items"]:
        support_by_seller[str(row["seller_uid"])].add(str(row["item_uid"]))
    delta = _profile_provenance_delta_receipt(
        world_uid=accepted.world_uid,
        original=original_provenance,
        counterfactual=provenance,
        support_by_seller=support_by_seller,
        original_items=original_redacted,
        counterfactual_items=redacted,
        expected_original_sha256=accepted.profile_provenance_sha256,
        expected_counterfactual_sha256=common.canonical_sha256(
            repeated_provenance
        ),
    )
    repeated_delta = _profile_provenance_delta_receipt(
        world_uid=accepted.world_uid,
        original=original_provenance,
        counterfactual=repeated_provenance,
        support_by_seller=support_by_seller,
        original_items=original_redacted,
        counterfactual_items=repeated_redacted,
        expected_original_sha256=accepted.profile_provenance_sha256,
        expected_counterfactual_sha256=common.canonical_sha256(provenance),
    )
    if common.canonical_json_bytes(delta) != common.canonical_json_bytes(
        repeated_delta
    ):
        raise ScientificQualityAuditError("Profile provenance delta replay drift")
    return list(profiles), list(redacted), dict(provenance), delta


def replay_and_audit_worlds(
    policy: Mapping[str, Any],
    *,
    blind_counters: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    builder_policy = scientific.load_policy()
    context = scientific.build_execution_context(
        builder_policy, execution_mode=str(policy["input"]["execution_mode"])
    )
    if context.output_root != common.repo_path(str(policy["input"]["dataset_root"])):
        raise ScientificQualityAuditError("Quality input root drift")
    root_manifest = common.load_json(context.output_root / "root_manifest.json")
    if (
        common.sha256_file(context.output_root / "root_manifest.json")
        != policy["pins"]["dataset_root_manifest"]["sha256"]
        or root_manifest.get("canonical_self_hash")
        != policy["input"]["root_manifest_canonical_self_hash"]
    ):
        raise DatasetInvalidationError("Input root manifest drift")
    split_manifests = _verify_quality_input_tree(context.output_root, root_manifest)
    if blind_counters is None:
        blind_counters = _new_blind_boundary_counters()
    sealed_registry_isolation_receipt = _run_blind_registry_isolation_scan(
        policy=policy,
        dataset_root=context.output_root,
        root_manifest=root_manifest,
        split_manifests=split_manifests,
        blind_counters=blind_counters,
    )
    sealed_scan_receipts = {
        split: _run_blind_literal_scan(
            policy=policy,
            dataset_root=context.output_root,
            split=split,
            split_manifest=split_manifests[split],
            blind_counters=blind_counters,
        )
        for split in ("audit_a", "audit_b")
    }
    persisted = {
        split: _load_persisted_split(
            context.output_root,
            split,
            read_truth=split in {"train", "development"},
            blind_counters=blind_counters,
        )
        for split in scientific.SPLITS
    }
    _validate_observed_split_isolation(
        persisted=persisted,
        split_manifests=split_manifests,
        root_manifest=root_manifest,
    )
    template, fixture, style_profile = scientific.load_release_inputs(context)
    historical = collision.load_historical_exclusion_registries()
    _validate_observed_document_registries(
        persisted=persisted,
        split_manifests=split_manifests,
        root_manifest=root_manifest,
        historical=historical,
    )
    current_item_hashes: set[str] = set()
    current_seller_hashes: set[str] = set()
    current_identity_hashes: set[str] = set()
    privileged_registry_sets = {
        split: {"controller": set(), "query": set(), "identity": set()}
        for split in ("train", "development")
    }
    bundles: dict[str, dict[str, Any]] = {
        split: {
            "original_profiles": [],
            "counterfactual_profiles": [],
            "original_items": [],
            "counterfactual_items": [],
            "observed_items": [],
            "source_items": [],
            "endpoints": [],
            "labels": [],
            "eligible_pair_uids": set(),
            "ordered_world_uids": [],
        }
        for split in ("train", "development")
    }
    persisted_groups = {}
    for split, data in persisted.items():
        item_groups = _group(data["items"], "world_uid")
        seller_world = {
            str(row["seller_uid"]): world_uid
            for world_uid, rows in item_groups.items()
            for row in rows
        }
        if len(seller_world) != len(data["profiles"]):
            raise DatasetInvalidationError("Persisted seller/world join drift")
        profile_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in data["profiles"]:
            seller_uid = str(row["seller_uid"])
            if seller_uid not in seller_world:
                raise DatasetInvalidationError("Profile lacks persisted world join")
            profile_groups[seller_world[seller_uid]].append(dict(row))
        grouped = {
            "profiles": dict(profile_groups),
            "items": item_groups,
            **{
                name: _group(data[name], "world_uid")
                for name in (
                    "worlds",
                    "endpoints",
                    "identity33",
                )
            },
        }
        if data["truth_read"]:
            for name in (
                "collision_attempts",
                "identity_allocation",
                "world_audit",
                "labels",
                "membership",
                "qrels",
            ):
                grouped[name] = _group(data[name], "world_uid")
        persisted_groups[split] = grouped
    records = sorted(
        context.world_records,
        key=lambda row: (
            scientific.SPLITS.index(str(row["split"])),
            int(row["split_ordinal"]),
        ),
    )
    scanned_text_count = 0
    changed_text_counts: Counter[str] = Counter()
    mapping_hashes: list[str] = []
    path_alignment_receipts: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    fixed_support_receipts: list[dict[str, Any]] = []
    provenance_delta_receipts: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        split = str(record["split"])
        world_uid = str(record["world_uid"])
        groups = persisted_groups[split]
        required_groups = ("worlds", "profiles", "items", "endpoints", "identity33")
        if any(world_uid not in groups[name] for name in required_groups):
                raise DatasetInvalidationError(
                    f"Persisted observed world rowset missing: {split}/{world_uid}"
                )
        _assert_json_rows_equal(
            [{"world_uid": world_uid, "split_ordinal": int(record["split_ordinal"])}],
            groups["worlds"][world_uid],
            label=f"{split}/world/{world_uid}",
        )
        endpoint_fields = tuple(
            context.effective_policy["relational_integrity"]
            ["pair_projection_contract"]["complete_model_pair_endpoints_schema"]
        )
        identity_fields = (
            "canonical_pair_uid",
            "world_uid",
            *tuple(context.effective_policy["history_features"]["feature_names"]),
        )
        if split in {"audit_a", "audit_b"}:
            if persisted[split]["truth_read"] is not False:
                raise ScientificQualityAuditError(
                    "Audit truth was read before prediction freeze"
                )
            scanned_text_count += _audit_blind_observed_world(
                policy=policy,
                record=record,
                profiles=groups["profiles"][world_uid],
                items=groups["items"][world_uid],
                endpoints=groups["endpoints"][world_uid],
                identity33=groups["identity33"][world_uid],
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
                historical=historical,
                current_item_hashes=current_item_hashes,
                current_seller_hashes=current_seller_hashes,
            )
            if position % 10 == 0 or position == len(records):
                print(
                    json.dumps(
                        {
                            "event": "quality_replay_progress",
                            "worlds_complete": position,
                            "worlds_total": len(records),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
            )
            continue
        if split not in {"train", "development"}:
            raise ScientificQualityAuditError("Unknown split reached privileged replay")
        _reject_blind_privileged_replay(split, blind_counters)
        accepted = world_module.build_scientific_world(
            policy=context.effective_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode=context.base_mode,
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                context.effective_policy, mode=context.base_mode, split=split
            ),
            document_variation_key=context.document_variation_key,
            anonymous_handle_key=context.anonymous_handle_key,
            historical_item_hashes=historical.item_document_hashes,
            historical_seller_hashes=historical.seller_document_hashes,
            historical_identity_hashes=historical.identity_value_hashes,
            current_item_hashes=current_item_hashes,
            current_seller_hashes=current_seller_hashes,
            current_identity_hashes=current_identity_hashes,
        )
        privileged_registry_sets[split]["controller"].update(
            str(row["controller_uid"]) for row in accepted.controller_membership
        )
        privileged_registry_sets[split]["query"].update(
            str(row["query_uid"]) for row in accepted.qrels
        )
        privileged_registry_sets[split]["identity"].update(
            accepted.identity_registry_delta
        )
        expected_profiles = [
            dataset_builder._project_model_seller_profile(row)
            for row in sorted(
                accepted.seller_profiles,
                key=lambda value: str(value["seller_uid"]).encode("utf-8"),
            )
        ]
        expected_items = [
            dataset_builder._project_model_redacted_item(row)
            for row in sorted(
                accepted.redacted_items,
                key=lambda value: str(value["item_uid"]).encode("utf-8"),
            )
        ]
        _assert_json_rows_equal(expected_profiles, groups["profiles"][world_uid], label=f"{split}/profiles/{world_uid}")
        _assert_json_rows_equal(expected_items, groups["items"][world_uid], label=f"{split}/items/{world_uid}")
        expected_endpoints = _csv_string_rows(
            sorted(
                accepted.world["public"]["complete_model_pair_endpoints"],
                key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"),
            ),
            endpoint_fields,
        )
        if expected_endpoints != groups["endpoints"][world_uid]:
            raise DatasetInvalidationError("Persisted endpoint replay drift")
        if _csv_string_rows(accepted.identity33, identity_fields) != groups["identity33"][world_uid]:
            raise DatasetInvalidationError("Persisted identity33 replay drift")
        expected_collision = {
            "world_uid": accepted.world_uid,
            "split": accepted.split,
            "split_ordinal": accepted.split_ordinal,
            "accepted_candidate_index": accepted.candidate_index,
            "candidates_examined": accepted.candidates_examined,
            "rejection_counts": accepted.rejection_counts,
            "item_registry_delta_count": len(accepted.item_registry_delta),
            "item_registry_delta_sha256": common.canonical_sha256(
                accepted.item_registry_delta
            ),
            "seller_registry_delta_count": len(accepted.seller_registry_delta),
            "seller_registry_delta_sha256": common.canonical_sha256(
                accepted.seller_registry_delta
            ),
            "natural_output_sha256": accepted.natural_output_sha256,
        }
        _assert_json_rows_equal(
            [expected_collision],
            groups["collision_attempts"][world_uid],
            label=f"{split}/collision/{world_uid}",
        )
        _assert_json_rows_equal(
            [
                {
                    "world_uid": accepted.world_uid,
                    "split": accepted.split,
                    "split_ordinal": accepted.split_ordinal,
                    "identity_registry_delta_count": len(
                        accepted.identity_registry_delta
                    ),
                    "identity_registry_delta_sha256": common.canonical_sha256(
                        accepted.identity_registry_delta
                    ),
                    "receipt": accepted.identity_allocation_receipt,
                }
            ],
            groups["identity_allocation"][world_uid],
            label=f"{split}/identity-allocation/{world_uid}",
        )
        _assert_json_rows_equal(
            [dataset_builder._private_world_audit_row(accepted)],
            groups["world_audit"][world_uid],
            label=f"{split}/private-world-audit/{world_uid}",
        )
        forbidden_literals = _private_leak_literals(accepted)
        scanned_text_count += _scan_visible_text(
            profiles=expected_profiles,
            redacted_items=expected_items,
            forbidden_markers=policy["row_audit"]["forbidden_markers"],
            forbidden_literals=tuple(forbidden_literals),
        )
        if split in {"train", "development"}:
            _validate_graph_and_truth(
                accepted=accepted,
                persisted_labels=groups["labels"][world_uid],
                persisted_membership=groups["membership"][world_uid],
            )
            _assert_json_rows_equal(
                accepted.controller_membership,
                groups["membership"][world_uid],
                label=f"{split}/controller-membership/{world_uid}",
            )
            _assert_json_rows_equal(
                _csv_string_rows(
                    accepted.pair_labels,
                    ("canonical_pair_uid", "world_uid", "label"),
                ),
                groups["labels"][world_uid],
                label=f"{split}/pair-labels/{world_uid}",
            )
            _assert_json_rows_equal(
                accepted.qrels,
                groups["qrels"][world_uid],
                label=f"{split}/qrels/{world_uid}",
            )
            counterfactual = counterfactual_text.rerender_counterfactual_world(
                context.effective_policy,
                mode=context.base_mode,
                split=split,
                template=template,
                sellers=accepted.world["public"]["sellers"],
                items=accepted.world["public"]["items"],
                identity_slots_audit=accepted.world["private"]["identity_slots_audit"],
                noise_slots_audit=accepted.world["private"]["noise_slots_audit"],
                render_asts=accepted.world["private"]["render_asts"],
                override_audit=accepted.world["private"]["override_audit"],
            )
            cf_profiles, cf_redacted, _cf_provenance, provenance_delta = (
                _recompute_counterfactual_identity33(
                policy=context.effective_policy,
                mode=context.base_mode,
                split=split,
                template=template,
                accepted=accepted,
                counterfactual=counterfactual,
                )
            )
            projected_cf_profiles = [
                dataset_builder._project_model_seller_profile(row)
                for row in cf_profiles
            ]
            projected_cf_items = [
                dataset_builder._project_model_redacted_item(row)
                for row in cf_redacted
            ]
            fixed_support = _fixed_support_slot_contract(
                split=split,
                world_uid=world_uid,
                original_items=expected_items,
                counterfactual_items=projected_cf_items,
            )
            if (
                int(fixed_support["title_slot_changed_count"])
                != int(counterfactual["audit"]["changed_title_count"])
                or int(fixed_support["description_slot_changed_count"])
                != int(counterfactual["audit"]["changed_description_count"])
                or int(fixed_support["visible_seller_changed_count"])
                != int(counterfactual["audit"]["visible_seller_changed_count"])
            ):
                raise ScientificQualityAuditError(
                    "Counterfactual fixed-support dose replay drift"
                )
            original_by_seller = {
                str(row["seller_uid"]): row for row in expected_profiles
            }
            cf_by_seller = {
                str(row["seller_uid"]): row for row in projected_cf_profiles
            }
            if set(original_by_seller) != set(cf_by_seller):
                raise ScientificQualityAuditError(
                    "Counterfactual seller-profile keyset drift"
                )
            profile_text_changed_count = sum(
                any(
                    str(original_by_seller[seller_uid][field])
                    != str(cf_by_seller[seller_uid][field])
                    for field in VISIBLE_PROFILE_FIELDS
                )
                for seller_uid in original_by_seller
            )
            dose = {
                "world_uid": world_uid,
                "source_seller_changed_count": int(
                    counterfactual["audit"]["source_seller_changed_count"]
                ),
                "effective_style_uid_changed_count": int(
                    counterfactual["audit"]["effective_style_uid_changed_count"]
                ),
                "effective_style_factor_tuple_changed_count": int(
                    counterfactual["audit"]
                    ["effective_style_factor_tuple_changed_count"]
                ),
                "seller_profile_text_changed_count": profile_text_changed_count,
                "visible_title_slot_changed_count": int(
                    fixed_support["title_slot_changed_count"]
                ),
                "visible_description_slot_changed_count": int(
                    fixed_support["description_slot_changed_count"]
                ),
                "visible_seller_changed_count": int(
                    fixed_support["visible_seller_changed_count"]
                ),
                "zero_dose_seller_count": int(
                    counterfactual["audit"]["zero_dose_seller_count"]
                ),
                "zero_visible_dose_seller_count": int(
                    counterfactual["audit"]["zero_visible_dose_seller_count"]
                ),
                "production_profile_slot_changed_count": sum(
                    int(row["changed_rank_aligned_slot_count"])
                    for row in provenance_delta["rows"]
                ),
                "production_profile_support_changed_count": sum(
                    int(row["source_item_uid_changed_count"])
                    for row in provenance_delta["rows"]
                ),
            }
            dose_contract = policy["text_counterfactual"]["intervention_dose"]
            _validate_counterfactual_dose(
                dose=dose,
                counterfactual_audit=counterfactual["audit"],
                dose_contract=dose_contract,
            )
            dose_rows.append(dose)
            fixed_support_receipts.append(fixed_support)
            provenance_delta_receipts.append(provenance_delta)
            scanned_text_count += _scan_visible_text(
                profiles=projected_cf_profiles,
                redacted_items=projected_cf_items,
                forbidden_markers=policy["row_audit"]["forbidden_markers"],
                forbidden_literals=tuple(forbidden_literals),
                failure_domain="auditor_counterfactual",
            )
            bundle = bundles[split]
            bundle["ordered_world_uids"].append(world_uid)
            bundle["original_profiles"].extend(expected_profiles)
            bundle["counterfactual_profiles"].extend(projected_cf_profiles)
            bundle["original_items"].extend(copy.deepcopy(expected_items))
            bundle["counterfactual_items"].extend(
                copy.deepcopy(projected_cf_items)
            )
            # Missingness and time-bucket shortcuts must be measured on the
            # exact model-visible item projection, not the private pre-redaction
            # source rows used by the counterfactual renderer.
            bundle["observed_items"].extend(copy.deepcopy(expected_items))
            bundle["source_items"].extend(
                copy.deepcopy(accepted.world["public"]["items"])
            )
            bundle["endpoints"].extend(copy.deepcopy(expected_endpoints))
            label_index = {
                str(row["canonical_pair_uid"]): int(row["label"])
                for row in groups["labels"][world_uid]
            }
            bundle["labels"].extend(
                label_index[str(row["canonical_pair_uid"])] for row in expected_endpoints
            )
            excluded = _mechanism_neutral_exclusions(
                accepted.world["private"]["negative_flags"], label_index
            )
            path_alignment_receipts.append(
                _world_three_path_alignment_receipt(
                    split=split,
                    split_ordinal=int(record["split_ordinal"]),
                    world_uid=world_uid,
                    target_source_pairs=counterfactual["audit"][
                        "target_source_pairs"
                    ],
                    expected_mapping_sha256=str(
                        counterfactual["audit"]["mapping_sha256"]
                    ),
                    fixed_support=fixed_support,
                    provenance_delta=provenance_delta,
                    original_profiles=expected_profiles,
                    counterfactual_profiles=projected_cf_profiles,
                    original_items=expected_items,
                    counterfactual_items=projected_cf_items,
                    endpoints=expected_endpoints,
                    excluded_pair_uids=excluded,
                    dose=dose,
                )
            )
            bundle["eligible_pair_uids"].update(
                str(row["canonical_pair_uid"])
                for row in expected_endpoints
                if str(row["canonical_pair_uid"]) not in excluded
            )
            changed_text_counts["titles"] += int(counterfactual["audit"]["changed_title_count"])
            changed_text_counts["descriptions"] += int(counterfactual["audit"]["changed_description_count"])
            changed_text_counts["style_source_changed"] += dose[
                "source_seller_changed_count"
            ]
            changed_text_counts["style_factor_changed"] += dose[
                "effective_style_factor_tuple_changed_count"
            ]
            changed_text_counts["profile_text_changed"] += dose[
                "seller_profile_text_changed_count"
            ]
            mapping_hashes.append(str(counterfactual["audit"]["mapping_sha256"]))
        if position % 10 == 0 or position == len(records):
            print(json.dumps({"event": "quality_replay_progress", "worlds_complete": position, "worlds_total": len(records)}, ensure_ascii=False), flush=True)
    if (
        len(current_item_hashes) != int(root_manifest["item_document_registry_count"])
        or common.canonical_sha256(sorted(current_item_hashes))
        != str(root_manifest["item_document_registry_sha256"])
        or len(current_seller_hashes)
        != int(root_manifest["seller_document_registry_count"])
        or common.canonical_sha256(sorted(current_seller_hashes))
        != str(root_manifest["seller_document_registry_sha256"])
    ):
        raise DatasetInvalidationError("Observed root document registry drift")
    for split in ("train", "development"):
        bundle = bundles[split]
        manifest = split_manifests[split]
        for kind in ("controller", "query"):
            values = privileged_registry_sets[split][kind]
            expected = manifest["uid_registries"][kind]
            if (
                len(values) != int(expected["count"])
                or common.canonical_sha256(sorted(values))
                != str(expected["sha256"])
            ):
                raise DatasetInvalidationError(
                    f"Privileged split UID registry drift: {split}/{kind}"
                )
        identity_values = privileged_registry_sets[split]["identity"]
        if (
            len(identity_values) != int(manifest["identity_value_registry_count"])
            or common.canonical_sha256(sorted(identity_values))
            != str(manifest["identity_value_registry_sha256"])
        ):
            raise DatasetInvalidationError(
                f"Privileged split identity registry drift: {split}"
            )
        if (
            len(bundle["endpoints"]) != 50 * 378
            or len(bundle["eligible_pair_uids"]) != 50 * 372
            or sum(bundle["labels"]) != 50 * 20
            or len(bundle["ordered_world_uids"]) != 50
            or len(set(bundle["ordered_world_uids"])) != 50
        ):
            raise DatasetInvalidationError(f"Design probe cardinality drift: {split}")
    _validate_final_blind_boundary_counters(blind_counters)
    if (
        len(path_alignment_receipts) != 100
        or len(dose_rows) != 100
        or len(fixed_support_receipts) != 100
        or len(provenance_delta_receipts) != 100
    ):
        raise ScientificQualityAuditError(
            "Quality receipt cardinality drift"
        )
    return bundles, {
        "world_count": len(records),
        "scanned_visible_text_count": scanned_text_count,
        "counterfactual_changed_counts": dict(changed_text_counts),
        "counterfactual_mapping_set_sha256": common.canonical_sha256(sorted(mapping_hashes)),
        "counterfactual_path_alignment_receipts": path_alignment_receipts,
        "counterfactual_path_alignment_receipts_sha256": common.canonical_sha256(
            path_alignment_receipts
        ),
        "counterfactual_dose_rows": dose_rows,
        "counterfactual_dose_rows_sha256": common.canonical_sha256(dose_rows),
        "fixed_support_receipts_sha256": common.canonical_sha256(
            fixed_support_receipts
        ),
        "fixed_support_receipts": fixed_support_receipts,
        "profile_provenance_delta_receipts_sha256": common.canonical_sha256(
            provenance_delta_receipts
        ),
        "profile_provenance_delta_receipts": provenance_delta_receipts,
        "profile_provenance_delta_world_count": len(provenance_delta_receipts),
        "counterfactual_dose_minimums": {
            name: min(int(row[name]) for row in dose_rows)
            for name in (
                "source_seller_changed_count",
                "effective_style_uid_changed_count",
                "effective_style_factor_tuple_changed_count",
                "seller_profile_text_changed_count",
                "visible_title_slot_changed_count",
                "visible_description_slot_changed_count",
                "visible_seller_changed_count",
            )
        },
        "counterfactual_zero_dose_maximum": max(
            int(row["zero_dose_seller_count"]) for row in dose_rows
        ),
        "counterfactual_zero_visible_dose_maximum": max(
            int(row["zero_visible_dose_seller_count"]) for row in dose_rows
        ),
        "blind_boundary_counters": copy.deepcopy(blind_counters),
        "sealed_literal_scan_receipts": sealed_scan_receipts,
        "sealed_literal_scan_receipts_sha256": common.canonical_sha256(
            sealed_scan_receipts
        ),
        "sealed_registry_isolation_receipt": sealed_registry_isolation_receipt,
        "sealed_registry_isolation_receipt_sha256": common.canonical_sha256(
            sealed_registry_isolation_receipt
        ),
        "root_tree_reverified": True,
    }


def evaluate_metadata_shortcuts(
    policy: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    names: tuple[str, ...] | None = None
    pair_uids: dict[str, np.ndarray] = {}
    world_uids: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "development"):
        bundle = bundles[split]
        matrix, observed_names, pairs, worlds = build_metadata_matrix(
            profiles=bundle["original_profiles"],
            observed_items=bundle["observed_items"],
            source_items=bundle["source_items"],
            endpoints=bundle["endpoints"],
        )
        if names is None:
            names = observed_names
        elif names != observed_names:
            raise ScientificQualityAuditError("Metadata feature order drift")
        matrices[split] = matrix
        pair_uids[split] = pairs
        world_uids[split] = worlds
        labels[split] = np.asarray(bundle["labels"], dtype=np.int8)
    expected_names = tuple(policy["metadata_probe"]["feature_names_in_order"])
    if names != expected_names:
        raise ScientificQualityAuditError("Metadata policy feature order drift")
    scores = _fit_probe_family(
        matrices["train"],
        labels["train"],
        matrices["development"],
        config=policy["metadata_probe"]["models"],
    )
    single_values = {
        name: symmetric_auc(labels["development"], matrices["development"][:, index])
        for index, name in enumerate(expected_names)
    }
    model_metrics = {
        name: {
            "symmetric_roc_auc": _rank_metrics(labels["development"], values)[0],
            "average_precision": _rank_metrics(labels["development"], values)[1],
            "average_precision_uplift": _rank_metrics(labels["development"], values)[1]
            - float(policy["metadata_probe"]["average_precision_baseline"]),
        }
        for name, values in scores.items()
    }
    bootstrap = _bootstrap_family_upper(
        labels=labels["development"],
        row_world_uids=world_uids["development"],
        ordered_world_uids=bundles["development"]["ordered_world_uids"],
        score_family=tuple(scores.values()),
        replicates=int(policy["bootstrap"]["replicates"]),
        seed=int(policy["bootstrap"]["metadata_design_seed"]),
        baseline=float(policy["metadata_probe"]["average_precision_baseline"]),
    )
    gates = policy["metadata_probe"]["hard_gates"]
    checks = {
        "single_feature_symmetric_auc": max(single_values.values())
        <= float(gates["maximum_single_feature_symmetric_auc"]),
        "family_symmetric_auc": max(
            row["symmetric_roc_auc"] for row in model_metrics.values()
        )
        <= float(gates["maximum_family_symmetric_auc"]),
        "family_average_precision_uplift": max(
            row["average_precision_uplift"] for row in model_metrics.values()
        )
        <= float(gates["maximum_family_average_precision_uplift"]),
        "bootstrap_auc_upper": bootstrap["auc_95_upper"]
        <= float(gates["bootstrap_95_upper_symmetric_auc"]),
        "bootstrap_ap_upper": bootstrap["ap_uplift_95_upper"]
        <= float(gates["bootstrap_95_upper_average_precision_uplift"]),
    }
    return {
        "feature_count": len(expected_names),
        "development_pair_count": len(labels["development"]),
        "development_positive_count": int(labels["development"].sum()),
        "average_precision_baseline": float(
            policy["metadata_probe"]["average_precision_baseline"]
        ),
        "single_feature_maximum": {
            "name": max(single_values, key=single_values.get),
            "symmetric_roc_auc": max(single_values.values()),
        },
        "single_feature_metrics": single_values,
        "model_metrics": model_metrics,
        "bootstrap": bootstrap,
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def _subset_views(
    views: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: matrix[mask] for name, matrix in views.items()}


def evaluate_text_counterfactual(
    policy: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    prepared: dict[str, dict[str, Any]] = {}
    for split in ("train", "development"):
        bundle = bundles[split]
        original_p, original_p_names, pair_uids, world_uids = build_text_views(
            profiles=bundle["original_profiles"], endpoints=bundle["endpoints"]
        )
        counterfactual_p, counterfactual_p_names, cf_pairs, cf_worlds = build_text_views(
            profiles=bundle["counterfactual_profiles"], endpoints=bundle["endpoints"]
        )
        original_f, original_f_names, f_pairs, f_worlds = (
            build_fixed_support_text_views(
                items=bundle["original_items"], endpoints=bundle["endpoints"]
            )
        )
        counterfactual_f, counterfactual_f_names, cf_f_pairs, cf_f_worlds = (
            build_fixed_support_text_views(
                items=bundle["counterfactual_items"], endpoints=bundle["endpoints"]
            )
        )
        original_numeric, numeric_names, n_pairs, n_worlds = (
            build_production_numeric_matrix(
                profiles=bundle["original_profiles"], endpoints=bundle["endpoints"]
            )
        )
        counterfactual_numeric, cf_numeric_names, cf_n_pairs, cf_n_worlds = (
            build_production_numeric_matrix(
                profiles=bundle["counterfactual_profiles"],
                endpoints=bundle["endpoints"],
            )
        )
        if (
            original_p_names != counterfactual_p_names
            or original_f_names != counterfactual_f_names
            or numeric_names != cf_numeric_names
            or not np.array_equal(pair_uids, cf_pairs)
            or not np.array_equal(world_uids, cf_worlds)
            or any(
                not np.array_equal(pair_uids, values)
                for values in (f_pairs, cf_f_pairs, n_pairs, cf_n_pairs)
            )
            or any(
                not np.array_equal(world_uids, values)
                for values in (f_worlds, cf_f_worlds, n_worlds, cf_n_worlds)
            )
        ):
            raise ScientificQualityAuditError("Original/counterfactual view join drift")
        original = {**original_p, **original_f}
        counterfactual = {**counterfactual_p, **counterfactual_f}
        original_names = {**original_p_names, **original_f_names}
        counterfactual_names = {**counterfactual_p_names, **counterfactual_f_names}
        joint_names = tuple(
            [f"p::{name}" for name in original_p_names["p_full"]]
            + [f"fs::{name}" for name in original_f_names["fs_full"]]
            + [f"numeric::{name}" for name in numeric_names]
        )
        original["u_joint_full"] = np.column_stack(
            (original_p["p_full"], original_f["fs_full"], original_numeric)
        )
        counterfactual["u_joint_full"] = np.column_stack(
            (
                counterfactual_p["p_full"],
                counterfactual_f["fs_full"],
                counterfactual_numeric,
            )
        )
        original_names["u_joint_full"] = joint_names
        counterfactual_names["u_joint_full"] = joint_names
        if tuple(original) != tuple(policy["text_counterfactual"]["views"]):
            raise ScientificQualityAuditError("Text path/view order drift")
        for view_name, contract in policy["text_counterfactual"]["views"].items():
            observed_names = original_names[view_name]
            if (
                observed_names != counterfactual_names[view_name]
                or int(contract["expected_width"]) != len(observed_names)
                or str(contract["feature_names_canonical_sha256"])
                != common.canonical_sha256(list(observed_names))
            ):
                raise ScientificQualityAuditError("Text policy feature order drift")
        eligible = np.asarray(
            [pair_uid in bundle["eligible_pair_uids"] for pair_uid in pair_uids],
            dtype=bool,
        )
        labels = np.asarray(bundle["labels"], dtype=np.int8)[eligible]
        if int(eligible.sum()) != 50 * 372 or int(labels.sum()) != 50 * 20:
            raise ScientificQualityAuditError("Mechanism-neutral text mask drift")
        prepared[split] = {
            "original": _subset_views(original, eligible),
            "counterfactual": _subset_views(counterfactual, eligible),
            "feature_names": original_names,
            "labels": labels,
            "world_uids": world_uids[eligible],
        }
    config = policy["text_counterfactual"]["models"]
    cf_scores: dict[str, np.ndarray] = {}
    original_metrics: dict[str, dict[str, float]] = {}
    cf_metrics: dict[str, dict[str, float]] = {}
    single_values: dict[str, float] = {}
    baseline = float(policy["text_counterfactual"]["average_precision_baseline"])
    for view_name in policy["text_counterfactual"]["views"]:
        fitted_cf = _fit_probe_family(
            prepared["train"]["counterfactual"][view_name],
            prepared["train"]["labels"],
            prepared["development"]["counterfactual"][view_name],
            config=config,
        )
        fitted_original = _fit_probe_family(
            prepared["train"]["original"][view_name],
            prepared["train"]["labels"],
            prepared["development"]["original"][view_name],
            config=config,
        )
        for model_name, scores in fitted_cf.items():
            key = f"{view_name}__{model_name}"
            cf_scores[key] = scores
            auc, ap = _rank_metrics(prepared["development"]["labels"], scores)
            cf_metrics[key] = {
                "symmetric_roc_auc": auc,
                "average_precision": ap,
                "average_precision_uplift": ap - baseline,
            }
        for model_name, scores in fitted_original.items():
            key = f"{view_name}__{model_name}"
            auc, ap = _rank_metrics(prepared["development"]["labels"], scores)
            original_metrics[key] = {
                "symmetric_roc_auc": auc,
                "average_precision": ap,
                "average_precision_uplift": ap - baseline,
            }
        matrix = prepared["development"]["counterfactual"][view_name]
        for index, feature_name in enumerate(
            prepared["development"]["feature_names"][view_name]
        ):
            single_values[f"{view_name}::{feature_name}"] = symmetric_auc(
                prepared["development"]["labels"], matrix[:, index]
            )
    bootstrap = _bootstrap_family_upper(
        labels=prepared["development"]["labels"],
        row_world_uids=prepared["development"]["world_uids"],
        ordered_world_uids=bundles["development"]["ordered_world_uids"],
        score_family=tuple(cf_scores.values()),
        replicates=int(policy["bootstrap"]["replicates"]),
        seed=int(policy["bootstrap"]["text_design_seed"]),
        baseline=baseline,
    )
    gates = policy["text_counterfactual"]["hard_gates"]
    checks = {
        "single_feature_symmetric_auc": max(single_values.values())
        <= float(gates["maximum_single_feature_symmetric_auc"]),
        "fourteen_model_symmetric_auc": max(
            row["symmetric_roc_auc"] for row in cf_metrics.values()
        )
        <= float(gates["maximum_fourteen_model_symmetric_auc"]),
        "fourteen_model_average_precision_uplift": max(
            row["average_precision_uplift"] for row in cf_metrics.values()
        )
        <= float(gates["maximum_fourteen_model_average_precision_uplift"]),
        "bootstrap_auc_upper": bootstrap["auc_95_upper"]
        <= float(gates["bootstrap_95_upper_symmetric_auc"]),
        "bootstrap_ap_upper": bootstrap["ap_uplift_95_upper"]
        <= float(gates["bootstrap_95_upper_average_precision_uplift"]),
    }
    path_views = {
        "fixed_support": ("fs_full", "fs_title", "fs_template_surface"),
        "production_step3": ("p_full", "p_topic", "p_template_surface"),
        "joint_visible_input": ("u_joint_full",),
    }
    path_maxima = {}
    for path_name, views in path_views.items():
        view_prefixes = tuple(f"{view}__" for view in views)
        model_rows = {
            name: row
            for name, row in cf_metrics.items()
            if name.startswith(view_prefixes)
        }
        feature_rows = {
            name: value
            for name, value in single_values.items()
            if name.startswith(tuple(f"{view}::" for view in views))
        }
        if len(model_rows) != len(views) * 2 or not feature_rows:
            raise ScientificQualityAuditError("Text path family cardinality drift")
        path_maxima[path_name] = {
            "model_count": len(model_rows),
            "single_feature_symmetric_auc_maximum": max(feature_rows.values()),
            "model_symmetric_auc_maximum": max(
                row["symmetric_roc_auc"] for row in model_rows.values()
            ),
            "model_average_precision_uplift_maximum": max(
                row["average_precision_uplift"] for row in model_rows.values()
            ),
        }
    return {
        "view_count": len(policy["text_counterfactual"]["views"]),
        "model_count": len(cf_metrics),
        "development_pair_count": len(prepared["development"]["labels"]),
        "development_positive_count": int(prepared["development"]["labels"].sum()),
        "average_precision_baseline": baseline,
        "single_feature_maximum": {
            "name": max(single_values, key=single_values.get),
            "symmetric_roc_auc": max(single_values.values()),
        },
        "counterfactual_model_metrics": cf_metrics,
        "original_text_descriptive_metrics": original_metrics,
        "path_maxima": path_maxima,
        "bootstrap": bootstrap,
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def _evidence_binding(
    policy: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    launch_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    policy_path = DEFAULT_POLICY_PATH
    dataset_manifest_path = common.repo_path(
        str(policy["pins"]["dataset_root_manifest"]["path"])
    )
    return {
        "scientific_contract": copy.deepcopy(policy["pins"]["scientific_contract"]),
        "quality_audit_c_amendment": copy.deepcopy(
            policy["pins"]["quality_audit_c_amendment"]
        ),
        "launch": copy.deepcopy(dict(launch_evidence)),
        "quality_policy": {
            "path": policy_path.relative_to(ROOT).as_posix(),
            "size_bytes": policy_path.stat().st_size,
            "sha256": common.sha256_file(policy_path),
            "canonical_self_hash": str(policy["canonical_self_hash"]),
        },
        "dataset_root_manifest": {
            "path": dataset_manifest_path.relative_to(ROOT).as_posix(),
            "size_bytes": dataset_manifest_path.stat().st_size,
            "sha256": common.sha256_file(dataset_manifest_path),
            "canonical_self_hash": str(root_manifest["canonical_self_hash"]),
        },
        "source_and_test_pins": {
            name: copy.deepcopy(spec)
            for name, spec in policy["pins"].items()
            if name
            in {
                "dataset_builder_policy",
                "style_derangement",
                "counterfactual_text",
                "blind_literal_scan",
                "quality_audit",
                "quality_tests",
            }
        },
        "runtime": copy.deepcopy(policy["runtime"]),
    }


def _write_failure_receipt(
    *,
    policy: Mapping[str, Any],
    output_root: Path,
    stage: str,
    error: Exception,
    blind_counters: Mapping[str, Mapping[str, int]],
    partial_results: Mapping[str, Any],
    classification: str,
    launch_evidence: Mapping[str, Any],
) -> Path:
    failure_root = output_root.with_name(f"{output_root.name}_FAILED")
    temporary = failure_root.with_name(f".{failure_root.name}.building")
    if failure_root.exists() or temporary.exists():
        raise ScientificQualityAuditError(
            "Immutable quality failure receipt already exists"
        )
    dataset_manifest_path = common.repo_path(
        str(policy["pins"]["dataset_root_manifest"]["path"])
    )
    root_manifest = common.load_json(dataset_manifest_path)
    if classification not in {
        "DATASET_INVALIDATED",
        "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
    }:
        raise ScientificQualityAuditError("Unknown failure classification")
    partial_summary = {
        name: {
            "canonical_sha256": common.canonical_sha256(value),
            "passed": value.get("passed") if isinstance(value, Mapping) else None,
        }
        for name, value in partial_results.items()
    }
    receipt = {
        "version": VERSION,
        "status": classification,
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "failure_reason_code": (
            "DATASET_GATE_OR_ROW_CONTRACT_FAILED"
            if classification == "DATASET_INVALIDATED"
            else "AUDITOR_IMPLEMENTATION_OR_RUNTIME_FAILED"
        ),
        "exception_message_sha256": common.sha256_bytes(
            str(error).encode("utf-8")
        ),
        "claim_boundary": {
            "formal_dataset_quality_pass": False,
            "training_qualified": False,
            "scientific_use_forbidden": True,
        },
        "formal_seed_created": False,
        "formal_rows_created": 0,
        "training_started": False,
        "blind_boundary_counters": copy.deepcopy(blind_counters),
        "completed_partial_result_summaries": partial_summary,
        "evidence_binding": _evidence_binding(
            policy, root_manifest, launch_evidence
        ),
        # This receipt is committed before any invalidated-dataset cleanup.
        # Only cleanup_receipt.json may claim that deletion completed.
        "input_dataset_retained_at_decision": True,
        "input_dataset_state_at_decision": (
            "PRESENT_PENDING_CLEANUP"
            if classification == "DATASET_INVALIDATED"
            else "PRESENT_AND_PRESERVED"
        ),
        "dataset_quality_conclusion_reached": classification
        == "DATASET_INVALIDATED",
        "cleanup_required": classification == "DATASET_INVALIDATED",
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        common.write_json(temporary / "decision_receipt.json", receipt)
        temporary.replace(failure_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return failure_root


def _cleanup_invalidated_dataset(
    *, policy: Mapping[str, Any], failure_root: Path
) -> dict[str, Any]:
    cleanup_path = failure_root / "cleanup_receipt.json"
    intent_path = failure_root / "cleanup_intent.json"
    decision_path = failure_root / "decision_receipt.json"
    if not decision_path.is_file():
        raise ScientificQualityAuditError("Cleanup decision receipt is missing")
    decision = common.load_json(decision_path)
    if (
        decision.get("canonical_self_hash") != _canonical_self_hash(decision)
        or decision.get("status") != "DATASET_INVALIDATED"
        or decision.get("cleanup_required") is not True
    ):
        raise ScientificQualityAuditError("Cleanup decision receipt drift")
    if cleanup_path.exists():
        existing = common.load_json(cleanup_path)
        if not intent_path.is_file():
            raise ScientificQualityAuditError("Cleanup intent is missing")
        existing_intent = common.load_json(intent_path)
        if (
            set(existing)
            != {
                "version",
                "status",
                "safe_target_verified",
                "cleanup_intent_canonical_self_hash",
                "decision_receipt_canonical_self_hash",
                "dataset_root_sha256_before_cleanup",
                "input_dataset_deleted",
                "deletion_error_type",
                "canonical_self_hash",
            }
            or existing.get("canonical_self_hash") != _canonical_self_hash(existing)
            or existing.get("status")
            != "DATASET_INVALIDATION_CLEANUP_COMPLETE"
            or existing.get("safe_target_verified") is not True
            or existing.get("cleanup_intent_canonical_self_hash")
            != existing_intent.get("canonical_self_hash")
            or existing.get("decision_receipt_canonical_self_hash")
            != decision["canonical_self_hash"]
            or existing.get("dataset_root_sha256_before_cleanup")
            != str(policy["input"]["root_manifest_canonical_self_hash"])
            or existing.get("input_dataset_deleted") is not True
            or existing.get("deletion_error_type") is not None
        ):
            raise ScientificQualityAuditError("Cleanup receipt binding drift")
        return existing
    dataset_root = common.repo_path(str(policy["input"]["dataset_root"])).resolve()
    pinned_manifest = common.repo_path(
        str(policy["pins"]["dataset_root_manifest"]["path"])
    ).resolve()
    expected_parent = (ROOT / "reports" / "step28_v13_v1_13_scientific_builder").resolve()
    safe_shape = (
        dataset_root.parent == expected_parent
        and pinned_manifest == dataset_root / "root_manifest.json"
        and dataset_root.name.startswith("design_preflight_")
    )
    manifest_matches = (
        dataset_root.is_dir()
        and pinned_manifest.is_file()
        and pinned_manifest.stat().st_size
        == policy["pins"]["dataset_root_manifest"]["size_bytes"]
        and common.sha256_file(pinned_manifest)
        == policy["pins"]["dataset_root_manifest"]["sha256"]
    )
    intended = {
        "version": VERSION,
        "status": "DATASET_INVALIDATION_CLEANUP_INTENT_COMMITTED",
        "dataset_root_relative_path": Path(
            policy["input"]["dataset_root"]
        ).as_posix(),
        "dataset_root_resolved_path_sha256": common.sha256_bytes(
            dataset_root.as_posix().encode("utf-8")
        ),
        "dataset_root_manifest_pin": copy.deepcopy(
            policy["pins"]["dataset_root_manifest"]
        ),
        "decision_receipt": {
            "size_bytes": decision_path.stat().st_size,
            "sha256": common.sha256_file(decision_path),
            "canonical_self_hash": decision["canonical_self_hash"],
        },
        "safe_target_verified_before_cleanup": safe_shape and manifest_matches,
        "canonical_self_hash": None,
    }
    intended["canonical_self_hash"] = _canonical_self_hash(intended)
    if intent_path.exists():
        observed_intent = common.load_json(intent_path)
        if observed_intent != intended:
            # A valid prior intent remains authoritative after its target has
            # been deleted, so compare all immutable bindings except the
            # pre-delete observation recomputed above.
            expected_recovery = copy.deepcopy(intended)
            expected_recovery["safe_target_verified_before_cleanup"] = True
            expected_recovery["canonical_self_hash"] = _canonical_self_hash(
                expected_recovery
            )
            if observed_intent != expected_recovery:
                raise ScientificQualityAuditError("Cleanup intent binding drift")
        intent = observed_intent
    else:
        if not (safe_shape and manifest_matches):
            raise ScientificQualityAuditError(
                "Unsafe invalidated-dataset cleanup target"
            )
        common.write_json(intent_path, intended)
        intent = intended
    safe = bool(intent["safe_target_verified_before_cleanup"])
    deletion_error_type: str | None = None
    completed = False
    if safe:
        if dataset_root.exists():
            try:
                shutil.rmtree(dataset_root)
                completed = not dataset_root.exists()
            except Exception as exc:  # pragma: no cover - exercised by mock
                deletion_error_type = type(exc).__name__
        else:
            completed = True
    receipt = {
        "version": VERSION,
        "status": (
            "DATASET_INVALIDATION_CLEANUP_COMPLETE"
            if completed
            else "DATASET_INVALIDATION_CLEANUP_FAILED"
        ),
        "safe_target_verified": safe,
        "cleanup_intent_canonical_self_hash": intent["canonical_self_hash"],
        "decision_receipt_canonical_self_hash": decision["canonical_self_hash"],
        "dataset_root_sha256_before_cleanup": str(
            policy["input"]["root_manifest_canonical_self_hash"]
        ),
        "input_dataset_deleted": completed,
        "deletion_error_type": deletion_error_type,
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
    common.write_json(cleanup_path, receipt)
    return receipt


def recover_cleanup_receipt() -> dict[str, Any]:
    """Complete only an already committed deletion transaction; never delete data."""

    policy = load_policy_for_cleanup_recovery()
    dataset_root = common.repo_path(str(policy["input"]["dataset_root"])).resolve()
    output_root = common.repo_path(str(policy["input"]["output_root"])).resolve()
    temp_root = output_root.parent / f".{output_root.name}.building"
    failure_root = output_root.with_name(f"{output_root.name}_FAILED")
    if (
        dataset_root.exists()
        or output_root.exists()
        or temp_root.exists()
        or not failure_root.is_dir()
        or not (failure_root / "decision_receipt.json").is_file()
        or not (failure_root / "cleanup_intent.json").is_file()
    ):
        raise ScientificQualityAuditError("Cleanup recovery state is not reachable")
    return _cleanup_invalidated_dataset(policy=policy, failure_root=failure_root)


def run_audit(*, launch_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        frame = inspect.currentframe()
        caller = (
            Path(frame.f_back.f_code.co_filename)
            if frame is not None and frame.f_back is not None
            else Path("")
        )
        del frame
        verified_launch = _verify_launch_evidence(
            launch_evidence,
            caller_path=caller,
        )
        policy = load_policy()
        output_root = common.repo_path(str(policy["input"]["output_root"]))
        temp_root = output_root.parent / f".{output_root.name}.building"
        failure_root = output_root.with_name(f"{output_root.name}_FAILED")
        if output_root.exists() or temp_root.exists() or failure_root.exists():
            raise ScientificQualityAuditError(
                "Immutable quality PASS/FAIL output already exists"
            )
    except Exception as exc:
        if isinstance(exc, AuditLaunchPreflightError):
            raise
        raise AuditLaunchPreflightError(
            "Quality audit launch preflight failed before dataset access"
        ) from exc
    blind_counters = _new_blind_boundary_counters()
    partial_results: dict[str, Any] = {}
    stage = "row_replay"
    try:
        bundles, row_audit = replay_and_audit_worlds(
            policy, blind_counters=blind_counters
        )
        partial_results["row_audit"] = row_audit
        stage = "metadata_shortcut_probe"
        metadata = evaluate_metadata_shortcuts(policy, bundles)
        partial_results["metadata_shortcut_audit"] = metadata
        print(
            json.dumps(
                {
                    "event": "metadata_quality_complete",
                    "passed": metadata["passed"],
                    "maximum": metadata["single_feature_maximum"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        stage = "text_counterfactual_probe"
        text_audit = evaluate_text_counterfactual(policy, bundles)
        partial_results["text_counterfactual_audit"] = text_audit
        print(
            json.dumps(
                {
                    "event": "text_quality_complete",
                    "passed": text_audit["passed"],
                    "maximum": text_audit["single_feature_maximum"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        passed = metadata["passed"] and text_audit["passed"]
        root_manifest = common.load_json(
            common.repo_path(str(policy["pins"]["dataset_root_manifest"]["path"]))
        )
        result = {
            "version": VERSION,
            "status": (
                "PASS_DESIGN_QUALITY_NOT_TRAINING_QUALIFIED"
                if passed
                else "FAIL_DESIGN_QUALITY_NOT_TRAINING_QUALIFIED"
            ),
            "scientific_use_forbidden": True,
            "formal_seed_created": False,
            "formal_rows_created": 0,
            "training_started": False,
            "row_audit": row_audit,
            "metadata_shortcut_audit": metadata,
            "text_counterfactual_audit": text_audit,
            "evidence_binding": _evidence_binding(
                policy, root_manifest, verified_launch
            ),
            "canonical_self_hash": None,
        }
        result["canonical_self_hash"] = _canonical_self_hash(result)
        if not passed:
            partial_results["failed_gate_result"] = result
            print(
                json.dumps(
                    {
                        "event": "design_quality_gate_failed",
                        "result_canonical_self_hash": result[
                            "canonical_self_hash"
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            raise DatasetInvalidationError("Design quality gate failed")
        stage = "publish_pass_receipt"
        temp_root.mkdir(parents=True, exist_ok=False)
        common.write_json(temp_root / "quality_audit.json", result)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temp_root.replace(output_root)
        return result
    except Exception as exc:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        classification = _failure_classification(exc)
        if not failure_root.exists():
            failure_root = _write_failure_receipt(
                policy=policy,
                output_root=output_root,
                stage=stage,
                error=exc,
                blind_counters=blind_counters,
                partial_results=partial_results,
                classification=classification,
                launch_evidence=verified_launch,
            )
        if classification == "DATASET_INVALIDATED":
            _cleanup_invalidated_dataset(policy=policy, failure_root=failure_root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-policy",
        action="store_true",
        help="Validate pins and runtime without reading dataset rows.",
    )
    args = parser.parse_args()
    if args.validate_policy:
        policy = load_policy()
        print(json.dumps({"status": "PASS_POLICY_ONLY", "version": policy["version"]}, ensure_ascii=False))
        return
    raise ScientificQualityAuditError(
        "Direct quality execution is forbidden; use the reviewed launch guard"
    )


if __name__ == "__main__":
    main()
