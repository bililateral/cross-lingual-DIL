#!/usr/bin/env python3
"""Shared contracts for Step 7-v4 raw-item authorship source selection."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np



class _LazySourceModule:
    """Avoid importing the raw-data stack inside the isolated GPU encoder."""

    _module = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module("step7_v3_1_source_data")
        return getattr(self._module, name)


source = _LazySourceModule()


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step7_v4_raw_item_authorship_selection_policy.json"
EXPECTED_VERSION = "2026-07-23-step7-v4-raw-item-authorship-selection-v2"
REQUIRED_SENTENCE_TRANSFORMERS_VERSION = "5.6.0"
FIELD_NAMES = ["title", "description"]
STYLOMETRY_STATISTICS = [
    "log1p_character_count",
    "digit_character_ratio",
    "punctuation_character_ratio",
    "symbol_character_ratio",
    "space_character_ratio",
    "newline_per_100_characters",
    "sentence_boundary_per_100_characters",
    "bracket_character_ratio",
    "delimiter_character_ratio",
    "repeated_punctuation_character_ratio",
    "bullet_line_share",
]
AGGREGATE_SUFFIXES = [
    "field_equal_centroid_cosine",
    "field_equal_symmetric_top3_cosine",
    "title_centroid_cosine",
    "title_symmetric_top3_cosine",
    "description_centroid_cosine",
    "description_symmetric_top3_cosine",
]
MODEL_KEYS = [
    "pcm_multilingual_authorship",
    "mstyledistance",
    "multilingual_e5_large",
    "labse",
]
MULTILINE_IDENTITY_REDACTION_RULE_NAMES = frozenset(
    {
        "contact_cued_split_plus_phone",
        "pgp_block_truncated_or_complete",
        "pgp_block",
        "pgp_fingerprint",
        "split_plus_country_area_phone",
    }
)
V4_MARKET_IDENTITY_LIST_MEMBER_PATTERN = (
    r"(?:dream[ \t]+market|agora[ \t]+market|wall[ \t]*street|"
    r"grey(?:[ \t]+market)?|samsara|alphabay[ \t]+market)"
)
V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST = frozenset(
    {
        "account",
        "discount",
        "dumpstrack",
        "exchange",
        "legitdocuments",
        "password",
        "premiumaccount",
        "without",
    }
)
V4_ADDITIONAL_IDENTITY_RULES = (
    (
        "repeated_email_domain_chain",
        re.compile(
            r"(?i)(?<![a-z0-9_+%\-])"
            r"[a-z0-9][a-z0-9._%+\-]{0,63}"
            r"@[a-z0-9](?:[a-z0-9.\-]{0,252}[a-z0-9])?"
            r"(?:@[a-z0-9](?:[a-z0-9.\-]{0,252}[a-z0-9])?){1,4}"
            r"(?![a-z0-9.\-])"
        ),
    ),
    (
        "contact_service_at_mixed_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:wicker|(?:"
            r"w[\W_]{0,3}(?:[i1!][\W_]{0,3})?c[\W_]{0,3}k[\W_]{0,3}(?:r|e)|"
            r"w[\W_]{0,3}[i1!][\W_]{0,3}k[\W_]{0,3}r|"
            r"v[\W_]{0,2}v[\W_]{0,2}[i1!][\W_]{0,2}c[\W_]{0,2}k[\W_]{0,2}r"
            r")|telegram|kik|wh?ats?app|jabber|xmpp|wechat|"
            r"snapchat|discord|icq)"
            r"[ \t.:,;=@_/\-]{1,24}(?:at[ \t.:,;=@_/\-]+)?"
            r"(?=[a-z0-9_.-]{4,64}(?![a-z0-9_.-]))"
            r"(?=[a-z0-9_.-]*[a-z])(?=[a-z0-9_.-]*\d)"
            r"[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
    (
        "contact_cued_split_plus_phone",
        re.compile(
            r"(?ix)(?<![a-z0-9])"
            r"(?:wh?ats?app|phone|telephone|tel|mobile|call|text|sms|"
            r"contact|reach|message)"
            r"[^|\n]{0,48}?\+[ \t\u00a0]*"
            r"(?:\n[ \t\u00a0]*){1,3}"
            r"\d(?:[ \t\u00a0().-]*\d){5,15}"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "split_plus_country_area_phone",
        re.compile(
            r"(?ix)(?<![a-z0-9])\+[ \t\u00a0]*"
            r"\d{1,3}[ \t\u00a0]*"
            r"(?:\([ \t\u00a0]*\d{2,4}[ \t\u00a0]*\)|"
            r"\d{2,4})[ \t\u00a0]*"
            r"(?:\n[ \t\u00a0]*){1,3}"
            r"\d{3,4}(?:[ \t\u00a0.-]*\d){3,7}"
            r"(?![a-z0-9])"
        ),
    ),
    (
        "market_identity_list_before_vendor_cue",
        re.compile(
            rf"(?ix)(?<![a-z0-9])"
            rf"{V4_MARKET_IDENTITY_LIST_MEMBER_PATTERN}"
            rf"(?:[ \t]*,[ \t]*{V4_MARKET_IDENTITY_LIST_MEMBER_PATTERN})"
            rf"{{2,11}}[ \t]+(?:verified[ \t]+)?vendor\b[ \t]*[.!]?"
        ),
    ),
    (
        "google_voice_attached_phone",
        re.compile(
            r"(?ix)(?<![a-z0-9])google[ \t\u00a0]+voice"
            r"[^a-z0-9|\n]{0,24}\+?[0-9]"
            r"(?:[ \t\u00a0().-]*[0-9]){5,15}(?![a-z0-9])"
        ),
    ),
    (
        "concatenated_whatsapp_phone_wickr_mixed_handle",
        re.compile(
            r"(?ix)(?<![a-z0-9])wh?ats?app"
            r"[^a-z0-9|\n]{0,24}\+?[0-9]"
            r"(?:[ \t\u00a0().-]*[0-9]){5,15}"
            r"[^a-z0-9|\n]{0,12}"
            r"w[\W_]{0,3}[i1!][\W_]{0,3}c[\W_]{0,3}k[\W_]{0,3}r"
            r"[^a-z0-9|\n]{0,24}"
            r"(?=[a-z0-9_.-]{4,64}(?![a-z0-9_.-]))"
            r"(?=[a-z0-9_.-]*[a-z])(?=[a-z0-9_.-]*[0-9])"
            r"[a-z0-9][a-z0-9_.-]{3,63}"
        ),
    ),
)
FIXED_FINAL_AUDIT_CONTENT_COLLISION_REVIEW_BASIS = (
    "fixed_english_clean_snapshot_manual_content_review_before_model_"
    "training_without_pair_labels_or_evidence_types"
)
FIXED_FINAL_AUDIT_CONTENT_COLLISION_FIELDS = frozenset(
    {
        "rule_name",
        "seller_uid_sha256",
        "clean_text_sha256",
        "matched_surface_sha256",
        "expected_match_count",
        "reason",
    }
)
# Keep GPU policy validation independent from the parent raw-data stack.  Only
# false positives that were explicitly reviewed and frozen here may appear in
# the policy.  The CPU corpus audit still replays the complete parent rule set.
FIXED_FINAL_AUDIT_CONTENT_COLLISION_ALLOWED_RULE_NAMES = frozenset(
    {"audit_contact_cued_phone"}
)
MODEL_NATIVE_MAX_SEQ_LENGTHS = {
    "pcm_multilingual_authorship": 512,
    "mstyledistance": 512,
    "multilingual_e5_large": 512,
    "labse": 256,
}
SHARED_TOKEN_BUDGET = min(MODEL_NATIVE_MAX_SEQ_LENGTHS.values())
BLOCK_MODEL_KEYS = {
    "pcm6": "pcm_multilingual_authorship",
    "mstyle6": "mstyledistance",
    "e5_6": "multilingual_e5_large",
    "labse6": "labse",
}
PUBLIC_OUTPUT_ROLES = [
    "pair_manifest",
    "gpu_pair_manifest",
    "raw_item_lineage",
    "unique_text_corpus",
    "seller_text_index",
    "gpu_seller_text_index",
    "seller_stylometry",
    "pair_stylometry",
    "legacy_pair_features",
]
PRIVATE_OUTPUT_ROLES = [
    "train_labels",
    "valid_labels",
    "train_evidence",
    "valid_evidence",
]
SENTENCE_BOUNDARY_CHARACTERS = frozenset(".!?。！？…")
DELIMITER_CHARACTERS = frozenset("|/\\:_-=+;,")
BULLET_PREFIX_CHARACTERS = frozenset("-*•·▪◦>–—")
SHORTCUT_AUDIT_ONLY_FEATURE_NAMES = [
    "same_market_bool",
    "same_source_dataset_bool",
]
LEGACY18_FEATURE_NAMES = [
    "clean_category_jaccard",
    "clean_shared_title_bool",
    "clean_shared_description_bool",
    "clean_shared_title_count_capped",
    "clean_shared_description_count_capped",
    "clean_shared_category_count_capped",
    "clean_shared_title_idf_sum",
    "clean_shared_description_idf_sum",
    "clean_shared_title_idf_mean",
    "clean_shared_description_idf_mean",
    "item_count_train_percentile_gap_abs",
    "title_length_median_train_percentile_gap_abs",
    "description_length_median_train_percentile_gap_abs",
    "digit_ratio_mean_train_percentile_gap_abs",
    "punct_ratio_mean_train_percentile_gap_abs",
    "repeated_title_share_train_percentile_gap_abs",
    "repeated_description_share_train_percentile_gap_abs",
    "max_category_share_train_percentile_gap_abs",
]
IMPLEMENTATION_HASH_RE = re.compile(r"[0-9a-f]{64}")
C1_WINDOWS_1252_REPAIRS: dict[int, str] = {}
for _codepoint in range(0x80, 0xA0):
    try:
        C1_WINDOWS_1252_REPAIRS[_codepoint] = bytes([_codepoint]).decode("cp1252")
    except UnicodeDecodeError:
        continue


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def v4_contextual_alias_deletion_tokens(
    registry: set[str] | frozenset[str],
) -> set[str]:
    """Build cue-gated one-character omissions without audited content words."""

    parent_tokens = source.contextual_alias_deletion_tokens(registry)
    return set(parent_tokens) - V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validated_fixed_final_audit_content_collisions(
    contract: dict | None,
) -> list[dict]:
    """Validate and return exact snapshot-pinned audit false positives."""

    if contract is None:
        return []
    if set(contract) != {
        "review_basis",
        "expected_match_count",
        "collisions",
        "labels_or_evidence_types_read",
    }:
        raise ValueError(
            "Step7-v4 fixed final-audit content-collision schema drift"
        )
    entries = contract.get("collisions")
    if (
        contract.get("review_basis")
        != FIXED_FINAL_AUDIT_CONTENT_COLLISION_REVIEW_BASIS
        or contract.get("labels_or_evidence_types_read") is not False
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError(
            "Step7-v4 fixed final-audit content-collision contract drift"
        )
    keys = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != FIXED_FINAL_AUDIT_CONTENT_COLLISION_FIELDS
            or entry.get("rule_name")
            not in FIXED_FINAL_AUDIT_CONTENT_COLLISION_ALLOWED_RULE_NAMES
            or type(entry.get("expected_match_count")) is not int
            or int(entry["expected_match_count"]) <= 0
            or not isinstance(entry.get("reason"), str)
            or not entry["reason"]
            or any(
                IMPLEMENTATION_HASH_RE.fullmatch(str(entry.get(field, "")))
                is None
                for field in (
                    "seller_uid_sha256",
                    "clean_text_sha256",
                    "matched_surface_sha256",
                )
            )
        ):
            raise ValueError(
                "Step7-v4 fixed final-audit content-collision entry drift"
            )
        keys.append(
            (
                entry["rule_name"],
                entry["seller_uid_sha256"],
                entry["clean_text_sha256"],
                entry["matched_surface_sha256"],
            )
        )
    if (
        len(keys) != len(set(keys))
        or entries
        != sorted(
            entries,
            key=lambda entry: (
                entry["rule_name"],
                entry["seller_uid_sha256"],
                entry["clean_text_sha256"],
                entry["matched_surface_sha256"],
            ),
        )
        or int(contract.get("expected_match_count", -1))
        != sum(int(entry["expected_match_count"]) for entry in entries)
    ):
        raise ValueError(
            "Step7-v4 fixed final-audit content-collision inventory drift"
        )
    return entries


def verify_canonical_self_hash(payload: dict, field: str, role: str) -> str:
    observed = payload.get(field)
    without_hash = dict(payload)
    without_hash.pop(field, None)
    if observed != canonical_hash(without_hash):
        raise ValueError(f"Step7-v4 {role} self-hash drift")
    return str(observed)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def render_csv(rows: list[dict], fieldnames: list[str] | None = None) -> bytes:
    if not rows and not fieldnames:
        raise ValueError("Step7-v4 cannot infer an empty CSV schema")
    columns = list(fieldnames or rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != columns:
            raise ValueError("Step7-v4 CSV row schema/order drift")
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                f"Step7-v4 refuses to overwrite a non-identical artifact: {path}"
            )
        return
    path.write_bytes(payload)


def write_csv_immutable(
    path: Path, rows: list[dict], fieldnames: list[str] | None = None
) -> None:
    write_bytes_immutable(path, render_csv(rows, fieldnames))


def write_json_immutable(path: Path, payload: dict) -> None:
    rendered = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    write_bytes_immutable(path, rendered)


def write_jsonl_immutable(path: Path, rows: Iterable[dict]) -> None:
    buffer = io.StringIO()
    for row in rows:
        buffer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        buffer.write("\n")
    write_bytes_immutable(path, buffer.getvalue().encode("utf-8"))


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step7-v4 artifact: {path}")
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file_record(record: dict, role: str) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"Step7-v4 file-record schema drift: {role}")
    path = resolve(record["path"])
    observed = file_record(path)
    if observed != record:
        raise ValueError(
            f"Step7-v4 file-record drift: role={role} expected={record} "
            f"observed={observed}"
        )
    return path


def load_policy(path: Path = DEFAULT_POLICY, *, require_frozen: bool = True) -> dict:
    policy = load_json(path)
    validate_policy(policy, require_frozen=require_frozen)
    return policy


def stylometry_feature_names() -> list[str]:
    return [
        f"style_{field}_{statistic}_abs_gap"
        for field in FIELD_NAMES
        for statistic in STYLOMETRY_STATISTICS
    ]


def encoder_feature_names(model_cfg: dict) -> list[str]:
    prefix = str(model_cfg["feature_prefix"])
    return [f"{prefix}_{suffix}" for suffix in AGGREGATE_SUFFIXES]


def frequency_audit_feature_names(model_cfg: dict) -> list[str]:
    return [f"{name}__multiplicity_weighted_audit" for name in encoder_feature_names(model_cfg)]


def feature_blocks(policy: dict) -> dict[str, list[str]]:
    blocks = {
        "legacy18": list(LEGACY18_FEATURE_NAMES),
        "stylometry22": stylometry_feature_names(),
    }
    for block_name, model_key in BLOCK_MODEL_KEYS.items():
        blocks[block_name] = encoder_feature_names(
            policy["embedding_models"][model_key]
        )
    return blocks


def candidate_specs(policy: dict) -> list[dict]:
    blocks = feature_blocks(policy)
    output = []
    for candidate in policy["candidates"]:
        names = [
            name
            for block_name in candidate["blocks"]
            for name in blocks[block_name]
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Step7-v4 candidate contains duplicate features: {candidate['id']}"
            )
        output.append({**candidate, "feature_names": names})
    return output


def validate_policy(policy: dict, *, require_frozen: bool = True) -> None:
    if policy.get("version") != EXPECTED_VERSION:
        raise ValueError("Step7-v4 policy version drift")
    implementation = policy.get("implementation", {})
    expected_implementation_keys = {
        "common",
        "preparation",
        "sync_builder",
        "workspace_materializer",
        "encoder",
        "selector",
        "runner",
        "redaction_module",
        "parent_preparation",
        "selection_solver",
    }
    if set(implementation) != expected_implementation_keys:
        raise ValueError("Step7-v4 implementation universe drift")
    for role, record in implementation.items():
        if set(record) != {"path", "sha256"} or not str(record["path"]):
            raise ValueError(f"Step7-v4 implementation record drift: {role}")
        digest = str(record["sha256"])
        if require_frozen and IMPLEMENTATION_HASH_RE.fullmatch(digest) is None:
            raise ValueError(f"Step7-v4 implementation hash is not frozen: {role}")

    required_inputs = {
        "parent_source_policy",
        "item_manifest",
        "market_item_snapshot",
        "agora_snapshot",
        "parent_pair_manifest",
        "parent_safe_features",
        "parent_train_labels",
        "parent_valid_labels",
    }
    if set(policy.get("inputs", {})) != required_inputs:
        raise ValueError("Step7-v4 input universe drift")
    for role, record in policy["inputs"].items():
        if set(record) != {"path", "sha256"}:
            raise ValueError(f"Step7-v4 input schema drift: {role}")
        if IMPLEMENTATION_HASH_RE.fullmatch(str(record["sha256"])) is None:
            raise ValueError(f"Step7-v4 input hash drift: {role}")

    quarantine = policy.get("parent_fragment_quarantine", {})
    expected_quarantine_keys = {
        "reason",
        "decision_basis",
        "expected_parent_pair_count",
        "excluded_pair_count",
        "pair_uid_sha256",
        "split_name",
        "component_id",
        "seller_uid_sha256",
        "must_be_an_isolated_component",
        "raw_source_dataset",
        "raw_rows",
    }
    if set(quarantine) != expected_quarantine_keys:
        raise ValueError("Step7-v4 parent-fragment quarantine schema drift")
    seller_hashes = quarantine["seller_uid_sha256"]
    if (
        quarantine["decision_basis"]
        != "label_free_exact_raw_workbook_structure_review"
        or int(quarantine["expected_parent_pair_count"]) <= 0
        or int(quarantine["excluded_pair_count"]) != 1
        or quarantine["split_name"] != "valid"
        or not str(quarantine["component_id"])
        or quarantine["must_be_an_isolated_component"] is not True
        or IMPLEMENTATION_HASH_RE.fullmatch(
            str(quarantine["pair_uid_sha256"])
        )
        is None
        or not isinstance(seller_hashes, list)
        or len(seller_hashes) != 2
        or seller_hashes != sorted(set(seller_hashes))
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(value)) is None
            for value in seller_hashes
        )
    ):
        raise ValueError("Step7-v4 parent-fragment quarantine contract drift")
    raw_quarantine_rows = quarantine["raw_rows"]
    expected_raw_quarantine_keys = {
        "source_row_number",
        "seller_uid_sha256",
        "item_uid_sha256",
        "canonical_cell_values_sha256",
        "cell_string_lengths",
        "structural_failure",
    }
    if (
        not isinstance(raw_quarantine_rows, list)
        or len(raw_quarantine_rows) != 2
        or [int(row["source_row_number"]) for row in raw_quarantine_rows]
        != sorted({int(row["source_row_number"]) for row in raw_quarantine_rows})
        or {row.get("seller_uid_sha256") for row in raw_quarantine_rows}
        != set(seller_hashes)
    ):
        raise ValueError("Step7-v4 quarantined raw-row universe drift")
    for row in raw_quarantine_rows:
        lengths = row.get("cell_string_lengths")
        if (
            set(row) != expected_raw_quarantine_keys
            or int(row["source_row_number"]) < 2
            or any(
                IMPLEMENTATION_HASH_RE.fullmatch(str(row[name])) is None
                for name in (
                    "seller_uid_sha256",
                    "item_uid_sha256",
                    "canonical_cell_values_sha256",
                )
            )
            or not isinstance(lengths, list)
            or len(lengths) != 9
            or any(type(value) is not int or value < 0 for value in lengths)
            or not str(row.get("structural_failure", ""))
        ):
            raise ValueError("Step7-v4 quarantined raw-row contract drift")

    boundary = policy["supervision_boundary"]
    expected_counts = boundary["expected_counts"]
    if (
        any(
            expected_counts[split]["positive"]
            + expected_counts[split]["negative"]
            != expected_counts[split]["total"]
            for split in ("train", "valid", "test")
        )
        or sum(
            expected_counts[split]["total"]
            for split in ("train", "valid", "test")
        )
        != expected_counts["total"]
        or int(quarantine["expected_parent_pair_count"])
        - int(quarantine["excluded_pair_count"])
        != int(expected_counts["total"])
        or sum(boundary["expected_seller_count_by_split"].values())
        != int(boundary["expected_total_unique_sellers"])
    ):
        raise ValueError("Step7-v4 split counts do not sum")
    if boundary["historical_test_labels_may_be_materialized"]:
        raise ValueError("Step7-v4 historical test labels must remain unopened")

    raw = policy["raw_item_boundary"]
    if quarantine["raw_source_dataset"] not in raw["allowed_source_datasets"]:
        raise ValueError("Step7-v4 quarantine raw source is outside the boundary")
    if raw["required_fields"] != FIELD_NAMES:
        raise ValueError("Step7-v4 raw field order drift")
    if raw["expected_selected_item_count"] != sum(
        raw["expected_selected_item_count_by_source"].values()
    ):
        raise ValueError("Step7-v4 raw item counts do not sum")
    if sorted(raw["allowed_source_datasets"]) != sorted(
        raw["expected_selected_item_count_by_source"]
    ):
        raise ValueError("Step7-v4 raw source universe drift")
    if (
        len(raw["market_item_exact_header"])
        != len(raw["market_item_column_order"])
        or len(raw["agora_exact_header"]) != len(raw["agora_column_order"])
    ):
        raise ValueError("Step7-v4 exact workbook header width drift")

    clean = policy["clean_text_contract"]
    validated_fixed_final_audit_content_collisions(
        clean.get("fixed_final_audit_content_collision_handling")
    )
    collision = clean.get("seller_local_content_collision_handling", {})
    fuzzy_collision = clean.get(
        "one_character_omission_content_collision_handling", {}
    )
    expected_collision_keys = {
        "collision_terms_source",
        "collision_compact_count",
        "collision_compact_registry_canonical_sha256",
        "expected_removed_from_global_contextual_registry_count",
        "expected_removed_from_global_contextual_registry_canonical_sha256",
        "unconditional_local_redaction_allowed",
        "identity_context_gated_redaction_enabled",
        "one_character_omission_matching_allowed",
        "uncued_ambiguous_occurrences_are_retained_and_audited",
        "expected_affected_seller_count",
        "expected_affected_seller_uid_to_tokens_canonical_sha256",
    }
    if (
        clean.get("protected_content_occurrence_matching")
        != (
            "case_insensitive_exact_surface_with_ascii_alphanumeric_boundaries_"
            "excluding_pre_redaction_high_confidence_identity_spans"
        )
        or clean.get("protected_content_retention_aggregation")
        != (
            "sum_per_source_item_field_occurrence_before_corpus_aggregation_"
            "no_cross_field_cancellation"
        )
        or clean.get("non_pgp_identity_matching_scope")
        != (
            "single_logical_line_except_two_fixed_split_plus_phone_rules_"
            "no_other_non_pgp_lf_or_cr_crossing"
        )
        or clean.get("identity_match_evaluation")
        != (
            "single_non_cascading_sweep_on_normalized_original_text_then_"
            "merge_spans_and_redact"
        )
        or clean.get("post_redaction_fuzzy_alias_audit")
        != (
            "exact_plural_and_identity_suffix_contextual_forms_fail_closed_"
            "but_one_character_omission_forms_are_censused_as_new_adjacency_"
            "collisions_not_identity_residue"
        )
        or set(fuzzy_collision)
        != {
            "review_basis",
            "denied_surface_tokens",
            "denied_surface_tokens_canonical_sha256",
            "expected_removed_token_count",
            "source_alias_anchor_sha256s",
            "expected_retained_surface_sha256_counts",
            "expected_retained_match_occurrence_count",
            "labels_or_evidence_types_read",
        }
        or fuzzy_collision.get("review_basis")
        != (
            "fixed_english_raw_snapshot_content_collision_review_before_"
            "model_training_without_pair_labels_or_evidence_types"
        )
        or fuzzy_collision.get("denied_surface_tokens")
        != sorted(V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST)
        or fuzzy_collision.get("denied_surface_tokens_canonical_sha256")
        != canonical_hash(sorted(V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST))
        or fuzzy_collision.get("expected_removed_token_count")
        != len(V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST)
        or len(fuzzy_collision.get("source_alias_anchor_sha256s", []))
        != len(V4_CONTEXTUAL_ALIAS_DELETION_CONTENT_DENYLIST)
        or fuzzy_collision.get("source_alias_anchor_sha256s")
        != sorted(fuzzy_collision.get("source_alias_anchor_sha256s", []))
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(value)) is None
            for value in fuzzy_collision.get("source_alias_anchor_sha256s", [])
        )
        or len(fuzzy_collision.get("expected_retained_surface_sha256_counts", {}))
        <= 0
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(key)) is None
            or type(value) is not int
            or value <= 0
            for key, value in fuzzy_collision.get(
                "expected_retained_surface_sha256_counts", {}
            ).items()
        )
        or fuzzy_collision.get("expected_retained_match_occurrence_count")
        != sum(
            fuzzy_collision.get(
                "expected_retained_surface_sha256_counts", {}
            ).values()
        )
        or fuzzy_collision.get("labels_or_evidence_types_read") is not False
        or clean.get("identity_bridge_evaluation")
        != (
            "only_mask_preexisting_seller_local_identity_spans_to_detect_"
            "audited_global_aliases_no_other_cascade"
        )
        or clean.get("multiline_identity_redaction_rule_allowlist")
        != sorted(MULTILINE_IDENTITY_REDACTION_RULE_NAMES)
        or clean.get("v4_additional_identity_rule_names")
        != [name for name, _pattern in V4_ADDITIONAL_IDENTITY_RULES]
        or set(collision) != expected_collision_keys
        or collision.get("collision_terms_source")
        != (
            "union_of_parent_protected_content_words_and_"
            "protected_identity_collision_terms"
        )
        or int(collision.get("collision_compact_count", 0)) <= 0
        or int(collision.get("expected_affected_seller_count", 0)) <= 0
        or int(
            collision.get(
                "expected_removed_from_global_contextual_registry_count", 0
            )
        )
        <= 0
        or collision.get("unconditional_local_redaction_allowed") is not False
        or collision.get("identity_context_gated_redaction_enabled") is not True
        or collision.get("one_character_omission_matching_allowed") is not False
        or collision.get("uncued_ambiguous_occurrences_are_retained_and_audited")
        is not True
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(collision.get(key, ""))) is None
            for key in (
                "collision_compact_registry_canonical_sha256",
                "expected_removed_from_global_contextual_registry_canonical_sha256",
                "expected_affected_seller_uid_to_tokens_canonical_sha256",
            )
        )
    ):
        raise ValueError("Step7-v4 seller-local content-collision contract drift")

    stylometry = policy["stylometry"]
    if stylometry["fields"] != FIELD_NAMES:
        raise ValueError("Step7-v4 stylometry field order drift")
    if stylometry["statistics"] != STYLOMETRY_STATISTICS:
        raise ValueError("Step7-v4 stylometry statistic order drift")
    if len(stylometry_feature_names()) != 22:
        raise AssertionError("Step7-v4 stylometry feature count drift")

    chunking = policy["shared_chunking"]
    if (
        int(chunking["token_budget_including_model_prefix_and_special_tokens"])
        != SHARED_TOKEN_BUDGET
    ):
        raise ValueError("Step7-v4 chunk token budget drift")
    if (
        not chunking["all_four_tokenizers_must_fit_every_chunk"]
        or not chunking["same_exact_chunk_text_and_order_for_every_encoder"]
        or not chunking["exact_character_reconstruction_per_unique_text"]
        or chunking["sampling_or_dropping_allowed"]
        or chunking["maximum_chunks_per_text"] is not None
    ):
        raise ValueError("Step7-v4 shared chunk completeness was relaxed")

    models = policy["embedding_models"]
    if list(models) != MODEL_KEYS:
        raise ValueError("Step7-v4 model order/universe drift")
    prefixes = set()
    for model_key, cfg in models.items():
        expected_native_max = MODEL_NATIVE_MAX_SEQ_LENGTHS[model_key]
        if (
            int(cfg["native_max_seq_length"]) != expected_native_max
            or int(cfg["expected_dimension"]) <= 0
        ):
            raise ValueError(f"Step7-v4 model dimensional contract drift: {model_key}")
        if SHARED_TOKEN_BUDGET > expected_native_max:
            raise ValueError(
                f"Step7-v4 shared chunks exceed the native model window: {model_key}"
            )
        if cfg.get("sentence_transformer_prompt") != "":
            raise ValueError(
                f"Step7-v4 hidden SentenceTransformer prompt is forbidden: {model_key}"
            )
        if int(cfg["expected_file_count"]) <= 0 or int(cfg["expected_total_size_bytes"]) <= 0:
            raise ValueError(f"Step7-v4 model payload contract drift: {model_key}")
        if IMPLEMENTATION_HASH_RE.fullmatch(str(cfg["expected_content_sha256"])) is None:
            raise ValueError(f"Step7-v4 model content hash drift: {model_key}")
        if cfg["feature_prefix"] in prefixes:
            raise ValueError("Step7-v4 model feature prefixes collide")
        prefixes.add(cfg["feature_prefix"])

    aggregation = policy["aggregation"]
    if aggregation["aggregate_suffixes"] != AGGREGATE_SUFFIXES:
        raise ValueError("Step7-v4 aggregate order drift")
    if int(aggregation["top_k_item_matches"]) != 3:
        raise ValueError("Step7-v4 top-k aggregation drift")
    if int(aggregation["similarity_block_rows"]) <= 0:
        raise ValueError("Step7-v4 similarity block size is invalid")
    if aggregation["unique_text_weighting_primary"] != "equal_weight_per_unique_clean_text":
        raise ValueError("Step7-v4 primary multiplicity isolation was relaxed")
    gpu_execution = policy["gpu_execution"]
    if (
        int(gpu_execution["random_seed"]) <= 0
        or gpu_execution.get("required_sentence_transformers_version")
        != REQUIRED_SENTENCE_TRANSFORMERS_VERSION
        or gpu_execution["expected_runtime"].get(
            "deterministic_algorithms_enabled"
        )
        is not True
        or gpu_execution["expected_runtime"].get("cuda_matmul_allow_tf32")
        is not False
        or gpu_execution["expected_runtime"].get("cudnn_allow_tf32") is not False
    ):
        raise ValueError("Step7-v4 deterministic GPU contract was relaxed")

    declared_blocks = policy["feature_blocks"]
    if set(declared_blocks) != {
        "legacy18",
        "stylometry22",
        "pcm6",
        "mstyle6",
        "e5_6",
        "labse6",
    }:
        raise ValueError("Step7-v4 feature-block universe drift")
    blocks = feature_blocks(policy)
    candidates = policy["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)) or not ids or ids[0] != "control__intercept":
        raise ValueError("Step7-v4 candidate IDs/order drift")
    for candidate in candidates:
        if set(candidate) != {"id", "blocks", "role"}:
            raise ValueError(f"Step7-v4 candidate schema drift: {candidate}")
        if any(block not in blocks for block in candidate["blocks"]):
            raise ValueError(f"Step7-v4 candidate uses unknown block: {candidate['id']}")
    candidate_specs(policy)
    candidate_blocks = {candidate["id"]: candidate["blocks"] for candidate in candidates}
    for raw_id, matched_id, block_name in (
        ("encoder__e5", "matched__e5_stylometry", "e5_6"),
        ("encoder__labse", "matched__labse_stylometry", "labse6"),
        ("style__pcm", "style__pcm_stylometry", "pcm6"),
        ("style__mstyle", "style__mstyle_stylometry", "mstyle6"),
    ):
        if candidate_blocks.get(raw_id) != [block_name] or candidate_blocks.get(
            matched_id
        ) != [block_name, "stylometry22"]:
            raise ValueError("Step7-v4 encoder/stylometry comparison is not matched")
    if candidate_blocks.get("control__legacy_stylometry") != [
        "legacy18",
        "stylometry22",
    ]:
        raise ValueError("Step7-v4 legacy/stylometry matched control drift")
    for candidate_id, block_name in (
        ("fusion__legacy_e5_stylometry", "e5_6"),
        ("fusion__legacy_labse_stylometry", "labse6"),
        ("fusion__legacy_pcm_stylometry", "pcm6"),
        ("fusion__legacy_mstyle_stylometry", "mstyle6"),
    ):
        if candidate_blocks.get(candidate_id) != [
            "legacy18",
            block_name,
            "stylometry22",
        ]:
            raise ValueError("Step7-v4 legacy encoder comparison is not matched")
    for candidate_id, semantic_block in (
        ("fusion__legacy_style_e5", "e5_6"),
        ("fusion__legacy_style_labse", "labse6"),
    ):
        if candidate_blocks.get(candidate_id) != [
            "legacy18",
            "pcm6",
            "mstyle6",
            "stylometry22",
            semantic_block,
        ]:
            raise ValueError("Step7-v4 full style/semantic fusion comparison drift")

    training = policy["training"]
    if (
        int(training["outer_fold_count"]) != 5
        or len(training["outer_seeds"]) != 5
        or len(set(training["outer_seeds"])) != 5
        or int(training["inner_fold_count"]) != 4
        or training["evidence_type_training_weight"] != "forbidden"
        or training["boundary_optimum_is_failure"]
        or training.get("l2_parameterization")
        != "weighted_mean_logloss_plus_half_l2_squared_coefficient_norm"
    ):
        raise ValueError("Step7-v4 nested-training discipline drift")
    l2_grid = [float(value) for value in training["l2_initial_grid"]]
    if l2_grid != sorted(set(l2_grid)) or any(value <= 0.0 for value in l2_grid):
        raise ValueError("Step7-v4 L2 grid is invalid")
    if not (
        float(training["l2_minimum"]) < l2_grid[0]
        and float(training["l2_maximum"]) > l2_grid[-1]
    ):
        raise ValueError("Step7-v4 adaptive L2 limits do not extend the grid")
    bootstrap = policy["evaluation"]["bootstrap"]
    if (
        int(bootstrap["resamples"]) <= 0
        or not 0.0 < float(bootstrap["confidence"]) < 1.0
    ):
        raise ValueError("Step7-v4 bootstrap contract is invalid")
    unique_rule = policy["selection_rule"]["unique_provisional_m0_requires"]
    expected_unique_keys = {
        "winner_rate_across_outer_repeats_at_least",
        "component_bootstrap_probability_delta_above_runner_up_at_least",
        "simultaneous_component_bootstrap_probability_winner_above_all_candidates_at_least",
        "no_exact_clone_nested_winner_rate_across_outer_repeats_at_least",
        "no_exact_clone_component_bootstrap_probability_winner_above_all_candidates_at_least",
        "all_formal_fits_converged",
    }
    if set(unique_rule) != expected_unique_keys or any(
        not 0.0 <= float(unique_rule[key]) <= 1.0
        for key in expected_unique_keys - {"all_formal_fits_converged"}
    ):
        raise ValueError("Step7-v4 provisional-M0 gate drift")
    increment_rule = policy["selection_rule"]["style_increment_claim_requires"]
    if set(increment_rule) != {
        "preregistered_primary_style_candidate_component_bootstrap_ci_lower_above_simple_style_control",
        "preregistered_primary_style_candidate_component_bootstrap_ci_lower_above_best_semantic_control",
    } or any(float(value) != 0.0 for value in increment_rule.values()):
        raise ValueError("Step7-v4 preregistered style-increment gate drift")

    outputs = policy["outputs"]
    expected_output_keys = {
        "root",
        *PUBLIC_OUTPUT_ROLES,
        *PRIVATE_OUTPUT_ROLES,
        "preparation_manifest",
        "development_labels_manifest",
        "gpu_sync_manifest",
        "shared_chunks",
        "shared_chunks_manifest",
        "model_runtime_manifest_template",
        "pair_scores_template",
        "gpu_output_manifest",
        "selection_summary",
        "train_selection_lock",
        "model_artifacts",
        "train_oof_predictions",
        "blind_valid_predictions",
        "blind_scoring_lock",
        "valid_predictions",
    }
    if set(outputs) != expected_output_keys:
        raise ValueError("Step7-v4 output universe drift")
    root = str(outputs["root"]).rstrip("/")
    if not root.startswith("reports/step7_v4_raw_item_authorship_selection/"):
        raise ValueError("Step7-v4 output root drift")
    for key, value in outputs.items():
        if key == "root":
            continue
        if not str(value).startswith(root + "/"):
            raise ValueError(f"Step7-v4 output escapes versioned root: {key}")
    output_paths = [str(value) for key, value in outputs.items() if key != "root"]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Step7-v4 output roles collide on one path")


def verify_implementation_files(
    policy: dict, names: Iterable[str] | None = None
) -> dict[str, dict]:
    records = {}
    selected = (
        sorted(policy["implementation"])
        if names is None
        else sorted(str(name) for name in names)
    )
    if not set(selected).issubset(policy["implementation"]):
        raise ValueError("Step7-v4 requested an unknown implementation role")
    for role in selected:
        expected = policy["implementation"][role]
        path = resolve(expected["path"])
        observed = file_record(path)
        if observed["sha256"] != expected["sha256"]:
            raise ValueError(
                f"Step7-v4 implementation drift: role={role} "
                f"expected={expected['sha256']} observed={observed['sha256']}"
            )
        records[role] = observed
    return records


def verify_inputs(policy: dict, names: Iterable[str] | None = None) -> dict[str, dict]:
    selected = list(names) if names is not None else list(policy["inputs"])
    output = {}
    for name in selected:
        spec = policy["inputs"][name]
        path = resolve(spec["path"])
        observed = file_record(path)
        if observed["sha256"] != str(spec["sha256"]).casefold():
            raise ValueError(
                f"Step7-v4 input drift: role={name} expected={spec['sha256']} "
                f"observed={observed['sha256']}"
            )
        output[name] = observed
    return output


def _repair_valid_utf8_mojibake_sequences(text: str) -> tuple[str, int]:
    output = []
    repair_count = 0
    index = 0
    while index < len(text):
        first = ord(text[index])
        expected_length = 0
        if 0xC2 <= first <= 0xDF:
            expected_length = 2
        elif 0xE0 <= first <= 0xEF:
            expected_length = 3
        elif 0xF0 <= first <= 0xF4:
            expected_length = 4
        if expected_length and index + expected_length <= len(text):
            codepoints = [ord(character) for character in text[index : index + expected_length]]
            if all(value <= 0xFF for value in codepoints) and all(
                0x80 <= value <= 0xBF for value in codepoints[1:]
            ):
                try:
                    decoded = bytes(codepoints).decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    pass
                else:
                    output.append(decoded)
                    repair_count += 1
                    index += expected_length
                    continue
        output.append(text[index])
        index += 1
    return "".join(output), repair_count


def _normalize_raw_field_with_diagnostics(value: object) -> tuple[str, dict]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "", {
            "utf8_mojibake_sequence_repair_count": 0,
            "windows_1252_c1_repair_count": 0,
        }
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text, utf8_repair_count = _repair_valid_utf8_mojibake_sequences(text)
    repaired = []
    c1_repair_count = 0
    for character in text:
        codepoint = ord(character)
        if 0x80 <= codepoint <= 0x9F:
            replacement = C1_WINDOWS_1252_REPAIRS.get(codepoint)
            if replacement is None:
                raise ValueError(
                    "Step7-v4 raw text contains an undefined Windows-1252 C1 "
                    f"control: U+{codepoint:04X}"
                )
            repaired.append(replacement)
            c1_repair_count += 1
        else:
            repaired.append(character)
    text = "".join(repaired)
    text.encode("utf-8", errors="strict")
    unsupported = [
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}
    ]
    if unsupported:
        codepoints = sorted({f"U+{ord(character):04X}" for character in unsupported})
        raise ValueError(
            "Step7-v4 raw text contains unsupported control characters: "
            + ",".join(codepoints[:8])
        )
    return text.strip(), {
        "utf8_mojibake_sequence_repair_count": utf8_repair_count,
        "windows_1252_c1_repair_count": c1_repair_count,
    }


def normalize_raw_field(value: object) -> str:
    return _normalize_raw_field_with_diagnostics(value)[0]


def seller_identity_bridged_audited_alias_spans(
    line: str,
    seller_literals: Sequence[str],
    seller_phrase_tokens: set[str] | frozenset[str],
    audited_global_phrases: set[str] | frozenset[str],
) -> list[tuple[int, int, str]]:
    """Find audited aliases formed only by masking seller-local identity spans."""

    local_identity_spans = set()
    for literal in seller_literals:
        pattern = source.identity_literal_pattern(literal)
        local_identity_spans.update(
            (match.start(), match.end()) for match in pattern.finditer(line)
        )
    local_identity_spans.update(
        source.unconditional_alias_spans(line, seller_phrase_tokens)
    )
    if not local_identity_spans or not audited_global_phrases:
        return []
    masked_positions = []
    covered = [False] * len(line)
    for start, end in sorted(local_identity_spans):
        for index in range(start, end):
            covered[index] = True
    masked_characters = []
    for index, character in enumerate(line):
        if covered[index]:
            continue
        masked_characters.append(character)
        masked_positions.append(index)
    masked = "".join(masked_characters)
    bridged = []
    for masked_start, masked_end in source.unconditional_alias_spans(
        masked, audited_global_phrases
    ):
        if masked_start >= masked_end or not masked_positions:
            raise ValueError("Step7-v4 bridged audited alias span drift")
        start = masked_positions[masked_start]
        end = masked_positions[masked_end - 1] + 1
        if not any(
            local_start < end and local_end > start
            for local_start, local_end in local_identity_spans
        ):
            continue
        compact = source.compact_identifier(masked[masked_start:masked_end])
        anchor = source.anchored_alias_registry_token(
            compact, frozenset(audited_global_phrases)
        )
        if anchor is None:
            raise ValueError("Step7-v4 bridged audited alias anchor drift")
        bridged.append((start, end, anchor))
    return sorted(set(bridged))


def _redact_identifiers_non_cascading_line_bounded(
    text: str,
    *,
    seller_literals: Sequence[str],
    seller_phrase_tokens: set[str] | frozenset[str],
    global_tokens: set[str] | frozenset[str],
    contextual_aliases: set[str] | frozenset[str],
    contextual_alias_deletions: set[str] | frozenset[str],
    audited_global_phrases: set[str] | frozenset[str],
) -> tuple[str, dict]:
    """Locate all identity spans on the original text, then remove them once.

    Several inherited obfuscated-handle regexes intentionally accept arbitrary
    non-word separators.  In Python that includes ``\n``.  Applying those rules
    to a whole item lets a removed phone/handle expose the first product word on
    the same or following line and causes a later rule to consume it as a fake
    handle.  V4 therefore forbids cascading matches: every span is discovered
    against the same normalized input.  Only the fixed multiline allowlist may
    span lines: PGP material plus two fixed phone layouts covering either a
    contact-cued ``+`` before the line break or a strict
    ``+country-code(area-code)`` prefix before the line break.
    """

    all_rule_names = {
        name
        for name, _pattern in (
            *source.GENERIC_IDENTIFIER_RULES,
            *source.OBFUSCATED_CONTACT_RULES,
            *V4_ADDITIONAL_IDENTITY_RULES,
        )
    }
    if not MULTILINE_IDENTITY_REDACTION_RULE_NAMES.issubset(all_rule_names):
        raise ValueError("Step7-v4 multiline redaction allowlist drift")
    original = str(text or "")
    logical_lines: list[tuple[int, str]] = []
    offset = 0
    for line in original.split("\n"):
        logical_lines.append((offset, line))
        offset += len(line) + 1

    spans: list[tuple[int, int]] = []
    generic_match_count = 0
    for rule_name, pattern in (
        *source.GENERIC_IDENTIFIER_RULES,
        *source.OBFUSCATED_CONTACT_RULES,
        *V4_ADDITIONAL_IDENTITY_RULES,
    ):
        if rule_name in MULTILINE_IDENTITY_REDACTION_RULE_NAMES:
            matches = list(pattern.finditer(original))
            spans.extend((match.start(), match.end()) for match in matches)
            generic_match_count += len(matches)
        else:
            for line_offset, line in logical_lines:
                matches = list(pattern.finditer(line))
                spans.extend(
                    (line_offset + match.start(), line_offset + match.end())
                    for match in matches
                )
                generic_match_count += len(matches)

    audited_phrase_counts: Counter[str] = Counter()
    audited_phrase_match_count = 0
    for line_offset, line in logical_lines:
        matches = source.unconditional_alias_spans(
            line, audited_global_phrases
        )
        spans.extend(
            (line_offset + start, line_offset + end) for start, end in matches
        )
        audited_phrase_match_count += len(matches)
        audited_phrase_counts.update(
            source.compact_identifier(line[start:end]) for start, end in matches
        )
        bridged_matches = seller_identity_bridged_audited_alias_spans(
            line,
            seller_literals,
            seller_phrase_tokens,
            audited_global_phrases,
        )
        spans.extend(
            (line_offset + start, line_offset + end)
            for start, end, _anchor in bridged_matches
        )
        audited_phrase_match_count += len(bridged_matches)
        audited_phrase_counts.update(
            anchor for _start, _end, anchor in bridged_matches
        )

    seller_literal_match_count = 0
    for literal in seller_literals:
        pattern = source.identity_literal_pattern(literal)
        for line_offset, line in logical_lines:
            matches = list(pattern.finditer(line))
            spans.extend(
                (line_offset + match.start(), line_offset + match.end())
                for match in matches
            )
            seller_literal_match_count += len(matches)

    seller_phrase_match_count = 0
    for line_offset, line in logical_lines:
        matches = source.unconditional_alias_spans(line, seller_phrase_tokens)
        spans.extend(
            (line_offset + start, line_offset + end) for start, end in matches
        )
        seller_phrase_match_count += len(matches)

    global_identifier_counts: Counter[str] = Counter()
    global_identifier_match_count = 0
    for line_offset, line in logical_lines:
        for match in source.IDENTIFIER_TOKEN_RE.finditer(line):
            if not source.matches_global_identity_token(
                match.group(0), global_tokens
            ):
                continue
            spans.append(
                (line_offset + match.start(), line_offset + match.end())
            )
            global_identifier_match_count += 1
            global_identifier_counts[
                source.canonical_identifier_token(match.group(0))
            ] += 1

    contextual_alias_match_count = 0
    one_character_omission_surface_counts: Counter[str] = Counter()
    for line_offset, line in logical_lines:
        matches = source.contextual_alias_spans(
            line, contextual_aliases, contextual_alias_deletions
        )
        spans.extend(
            (
                line_offset + int(match["redact_start"]),
                line_offset + int(match["redact_end"]),
            )
            for match in matches
        )
        contextual_alias_match_count += len(matches)
        one_character_omission_surface_counts.update(
            str(match["compact_alias"])
            for match in matches
            if match["match_kind"] == "one_character_omission"
        )

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not 0 <= start < end <= len(original):
            raise ValueError("Step7-v4 identity redaction produced an invalid span")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    chunks = []
    cursor = 0
    for start, end in merged:
        chunks.extend((original[cursor:start], " "))
        cursor = end
    chunks.append(original[cursor:])
    clean = source.normalize_redacted_text("".join(chunks))
    return clean, {
        "generic_identifier_match_count": int(generic_match_count),
        "seller_local_alias_match_count": int(seller_literal_match_count),
        "seller_local_alias_phrase_match_count": int(
            seller_phrase_match_count
        ),
        "audited_global_identity_phrase_match_count": int(
            audited_phrase_match_count
        ),
        "audited_global_identity_phrase_counts": dict(
            sorted(audited_phrase_counts.items())
        ),
        "global_identifier_token_match_count": int(
            global_identifier_match_count
        ),
        "global_identifier_token_counts": dict(
            sorted(global_identifier_counts.items())
        ),
        "contextual_alias_match_count": int(contextual_alias_match_count),
        "one_character_omission_surface_counts": dict(
            sorted(one_character_omission_surface_counts.items())
        ),
        "redaction_pass_count": 1,
    }


def redact_raw_field(
    raw_text: object,
    *,
    seller_uid: str,
    seller_literals: Sequence[str],
    seller_phrase_tokens: set[str] | frozenset[str],
    global_tokens: set[str] | frozenset[str],
    contextual_aliases: set[str] | frozenset[str],
    contextual_alias_deletions: set[str] | frozenset[str],
    seller_contextual_collision_tokens: set[str] | frozenset[str],
    audited_global_phrases: set[str] | frozenset[str],
) -> tuple[str, dict]:
    normalized, normalization_diagnostics = _normalize_raw_field_with_diagnostics(
        raw_text
    )
    if not normalized:
        return "", {
            "one_character_omission_surface_counts": {},
            "post_redaction_one_character_omission_collision_count": 0,
            "raw_character_count": 0,
            "clean_character_count": 0,
            "empty_input": True,
            "empty_after_redaction": False,
            **normalization_diagnostics,
        }
    combined_contextual_aliases = frozenset(contextual_aliases) | frozenset(
        seller_contextual_collision_tokens
    )
    clean, diagnostics = _redact_identifiers_non_cascading_line_bounded(
        normalized,
        seller_literals=seller_literals,
        seller_phrase_tokens=seller_phrase_tokens,
        global_tokens=global_tokens,
        contextual_aliases=combined_contextual_aliases,
        contextual_alias_deletions=contextual_alias_deletions,
        audited_global_phrases=audited_global_phrases,
    )
    post_redaction_fuzzy_collision_count = 0
    if clean:
        for line in clean.split("\n"):
            if not line:
                continue
            source.assert_no_identifier_residue(
                line,
                seller_literals,
                seller_uid,
                global_tokens=global_tokens,
                seller_local_phrase_tokens=seller_phrase_tokens,
                audited_global_phrase_tokens=audited_global_phrases,
            )
            exact_contextual_matches = source.contextual_alias_spans(
                line, combined_contextual_aliases, frozenset()
            )
            if exact_contextual_matches:
                raise ValueError(
                    "Step7-v4 left an exact line-local context-gated known alias after "
                    f"redaction for seller_hash={sha256_text(seller_uid)[:16]}"
                )
            post_redaction_fuzzy_collision_count += sum(
                match["match_kind"] == "one_character_omission"
                for match in source.contextual_alias_spans(
                    line,
                    combined_contextual_aliases,
                    contextual_alias_deletions,
                )
            )
    return clean, {
        **diagnostics,
        "post_redaction_one_character_omission_collision_count": int(
            post_redaction_fuzzy_collision_count
        ),
        "raw_character_count": len(normalized),
        "clean_character_count": len(clean),
        "empty_input": False,
        "empty_after_redaction": not clean,
        **normalization_diagnostics,
    }


def _repeated_punctuation_character_count(text: str) -> int:
    count = 0
    run = 0
    for character in text + " ":
        if unicodedata.category(character).startswith("P"):
            run += 1
        else:
            if run >= 2:
                count += run
            run = 0
    return count


def _bullet_line_share(text: str) -> float:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_count = 0
    for line in lines:
        if line[0] in BULLET_PREFIX_CHARACTERS or re.match(r"^\d{1,3}[.)]\s", line):
            bullet_count += 1
    return float(bullet_count / len(lines))


def text_stylometry(text: str) -> dict[str, float]:
    value = str(text)
    if not value:
        raise ValueError("Step7-v4 stylometry requires nonempty clean text")
    length = len(value)
    categories = Counter(unicodedata.category(character)[0] for character in value)
    statistics = {
        "log1p_character_count": math.log1p(length),
        "digit_character_ratio": sum(character.isdigit() for character in value) / length,
        "punctuation_character_ratio": categories["P"] / length,
        "symbol_character_ratio": categories["S"] / length,
        "space_character_ratio": sum(character in {" ", "\t"} for character in value) / length,
        "newline_per_100_characters": 100.0 * value.count("\n") / length,
        "sentence_boundary_per_100_characters": 100.0
        * sum(character in SENTENCE_BOUNDARY_CHARACTERS for character in value)
        / length,
        "bracket_character_ratio": sum(
            unicodedata.category(character) in {"Ps", "Pe"} for character in value
        )
        / length,
        "delimiter_character_ratio": sum(
            character in DELIMITER_CHARACTERS for character in value
        )
        / length,
        "repeated_punctuation_character_ratio": _repeated_punctuation_character_count(value)
        / length,
        "bullet_line_share": _bullet_line_share(value),
    }
    if list(statistics) != STYLOMETRY_STATISTICS or not all(
        math.isfinite(number) for number in statistics.values()
    ):
        raise AssertionError("Step7-v4 stylometry output drift")
    return statistics


def build_seller_stylometry_rows(
    seller_text_rows: list[dict],
    unique_text_rows: list[dict],
    seller_split: dict[str, str],
) -> list[dict]:
    text_index = {row["text_uid"]: row["text"] for row in unique_text_rows}
    if len(text_index) != len(unique_text_rows):
        raise ValueError("Step7-v4 unique text corpus contains duplicate text_uid")
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    seen = set()
    for row in seller_text_rows:
        key = (row["seller_uid"], row["field_name"], row["text_uid"])
        if key in seen:
            raise ValueError("Step7-v4 seller text index contains a duplicate mapping")
        seen.add(key)
        if row["field_name"] not in FIELD_NAMES:
            raise ValueError("Step7-v4 seller text index contains an unknown field")
        if row["text_uid"] not in text_index:
            raise ValueError("Step7-v4 seller text index references a missing text")
        if int(row["multiplicity"]) <= 0:
            raise ValueError("Step7-v4 seller text multiplicity must be positive")
        grouped[(row["seller_uid"], row["field_name"])].append(
            text_stylometry(text_index[row["text_uid"]])
        )

    output = []
    for seller_uid in sorted(seller_split):
        row: dict[str, object] = {
            "seller_uid": seller_uid,
            "split_name": seller_split[seller_uid],
        }
        for field in FIELD_NAMES:
            values = grouped.get((seller_uid, field), [])
            for statistic in STYLOMETRY_STATISTICS:
                row[f"{field}_{statistic}"] = (
                    ""
                    if not values
                    else f"{float(np.median([item[statistic] for item in values])):.12f}"
                )
        output.append(row)
    return output


def build_pair_stylometry_rows(
    pair_rows: list[dict], seller_rows: list[dict]
) -> list[dict]:
    seller_index = {row["seller_uid"]: row for row in seller_rows}
    if len(seller_index) != len(seller_rows):
        raise ValueError("Step7-v4 seller stylometry contains duplicate sellers")
    output = []
    for pair in pair_rows:
        left = seller_index[pair["seller_uid_left"]]
        right = seller_index[pair["seller_uid_right"]]
        row: dict[str, object] = {"pair_uid": pair["pair_uid"]}
        for field in FIELD_NAMES:
            for statistic in STYLOMETRY_STATISTICS:
                source_name = f"{field}_{statistic}"
                target_name = f"style_{field}_{statistic}_abs_gap"
                if left[source_name] == "" or right[source_name] == "":
                    row[target_name] = ""
                else:
                    row[target_name] = (
                        f"{abs(float(left[source_name]) - float(right[source_name])):.12f}"
                    )
        if list(row)[1:] != stylometry_feature_names():
            raise AssertionError("Step7-v4 pair stylometry output order drift")
        output.append(row)
    return output


def model_content_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Step7-v4 model directory is missing: {path}")
    records = []
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        if ".cache" in item.parts or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        records.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    if not records:
        raise ValueError(f"Step7-v4 model directory is empty: {path}")
    return {
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "content_sha256": canonical_hash(records),
        "files": records,
    }


def validate_model_payload(model_key: str, cfg: dict) -> dict:
    observed = model_content_fingerprint(resolve(cfg["local_path"]))
    expected = {
        "file_count": int(cfg["expected_file_count"]),
        "total_size_bytes": int(cfg["expected_total_size_bytes"]),
        "content_sha256": str(cfg["expected_content_sha256"]).casefold(),
    }
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            raise ValueError(
                f"Step7-v4 model payload drift: model={model_key} field={field} "
                f"expected={expected_value} observed={observed[field]}"
            )
    return observed


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Step7-v4 encountered a zero or non-finite vector")
    return value / norm


def unit_mean(matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Step7-v4 unit mean requires a finite nonempty matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Step7-v4 unit mean encountered a zero vector")
    normalized = values / norms[:, None]
    if weights is None:
        mean = np.mean(normalized, axis=0)
    else:
        item_weights = np.asarray(weights, dtype=np.float64)
        if (
            item_weights.shape != (len(values),)
            or not np.all(np.isfinite(item_weights))
            or np.any(item_weights <= 0.0)
        ):
            raise ValueError("Step7-v4 unit mean weights are invalid")
        mean = np.average(normalized, axis=0, weights=item_weights)
    return _unit_vector(mean)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.dot(_unit_vector(left), _unit_vector(right)))
    if not math.isfinite(value):
        raise ValueError("Step7-v4 cosine is non-finite")
    return max(-1.0, min(1.0, value))


def _top_k_mean(values: np.ndarray, k: int) -> float:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Step7-v4 top-k requires a nonempty vector")
    selected = np.partition(values, len(values) - min(k, len(values)))[
        -min(k, len(values)) :
    ]
    return float(np.mean(selected))


def _top_k_mean_with_multiplicity(
    values: np.ndarray, multiplicities: np.ndarray, k: int
) -> float:
    if values.ndim != 1 or multiplicities.shape != values.shape or len(values) == 0:
        raise ValueError("Step7-v4 multiplicity top-k shape mismatch")
    remaining = int(k)
    total = 0.0
    used = 0
    # At most ``k`` distinct source rows can contribute to the first ``k``
    # replicated values.  Restricting the sort to those rows is exactly
    # equivalent to explicitly repeating every source row, but remains bounded
    # for sellers with very large raw-item multiplicities.
    selected_count = min(int(k), len(values))
    selected = np.argpartition(values, len(values) - selected_count)[
        -selected_count:
    ]
    order = selected[
        np.lexsort((selected, -values[selected]))
    ]
    for index in order:
        copies = min(remaining, int(multiplicities[index]))
        if copies > 0:
            total += float(values[index]) * copies
            used += copies
            remaining -= copies
        if remaining == 0:
            break
    if used == 0:
        raise ValueError("Step7-v4 multiplicity top-k selected no values")
    return total / used


def _merge_row_top_k(
    existing: np.ndarray | None, candidates: np.ndarray, k: int
) -> np.ndarray:
    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Step7-v4 top-k accumulator requires a nonempty matrix")
    combined = values if existing is None else np.concatenate((existing, values), axis=1)
    keep = min(int(k), combined.shape[1])
    if keep <= 0:
        raise ValueError("Step7-v4 top-k accumulator requires positive k")
    indices = np.argpartition(combined, combined.shape[1] - keep, axis=1)[
        :, -keep:
    ]
    return np.take_along_axis(combined, indices, axis=1)


def symmetric_top_k_cosine(
    left: np.ndarray,
    right: np.ndarray,
    k: int,
    *,
    left_multiplicities: np.ndarray | None = None,
    right_multiplicities: np.ndarray | None = None,
    similarity_block_rows: int = 256,
) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 2
        or right_values.ndim != 2
        or len(left_values) == 0
        or len(right_values) == 0
        or left_values.shape[1] != right_values.shape[1]
        or int(k) <= 0
        or int(similarity_block_rows) <= 0
    ):
        raise ValueError("Step7-v4 symmetric top-k matrix shape mismatch")
    left_norms = np.linalg.norm(left_values, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_values, axis=1, keepdims=True)
    if (
        not np.all(np.isfinite(left_values))
        or not np.all(np.isfinite(right_values))
        or np.any(left_norms <= 1e-12)
        or np.any(right_norms <= 1e-12)
    ):
        raise ValueError("Step7-v4 symmetric top-k encountered an invalid vector")
    left_unit = left_values / left_norms
    right_unit = right_values / right_norms

    weighted = not (
        left_multiplicities is None and right_multiplicities is None
    )
    if left_multiplicities is None or right_multiplicities is None:
        if weighted:
            raise ValueError("Step7-v4 multiplicity top-k requires both weight vectors")
        left_weights = right_weights = None
    else:
        left_weights = np.asarray(left_multiplicities, dtype=np.int64)
        right_weights = np.asarray(right_multiplicities, dtype=np.int64)
        if (
            left_weights.shape != (len(left_values),)
            or right_weights.shape != (len(right_values),)
            or np.any(left_weights <= 0)
            or np.any(right_weights <= 0)
        ):
            raise ValueError("Step7-v4 multiplicities are invalid")

    left_score_parts = []
    right_top_values: np.ndarray | None = None
    for start in range(0, len(left_unit), int(similarity_block_rows)):
        stop = min(len(left_unit), start + int(similarity_block_rows))
        similarities = np.clip(left_unit[start:stop] @ right_unit.T, -1.0, 1.0)
        if weighted:
            assert left_weights is not None and right_weights is not None
            left_score_parts.append(
                np.asarray(
                    [
                        _top_k_mean_with_multiplicity(row, right_weights, int(k))
                        for row in similarities
                    ],
                    dtype=np.float64,
                )
            )
            # A source item can occupy at most k slots in a top-k result, so
            # clipping and locally repeating multiplicities is exact while
            # keeping every block bounded by block_rows * k.
            repeated_indices = np.repeat(
                np.arange(stop - start),
                np.minimum(left_weights[start:stop], int(k)),
            )
            right_candidates = similarities[repeated_indices].T
        else:
            left_score_parts.append(
                np.asarray(
                    [_top_k_mean(row, int(k)) for row in similarities],
                    dtype=np.float64,
                )
            )
            right_candidates = similarities.T
        right_top_values = _merge_row_top_k(
            right_top_values, right_candidates, int(k)
        )

    left_scores = np.concatenate(left_score_parts)
    if len(left_scores) != len(left_values) or right_top_values is None:
        raise AssertionError("Step7-v4 batched top-k accumulation is incomplete")
    right_scores = np.mean(right_top_values, axis=1)
    if weighted:
        assert left_weights is not None and right_weights is not None
        value = (
            np.average(left_scores, weights=left_weights)
            + np.average(right_scores, weights=right_weights)
        ) / 2.0
    else:
        value = (np.mean(left_scores) + np.mean(right_scores)) / 2.0
    if not math.isfinite(float(value)):
        raise ValueError("Step7-v4 batched top-k result is non-finite")
    return float(value)


def _field_pair_aggregates(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    left_multiplicities: np.ndarray,
    right_multiplicities: np.ndarray,
    top_k: int,
    similarity_block_rows: int,
) -> tuple[dict[str, float], dict[str, float]]:
    primary = {
        "centroid_cosine": cosine(unit_mean(left_vectors), unit_mean(right_vectors)),
        "symmetric_top3_cosine": symmetric_top_k_cosine(
            left_vectors,
            right_vectors,
            top_k,
            similarity_block_rows=similarity_block_rows,
        ),
    }
    weighted = {
        "centroid_cosine": cosine(
            unit_mean(left_vectors, left_multiplicities),
            unit_mean(right_vectors, right_multiplicities),
        ),
        "symmetric_top3_cosine": symmetric_top_k_cosine(
            left_vectors,
            right_vectors,
            top_k,
            left_multiplicities=left_multiplicities,
            right_multiplicities=right_multiplicities,
            similarity_block_rows=similarity_block_rows,
        ),
    }
    return primary, weighted


def aggregate_pair_vectors(
    left: dict[str, tuple[np.ndarray, np.ndarray]],
    right: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    top_k: int,
    similarity_block_rows: int = 256,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    field_primary: dict[str, dict[str, float]] = {}
    field_weighted: dict[str, dict[str, float]] = {}
    for field in FIELD_NAMES:
        if field not in left or field not in right:
            continue
        left_vectors, left_multiplicities = left[field]
        right_vectors, right_multiplicities = right[field]
        primary, weighted = _field_pair_aggregates(
            left_vectors,
            right_vectors,
            left_multiplicities,
            right_multiplicities,
            top_k,
            similarity_block_rows,
        )
        field_primary[field] = primary
        field_weighted[field] = weighted

    if not field_primary:
        raise ValueError("Step7-v4 pair has no mutually present title or description")

    def flatten(values: dict[str, dict[str, float]]) -> dict[str, float | None]:
        centroid_values = [values[field]["centroid_cosine"] for field in FIELD_NAMES if field in values]
        top_values = [
            values[field]["symmetric_top3_cosine"] for field in FIELD_NAMES if field in values
        ]
        return {
            "field_equal_centroid_cosine": float(np.mean(centroid_values)),
            "field_equal_symmetric_top3_cosine": float(np.mean(top_values)),
            "title_centroid_cosine": (
                values["title"]["centroid_cosine"] if "title" in values else None
            ),
            "title_symmetric_top3_cosine": (
                values["title"]["symmetric_top3_cosine"] if "title" in values else None
            ),
            "description_centroid_cosine": (
                values["description"]["centroid_cosine"]
                if "description" in values
                else None
            ),
            "description_symmetric_top3_cosine": (
                values["description"]["symmetric_top3_cosine"]
                if "description" in values
                else None
            ),
        }

    primary_output = flatten(field_primary)
    weighted_output = flatten(field_weighted)
    if list(primary_output) != AGGREGATE_SUFFIXES or list(weighted_output) != AGGREGATE_SUFFIXES:
        raise AssertionError("Step7-v4 pair aggregate order drift")
    return primary_output, weighted_output


def text_vectors_from_chunks(
    chunk_rows: list[dict], embedding_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    matrix = np.asarray(embedding_matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(chunk_rows):
        raise ValueError("Step7-v4 chunk embedding row-count mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    expected_index: dict[str, int] = defaultdict(int)
    for row_index, row in enumerate(chunk_rows):
        text_uid = str(row["text_uid"])
        chunk_index = int(row["chunk_index"])
        if chunk_index != expected_index[text_uid]:
            raise ValueError("Step7-v4 chunk order is not contiguous per text")
        expected_index[text_uid] += 1
        grouped[text_uid].append(row_index)
    return {
        text_uid: unit_mean(matrix[np.asarray(indices, dtype=int)])
        for text_uid, indices in sorted(grouped.items())
    }


def seller_vector_index(
    seller_text_rows: list[dict], text_vectors: dict[str, np.ndarray]
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in seller_text_rows:
        text_uid = str(row["text_uid"])
        if text_uid not in text_vectors:
            raise ValueError("Step7-v4 seller mapping references a missing text vector")
        grouped[str(row["seller_uid"])][str(row["field_name"])].append(
            (text_uid, int(row["multiplicity"]))
        )
    output = {}
    for seller_uid, fields in sorted(grouped.items()):
        output[seller_uid] = {}
        for field, items in sorted(fields.items(), key=lambda item: FIELD_NAMES.index(item[0])):
            ordered = sorted(items)
            output[seller_uid][field] = (
                np.vstack([text_vectors[text_uid] for text_uid, _count in ordered]),
                np.asarray([count for _text_uid, count in ordered], dtype=np.int64),
            )
    return output


def compute_pair_score_rows(
    pair_rows: list[dict],
    seller_text_rows: list[dict],
    chunk_rows: list[dict],
    embedding_matrix: np.ndarray,
    model_cfg: dict,
    *,
    top_k: int,
    decimal_places: int,
    similarity_block_rows: int = 256,
) -> list[dict]:
    text_vectors = text_vectors_from_chunks(chunk_rows, embedding_matrix)
    sellers = seller_vector_index(seller_text_rows, text_vectors)
    feature_names = encoder_feature_names(model_cfg)
    audit_names = frequency_audit_feature_names(model_cfg)
    output = []
    for pair in pair_rows:
        left_uid = pair["seller_uid_left"]
        right_uid = pair["seller_uid_right"]
        if left_uid not in sellers or right_uid not in sellers:
            raise ValueError("Step7-v4 pair references a seller without vectors")
        primary, weighted = aggregate_pair_vectors(
            sellers[left_uid],
            sellers[right_uid],
            top_k=top_k,
            similarity_block_rows=similarity_block_rows,
        )
        row: dict[str, object] = {"pair_uid": pair["pair_uid"]}
        for feature_name, suffix in zip(feature_names, AGGREGATE_SUFFIXES, strict=True):
            value = primary[suffix]
            row[feature_name] = "" if value is None else f"{value:.{decimal_places}f}"
        for feature_name, suffix in zip(audit_names, AGGREGATE_SUFFIXES, strict=True):
            value = weighted[suffix]
            row[feature_name] = "" if value is None else f"{value:.{decimal_places}f}"
        output.append(row)
    return output


def build_opaque_gpu_indices(
    pair_rows: list[dict], seller_text_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Replace identity-bearing seller UIDs with stable ordinal GPU tokens."""

    sellers = sorted(
        {
            row[endpoint]
            for row in pair_rows
            for endpoint in ("seller_uid_left", "seller_uid_right")
        }
    )
    seller_tokens = {
        seller_uid: f"seller_{index:06d}"
        for index, seller_uid in enumerate(sellers, start=1)
    }
    gpu_pairs = [
        {
            "pair_uid": f"pair_{index:06d}",
            "seller_uid_left": seller_tokens[row["seller_uid_left"]],
            "seller_uid_right": seller_tokens[row["seller_uid_right"]],
        }
        for index, row in enumerate(pair_rows, start=1)
    ]
    gpu_sellers = [
        {
            "seller_uid": seller_tokens[row["seller_uid"]],
            "field_name": row["field_name"],
            "text_uid": row["text_uid"],
            "multiplicity": int(row["multiplicity"]),
        }
        for row in seller_text_rows
    ]
    if len({row["pair_uid"] for row in gpu_pairs}) != len(gpu_pairs):
        raise AssertionError("Step7-v4 opaque GPU pair tokens collide")
    if {row["seller_uid"] for row in gpu_sellers} != set(seller_tokens.values()):
        raise ValueError("Step7-v4 opaque GPU seller index is incomplete")
    return gpu_pairs, gpu_sellers


def exact_overlap_audit_by_pair(
    pair_rows: list[dict], seller_text_rows: list[dict]
) -> dict[str, dict[str, bool]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in seller_text_rows:
        grouped[row["seller_uid"]][row["field_name"]].add(row["text_uid"])
    output = {}
    for pair in pair_rows:
        left = grouped[pair["seller_uid_left"]]
        right = grouped[pair["seller_uid_right"]]
        title = bool(left["title"] & right["title"])
        description = bool(left["description"] & right["description"])
        output[pair["pair_uid"]] = {
            "exact_clean_title_overlap": title,
            "exact_clean_description_overlap": description,
            "any_exact_clean_text_overlap": title or description,
        }
    return output


def _compile_unconditional_alias_count(
    registry: set[str] | frozenset[str],
):
    """Compile the parent unconditional-alias matcher without changing semantics."""

    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry:
        return lambda _text: 0
    if any(
        re.fullmatch(r"[a-z0-9]{1,96}", token) is None
        for token in token_registry
    ):
        raise ValueError("Step7-v4 unconditional alias registry is invalid")
    forms: dict[str, str] = {token: token for token in token_registry}
    for token in sorted(token_registry):
        forms.setdefault(token + "s", token)
    for suffix in source.IDENTITY_HANDLE_SUFFIXES:
        for token in sorted(token_registry):
            forms.setdefault(token + suffix, token)
    maximum_form_length = max(
        len(token)
        + max(1, *(len(suffix) for suffix in source.IDENTITY_HANDLE_SUFFIXES))
        for token in token_registry
    )

    def count(text: str) -> int:
        words = list(source.UNCONDITIONAL_ALIAS_WORD_RE.finditer(text))
        matched_count = 0
        index = 0
        while index < len(words):
            matched_word_count = 0
            compact = ""
            for end_index in range(index, len(words)):
                if end_index > index:
                    broad_gap = text[
                        words[end_index - 1].end() : words[end_index].start()
                    ]
                    if (
                        source.CONTEXTUAL_ALIAS_PHRASE_GAP_RE.fullmatch(
                            broad_gap
                        )
                        is None
                    ):
                        break
                compact += source.compact_identifier(words[end_index].group(0))
                if len(compact) > maximum_form_length:
                    break
                anchor = forms.get(compact)
                if anchor is None:
                    continue
                gap_pattern = (
                    source.CONTEXTUAL_ALIAS_PHRASE_GAP_RE
                    if anchor in source.AUDITED_GLOBAL_IDENTITY_DOT_SEPARATOR_TOKENS
                    else source.UNCONDITIONAL_ALIAS_PHRASE_GAP_RE
                )
                if any(
                    gap_pattern.fullmatch(text[left.end() : right.start()]) is None
                    for left, right in zip(
                        words[index:end_index],
                        words[index + 1 : end_index + 1],
                    )
                ):
                    continue
                # The parent matcher intentionally retains the longest valid
                # match beginning at this word.
                matched_word_count = end_index - index + 1
            if matched_word_count:
                matched_count += 1
                index += matched_word_count
            else:
                index += 1
        return matched_count

    return count


def _compile_contextual_alias_count(
    registry: set[str] | frozenset[str],
    deletion_registry: set[str] | frozenset[str] | None,
):
    """Compile the parent cue-gated alias matcher for repeated corpus scans."""

    token_registry = frozenset(str(token) for token in registry)
    if not token_registry:
        return lambda _text: 0
    deletion_tokens = frozenset(str(token) for token in (deletion_registry or ()))
    forms: dict[str, str] = {token: "exact" for token in token_registry}
    for token in sorted(deletion_tokens):
        forms.setdefault(token, "one_character_omission")
    for token in sorted(token_registry):
        forms.setdefault(token + "s", "known_alias_plural")
    for suffix in source.IDENTITY_HANDLE_SUFFIXES:
        for token in sorted(token_registry):
            forms.setdefault(
                token + suffix, "known_alias_plus_identity_suffix"
            )

    def count(text: str) -> int:
        words = list(source.CONTEXTUAL_ALIAS_WORD_RE.finditer(text))
        matched_count = 0
        for index, _first in enumerate(words):
            maximum_word_count = min(3, len(words) - index)
            for word_count in range(maximum_word_count, 0, -1):
                selected = words[index : index + word_count]
                if any(
                    source.CONTEXTUAL_ALIAS_PHRASE_GAP_RE.fullmatch(
                        text[left.end() : right.start()]
                    )
                    is None
                    for left, right in zip(selected, selected[1:])
                ):
                    continue
                alias_start = selected[0].start()
                alias_end = selected[-1].end()
                compact = source.compact_identifier(text[alias_start:alias_end])
                if compact not in forms:
                    continue
                window_start = max(0, alias_start - 96)
                before = text[window_start:alias_start].casefold()
                after = text[alias_end : min(len(text), alias_end + 96)].casefold()
                if (
                    source.CONTEXTUAL_ALIAS_BEFORE_CUE_RE.search(before) is not None
                    or source.CONTEXTUAL_ALIAS_AFTER_CUE_RE.match(after) is not None
                ):
                    matched_count += 1
                    break
        return matched_count

    return count


def exact_final_corpus_identity_residue_scan(
    corpus_rows: Iterable[dict],
    seller_literals_by_uid: dict[str, list[str]],
    global_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_tokens: set[str] | frozenset[str] | None = None,
    contextual_alias_deletion_registry: set[str] | frozenset[str] | None = None,
    seller_phrase_tokens_by_uid: dict[str, set[str]] | None = None,
    audited_global_phrase_tokens: set[str] | frozenset[str] | None = None,
    seller_contextual_collision_tokens_by_uid: dict[str, set[str]] | None = None,
    *,
    fixed_content_collision_contract: dict | None = None,
    fail_on_residue: bool = True,
) -> dict:
    """Scan final text with compiled v3.1 matchers and v4 collision semantics.

    Raw item histories contain far more rows than the parent seller-level
    corpus.  The parent implementation recompiles every seller literal and
    reconstructs global alias lookup state for every row.  The full contextual
    counter still exactly replays that matcher.  V4 separately counts exact,
    plural, and identity-suffix matches as fail-closed residue, while generated
    one-character omissions are audit-only after redaction because deletion can
    create a cue adjacency that did not exist in the original field.
    """

    fixed_content_collisions = validated_fixed_final_audit_content_collisions(
        fixed_content_collision_contract
    )
    expected_fixed_collision_counts = {
        (
            entry["rule_name"],
            entry["seller_uid_sha256"],
            entry["clean_text_sha256"],
            entry["matched_surface_sha256"],
        ): int(entry["expected_match_count"])
        for entry in fixed_content_collisions
    }
    observed_fixed_collision_counts: Counter[tuple[str, str, str, str]] = (
        Counter()
    )
    literal_patterns = {
        seller_uid: tuple(
            source.identity_literal_pattern(literal) for literal in literals
        )
        for seller_uid, literals in seller_literals_by_uid.items()
    }
    local_phrase_counts = {
        seller_uid: _compile_unconditional_alias_count(tokens)
        for seller_uid, tokens in (seller_phrase_tokens_by_uid or {}).items()
    }
    global_phrase_count = _compile_unconditional_alias_count(
        audited_global_phrase_tokens or frozenset()
    )
    contextual_count = _compile_contextual_alias_count(
        contextual_alias_tokens or frozenset(),
        contextual_alias_deletion_registry,
    )
    exact_contextual_count = _compile_contextual_alias_count(
        contextual_alias_tokens or frozenset(), frozenset()
    )
    local_contextual_counts = {
        seller_uid: _compile_contextual_alias_count(tokens, frozenset())
        for seller_uid, tokens in (
            seller_contextual_collision_tokens_by_uid or {}
        ).items()
    }
    token_registry = global_tokens or frozenset()
    global_token_cache: dict[str, bool] = {}
    fixed_audit_cache: dict[str, tuple[dict[str, int], ...]] = {}
    global_phrase_cache: dict[str, int] = {}
    contextual_cache: dict[str, int] = {}
    exact_contextual_cache: dict[str, int] = {}

    pattern_counts: Counter[str] = Counter()
    local_literal_residue_count = 0
    global_token_residue_count = 0
    contextual_alias_residue_count = 0
    exact_contextual_alias_residue_count = 0
    seller_local_contextual_collision_residue_count = 0
    local_phrase_residue_count = 0
    audited_global_phrase_residue_count = 0
    scanned_rows = 0
    for row in corpus_rows:
        scanned_rows += 1
        seller_uid = str(row["seller_uid"])
        text = str(row["model_text"])
        fixed_counts = fixed_audit_cache.get(text)
        if fixed_counts is None:
            fixed_counts = tuple(
                dict(
                    Counter(
                        sha256_text(match.group(0))
                        for match in pattern.finditer(text)
                    )
                )
                for _rule_name, pattern in source.FINAL_CORPUS_AUDIT_RULES
            )
            fixed_audit_cache[text] = fixed_counts
        seller_uid_sha256 = sha256_text(seller_uid)
        clean_text_sha256 = sha256_text(text)
        for (rule_name, _pattern), surface_counts in zip(
            source.FINAL_CORPUS_AUDIT_RULES, fixed_counts, strict=True
        ):
            for surface_sha256, count in surface_counts.items():
                fixed_key = (
                    rule_name,
                    seller_uid_sha256,
                    clean_text_sha256,
                    surface_sha256,
                )
                if fixed_key in expected_fixed_collision_counts:
                    observed_fixed_collision_counts[fixed_key] += count
                else:
                    pattern_counts[rule_name] += count
        for pattern in literal_patterns.get(seller_uid, ()):
            local_literal_residue_count += sum(1 for _ in pattern.finditer(text))
        local_counter = local_phrase_counts.get(seller_uid)
        if local_counter is not None:
            local_phrase_residue_count += local_counter(text)
        global_phrase_matches = global_phrase_cache.get(text)
        if global_phrase_matches is None:
            global_phrase_matches = global_phrase_count(text)
            global_phrase_cache[text] = global_phrase_matches
        audited_global_phrase_residue_count += global_phrase_matches
        if token_registry:
            for match in source.IDENTIFIER_TOKEN_RE.finditer(text):
                surface = match.group(0)
                matched = global_token_cache.get(surface)
                if matched is None:
                    matched = source.matches_global_identity_token(
                        surface, token_registry
                    )
                    global_token_cache[surface] = matched
                global_token_residue_count += int(matched)
        contextual_matches = contextual_cache.get(text)
        if contextual_matches is None:
            contextual_matches = contextual_count(text)
            contextual_cache[text] = contextual_matches
        contextual_alias_residue_count += contextual_matches
        exact_contextual_matches = exact_contextual_cache.get(text)
        if exact_contextual_matches is None:
            exact_contextual_matches = exact_contextual_count(text)
            exact_contextual_cache[text] = exact_contextual_matches
        if exact_contextual_matches > contextual_matches:
            raise ValueError("Step7-v4 contextual alias matcher monotonicity drift")
        exact_contextual_alias_residue_count += exact_contextual_matches
        local_contextual_counter = local_contextual_counts.get(seller_uid)
        if local_contextual_counter is not None:
            seller_local_contextual_collision_residue_count += (
                local_contextual_counter(text)
            )

    observed_fixed_collision_count_map = {
        key: int(observed_fixed_collision_counts.get(key, 0))
        for key in expected_fixed_collision_counts
    }
    if observed_fixed_collision_count_map != expected_fixed_collision_counts:
        raise ValueError(
            "Step7-v4 fixed final-audit content-collision replay drift: "
            "expected="
            f"{canonical_hash(sorted((*key, value) for key, value in expected_fixed_collision_counts.items()))} "
            "observed="
            f"{canonical_hash(sorted((*key, value) for key, value in observed_fixed_collision_count_map.items()))}"
        )
    fixed_content_collision_count = int(
        sum(observed_fixed_collision_counts.values())
    )
    fixed_content_collision_counts_by_rule: Counter[str] = Counter()
    for (
        rule_name,
        _seller_uid_sha256,
        _clean_text_sha256,
        _surface_sha256,
    ), count in observed_fixed_collision_counts.items():
        fixed_content_collision_counts_by_rule[rule_name] += int(count)
    pattern_residue_count = int(sum(pattern_counts.values()))
    total = (
        pattern_residue_count
        + int(local_literal_residue_count)
        + int(global_token_residue_count)
        + int(exact_contextual_alias_residue_count)
        + int(seller_local_contextual_collision_residue_count)
        + int(local_phrase_residue_count)
        + int(audited_global_phrase_residue_count)
    )
    result = {
        "status": "pass" if total == 0 else "fail",
        "claim_scope": (
            "fixed_high_precision_exact_identity_residue_absence_with_"
            "post_redaction_one_character_omission_collisions_censused_"
            "separately"
        ),
        "unknown_identifier_absence_proven": False,
        "scan_scope": "serialized_final_model_text_each_seller_row_independently",
        "seller_row_count": scanned_rows,
        "independent_pattern_match_count_including_pinned_content_collisions": (
            pattern_residue_count + fixed_content_collision_count
        ),
        "pattern_residue_count": pattern_residue_count,
        "pattern_residue_count_by_rule": {
            key: int(value)
            for key, value in sorted(pattern_counts.items())
            if value
        },
        "pinned_non_identity_pattern_collision_count": (
            fixed_content_collision_count
        ),
        "pinned_non_identity_pattern_collision_count_by_rule": {
            key: int(value)
            for key, value in sorted(
                fixed_content_collision_counts_by_rule.items()
            )
            if value
        },
        "pinned_non_identity_pattern_collision_contract_sha256": (
            canonical_hash(fixed_content_collisions)
        ),
        "seller_local_identity_literal_residue_count": int(
            local_literal_residue_count
        ),
        "seller_local_separator_variant_residue_count": int(
            local_phrase_residue_count
        ),
        "audited_global_identity_phrase_residue_count": int(
            audited_global_phrase_residue_count
        ),
        "known_global_high_confidence_handle_residue_count": int(
            global_token_residue_count
        ),
        "context_gated_known_alias_residue_count": int(
            contextual_alias_residue_count
        ),
        "context_gated_exact_known_alias_residue_count": int(
            exact_contextual_alias_residue_count
        ),
        "post_redaction_one_character_omission_collision_census_count": int(
            contextual_alias_residue_count
            - exact_contextual_alias_residue_count
        ),
        "seller_local_context_gated_collision_residue_count": int(
            seller_local_contextual_collision_residue_count
        ),
        "total_residue_count": int(total),
    }
    if total and fail_on_residue:
        raise ValueError(
            "Step7-v4 final serialized corpus identity scan failed: "
            f"pattern={pattern_residue_count} "
            f"pinned_non_identity_pattern_collision="
            f"{fixed_content_collision_count} "
            f"local={local_literal_residue_count} global={global_token_residue_count} "
            f"local_phrase={local_phrase_residue_count} "
            f"audited_global_phrase={audited_global_phrase_residue_count} "
            f"exact_contextual_alias={exact_contextual_alias_residue_count} "
            "one_character_omission_collision_census="
            f"{contextual_alias_residue_count - exact_contextual_alias_residue_count} "
            "local_contextual_collision="
            f"{seller_local_contextual_collision_residue_count}"
        )
    return result


def seller_local_collision_residual_census(
    corpus_rows: Iterable[dict],
    seller_contextual_collision_tokens_by_uid: dict[str, set[str]],
) -> dict:
    """Inventory deliberately retained, uncued ambiguous local aliases."""

    token_counts: Counter[str] = Counter()
    weighted_token_counts: Counter[str] = Counter()
    affected_seller_rows = 0
    affected_sellers = set()
    for row in corpus_rows:
        seller_uid = str(row["seller_uid"])
        tokens = seller_contextual_collision_tokens_by_uid.get(seller_uid, set())
        if not tokens:
            continue
        affected_seller_rows += 1
        affected_sellers.add(seller_uid)
        text = str(row["model_text"])
        multiplicity = int(row.get("multiplicity", 1))
        if multiplicity <= 0:
            raise ValueError("Step7-v4 collision census multiplicity is invalid")
        for token in sorted(tokens):
            count = len(source.unconditional_alias_spans(text, {token}))
            if count:
                token_counts[token] += count
                weighted_token_counts[token] += count * multiplicity
    return {
        "status": "audit_uncued_ambiguous_local_alias_collisions_retained",
        "claim_scope": (
            "exact_plural_or_identity_suffix_forms_without_an_identity_cue_"
            "are_inventory_not_proof_of_identity"
        ),
        "affected_seller_count": len(affected_sellers),
        "affected_seller_unique_text_row_count": affected_seller_rows,
        "retained_unique_text_occurrence_count": int(sum(token_counts.values())),
        "retained_source_weighted_occurrence_count": int(
            sum(weighted_token_counts.values())
        ),
        "retained_token_sha256_counts": {
            sha256_text(token): int(count)
            for token, count in sorted(token_counts.items())
        },
        "retained_token_sha256_source_weighted_counts": {
            sha256_text(token): int(count)
            for token, count in sorted(weighted_token_counts.items())
        },
        "registry_seller_uid_to_tokens_canonical_sha256": canonical_hash(
            {
                seller_uid: sorted(tokens)
                for seller_uid, tokens in sorted(
                    seller_contextual_collision_tokens_by_uid.items()
                )
                if tokens
            }
        ),
        "context_cued_occurrences_are_checked_by_final_identity_scan": True,
        "unknown_or_ambiguous_identifier_absence_proven": False,
    }


def audited_identity_embedded_residual_census(
    corpus_rows: Iterable[dict], registry: set[str] | frozenset[str]
) -> dict:
    """Exact v3.1 embedded-alias census with trie scanning and word caching.

    The parent implementation checks every audited alias for every word.  Raw
    item histories contain millions of repeated product words, making that
    mathematically exact implementation unnecessarily quadratic.  This version
    preserves the same surface, anchor, and alias-surface-pair counts while a
    trie finds every strict substring and a cache reuses repeated word results.
    """

    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry or any(
        re.fullmatch(r"[a-z0-9]{1,96}", token) is None for token in token_registry
    ):
        raise ValueError("Step7-v4 embedded identity census registry is invalid")
    terminal = ""
    trie: dict = {}
    for alias in sorted(token_registry):
        node = trie
        for character in alias:
            node = node.setdefault(character, {})
        node.setdefault(terminal, []).append(alias)

    def embedded_aliases(surface: str) -> tuple[str, ...]:
        matches = set()
        for start, character in enumerate(surface):
            node = trie.get(character)
            if node is None:
                continue
            position = start + 1
            if terminal in node:
                matches.update(node[terminal])
            while position < len(surface):
                node = node.get(surface[position])
                if node is None:
                    break
                if terminal in node:
                    matches.update(node[terminal])
                position += 1
        return tuple(
            alias for alias in sorted(matches) if len(alias) < len(surface)
        )

    anchor_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    scanned_rows = 0
    scanned_word_count = 0
    surface_cache: dict[str, tuple[str, ...]] = {}
    for row in corpus_rows:
        scanned_rows += 1
        if not str(row.get("seller_uid", "")):
            raise ValueError("Step7-v4 embedded identity census row lacks seller_uid")
        model_text = str(row.get("model_text", ""))
        for word_match in re.finditer(r"(?i)[a-z0-9]+", model_text):
            scanned_word_count += 1
            surface = word_match.group(0).casefold()
            if source.anchored_alias_registry_token(surface, token_registry) is not None:
                continue
            matched_aliases = surface_cache.get(surface)
            if matched_aliases is None:
                matched_aliases = embedded_aliases(surface)
                surface_cache[surface] = matched_aliases
            for alias in matched_aliases:
                anchor_counts[alias] += 1
                surface_counts[surface] += 1
                pair_counts[(alias, surface)] += 1
    return {
        "status": "pass_fixed_snapshot_embedded_identity_census_completed",
        "scan_scope": (
            "every_ascii_alphanumeric_word_in_serialized_final_model_text_"
            "after_anchored_identity_forms_are_excluded"
        ),
        "matching_contract": (
            "strict_substring_of_audited_seller_or_market_identity_registry_"
            "without_reusing_redaction_matchers"
        ),
        "registry_token_count": len(token_registry),
        "scanned_seller_row_count": scanned_rows,
        "matched_registry_token_count": len(anchor_counts),
        "matched_occurrence_count": int(sum(anchor_counts.values())),
        "matched_alias_surface_pair_count": len(pair_counts),
        "matched_surface_count": len(surface_counts),
        "matched_anchor_sha256_counts": {
            sha256_text(anchor): int(count)
            for anchor, count in sorted(anchor_counts.items())
        },
        "matched_surface_sha256_counts": {
            sha256_text(surface): int(count)
            for surface, count in sorted(surface_counts.items())
        },
        "matched_alias_surface_pair_sha256_counts": {
            sha256_text(anchor + "\0" + surface): int(count)
            for (anchor, surface), count in sorted(pair_counts.items())
        },
        "unknown_or_ambiguous_identifier_absence_proven": False,
        "implementation_audit": {
            "algorithm": "exact_alias_trie_with_repeated_surface_cache",
            "scanned_word_count": scanned_word_count,
            "unique_nonanchored_surface_count": len(surface_cache),
        },
    }


def full_known_alias_residual_census(
    corpus_rows: Iterable[dict], registry: set[str] | frozenset[str]
) -> dict:
    """Exact parent known-alias census with precompiled surface lookup.

    Matching still occurs longest-first at each word boundary and advances past
    the longest match exactly as v3.1 does.  Exact aliases retain precedence
    over plurals, which retain precedence over suffix forms.  The optimization
    only replaces repeated suffix classification with a precomputed dictionary
    and caches repeated serialized segments.
    """

    token_registry = frozenset(str(token).casefold() for token in registry)
    if not token_registry or any(
        re.fullmatch(r"[a-z0-9]{1,96}", token) is None for token in token_registry
    ):
        raise ValueError("Step7-v4 full known-alias census registry is invalid")
    forms: dict[str, tuple[str, str]] = {
        token: (token, "exact") for token in token_registry
    }
    for token in sorted(token_registry):
        plural = token + "s"
        forms.setdefault(plural, (token, "known_alias_plural"))
    for suffix in source.IDENTITY_HANDLE_SUFFIXES:
        for token in sorted(token_registry):
            forms.setdefault(token + suffix, (token, f"known_alias_plus_{suffix}"))
    maximum_form_length = max(len(form) for form in forms)
    anchor_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    seller_sets: dict[str, set[str]] = defaultdict(set)
    match_kind_counts: Counter[str] = Counter()
    scanned_rows = 0
    scanned_segments = 0
    segment_cache: dict[str, tuple[tuple[str, str, str], ...]] = {}

    def scan_segment(segment: str) -> tuple[tuple[str, str, str], ...]:
        cached = segment_cache.get(segment)
        if cached is not None:
            return cached
        words = list(source.KNOWN_ALIAS_CENSUS_WORD_RE.finditer(segment))
        matches = []
        word_index = 0
        while word_index < len(words):
            compact = ""
            longest: tuple[int, str, str, str] | None = None
            for end_index in range(word_index, len(words)):
                if end_index > word_index:
                    gap = segment[
                        words[end_index - 1].end() : words[end_index].start()
                    ]
                    if source.KNOWN_ALIAS_CENSUS_GAP_RE.fullmatch(gap) is None:
                        break
                compact += words[end_index].group(0).casefold()
                if len(compact) > maximum_form_length:
                    break
                classified = forms.get(compact)
                if classified is not None:
                    longest = (
                        end_index,
                        classified[0],
                        classified[1],
                        compact,
                    )
            if longest is None:
                word_index += 1
                continue
            end_index, anchor, match_kind, compact_surface = longest
            matches.append((anchor, match_kind, compact_surface))
            word_index = end_index + 1
        result = tuple(matches)
        segment_cache[segment] = result
        return result

    for row in corpus_rows:
        scanned_rows += 1
        seller_uid = str(row.get("seller_uid", ""))
        if not seller_uid:
            raise ValueError("Step7-v4 full alias census row lacks seller_uid")
        for segment in re.split(r"\|\||\r?\n", str(row.get("model_text", ""))):
            scanned_segments += 1
            for anchor, match_kind, compact_surface in scan_segment(segment):
                anchor_counts[anchor] += 1
                surface_counts[compact_surface] += 1
                seller_sets[anchor].add(seller_uid)
                match_kind_counts[match_kind] += 1
    return {
        "status": "pass_full_fixed_snapshot_known_alias_census_completed",
        "scan_scope": (
            "serialized_final_model_text_each_seller_row_newline_field_and_"
            "double_pipe_value_scanned_independently"
        ),
        "matching_contract": (
            "independent_longest_separator_invariant_exact_then_registry_"
            "anchored_plural_or_identity_suffix"
        ),
        "registry_token_count": len(token_registry),
        "scanned_seller_row_count": scanned_rows,
        "scanned_segment_count": scanned_segments,
        "matched_registry_token_count": len(anchor_counts),
        "matched_occurrence_count": int(sum(anchor_counts.values())),
        "matched_surface_count": len(surface_counts),
        "match_kind_counts": {
            key: int(value) for key, value in sorted(match_kind_counts.items())
        },
        "matched_anchor_sha256_counts": {
            sha256_text(anchor): int(count)
            for anchor, count in sorted(anchor_counts.items())
        },
        "matched_surface_sha256_counts": {
            sha256_text(surface): int(count)
            for surface, count in sorted(surface_counts.items())
        },
        "matched_anchor_sha256_seller_counts": {
            sha256_text(anchor): len(seller_sets[anchor])
            for anchor in sorted(seller_sets)
        },
        "unknown_or_ambiguous_identifier_absence_proven": False,
        "implementation_audit": {
            "algorithm": "exact_precompiled_surface_lookup_with_segment_cache",
            "precompiled_surface_count": len(forms),
            "unique_serialized_segment_count": len(segment_cache),
        },
    }


def validate_pair_manifest(policy: dict, rows: list[dict]) -> None:
    expected_schema = [
        "pair_uid",
        "split_name",
        "component_id",
        "seller_uid_left",
        "seller_uid_right",
    ]
    if not rows or list(rows[0]) != expected_schema:
        raise ValueError("Step7-v4 pair manifest schema drift")
    if any(list(row) != expected_schema for row in rows):
        raise ValueError("Step7-v4 pair manifest row schema drift")
    if len({row["pair_uid"] for row in rows}) != len(rows):
        raise ValueError("Step7-v4 pair manifest contains duplicate pairs")
    counts = Counter(row["split_name"] for row in rows)
    expected = policy["supervision_boundary"]["expected_counts"]
    if counts != Counter({split: expected[split]["total"] for split in ("train", "valid", "test")}):
        raise ValueError("Step7-v4 pair split counts drift")
    seller_split = {}
    component_split = {}
    endpoint_pairs = set()
    for row in rows:
        split = row["split_name"]
        component = row["component_id"]
        if (
            not component
            or row["seller_uid_left"] == row["seller_uid_right"]
            or not row["seller_uid_left"]
            or not row["seller_uid_right"]
        ):
            raise ValueError("Step7-v4 pair manifest contains an invalid pair")
        endpoint_key = tuple(
            sorted((row["seller_uid_left"], row["seller_uid_right"]))
        )
        if endpoint_key in endpoint_pairs:
            raise ValueError("Step7-v4 pair manifest repeats an unordered seller pair")
        endpoint_pairs.add(endpoint_key)
        previous_component = component_split.setdefault(component, split)
        if previous_component != split:
            raise ValueError("Step7-v4 component crosses split boundaries")
        for endpoint in ("seller_uid_left", "seller_uid_right"):
            seller = row[endpoint]
            previous_seller = seller_split.setdefault(seller, split)
            if previous_seller != split:
                raise ValueError("Step7-v4 seller crosses split boundaries")
    expected_sellers = policy["supervision_boundary"]["expected_total_unique_sellers"]
    if len(seller_split) != expected_sellers:
        raise ValueError("Step7-v4 seller universe drift")
    observed_component_counts = Counter(component_split.values())
    observed_seller_counts = Counter(seller_split.values())
    if observed_component_counts != Counter(
        policy["supervision_boundary"]["expected_component_count_by_split"]
    ):
        raise ValueError("Step7-v4 component count by split drift")
    if observed_seller_counts != Counter(
        policy["supervision_boundary"]["expected_seller_count_by_split"]
    ):
        raise ValueError("Step7-v4 seller count by split drift")


def validate_public_artifact_schemas(policy: dict) -> dict:
    outputs = policy["outputs"]
    pair_rows = load_csv(resolve(outputs["pair_manifest"]))
    validate_pair_manifest(policy, pair_rows)
    pair_uids = [row["pair_uid"] for row in pair_rows]
    quarantine = policy["parent_fragment_quarantine"]
    if (
        quarantine["pair_uid_sha256"]
        in {sha256_text(row["pair_uid"]) for row in pair_rows}
        or set(quarantine["seller_uid_sha256"])
        & {
            sha256_text(row[endpoint])
            for row in pair_rows
            for endpoint in ("seller_uid_left", "seller_uid_right")
        }
    ):
        raise ValueError("Step7-v4 quarantined parent fragment leaked into pairs")

    raw_lineage = load_csv(resolve(outputs["raw_item_lineage"]))
    raw_lineage_schema = [
        "source_dataset",
        "source_row_number",
        "item_uid",
        "seller_uid",
        "split_name",
        "title_text_uid",
        "description_text_uid",
    ]
    if not raw_lineage or any(list(row) != raw_lineage_schema for row in raw_lineage):
        raise ValueError("Step7-v4 raw item lineage schema drift")
    raw_keys = [
        (row["source_dataset"], int(row["source_row_number"])) for row in raw_lineage
    ]
    if len(raw_keys) != len(set(raw_keys)) or raw_keys != sorted(raw_keys):
        raise ValueError("Step7-v4 raw item lineage contains duplicate source rows")
    raw_expected = policy["raw_item_boundary"]["expected_selected_item_count"]
    if len(raw_lineage) != raw_expected:
        raise ValueError("Step7-v4 raw item lineage count drift")
    forbidden_raw_keys = {
        (quarantine["raw_source_dataset"], int(row["source_row_number"]))
        for row in quarantine["raw_rows"]
    }
    if forbidden_raw_keys & set(raw_keys):
        raise ValueError("Step7-v4 quarantined raw fragment leaked into lineage")
    pair_seller_split = {
        row[endpoint]: row["split_name"]
        for row in pair_rows
        for endpoint in ("seller_uid_left", "seller_uid_right")
    }
    source_counts = Counter()
    raw_occurrences = set()
    item_uids = set()
    for row in raw_lineage:
        source_name = row["source_dataset"]
        row_number = int(row["source_row_number"])
        seller_uid = row["seller_uid"]
        if (
            source_name not in policy["raw_item_boundary"]["allowed_source_datasets"]
            or row_number < 2
            or seller_uid not in pair_seller_split
            or row["split_name"] != pair_seller_split[seller_uid]
            or not row["item_uid"]
            or row["item_uid"] in item_uids
        ):
            raise ValueError("Step7-v4 raw item lineage identity/split drift")
        item_uids.add(row["item_uid"])
        source_counts[source_name] += 1
        for field in FIELD_NAMES:
            text_uid = row[f"{field}_text_uid"]
            if text_uid:
                raw_occurrences.add(
                    (seller_uid, field, text_uid, source_name, row_number)
                )
    if source_counts != Counter(
        policy["raw_item_boundary"]["expected_selected_item_count_by_source"]
    ):
        raise ValueError("Step7-v4 raw item lineage source-count drift")

    unique_rows = load_jsonl(resolve(outputs["unique_text_corpus"]))
    unique_schema = ["text_uid", "text", "text_sha256"]
    if not unique_rows or any(list(row) != unique_schema for row in unique_rows):
        raise ValueError("Step7-v4 unique-text corpus schema drift")
    if len({row["text_uid"] for row in unique_rows}) != len(unique_rows):
        raise ValueError("Step7-v4 unique-text corpus contains duplicate IDs")
    for row in unique_rows:
        if (
            not row["text"]
            or row["text_uid"] != row["text_sha256"]
            or sha256_text(row["text"]) != row["text_sha256"]
        ):
            raise ValueError("Step7-v4 unique-text hash/content drift")

    seller_rows = load_jsonl(resolve(outputs["seller_text_index"]))
    seller_schema = [
        "seller_uid",
        "split_name",
        "field_name",
        "text_uid",
        "multiplicity",
        "source_lineage",
    ]
    if not seller_rows or any(list(row) != seller_schema for row in seller_rows):
        raise ValueError("Step7-v4 seller-text index schema drift")
    text_uids = {row["text_uid"] for row in unique_rows}
    for row in raw_lineage:
        for field in ("title_text_uid", "description_text_uid"):
            if row[field] and row[field] not in text_uids:
                raise ValueError("Step7-v4 raw lineage references a missing clean text")
    seen = set()
    source_row_count = 0
    mapped_occurrences = set()
    for row in seller_rows:
        key = (row["seller_uid"], row["field_name"], row["text_uid"])
        if key in seen or row["text_uid"] not in text_uids:
            raise ValueError("Step7-v4 seller-text mapping duplicate/missing text")
        seen.add(key)
        if (
            row["field_name"] not in FIELD_NAMES
            or int(row["multiplicity"]) <= 0
            or row["seller_uid"] not in pair_seller_split
            or row["split_name"] != pair_seller_split[row["seller_uid"]]
            or not isinstance(row["source_lineage"], list)
            or not row["source_lineage"]
        ):
            raise ValueError("Step7-v4 seller-text mapping value drift")
        lineage_count = 0
        previous_source = None
        for item in row["source_lineage"]:
            if list(item) != ["source_dataset", "source_row_numbers"]:
                raise ValueError("Step7-v4 seller-text source-lineage schema drift")
            source_name = item["source_dataset"]
            row_numbers = [int(value) for value in item["source_row_numbers"]]
            if (
                source_name not in policy["raw_item_boundary"]["allowed_source_datasets"]
                or (previous_source is not None and source_name <= previous_source)
                or row_numbers != sorted(set(row_numbers))
                or any(value < 2 for value in row_numbers)
            ):
                raise ValueError("Step7-v4 seller-text source-lineage value drift")
            previous_source = source_name
            lineage_count += len(row_numbers)
            for row_number in row_numbers:
                occurrence = (
                    row["seller_uid"],
                    row["field_name"],
                    row["text_uid"],
                    source_name,
                    row_number,
                )
                if occurrence in mapped_occurrences:
                    raise ValueError("Step7-v4 seller-text source occurrence repeats")
                mapped_occurrences.add(occurrence)
        if lineage_count != int(row["multiplicity"]):
            raise ValueError("Step7-v4 multiplicity does not match source lineage")
        source_row_count += lineage_count
    if mapped_occurrences != raw_occurrences:
        raise ValueError("Step7-v4 raw lineage and seller-text mappings are not bijective")
    if {row["text_uid"] for row in seller_rows} != text_uids:
        raise ValueError("Step7-v4 unique-text corpus contains an unused text")

    gpu_pair_rows = load_csv(resolve(outputs["gpu_pair_manifest"]))
    gpu_pair_schema = ["pair_uid", "seller_uid_left", "seller_uid_right"]
    if (
        not gpu_pair_rows
        or any(list(row) != gpu_pair_schema for row in gpu_pair_rows)
        or any(
            row["pair_uid"] != f"pair_{index:06d}"
            for index, row in enumerate(gpu_pair_rows, start=1)
        )
    ):
        raise ValueError("Step7-v4 opaque GPU pair manifest schema/order drift")
    gpu_seller_rows = load_jsonl(resolve(outputs["gpu_seller_text_index"]))
    gpu_seller_schema = ["seller_uid", "field_name", "text_uid", "multiplicity"]
    if not gpu_seller_rows or any(
        list(row) != gpu_seller_schema for row in gpu_seller_rows
    ):
        raise ValueError("Step7-v4 opaque GPU seller index schema drift")
    expected_gpu_pairs, expected_gpu_sellers = build_opaque_gpu_indices(
        pair_rows, seller_rows
    )
    if gpu_pair_rows != expected_gpu_pairs or gpu_seller_rows != expected_gpu_sellers:
        raise ValueError("Step7-v4 opaque GPU indices do not replay full public indices")

    seller_stylometry = load_csv(resolve(outputs["seller_stylometry"]))
    expected_seller_schema = [
        "seller_uid",
        "split_name",
        *[
            f"{field}_{statistic}"
            for field in FIELD_NAMES
            for statistic in STYLOMETRY_STATISTICS
        ],
    ]
    if not seller_stylometry or any(list(row) != expected_seller_schema for row in seller_stylometry):
        raise ValueError("Step7-v4 seller stylometry schema drift")
    if [row["seller_uid"] for row in seller_stylometry] != sorted(pair_seller_split):
        raise ValueError("Step7-v4 seller stylometry order/universe drift")
    for row in seller_stylometry:
        if row["split_name"] != pair_seller_split[row["seller_uid"]]:
            raise ValueError("Step7-v4 seller stylometry split drift")
        for name in expected_seller_schema[2:]:
            value = str(row[name]).strip()
            if value and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError("Step7-v4 seller stylometry numeric drift")

    pair_stylometry = load_csv(resolve(outputs["pair_stylometry"]))
    expected_pair_style_schema = ["pair_uid", *stylometry_feature_names()]
    if not pair_stylometry or any(list(row) != expected_pair_style_schema for row in pair_stylometry):
        raise ValueError("Step7-v4 pair stylometry schema drift")
    if [row["pair_uid"] for row in pair_stylometry] != pair_uids:
        raise ValueError("Step7-v4 pair stylometry row order drift")
    for row in pair_stylometry:
        for name in expected_pair_style_schema[1:]:
            value = str(row[name]).strip()
            if value and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError("Step7-v4 pair stylometry numeric drift")

    legacy_rows = load_csv(resolve(outputs["legacy_pair_features"]))
    expected_legacy_schema = [
        "pair_uid",
        *SHORTCUT_AUDIT_ONLY_FEATURE_NAMES,
        *LEGACY18_FEATURE_NAMES,
    ]
    if not legacy_rows or any(list(row) != expected_legacy_schema for row in legacy_rows):
        raise ValueError("Step7-v4 legacy feature schema drift")
    if [row["pair_uid"] for row in legacy_rows] != pair_uids:
        raise ValueError("Step7-v4 legacy feature row order drift")
    for row in legacy_rows:
        for name in expected_legacy_schema[1:]:
            if not math.isfinite(float(row[name])):
                raise ValueError("Step7-v4 legacy feature numeric drift")

    return {
        "pair_rows": pair_rows,
        "raw_item_lineage_rows": raw_lineage,
        "unique_text_rows": unique_rows,
        "seller_text_rows": seller_rows,
        "gpu_pair_rows": gpu_pair_rows,
        "gpu_seller_text_rows": gpu_seller_rows,
        "seller_stylometry_rows": seller_stylometry,
        "pair_stylometry_rows": pair_stylometry,
        "legacy_rows": legacy_rows,
        "source_lineage_field_occurrence_count": source_row_count,
    }


def validate_text_diagnostics(
    policy: dict, parent_policy: dict, diagnostics: dict
) -> None:
    expected_keys = {
        "raw_character_count",
        "clean_character_count_occurrence_weighted",
        "aggregate_character_retention",
        "field_counts",
        "redaction_counts",
        "removed_global_identifier_token_sha256_counts",
        "removed_audited_phrase_sha256_counts",
        "removed_one_character_omission_surface_sha256_counts",
        "protected_content_occurrence_matching",
        "protected_content_retention_aggregation",
        "protected_content_retention",
    }
    if set(diagnostics) != expected_keys:
        raise ValueError("Step7-v4 text-diagnostics schema drift")
    raw_characters = diagnostics.get("raw_character_count")
    clean_characters = diagnostics.get(
        "clean_character_count_occurrence_weighted"
    )
    retention = diagnostics.get("aggregate_character_retention")
    minimum_retention = float(
        policy["clean_text_contract"]["minimum_aggregate_character_retention"]
    )
    if (
        type(raw_characters) is not int
        or raw_characters <= 0
        or type(clean_characters) is not int
        or not 0 <= clean_characters <= raw_characters
        or not isinstance(retention, (int, float))
        or not math.isfinite(float(retention))
        or abs(float(retention) - clean_characters / raw_characters) > 1e-15
        or float(retention) < minimum_retention
    ):
        raise ValueError("Step7-v4 aggregate text-retention audit drift")

    expected_field_keys = {
        f"{field}_{suffix}"
        for field in FIELD_NAMES
        for suffix in (
            "source_occurrence_count",
            "raw_nonempty_count",
            "clean_nonempty_count",
            "empty_after_redaction_count",
        )
    }
    field_counts = diagnostics.get("field_counts", {})
    expected_items = int(policy["raw_item_boundary"]["expected_selected_item_count"])
    if set(field_counts) != expected_field_keys or any(
        type(value) is not int or value < 0 for value in field_counts.values()
    ):
        raise ValueError("Step7-v4 text field-count audit drift")
    for field in FIELD_NAMES:
        source_count = field_counts[f"{field}_source_occurrence_count"]
        raw_nonempty = field_counts[f"{field}_raw_nonempty_count"]
        clean_nonempty = field_counts[f"{field}_clean_nonempty_count"]
        empty_after = field_counts[f"{field}_empty_after_redaction_count"]
        if (
            source_count != expected_items
            or not 0 <= raw_nonempty <= source_count
            or not 0 <= clean_nonempty <= raw_nonempty
            or clean_nonempty + empty_after != raw_nonempty
        ):
            raise ValueError(f"Step7-v4 {field} nonempty-field audit drift")

    expected_redaction_keys = {
        "generic_identifier_match_count",
        "seller_local_alias_match_count",
        "seller_local_alias_phrase_match_count",
        "audited_global_identity_phrase_match_count",
        "global_identifier_token_match_count",
        "contextual_alias_match_count",
        "post_redaction_one_character_omission_collision_count",
        "raw_character_count",
        "clean_character_count",
        "empty_input",
        "empty_after_redaction",
        "utf8_mojibake_sequence_repair_count",
        "windows_1252_c1_repair_count",
        "maximum_redaction_pass_count",
    }
    redaction_counts = diagnostics.get("redaction_counts", {})
    if (
        set(redaction_counts) != expected_redaction_keys
        or any(
            type(value) is not int or value < 0
            for value in redaction_counts.values()
        )
        or not 1 <= int(redaction_counts["maximum_redaction_pass_count"]) <= 8
        or int(redaction_counts["raw_character_count"]) != raw_characters
        or int(redaction_counts["clean_character_count"]) != clean_characters
    ):
        raise ValueError("Step7-v4 redaction-counter audit drift")

    for role in (
        "removed_global_identifier_token_sha256_counts",
        "removed_audited_phrase_sha256_counts",
        "removed_one_character_omission_surface_sha256_counts",
    ):
        counts = diagnostics.get(role, {})
        if not isinstance(counts, dict) or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(key)) is None
            or type(value) is not int
            or value <= 0
            for key, value in counts.items()
        ):
            raise ValueError(f"Step7-v4 {role} audit drift")

    fuzzy_collision = policy["clean_text_contract"][
        "one_character_omission_content_collision_handling"
    ]
    if diagnostics["removed_one_character_omission_surface_sha256_counts"] != (
        fuzzy_collision["expected_retained_surface_sha256_counts"]
    ):
        raise ValueError(
            "Step7-v4 reviewed one-character-omission match census drift"
        )

    clean_contract = policy["clean_text_contract"]
    if (
        diagnostics.get("protected_content_occurrence_matching")
        != clean_contract["protected_content_occurrence_matching"]
        or diagnostics.get("protected_content_retention_aggregation")
        != clean_contract["protected_content_retention_aggregation"]
    ):
        raise ValueError("Step7-v4 protected-content counting contract drift")
    quality = parent_policy["clean_text_contract"]["quality_gates"]
    protected_terms = list(
        dict.fromkeys(
            [
                *quality["protected_content_words"],
                *quality["protected_identity_collision_terms"],
            ]
        )
    )
    protected = diagnostics.get("protected_content_retention", {})
    expected_term_keys = {
        "raw_count",
        "intentional_identity_span_count",
        "eligible_content_count",
        "clean_count",
        "total_removed_count",
        "unexplained_removed_count",
        "created_surplus_count",
        "retention",
    }
    minimum_protected = min(
        float(quality["minimum_protected_word_retention"]),
        float(quality["minimum_protected_identity_collision_term_retention"]),
    )
    if set(protected) != set(protected_terms):
        raise ValueError("Step7-v4 protected-content term universe drift")
    for term in protected_terms:
        record = protected[term]
        integer_keys = expected_term_keys - {"retention"}
        if set(record) != expected_term_keys or any(
            type(record.get(key)) is not int or int(record[key]) < 0
            for key in integer_keys
        ):
            raise ValueError(f"Step7-v4 protected-content schema drift: {term}")
        raw_count = int(record["raw_count"])
        intentional = int(record["intentional_identity_span_count"])
        eligible = int(record["eligible_content_count"])
        clean_count = int(record["clean_count"])
        removed = int(record["total_removed_count"])
        unexplained = int(record["unexplained_removed_count"])
        created = int(record["created_surplus_count"])
        ratio = float(record["retention"])
        expected_ratio = 1.0 if eligible == 0 else 1.0 - unexplained / eligible
        if (
            intentional > raw_count
            or eligible != raw_count - intentional
            or clean_count > eligible
            or removed != raw_count - clean_count
            or unexplained != eligible - clean_count
            or created != 0
            or not math.isfinite(ratio)
            or abs(ratio - expected_ratio) > 1e-15
            or (eligible > 0 and ratio < minimum_protected)
        ):
            raise ValueError(
                f"Step7-v4 protected-content arithmetic drift: {term}"
            )


def validate_preparation_artifacts(policy: dict) -> tuple[dict, dict]:
    manifest_path = resolve(policy["outputs"]["preparation_manifest"])
    manifest = load_json(manifest_path)
    verify_canonical_self_hash(
        manifest, "manifest_content_sha256", "public preparation manifest"
    )
    if (
        manifest.get("step") != "step7_v4_prepare_source_data_public"
        or manifest.get("version") != EXPECTED_VERSION
        or manifest.get("labels_read") is not False
        or manifest.get("evidence_types_read") is not False
        or manifest.get("pair_label_or_evidence_bearing_input_files_opened")
        is not False
        or manifest.get("labelled_component_assignment_file_opened") is not False
        or manifest.get("frozen_label_free_parent_pair_projection_used")
        is not True
        or manifest.get("historical_test_labels_read") is not False
        or manifest.get("policy_sha256") != sha256_file(DEFAULT_POLICY)
    ):
        raise ValueError("Step7-v4 preparation manifest boundary drift")
    if set(manifest.get("outputs", {})) != set(PUBLIC_OUTPUT_ROLES):
        raise ValueError("Step7-v4 preparation output manifest universe drift")
    public_input_roles = {
        "parent_source_policy",
        "item_manifest",
        "market_item_snapshot",
        "agora_snapshot",
        "parent_pair_manifest",
        "parent_safe_features",
    }
    if set(manifest.get("input_files", {})) != public_input_roles:
        raise ValueError("Step7-v4 preparation input manifest universe drift")
    for role in public_input_roles:
        record = manifest["input_files"][role]
        spec = policy["inputs"][role]
        if (
            record.get("path") != str(spec["path"]).replace("\\", "/")
            or record.get("sha256") != spec["sha256"]
            or int(record.get("size_bytes", 0)) <= 0
        ):
            raise ValueError(f"Step7-v4 preparation input provenance drift: {role}")
    if manifest.get("parent_full_report_regeneration_skipped") is not True:
        raise ValueError("Step7-v4 parent replay scope was not declared")
    quarantine_cfg = policy["parent_fragment_quarantine"]
    quarantine = manifest.get("parent_fragment_quarantine", {})
    expected_quarantine_keys = {
        "status",
        "decision_basis",
        "reason",
        "parent_pair_count",
        "effective_pair_count",
        "excluded_pair_count",
        "excluded_pair_uid_sha256",
        "excluded_seller_uid_sha256",
        "excluded_split_name",
        "excluded_component_id",
        "isolated_component_verified",
        "filtered_identity_registry_replays_frozen_legacy18",
        "labels_or_evidence_types_read",
        "step2_fragment_rows_verified",
        "raw_workbook_fragment_rows_verified",
        "raw_fragment_contract_sha256",
    }
    if (
        set(quarantine) != expected_quarantine_keys
        or quarantine.get("status")
        != "pass_exact_label_free_raw_fragment_component_quarantined"
        or quarantine.get("decision_basis") != quarantine_cfg["decision_basis"]
        or quarantine.get("reason") != quarantine_cfg["reason"]
        or int(quarantine.get("parent_pair_count", -1))
        != int(quarantine_cfg["expected_parent_pair_count"])
        or int(quarantine.get("effective_pair_count", -1))
        != int(policy["supervision_boundary"]["expected_counts"]["total"])
        or int(quarantine.get("excluded_pair_count", -1))
        != int(quarantine_cfg["excluded_pair_count"])
        or quarantine.get("excluded_pair_uid_sha256")
        != quarantine_cfg["pair_uid_sha256"]
        or quarantine.get("excluded_seller_uid_sha256")
        != quarantine_cfg["seller_uid_sha256"]
        or quarantine.get("excluded_split_name") != quarantine_cfg["split_name"]
        or quarantine.get("excluded_component_id")
        != quarantine_cfg["component_id"]
        or quarantine.get("isolated_component_verified") is not True
        or quarantine.get("filtered_identity_registry_replays_frozen_legacy18")
        is not True
        or quarantine.get("labels_or_evidence_types_read") is not False
        or int(quarantine.get("step2_fragment_rows_verified", -1))
        != len(quarantine_cfg["raw_rows"])
        or int(quarantine.get("raw_workbook_fragment_rows_verified", -1))
        != len(quarantine_cfg["raw_rows"])
        or quarantine.get("raw_fragment_contract_sha256")
        != canonical_hash(quarantine_cfg["raw_rows"])
    ):
        raise ValueError("Step7-v4 parent-fragment quarantine audit drift")
    item_audit = manifest.get("item_manifest_audit", {})
    expected_item_audit_keys = {
        "selected_item_count",
        "selected_item_count_by_source",
        "quarantined_fragment_item_count",
        "quarantined_fragment_source_row_contract_sha256",
        "ignored_rows_for_pair_universe_sellers",
        "ignored_rows_by_boundary",
    }
    raw_cfg = policy["raw_item_boundary"]
    if (
        set(item_audit) != expected_item_audit_keys
        or int(item_audit.get("selected_item_count", -1))
        != int(raw_cfg["expected_selected_item_count"])
        or item_audit.get("selected_item_count_by_source")
        != raw_cfg["expected_selected_item_count_by_source"]
        or int(item_audit.get("quarantined_fragment_item_count", -1))
        != len(quarantine_cfg["raw_rows"])
        or item_audit.get("quarantined_fragment_source_row_contract_sha256")
        != canonical_hash(quarantine_cfg["raw_rows"])
        or int(item_audit.get("ignored_rows_for_pair_universe_sellers", -1))
        != sum(
            int(value)
            for value in item_audit.get("ignored_rows_by_boundary", {}).values()
        )
    ):
        raise ValueError("Step7-v4 Step2 item-manifest audit drift")
    workbook_audit = manifest.get("workbook_audit", {})
    expected_workbook_keys = {
        "worksheet_title",
        "worksheet_max_row",
        "worksheet_max_column",
        "selected_row_count",
        "quarantined_fragment_row_count",
        "conceptual_column_order",
        "exact_header_verified",
        "step2_seller_and_item_identity_replay_count",
    }
    quarantine_source_counts = Counter(
        quarantine_cfg["raw_source_dataset"]
        for _row in quarantine_cfg["raw_rows"]
    )
    if set(workbook_audit) != set(raw_cfg["allowed_source_datasets"]):
        raise ValueError("Step7-v4 workbook audit source universe drift")
    for source_name in raw_cfg["allowed_source_datasets"]:
        record = workbook_audit[source_name]
        columns = (
            raw_cfg["market_item_column_order"]
            if source_name == "market_item.xlsx"
            else raw_cfg["agora_column_order"]
        )
        header = (
            raw_cfg["market_item_exact_header"]
            if source_name == "market_item.xlsx"
            else raw_cfg["agora_exact_header"]
        )
        selected_count = int(
            raw_cfg["expected_selected_item_count_by_source"][source_name]
        )
        quarantined_count = int(quarantine_source_counts[source_name])
        if (
            set(record) != expected_workbook_keys
            or not str(record.get("worksheet_title", ""))
            or int(record.get("worksheet_max_row", 0)) < 2
            or int(record.get("worksheet_max_column", 0)) != len(columns)
            or int(record.get("selected_row_count", -1)) != selected_count
            or int(record.get("quarantined_fragment_row_count", -1))
            != quarantined_count
            or record.get("conceptual_column_order") != columns
            or record.get("exact_header_verified") != header
            or int(
                record.get("step2_seller_and_item_identity_replay_count", -1)
            )
            != selected_count + quarantined_count
        ):
            raise ValueError(f"Step7-v4 workbook audit drift: {source_name}")
    parent_policy = load_json(resolve(policy["inputs"]["parent_source_policy"]["path"]))
    validate_text_diagnostics(
        policy, parent_policy, manifest.get("text_diagnostics", {})
    )
    parent_replayed_roles = {
        "seller_profiles",
        "item_identity_signals",
    }
    parent_records = manifest.get("parent_replayed_input_files", {})
    if set(parent_records) != parent_replayed_roles:
        raise ValueError("Step7-v4 parent replay input universe drift")
    for role in parent_replayed_roles:
        record = parent_records[role]
        spec = parent_policy["inputs"][role]
        if (
            record.get("path") != str(spec["path"]).replace("\\", "/")
            or record.get("sha256") != spec["sha256"]
            or int(record.get("size_bytes", 0)) <= 0
        ):
            raise ValueError(f"Step7-v4 parent replay provenance drift: {role}")
    expected_implementation = {
        "producer": file_record(resolve(policy["implementation"]["preparation"]["path"])),
        "common": file_record(resolve(policy["implementation"]["common"]["path"])),
        "redaction_module": file_record(
            resolve(policy["implementation"]["redaction_module"]["path"])
        ),
        "parent_preparation": file_record(
            resolve(policy["implementation"]["parent_preparation"]["path"])
        ),
    }
    if manifest.get("implementation") != expected_implementation:
        raise ValueError("Step7-v4 preparation implementation provenance drift")
    for role in PUBLIC_OUTPUT_ROLES:
        path = verify_file_record(manifest["outputs"][role], role)
        if path != resolve(policy["outputs"][role]):
            raise ValueError(f"Step7-v4 preparation output path drift: {role}")
    bundle = validate_public_artifact_schemas(policy)
    counts = manifest["counts"]
    if int(counts["pair_count"]) != len(bundle["pair_rows"]):
        raise ValueError("Step7-v4 preparation pair count drift")
    if int(counts["seller_count"]) != len(
        {
            row[endpoint]
            for row in bundle["pair_rows"]
            for endpoint in ("seller_uid_left", "seller_uid_right")
        }
    ):
        raise ValueError("Step7-v4 preparation seller count drift")
    if int(counts["raw_item_lineage_count"]) != len(
        bundle["raw_item_lineage_rows"]
    ):
        raise ValueError("Step7-v4 preparation raw lineage count drift")
    if int(counts["global_unique_clean_text_count"]) != len(
        bundle["unique_text_rows"]
    ):
        raise ValueError("Step7-v4 preparation unique text count drift")
    if int(counts["seller_unique_text_mapping_count"]) != len(
        bundle["seller_text_rows"]
    ):
        raise ValueError("Step7-v4 preparation seller mapping count drift")
    if int(counts["gpu_opaque_pair_count"]) != len(
        bundle["gpu_pair_rows"]
    ) or int(counts["gpu_opaque_seller_text_mapping_count"]) != len(
        bundle["gpu_seller_text_rows"]
    ):
        raise ValueError("Step7-v4 preparation opaque GPU index count drift")
    for field in FIELD_NAMES:
        if int(counts[f"{field}_seller_unique_text_mapping_count"]) != sum(
            row["field_name"] == field for row in bundle["seller_text_rows"]
        ):
            raise ValueError(f"Step7-v4 preparation {field} mapping count drift")
    identity = manifest.get("identity_audit", {})
    high_precision = identity.get("high_precision_final_serialized_scan", {})
    if (
        high_precision.get("status") != "pass"
        or int(high_precision.get("total_residue_count", -1)) != 0
        or identity.get("unknown_or_ambiguous_identifier_absence_proven")
        is not False
    ):
        raise ValueError("Step7-v4 preparation identity audit did not pass")
    retained_collision = identity.get(
        "retained_uncued_local_collision_census", {}
    )
    expected_retained_collision_keys = {
        "status",
        "claim_scope",
        "affected_seller_count",
        "affected_seller_unique_text_row_count",
        "retained_unique_text_occurrence_count",
        "retained_source_weighted_occurrence_count",
        "retained_token_sha256_counts",
        "retained_token_sha256_source_weighted_counts",
        "registry_seller_uid_to_tokens_canonical_sha256",
        "context_cued_occurrences_are_checked_by_final_identity_scan",
        "unknown_or_ambiguous_identifier_absence_proven",
    }
    retained_counts = retained_collision.get("retained_token_sha256_counts", {})
    retained_weighted_counts = retained_collision.get(
        "retained_token_sha256_source_weighted_counts", {}
    )
    collision_policy = policy["clean_text_contract"][
        "seller_local_content_collision_handling"
    ]
    if (
        set(retained_collision) != expected_retained_collision_keys
        or retained_collision.get("status")
        != "audit_uncued_ambiguous_local_alias_collisions_retained"
        or int(retained_collision.get("affected_seller_count", -1))
        != int(collision_policy["expected_affected_seller_count"])
        or int(
            retained_collision.get("affected_seller_unique_text_row_count", -1)
        )
        <= 0
        or int(retained_collision.get("retained_unique_text_occurrence_count", -1))
        != sum(int(value) for value in retained_counts.values())
        or int(
            retained_collision.get("retained_source_weighted_occurrence_count", -1)
        )
        != sum(int(value) for value in retained_weighted_counts.values())
        or set(retained_counts) != set(retained_weighted_counts)
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(key)) is None
            or type(value) is not int
            or value <= 0
            for counts in (retained_counts, retained_weighted_counts)
            for key, value in counts.items()
        )
        or retained_collision.get(
            "registry_seller_uid_to_tokens_canonical_sha256"
        )
        != collision_policy[
            "expected_affected_seller_uid_to_tokens_canonical_sha256"
        ]
        or retained_collision.get(
            "context_cued_occurrences_are_checked_by_final_identity_scan"
        )
        is not True
        or retained_collision.get(
            "unknown_or_ambiguous_identifier_absence_proven"
        )
        is not False
    ):
        raise ValueError("Step7-v4 retained local-collision census drift")
    registry = manifest.get("identity_registry_provenance", {})
    expected_registry_keys = {
        "global_identity_token_count",
        "global_identity_token_registry_sha256",
        "contextual_alias_token_count",
        "contextual_alias_registry_sha256",
        "contextual_alias_deletion_token_count",
        "contextual_alias_deletion_registry_sha256",
        "audited_global_phrase_token_count",
        "audited_global_phrase_registry_sha256",
        "seller_local_literal_registry_sha256",
        "seller_local_phrase_registry_sha256",
        "seller_local_content_collision_registry",
        "quarantined_invalid_profile_count",
        "quarantined_invalid_profile_uid_sha256_list",
        "labels_or_evidence_types_read",
    }
    if (
        set(registry) != expected_registry_keys
        or registry.get("labels_or_evidence_types_read") is not False
        or any(
            int(registry[key]) <= 0
            for key in (
                "global_identity_token_count",
                "contextual_alias_token_count",
                "contextual_alias_deletion_token_count",
                "audited_global_phrase_token_count",
            )
        )
        or int(registry.get("quarantined_invalid_profile_count", -1))
        != len(quarantine_cfg["seller_uid_sha256"])
        or registry.get("quarantined_invalid_profile_uid_sha256_list")
        != quarantine_cfg["seller_uid_sha256"]
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(value)) is None
            for key, value in registry.items()
            if key.endswith("_sha256")
        )
    ):
        raise ValueError("Step7-v4 identity registry provenance drift")
    local_collision_registry = registry["seller_local_content_collision_registry"]
    expected_local_collision_registry_keys = {
        "status",
        "collision_compact_count",
        "collision_compact_registry_canonical_sha256",
        "affected_seller_count",
        "affected_seller_uid_to_tokens_canonical_sha256",
        "affected_seller_uid_sha256_list",
        "removed_from_unconditional_literal_registry_count",
        "removed_from_unconditional_phrase_registry_count",
        "removed_from_global_contextual_registry_count",
        "removed_from_global_contextual_registry_canonical_sha256",
        "one_character_omission_matching_allowed",
        "uncued_ambiguous_occurrences_are_retained_and_audited",
        "labels_or_evidence_types_read",
    }
    if (
        set(local_collision_registry) != expected_local_collision_registry_keys
        or local_collision_registry.get("status")
        != "pass_collision_prone_local_aliases_are_context_gated"
        or int(local_collision_registry.get("collision_compact_count", -1))
        != int(collision_policy["collision_compact_count"])
        or local_collision_registry.get(
            "collision_compact_registry_canonical_sha256"
        )
        != collision_policy["collision_compact_registry_canonical_sha256"]
        or int(local_collision_registry.get("affected_seller_count", -1))
        != int(collision_policy["expected_affected_seller_count"])
        or local_collision_registry.get(
            "affected_seller_uid_to_tokens_canonical_sha256"
        )
        != collision_policy[
            "expected_affected_seller_uid_to_tokens_canonical_sha256"
        ]
        or len(local_collision_registry.get("affected_seller_uid_sha256_list", []))
        != int(collision_policy["expected_affected_seller_count"])
        or any(
            IMPLEMENTATION_HASH_RE.fullmatch(str(value)) is None
            for value in local_collision_registry.get(
                "affected_seller_uid_sha256_list", []
            )
        )
        or int(
            local_collision_registry.get(
                "removed_from_unconditional_literal_registry_count", 0
            )
        )
        <= 0
        or int(
            local_collision_registry.get(
                "removed_from_unconditional_phrase_registry_count", 0
            )
        )
        <= 0
        or int(
            local_collision_registry.get(
                "removed_from_global_contextual_registry_count", -1
            )
        )
        != int(
            collision_policy[
                "expected_removed_from_global_contextual_registry_count"
            ]
        )
        or local_collision_registry.get(
            "removed_from_global_contextual_registry_canonical_sha256"
        )
        != collision_policy[
            "expected_removed_from_global_contextual_registry_canonical_sha256"
        ]
        or local_collision_registry.get(
            "one_character_omission_matching_allowed"
        )
        is not False
        or local_collision_registry.get(
            "uncued_ambiguous_occurrences_are_retained_and_audited"
        )
        is not True
        or local_collision_registry.get("labels_or_evidence_types_read")
        is not False
    ):
        raise ValueError("Step7-v4 local content-collision registry audit drift")
    return manifest, bundle


def project_private_label_rows(
    rows: list[dict],
    split: str,
    *,
    expected_pair_uids: list[str] | None = None,
    allowed_excluded_pair_uid_sha256: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Project one frozen parent split into labels and separate evidence."""

    expected_schema = [
        "pair_uid",
        "review_label",
        "evidence_type",
        "component_id",
        "identity_rule_control_score",
    ]
    if not rows or any(list(row) != expected_schema for row in rows):
        raise ValueError(f"Step7-v4 parent private-label schema drift: {split}")
    if len({row["pair_uid"] for row in rows}) != len(rows):
        raise ValueError(f"Step7-v4 parent private labels repeat a pair: {split}")
    projected_rows = rows
    if expected_pair_uids is not None:
        expected_set = set(expected_pair_uids)
        if len(expected_set) != len(expected_pair_uids):
            raise ValueError(f"Step7-v4 effective pair IDs repeat: {split}")
        projected_rows = [row for row in rows if row["pair_uid"] in expected_set]
        excluded_hashes = {
            sha256_text(row["pair_uid"])
            for row in rows
            if row["pair_uid"] not in expected_set
        }
        if (
            [row["pair_uid"] for row in projected_rows] != expected_pair_uids
            or excluded_hashes != (allowed_excluded_pair_uid_sha256 or set())
        ):
            raise ValueError(
                f"Step7-v4 private-label quarantine projection drift: {split}"
            )
    labels = [
        {
            "pair_uid": row["pair_uid"],
            "review_label": row["review_label"],
            "component_id": row["component_id"],
        }
        for row in projected_rows
    ]
    evidence = [
        {"pair_uid": row["pair_uid"], "evidence_type": row["evidence_type"]}
        for row in projected_rows
    ]
    return labels, evidence


def validate_private_label_artifacts(policy: dict, pair_rows: list[dict]) -> dict:
    manifest_path = resolve(policy["outputs"]["development_labels_manifest"])
    manifest = load_json(manifest_path)
    verify_canonical_self_hash(
        manifest, "manifest_content_sha256", "private development manifest"
    )
    expected_label_inputs = verify_inputs(
        policy, ("parent_train_labels", "parent_valid_labels")
    )
    if (
        manifest.get("step") != "step7_v4_prepare_source_data_private_labels"
        or manifest.get("version") != EXPECTED_VERSION
        or manifest.get("historical_test_labels_materialized") is not False
        or set(manifest.get("outputs", {})) != set(PRIVATE_OUTPUT_ROLES)
        or manifest.get("public_preparation_manifest_sha256")
        != sha256_file(resolve(policy["outputs"]["preparation_manifest"]))
        or manifest.get("policy_sha256") != sha256_file(DEFAULT_POLICY)
        or manifest.get("producer_sha256")
        != sha256_file(resolve(policy["implementation"]["preparation"]["path"]))
        or manifest.get("label_inputs") != expected_label_inputs
        or manifest.get("selection_label_columns")
        != ["pair_uid", "review_label", "component_id"]
        or manifest.get("diagnostic_evidence_columns")
        != ["pair_uid", "evidence_type"]
        or manifest.get("identity_rule_control_score_materialized") is not False
        or manifest.get("evidence_is_physically_separate_from_selection_labels")
        is not True
        or manifest.get("parent_fragment_quarantine_projection")
        != {
            "decision_basis": policy["parent_fragment_quarantine"][
                "decision_basis"
            ],
            "excluded_pair_uid_sha256": policy["parent_fragment_quarantine"][
                "pair_uid_sha256"
            ],
            "excluded_pair_count_by_split": {"train": 0, "valid": 1},
            "labels_or_evidence_types_used_to_choose_exclusion": False,
        }
    ):
        raise ValueError("Step7-v4 private-label manifest step drift")
    pair_by_split = defaultdict(list)
    pair_index = {}
    for row in pair_rows:
        pair_by_split[row["split_name"]].append(row["pair_uid"])
        pair_index[row["pair_uid"]] = row
    output = {}
    for split in ("train", "valid"):
        label_role = f"{split}_labels"
        evidence_role = f"{split}_evidence"
        label_path = verify_file_record(
            manifest["outputs"][label_role], label_role
        )
        evidence_path = verify_file_record(
            manifest["outputs"][evidence_role], evidence_role
        )
        if label_path != resolve(policy["outputs"][label_role]) or evidence_path != resolve(
            policy["outputs"][evidence_role]
        ):
            raise ValueError(f"Step7-v4 private label/evidence path drift: {split}")
        labels = load_csv(label_path)
        evidence = load_csv(evidence_path)
        parent_role = f"parent_{split}_labels"
        parent_rows = load_csv(resolve(policy["inputs"][parent_role]["path"]))
        allowed_excluded_hashes = (
            {policy["parent_fragment_quarantine"]["pair_uid_sha256"]}
            if split == policy["parent_fragment_quarantine"]["split_name"]
            else set()
        )
        expected_labels, expected_evidence = project_private_label_rows(
            parent_rows,
            split,
            expected_pair_uids=pair_by_split[split],
            allowed_excluded_pair_uid_sha256=allowed_excluded_hashes,
        )
        if labels != expected_labels or evidence != expected_evidence:
            raise ValueError(
                f"Step7-v4 private label/evidence parent replay drift: {split}"
            )
        label_schema = ["pair_uid", "review_label", "component_id"]
        evidence_schema = ["pair_uid", "evidence_type"]
        if not labels or any(list(row) != label_schema for row in labels):
            raise ValueError(f"Step7-v4 private selection-label schema drift: {split}")
        if not evidence or any(list(row) != evidence_schema for row in evidence):
            raise ValueError(f"Step7-v4 private diagnostic-evidence schema drift: {split}")
        expected_pair_uids = pair_by_split[split]
        if [row["pair_uid"] for row in labels] != expected_pair_uids or [
            row["pair_uid"] for row in evidence
        ] != expected_pair_uids:
            raise ValueError(f"Step7-v4 private label/evidence pair order drift: {split}")
        if any(
            row["review_label"] not in {"positive", "negative"}
            or row["component_id"]
            != pair_index[row["pair_uid"]]["component_id"]
            for row in labels
        ):
            raise ValueError(
                f"Step7-v4 private label value/component drift: {split}"
            )
        label_counts = Counter(row["review_label"] for row in labels)
        expected = policy["supervision_boundary"]["expected_counts"][split]
        if label_counts != Counter(
            {"positive": expected["positive"], "negative": expected["negative"]}
        ):
            raise ValueError(f"Step7-v4 private label count drift: {split}")
        if any(not str(row["evidence_type"]).strip() for row in evidence):
            raise ValueError(f"Step7-v4 private evidence value is empty: {split}")
        output[split] = {"labels": labels, "evidence": evidence}
    return output
