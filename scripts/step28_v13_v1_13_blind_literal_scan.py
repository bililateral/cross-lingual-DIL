#!/usr/bin/env python3
"""Relation-free exact-literal scan for blinded Step28-v13 audit splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


VERSION = "2026-08-12-step28-v13-v1-13-blind-literal-scan-v4"
SEALED_REGISTRY_VERSION = (
    "2026-08-12-step28-v13-v1-13-sealed-literal-registry-builder-v2"
)
AUDIT_SPLITS = frozenset({"audit_a", "audit_b"})
SPLITS = ("train", "development", "audit_a", "audit_b")
PROFILE_TEXT_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
PRIVATE_LITERAL_FILES = (
    "private/controller_membership.jsonl",
    "private/qrels.jsonl",
    "private/world_generation_audit.jsonl",
    "private/document_collision_attempts.jsonl",
    "private/identity_allocation_receipts.jsonl",
)
SEALED_REGISTRY_CATEGORIES = (
    "controller_membership_string",
    "document_collision_receipt_string",
    "full_world_forbidden",
    "identity_allocation_receipt_string",
    "pair_label_string",
    "persisted_world_audit_string",
    "qrels_string",
)
SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES = (
    "private/controller_membership.jsonl",
    "private/pair_labels.csv",
    "private/qrels.jsonl",
    "private/world_generation_audit.jsonl",
    "private/document_collision_attempts.jsonl",
    "private/identity_allocation_receipts.jsonl",
)
SEALED_REGISTRY_FALSE_AUTHORIZATION_FIELDS = (
    "formal_generation_authorized",
    "model_training_authorized",
    "formal_seed_authorized",
    "audit_truth_release_authorized",
    "quality_audit_run_authorized",
    "formal_500x4_generation_authorized",
    "design_dataset_training_qualified",
)
SEALED_REGISTRY_PUBLIC_RECEIPT_KEYS = frozenset(
    {
        "version",
        "status",
        "transaction_id",
        "dataset_root_manifest",
        "literal_authority_source",
        "builder_policy",
        "worlds_replayed",
        "audit_worlds_projected",
        "private_input_files_semantically_replayed",
        "private_input_file_count",
        "split_commitments",
        "private_values_returned",
        "private_relations_returned",
        "labels_returned",
        "labels_opened_for_exact_replay",
        "labels_used_for_candidate_selection",
        "labels_used_for_literal_selection",
        "pair_label_rows_replayed",
        "qrels_returned",
        "observed_rows_modified",
        "candidate_selection_changed",
        "derangement_changed",
        "quality_probe_run",
        *SEALED_REGISTRY_FALSE_AUTHORIZATION_FIELDS,
        "source_closure",
        "sealed_registry",
        "builder_source",
        "canonical_self_hash",
    }
)
OBSERVED_TEXT_FILES = (
    "observed/model_seller_profiles.jsonl",
    "observed/redacted_items.jsonl",
)
EXPLICIT_LITERAL_FIELDS = frozenset(
    {
        "downstream_canonical_value",
        "identity_value",
        "raw_surface",
    }
)
PRIVATE_MAPPING_KEY_FIELD_PATTERNS = (
    re.compile(r"^occurrence_counts$"),
    re.compile(r".*_(?:registry|registries|counts)$"),
    re.compile(r".*_by_(?:uid|seller|controller|query)$"),
)
SHORT_PRIVATE_MARKERS = frozenset({"bat", "qq", "tg", "wx"})
ASCII_WORD = re.compile(r"[0-9a-z_]", re.IGNORECASE)
DIGIT_SEPARATOR = re.compile(r"(?<=\d)[\W_]+(?=\d)")
IDENTITY_PREFIX = re.compile(
    r"(?i)^(?:telegram|tg|wechat|wx|qq|bat|\u5fae\u4fe1|\u7535\u62a5|\u8759\u8760)"
    r"\s*(?:[:\uff1a/@_-]\s*)+(.+)$"
)


class BlindLiteralScanError(RuntimeError):
    """Raised without echoing any sealed value."""


class BlindLiteralInputError(BlindLiteralScanError):
    """Raised when sealed dataset bytes violate the scanner input contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BlindLiteralInputError("sealed row is not an object")
                rows.append(value)
    except OSError as exc:
        raise BlindLiteralScanError("sealed input cannot be read") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BlindLiteralInputError("sealed input is malformed") from exc
    if not rows:
        raise BlindLiteralInputError("sealed input is empty")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BlindLiteralScanError("sealed input cannot be read") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BlindLiteralInputError("sealed input is malformed") from exc
    if not isinstance(value, dict):
        raise BlindLiteralInputError("sealed manifest is not an object")
    return value


def _canonical_self_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return _canonical_sha256(payload)


def _require_sorted_unique_strings(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value), key=lambda item: item.encode("utf-8"))
    ):
        raise BlindLiteralInputError(f"sealed registry {label} drift")
    return value


def validate_sealed_registry_public_receipt_structure(
    value: Any,
) -> dict[str, Any]:
    """Shared strict public-receipt structure for recovery and quality use."""

    if (
        not isinstance(value, dict)
        or set(value) != SEALED_REGISTRY_PUBLIC_RECEIPT_KEYS
        or value.get("version") != SEALED_REGISTRY_VERSION
        or value.get("status")
        != "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO"
        or value.get("canonical_self_hash") != _canonical_self_hash(value)
        or re.fullmatch(r"[0-9a-f]{64}", value.get("transaction_id", ""))
        is None
        or value.get("worlds_replayed") != 104
        or value.get("audit_worlds_projected") != 4
        or tuple(value.get("private_input_files_semantically_replayed", ()))
        != SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
        or value.get("private_input_file_count")
        != len(SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES)
        or value.get("labels_opened_for_exact_replay") is not True
        or value.get("labels_used_for_candidate_selection") is not False
        or value.get("labels_used_for_literal_selection") is not False
        or value.get("pair_label_rows_replayed") != 39_312
        or value.get("candidate_selection_changed") is not False
        or value.get("derangement_changed") is not False
        or value.get("quality_probe_run") is not False
        or any(value.get(name) is not False for name in SEALED_REGISTRY_FALSE_AUTHORIZATION_FIELDS)
        or any(
            value.get(name) != 0
            for name in (
                "private_values_returned",
                "private_relations_returned",
                "labels_returned",
                "qrels_returned",
                "observed_rows_modified",
            )
        )
    ):
        raise BlindLiteralInputError(
            "sealed registry public receipt structure drift"
        )
    for name in (
        "dataset_root_manifest",
        "builder_policy",
    ):
        record = value.get(name)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "canonical_self_hash",
        }:
            raise BlindLiteralInputError(
                "sealed registry public receipt input pin drift"
            )
    for name in ("literal_authority_source", "builder_source"):
        record = value.get(name)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise BlindLiteralInputError(
                "sealed registry public receipt source pin drift"
            )
    sealed = value.get("sealed_registry")
    if not isinstance(sealed, dict) or set(sealed) != {
        "path",
        "size_bytes",
        "sha256",
        "canonical_self_hash",
    }:
        raise BlindLiteralInputError(
            "sealed registry public receipt sidecar pin drift"
        )
    source_closure = value.get("source_closure")
    if (
        not isinstance(source_closure, dict)
        or source_closure.get("packer") != value.get("builder_source")
        or source_closure.get("step28_v13_v1_13_blind_literal_scan")
        != value.get("literal_authority_source")
    ):
        raise BlindLiteralInputError(
            "sealed registry public receipt source closure drift"
        )
    split_commitments = value.get("split_commitments")
    if not isinstance(split_commitments, dict) or tuple(split_commitments) != (
        "audit_a",
        "audit_b",
    ):
        raise BlindLiteralInputError(
            "sealed registry public receipt split universe drift"
        )
    for split_value in split_commitments.values():
        if not isinstance(split_value, dict) or set(split_value) != {
            "world_count",
            "forbidden_literal_count",
            "category_commitments",
            "allowed_noise_raw_surface_commitment",
        }:
            raise BlindLiteralInputError(
                "sealed registry public receipt split contract drift"
            )
        categories = split_value["category_commitments"]
        noise = split_value["allowed_noise_raw_surface_commitment"]
        if (
            split_value["world_count"] != 2
            or isinstance(split_value["forbidden_literal_count"], bool)
            or not isinstance(split_value["forbidden_literal_count"], int)
            or split_value["forbidden_literal_count"] <= 0
            or not isinstance(categories, dict)
            or tuple(categories) != SEALED_REGISTRY_CATEGORIES
            or not isinstance(noise, dict)
            or set(noise) != {"count", "sha256"}
            or isinstance(noise.get("count"), bool)
            or not isinstance(noise.get("count"), int)
            or noise.get("count", 0) <= 0
        ):
            raise BlindLiteralInputError(
                "sealed registry public receipt split commitment drift"
            )
        for commitment in (*categories.values(), noise):
            if (
                not isinstance(commitment, dict)
                or set(commitment) != {"count", "sha256"}
                or isinstance(commitment.get("count"), bool)
                or not isinstance(commitment.get("count"), int)
                or commitment.get("count", 0) <= 0
                or re.fullmatch(r"[0-9a-f]{64}", commitment.get("sha256", ""))
                is None
            ):
                raise BlindLiteralInputError(
                    "sealed registry public receipt commitment drift"
                )
    return value


def _load_sealed_registry(
    path: Path, *, dataset_root: Path, split: str
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Validate the relation-free sidecar and return only this split's literals."""

    path = path.resolve()
    if not path.is_file():
        raise BlindLiteralInputError("sealed literal registry is missing")
    value = _read_json(path)
    expected_keys = {
        "version",
        "status",
        "transaction_id",
        "dataset_root_manifest",
        "literal_authority_source",
        "builder_policy",
        "split_order",
        "private_relations_persisted",
        "labels_persisted",
        "labels_opened_for_exact_replay",
        "labels_used_for_candidate_selection",
        "labels_used_for_literal_selection",
        "pair_label_rows_replayed",
        "qrels_persisted",
        "observed_rows_modified",
        "private_input_files_semantically_replayed",
        "split_registries",
        "canonical_self_hash",
    }
    if (
        set(value) != expected_keys
        or value.get("version") != SEALED_REGISTRY_VERSION
        or value.get("status") != "SEALED_RELATION_FREE_LITERAL_REGISTRY"
        or not isinstance(value.get("transaction_id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value.get("transaction_id", ""))
        is None
        or value.get("canonical_self_hash") != _canonical_self_hash(value)
        or tuple(value.get("split_order", ())) != tuple(sorted(AUDIT_SPLITS))
        or value.get("private_relations_persisted") is not False
        or value.get("labels_persisted") is not False
        or value.get("labels_opened_for_exact_replay") is not True
        or value.get("labels_used_for_candidate_selection") is not False
        or value.get("labels_used_for_literal_selection") is not False
        or value.get("pair_label_rows_replayed") != 39_312
        or value.get("qrels_persisted") is not False
        or value.get("observed_rows_modified") != 0
        or tuple(value.get("private_input_files_semantically_replayed", ()))
        != SEALED_REGISTRY_SEMANTIC_PRIVATE_FILES
    ):
        raise BlindLiteralInputError("sealed literal registry contract drift")

    root_manifest_path = dataset_root / "root_manifest.json"
    root_manifest = _read_json(root_manifest_path)
    root_pin = value.get("dataset_root_manifest")
    if (
        not isinstance(root_pin, dict)
        or set(root_pin)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not str(root_pin.get("path", "")).endswith("root_manifest.json")
        or isinstance(root_pin.get("size_bytes"), bool)
        or not isinstance(root_pin.get("size_bytes"), int)
        or root_pin.get("size_bytes") != root_manifest_path.stat().st_size
        or root_pin.get("sha256") != _sha256_file(root_manifest_path)
        or root_pin.get("canonical_self_hash") != root_manifest.get("canonical_self_hash")
        or root_manifest.get("canonical_self_hash") != _canonical_self_hash(root_manifest)
    ):
        raise BlindLiteralInputError("sealed registry root-manifest binding drift")
    source_path = Path(__file__).resolve()
    source_pin = value.get("literal_authority_source")
    if (
        not isinstance(source_pin, dict)
        or set(source_pin) != {"path", "size_bytes", "sha256"}
        or not str(source_pin.get("path", "")).endswith(
            "scripts/step28_v13_v1_13_blind_literal_scan.py"
        )
        or source_pin.get("size_bytes") != source_path.stat().st_size
        or source_pin.get("sha256") != _sha256_file(source_path)
    ):
        raise BlindLiteralInputError("sealed registry literal authority drift")
    repo_root = source_path.parents[1]
    builder_policy_pin = value.get("builder_policy")
    if not isinstance(builder_policy_pin, dict) or set(builder_policy_pin) != {
        "path",
        "size_bytes",
        "sha256",
        "canonical_self_hash",
    }:
        raise BlindLiteralInputError("sealed registry builder-policy pin drift")
    policy_path = (repo_root / str(builder_policy_pin.get("path", ""))).resolve()
    expected_policy_path = (
        repo_root
        / "schema"
        / "step28_v13_v1_13_scientific_dataset_builder_policy.json"
    ).resolve()
    if policy_path != expected_policy_path or not policy_path.is_file():
        raise BlindLiteralInputError("sealed registry builder-policy path drift")
    builder_policy = _read_json(policy_path)
    if (
        builder_policy_pin.get("size_bytes") != policy_path.stat().st_size
        or builder_policy_pin.get("sha256") != _sha256_file(policy_path)
        or builder_policy_pin.get("canonical_self_hash")
        != builder_policy.get("canonical_self_hash")
        or builder_policy.get("canonical_self_hash")
        != _canonical_self_hash(builder_policy)
        or root_manifest.get("builder_policy_canonical_self_hash")
        != builder_policy.get("canonical_self_hash")
    ):
        raise BlindLiteralInputError("sealed registry builder-policy drift")

    split_registries = value.get("split_registries")
    if not isinstance(split_registries, dict) or set(split_registries) != AUDIT_SPLITS:
        raise BlindLiteralInputError("sealed registry split universe drift")
    split_value = split_registries.get(split)
    if not isinstance(split_value, dict) or set(split_value) != {
        "world_count",
        "categories",
        "allowed_noise_raw_surfaces",
        "forbidden_literal_count",
        "category_commitments",
        "allowed_noise_raw_surface_commitment",
    }:
        raise BlindLiteralInputError("sealed registry split contract drift")
    if isinstance(split_value["world_count"], bool) or not isinstance(
        split_value["world_count"], int
    ) or split_value["world_count"] <= 0:
        raise BlindLiteralInputError("sealed registry world count drift")

    categories_value = split_value["categories"]
    commitments = split_value["category_commitments"]
    if (
        not isinstance(categories_value, dict)
        or tuple(categories_value) != SEALED_REGISTRY_CATEGORIES
        or not isinstance(commitments, dict)
        or tuple(commitments) != SEALED_REGISTRY_CATEGORIES
    ):
        raise BlindLiteralInputError("sealed registry category universe drift")
    categories: dict[str, set[str]] = {}
    for category in SEALED_REGISTRY_CATEGORIES:
        rows = _require_sorted_unique_strings(
            categories_value[category], label="category values"
        )
        commitment = commitments[category]
        if (
            not isinstance(commitment, dict)
            or set(commitment) != {"count", "sha256"}
            or commitment.get("count") != len(rows)
            or commitment.get("sha256") != _canonical_sha256(rows)
        ):
            raise BlindLiteralInputError("sealed registry category commitment drift")
        categories[category] = set(rows)
    union = set().union(*categories.values())
    if split_value["forbidden_literal_count"] != len(union):
        raise BlindLiteralInputError("sealed registry literal cardinality drift")

    allowed_noise = _require_sorted_unique_strings(
        split_value["allowed_noise_raw_surfaces"], label="allowed noise surface"
    )
    allowed_commitment = split_value["allowed_noise_raw_surface_commitment"]
    if (
        not isinstance(allowed_commitment, dict)
        or set(allowed_commitment) != {"count", "sha256"}
        or allowed_commitment.get("count") != len(allowed_noise)
        or allowed_commitment.get("sha256") != _canonical_sha256(allowed_noise)
    ):
        raise BlindLiteralInputError("sealed registry noise commitment drift")
    binding = {
        "transaction_id": value["transaction_id"],
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "canonical_self_hash": value["canonical_self_hash"],
        "world_count": split_value["world_count"],
        "forbidden_literal_count": len(union),
        "category_counts": {
            category: len(categories[category])
            for category in SEALED_REGISTRY_CATEGORIES
        },
        "allowed_noise_raw_surface_count": len(allowed_noise),
    }
    return categories, binding


def _identity_value_hash(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.strip().casefold())
    if not normalized:
        raise BlindLiteralInputError("empty sealed identity value")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _manifest_file_record(
    manifest: dict[str, Any], relative: str
) -> dict[str, Any]:
    matches = [
        record
        for record in manifest.get("files", ())
        if isinstance(record, dict) and record.get("path") == relative
    ]
    if len(matches) != 1:
        raise BlindLiteralInputError("sealed manifest file registry drift")
    record = matches[0]
    if set(record) != {"path", "row_count", "size_bytes", "sha256"}:
        raise BlindLiteralInputError("sealed manifest file record drift")
    return record


def _literal_category(field: str) -> str:
    if field == "controller_uid":
        return "controller_uid"
    if field == "query_uid":
        return "query_uid"
    if field in EXPLICIT_LITERAL_FIELDS:
        return "identity_surface"
    if field == "mechanism":
        return "mechanism_marker"
    if field == "flag":
        return "negative_flag_marker"
    if field == "override_kind":
        return "override_marker"
    if field in {"identity_type", "fixed_type", "allowed_types"}:
        return "identity_type_marker"
    return "private_uid"


def _collect_literal_fields(
    value: Any,
    *,
    field: str = "",
) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and key
                and any(pattern.fullmatch(field) for pattern in PRIVATE_MAPPING_KEY_FIELD_PATTERNS)
            ):
                yield "private_uid", key
            yield from _collect_literal_fields(child, field=str(key))
        return
    if isinstance(value, list):
        for child in value:
            yield from _collect_literal_fields(child, field=field)
        return
    if not isinstance(value, str) or not value:
        return
    # Every private string is inventoried.  Restricting collection to UID/value
    # suffixes previously omitted mechanisms, flags, overrides and other private
    # generation markers, creating a false PASS path.
    yield _literal_category(field) if (
        field.endswith(("_uid", "_uids"))
        or field in EXPLICIT_LITERAL_FIELDS
        or field in {"mechanism", "flag", "override_kind", "identity_type", "fixed_type", "allowed_types"}
    ) else "private_string", value


def collect_complete_world_forbidden_literals(
    *,
    world_uid: str,
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    public_pair_endpoints: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    private_world: Mapping[str, Any],
    persisted_private_world_audit: Mapping[str, Any],
) -> set[str]:
    """Single authority for exact private values forbidden in visible text."""

    literals = {
        str(world_uid),
        *(str(row["seller_uid"]) for row in public_sellers),
        *(str(row["item_uid"]) for row in public_items),
        *(str(row["canonical_pair_uid"]) for row in public_pair_endpoints),
        *(str(row["query_uid"]) for row in qrels),
    }
    sensitive_value_fields = {
        "normalized_value",
        "identity_value",
        "mechanism",
        "mechanism_name",
        "flag",
        "override_kind",
    }

    def add_identity_equivalents(identity_type: str, raw_value: str) -> None:
        if not raw_value:
            return
        stripped = raw_value.strip()
        variants = {
            raw_value,
            stripped,
            stripped.lower(),
            stripped.upper(),
            stripped.casefold(),
            unicodedata.normalize("NFC", stripped),
            unicodedata.normalize("NFKC", stripped),
        }
        if identity_type == "telegram":
            handle = stripped.removeprefix("@").casefold()
            variants.update(
                {
                    handle,
                    f"@{handle}",
                    f"t.me/{handle}",
                    f"https://t.me/{handle}",
                }
            )
        elif identity_type == "external_url":
            without_scheme = re.sub(r"^https?://", "", stripped, flags=re.I)
            variants.update(
                {
                    without_scheme,
                    without_scheme.removeprefix("www."),
                    f"http://{without_scheme}",
                    f"https://{without_scheme}",
                }
            )
        elif identity_type == "phone":
            compact = re.sub(r"[\s()\-]", "", stripped)
            variants.add(compact)
            variants.add(compact.removeprefix("+"))
        literals.update(value for value in variants if value)

    for row in private_world.get("identity_assets", ()):
        add_identity_equivalents(
            str(row.get("identity_type", "")), str(row.get("identity_value", ""))
        )
    for row in private_world.get("identity_slots_audit", ()):
        add_identity_equivalents(
            str(row.get("identity_type", "")), str(row.get("raw_surface", ""))
        )

    fully_private_string_sections = {
        "mechanism_assignments",
        "positive_targets",
        "negative_flags",
        "override_audit",
    }

    def visit(
        value: Any,
        field: str = "",
        path: tuple[str, ...] = (),
        collect_all_strings: bool = False,
    ) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                name_text = str(name)
                visit(
                    child,
                    name_text,
                    (*path, name_text),
                    collect_all_strings
                    or (not path and name_text in fully_private_string_sections),
                )
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, field, path, collect_all_strings)
        elif isinstance(value, str) and (
            collect_all_strings
            or field.endswith("_uid")
            or field.endswith("_handle")
            or field in sensitive_value_fields
            or (field == "raw_surface" and any("identity" in name for name in path))
        ):
            if value:
                literals.add(value)

    visit(private_world)
    for _category, literal in _collect_literal_fields(
        persisted_private_world_audit
    ):
        literals.add(literal)
    return literals


def _collapse_digit_separators(value: str) -> str:
    previous = ""
    collapsed = value
    while collapsed != previous:
        previous = collapsed
        collapsed = DIGIT_SEPARATOR.sub("", collapsed)
    return collapsed


def _normalized_forms(value: str) -> set[str]:
    """Return symmetric comparison forms for both sealed and visible text."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    forms = {normalized, _collapse_digit_separators(normalized)}
    return {form for form in forms if form}


def _variants(value: str) -> set[str]:
    stripped = value.strip().strip("<>[](){}\"'")
    candidates = {
        value,
        stripped,
        stripped.lower(),
        stripped.upper(),
        stripped.casefold(),
        unicodedata.normalize("NFC", stripped),
        unicodedata.normalize("NFKC", stripped),
    }
    if stripped.startswith("@"):
        candidates.add(stripped[1:])
    lowered = stripped.casefold()
    for prefix in ("https://t.me/", "http://t.me/", "t.me/"):
        if lowered.startswith(prefix):
            candidates.add(stripped[len(prefix) :])
    prefix_match = IDENTITY_PREFIX.match(stripped)
    if prefix_match:
        candidates.add(prefix_match.group(1).lstrip("@"))
    url_value = stripped
    if lowered.startswith("www."):
        url_value = "https://" + stripped
    if "://" in url_value:
        parsed = urlsplit(url_value)
        if parsed.netloc:
            suffix = parsed.path
            if parsed.query:
                suffix += "?" + parsed.query
            host = parsed.netloc
            candidates.add(host + suffix)
            if host.casefold().startswith("www."):
                candidates.add(host[4:] + suffix)
            if parsed.hostname and parsed.hostname.casefold() != host.casefold():
                candidates.add(parsed.hostname + suffix)
    digits = "".join(character for character in stripped if character.isdecimal())
    if len(digits) >= 5:
        candidates.add(digits)
    variants: set[str] = set()
    for candidate in candidates:
        variants.update(_normalized_forms(candidate))
    return {
        candidate
        for candidate in variants
        if candidate
        and (len(candidate) >= 4 or candidate in SHORT_PRIVATE_MARKERS)
    }


def _literal_in_normalized_text(literal: str, normalized_text: str) -> bool:
    if len(literal) >= 4:
        return literal in normalized_text
    if literal not in SHORT_PRIVATE_MARKERS:
        return False
    start = 0
    while True:
        index = normalized_text.find(literal, start)
        if index < 0:
            return False
        before = normalized_text[index - 1] if index else ""
        after_index = index + len(literal)
        after = normalized_text[after_index] if after_index < len(normalized_text) else ""
        if (
            (not before or ASCII_WORD.fullmatch(before) is None)
            and (not after or ASCII_WORD.fullmatch(after) is None)
        ):
            return True
        start = index + 1


def contains_private_literal(text: str, literals: Iterable[str]) -> bool:
    """Check sealed literals without returning or echoing the matching value."""

    text_forms = _normalized_forms(text)
    literal_forms: set[str] = set()
    for literal in literals:
        if literal:
            literal_forms.update(_variants(str(literal)))
    return any(
        _literal_in_normalized_text(literal, text_form)
        for text_form in text_forms
        for literal in literal_forms
    )


def _visible_texts(split_root: Path) -> list[str]:
    profiles = _read_jsonl(split_root / OBSERVED_TEXT_FILES[0])
    items = _read_jsonl(split_root / OBSERVED_TEXT_FILES[1])
    texts: list[str] = []
    for row in profiles:
        if any(field not in row or not isinstance(row[field], str) for field in PROFILE_TEXT_FIELDS):
            raise BlindLiteralInputError("observed profile schema drift")
        texts.extend(row[field] for field in PROFILE_TEXT_FIELDS)
    for row in items:
        if not isinstance(row.get("title"), str) or not isinstance(row.get("description"), str):
            raise BlindLiteralInputError("observed item schema drift")
        texts.extend((row["title"], row["description"]))
    return texts


def scan(dataset_root: Path, split: str, sealed_registry: Path) -> dict[str, Any]:
    if split not in AUDIT_SPLITS:
        raise BlindLiteralScanError("only blinded audit splits are allowed")
    dataset_root = dataset_root.resolve()
    split_root = (dataset_root / split).resolve()
    if dataset_root not in split_root.parents or not split_root.is_dir():
        raise BlindLiteralScanError("unsafe or missing split root")
    required = [split_root / relative for relative in (*PRIVATE_LITERAL_FILES, *OBSERVED_TEXT_FILES)]
    if any(not path.is_file() for path in required):
        raise BlindLiteralInputError("required scan input is missing")

    literals: dict[str, set[str]] = {}
    private_row_counts: dict[str, int] = {}
    for relative in PRIVATE_LITERAL_FILES:
        rows = _read_jsonl(split_root / relative)
        private_row_counts[relative] = len(rows)
        for row in rows:
            for category, literal in _collect_literal_fields(row):
                literals.setdefault(category, set()).update(_variants(literal))
    sealed_categories, sealed_binding = _load_sealed_registry(
        sealed_registry, dataset_root=dataset_root, split=split
    )
    for category, values in sealed_categories.items():
        normalized_category = f"sealed_registry:{category}"
        for literal in values:
            literals.setdefault(normalized_category, set()).update(_variants(literal))
    if not literals or not literals.get("controller_uid") or not literals.get("query_uid"):
        raise BlindLiteralInputError("sealed literal inventory is incomplete")

    normalized = {category: set(values) for category, values in literals.items()}
    texts = _visible_texts(split_root)
    hit_counts: Counter[str] = Counter()
    for text in texts:
        for category, values in normalized.items():
            if contains_private_literal(text, values):
                hit_counts[category] += 1

    input_binding = {
        relative: {
            "size_bytes": (split_root / relative).stat().st_size,
            "sha256": _sha256_file(split_root / relative),
        }
        for relative in (*PRIVATE_LITERAL_FILES, *OBSERVED_TEXT_FILES)
    }
    hit_count = sum(hit_counts.values())
    receipt = {
        "version": VERSION,
        "status": "PASS_NO_PRIVATE_LITERAL_HIT" if hit_count == 0 else "FAIL_PRIVATE_LITERAL_HIT",
        "split": split,
        "private_relation_rows_returned": 0,
        "private_values_returned": 0,
        "labels_opened": False,
        "world_reconstructed": False,
        "private_literal_file_count": len(PRIVATE_LITERAL_FILES),
        "private_row_counts": private_row_counts,
        "sealed_registry_binding": sealed_binding,
        "literal_category_counts": {
            category: len(values) for category, values in sorted(normalized.items())
        },
        "visible_text_count": len(texts),
        "hit_count": hit_count,
        "hit_category_counts": dict(sorted(hit_counts.items())),
        "input_binding": input_binding,
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "canonical_self_hash"}
    )
    return receipt


def scan_registry_isolation(dataset_root: Path) -> dict[str, Any]:
    """Verify cross-split private registries without returning any raw value."""

    dataset_root = dataset_root.resolve()
    root_manifest_path = dataset_root / "root_manifest.json"
    if not dataset_root.is_dir() or not root_manifest_path.is_file():
        raise BlindLiteralScanError("unsafe or missing dataset root")
    root_manifest = _read_json(root_manifest_path)
    if (
        root_manifest.get("canonical_self_hash") != _canonical_self_hash(root_manifest)
        or tuple(root_manifest.get("split_order", ())) != SPLITS
    ):
        raise BlindLiteralInputError("root manifest contract drift")

    registry_sets: dict[str, dict[str, set[str]]] = {}
    input_binding: dict[str, dict[str, Any]] = {
        "root_manifest.json": {
            "size_bytes": root_manifest_path.stat().st_size,
            "sha256": _sha256_file(root_manifest_path),
        }
    }
    split_commitments: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        split_root = (dataset_root / split).resolve()
        if dataset_root not in split_root.parents or not split_root.is_dir():
            raise BlindLiteralScanError("unsafe or missing split root")
        manifest_path = split_root / "split_manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("canonical_self_hash") != _canonical_self_hash(manifest)
            or manifest.get("split") != split
            or root_manifest.get("split_manifest_self_hashes", {}).get(split)
            != manifest.get("canonical_self_hash")
        ):
            raise BlindLiteralInputError("split manifest contract drift")
        input_binding[f"{split}/split_manifest.json"] = {
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
        }
        rows_by_name: dict[str, list[dict[str, Any]]] = {}
        for relative in PRIVATE_LITERAL_FILES:
            record = _manifest_file_record(manifest, relative)
            path = split_root / relative
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or _sha256_file(path) != record["sha256"]
            ):
                raise BlindLiteralInputError("sealed registry input bytes drift")
            rows = _read_jsonl(path)
            if len(rows) != int(record["row_count"]):
                raise BlindLiteralInputError("sealed registry input row-count drift")
            rows_by_name[relative] = rows
            input_binding[f"{split}/{relative}"] = {
                "size_bytes": path.stat().st_size,
                "sha256": record["sha256"],
            }

        controller_values = {
            str(row.get("controller_uid", ""))
            for row in rows_by_name["private/controller_membership.jsonl"]
        }
        query_values = {
            str(row.get("query_uid", ""))
            for row in rows_by_name["private/qrels.jsonl"]
        }
        identity_values: set[str] = set()
        for row in rows_by_name["private/world_generation_audit.jsonl"]:
            assets = row.get("identity_assets")
            if not isinstance(assets, list):
                raise BlindLiteralInputError("sealed identity asset registry drift")
            for asset in assets:
                if not isinstance(asset, dict) or not isinstance(
                    asset.get("identity_value"), str
                ):
                    raise BlindLiteralInputError("sealed identity asset row drift")
                identity_values.add(_identity_value_hash(asset["identity_value"]))
        if "" in controller_values or "" in query_values or not identity_values:
            raise BlindLiteralInputError("sealed private registry is incomplete")
        registry_sets[split] = {
            "identity_value": identity_values,
            "controller_uid": controller_values,
            "query_uid": query_values,
        }
        computed = {
            "identity_value": {
                "count": len(identity_values),
                "sha256": _canonical_sha256(sorted(identity_values)),
            },
            "controller_uid": {
                "count": len(controller_values),
                "sha256": _canonical_sha256(sorted(controller_values)),
            },
            "query_uid": {
                "count": len(query_values),
                "sha256": _canonical_sha256(sorted(query_values)),
            },
        }
        expected = {
            "identity_value": {
                "count": manifest.get("identity_value_registry_count"),
                "sha256": manifest.get("identity_value_registry_sha256"),
            },
            "controller_uid": dict(manifest.get("uid_registries", {}).get("controller", {})),
            "query_uid": dict(manifest.get("uid_registries", {}).get("query", {})),
        }
        if computed != expected:
            raise BlindLiteralInputError("sealed split registry commitment drift")
        split_commitments[split] = computed

    overlap_counts: dict[str, int] = {}
    union_commitments: dict[str, dict[str, Any]] = {}
    for kind in ("identity_value", "controller_uid", "query_uid"):
        pairwise_overlap = 0
        union: set[str] = set()
        for index, left in enumerate(SPLITS):
            union.update(registry_sets[left][kind])
            for right in SPLITS[index + 1 :]:
                pairwise_overlap += len(
                    registry_sets[left][kind] & registry_sets[right][kind]
                )
        overlap_counts[kind] = pairwise_overlap
        union_commitments[kind] = {
            "count": len(union),
            "sha256": _canonical_sha256(sorted(union)),
        }
    expected_union = {
        "identity_value": {
            "count": root_manifest.get("identity_value_registry_count"),
            "sha256": root_manifest.get("identity_value_registry_sha256"),
        },
        "controller_uid": dict(
            root_manifest.get("uid_registries", {}).get("controller", {})
        ),
        "query_uid": dict(root_manifest.get("uid_registries", {}).get("query", {})),
    }
    if union_commitments != expected_union:
        raise BlindLiteralInputError("sealed root registry commitment drift")
    if any(overlap_counts.values()):
        raise BlindLiteralInputError("sealed cross-split registry overlap")

    receipt = {
        "version": VERSION,
        "status": "PASS_PRIVATE_REGISTRY_SPLIT_ISOLATION",
        "split_order": list(SPLITS),
        "private_relation_rows_returned": 0,
        "private_values_returned": 0,
        "labels_opened": False,
        "world_reconstructed": False,
        "split_commitments": split_commitments,
        "union_commitments": union_commitments,
        "cross_split_overlap_counts": overlap_counts,
        "input_binding": input_binding,
        "canonical_self_hash": None,
    }
    receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--mode",
        choices=("literal-scan", "registry-isolation"),
        default="literal-scan",
    )
    parser.add_argument("--split", choices=sorted(AUDIT_SPLITS))
    parser.add_argument("--sealed-registry")
    args = parser.parse_args()
    try:
        if args.mode == "literal-scan":
            if args.split is None or args.sealed_registry is None:
                raise BlindLiteralScanError(
                    "literal scan requires an audit split and sealed registry"
                )
            receipt = scan(
                Path(args.dataset_root), args.split, Path(args.sealed_registry)
            )
        else:
            if args.split is not None or args.sealed_registry is not None:
                raise BlindLiteralScanError(
                    "registry isolation does not accept a split or sealed registry"
                )
            receipt = scan_registry_isolation(Path(args.dataset_root))
    except BlindLiteralInputError as exc:
        error = {
            "version": VERSION,
            "status": "FAIL_SEALED_INPUT_INVALID",
            "exception_type": type(exc).__name__,
            "private_values_returned": 0,
            "private_relation_rows_returned": 0,
            "canonical_self_hash": None,
        }
        error["canonical_self_hash"] = _canonical_sha256(
            {key: value for key, value in error.items() if key != "canonical_self_hash"}
        )
        print(_canonical_bytes(error).decode("utf-8"), flush=True)
        raise SystemExit(3)
    except Exception as exc:
        error = {
            "version": VERSION,
            "status": "SCANNER_EXECUTION_FAILED",
            "exception_type": type(exc).__name__,
            "private_values_returned": 0,
            "private_relation_rows_returned": 0,
            "canonical_self_hash": None,
        }
        error["canonical_self_hash"] = _canonical_sha256(
            {key: value for key, value in error.items() if key != "canonical_self_hash"}
        )
        print(_canonical_bytes(error).decode("utf-8"), flush=True)
        raise SystemExit(2)
    print(_canonical_bytes(receipt).decode("utf-8"), flush=True)


if __name__ == "__main__":
    main()
