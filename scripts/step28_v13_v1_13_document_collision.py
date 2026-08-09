#!/usr/bin/env python3
"""Pure Step28-v13 v1.13 document-collision contract primitives.

This module is intentionally unable to create seeds, derive formal
capabilities, generate candidate text, write dataset rows, or train models.
It only defines exact document bytes, validates the frozen historical hash
registries, verifies one pinned style-profile fact, and derives an in-memory
candidate key from an already isolated document-variation capability.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import step28_v13_common as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_document_collision_policy.json"
)
POLICY_VERSION = "2026-08-09-step28-v13-v1-13-document-collision-policy-v1"
POLICY_STATUS = "DESIGN_ONLY_IMPLEMENTATION_IN_PROGRESS_NO_FORMAL_AUTHORIZATION"
SPLITS = ("train", "development", "audit_a", "audit_b")
SELLER_DOCUMENT_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
CANDIDATE_DOMAIN = b"step28-v13-v1.13-document-candidate"
FIELD_SEPARATOR = b"\x1f"
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORLD_UID_RE = re.compile(r"^w_[0-9a-f]{64}$")
FORMAL_AUTHORIZATION_KEYS = frozenset(
    {
        "audit_truth_access",
        "candidate_generation",
        "formal_capability_derivation",
        "formal_dataset_generation",
        "formal_model_training",
        "formal_seed_ceremony",
    }
)


class CollisionContractError(common.ContractError):
    """Raised when the frozen v1.13 collision contract fails closed."""


@dataclass(frozen=True)
class HistoricalExclusionRegistries:
    """Immutable, hash-only exclusions reconstructed from frozen evidence."""

    item_document_hashes: frozenset[str]
    seller_document_hashes: frozenset[str]
    identity_value_hashes: frozenset[str]
    uid_hashes: Mapping[str, frozenset[str]]
    consumed_capability_commitments: frozenset[str]
    successful_v1_2_item_row_count: int
    successful_v1_2_seller_row_count: int
    successful_v1_2_item_unique_count: int
    successful_v1_2_seller_unique_count: int
    failed_v1_12_item_unique_count: int
    failed_v1_12_seller_unique_count: int


def _require_exact_keys(
    value: Mapping[str, Any], expected: Collection[str], *, label: str
) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise CollisionContractError(
            f"{label} keyset drift: missing={sorted(expected_set - observed)} "
            f"extra={sorted(observed - expected_set)}"
        )


def _require_plain_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CollisionContractError(f"{label} must be a JSON integer")
    return value


def _validate_hex_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None:
        raise CollisionContractError(f"{label} is not a lowercase SHA-256 value")
    return value


def _validate_canonical_self_hash(
    document: Mapping[str, Any], *, expected: str | None, label: str
) -> str:
    claimed = _validate_hex_sha256(
        document.get("canonical_self_hash"), label=f"{label}.canonical_self_hash"
    )
    payload = dict(document)
    payload.pop("canonical_self_hash")
    observed = common.canonical_sha256(payload)
    if claimed != observed:
        raise CollisionContractError(
            f"{label} canonical self-hash drift: claimed={claimed} observed={observed}"
        )
    if expected is not None and claimed != expected:
        raise CollisionContractError(
            f"{label} canonical self-hash does not match the policy pin"
        )
    return claimed


def _verify_pinned_json(
    spec: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    path = common.verify_file_pin(spec, label=label)
    try:
        value = common.load_json(path)
    except common.ContractError as exc:
        raise CollisionContractError(f"Invalid strict JSON for {label}") from exc
    if not isinstance(value, dict):
        raise CollisionContractError(f"{label} must be a JSON object")
    if "canonical_self_hash" in spec:
        _validate_canonical_self_hash(
            value, expected=str(spec["canonical_self_hash"]), label=label
        )
    return path, value


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise CollisionContractError(
            "Only the canonical v1.13 document-collision policy may produce PASS"
        )
    try:
        policy = common.load_json(path)
    except common.ContractError as exc:
        raise CollisionContractError("Invalid strict JSON for v1.13 policy") from exc
    if not isinstance(policy, dict):
        raise CollisionContractError("V1.13 collision policy must be a JSON object")
    _require_exact_keys(
        policy,
        {
            "version",
            "status",
            "claim_boundary",
            "formal_authorizations",
            "split_order",
            "document_contract",
            "candidate_key_contract",
            "style_title_missing_attestation",
            "historical_sources",
            "canonical_self_hash",
        },
        label="v1.13 policy",
    )
    _validate_canonical_self_hash(policy, expected=None, label="v1.13 policy")
    if policy.get("version") != POLICY_VERSION or policy.get("status") != POLICY_STATUS:
        raise CollisionContractError("V1.13 collision policy version/status drift")
    if tuple(policy.get("split_order", ())) != SPLITS:
        raise CollisionContractError("V1.13 split order drift")

    authorizations = policy.get("formal_authorizations")
    if not isinstance(authorizations, dict):
        raise CollisionContractError("Formal authorization block is missing")
    _require_exact_keys(
        authorizations, FORMAL_AUTHORIZATION_KEYS, label="formal_authorizations"
    )
    if any(value is not False for value in authorizations.values()):
        raise CollisionContractError("This design-only policy cannot authorize formal work")

    document = policy.get("document_contract")
    if not isinstance(document, dict):
        raise CollisionContractError("Document contract block is missing")
    common.verify_file_pin(document["contract"], label="frozen v1.13 design contract")
    if document.get("hash_algorithm") != "sha256":
        raise CollisionContractError("Document hash algorithm drift")
    item = document.get("item_document", {})
    if item != {
        "encoding": "utf-8",
        "fields_in_canonical_json": ["description", "title"],
        "json_allow_nan": False,
        "json_ensure_ascii": False,
        "json_separators": [",", ":"],
        "json_sort_keys": True,
        "postprocess_production_redactor_output": False,
    }:
        raise CollisionContractError("Item-document byte contract drift")
    seller = document.get("seller_document", {})
    if seller != {
        "drop_empty_after_strip": True,
        "encoding": "utf-8",
        "field_order": list(SELLER_DOCUMENT_FIELDS),
        "field_strip": True,
        "separator_hex": "0a",
    }:
        raise CollisionContractError("Seller-document byte contract drift")

    candidate = policy.get("candidate_key_contract", {})
    if candidate != {
        "candidate_index_max_inclusive": 31,
        "candidate_index_min_inclusive": 0,
        "capability_key_bytes": 32,
        "domain_utf8": CANDIDATE_DOMAIN.decode("ascii"),
        "field_separator_hex": FIELD_SEPARATOR.hex(),
        "mac": "hmac-sha256",
        "message_fields": [
            "domain_utf8",
            "split_utf8",
            "world_uid_utf8",
            "candidate_index_uint64_be",
        ],
        "must_not_replace_text_key_hex": True,
    }:
        raise CollisionContractError("Candidate-key byte contract drift")
    return policy


def item_document_bytes(*, title: str, description: str) -> bytes:
    """Return exact bytes for a production-redacted item document."""

    if not isinstance(title, str) or not isinstance(description, str):
        raise CollisionContractError("Item title and description must both be strings")
    return json.dumps(
        {"description": description, "title": title},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def item_document_hash(*, title: str, description: str) -> str:
    return hashlib.sha256(
        item_document_bytes(title=title, description=description)
    ).hexdigest()


def seller_document_bytes(profile: Mapping[str, Any]) -> bytes:
    """Return exact UTF-8 bytes for the five frozen M0-visible fields."""

    if not isinstance(profile, Mapping):
        raise CollisionContractError("Seller profile must be a mapping")
    values: list[str] = []
    for field in SELLER_DOCUMENT_FIELDS:
        if field not in profile:
            raise CollisionContractError(f"Seller profile lacks frozen field: {field}")
        value = profile[field]
        if not isinstance(value, str):
            raise CollisionContractError(f"Seller field must be a string: {field}")
        stripped = value.strip()
        if stripped:
            values.append(stripped)
    return "\n".join(values).encode("utf-8")


def seller_document_hash(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(seller_document_bytes(profile)).hexdigest()


def validate_row_hash_multiplicity(
    *,
    row_count: int,
    row_hashes: Sequence[str],
    registry_hashes: Collection[str],
    label: str,
) -> None:
    """Enforce row == hash-row == unique-hash == registry cardinality."""

    count = _require_plain_int(row_count, label=f"{label}.row_count")
    if count < 0:
        raise CollisionContractError(f"{label}.row_count cannot be negative")
    if not isinstance(row_hashes, Sequence) or isinstance(
        row_hashes, (str, bytes, Mapping)
    ):
        raise CollisionContractError(f"{label}.row_hashes must be a sequence of hashes")
    materialized = list(row_hashes)
    for index, value in enumerate(materialized):
        _validate_hex_sha256(value, label=f"{label}.row_hashes[{index}]")
    row_set = set(materialized)
    if isinstance(registry_hashes, (str, bytes, Mapping)):
        raise CollisionContractError(
            f"{label}.registry_hashes must be a collection of individual hashes"
        )
    registry_materialized = list(registry_hashes)
    for index, value in enumerate(registry_materialized):
        _validate_hex_sha256(value, label=f"{label}.registry_hashes[{index}]")
    registry_set = set(registry_materialized)
    if len(registry_materialized) != len(registry_set):
        raise CollisionContractError(f"{label} registry contains duplicate hashes")
    if not (
        count
        == len(materialized)
        == len(row_set)
        == len(registry_materialized)
        == len(registry_set)
    ):
        raise CollisionContractError(
            f"{label} multiplicity failure: rows={count} hash_rows={len(materialized)} "
            f"unique_hashes={len(row_set)} registry_rows={len(registry_materialized)} "
            f"registry_unique={len(registry_set)}"
        )
    if row_set != registry_set:
        raise CollisionContractError(f"{label} registry membership mismatch")


def derive_candidate_key(
    document_variation_key: bytes,
    *,
    split: str,
    world_uid: str,
    candidate_index: int,
) -> bytes:
    """Derive one candidate key without mutating any source capability."""

    if not isinstance(document_variation_key, bytes) or len(document_variation_key) != 32:
        raise CollisionContractError("document_variation capability must be 32 bytes")
    if split not in SPLITS:
        raise CollisionContractError(f"Unknown split for candidate derivation: {split}")
    if not isinstance(world_uid, str) or WORLD_UID_RE.fullmatch(world_uid) is None:
        raise CollisionContractError("Candidate derivation requires a canonical world UID")
    index = _require_plain_int(candidate_index, label="candidate_index")
    if not 0 <= index <= 31:
        raise CollisionContractError("candidate_index must be in the frozen range 0..31")
    message = FIELD_SEPARATOR.join(
        (
            CANDIDATE_DOMAIN,
            split.encode("ascii"),
            world_uid.encode("ascii"),
            index.to_bytes(8, byteorder="big", signed=False),
        )
    )
    return hmac.new(document_variation_key, message, hashlib.sha256).digest()


def derive_candidate_key_hex(
    document_variation_key_hex: str,
    *,
    split: str,
    world_uid: str,
    candidate_index: int,
) -> str:
    _validate_hex_sha256(
        document_variation_key_hex, label="document_variation_key_hex"
    )
    return derive_candidate_key(
        bytes.fromhex(document_variation_key_hex),
        split=split,
        world_uid=world_uid,
        candidate_index=candidate_index,
    ).hex()


def verify_style_title_missing_attestation(
    policy: Mapping[str, Any],
) -> dict[str, float]:
    spec = policy["style_title_missing_attestation"]
    expected_keys = list(spec["expected_quantile_keys"])
    _, attestation = _verify_pinned_json(spec["attestation"], label="title-missing attestation")
    if (
        attestation.get("version") != spec["attestation"]["version"]
        or attestation.get("status") != spec["attestation"]["status"]
        or attestation.get("source") != spec["source"]
        or attestation.get("expected_quantile_keys") != expected_keys
        or attestation.get("exact_keyset_match") is not True
        or attestation.get("all_values_numeric_zero") is not True
        or attestation.get("protected_labels_read") is not False
        or attestation.get("formal_seed_created") is not False
        or _require_plain_int(
            attestation.get("formal_rows_created"),
            label="title-missing attestation.formal_rows_created",
        )
        != 0
        or attestation.get("model_training_started") is not False
    ):
        raise CollisionContractError("Title-missing attestation contract drift")
    _, source = _verify_pinned_json(spec["source"], label="style-profile source")
    try:
        observed = source["seller_equal_weight_quantiles"]["title_missing"]
    except (KeyError, TypeError) as exc:
        raise CollisionContractError("Pinned style profile lacks title_missing quantiles") from exc
    if not isinstance(observed, dict) or list(observed) != expected_keys:
        raise CollisionContractError("Style-profile title_missing quantile key/order drift")
    output: dict[str, float] = {}
    for key in expected_keys:
        value = observed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            raise CollisionContractError(
                f"Style-profile title_missing quantile is not numeric zero: {key}"
            )
        output[key] = float(value)
    attested_values = attestation.get("observed_values")
    if not isinstance(attested_values, dict) or list(attested_values) != expected_keys:
        raise CollisionContractError("Attested title-missing key/order drift")
    for key in expected_keys:
        value = attested_values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            raise CollisionContractError(
                f"Attested title-missing value is not numeric zero: {key}"
            )
    if attested_values != output:
        raise CollisionContractError("Attested title-missing values do not replay")
    return output


def _iter_jsonl_objects(path: Path, *, label: str) -> Iterator[dict[str, Any]]:
    with open(common.filesystem_path(path), "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise CollisionContractError(f"Blank JSONL row in {label}:{line_number}")
            try:
                value = json.loads(
                    raw_line.decode("utf-8"),
                    object_pairs_hook=common._reject_duplicate_pairs,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                common.ContractError,
            ) as exc:
                raise CollisionContractError(
                    f"Invalid UTF-8 JSONL row in {label}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise CollisionContractError(
                    f"Non-object JSONL row in {label}:{line_number}"
                )
            yield value


def _validate_sorted_hash_list(
    values: Any,
    *,
    expected_count: int,
    expected_digest: str | None,
    label: str,
) -> frozenset[str]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise CollisionContractError(f"{label} count drift")
    previous: str | None = None
    for index, value in enumerate(values):
        current = _validate_hex_sha256(value, label=f"{label}[{index}]")
        if previous is not None and current <= previous:
            raise CollisionContractError(f"{label} must be strictly sorted and unique")
        previous = current
    if expected_digest is not None:
        _validate_hex_sha256(expected_digest, label=f"{label}.digest")
        observed_digest = common.canonical_sha256(values)
        if observed_digest != expected_digest:
            raise CollisionContractError(f"{label} digest drift")
    return frozenset(values)


def _manifest_member(
    manifest: Mapping[str, Any], *, relative_path: str, expected: Mapping[str, Any]
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CollisionContractError("Split manifest lacks files list")
    paths: list[str] = []
    matches: list[Mapping[str, Any]] = []
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CollisionContractError("Malformed split-manifest file record")
        paths.append(record["path"])
        if record["path"] == relative_path:
            matches.append(record)
    if len(paths) != len(set(paths)):
        raise CollisionContractError("Duplicate path in split-manifest files")
    if len(matches) != 1:
        raise CollisionContractError(f"Split manifest must pin exactly one {relative_path}")
    record = matches[0]
    if (
        record.get("sha256") != expected["sha256"]
        or record.get("size_bytes") != expected["size_bytes"]
        or record.get("model_mount_allowed") is not True
    ):
        raise CollisionContractError(f"Split-manifest member drift: {relative_path}")


def _load_successful_v1_2(
    spec: Mapping[str, Any],
) -> tuple[frozenset[str], frozenset[str], int, int]:
    _, release = _verify_pinned_json(
        spec["release_manifest"], label="successful v1.2 release manifest"
    )
    release_pin = spec["release_manifest"]
    if (
        release.get("version") != release_pin["version"]
        or release.get("status") != release_pin["status"]
        or release.get("run_id") != release_pin["run_id"]
    ):
        raise CollisionContractError("Successful v1.2 release identity drift")
    receipts = release.get("split_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(SPLITS):
        raise CollisionContractError("Successful v1.2 split receipt set drift")

    all_items: set[str] = set()
    all_sellers: set[str] = set()
    item_rows_total = 0
    seller_rows_total = 0
    for split in SPLITS:
        split_spec = spec["splits"][split]
        _, manifest = _verify_pinned_json(
            split_spec["manifest"], label=f"successful v1.2 {split} manifest"
        )
        if (
            manifest.get("status") != "PASS_SPLIT_DATASET_READY"
            or manifest.get("split") != split
            or manifest.get("world_count") != 500
            or manifest.get("item_count") != split_spec["item_row_count"]
            or manifest.get("seller_count") != split_spec["seller_row_count"]
        ):
            raise CollisionContractError(f"Successful v1.2 {split} manifest facts drift")
        receipt = receipts[split]
        if (
            not isinstance(receipt, dict)
            or receipt.get("manifest_sha256") != split_spec["manifest"]["sha256"]
            or receipt.get("manifest_self_sha256")
            != split_spec["manifest"]["canonical_self_hash"]
        ):
            raise CollisionContractError(f"Release-to-split manifest pin drift: {split}")

        _manifest_member(
            manifest,
            relative_path="observed/redacted_items.jsonl",
            expected=split_spec["item_rows"],
        )
        _manifest_member(
            manifest,
            relative_path="observed/seller_profiles.jsonl",
            expected=split_spec["seller_rows"],
        )
        item_path = common.verify_file_pin(
            split_spec["item_rows"], label=f"successful v1.2 {split} item rows"
        )
        seller_path = common.verify_file_pin(
            split_spec["seller_rows"], label=f"successful v1.2 {split} seller rows"
        )

        split_items: set[str] = set()
        split_item_rows = 0
        for row in _iter_jsonl_objects(item_path, label=f"v1.2 {split} items"):
            if "title" not in row or "description" not in row:
                raise CollisionContractError(f"V1.2 {split} item lacks visible text fields")
            split_items.add(
                item_document_hash(title=row["title"], description=row["description"])
            )
            split_item_rows += 1

        split_sellers: set[str] = set()
        split_seller_rows = 0
        for row in _iter_jsonl_objects(seller_path, label=f"v1.2 {split} sellers"):
            split_sellers.add(seller_document_hash(row))
            split_seller_rows += 1

        if (
            split_item_rows != split_spec["item_row_count"]
            or len(split_items) != split_spec["item_document_unique_count"]
            or common.canonical_sha256(sorted(split_items))
            != split_spec["item_document_hashes_sha256"]
            or split_seller_rows != split_spec["seller_row_count"]
            or len(split_sellers) != split_spec["seller_document_unique_count"]
            or common.canonical_sha256(sorted(split_sellers))
            != split_spec["seller_document_hashes_sha256"]
        ):
            raise CollisionContractError(f"Successful v1.2 {split} document replay drift")
        if all_items.intersection(split_items) or all_sellers.intersection(split_sellers):
            raise CollisionContractError(f"Successful v1.2 cross-split document collision: {split}")
        all_items.update(split_items)
        all_sellers.update(split_sellers)
        item_rows_total += split_item_rows
        seller_rows_total += split_seller_rows

    if (
        item_rows_total != spec["all_item_row_count"]
        or seller_rows_total != spec["all_seller_row_count"]
        or len(all_items) != spec["all_item_document_unique_count"]
        or len(all_sellers) != spec["all_seller_document_unique_count"]
        or common.canonical_sha256(sorted(all_items))
        != spec["all_item_document_hashes_sha256"]
        or common.canonical_sha256(sorted(all_sellers))
        != spec["all_seller_document_hashes_sha256"]
    ):
        raise CollisionContractError("Successful v1.2 aggregate document replay drift")
    return frozenset(all_items), frozenset(all_sellers), item_rows_total, seller_rows_total


def _load_failed_v1_12(
    spec: Mapping[str, Any],
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    Mapping[str, frozenset[str]],
    frozenset[str],
    dict[str, Any],
]:
    _, archive = _verify_pinned_json(spec["archive"], label="failed v1.12 exclusion archive")
    archive_pin = spec["archive"]
    if (
        archive.get("version") != archive_pin["version"]
        or archive.get("status") != archive_pin["status"]
        or archive.get("uid_hash_method") != spec["uid_hash_method"]
        or archive.get("archive_content_scope")
        != "THIS_HASH_ONLY_ARCHIVE_ONLY_NOT_THE_STILL_EXISTING_FAILED_RUN_PAYLOAD"
        or archive.get("future_registry_must_not_require_deleted_failed_payloads")
        is not True
        or archive.get("labels_or_oracle_rows_persisted_in_this_archive") is not False
        or archive.get("raw_identity_values_persisted_in_this_archive") is not False
        or archive.get("raw_item_or_seller_text_persisted_in_this_archive") is not False
        or archive.get("raw_private_keys_persisted_in_this_archive") is not False
        or archive.get("raw_uids_persisted_in_this_archive") is not False
        or archive.get("scientific_metrics_produced") is not False
    ):
        raise CollisionContractError("Failed v1.12 archive boundary drift")

    base_pin = spec["base_identity_archive"]
    archive_base_pin = archive.get("base_identity_exclusion_archive")
    if not isinstance(archive_base_pin, dict) or archive_base_pin != {
        "canonical_self_hash": base_pin["canonical_self_hash"],
        "path": base_pin["path"],
        "sha256": base_pin["sha256"],
        "size_bytes": base_pin["size_bytes"],
    }:
        raise CollisionContractError("V1.12-to-base identity archive pin drift")
    _, base = _verify_pinned_json(base_pin, label="identity exclusions through v1.11")
    if (
        base.get("version") != base_pin["version"]
        or base.get("status") != base_pin["status"]
        or base.get("combined_unique_value_hash_count")
        != base_pin["combined_unique_value_hash_count"]
        or base.get("raw_identity_values_persisted") is not False
        or base.get("raw_private_keys_persisted") is not False
        or base.get("scientific_metrics_produced") is not False
        or base.get("future_registry_must_not_require_deleted_failed_payloads")
        is not True
    ):
        raise CollisionContractError("Base identity exclusion boundary drift")
    base_identity = _validate_sorted_hash_list(
        base.get("combined_value_hashes"),
        expected_count=base_pin["combined_unique_value_hash_count"],
        expected_digest=None,
        label="identity exclusions through v1.11",
    )

    identity_count = spec["v1_12_identity_value_hash_count"]
    v1_12_identity = _validate_sorted_hash_list(
        archive.get("v1_12_identity_value_hashes"),
        expected_count=identity_count,
        expected_digest=archive.get("v1_12_identity_value_hashes_sha256"),
        label="v1.12 identity value hashes",
    )
    if archive.get("v1_12_identity_value_hash_count") != identity_count:
        raise CollisionContractError("V1.12 identity count field drift")
    combined_identity = base_identity.union(v1_12_identity)
    if (
        len(combined_identity) != spec["combined_identity_value_hash_count_through_v1_12"]
        or archive.get("combined_identity_value_hash_count_through_v1_12")
        != len(combined_identity)
        or common.canonical_sha256(sorted(combined_identity))
        != spec["combined_identity_value_hashes_sha256_through_v1_12"]
        or archive.get("combined_identity_value_hashes_sha256_through_v1_12")
        != spec["combined_identity_value_hashes_sha256_through_v1_12"]
    ):
        raise CollisionContractError("Combined identity exclusion replay drift")

    item_count = spec["v1_12_item_document_hash_count"]
    items = _validate_sorted_hash_list(
        archive.get("v1_12_item_document_hashes"),
        expected_count=item_count,
        expected_digest=archive.get("v1_12_item_document_hashes_sha256"),
        label="v1.12 item document hashes",
    )
    seller_count = spec["v1_12_seller_document_hash_count"]
    sellers = _validate_sorted_hash_list(
        archive.get("v1_12_seller_document_hashes"),
        expected_count=seller_count,
        expected_digest=archive.get("v1_12_seller_document_hashes_sha256"),
        label="v1.12 seller document hashes",
    )
    if (
        archive.get("v1_12_item_document_hash_count") != item_count
        or archive.get("v1_12_seller_document_hash_count") != seller_count
    ):
        raise CollisionContractError("V1.12 document count field drift")

    uid_container = archive.get("v1_12_uid_hash_registries")
    if not isinstance(uid_container, dict) or set(uid_container) != set(spec["uid_hash_counts"]):
        raise CollisionContractError("V1.12 UID registry type set drift")
    uid_hashes: dict[str, frozenset[str]] = {}
    for name, expected_count in spec["uid_hash_counts"].items():
        entry = uid_container[name]
        if not isinstance(entry, dict) or entry.get("count") != expected_count:
            raise CollisionContractError(f"V1.12 UID registry count drift: {name}")
        uid_hashes[name] = _validate_sorted_hash_list(
            entry.get("hashes"),
            expected_count=expected_count,
            expected_digest=entry.get("hashes_sha256"),
            label=f"v1.12 {name} hashes",
        )

    commitment_sets: list[frozenset[str]] = []
    for field in (
        "forbidden_master_seed_commitments",
        "consumed_generator_capability_commitments",
        "consumed_m1_capability_commitments",
    ):
        values = archive.get(field)
        digest = archive.get(f"{field}_sha256")
        if not isinstance(values, list):
            raise CollisionContractError(f"Missing v1.12 commitment list: {field}")
        if len(values) != spec["commitment_counts"][field]:
            raise CollisionContractError(f"V1.12 commitment count drift: {field}")
        commitment_sets.append(
            _validate_sorted_hash_list(
                values,
                expected_count=len(values),
                expected_digest=digest,
                label=field,
            )
        )
    commitments = frozenset().union(*commitment_sets)
    if (
        len(commitments) != spec["consumed_commitment_count"]
        or sum(len(value) for value in commitment_sets) != len(commitments)
    ):
        raise CollisionContractError("V1.12 consumed commitments are not 37 unique values")

    return (
        items,
        sellers,
        frozenset(combined_identity),
        MappingProxyType(uid_hashes),
        commitments,
        archive,
    )


def load_historical_exclusion_registries(
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> HistoricalExclusionRegistries:
    """Rebuild and verify all frozen v1.2/v1.12 hash-only exclusions."""

    policy = load_policy(policy_path)
    sources = policy["historical_sources"]
    v1_2_items, v1_2_sellers, item_rows, seller_rows = _load_successful_v1_2(
        sources["successful_v1_2"]
    )
    (
        v1_12_items,
        v1_12_sellers,
        identities,
        uid_hashes,
        commitments,
        archive,
    ) = _load_failed_v1_12(sources["failed_v1_12"])

    expected_item_overlap = _validate_sorted_hash_list(
        archive.get("historical_v1_2_item_document_intersection_hashes"),
        expected_count=archive.get("historical_v1_2_item_document_intersection_count"),
        expected_digest=archive.get(
            "historical_v1_2_item_document_intersection_hashes_sha256"
        ),
        label="v1.12 recorded v1.2 item-document overlap",
    )
    observed_item_overlap = v1_2_items.intersection(v1_12_items)
    if observed_item_overlap != expected_item_overlap or len(observed_item_overlap) != 35:
        raise CollisionContractError("V1.2/v1.12 item-document overlap replay drift")
    observed_seller_overlap = v1_2_sellers.intersection(v1_12_sellers)
    if (
        observed_seller_overlap
        or archive.get("historical_v1_2_seller_document_intersection_count") != 0
    ):
        raise CollisionContractError("V1.2/v1.12 seller-document overlap replay drift")

    return HistoricalExclusionRegistries(
        item_document_hashes=frozenset(v1_2_items.union(v1_12_items)),
        seller_document_hashes=frozenset(v1_2_sellers.union(v1_12_sellers)),
        identity_value_hashes=identities,
        uid_hashes=uid_hashes,
        consumed_capability_commitments=commitments,
        successful_v1_2_item_row_count=item_rows,
        successful_v1_2_seller_row_count=seller_rows,
        successful_v1_2_item_unique_count=len(v1_2_items),
        successful_v1_2_seller_unique_count=len(v1_2_sellers),
        failed_v1_12_item_unique_count=len(v1_12_items),
        failed_v1_12_seller_unique_count=len(v1_12_sellers),
    )


def _summary(registries: HistoricalExclusionRegistries) -> dict[str, Any]:
    return {
        "status": "PASS_HASH_ONLY_HISTORICAL_REGISTRY_REPLAY",
        "formal_authorization": False,
        "item_document_exclusion_count": len(registries.item_document_hashes),
        "seller_document_exclusion_count": len(registries.seller_document_hashes),
        "identity_value_exclusion_count": len(registries.identity_value_hashes),
        "uid_hash_counts": {
            key: len(registries.uid_hashes[key]) for key in sorted(registries.uid_hashes)
        },
        "consumed_capability_commitment_count": len(
            registries.consumed_capability_commitments
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-config-only", action="store_true")
    mode.add_argument("--verify-historical-registry", action="store_true")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    title_missing = verify_style_title_missing_attestation(policy)
    if args.validate_config_only:
        output = {
            "status": "PASS_V1_13_COLLISION_CONFIG_ONLY",
            "formal_authorization": False,
            "split_order": list(SPLITS),
            "title_missing_quantiles": title_missing,
        }
    else:
        output = _summary(load_historical_exclusion_registries(args.policy))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
