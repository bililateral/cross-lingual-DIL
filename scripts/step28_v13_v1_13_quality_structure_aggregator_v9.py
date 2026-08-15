#!/usr/bin/env python3
"""Aggregate v9 label-free structural shortcut gates without opening truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import step28_v13_v1_13_quality_channel_policy_v9 as channel_policy
import step28_v13_v1_13_quality_channel_views_v9 as channel


VERSION = "2026-08-14-step28-v13-v1-13-quality-structure-aggregator-v9"
SPLITS = ("train", "development", "audit_a", "audit_b")
ZERO_TOLERANCE_FIELDS = (
    "registered_visible_occurrence_multiset_difference_count",
    "literal_code_hits_in_masked",
    "literal_code_hits_in_neutralized",
    "unregistered_code_hits",
    "unregistered_clone_foreign_code_hits",
    "view_keyset_difference_count",
    "neutralized_legal_code_permutation_byte_difference_count",
    "audit_truth_open_count",
    "audit_truth_read_count",
    "audit_truth_materialized_row_count",
    "generator_quality_result_read_count",
    "candidate_quality_result_read_count",
    "view_builder_quality_result_read_count",
)
STRUCTURE_AUDIT_FIELDS = (
    "version",
    "world_uid",
    "item_count",
    "seller_count",
    "registered_code_count",
    "registered_item_occurrence_count",
    "registered_visible_occurrence_expected_count",
    "registered_visible_occurrence_actual_count",
    "registered_visible_occurrence_multiset_difference_count",
    "literal_code_hits_in_masked",
    "literal_code_hits_in_neutralized",
    "unregistered_code_hits",
    "unregistered_clone_foreign_code_hits",
    "view_keyset_difference_count",
    "neutralized_legal_code_permutation_byte_difference_count",
    "clone_directions",
    "neutral_receipt",
    "full_item_sha256",
    "masked_item_sha256",
    "neutral_item_sha256",
    "full_profile_sha256",
    "masked_profile_sha256",
    "neutral_profile_sha256",
    "forbidden_capability_mounted",
    "audit_truth_open_count",
    "audit_truth_read_count",
    "audit_truth_materialized_row_count",
    "generator_quality_result_read_count",
    "candidate_quality_result_read_count",
    "view_builder_quality_result_read_count",
)
FORBIDDEN_CAPABILITY_FIELDS = (
    "audit_truth",
    "generator_quality_result",
    "candidate_quality_result",
    "view_builder_quality_result",
)
NEUTRAL_RECEIPT_FIELDS = (
    "version",
    "neutral_render_code_ordinal_zero",
    "neutral_code_family_rule",
    "neutral_code_family_count",
    "neutral_code_family_sha256",
    "original_code_value_argument_count",
    "original_code_value_read_count",
    "neutral_metadata_source_value_read_count",
    "neutral_metadata_source_value_read_counts",
    "neutralizer_input_capability",
    "neutralizer_input_fields",
    "neutral_profiles_recomputed_after_code_collapse",
    "neutral_profile_safe_item_sha256",
    "clone_count",
    "title_template_mapping",
    "description_template_mapping",
    "per_item_template_mapping",
    "non_code_projection_commitment",
    "non_code_projection_nodes",
    "neutral_item_sha256",
    "neutral_profile_sha256",
)
NEUTRAL_ITEM_METADATA_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
    "category",
)
NON_CODE_PROJECTION_COMMITMENT_FIELDS = (
    "verified",
    "source_sha256",
    "neutral_sha256",
    "ast_row_count",
    "identity_slot_count",
    "noise_slot_count",
    "absolute_offsets_compared",
    "relative_ast_boundaries_compared",
    "allowed_removed_nodes",
)
ALLOWED_REMOVED_NODES = (
    "registered_code_carrier",
    "removed_literal_tokens",
    "derived_title_modifier",
    "conditional_english_tag_visibility",
)
MAXIMUM_ITEMS_PER_SELLER = 8


class QualityStructureAggregationError(ValueError):
    """Raised when structural evidence is incomplete or malformed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityStructureAggregationError(f"{name} must be a nonnegative integer")
    return value


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityStructureAggregationError(f"{name} must be nonempty text")
    return value


def _required_sha256(value: object, *, name: str) -> str:
    text = _required_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise QualityStructureAggregationError(f"{name} must be lowercase SHA-256")
    return text


def _validate_non_code_projection_commitment(
    value: object, *, item_count: int
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(NON_CODE_PROJECTION_COMMITMENT_FIELDS)
    ):
        raise QualityStructureAggregationError(
            "Non-code projection commitment schema drift"
        )
    source_sha256 = _required_sha256(
        value.get("source_sha256"), name="non-code source SHA-256"
    )
    neutral_sha256 = _required_sha256(
        value.get("neutral_sha256"), name="non-code neutral SHA-256"
    )
    ast_row_count = _nonnegative_int(
        value.get("ast_row_count"), name="non-code AST row count"
    )
    _nonnegative_int(
        value.get("identity_slot_count"), name="non-code identity slot count"
    )
    _nonnegative_int(
        value.get("noise_slot_count"), name="non-code noise slot count"
    )
    if (
        value.get("verified") is not True
        or source_sha256 != neutral_sha256
        or ast_row_count != item_count
        or value.get("absolute_offsets_compared") is not False
        or value.get("relative_ast_boundaries_compared") is not True
        or value.get("allowed_removed_nodes") != list(ALLOWED_REMOVED_NODES)
    ):
        raise QualityStructureAggregationError(
            "Non-code projection commitment closure drift"
        )


def _validate_critical_structure_receipt(
    row: Mapping[str, Any], *, maximum_item_count: int
) -> None:
    """Validate receipt fields that can close without mounting model views."""

    item_count = _nonnegative_int(row.get("item_count"), name="item count")
    if (
        isinstance(maximum_item_count, bool)
        or not isinstance(maximum_item_count, int)
        or maximum_item_count <= 0
        or item_count > maximum_item_count
        or _nonnegative_int(
            row.get("registered_code_count"), name="registered code count"
        )
        != item_count
    ):
        raise QualityStructureAggregationError("Item-count capacity closure drift")
    for field in (
        "full_item_sha256",
        "masked_item_sha256",
        "neutral_item_sha256",
        "full_profile_sha256",
        "masked_profile_sha256",
        "neutral_profile_sha256",
    ):
        _required_sha256(row.get(field), name=field)
    clones = row.get("clone_directions")
    if not isinstance(clones, list) or any(
        not isinstance(value, Mapping)
        or set(value) != {"source_item_uid", "target_item_uid"}
        or not isinstance(value["source_item_uid"], str)
        or not value["source_item_uid"]
        or not isinstance(value["target_item_uid"], str)
        or not value["target_item_uid"]
        or value["source_item_uid"] == value["target_item_uid"]
        for value in clones
    ):
        raise QualityStructureAggregationError("Clone-direction receipt drift")
    clone_pairs = {
        (str(value["source_item_uid"]), str(value["target_item_uid"]))
        for value in clones
    }
    if len(clone_pairs) != len(clones):
        raise QualityStructureAggregationError("Duplicate clone-direction receipt")

    neutral = row.get("neutral_receipt")
    if not isinstance(neutral, Mapping) or set(neutral) != set(NEUTRAL_RECEIPT_FIELDS):
        raise QualityStructureAggregationError("Neutral receipt exact schema drift")
    neutral_codes: list[str] = []
    for ordinal in range(item_count):
        digits: list[str] = []
        value = ordinal
        for _position in range(8):
            digits.append(channel.CODE_ALPHABET[value & 0xF])
            value >>= 4
        neutral_codes.append(
            "Q" + "".join(reversed(digits)) + "BA"
        )
    expected_neutral_family_sha256 = hashlib.sha256(
        _canonical_json_bytes(neutral_codes)
    ).hexdigest()
    if (
        _nonnegative_int(
            neutral.get("neutral_code_family_count"),
            name="neutral code family count",
        )
        != item_count
        or _nonnegative_int(
            neutral.get("original_code_value_argument_count"),
            name="original code argument count",
        )
        != 0
        or neutral.get("neutral_render_code_ordinal_zero") != "QAAAAAAAABA"
        or neutral.get("neutral_code_family_rule")
        != "Q_plus_item_ordinal_as_eight_base16_A_to_P_digits_plus_BA"
        or neutral.get("neutral_code_family_sha256")
        != expected_neutral_family_sha256
        or _nonnegative_int(
            neutral.get("original_code_value_read_count"),
            name="original code read count",
        )
        != 0
        or neutral.get("neutralizer_input_fields")
        != list(NEUTRAL_ITEM_METADATA_FIELDS)
        or neutral.get("neutralizer_input_capability")
        != "NeutralItemProjection[NeutralItemMetadata]"
        or neutral.get("neutral_profiles_recomputed_after_code_collapse") is not True
        or _nonnegative_int(neutral.get("clone_count"), name="neutral clone count")
        != len(clones)
        or neutral.get("neutral_item_sha256") != row.get("neutral_item_sha256")
        or neutral.get("neutral_profile_sha256")
        != row.get("neutral_profile_sha256")
    ):
        raise QualityStructureAggregationError("Neutral receipt critical closure drift")
    expected_reads = item_count * len(NEUTRAL_ITEM_METADATA_FIELDS)
    read_counts = neutral.get("neutral_metadata_source_value_read_counts")
    if (
        _nonnegative_int(
            neutral.get("neutral_metadata_source_value_read_count"),
            name="neutral metadata read count",
        )
        != expected_reads
        or not isinstance(read_counts, Mapping)
        or dict(read_counts)
        != {field: item_count for field in NEUTRAL_ITEM_METADATA_FIELDS}
    ):
        raise QualityStructureAggregationError("Neutral metadata read receipt drift")
    for field in (
        "neutral_code_family_sha256",
        "neutral_profile_safe_item_sha256",
        "neutral_item_sha256",
        "neutral_profile_sha256",
    ):
        _required_sha256(neutral.get(field), name=f"neutral {field}")
    _validate_non_code_projection_commitment(
        neutral.get("non_code_projection_commitment"), item_count=item_count
    )
    if any(
        not isinstance(neutral.get(field), list) or len(neutral[field]) != 8
        for field in ("title_template_mapping", "description_template_mapping")
    ):
        raise QualityStructureAggregationError("Neutral template mapping drift")
    per_item = neutral.get("per_item_template_mapping")
    nodes = neutral.get("non_code_projection_nodes")
    if not isinstance(per_item, list) or not isinstance(nodes, list):
        raise QualityStructureAggregationError("Neutral per-item receipt drift")
    per_item_uids = [
        value.get("item_uid") if isinstance(value, Mapping) else None
        for value in per_item
    ]
    node_uids = [
        value.get("item_uid") if isinstance(value, Mapping) else None
        for value in nodes
    ]
    if (
        len(per_item_uids) != item_count
        or len(node_uids) != item_count
        or any(not isinstance(value, str) or not value for value in per_item_uids)
        or any(not isinstance(value, str) or not value for value in node_uids)
        or len(set(per_item_uids)) != item_count
        or set(per_item_uids) != set(node_uids)
    ):
        raise QualityStructureAggregationError("Neutral per-item UID closure drift")


def _aggregate(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_world_counts: Mapping[str, int],
    expected_sellers_per_world: int,
    maximum_position_deviation: float,
    enforce_position_margin: bool,
    claim_boundary: str,
) -> dict[str, Any]:
    if set(public_rows_by_split) != set(SPLITS) or set(structure_rows_by_split) != set(SPLITS):
        raise QualityStructureAggregationError("Structural split universe drift")
    if set(expected_world_counts) != set(SPLITS):
        raise QualityStructureAggregationError("Expected split universe drift")
    if (
        isinstance(expected_sellers_per_world, bool)
        or not isinstance(expected_sellers_per_world, int)
        or expected_sellers_per_world <= 0
        or not math.isfinite(maximum_position_deviation)
        or maximum_position_deviation < 0.0
    ):
        raise QualityStructureAggregationError("Structural scalar contract drift")

    global_owner: dict[str, tuple[str, str]] = {}
    prior_world_code_hits = 0
    aggregate_zero_counts: Counter[str] = Counter()
    split_receipts: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for split in SPLITS:
        expected_world_count = _nonnegative_int(
            expected_world_counts[split], name=f"{split} world count"
        )
        public_rows = tuple(public_rows_by_split[split])
        structure_rows = tuple(structure_rows_by_split[split])
        public_by_world: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in public_rows:
            if not isinstance(row, Mapping):
                raise QualityStructureAggregationError("Public structural row type drift")
            world_uid = _required_text(row.get("world_uid"), name="public world UID")
            _required_text(row.get("seller_uid"), name="public seller UID")
            owned_codes = row.get("owned_codes")
            if not isinstance(owned_codes, list) or not owned_codes:
                raise QualityStructureAggregationError("Owned code inventory drift")
            public_by_world[world_uid].append(row)
        audits_by_world: dict[str, Mapping[str, Any]] = {}
        for row in structure_rows:
            if not isinstance(row, Mapping):
                raise QualityStructureAggregationError("Structure audit row type drift")
            if set(row) != set(STRUCTURE_AUDIT_FIELDS):
                raise QualityStructureAggregationError(
                    "Structure audit exact schema drift"
                )
            _validate_critical_structure_receipt(
                row,
                maximum_item_count=(
                    expected_sellers_per_world * MAXIMUM_ITEMS_PER_SELLER
                ),
            )
            capability_mounted = row["forbidden_capability_mounted"]
            if capability_mounted != {
                name: False for name in FORBIDDEN_CAPABILITY_FIELDS
            }:
                raise QualityStructureAggregationError(
                    "Forbidden structure capability was mounted"
                )
            world_uid = _required_text(row.get("world_uid"), name="audit world UID")
            if world_uid in audits_by_world:
                raise QualityStructureAggregationError("Duplicate structure audit world")
            audits_by_world[world_uid] = row
        if (
            len(public_by_world) != expected_world_count
            or len(audits_by_world) != expected_world_count
            or set(public_by_world) != set(audits_by_world)
        ):
            raise QualityStructureAggregationError("Structural world universe drift")

        position_counts = [Counter() for _ in range(10)]
        registered_code_count = 0
        for world_uid in sorted(public_by_world, key=lambda value: value.encode("utf-8")):
            rows = public_by_world[world_uid]
            seller_uids = [_required_text(row["seller_uid"], name="seller UID") for row in rows]
            if (
                len(rows) != expected_sellers_per_world
                or len(seller_uids) != len(set(seller_uids))
            ):
                raise QualityStructureAggregationError("Per-world seller universe drift")
            world_codes: set[str] = set()
            for row in rows:
                seller_uid = str(row["seller_uid"])
                owned_codes = tuple(row["owned_codes"])
                if len(owned_codes) != len(set(owned_codes)):
                    raise QualityStructureAggregationError("Seller owned-code duplicate")
                for value in owned_codes:
                    code = _required_text(value, name="owned code")
                    if channel.RAW_CODE_RE.fullmatch(code) is None:
                        raise QualityStructureAggregationError("Owned code syntax drift")
                    if code in world_codes:
                        raise QualityStructureAggregationError("Same-world code ownership collision")
                    world_codes.add(code)
                    owner = global_owner.get(code)
                    if owner is not None and owner[0] != world_uid:
                        prior_world_code_hits += 1
                    else:
                        global_owner[code] = (world_uid, seller_uid)
                    for position, symbol in enumerate(code[1:]):
                        position_counts[position][symbol] += 1
            audit = audits_by_world[world_uid]
            if (
                _nonnegative_int(audit.get("seller_count"), name="audit seller count")
                != expected_sellers_per_world
                or _nonnegative_int(
                    audit.get("registered_code_count"), name="registered code count"
                )
                != len(world_codes)
                or _nonnegative_int(
                    audit.get("registered_visible_occurrence_expected_count"),
                    name="expected visible occurrence count",
                )
                != _nonnegative_int(
                    audit.get("registered_visible_occurrence_actual_count"),
                    name="actual visible occurrence count",
                )
            ):
                raise QualityStructureAggregationError("Per-world structure closure drift")
            for field in ZERO_TOLERANCE_FIELDS:
                aggregate_zero_counts[field] += _nonnegative_int(
                    audit.get(field), name=field
                )
            registered_code_count += len(world_codes)

        maximum_deviation = 0.0
        if registered_code_count:
            for counter in position_counts:
                if sum(counter.values()) != registered_code_count:
                    raise QualityStructureAggregationError("Code position count drift")
                maximum_deviation = max(
                    maximum_deviation,
                    *(
                        abs(counter[symbol] / registered_code_count - 1.0 / 16.0)
                        for symbol in channel.CODE_ALPHABET
                    ),
                )
        if enforce_position_margin and split in {"train", "development"}:
            if maximum_deviation > maximum_position_deviation:
                failures.append(f"{split}:code_character_position_margin")
        split_receipts[split] = {
            "world_count": expected_world_count,
            "seller_row_count": len(public_rows),
            "registered_code_count": registered_code_count,
            "code_character_position_maximum_absolute_deviation": maximum_deviation,
        }

    aggregate_zero_counts["prior_world_code_hits"] = prior_world_code_hits
    for field, value in sorted(aggregate_zero_counts.items()):
        if value != 0:
            failures.append(field)
    is_fixture = claim_boundary == "FIXTURE_ONLY_NO_DATASET_CONCLUSION"
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": (
            (
                "FIXTURE_STRUCTURE_PASS_NO_DATASET_CONCLUSION"
                if not failures
                else "FIXTURE_STRUCTURE_GATE_TRIGGERED_NO_DATASET_CONCLUSION"
            )
            if is_fixture
            else ("PASS" if not failures else "DATASET_INVALIDATED")
        ),
        "claim_boundary": claim_boundary,
        "split_receipts": split_receipts,
        "zero_tolerance_counts": dict(sorted(aggregate_zero_counts.items())),
        "gate_failures": sorted(set(failures), key=lambda value: value.encode("utf-8")),
        "truth_label_row_count_read": aggregate_zero_counts[
            "audit_truth_materialized_row_count"
        ],
        "audit_truth_open_count": aggregate_zero_counts["audit_truth_open_count"],
        "audit_truth_read_count": aggregate_zero_counts["audit_truth_read_count"],
        "audit_truth_materialized_row_count": aggregate_zero_counts[
            "audit_truth_materialized_row_count"
        ],
        "forbidden_read_counts": {
            "audit_truth": {
                "open_count": aggregate_zero_counts["audit_truth_open_count"],
                "read_count": aggregate_zero_counts["audit_truth_read_count"],
                "materialized_row_count": aggregate_zero_counts[
                    "audit_truth_materialized_row_count"
                ],
            },
            "generator_quality_result": aggregate_zero_counts[
                "generator_quality_result_read_count"
            ],
            "candidate_quality_result": aggregate_zero_counts[
                "candidate_quality_result_read_count"
            ],
            "view_builder_quality_result": aggregate_zero_counts[
                "view_builder_quality_result_read_count"
            ],
        },
    }
    receipt["canonical_self_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def aggregate_fixture_structure(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_world_counts: Mapping[str, int],
    expected_sellers_per_world: int,
) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3
        for value in expected_world_counts.values()
    ):
        raise QualityStructureAggregationError("Fixture world boundary widened")
    return _aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=structure_rows_by_split,
        expected_world_counts=expected_world_counts,
        expected_sellers_per_world=expected_sellers_per_world,
        maximum_position_deviation=1.0,
        enforce_position_margin=False,
        claim_boundary="FIXTURE_ONLY_NO_DATASET_CONCLUSION",
    )


def aggregate_formal_structure(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    channel_policy.validate_policy(policy)
    if policy["authorization"]["quality_audit_run"] is not True:
        raise QualityStructureAggregationError("Formal quality audit remains unauthorized")
    return _aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=structure_rows_by_split,
        expected_world_counts=policy["design_scale"]["world_counts"],
        expected_sellers_per_world=policy["design_scale"]["seller_count_per_world"],
        maximum_position_deviation=policy["quality_gates"][
            "code_character_position_maximum_absolute_deviation_from_one_sixteenth"
        ],
        enforce_position_margin=True,
        claim_boundary="V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
    )
