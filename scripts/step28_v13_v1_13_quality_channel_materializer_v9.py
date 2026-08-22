#!/usr/bin/env python3
"""Materialize V9.1 full/masked/neutral model views before private pair truth.

The public entry accepts explicit label-free projections.  It has no pair,
controller, qrels, positive-target, negative-flag, or quality-result argument.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import inspect
from itertools import zip_longest
import json
import math
from typing import Any

import step28_v13_common as common
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_v1_13_pure_natural_renderer_v9 as pure_renderer
import step28_v13_v1_13_quality_channel_views_v9 as channel
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_world_builder as world_builder


VERSION = "2026-08-21-step28-v13-v1-13-quality-channel-materializer-v9-1"
NEUTRAL_RENDER_CODE = "QAAAAAAAABA"
NEUTRAL_CODE_ORDINAL_DIGITS = 8
NEUTRAL_CODE_DERIVED_SUFFIX = "BA"
MODEL_PROFILE_TEXT_FIELDS = scientific.MODEL_PROFILE_TEXT_FIELDS
NEUTRAL_ITEM_METADATA_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
    "category",
)
PUBLIC_ITEM_FIELDS = (*NEUTRAL_ITEM_METADATA_FIELDS, "title", "description")
FORBIDDEN_ACCESS_CAPABILITY_FIELDS = (
    "audit_truth",
    "generator_quality_result",
    "candidate_quality_result",
    "view_builder_quality_result",
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


class QualityChannelMaterializationError(common.ContractError):
    """Raised when a label-free model view cannot close exactly."""


@dataclass(frozen=True)
class MaterializedChannelViews:
    masked_redacted_items: tuple[dict[str, Any], ...]
    neutral_redacted_items: tuple[dict[str, Any], ...]
    masked_seller_profiles: tuple[dict[str, Any], ...]
    neutral_seller_profiles: tuple[dict[str, Any], ...]
    public_code_probe_input: tuple[dict[str, Any], ...]
    text_probe_eligibility_input: tuple[dict[str, Any], ...]
    channel_structure_audit: dict[str, Any]


@dataclass(frozen=True)
class NeutralItemMetadata:
    """Code-free capability passed into the neutral renderer."""

    world_uid: str
    seller_uid: str
    item_uid: str
    time_bucket: int
    category: str


@dataclass(frozen=True)
class NeutralItemProjection:
    """Runtime-audited, code-free capability for the neutral renderer."""

    rows: tuple[NeutralItemMetadata, ...]
    source_value_read_counts: tuple[tuple[str, int], ...]
    source_value_read_count: int
    forbidden_value_read_count: int


@dataclass(frozen=True)
class ForbiddenAccessObservation:
    """Runtime wiring receipt for capabilities absent from the view builder."""

    capability_mounted: tuple[tuple[str, bool], ...]
    read_counts: tuple[tuple[str, int], ...]


def _forbidden_access_observation(
    *,
    entrypoint_parameter_names: Sequence[str],
    accessed_capability_names: Sequence[str],
) -> ForbiddenAccessObservation:
    """Compute mount/read counters from the actual entrypoint wiring ledger."""

    parameter_names = tuple(entrypoint_parameter_names)
    accessed = tuple(accessed_capability_names)
    if len(parameter_names) != len(set(parameter_names)):
        raise QualityChannelMaterializationError(
            "View-builder entrypoint parameter ledger is duplicated"
        )
    unknown_accesses = set(accessed) - set(FORBIDDEN_ACCESS_CAPABILITY_FIELDS)
    read_counter = Counter(accessed)
    capability_mounted = tuple(
        (name, name in parameter_names)
        for name in FORBIDDEN_ACCESS_CAPABILITY_FIELDS
    )
    if unknown_accesses or any(value for _name, value in capability_mounted):
        raise QualityChannelMaterializationError(
            "A forbidden quality/truth capability reached the view-builder wiring"
        )
    return ForbiddenAccessObservation(
        capability_mounted=capability_mounted,
        read_counts=tuple(
            (name, int(read_counter[name]))
            for name in FORBIDDEN_ACCESS_CAPABILITY_FIELDS
        ),
    )


def _byte_difference_count(left: bytes, right: bytes) -> int:
    return sum(
        left_value != right_value
        for left_value, right_value in zip_longest(left, right, fillvalue=-1)
    )


class _NeutralMetadataReadGuard(Mapping[str, Any]):
    def __init__(
        self,
        source: Mapping[str, Any],
        *,
        value_read_counts: dict[str, int],
        forbidden_read_count: list[int],
    ) -> None:
        self._source = source
        self._value_read_counts = value_read_counts
        self._forbidden_read_count = forbidden_read_count

    def __getitem__(self, key: str) -> Any:
        if key not in NEUTRAL_ITEM_METADATA_FIELDS:
            self._forbidden_read_count[0] += 1
            raise QualityChannelMaterializationError(
                "Neutral metadata projection attempted a forbidden value read"
            )
        self._value_read_counts[key] += 1
        return self._source[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


def _clone(value: Any) -> Any:
    return json.loads(common.canonical_json_bytes(value).decode("utf-8"))


def _sorted_rows(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        _clone(row)
        for row in sorted(rows, key=lambda row: str(row[key]).encode("utf-8"))
    )


def _row_index(
    rows: Sequence[Mapping[str, Any]], *, key: str, name: str
) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): _clone(row) for row in rows}
    if len(output) != len(rows) or any(not value for value in output):
        raise QualityChannelMaterializationError(f"{name} keyset is not unique")
    return output


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _project_neutral_item_metadata(
    public_items: Sequence[Mapping[str, Any]],
) -> NeutralItemProjection:
    """Project metadata without ever reading title or description values."""

    projected: list[NeutralItemMetadata] = []
    seen: set[str] = set()
    value_read_counts = {field: 0 for field in NEUTRAL_ITEM_METADATA_FIELDS}
    forbidden_read_count = [0]
    for source in public_items:
        if set(source) != set(PUBLIC_ITEM_FIELDS):
            raise QualityChannelMaterializationError(
                "Public item schema is not the frozen seven-field schema"
            )
        row = _NeutralMetadataReadGuard(
            source,
            value_read_counts=value_read_counts,
            forbidden_read_count=forbidden_read_count,
        )
        item_uid = str(row["item_uid"])
        time_bucket = row["time_bucket"]
        if (
            not item_uid
            or item_uid in seen
            or isinstance(time_bucket, bool)
            or not isinstance(time_bucket, int)
            or not 0 <= time_bucket <= 3
        ):
            raise QualityChannelMaterializationError(
                "Neutral item metadata key or time bucket drift"
            )
        seen.add(item_uid)
        projected.append(
            NeutralItemMetadata(
                world_uid=str(row["world_uid"]),
                seller_uid=str(row["seller_uid"]),
                item_uid=item_uid,
                time_bucket=time_bucket,
                category=str(row["category"]),
            )
        )
    rows = tuple(sorted(projected, key=lambda row: row.item_uid.encode("utf-8")))
    expected_read_count = len(rows) * len(NEUTRAL_ITEM_METADATA_FIELDS)
    if (
        forbidden_read_count[0] != 0
        or sum(value_read_counts.values()) != expected_read_count
        or any(value != len(rows) for value in value_read_counts.values())
    ):
        raise QualityChannelMaterializationError(
            "Neutral metadata runtime read audit did not close exactly"
        )
    return NeutralItemProjection(
        rows=rows,
        source_value_read_counts=tuple(sorted(value_read_counts.items())),
        source_value_read_count=expected_read_count,
        forbidden_value_read_count=forbidden_read_count[0],
    )


def _neutral_render_code(item_ordinal: int) -> str:
    if (
        isinstance(item_ordinal, bool)
        or not isinstance(item_ordinal, int)
        or not 0 <= item_ordinal < 16**NEUTRAL_CODE_ORDINAL_DIGITS
    ):
        raise QualityChannelMaterializationError(
            "Neutral item ordinal is outside the frozen code-family domain"
        )
    digits = []
    value = item_ordinal
    for _position in range(NEUTRAL_CODE_ORDINAL_DIGITS):
        digits.append(channel.CODE_ALPHABET[value & 0xF])
        value >>= 4
    code = "Q" + "".join(reversed(digits)) + NEUTRAL_CODE_DERIVED_SUFFIX
    if channel.CODE_RE.fullmatch(code) is None:
        raise QualityChannelMaterializationError("Neutral code family is malformed")
    return code


def _inverse_capacity_map(
    skeletons: Sequence[str], *, description: bool
) -> tuple[dict[int, int], tuple[dict[str, Any], ...]]:
    forward = pure_renderer._capacity_index_map(skeletons, description=description)
    inverse = {target: source for source, target in forward.items()}
    if len(inverse) != 8:
        raise QualityChannelMaterializationError("Capacity skeleton inverse is incomplete")
    rows = tuple(
        {
            "source_template_id": source,
            "neutral_template_id": source,
            "production_template_id": target,
            "is_code_carrier": "{code}" in str(skeletons[target]),
            "carrier_ast_node_id": (
                f"{'description' if description else 'title'}.template.{target}.code"
            ),
            "removed_literal_tokens": (
                []
                if source == target
                else [
                    pure_renderer.DESCRIPTION_CODE_TWIN_INSERTION
                    if description
                    else pure_renderer.TITLE_CODE_TWIN_SUFFIX
                ]
            ),
            "retained_non_code_placeholders": sorted(
                name
                for name in (
                    "product",
                    "attribute",
                    "title_modifier",
                    "delivery",
                    "service",
                    "noise_clause",
                    "context_guard",
                    "identity_clause",
                )
                if "{" + name + "}" in str(skeletons[source])
            ),
            "neutral_carrier_ast": str(skeletons[source]),
        }
        for source, target in sorted(forward.items())
    )
    return inverse, rows


def _neutral_template_index(
    *, current: int, skeletons: Sequence[str], inverse: Mapping[int, int]
) -> int:
    if isinstance(current, bool) or not isinstance(current, int):
        raise QualityChannelMaterializationError("Skeleton index is not an integer")
    if current in inverse:
        return int(inverse[current])
    if 0 <= current < 8 and "{code}" not in str(skeletons[current]):
        return current
    raise QualityChannelMaterializationError("Skeleton lacks a unique neutral base")


def _literal_positions(text: str, literal: str) -> tuple[int, ...]:
    positions: list[int] = []
    cursor = 0
    while True:
        start = text.find(literal, cursor)
        if start < 0:
            return tuple(positions)
        positions.append(start)
        cursor = start + len(literal)


def _render_registered_carrier(
    *,
    field: str,
    ast: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    effective_styles: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, int]:
    """Render the carrier from its AST; do not infer its span from final text."""

    seller_uid = str(ast["seller_uid"])
    style = effective_styles.get(seller_uid)
    if style is None:
        raise QualityChannelMaterializationError(
            "Registered carrier style projection is incomplete"
        )
    code = channel._validate_code(ast["code"])
    index_key = f"{field}_skeleton_index"
    skeleton_key = f"{field}_skeletons"
    nonempty_key = f"{field}_nonempty"
    try:
        skeleton_index = int(ast[index_key])
        skeleton = str(safe_library[skeleton_key][skeleton_index])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise QualityChannelMaterializationError(
            "Registered carrier template lookup failed"
        ) from exc
    if not bool(ast[nonempty_key]):
        return "", code, skeleton_index
    if field == "title":
        rendered = pure_renderer._render_base_title(
            skeleton=skeleton,
            product=str(ast["product"]),
            attribute=str(ast["attribute"]),
            code=code,
            style=style,
            library=safe_library,
        )
    elif field == "description":
        rendered = pure_renderer._render_base_description(
            skeleton=skeleton,
            product=str(ast["product"]),
            attribute=str(ast["attribute"]),
            code=code,
            delivery=str(ast["delivery"]),
            service=str(ast["service"]),
            style=style,
            library=safe_library,
        )
    else:
        raise QualityChannelMaterializationError(
            "Registered carrier field is not title or description"
        )
    rendered = production.source.normalize_redacted_text(rendered)
    if len(_literal_positions(rendered, code)) != skeleton.count("{code}"):
        raise QualityChannelMaterializationError(
            "AST-rendered carrier code count does not match its template"
        )
    return rendered, code, skeleton_index


def _registered_item_spans(
    *,
    row: Mapping[str, Any],
    ast: Mapping[str, Any],
    ast_by_item: Mapping[str, Mapping[str, Any]],
    clone_source_by_target: Mapping[str, str],
    safe_library: Mapping[str, Any],
    all_codes: set[str],
    effective_styles: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    item_uid = str(row["item_uid"])
    own_code = str(ast["code"])
    if own_code not in all_codes:
        raise QualityChannelMaterializationError("Item own code is not registered")
    output: dict[str, str] = {}
    receipt_rows: list[dict[str, Any]] = []
    for field in ("title", "description"):
        text = str(row[field])
        node_owner = item_uid
        skeleton_ast = ast
        if field == "title" and item_uid in clone_source_by_target:
            node_owner = clone_source_by_target[item_uid]
            skeleton_ast = ast_by_item[node_owner]
        rendered_base, visible_code, skeleton_index = _render_registered_carrier(
            field=field,
            ast=skeleton_ast,
            safe_library=safe_library,
            effective_styles=effective_styles,
        )
        positions = _literal_positions(rendered_base, visible_code)
        last_carrier_end = (
            positions[-1] + len(visible_code) if positions else 0
        )
        carrier_prefix_matches = (
            text[:last_carrier_end] == rendered_base[:last_carrier_end]
        )
        if (field == "title" and text != rendered_base) or not carrier_prefix_matches:
            common_prefix_length = 0
            for visible_character, base_character in zip(text, rendered_base):
                if visible_character != base_character:
                    break
                common_prefix_length += 1
            raise QualityChannelMaterializationError(
                "Visible carrier bytes disagree with independent AST rendering: "
                f"item_uid={item_uid} field={field} node_owner={node_owner} "
                f"visible_length={len(text)} base_length={len(rendered_base)} "
                f"common_prefix_length={common_prefix_length} "
                f"carrier_positions={positions} "
                f"visible_sha256={_sha256_text(text)} "
                f"base_sha256={_sha256_text(rendered_base)}"
            )
        spans = tuple(
            channel.RegisteredCodeSpan(
                start,
                start + len(visible_code),
                visible_code,
                f"{node_owner}.{field}.template.{skeleton_index}.code.{ordinal}",
            )
            for ordinal, start in enumerate(positions)
        )
        masked, _counts = channel.mask_registered_code_spans(
            text, registered_codes=all_codes, registered_spans=spans
        )
        output[field] = masked
        receipt_rows.extend(
            {
                "item_uid": item_uid,
                "field": field,
                "code": span.code,
                "is_own": span.code == own_code,
            }
            for span in spans
        )
    return output, tuple(receipt_rows)


def _build_profiles(
    *,
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    profile_safe = production.build_profile_safe_items(
        profile_policy, items=public_items, redacted_items=redacted_items
    )
    profiles, audit = profiles_module.build_world_profiles(
        profile_policy,
        mode=mode,
        split=split,
        sellers=public_sellers,
        items=profile_safe,
    )
    if audit.get("labels_or_private_structure_read") is not False or audit.get(
        "seller_count"
    ) != 28:
        raise QualityChannelMaterializationError("Seller-profile rebuild did not close")
    return _sorted_rows(profiles, "seller_uid")


def _persisted_profile_rows(
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the exact rows later written and mounted as a model view."""

    if MODEL_PROFILE_TEXT_FIELDS != channel.PROFILE_FIELDS:
        raise QualityChannelMaterializationError(
            "Model seller-profile text projection drift"
        )
    try:
        return scientific.project_model_seller_profiles(profiles)
    except scientific.ScientificBuilderError as exc:
        raise QualityChannelMaterializationError(str(exc)) from exc


def _persisted_profile_sha256(
    profiles: Sequence[Mapping[str, Any]],
) -> str:
    return common.canonical_sha256(_persisted_profile_rows(profiles))


def _replace_neutral_code(
    text: str, *, node_prefix: str, registered_neutral_codes: Sequence[str]
) -> str:
    inventory = tuple(registered_neutral_codes)
    if not inventory or len(inventory) != len(set(inventory)):
        raise QualityChannelMaterializationError(
            "Neutral code family inventory is empty or duplicated"
        )
    allowed = set(inventory)
    matches = list(channel.RAW_CODE_RE.finditer(text))
    if any(match.group(0) not in allowed for match in matches):
        raise QualityChannelMaterializationError("Neutral view contains a non-neutral code")
    spans = tuple(
        channel.RegisteredCodeSpan(
            match.start(),
            match.end(),
            match.group(0),
            f"{node_prefix}.{ordinal}",
        )
        for ordinal, match in enumerate(matches)
    )
    if not spans:
        return text
    masked, _counts = channel.mask_registered_code_spans(
        text, registered_codes=inventory, registered_spans=spans
    )
    return masked


def _normalized_non_code_ast_rows(
    render_asts: Sequence[Mapping[str, Any]],
    *,
    safe_library: Mapping[str, Any],
    title_inverse: Mapping[int, int],
    description_inverse: Mapping[int, int],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in render_asts:
        item_uid = str(source["item_uid"])
        if not item_uid or item_uid in seen:
            raise QualityChannelMaterializationError(
                "Non-code AST commitment item key drift"
            )
        seen.add(item_uid)
        row = {
            key: _clone(value)
            for key, value in source.items()
            if key not in {"code", "title_skeleton_index", "description_skeleton_index"}
        }
        row["title_skeleton_index"] = _neutral_template_index(
            current=int(source["title_skeleton_index"]),
            skeletons=safe_library["title_skeletons"],
            inverse=title_inverse,
        )
        row["description_skeleton_index"] = _neutral_template_index(
            current=int(source["description_skeleton_index"]),
            skeletons=safe_library["description_skeletons"],
            inverse=description_inverse,
        )
        rows.append(row)
    return _sorted_rows(rows, "item_uid")


def _identity_projection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["item_uid"])].append(row)
    output: list[dict[str, Any]] = []
    fields = (
        "slot_uid",
        "bundle_uid",
        "item_uid",
        "seller_uid",
        "field_name",
        "identity_uid",
        "identity_type",
        "downstream_canonical_value",
        "raw_surface",
        "parser_expectation",
        "expected_seller_facing_context",
        "expected_product_data_risk_context",
        "expected_direct_identity_eligible",
        "expected_support_only",
        "planned_role",
        "time_bucket",
    )
    for item_uid in sorted(groups, key=lambda value: value.encode("utf-8")):
        ordered = sorted(
            groups[item_uid],
            key=lambda row: (int(row["start"]), int(row["end"]), str(row["slot_uid"])),
        )
        for ordinal, source in enumerate(ordered):
            row = {field: _clone(source[field]) for field in fields}
            row["relative_slot_ordinal"] = ordinal
            row["previous_ast_node_kind"] = (
                "base_description" if ordinal == 0 else "context_guard"
            )
            row["next_ast_node_kind"] = "context_guard"
            output.append(row)
    return tuple(output)


def _noise_projection_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    projected = [
        {
            "noise_slot_uid": str(row["noise_slot_uid"]),
            "item_uid": str(row["item_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "field_name": str(row["field_name"]),
            "raw_surface": str(row["raw_surface"]),
            "parser_expectation": str(row["parser_expectation"]),
            "previous_ast_node_kind": "base_description",
            "next_ast_node_kind": "context_guard_or_end",
        }
        for row in rows
    ]
    return tuple(
        sorted(
            projected,
            key=lambda row: (
                row["item_uid"].encode("utf-8"),
                row["noise_slot_uid"].encode("utf-8"),
            ),
        )
    )


def _non_code_projection_commitment(
    *,
    source_asts_without_codes: Sequence[Mapping[str, Any]],
    neutral_asts: Sequence[Mapping[str, Any]],
    source_identity_slots: Sequence[Mapping[str, Any]],
    neutral_identity_slots: Sequence[Mapping[str, Any]],
    source_noise_slots: Sequence[Mapping[str, Any]],
    neutral_noise_slots: Sequence[Mapping[str, Any]],
    safe_library: Mapping[str, Any],
    title_inverse: Mapping[int, int],
    description_inverse: Mapping[int, int],
    effective_styles: Mapping[str, Mapping[str, Any]],
    visible_title_source_by_item: Mapping[str, str],
) -> dict[str, Any]:
    common_payload = {
        "effective_styles": {
            seller_uid: _clone(effective_styles[seller_uid])
            for seller_uid in sorted(
                effective_styles, key=lambda value: value.encode("utf-8")
            )
        },
        "visible_title_source_by_item": {
            item_uid: str(visible_title_source_by_item[item_uid])
            for item_uid in sorted(
                visible_title_source_by_item,
                key=lambda value: value.encode("utf-8"),
            )
        },
    }
    source_payload = {
        **common_payload,
        "render_asts": _normalized_non_code_ast_rows(
            source_asts_without_codes,
            safe_library=safe_library,
            title_inverse=title_inverse,
            description_inverse=description_inverse,
        ),
        "identity_slots": _identity_projection_rows(source_identity_slots),
        "noise_slots": _noise_projection_rows(source_noise_slots),
    }
    neutral_payload = {
        **common_payload,
        "render_asts": _normalized_non_code_ast_rows(
            neutral_asts,
            safe_library=safe_library,
            title_inverse=title_inverse,
            description_inverse=description_inverse,
        ),
        "identity_slots": _identity_projection_rows(neutral_identity_slots),
        "noise_slots": _noise_projection_rows(neutral_noise_slots),
    }
    source_hash = common.canonical_sha256(source_payload)
    neutral_hash = common.canonical_sha256(neutral_payload)
    if (
        common.canonical_json_bytes(source_payload)
        != common.canonical_json_bytes(neutral_payload)
    ):
        raise QualityChannelMaterializationError(
            "Non-code AST, slot, style, or clone projection changed during neutralization"
        )
    return {
        "verified": True,
        "source_sha256": source_hash,
        "neutral_sha256": neutral_hash,
        "ast_row_count": len(source_payload["render_asts"]),
        "identity_slot_count": len(source_payload["identity_slots"]),
        "noise_slot_count": len(source_payload["noise_slots"]),
        "absolute_offsets_compared": False,
        "relative_ast_boundaries_compared": True,
        "allowed_removed_nodes": [
            "registered_code_carrier",
            "removed_literal_tokens",
            "derived_title_modifier",
            "conditional_english_tag_visibility",
        ],
    }


def _neutralize_without_original_code_values(
    *,
    processing_policy: Mapping[str, Any],
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    processing_template: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    fixture: Mapping[str, Any],
    world_uid: str,
    public_sellers: Sequence[Mapping[str, Any]],
    public_item_projection: NeutralItemProjection,
    render_asts_without_codes: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
    effective_styles: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Render neutral views; the signature deliberately has no original code input."""

    if channel.CODE_RE.fullmatch(NEUTRAL_RENDER_CODE) is None:
        raise QualityChannelMaterializationError("Frozen neutral code is malformed")
    if not isinstance(public_item_projection, NeutralItemProjection):
        raise QualityChannelMaterializationError(
            "Neutral renderer did not receive the audited projection capability"
        )
    if public_item_projection.forbidden_value_read_count != 0:
        raise QualityChannelMaterializationError(
            "Neutral metadata projection recorded a forbidden value read"
        )
    public_by_item: dict[str, dict[str, Any]] = {}
    for metadata in public_item_projection.rows:
        if not isinstance(metadata, NeutralItemMetadata):
            raise QualityChannelMaterializationError(
                "Neutral renderer received a non-capability item input"
            )
        row = {
            "world_uid": metadata.world_uid,
            "seller_uid": metadata.seller_uid,
            "item_uid": metadata.item_uid,
            "time_bucket": metadata.time_bucket,
            "category": metadata.category,
            "title": "",
            "description": "",
        }
        if metadata.item_uid in public_by_item or set(row) != set(PUBLIC_ITEM_FIELDS):
            raise QualityChannelMaterializationError(
                "Neutral item capability keyset is duplicated or malformed"
            )
        public_by_item[metadata.item_uid] = row
    ast_by_item = _row_index(
        render_asts_without_codes, key="item_uid", name="code-free render AST"
    )
    if set(public_by_item) != set(ast_by_item):
        raise QualityChannelMaterializationError("Neutral item/AST universe drift")
    title_inverse, title_mapping = _inverse_capacity_map(
        safe_library["title_skeletons"], description=False
    )
    description_inverse, description_mapping = _inverse_capacity_map(
        safe_library["description_skeletons"], description=True
    )
    identity_by_item: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_slots_audit:
        identity_by_item[str(row["item_uid"])].append(_clone(row))
    noise_by_item = {
        str(row["item_uid"]): _clone(row) for row in noise_slots_audit
    }
    if len(noise_by_item) != len(noise_slots_audit):
        raise QualityChannelMaterializationError("Neutral noise projection is duplicated")
    role_to_family = profile_policy["identity_design"]["role_to_template_family"]
    items_by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    neutral_asts: list[dict[str, Any]] = []
    non_code_projection_nodes: list[dict[str, Any]] = []
    per_item_template_mapping: list[dict[str, Any]] = []
    visible_title_source_by_item = {
        item_uid: item_uid for item_uid in public_by_item
    }
    neutral_code_by_item: dict[str, str] = {}
    for item_ordinal, item_uid in enumerate(
        sorted(public_by_item, key=lambda value: value.encode("utf-8"))
    ):
        public = public_by_item[item_uid]
        ast = ast_by_item[item_uid]
        seller_uid = str(public["seller_uid"])
        if seller_uid not in effective_styles:
            raise QualityChannelMaterializationError("Neutral style projection is incomplete")
        style = effective_styles[seller_uid]
        neutral_code = _neutral_render_code(item_ordinal)
        neutral_code_by_item[item_uid] = neutral_code
        title_index = _neutral_template_index(
            current=int(ast["title_skeleton_index"]),
            skeletons=safe_library["title_skeletons"],
            inverse=title_inverse,
        )
        description_index = _neutral_template_index(
            current=int(ast["description_skeleton_index"]),
            skeletons=safe_library["description_skeletons"],
            inverse=description_inverse,
        )
        title_modifier = pure_renderer._title_modifier(
            neutral_code, safe_library
        )
        if title_modifier != "常规款":
            raise QualityChannelMaterializationError(
                "Frozen neutral code no longer selects the neutral title modifier"
            )
        title = (
            pure_renderer._render_base_title(
                skeleton=str(safe_library["title_skeletons"][title_index]),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                code=neutral_code,
                style=style,
                library=safe_library,
            )
            if bool(ast["title_nonempty"])
            else ""
        )
        base_description = (
            pure_renderer._render_base_description(
                skeleton=str(safe_library["description_skeletons"][description_index]),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                code=neutral_code,
                delivery=str(ast["delivery"]),
                service=str(ast["service"]),
                style=style,
                library=safe_library,
            )
            if bool(ast["description_nonempty"])
            else ""
        )
        english_tag = str(style["english_tag"])
        if bool(ast["title_nonempty"]) and english_tag and not title.endswith(
            " " + english_tag
        ):
            raise QualityChannelMaterializationError(
                "Neutral native title did not preserve the English-tag value"
            )
        non_code_projection_nodes.append(
            {
                "item_uid": item_uid,
                "derived_title_modifier_node_id": (
                    f"item.{item_uid}.title.derived_title_modifier"
                ),
                "derived_title_modifier_value": title_modifier,
                "conditional_english_tag_visibility_node_id": (
                    f"item.{item_uid}.title.conditional_english_tag_visibility"
                ),
                "english_tag_value_node_id": (
                    f"seller.{seller_uid}.style.english_tag_value"
                ),
                "english_tag_value_sha256": _sha256_text(english_tag),
                "english_tag_nonempty": bool(english_tag),
                "neutral_native_visibility": bool(
                    ast["title_nonempty"] and english_tag
                ),
            }
        )
        per_item_template_mapping.append(
            {
                "item_uid": item_uid,
                "production_title_template_id": int(ast["title_skeleton_index"]),
                "neutral_title_template_id": title_index,
                "neutral_title_is_code_carrier": (
                    "{code}" in str(safe_library["title_skeletons"][title_index])
                ),
                "production_description_template_id": int(
                    ast["description_skeleton_index"]
                ),
                "neutral_description_template_id": description_index,
                "neutral_description_is_code_carrier": (
                    "{code}"
                    in str(safe_library["description_skeletons"][description_index])
                ),
            }
        )
        slots = []
        for source in sorted(
            identity_by_item[item_uid], key=lambda row: str(row["slot_uid"]).encode("utf-8")
        ):
            if str(source["planned_role"]) not in role_to_family:
                raise QualityChannelMaterializationError("Identity role projection drift")
            slots.append(
                {
                    "slot_uid": str(source["slot_uid"]),
                    "bundle_uid": str(source["bundle_uid"]),
                    "identity_uid": str(source["identity_uid"]),
                    "role": str(source["planned_role"]),
                    "identity_type": str(source["identity_type"]),
                    "identity_value": str(source["raw_surface"]),
                }
            )
        noise = noise_by_item.get(item_uid)
        item_state = {
            "world_uid": str(public["world_uid"]),
            "seller_uid": seller_uid,
            "item_uid": item_uid,
            "time_bucket": int(public["time_bucket"]),
            "title_nonempty": bool(ast["title_nonempty"]),
            "description_nonempty": bool(ast["description_nonempty"]),
            "base_description": base_description,
            "noise_clause": "" if noise is None else str(noise["raw_surface"]),
            "identity_slots": slots,
        }
        items_by_seller[seller_uid].append(item_state)
        public["title"] = title
        ast["title_skeleton_index"] = title_index
        ast["description_skeleton_index"] = description_index
        ast["code"] = neutral_code
        neutral_asts.append(ast)
    identity_audit, identity_edit, noise_audit = world_builder._render_identity_slots(
        policy=profile_policy,
        template=processing_template,
        fixture=fixture,
        items_by_seller=items_by_seller,
        noise_records_by_item=noise_by_item,
    )
    for rows in items_by_seller.values():
        for item in rows:
            public_by_item[str(item["item_uid"])]["description"] = str(item["description"])
    clone_count = 0
    for override in override_audit:
        kind = str(override["override_kind"])
        if kind == "exact_title_clone":
            source = str(override["item_uid_left"])
            target = str(override["item_uid_right"])
            public_by_item[target]["title"] = public_by_item[source]["title"]
            visible_title_source_by_item[target] = source
            clone_count += 1
        elif kind != "high_semantic_similarity":
            raise QualityChannelMaterializationError("Neutral override kind drift")
    non_code_commitment = _non_code_projection_commitment(
        source_asts_without_codes=render_asts_without_codes,
        neutral_asts=neutral_asts,
        source_identity_slots=identity_slots_audit,
        neutral_identity_slots=identity_audit,
        source_noise_slots=noise_slots_audit,
        neutral_noise_slots=noise_audit,
        safe_library=safe_library,
        title_inverse=title_inverse,
        description_inverse=description_inverse,
        effective_styles=effective_styles,
        visible_title_source_by_item=visible_title_source_by_item,
    )
    projected_world = {
        "public": {
            "world": {"world_uid": world_uid},
            "sellers": _clone(public_sellers),
            "items": list(public_by_item.values()),
        },
        "private": {
            "identity_slots_audit": identity_audit,
            "identity_slots_edit": identity_edit,
            "noise_slots_audit": noise_audit,
            "render_asts": neutral_asts,
            "override_audit": _clone(override_audit),
        },
    }
    processed = production.process_world(
        processing_policy,
        mode=mode,
        split=split,
        template=processing_template,
        world=projected_world,
    )
    neutral_items = _sorted_rows(processed["public"]["redacted_items"], "item_uid")
    neutral_code_inventory = tuple(
        neutral_code_by_item[item_uid]
        for item_uid in sorted(
            neutral_code_by_item, key=lambda value: value.encode("utf-8")
        )
    )
    for row in neutral_items:
        item_uid = str(row["item_uid"])
        for field in ("title", "description"):
            row[field] = _replace_neutral_code(
                str(row[field]),
                node_prefix=f"neutral.item.{item_uid}.{field}.code",
                registered_neutral_codes=neutral_code_inventory,
            )
    collapsed_profile_safe_items = production.build_profile_safe_items(
        processing_policy,
        items=projected_world["public"]["items"],
        redacted_items=neutral_items,
    )
    profiles, profile_audit = profiles_module.build_world_profiles(
        profile_policy,
        mode=mode,
        split=split,
        sellers=public_sellers,
        items=collapsed_profile_safe_items,
    )
    if profile_audit.get("labels_or_private_structure_read") is not False:
        raise QualityChannelMaterializationError("Neutral profile builder read private truth")
    neutral_profiles = _sorted_rows(profiles, "seller_uid")
    if any(
        channel.RAW_CODE_RE.search(str(row[field])) is not None
        for row in neutral_profiles
        for field in MODEL_PROFILE_TEXT_FIELDS
    ):
        raise QualityChannelMaterializationError(
            "Post-collapse neutral profile still contains a legal item code"
        )
    receipt = {
        "version": scientific.PERSISTED_STRUCTURE_VERSION,
        "neutral_render_code_ordinal_zero": NEUTRAL_RENDER_CODE,
        "neutral_code_family_rule": (
            "Q_plus_item_ordinal_as_eight_base16_A_to_P_digits_plus_BA"
        ),
        "neutral_code_family_count": len(neutral_code_inventory),
        "neutral_code_family_sha256": common.canonical_sha256(
            neutral_code_inventory
        ),
        "original_code_value_argument_count": 0,
        "original_code_value_read_count": (
            public_item_projection.forbidden_value_read_count
        ),
        "neutral_metadata_source_value_read_count": (
            public_item_projection.source_value_read_count
        ),
        "neutral_metadata_source_value_read_counts": dict(
            public_item_projection.source_value_read_counts
        ),
        "neutralizer_input_capability": "NeutralItemProjection[NeutralItemMetadata]",
        "neutralizer_input_fields": list(NEUTRAL_ITEM_METADATA_FIELDS),
        "neutral_profiles_recomputed_after_code_collapse": True,
        "neutral_profile_safe_item_sha256": common.canonical_sha256(
            collapsed_profile_safe_items
        ),
        "clone_count": clone_count,
        "title_template_mapping": title_mapping,
        "description_template_mapping": description_mapping,
        "per_item_template_mapping": per_item_template_mapping,
        "non_code_projection_commitment": non_code_commitment,
        "non_code_projection_nodes": [
            {
                **row,
                "visible_title_source_item_uid": visible_title_source_by_item[
                    str(row["item_uid"])
                ],
            }
            for row in non_code_projection_nodes
        ],
        "neutral_item_sha256": common.canonical_sha256(neutral_items),
        "neutral_profile_sha256": _persisted_profile_sha256(neutral_profiles),
    }
    return neutral_items, neutral_profiles, receipt


def _materialize_neutral_from_render_asts(
    *,
    processing_policy: Mapping[str, Any],
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    processing_template: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    fixture: Mapping[str, Any],
    world_uid: str,
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
    effective_styles: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run the real neutral item/profile mount after removing code values."""

    projection = _project_neutral_item_metadata(public_items)
    code_free_asts: list[dict[str, Any]] = []
    for row in render_asts:
        projected = _clone(row)
        projected.pop("code", None)
        code_free_asts.append(projected)
    return _neutralize_without_original_code_values(
        processing_policy=processing_policy,
        profile_policy=profile_policy,
        mode=mode,
        split=split,
        processing_template=processing_template,
        safe_library=safe_library,
        fixture=fixture,
        world_uid=world_uid,
        public_sellers=public_sellers,
        public_item_projection=projection,
        render_asts_without_codes=code_free_asts,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        override_audit=override_audit,
        effective_styles=effective_styles,
    )


def _profile_numeric_projection(profile: Mapping[str, Any]) -> dict[str, float]:
    values = {
        "title_length_median": float(profile["title_length_stats"]["median"]),
        "description_length_median": float(
            profile["description_length_stats"]["median"]
        ),
        "digit_ratio_mean": float(profile["style_stats"]["digit_ratio_mean"]),
        "punct_ratio_mean": float(profile["style_stats"]["punct_ratio_mean"]),
        "repeated_title_share": float(
            profile["style_stats"]["repeated_title_share"]
        ),
        "repeated_description_share": float(
            profile["style_stats"]["repeated_description_share"]
        ),
    }
    if set(values) != set(channel.NUMERIC_DELTA_FIELDS) or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise QualityChannelMaterializationError("Numeric profile projection drift")
    return values


def _public_probe_rows(
    *,
    world_uid: str,
    code_by_item: Mapping[str, str],
    seller_by_item: Mapping[str, str],
    item_occurrences: Sequence[Mapping[str, Any]],
    full_profiles: Sequence[Mapping[str, Any]],
    neutral_profiles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    full_by_seller = _row_index(full_profiles, key="seller_uid", name="full profile")
    neutral_by_seller = _row_index(
        neutral_profiles, key="seller_uid", name="neutral profile"
    )
    sellers = set(seller_by_item.values())
    if set(full_by_seller) != sellers or set(neutral_by_seller) != sellers:
        raise QualityChannelMaterializationError("Profile seller universe drift")
    item_occurrences_by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in item_occurrences:
        item_uid = str(occurrence["item_uid"])
        item_occurrences_by_seller[seller_by_item[item_uid]].append(_clone(occurrence))
    rows: list[dict[str, Any]] = []
    for seller_uid in sorted(sellers, key=lambda value: value.encode("utf-8")):
        owned = sorted(
            (
                code_by_item[item_uid]
                for item_uid, owner in seller_by_item.items()
                if owner == seller_uid
            ),
            key=lambda value: value.encode("ascii"),
        )
        owned_set = set(owned)
        profile_occurrences: list[dict[str, Any]] = []
        full_profile = full_by_seller[seller_uid]
        for field in MODEL_PROFILE_TEXT_FIELDS:
            text = str(full_profile[field])
            for match in channel.RAW_CODE_RE.finditer(text):
                code = match.group(0)
                if code not in set(code_by_item.values()):
                    raise QualityChannelMaterializationError(
                        "Profile exposes an unregistered item code"
                    )
                profile_occurrences.append(
                    {
                        "field": field,
                        "code": code,
                        "is_own": code in owned_set,
                    }
                )
        full_numeric = _profile_numeric_projection(full_profile)
        neutral_numeric = _profile_numeric_projection(neutral_by_seller[seller_uid])
        numeric_deltas = {
            name: full_numeric[name] - neutral_numeric[name]
            for name in channel.NUMERIC_DELTA_FIELDS
        }
        rows.append(
            {
                "world_uid": world_uid,
                "seller_uid": seller_uid,
                "owned_codes": owned,
                "item_occurrences": sorted(
                    (
                        {
                            "field": str(row["field"]),
                            "code": str(row["code"]),
                            "is_own": bool(row["is_own"]),
                        }
                        for row in item_occurrences_by_seller[seller_uid]
                    ),
                    key=lambda row: (
                        str(row["field"]).encode("ascii"),
                        str(row["code"]).encode("ascii"),
                        bool(row["is_own"]),
                    ),
                ),
                "profile_occurrences": sorted(
                    profile_occurrences,
                    key=lambda row: (
                        str(row["field"]).encode("ascii"),
                        str(row["code"]).encode("ascii"),
                        bool(row["is_own"]),
                    ),
                ),
                "numeric_profile_deltas": numeric_deltas,
            }
        )
    return tuple(rows)


def _text_probe_eligibility_rows(
    *,
    world_uid: str,
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    pair_uids = {
        str(row["canonical_pair_uid"])
        for row in complete_model_pair_endpoints
        if str(row.get("world_uid", "")) == world_uid
    }
    if len(complete_model_pair_endpoints) != 378 or len(pair_uids) != 378:
        raise QualityChannelMaterializationError(
            "Text-probe pair endpoint universe is not exactly 378"
        )
    excluded = tuple(
        sorted(
            (str(row["canonical_pair_uid"]) for row in override_audit),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(excluded) != 6 or len(set(excluded)) != 6 or not set(excluded) <= pair_uids:
        raise QualityChannelMaterializationError(
            "Text-probe exclusion universe is not exactly six registered pairs"
        )
    excluded_set = set(excluded)
    rows = tuple(
        {
            "world_uid": world_uid,
            "canonical_pair_uid": pair_uid,
            "text_probe_eligible": pair_uid not in excluded_set,
        }
        for pair_uid in sorted(pair_uids, key=lambda value: value.encode("utf-8"))
    )
    if sum(bool(row["text_probe_eligible"]) for row in rows) != 372:
        raise QualityChannelMaterializationError(
            "Text-probe eligibility count is not exactly 372"
        )
    return rows


def materialize_label_free_channel_views(
    *,
    processing_policy: Mapping[str, Any],
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    processing_template: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    fixture: Mapping[str, Any],
    world_uid: str,
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
    effective_styles: Mapping[str, Mapping[str, Any]],
    full_redacted_items: Sequence[Mapping[str, Any]],
    full_seller_profiles: Sequence[Mapping[str, Any]],
) -> MaterializedChannelViews:
    """Create all label-free view payloads before `_build_private_truth`."""

    if not world_uid or split not in {"train", "development", "audit_a", "audit_b"}:
        raise QualityChannelMaterializationError("World/split binding drift")
    public_by_item = _row_index(public_items, key="item_uid", name="public item")
    ast_by_item = _row_index(render_asts, key="item_uid", name="render AST")
    full_by_item = _row_index(
        full_redacted_items, key="item_uid", name="full redacted item"
    )
    if set(public_by_item) != set(ast_by_item) or set(ast_by_item) != set(full_by_item):
        raise QualityChannelMaterializationError("Channel item universe drift")
    code_by_item = {item_uid: str(row["code"]) for item_uid, row in ast_by_item.items()}
    all_codes = {channel._validate_code(code) for code in code_by_item.values()}
    if len(all_codes) != len(code_by_item):
        raise QualityChannelMaterializationError("Channel code registry is not unique")
    seller_by_item = {
        item_uid: str(row["seller_uid"]) for item_uid, row in public_by_item.items()
    }
    clone_source_by_target = {
        str(row["item_uid_right"]): str(row["item_uid_left"])
        for row in override_audit
        if row["override_kind"] == "exact_title_clone"
    }
    masked_items: list[dict[str, Any]] = []
    item_occurrences: list[dict[str, Any]] = []
    for item_uid in sorted(full_by_item, key=lambda value: value.encode("utf-8")):
        masked_text, receipt_rows = _registered_item_spans(
            row=full_by_item[item_uid],
            ast=ast_by_item[item_uid],
            ast_by_item=ast_by_item,
            clone_source_by_target=clone_source_by_target,
            safe_library=safe_library,
            all_codes=all_codes,
            effective_styles=effective_styles,
        )
        row = _clone(full_by_item[item_uid])
        row.update(masked_text)
        masked_items.append(row)
        item_occurrences.extend(receipt_rows)
    masked_items_tuple = _sorted_rows(masked_items, "item_uid")
    masked_profiles = _build_profiles(
        profile_policy=profile_policy,
        mode=mode,
        split=split,
        public_sellers=public_sellers,
        public_items=public_items,
        redacted_items=masked_items_tuple,
    )
    neutral_items, neutral_profiles, neutral_receipt = (
        _materialize_neutral_from_render_asts(
            processing_policy=processing_policy,
            profile_policy=profile_policy,
            mode=mode,
            split=split,
            processing_template=processing_template,
            safe_library=safe_library,
            fixture=fixture,
            world_uid=world_uid,
            public_sellers=public_sellers,
            public_items=public_items,
            render_asts=render_asts,
            identity_slots_audit=identity_slots_audit,
            noise_slots_audit=noise_slots_audit,
            override_audit=override_audit,
            effective_styles=effective_styles,
        )
    )
    ordered_item_uids = sorted(ast_by_item, key=lambda value: value.encode("utf-8"))
    if len(ordered_item_uids) < 2:
        raise QualityChannelMaterializationError(
            "Legal code permutation requires at least two items"
        )
    ordered_codes = [code_by_item[item_uid] for item_uid in ordered_item_uids]
    rotated_codes = ordered_codes[1:] + ordered_codes[:1]
    permuted_code_by_item = dict(zip(ordered_item_uids, rotated_codes))
    permuted_render_asts: list[dict[str, Any]] = []
    for row in render_asts:
        projected = _clone(row)
        projected["code"] = permuted_code_by_item[str(projected["item_uid"])]
        permuted_render_asts.append(projected)
    permuted_neutral_items, permuted_neutral_profiles, _ = (
        _materialize_neutral_from_render_asts(
            processing_policy=processing_policy,
            profile_policy=profile_policy,
            mode=mode,
            split=split,
            processing_template=processing_template,
            safe_library=safe_library,
            fixture=fixture,
            world_uid=world_uid,
            public_sellers=public_sellers,
            public_items=public_items,
            render_asts=permuted_render_asts,
            identity_slots_audit=identity_slots_audit,
            noise_slots_audit=noise_slots_audit,
            override_audit=override_audit,
            effective_styles=effective_styles,
        )
    )
    neutral_mount_bytes = common.canonical_json_bytes(
        {"items": neutral_items, "profiles": neutral_profiles}
    )
    permuted_neutral_mount_bytes = common.canonical_json_bytes(
        {
            "items": permuted_neutral_items,
            "profiles": permuted_neutral_profiles,
        }
    )
    permutation_byte_difference_count = _byte_difference_count(
        neutral_mount_bytes, permuted_neutral_mount_bytes
    )
    forbidden_access = _forbidden_access_observation(
        entrypoint_parameter_names=tuple(
            inspect.signature(materialize_label_free_channel_views).parameters
        ),
        accessed_capability_names=(),
    )
    forbidden_capability_mounted = dict(forbidden_access.capability_mounted)
    forbidden_read_counts = dict(forbidden_access.read_counts)
    public_probe = _public_probe_rows(
        world_uid=world_uid,
        code_by_item=code_by_item,
        seller_by_item=seller_by_item,
        item_occurrences=item_occurrences,
        full_profiles=full_seller_profiles,
        neutral_profiles=neutral_profiles,
    )
    text_probe_eligibility = _text_probe_eligibility_rows(
        world_uid=world_uid,
        complete_model_pair_endpoints=complete_model_pair_endpoints,
        override_audit=override_audit,
    )
    expected_visible_occurrences: Counter[tuple[str, str, str]] = Counter()
    for row in public_probe:
        seller_uid = str(row["seller_uid"])
        for occurrence in (*row["item_occurrences"], *row["profile_occurrences"]):
            expected_visible_occurrences[
                (seller_uid, str(occurrence["field"]), str(occurrence["code"]))
            ] += 1
    actual_visible_occurrences: Counter[tuple[str, str, str]] = Counter()
    for row in full_redacted_items:
        seller_uid = str(row["seller_uid"])
        for field in ("title", "description"):
            for match in channel.RAW_CODE_RE.finditer(str(row[field])):
                actual_visible_occurrences[(seller_uid, field, match.group(0))] += 1
    for row in full_seller_profiles:
        seller_uid = str(row["seller_uid"])
        for field in MODEL_PROFILE_TEXT_FIELDS:
            for match in channel.RAW_CODE_RE.finditer(str(row[field])):
                actual_visible_occurrences[(seller_uid, field, match.group(0))] += 1
    if actual_visible_occurrences != expected_visible_occurrences:
        raise QualityChannelMaterializationError(
            "Registered/model-mounted code occurrence closure drift"
        )
    masked_literal_hits = sum(
        len(channel.RAW_CODE_RE.findall(str(row[field])))
        for row in masked_items_tuple
        for field in ("title", "description")
    ) + sum(
        len(channel.RAW_CODE_RE.findall(str(row[field])))
        for row in masked_profiles
        for field in MODEL_PROFILE_TEXT_FIELDS
    )
    neutral_literal_hits = sum(
        len(channel.RAW_CODE_RE.findall(str(row[field])))
        for row in neutral_items
        for field in ("title", "description")
    ) + sum(
        len(channel.RAW_CODE_RE.findall(str(row[field])))
        for row in neutral_profiles
        for field in MODEL_PROFILE_TEXT_FIELDS
    )
    unregistered_full_hits = sum(
        count
        for (_seller_uid, _field, code), count in actual_visible_occurrences.items()
        if code not in all_codes
    )
    owner_by_code = {
        code_by_item[item_uid]: seller_by_item[item_uid]
        for item_uid in code_by_item
    }
    unregistered_clone_foreign_hits = 0
    for row in full_redacted_items:
        item_uid = str(row["item_uid"])
        seller_uid = str(row["seller_uid"])
        source_item_uid = clone_source_by_target.get(item_uid)
        allowed_clone_code = (
            None if source_item_uid is None else code_by_item[source_item_uid]
        )
        for field in ("title", "description"):
            for code in channel.RAW_CODE_RE.findall(str(row[field])):
                if owner_by_code.get(code) != seller_uid and code != allowed_clone_code:
                    unregistered_clone_foreign_hits += 1
    full_item_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
        for row in full_redacted_items
    }
    masked_item_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
        for row in masked_items_tuple
    }
    neutral_item_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
        for row in neutral_items
    }
    full_profile_keys = {str(row["seller_uid"]) for row in full_seller_profiles}
    masked_profile_keys = {str(row["seller_uid"]) for row in masked_profiles}
    neutral_profile_keys = {str(row["seller_uid"]) for row in neutral_profiles}
    view_keyset_difference_count = sum(
        len(left ^ right)
        for left, right in (
            (full_item_keys, masked_item_keys),
            (full_item_keys, neutral_item_keys),
            (full_profile_keys, masked_profile_keys),
            (full_profile_keys, neutral_profile_keys),
        )
    )
    structure_audit = {
        "version": scientific.PERSISTED_STRUCTURE_VERSION,
        "world_uid": world_uid,
        "item_count": len(public_by_item),
        "seller_count": len(public_sellers),
        "registered_code_count": len(all_codes),
        "registered_item_occurrence_count": len(item_occurrences),
        "registered_visible_occurrence_expected_count": sum(
            expected_visible_occurrences.values()
        ),
        "registered_visible_occurrence_actual_count": sum(
            actual_visible_occurrences.values()
        ),
        "registered_visible_occurrence_multiset_difference_count": sum(
            (expected_visible_occurrences - actual_visible_occurrences).values()
        )
        + sum((actual_visible_occurrences - expected_visible_occurrences).values()),
        "literal_code_hits_in_masked": masked_literal_hits,
        "literal_code_hits_in_neutralized": neutral_literal_hits,
        "unregistered_code_hits": unregistered_full_hits,
        "unregistered_clone_foreign_code_hits": (
            unregistered_clone_foreign_hits
        ),
        "view_keyset_difference_count": view_keyset_difference_count,
        "neutralized_legal_code_permutation_byte_difference_count": (
            permutation_byte_difference_count
        ),
        "clone_directions": [
            {
                "source_item_uid": str(row["item_uid_left"]),
                "target_item_uid": str(row["item_uid_right"]),
            }
            for row in override_audit
            if row["override_kind"] == "exact_title_clone"
        ],
        "neutral_receipt": neutral_receipt,
        "full_item_sha256": common.canonical_sha256(full_redacted_items),
        "masked_item_sha256": common.canonical_sha256(masked_items_tuple),
        "neutral_item_sha256": common.canonical_sha256(neutral_items),
        "full_profile_sha256": _persisted_profile_sha256(full_seller_profiles),
        "masked_profile_sha256": _persisted_profile_sha256(masked_profiles),
        "neutral_profile_sha256": _persisted_profile_sha256(neutral_profiles),
        "forbidden_capability_mounted": forbidden_capability_mounted,
        "audit_truth_open_count": forbidden_read_counts["audit_truth"],
        "audit_truth_read_count": forbidden_read_counts["audit_truth"],
        "audit_truth_materialized_row_count": forbidden_read_counts["audit_truth"],
        "generator_quality_result_read_count": forbidden_read_counts[
            "generator_quality_result"
        ],
        "candidate_quality_result_read_count": forbidden_read_counts[
            "candidate_quality_result"
        ],
        "view_builder_quality_result_read_count": forbidden_read_counts[
            "view_builder_quality_result"
        ],
    }
    if tuple(structure_audit) != STRUCTURE_AUDIT_FIELDS:
        raise QualityChannelMaterializationError(
            "Structure audit exact schema/order drift"
        )
    return MaterializedChannelViews(
        masked_redacted_items=masked_items_tuple,
        neutral_redacted_items=neutral_items,
        masked_seller_profiles=masked_profiles,
        neutral_seller_profiles=neutral_profiles,
        public_code_probe_input=public_probe,
        text_probe_eligibility_input=text_probe_eligibility,
        channel_structure_audit=structure_audit,
    )
