#!/usr/bin/env python3
"""Label-free V9.2 replay of one accepted candidate after style derangement."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_pure_natural_renderer_v9 as pure_renderer
import step28_v13_v1_13_scientific_world_v9 as v9_world
import step28_v13_world_builder as world_builder
from step28_v13_v1_13_style_derangement import (
    StyleSourceDerangement,
    build_style_source_derangement,
)


VERSION = "2026-08-23-step28-v13-v1-13-counterfactual-text-v9-2"
FORBIDDEN_CAPABILITIES = (
    "controller_membership",
    "pair_labels",
    "qrels",
    "audit_a_truth",
    "audit_b_truth",
    "quality_results",
)


class CounterfactualTextV92Error(common.ContractError):
    """Raised when the V9 production replay changes more than author style."""


@dataclass(frozen=True)
class CounterfactualFullSurface:
    redacted_items: tuple[dict[str, Any], ...]
    seller_profiles: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


class _StyleReadGuard(Mapping[str, Any]):
    """Expose only frozen style factors while recording actual renderer reads."""

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        on_read: Callable[[str], None],
    ) -> None:
        if set(values) != set(pure_renderer.STYLE_FIELDS):
            raise CounterfactualTextV92Error("Style read-guard schema drift")
        self._values = dict(values)
        self._on_read = on_read

    def __getitem__(self, key: str) -> Any:
        if key not in self._values:
            raise CounterfactualTextV92Error("Renderer requested an unknown style field")
        self._on_read(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(pure_renderer.STYLE_FIELDS)

    def __len__(self) -> int:
        return len(pure_renderer.STYLE_FIELDS)


def _clone(value: Any) -> Any:
    return json.loads(common.canonical_json_bytes(value).decode("utf-8"))


def _sorted_rows(
    rows: Sequence[Mapping[str, Any]], key: str
) -> tuple[dict[str, Any], ...]:
    output = tuple(
        sorted(
            (_clone(dict(row)) for row in rows),
            key=lambda row: str(row[key]).encode("utf-8"),
        )
    )
    if len({str(row[key]) for row in output}) != len(output):
        raise CounterfactualTextV92Error(f"Duplicate {key} in replay input")
    return output


def _one_value_per_seller(
    rows: Sequence[Mapping[str, Any]],
    *,
    seller_uids: Sequence[str],
    value_field: str,
) -> dict[str, str]:
    values: dict[str, set[str]] = {seller_uid: set() for seller_uid in seller_uids}
    for row in rows:
        seller_uid = str(row["seller_uid"])
        if seller_uid not in values:
            raise CounterfactualTextV92Error("Replay AST seller universe drift")
        values[seller_uid].add(str(row[value_field]))
    if any(len(value) != 1 for value in values.values()):
        raise CounterfactualTextV92Error(
            f"Replay requires one {value_field} per seller"
        )
    return {seller_uid: next(iter(values[seller_uid])) for seller_uid in seller_uids}


def _style_projection(
    effective_styles: Mapping[str, Mapping[str, Any]],
    *,
    seller_uids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if set(effective_styles) != set(seller_uids):
        raise CounterfactualTextV92Error("Effective-style seller universe drift")
    output: dict[str, dict[str, Any]] = {}
    for seller_uid in seller_uids:
        style = _clone(dict(effective_styles[seller_uid]))
        if set(style) != set(pure_renderer.STYLE_FIELDS):
            raise CounterfactualTextV92Error("Effective-style factor schema drift")
        output[seller_uid] = style
    return output


def _identity_core(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            key: value
            for key, value in row.items()
            if key not in {"start", "end"}
        }
        for row in _sorted_rows(rows, "slot_uid")
    )


def _noise_core(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            key: value
            for key, value in row.items()
            if key not in {"start", "end"}
        }
        for row in _sorted_rows(rows, "noise_slot_uid")
    )


def _ast_without_style(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {key: value for key, value in row.items() if key != "effective_style_uid"}
        for row in _sorted_rows(rows, "item_uid")
    )


def _public_item_non_text_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {key: value for key, value in row.items() if key not in {"title", "description"}}
        for row in _sorted_rows(rows, "item_uid")
    )


def _model_item_key_and_empty_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "title_nonempty": bool(str(row.get("title", ""))),
            "description_nonempty": bool(str(row.get("description", ""))),
        }
        for row in _sorted_rows(rows, "item_uid")
    )


def _seller_key_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        str(row["seller_uid"])
        for row in _sorted_rows(rows, "seller_uid")
    )


def _equal_commitment(source: object, counterfactual: object) -> dict[str, Any]:
    source_sha256 = common.canonical_sha256(source)
    counterfactual_sha256 = common.canonical_sha256(counterfactual)
    return {
        "source_sha256": source_sha256,
        "counterfactual_sha256": counterfactual_sha256,
        "equal": source_sha256 == counterfactual_sha256,
    }


def _replay_commitment(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "public_items",
        "redacted_items",
        "seller_profiles",
        "identity33",
        "identity_slots_audit",
        "identity_slots_edit",
        "noise_slots_audit",
        "render_asts",
        "complete_model_pair_endpoints",
        "override_audit",
        "style_read_audit",
        "profile_provenance_sha256",
    )
    return {field: _clone(value[field]) for field in fields}


def _rerender_once(
    *,
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    base_template: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    fixture: Mapping[str, Any],
    public_world: Mapping[str, Any],
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    original_redacted_items: Sequence[Mapping[str, Any]],
    original_seller_profiles: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
    source_by_target: Mapping[str, str],
    original_style_uid_by_seller: Mapping[str, str],
    style_factors_by_seller: Mapping[str, Mapping[str, Any]],
    baseline_identity33: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pure_renderer.validate_safe_library(safe_library)
    sellers = _sorted_rows(public_sellers, "seller_uid")
    items = _sorted_rows(public_items, "item_uid")
    asts = list(_sorted_rows(render_asts, "item_uid"))
    seller_uids = tuple(str(row["seller_uid"]) for row in sellers)
    item_by_uid = {str(row["item_uid"]): row for row in items}
    identity_by_item: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_slots_audit:
        identity_by_item[str(row["item_uid"])].append(_clone(dict(row)))
    noise_by_item = {
        str(row["item_uid"]): _clone(dict(row))
        for row in noise_slots_audit
    }
    if len(noise_by_item) != len(noise_slots_audit):
        raise CounterfactualTextV92Error("Duplicate noise item in replay input")

    native: dict[str, dict[str, Any]] = {}
    counterfactual_asts: list[dict[str, Any]] = []
    style_read_audit: list[dict[str, Any]] = []
    for ast in asts:
        item_uid = str(ast["item_uid"])
        seller_uid = str(ast["seller_uid"])
        if item_uid not in item_by_uid or seller_uid not in source_by_target:
            raise CounterfactualTextV92Error("Replay item/seller universe drift")
        source_seller_uid = str(source_by_target[seller_uid])
        style = dict(style_factors_by_seller[source_seller_uid])
        counterfactual_ast = _clone(ast)
        counterfactual_ast["effective_style_uid"] = original_style_uid_by_seller[
            source_seller_uid
        ]
        counterfactual_asts.append(counterfactual_ast)
        try:
            title_skeleton = safe_library["title_skeletons"][
                int(ast["title_skeleton_index"])
            ]
            description_skeleton = safe_library["description_skeletons"][
                int(ast["description_skeleton_index"])
            ]
        except (IndexError, KeyError, TypeError) as exc:
            raise CounterfactualTextV92Error(
                "Accepted semantic skeleton is outside the V9 safe library"
            ) from exc
        title_read_fields: list[str] = []
        description_read_fields: list[str] = []
        title = (
            pure_renderer._render_base_title(
                skeleton=str(title_skeleton),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                code=str(ast["code"]),
                style=_StyleReadGuard(
                    style, on_read=title_read_fields.append
                ),
                library=safe_library,
            )
            if bool(ast["title_nonempty"])
            else ""
        )
        base_description = (
            pure_renderer._render_base_description(
                skeleton=str(description_skeleton),
                product=str(ast["product"]),
                attribute=str(ast["attribute"]),
                code=str(ast["code"]),
                delivery=str(ast["delivery"]),
                service=str(ast["service"]),
                style=_StyleReadGuard(
                    style, on_read=description_read_fields.append
                ),
                library=safe_library,
            )
            if bool(ast["description_nonempty"])
            else ""
        )
        noise = noise_by_item.get(item_uid)
        noise_clause = "" if noise is None else str(noise["raw_surface"])
        if bool(ast["noise_slot_uid"]) != (noise is not None):
            raise CounterfactualTextV92Error("Replay noise-slot lineage drift")
        native[item_uid] = {
            "title": title,
            "base_description": base_description,
            "noise_clause": noise_clause,
        }
        style_read_audit.append(
            {
                "item_uid": item_uid,
                "seller_uid": seller_uid,
                "style_source_seller_uid": source_seller_uid,
                "title_style_read_fields": title_read_fields,
                "description_style_read_fields": description_read_fields,
                "title_style_read_count": len(title_read_fields),
                "description_style_read_count": len(description_read_fields),
            }
        )

    clone_endpoint_rows: list[dict[str, str]] = []
    for row in override_audit:
        kind = str(row["override_kind"])
        if kind == "exact_title_clone":
            source_uid = str(row["item_uid_left"])
            target_uid = str(row["item_uid_right"])
            if (
                source_uid not in native
                or target_uid not in native
                or not native[source_uid]["title"]
                or not native[target_uid]["base_description"]
            ):
                raise CounterfactualTextV92Error(
                    "Registered exact-title clone cannot be replayed"
                )
            native[target_uid]["title"] = native[source_uid]["title"]
            clone_endpoint_rows.append(
                {
                    "source_item_uid": source_uid,
                    "target_item_uid": target_uid,
                }
            )
        elif kind != "high_semantic_similarity":
            raise CounterfactualTextV92Error("Unknown registered override in replay")

    role_to_family = profile_policy["identity_design"]["role_to_template_family"]
    items_by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for ast in counterfactual_asts:
        item_uid = str(ast["item_uid"])
        source_item = item_by_uid[item_uid]
        slots: list[dict[str, Any]] = []
        for source_slot in sorted(
            identity_by_item[item_uid],
            key=lambda row: str(row["slot_uid"]).encode("utf-8"),
        ):
            family = str(role_to_family[str(source_slot["planned_role"])])
            expected_clause = v9_world.text_renderer.identity_clause(
                template_family=family,
                identity_type=str(source_slot["identity_type"]),
                normalized_value=str(source_slot["raw_surface"]),
                template=base_template,
            )
            if expected_clause.count(str(source_slot["raw_surface"])) != 1:
                raise CounterfactualTextV92Error(
                    "Replay identity clause no longer round-trips"
                )
            slots.append(
                {
                    "slot_uid": str(source_slot["slot_uid"]),
                    "bundle_uid": str(source_slot["bundle_uid"]),
                    "identity_uid": str(source_slot["identity_uid"]),
                    "role": str(source_slot["planned_role"]),
                    "identity_type": str(source_slot["identity_type"]),
                    "identity_value": str(source_slot["raw_surface"]),
                }
            )
        items_by_seller[str(ast["seller_uid"])].append(
            {
                "world_uid": str(source_item["world_uid"]),
                "seller_uid": str(source_item["seller_uid"]),
                "item_uid": item_uid,
                "time_bucket": int(source_item["time_bucket"]),
                "title_nonempty": bool(ast["title_nonempty"]),
                "description_nonempty": bool(ast["description_nonempty"]),
                "base_description": native[item_uid]["base_description"],
                "noise_clause": native[item_uid]["noise_clause"],
                "identity_slots": slots,
            }
        )
    noise_records = {
        item_uid: {
            "noise_slot_uid": str(row["noise_slot_uid"]),
            "raw_surface": str(row["raw_surface"]),
        }
        for item_uid, row in noise_by_item.items()
    }
    new_identity_audit, new_identity_edit, new_noise_audit = (
        world_builder._render_identity_slots(
            policy=profile_policy,
            template=base_template,
            fixture=fixture,
            items_by_seller=items_by_seller,
            noise_records_by_item=noise_records,
        )
    )
    descriptions = {
        str(item["item_uid"]): str(item["description"])
        for rows in items_by_seller.values()
        for item in rows
    }
    replay_items = []
    for source in items:
        item_uid = str(source["item_uid"])
        ast = next(row for row in counterfactual_asts if row["item_uid"] == item_uid)
        row = _clone(source)
        if str(row["category"]) != str(ast["category"]):
            raise CounterfactualTextV92Error("Accepted category/AST drift")
        row["title"] = native[item_uid]["title"]
        row["description"] = descriptions[item_uid]
        replay_items.append(row)

    replay_world = {
        "public": {
            "world": _clone(dict(public_world)),
            "sellers": list(sellers),
            "items": replay_items,
            "complete_model_pair_endpoints": [
                _clone(dict(row)) for row in complete_model_pair_endpoints
            ],
        },
        "private": {
            "identity_slots_audit": new_identity_audit,
            "identity_slots_edit": new_identity_edit,
            "noise_slots_audit": new_noise_audit,
            "render_asts": counterfactual_asts,
            "override_audit": [_clone(dict(row)) for row in override_audit],
        },
    }
    profiles, provenance, identity33, redacted = v9_world._build_profiles_and_identity33(
        policy=profile_policy,
        mode=mode,
        split=split,
        template=base_template,
        world=replay_world,
        candidate_only_attributes=v9_world.CANDIDATE_ONLY_ATTRIBUTES,
    )
    baseline_identity = tuple(
        sorted(
            (_clone(dict(row)) for row in baseline_identity33),
            key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"),
        )
    )
    if common.canonical_json_bytes(identity33) != common.canonical_json_bytes(
        baseline_identity
    ):
        raise CounterfactualTextV92Error("Style replay changed Identity33")
    if _identity_core(identity_slots_audit) != _identity_core(new_identity_audit):
        raise CounterfactualTextV92Error("Style replay changed identity-slot core")
    if _noise_core(noise_slots_audit) != _noise_core(new_noise_audit):
        raise CounterfactualTextV92Error("Style replay changed noise-slot core")
    if _public_item_non_text_projection(public_items) != _public_item_non_text_projection(
        replay_items
    ):
        raise CounterfactualTextV92Error("Style replay changed public item semantics")
    if _model_item_key_and_empty_projection(original_redacted_items) != (
        _model_item_key_and_empty_projection(redacted)
    ):
        raise CounterfactualTextV92Error(
            "Style replay changed model-item keys or empty-field pattern"
        )
    if _seller_key_projection(original_seller_profiles) != _seller_key_projection(
        profiles
    ):
        raise CounterfactualTextV92Error("Style replay changed seller-profile universe")
    return {
        "public_items": tuple(_sorted_rows(replay_items, "item_uid")),
        "redacted_items": tuple(_sorted_rows(redacted, "item_uid")),
        "seller_profiles": tuple(_sorted_rows(profiles, "seller_uid")),
        "identity33": identity33,
        "identity_slots_audit": tuple(_sorted_rows(new_identity_audit, "slot_uid")),
        "identity_slots_edit": tuple(_sorted_rows(new_identity_edit, "slot_uid")),
        "noise_slots_audit": tuple(
            _sorted_rows(new_noise_audit, "noise_slot_uid")
        ),
        "render_asts": tuple(_sorted_rows(counterfactual_asts, "item_uid")),
        "complete_model_pair_endpoints": tuple(
            _sorted_rows(complete_model_pair_endpoints, "canonical_pair_uid")
        ),
        "override_audit": tuple(_clone(dict(row)) for row in override_audit),
        "style_read_audit": tuple(_sorted_rows(style_read_audit, "item_uid")),
        "profile_provenance_sha256": common.canonical_sha256(provenance),
        "clone_endpoint_rows": clone_endpoint_rows,
    }


def materialize_style_deranged_full_surface(
    *,
    profile_policy: Mapping[str, Any],
    mode: str,
    split: str,
    base_template: Mapping[str, Any],
    safe_library: Mapping[str, Any],
    fixture: Mapping[str, Any],
    world_uid: str,
    candidate_key: bytes,
    public_world: Mapping[str, Any],
    public_sellers: Sequence[Mapping[str, Any]],
    public_items: Sequence[Mapping[str, Any]],
    original_redacted_items: Sequence[Mapping[str, Any]],
    original_seller_profiles: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
    effective_styles: Mapping[str, Mapping[str, Any]],
    baseline_identity33: Sequence[Mapping[str, Any]],
) -> CounterfactualFullSurface:
    """Create one label-free deranged surface and independently replay it twice."""

    if not isinstance(candidate_key, bytes) or len(candidate_key) != 32:
        raise CounterfactualTextV92Error("Candidate key must be exactly 32 bytes")
    if str(public_world.get("world_uid", "")) != world_uid:
        raise CounterfactualTextV92Error("Public world binding drift")
    sellers = _sorted_rows(public_sellers, "seller_uid")
    seller_uids = tuple(str(row["seller_uid"]) for row in sellers)
    if len(seller_uids) != 28:
        raise CounterfactualTextV92Error("V9.2 requires exactly 28 sellers")
    styles = _style_projection(effective_styles, seller_uids=seller_uids)
    original_style_uids = _one_value_per_seller(
        render_asts,
        seller_uids=seller_uids,
        value_field="effective_style_uid",
    )
    mapping: StyleSourceDerangement = build_style_source_derangement(
        split=split,
        world_uid=world_uid,
        seller_uids=seller_uids,
    )
    source_by_target = mapping.as_mapping()
    kwargs = {
        "profile_policy": profile_policy,
        "mode": mode,
        "split": split,
        "base_template": base_template,
        "safe_library": safe_library,
        "fixture": fixture,
        "public_world": public_world,
        "public_sellers": public_sellers,
        "public_items": public_items,
        "original_redacted_items": original_redacted_items,
        "original_seller_profiles": original_seller_profiles,
        "complete_model_pair_endpoints": complete_model_pair_endpoints,
        "render_asts": render_asts,
        "identity_slots_audit": identity_slots_audit,
        "noise_slots_audit": noise_slots_audit,
        "override_audit": override_audit,
        "source_by_target": source_by_target,
        "original_style_uid_by_seller": original_style_uids,
        "style_factors_by_seller": styles,
        "baseline_identity33": baseline_identity33,
    }
    replay_a = _rerender_once(**kwargs)
    replay_b = _rerender_once(**kwargs)
    commitment_a = _replay_commitment(replay_a)
    commitment_b = _replay_commitment(replay_b)
    if common.canonical_json_bytes(commitment_a) != common.canonical_json_bytes(
        commitment_b
    ):
        raise CounterfactualTextV92Error("Independent V9 production replays diverged")

    original_asts = _sorted_rows(render_asts, "item_uid")
    if _ast_without_style(original_asts) != _ast_without_style(replay_a["render_asts"]):
        raise CounterfactualTextV92Error("A non-style AST field changed")
    original_items = _sorted_rows(public_items, "item_uid")
    original_item_by_uid = {str(row["item_uid"]): row for row in original_items}
    replay_item_by_uid = {
        str(row["item_uid"]): row for row in replay_a["public_items"]
    }
    if set(original_item_by_uid) != set(replay_item_by_uid):
        raise CounterfactualTextV92Error("Replay public item keyset drift")
    title_changes = sum(
        str(original_item_by_uid[item_uid]["title"])
        != str(replay_item_by_uid[item_uid]["title"])
        for item_uid in original_item_by_uid
    )
    description_changes = sum(
        str(original_item_by_uid[item_uid]["description"])
        != str(replay_item_by_uid[item_uid]["description"])
        for item_uid in original_item_by_uid
    )
    visible_change_sellers = {
        str(original_item_by_uid[item_uid]["seller_uid"])
        for item_uid in original_item_by_uid
        if (
            str(original_item_by_uid[item_uid]["title"])
            != str(replay_item_by_uid[item_uid]["title"])
            or str(original_item_by_uid[item_uid]["description"])
            != str(replay_item_by_uid[item_uid]["description"])
        )
    }
    factor_names = tuple(pure_renderer.STYLE_FIELDS)
    original_factor_tuples = {
        tuple(styles[seller_uid][name] for name in factor_names)
        for seller_uid in seller_uids
    }
    style_read_rows = tuple(replay_a["style_read_audit"])
    carrier_counts = {
        seller_uid: sum(
            int(int(row["title_style_read_count"]) > 0)
            + int(int(row["description_style_read_count"]) > 0)
            for row in style_read_rows
            if str(row["seller_uid"]) == seller_uid
        )
        for seller_uid in seller_uids
    }
    style_factor_read_counts = {
        seller_uid: sum(
            int(row["title_style_read_count"])
            + int(row["description_style_read_count"])
            for row in style_read_rows
            if str(row["seller_uid"]) == seller_uid
        )
        for seller_uid in seller_uids
    }
    uid_change_count = sum(
        original_style_uids[target]
        != original_style_uids[source_by_target[target]]
        for target in seller_uids
    )
    factor_change_count = sum(
        tuple(styles[target][name] for name in factor_names)
        != tuple(styles[source_by_target[target]][name] for name in factor_names)
        for target in seller_uids
    )
    expected_clone_endpoint_rows = [
        {
            "source_item_uid": str(row["item_uid_left"]),
            "target_item_uid": str(row["item_uid_right"]),
        }
        for row in override_audit
        if str(row["override_kind"]) == "exact_title_clone"
    ]
    audit: dict[str, Any] = {
        "version": VERSION,
        "world_uid": world_uid,
        "mapping": {
            "attempt": mapping.attempt,
            "seller_set_sha256": mapping.seller_set_sha256,
            "mapping_sha256": mapping.mapping_sha256,
            "target_source_pairs": [list(row) for row in mapping.target_source_pairs],
            "fixed_point_count": sum(
                target == source for target, source in mapping.target_source_pairs
            ),
        },
        "candidate_key_sha256": hashlib.sha256(candidate_key).hexdigest(),
        "forbidden_capability_mounted": {
            name: False for name in FORBIDDEN_CAPABILITIES
        },
        "double_replay": {
            "independent_production_replay_count": 2,
            "canonical_commitment_sha256": common.canonical_sha256(commitment_a),
            "byte_identical": True,
        },
        "invariants": {
            "non_effective_style_ast": _equal_commitment(
                _ast_without_style(original_asts),
                _ast_without_style(replay_a["render_asts"]),
            ),
            "public_item_non_text": _equal_commitment(
                _public_item_non_text_projection(public_items),
                _public_item_non_text_projection(replay_a["public_items"]),
            ),
            "model_item_key_and_empty_pattern": _equal_commitment(
                _model_item_key_and_empty_projection(original_redacted_items),
                _model_item_key_and_empty_projection(replay_a["redacted_items"]),
            ),
            "pair_endpoint_order": _equal_commitment(
                complete_model_pair_endpoints,
                replay_a["complete_model_pair_endpoints"],
            ),
            "identity33": _equal_commitment(
                baseline_identity33, replay_a["identity33"]
            ),
            "identity_slot_core": _equal_commitment(
                _identity_core(identity_slots_audit),
                _identity_core(replay_a["identity_slots_audit"]),
            ),
            "noise_slot_core": _equal_commitment(
                _noise_core(noise_slots_audit),
                _noise_core(replay_a["noise_slots_audit"]),
            ),
            "override_audit": _equal_commitment(
                override_audit, replay_a["override_audit"]
            ),
            "clone_endpoint_and_direction": _equal_commitment(
                expected_clone_endpoint_rows,
                replay_a["clone_endpoint_rows"],
            ),
            "seller_profile_keyset": _equal_commitment(
                _seller_key_projection(original_seller_profiles),
                _seller_key_projection(replay_a["seller_profiles"]),
            ),
        },
        "model_inputs": {
            "original_full_items_sha256": common.canonical_sha256(
                original_redacted_items
            ),
            "original_full_profiles_sha256": common.canonical_sha256(
                v9_world.channel_materializer._persisted_profile_rows(
                    original_seller_profiles
                )
            ),
            "counterfactual_full_items_sha256": common.canonical_sha256(
                replay_a["redacted_items"]
            ),
            "counterfactual_full_profiles_sha256": common.canonical_sha256(
                v9_world.channel_materializer._persisted_profile_rows(
                    replay_a["seller_profiles"]
                )
            ),
        },
        "style_structure": {
            "minimum_distinct_style_factor_tuples_required": 2,
            "observed_distinct_style_factor_tuple_count": len(
                original_factor_tuples
            ),
            "minimum_visible_carrier_fields_per_seller_required": 1,
            "minimum_observed_visible_carrier_fields_per_seller": min(
                carrier_counts.values()
            ),
            "visible_carrier_fields_by_seller_sha256": common.canonical_sha256(
                carrier_counts
            ),
            "renderer_style_item_read_count": sum(
                int(
                    int(row["title_style_read_count"])
                    + int(row["description_style_read_count"])
                    > 0
                )
                for row in style_read_rows
            ),
            "renderer_style_field_read_count": sum(carrier_counts.values()),
            "renderer_style_factor_read_count": sum(
                style_factor_read_counts.values()
            ),
            "minimum_actual_style_factor_reads_per_seller": min(
                style_factor_read_counts.values()
            ),
            "renderer_style_factor_reads_by_seller_sha256": common.canonical_sha256(
                style_factor_read_counts
            ),
            "renderer_style_read_audit_sha256": common.canonical_sha256(
                style_read_rows
            ),
            "effective_style_uid_changed_seller_count": uid_change_count,
            "effective_style_factor_tuple_changed_seller_count": factor_change_count,
            "visible_change_seller_count": len(visible_change_sellers),
            "title_change_count": title_changes,
            "description_change_count": description_changes,
            "visible_change_seller_set_sha256": common.canonical_sha256(
                sorted(visible_change_sellers, key=lambda value: value.encode("utf-8"))
            ),
        },
        "labels_or_retrieval_truth_read": False,
        "quality_result_read_count": 0,
    }
    audit["canonical_self_hash"] = common.canonical_sha256(audit)
    return CounterfactualFullSurface(
        redacted_items=tuple(replay_a["redacted_items"]),
        seller_profiles=tuple(replay_a["seller_profiles"]),
        audit=audit,
    )
