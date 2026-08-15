#!/usr/bin/env python3
"""V9 multi-world scientific generator for Step28-v13 v1.13.

The candidate loop can observe only anonymous text material and exact document
hash collisions.  Pair truth is projected only after a candidate is accepted.
Contribution sources are frozen as an unordered multiset; their derived output
rank may change when an otherwise admissible natural expression changes.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import step28_v13_common as common
import step28_v13_history_features as history_features
import step28_v13_identity_values as identity_values
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_text_renderer as text_renderer
import step28_v13_v1_13_candidate_parent as stage_parent
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_identity_remap as identity_remap
import step28_v13_v1_13_natural_variation as stage_variation
import step28_v13_v1_13_document_capacity_v9 as document_capacity
import step28_v13_v1_13_pure_natural_renderer_v9 as pure_renderer
import step28_v13_v1_13_quality_channel_materializer_v9 as channel_materializer
import step28_v13_world_builder as world_builder


HANDLE_DOMAIN = b"step28-v13-v1.13-scientific-anonymous-handle"
EXACT_CLONE_ENDPOINT_DOMAIN = (
    b"step28-v13-v1.13-scientific-exact-clone-endpoint-v1"
)
CANDIDATE_LIMIT = 32
CANDIDATE_ONLY_ATTRIBUTES = ("通用版",)
CANDIDATE_TEMPLATE_RELATIVE_PATH = (
    "schema/step28_v13_v1_13_candidate_text_templates_v9.json"
)
CANDIDATE_TEMPLATE_SHA256 = (
    "9a868644f76ad23be2e16973567a5b3c69d70bcbb23690453093dad3658e7a2a"
)
COLLISION_CATEGORIES = (
    "same_world_item_document",
    "same_world_seller_document",
    "historical_item_document",
    "historical_seller_document",
    "current_dataset_item_document",
    "current_dataset_seller_document",
)


class ScientificWorldError(common.ContractError):
    """Raised when one scientific world cannot close exactly."""


@dataclass(frozen=True)
class CandidateBinding:
    item_handle_to_uid: dict[str, str]
    noise_handle_to_slot_uid: dict[str, str]
    registered_overrides: tuple[dict[str, Any], ...]
    structural_parent_sha256: str
    baseline_identity33_sha256: str
    capacity_receipt: dict[str, Any]


@dataclass(frozen=True)
class CandidateObservation:
    world: dict[str, Any]
    redacted_items: tuple[dict[str, Any], ...]
    seller_profiles: tuple[dict[str, Any], ...]
    identity33: tuple[dict[str, Any], ...]
    profile_provenance_sha256: str
    profile_provenance_source_multiset_sha256: str
    item_document_hashes: tuple[str, ...]
    seller_document_hashes: tuple[str, ...]
    item_codes: tuple[str, ...]
    document_capacity_audit: dict[str, Any]
    natural_output_sha256: str


@dataclass(frozen=True)
class AcceptedScientificWorld:
    split: str
    split_ordinal: int
    world_uid: str
    candidate_index: int
    candidates_examined: int
    rejection_counts: dict[str, int]
    world: dict[str, Any]
    redacted_items: tuple[dict[str, Any], ...]
    seller_profiles: tuple[dict[str, Any], ...]
    masked_redacted_items: tuple[dict[str, Any], ...]
    neutral_redacted_items: tuple[dict[str, Any], ...]
    masked_seller_profiles: tuple[dict[str, Any], ...]
    neutral_seller_profiles: tuple[dict[str, Any], ...]
    public_code_probe_input: tuple[dict[str, Any], ...]
    text_probe_eligibility_input: tuple[dict[str, Any], ...]
    channel_structure_audit: dict[str, Any]
    identity33: tuple[dict[str, Any], ...]
    controller_membership: tuple[dict[str, Any], ...]
    pair_labels: tuple[dict[str, Any], ...]
    qrels: tuple[dict[str, Any], ...]
    identity_allocation_receipt: dict[str, Any]
    identity_registry_delta: tuple[str, ...]
    code_registry_delta: tuple[str, ...]
    item_registry_delta: tuple[str, ...]
    seller_registry_delta: tuple[str, ...]
    structural_parent_sha256: str
    candidate_zero_lineage_reference_sha256: str
    document_capacity_receipt: dict[str, Any]
    document_capacity_audit: dict[str, Any]
    profile_provenance_sha256: str
    identity33_sha256: str
    natural_output_sha256: str


def _canonical_clone(value: Any) -> Any:
    return json.loads(common.canonical_json_bytes(value).decode("utf-8"))


def _candidate_safe_library(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Extend the base whitelist by one attribute and code-bearing twins."""

    output = stage_variation._safe_library(
        base_policy=policy,
        template=template,
        fixture=fixture,
        split=split,
    )
    candidate_path = common.repo_path(CANDIDATE_TEMPLATE_RELATIVE_PATH)
    if common.sha256_file(candidate_path) != CANDIDATE_TEMPLATE_SHA256:
        raise ScientificWorldError("V9 candidate-template payload drift")
    candidate_template = common.load_json(candidate_path)
    normalized_candidate = _canonical_clone(candidate_template)
    candidate_attributes = normalized_candidate["generic_lexicon"]["attributes"]
    if candidate_attributes[-1:] != list(CANDIDATE_ONLY_ATTRIBUTES):
        raise ScientificWorldError("Candidate-only attribute extension drift")
    normalized_candidate["generic_lexicon"]["attributes"] = candidate_attributes[:-1]
    for split_name, split_library in normalized_candidate["split_libraries"].items():
        if split_name not in {"train", "development", "audit_a", "audit_b"}:
            raise ScientificWorldError("Candidate-template split drift")
        split_library["title_skeletons"] = split_library["title_skeletons"][:8]
        split_library["description_skeletons"] = split_library[
            "description_skeletons"
        ][:8]
    if common.canonical_json_bytes(normalized_candidate) != common.canonical_json_bytes(
        template
    ):
        raise ScientificWorldError("Candidate template changes a forbidden base field")
    if (
        tuple(CANDIDATE_ONLY_ATTRIBUTES) != ("通用版",)
        or "通用版" in output["attributes"]
    ):
        raise ScientificWorldError("Candidate-only attribute baseline drift")
    output["attributes"].append("通用版")
    output["attribute_permutation_classes"].append(["通用版"])
    candidate_split = candidate_template["split_libraries"][split]
    for values_field, classes_field in (
        ("title_skeletons", "title_skeleton_permutation_classes"),
        ("description_skeletons", "description_skeleton_permutation_classes"),
    ):
        base_count = len(output[values_field])
        candidate_values = [str(value) for value in candidate_split[values_field]]
        if base_count != 8 or candidate_values[:base_count] != output[values_field]:
            raise ScientificWorldError("Candidate skeleton base domain drift")
        output[values_field] = candidate_values
        output[classes_field].extend(
            [[index] for index in range(base_count, len(candidate_values))]
        )
    pure_renderer.validate_safe_library(output)
    return output


def _anonymous_handle(*, key: bytes, kind: str, value: str) -> str:
    if not isinstance(key, bytes) or len(key) != 32 or not kind or not value:
        raise ScientificWorldError("Anonymous handle input is malformed")
    digest = hmac.new(
        key,
        common.FIELD_SEPARATOR.join(
            (HANDLE_DOMAIN, kind.encode("ascii"), value.encode("utf-8"))
        ),
        hashlib.sha256,
    ).hexdigest()
    return f"h_{kind}_{digest[:32]}"


def _sorted_rows(
    rows: Sequence[Mapping[str, Any]], *fields: str
) -> list[dict[str, Any]]:
    return sorted(
        (_canonical_clone(dict(row)) for row in rows),
        key=lambda row: tuple(str(row[field]).encode("utf-8") for field in fields),
    )


def _structural_parent_projection(world: Mapping[str, Any]) -> dict[str, Any]:
    """Project everything a natural text candidate is forbidden to change."""

    public = world["public"]
    private = world["private"]
    render_fields = (
        "world_uid",
        "seller_uid",
        "item_uid",
        "time_bucket",
        "code",
        "effective_style_uid",
        "title_nonempty",
        "description_nonempty",
        "identity_slot_uids",
        "noise_slot_uid",
    )
    identity_slots = [
        {key: value for key, value in row.items() if key not in {"start", "end"}}
        for row in _sorted_rows(private["identity_slots_audit"], "slot_uid")
    ]
    identity_edits = [
        {key: value for key, value in row.items() if key not in {"start", "end"}}
        for row in _sorted_rows(private["identity_slots_edit"], "slot_uid")
    ]
    noise_slots = [
        {
            key: value
            for key, value in row.items()
            if key not in {"start", "end", "raw_surface"}
        }
        for row in _sorted_rows(private["noise_slots_audit"], "noise_slot_uid")
    ]
    return {
        "world": _canonical_clone(public["world"]),
        "sellers": _sorted_rows(public["sellers"], "seller_uid"),
        "item_structure": [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": row["item_uid"],
                "time_bucket": row["time_bucket"],
            }
            for row in _sorted_rows(public["items"], "item_uid")
        ],
        "pair_endpoints": _sorted_rows(
            public["complete_model_pair_endpoints"], "canonical_pair_uid"
        ),
        "controller_membership": _sorted_rows(
            private["controller_membership"], "controller_uid", "seller_uid"
        ),
        "controller_style_groups": _sorted_rows(
            private["controller_style_groups"], "controller_uid"
        ),
        "mechanism_assignments": _sorted_rows(
            private["mechanism_assignments"], "controller_uid"
        ),
        "identity_assets": _sorted_rows(
            private["identity_assets"], "identity_asset_uid"
        ),
        "identity_slots": identity_slots,
        "identity_slot_edits": identity_edits,
        "noise_slot_targets": noise_slots,
        "render_structure": [
            {field: row[field] for field in render_fields}
            for row in _sorted_rows(private["render_asts"], "item_uid")
        ],
        "positive_targets": _sorted_rows(
            private["positive_targets"], "controller_uid", "mechanism_slot_uid"
        ),
        "negative_flags": _sorted_rows(
            private["negative_flags"], "flag", "canonical_pair_uid"
        ),
        "override_audit": _sorted_rows(
            private["override_audit"], "asset_index", "override_kind"
        ),
        "exact_title_clone_endpoint_qualification": _canonical_clone(
            private["exact_title_clone_endpoint_qualification"]
        ),
        "solver_audit": _canonical_clone(private["solver_audit"]),
    }


def _profile_provenance_source_multiset_sha256(
    provenance: Mapping[str, Any],
) -> str:
    """Bind every contribution-source fact while ignoring only output rank.

    Natural text variation can reorder otherwise identical contribution rows.
    ``output_rank`` is a downstream presentation position, not source lineage.
    Multiplicity is retained, and every remaining provenance field is frozen.
    """

    expected_container_fields = {
        "version",
        "world_uid",
        "seller_count",
        "profile_count",
        "contribution_row_count",
        "raw_contribution_values_persisted",
        "private_audit_only",
        "rows",
        "rows_sha256",
    }
    if set(provenance) != expected_container_fields:
        raise ScientificWorldError("Profile provenance container schema drift")
    rows_value = provenance.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise ScientificWorldError("Profile provenance rows are malformed")
    if (
        provenance.get("contribution_row_count") != len(rows_value)
        or provenance.get("rows_sha256") != common.canonical_sha256(rows_value)
        or provenance.get("raw_contribution_values_persisted") is not False
        or provenance.get("private_audit_only") is not True
    ):
        raise ScientificWorldError("Profile provenance receipt drift")

    ranks_by_group: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    normalized_rows: list[dict[str, Any]] = []
    for source in rows_value:
        if not isinstance(source, Mapping) or set(source) != set(
            stage_parent.PROVENANCE_FIELDS
        ):
            raise ScientificWorldError("Profile provenance row schema drift")
        row = _canonical_clone(dict(source))
        if row["world_uid"] != provenance["world_uid"]:
            raise ScientificWorldError("Profile provenance world binding drift")
        rank = row.pop("output_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ScientificWorldError("Profile provenance output rank is malformed")
        ranks_by_group[(str(row["seller_uid"]), str(row["output_field"]))].append(
            rank
        )
        normalized_rows.append(row)

    for ranks in ranks_by_group.values():
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ScientificWorldError("Profile provenance output ranks are not closed")
    normalized_rows.sort(key=common.canonical_json_bytes)
    projection = {
        "version": provenance["version"],
        "world_uid": provenance["world_uid"],
        "seller_count": provenance["seller_count"],
        "profile_count": provenance["profile_count"],
        "contribution_row_count": provenance["contribution_row_count"],
        "raw_contribution_values_persisted": provenance[
            "raw_contribution_values_persisted"
        ],
        "private_audit_only": provenance["private_audit_only"],
        "rank_semantics": "excluded_downstream_presentation_position_only",
        "rows_without_output_rank": normalized_rows,
    }
    return common.canonical_sha256(projection)


def _rank_exact_clone_items(
    *,
    structure_key_hex: str,
    world_uid: str,
    asset_index: int,
    side: str,
    item_uids: Sequence[str],
) -> list[str]:
    try:
        key = bytes.fromhex(structure_key_hex)
    except ValueError as exc:
        raise ScientificWorldError("Structure key is not lowercase hexadecimal") from exc
    if len(key) != 32 or side not in {"source", "target"}:
        raise ScientificWorldError("Exact-title endpoint ranking authority drift")
    if not item_uids or len(set(item_uids)) != len(item_uids):
        raise ScientificWorldError("Exact-title endpoint candidate universe drift")

    def score(item_uid: str) -> tuple[bytes, bytes]:
        message = common.FIELD_SEPARATOR.join(
            (
                EXACT_CLONE_ENDPOINT_DOMAIN,
                world_uid.encode("utf-8"),
                str(asset_index).encode("ascii"),
                side.encode("ascii"),
                item_uid.encode("utf-8"),
            )
        )
        return hmac.new(key, message, hashlib.sha256).digest(), item_uid.encode(
            "utf-8"
        )

    return sorted(item_uids, key=score)


def _qualify_exact_title_clone_endpoints(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    split: str,
    structure_key_hex: str,
    world: dict[str, Any],
) -> dict[str, Any]:
    """Relocate only item endpoints so every registered title clone is renderable.

    Seller pairs, clone direction, pair truth, and every non-clone mechanism remain
    frozen. Selection reads only item ownership, structural nonempty flags, UIDs,
    and the already-authorized split structure key.
    """

    private = world["private"]
    if "exact_title_clone_endpoint_qualification" in private:
        raise ScientificWorldError("Exact-title endpoints were qualified twice")
    public_items = {
        str(row["item_uid"]): row for row in world["public"]["items"]
    }
    render_asts = {
        str(row["item_uid"]): row for row in private["render_asts"]
    }
    if (
        len(public_items) != len(world["public"]["items"])
        or len(render_asts) != len(private["render_asts"])
        or set(public_items) != set(render_asts)
    ):
        raise ScientificWorldError("Exact-title qualification item universe drift")
    world_uid = str(world["public"]["world"]["world_uid"])
    if not world_uid or split not in {"train", "development", "audit_a", "audit_b"}:
        raise ScientificWorldError("Exact-title qualification world/split drift")

    style_rows = stage_parent._effective_style_rows(
        policy=policy,
        template=template,
        mode=mode,
        world=world,
    )
    style_by_seller = {
        str(row["seller_uid"]): dict(row["style_factors"])
        for row in style_rows
    }
    title_skeletons = template["split_libraries"][split]["title_skeletons"]

    def base_title(item_uid: str) -> str:
        ast = render_asts[item_uid]
        if ast["title_nonempty"] is not True:
            return ""
        seller_uid = str(ast["seller_uid"])
        try:
            style = style_by_seller[seller_uid]
            skeleton = title_skeletons[int(ast["title_skeleton_index"])]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ScientificWorldError(
                "Exact-title base-title replay input drift"
            ) from exc
        return text_renderer.render_base_title(
            skeleton=str(skeleton),
            product=str(ast["product"]),
            attribute=str(ast["attribute"]),
            code=str(ast["code"]),
            style=style,
            template=template,
        )

    overrides = [_canonical_clone(row) for row in private["override_audit"]]
    clone_rows = [
        row for row in overrides if row["override_kind"] == "exact_title_clone"
    ]
    semantic_rows = [
        row
        for row in overrides
        if row["override_kind"] == "high_semantic_similarity"
    ]
    if len(overrides) != 6 or len(clone_rows) != 2 or len(semantic_rows) != 4:
        raise ScientificWorldError("Registered override count drift before qualification")

    # Undo the two already-materialized clones before choosing qualified items.
    for row in clone_rows:
        old_source = str(row["item_uid_left"])
        old_target = str(row["item_uid_right"])
        if (
            old_source not in public_items
            or old_target not in public_items
            or str(public_items[old_source]["seller_uid"])
            != str(row["seller_uid_left"])
            or str(public_items[old_target]["seller_uid"])
            != str(row["seller_uid_right"])
        ):
            raise ScientificWorldError("Original exact-title endpoint lineage drift")
        source_base = base_title(old_source)
        target_base = base_title(old_target)
        if (
            not source_base
            or not target_base
            or str(public_items[old_target]["title"]) != source_base
        ):
            raise ScientificWorldError("Original exact-title clone cannot be replayed")
        public_items[old_target]["title"] = target_base

    used_items = {
        str(row[field])
        for row in semantic_rows
        for field in ("item_uid_left", "item_uid_right")
    }
    audit_rows: list[dict[str, Any]] = []
    for registration_ordinal, row in enumerate(overrides):
        if row["override_kind"] != "exact_title_clone":
            continue
        asset_index = row["asset_index"]
        if isinstance(asset_index, bool) or not isinstance(asset_index, int):
            raise ScientificWorldError("Exact-title asset index drift")
        source_seller = str(row["seller_uid_left"])
        target_seller = str(row["seller_uid_right"])
        source_candidates = [
            item_uid
            for item_uid, ast in render_asts.items()
            if str(ast["seller_uid"]) == source_seller
            and ast["title_nonempty"] is True
            and item_uid not in used_items
        ]
        target_candidates = [
            item_uid
            for item_uid, ast in render_asts.items()
            if str(ast["seller_uid"]) == target_seller
            and ast["title_nonempty"] is True
            and ast["description_nonempty"] is True
            and item_uid not in used_items
        ]
        if not source_candidates or not target_candidates:
            raise ScientificWorldError(
                "No structurally qualified exact-title clone endpoint"
            )
        source_item = _rank_exact_clone_items(
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            asset_index=asset_index,
            side="source",
            item_uids=source_candidates,
        )[0]
        target_item = _rank_exact_clone_items(
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            asset_index=asset_index,
            side="target",
            item_uids=target_candidates,
        )[0]
        if source_item == target_item:
            raise ScientificWorldError("Exact-title source and target item are equal")
        original_source = str(row["item_uid_left"])
        original_target = str(row["item_uid_right"])
        row["item_uid_left"] = source_item
        row["item_uid_right"] = target_item
        used_items.update((source_item, target_item))
        source_title = base_title(source_item)
        target_title = base_title(target_item)
        if not source_title or not target_title:
            raise ScientificWorldError("Qualified exact-title endpoint lost its title")
        public_items[target_item]["title"] = source_title
        if not str(public_items[target_item]["description"]):
            raise ScientificWorldError(
                "Qualified exact-title target lacks a materialized description"
            )
        audit_rows.append(
            {
                "registration_ordinal": registration_ordinal,
                "asset_index": asset_index,
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "source_seller_uid": source_seller,
                "target_seller_uid": target_seller,
                "original_source_item_uid": original_source,
                "original_target_item_uid": original_target,
                "qualified_source_item_uid": source_item,
                "qualified_target_item_uid": target_item,
                "source_candidate_count": len(source_candidates),
                "target_candidate_count": len(target_candidates),
                "source_title_nonempty": True,
                "target_title_nonempty": True,
                "target_description_nonempty": True,
            }
        )

    endpoint_items = {
        str(row[field])
        for row in overrides
        for field in ("item_uid_left", "item_uid_right")
    }
    if len(endpoint_items) != 12 or len(audit_rows) != 2:
        raise ScientificWorldError("Qualified override endpoint uniqueness drift")
    receipt = {
        "version": "2026-08-11-step28-v13-v1-13-exact-clone-endpoints-v1",
        "selection_domain": EXACT_CLONE_ENDPOINT_DOMAIN.decode("ascii"),
        "world_uid": world_uid,
        "split": split,
        "seller_pairs_or_direction_changed": False,
        "labels_or_model_scores_read": False,
        "shortcut_probe_results_read": False,
        "row_count": len(audit_rows),
        "rows": audit_rows,
        "rows_sha256": common.canonical_sha256(audit_rows),
    }
    private["override_audit"] = overrides
    private["exact_title_clone_endpoint_qualification"] = receipt
    return receipt


def _build_profiles_and_identity33(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    template: Mapping[str, Any],
    world: Mapping[str, Any],
    candidate_only_attributes: Sequence[str] = (),
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    if tuple(candidate_only_attributes) == ():
        processing_policy = policy
        processing_template = template
    elif tuple(candidate_only_attributes) == CANDIDATE_ONLY_ATTRIBUTES:
        candidate_template_path = common.repo_path(CANDIDATE_TEMPLATE_RELATIVE_PATH)
        if common.sha256_file(candidate_template_path) != CANDIDATE_TEMPLATE_SHA256:
            raise ScientificWorldError("Candidate-only template hash drift")
        processing_template = common.load_json(candidate_template_path)
        without_extension = _canonical_clone(processing_template)
        attributes = without_extension["generic_lexicon"]["attributes"]
        if (
            not isinstance(attributes, list)
            or not attributes
            or attributes.pop() != CANDIDATE_ONLY_ATTRIBUTES[0]
        ):
            raise ScientificWorldError("Candidate-only template boundary drift")
        for split_library in without_extension["split_libraries"].values():
            split_library["title_skeletons"] = split_library[
                "title_skeletons"
            ][:8]
            split_library["description_skeletons"] = split_library[
                "description_skeletons"
            ][:8]
        if common.canonical_json_bytes(without_extension) != common.canonical_json_bytes(
            template
        ):
            raise ScientificWorldError("Candidate-only template boundary drift")
        processing_policy = _canonical_clone(policy)
        processing_policy["template_library"]["path"] = (
            CANDIDATE_TEMPLATE_RELATIVE_PATH
        )
        processing_policy["template_library"]["sha256"] = (
            CANDIDATE_TEMPLATE_SHA256
        )
    else:
        raise ScientificWorldError("Candidate-only production domain drift")
    processed = production.process_world(
        processing_policy,
        mode=mode,
        split=split,
        template=processing_template,
        world=world,
    )
    profiles, profile_audit = profiles_module.build_world_profiles(
        policy,
        mode=mode,
        split=split,
        sellers=world["public"]["sellers"],
        items=processed["public"]["profile_safe_items"],
    )
    provenance = stage_parent.build_profile_contribution_provenance(
        world_uid=str(world["public"]["world"]["world_uid"]),
        profiles=profiles,
        profile_safe_items=processed["public"]["profile_safe_items"],
    )
    if (
        profile_audit.get("labels_or_private_structure_read") is not False
        or profile_audit.get("seller_count") != 28
    ):
        raise ScientificWorldError("Seller-profile label-free audit did not close")
    item_index = stage_parent._history_item_index(world)
    history_rows = processed["public"]["history_safe_occurrences"]
    parsed = processed["private"]["parsed_identity_occurrences"]
    attestation = production.build_history_projection_attestation(
        policy,
        mode=mode,
        split=split,
        world_uid=str(world["public"]["world"]["world_uid"]),
        sellers=world["public"]["sellers"],
        items=world["public"]["items"],
        history_safe_occurrences=history_rows,
        history_item_index=item_index,
        parsed_rows=parsed,
        identity_slots_audit=world["private"]["identity_slots_audit"],
        noise_slots_audit=world["private"]["noise_slots_audit"],
        render_asts=world["private"]["render_asts"],
    )
    pair_schema = policy["relational_integrity"]["pair_projection_contract"][
        "complete_model_pair_endpoints_schema"
    ]
    endpoints = [
        {field: row[field] for field in pair_schema}
        for row in world["public"]["complete_model_pair_endpoints"]
    ]
    identity33, audit = history_features.build_identity33_all_pairs(
        policy,
        mode=mode,
        split=split,
        history_safe_occurrences=history_rows,
        history_item_index=item_index,
        projection_attestations=[attestation],
        complete_model_pair_endpoints=endpoints,
    )
    if (
        len(identity33) != 378
        or audit.get("feature_count") != 33
        or audit.get("identity33_sha256") != common.canonical_sha256(identity33)
    ):
        raise ScientificWorldError("Identity33 all-pair projection did not close")
    redacted = processed["public"]["redacted_items"]
    if len(redacted) != len(world["public"]["items"]):
        raise ScientificWorldError("Production redaction item count drift")
    return (
        tuple(_sorted_rows(profiles, "seller_uid")),
        _canonical_clone(provenance),
        tuple(_sorted_rows(identity33, "canonical_pair_uid")),
        tuple(_sorted_rows(redacted, "item_uid")),
    )


def _build_restricted_view(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    world: Mapping[str, Any],
    anonymous_handle_key: bytes,
    structural_parent_sha256: str,
    baseline_identity33_sha256: str,
    capacity_receipt: Mapping[str, Any],
) -> tuple[pure_renderer.RestrictedCandidateView, CandidateBinding]:
    item_rows = _sorted_rows(world["public"]["items"], "item_uid")
    ast_by_item = {
        str(row["item_uid"]): dict(row) for row in world["private"]["render_asts"]
    }
    if len(ast_by_item) != len(world["private"]["render_asts"]):
        raise ScientificWorldError("Render AST contains duplicate item UIDs")
    item_uid_to_handle = {
        str(row["item_uid"]): _anonymous_handle(
            key=anonymous_handle_key,
            kind="item",
            value=str(row["item_uid"]),
        )
        for row in item_rows
    }
    if len(set(item_uid_to_handle.values())) != len(item_uid_to_handle):
        raise ScientificWorldError("Anonymous item-handle collision")
    effective_rows = stage_parent._effective_style_rows(
        policy=policy,
        template=template,
        mode=mode,
        world=world,
    )
    style_by_seller = {
        str(row["seller_uid"]): dict(row["style_factors"])
        for row in effective_rows
    }
    safe_items: list[dict[str, Any]] = []
    for item in item_rows:
        item_uid = str(item["item_uid"])
        ast = ast_by_item.get(item_uid)
        style = style_by_seller.get(str(item["seller_uid"]))
        if ast is None or style is None or set(style) != set(pure_renderer.STYLE_FIELDS):
            raise ScientificWorldError("Restricted item lacks AST or effective style")
        row = {
            "item_handle": item_uid_to_handle[item_uid],
            "code": str(ast["code"]),
            "effective_style": style,
            "title_nonempty": bool(ast["title_nonempty"]),
            "description_nonempty": bool(ast["description_nonempty"]),
            "baseline_category": str(ast["category"]),
            "baseline_product": str(ast["product"]),
            "baseline_attribute": str(ast["attribute"]),
            "baseline_delivery": str(ast["delivery"]),
            "baseline_service": str(ast["service"]),
            "baseline_title_skeleton_index": int(ast["title_skeleton_index"]),
            "baseline_description_skeleton_index": int(
                ast["description_skeleton_index"]
            ),
        }
        if set(row) != set(pure_renderer.ITEM_VIEW_FIELDS):
            raise ScientificWorldError("Restricted item schema drift")
        safe_items.append(row)
    safe_items.sort(key=lambda row: row["item_handle"].encode("utf-8"))

    safe_library = _candidate_safe_library(
        policy=policy,
        template=template,
        fixture=fixture,
        split=split,
    )
    safe_library_sha256 = common.canonical_sha256(safe_library)
    noise_audit_by_slot = {
        str(row["noise_slot_uid"]): dict(row)
        for row in world["private"]["noise_slots_audit"]
    }
    if len(noise_audit_by_slot) != len(world["private"]["noise_slots_audit"]):
        raise ScientificWorldError("Noise audit contains duplicate slots")
    noise_targets: list[dict[str, Any]] = []
    noise_handle_to_slot_uid: dict[str, str] = {}
    for ast in _sorted_rows(world["private"]["render_asts"], "item_uid"):
        slot_uid = str(ast["noise_slot_uid"])
        if not slot_uid:
            continue
        audit = noise_audit_by_slot.get(slot_uid)
        if audit is None:
            raise ScientificWorldError("Noise target lacks its audit row")
        raw_surface = str(audit["raw_surface"])
        matches = [
            (template_index, value_index)
            for template_index in range(len(safe_library["must_ignore_templates"]))
            for value_index in range(len(safe_library["must_ignore_values"]))
            if text_renderer.must_ignore_clause(
                template_index=template_index,
                value=str(safe_library["must_ignore_values"][value_index]),
                template=template,
            )
            == raw_surface
        ]
        if len(matches) != 1:
            raise ScientificWorldError("Noise surface is not uniquely reversible")
        noise_handle = _anonymous_handle(
            key=anonymous_handle_key,
            kind="noise",
            value=slot_uid,
        )
        if noise_handle in noise_handle_to_slot_uid:
            raise ScientificWorldError("Anonymous noise-handle collision")
        noise_handle_to_slot_uid[noise_handle] = slot_uid
        template_index, value_index = matches[0]
        noise_targets.append(
            {
                "noise_handle": noise_handle,
                "item_handle": item_uid_to_handle[str(ast["item_uid"])],
                "baseline_template_index": template_index,
                "baseline_value_index": value_index,
            }
        )
    noise_targets.sort(key=lambda row: row["noise_handle"].encode("utf-8"))
    view_value = {
        "version": pure_renderer.VIEW_VERSION,
        "item_count": len(safe_items),
        "items": safe_items,
        "noise_targets": noise_targets,
        "safe_library": safe_library,
        "safe_library_sha256": safe_library_sha256,
    }
    pure_renderer.validate_restricted_view(view_value)
    view_bytes = common.canonical_json_bytes(view_value)
    view = pure_renderer.RestrictedCandidateView(
        view_bytes=view_bytes,
        view_sha256=hashlib.sha256(view_bytes).hexdigest(),
    )

    overrides: list[dict[str, Any]] = []
    for ordinal, source in enumerate(world["private"]["override_audit"]):
        row = {
            "registration_ordinal": ordinal,
            "override_kind": str(source["override_kind"]),
            "asset_index": int(source["asset_index"]),
            "canonical_pair_uid": str(source["canonical_pair_uid"]),
            "seller_uid_left": str(source["seller_uid_left"]),
            "seller_uid_right": str(source["seller_uid_right"]),
            "item_uid_left": str(source["item_uid_left"]),
            "item_uid_right": str(source["item_uid_right"]),
        }
        if row["override_kind"] not in {
            "high_semantic_similarity",
            "exact_title_clone",
        }:
            raise ScientificWorldError("Unknown registered override")
        overrides.append(row)
    binding = CandidateBinding(
        item_handle_to_uid={
            handle: uid
            for uid, handle in sorted(
                item_uid_to_handle.items(), key=lambda pair: pair[1].encode("utf-8")
            )
        },
        noise_handle_to_slot_uid={
            handle: noise_handle_to_slot_uid[handle]
            for handle in sorted(noise_handle_to_slot_uid)
        },
        registered_overrides=tuple(overrides),
        structural_parent_sha256=structural_parent_sha256,
        baseline_identity33_sha256=baseline_identity33_sha256,
        capacity_receipt=_canonical_clone(dict(capacity_receipt)),
    )
    return view, binding


def _audit_document_capacity(
    *,
    world: Mapping[str, Any],
    redacted_items: Sequence[Mapping[str, Any]],
    seller_profiles: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    ast_by_item = {
        str(row["item_uid"]): row for row in world["private"]["render_asts"]
    }
    redacted_by_item = {
        str(row["item_uid"]): row for row in redacted_items
    }
    if (
        len(ast_by_item) != len(world["private"]["render_asts"])
        or len(redacted_by_item) != len(redacted_items)
        or set(ast_by_item) != set(redacted_by_item)
    ):
        raise ScientificWorldError("Capacity audit item universe drift")
    code_by_item = {item_uid: str(row["code"]) for item_uid, row in ast_by_item.items()}
    if (
        any(document_capacity.CODE_RE.fullmatch(code) is None for code in code_by_item.values())
        or len(set(code_by_item.values())) != len(code_by_item)
    ):
        raise ScientificWorldError("Capacity audit item codes are malformed or duplicated")
    seller_by_item = {
        str(row["item_uid"]): str(row["seller_uid"])
        for row in world["public"]["items"]
    }
    if set(seller_by_item) != set(code_by_item):
        raise ScientificWorldError("Capacity audit item ownership drift")
    all_codes = set(code_by_item.values())
    allowed_foreign_by_item: defaultdict[str, set[str]] = defaultdict(set)
    clone_targets: dict[str, str] = {}
    for row in world["private"]["override_audit"]:
        if row["override_kind"] != "exact_title_clone":
            continue
        source_uid = str(row["item_uid_left"])
        target_uid = str(row["item_uid_right"])
        allowed_foreign_by_item[target_uid].add(code_by_item[source_uid])
        clone_targets[target_uid] = source_uid

    item_rows: list[dict[str, Any]] = []
    for item_uid in sorted(redacted_by_item, key=lambda value: value.encode("utf-8")):
        row = redacted_by_item[item_uid]
        title = str(row["title"])
        description = str(row["description"])
        own_code = code_by_item[item_uid]
        if own_code not in title + description:
            raise ScientificWorldError("An item lost its own capacity code")
        visible_codes = set(document_capacity.CODE_TOKEN_RE.findall(title + description))
        if not visible_codes <= all_codes:
            raise ScientificWorldError("An unregistered code-shaped token became visible")
        foreign_codes = visible_codes - {own_code}
        if not foreign_codes <= allowed_foreign_by_item[item_uid]:
            raise ScientificWorldError("An unregistered foreign item code became visible")
        if foreign_codes & set(document_capacity.CODE_TOKEN_RE.findall(description)):
            raise ScientificWorldError("A clone foreign code escaped into description")
        if any(code not in title for code in foreign_codes):
            raise ScientificWorldError("A clone foreign code is outside copied title")
        if item_uid in clone_targets:
            source_uid = clone_targets[item_uid]
            source_title = str(redacted_by_item[source_uid]["title"])
            if title != source_title or own_code not in description:
                raise ScientificWorldError("Exact-title clone capacity closure failed")
        item_rows.append(
            {
                "item_uid": item_uid,
                "seller_uid": seller_by_item[item_uid],
                "own_code_visible": True,
                "foreign_code_count": len(foreign_codes),
                "registered_clone_foreign_only": bool(foreign_codes),
            }
        )

    profile_by_seller = {
        str(row["seller_uid"]): row for row in seller_profiles
    }
    seller_uids = {str(row["seller_uid"]) for row in world["public"]["sellers"]}
    if len(profile_by_seller) != len(seller_profiles) or set(profile_by_seller) != seller_uids:
        raise ScientificWorldError("Capacity audit seller-profile universe drift")
    seller_rows: list[dict[str, Any]] = []
    for seller_uid in sorted(seller_uids, key=lambda value: value.encode("utf-8")):
        owned_codes = {
            code_by_item[item_uid]
            for item_uid, owner in seller_by_item.items()
            if owner == seller_uid
        }
        description_concat = str(profile_by_seller[seller_uid]["description_concat_top"])
        visible_codes = set(
            document_capacity.CODE_TOKEN_RE.findall(description_concat)
        )
        if not visible_codes <= all_codes:
            raise ScientificWorldError(
                "Seller description contains an unregistered code-shaped token"
            )
        owned_visible = visible_codes & owned_codes
        foreign_visible = visible_codes - owned_codes
        if not owned_visible or foreign_visible:
            raise ScientificWorldError(
                "Seller description profile lacks an exclusive owned code"
            )
        seller_rows.append(
            {
                "seller_uid": seller_uid,
                "owned_description_code_count": len(owned_visible),
                "foreign_description_code_count": 0,
            }
        )
    audit = {
        "version": document_capacity.VERSION,
        "item_count": len(item_rows),
        "unique_code_count": len(all_codes),
        "seller_count": len(seller_rows),
        "all_items_retain_own_code": True,
        "foreign_codes_are_registered_clone_titles_only": True,
        "all_seller_descriptions_retain_exclusive_owned_code": True,
        "item_rows_sha256": common.canonical_sha256(item_rows),
        "seller_rows_sha256": common.canonical_sha256(seller_rows),
        "labels_controllers_or_collision_registries_read": False,
    }
    return tuple(sorted(all_codes)), audit


def _assert_prior_item_codes_absent(
    *, observation: CandidateObservation, prior_item_codes: set[str]
) -> None:
    if not prior_item_codes:
        return
    visible_texts = [
        str(row[field])
        for row in observation.redacted_items
        for field in ("title", "description")
    ]
    visible_texts.extend(
        str(row[field])
        for row in observation.seller_profiles
        for field in (
            "category_concat_top",
            "signature_title_concat",
            "title_concat_top",
            "signature_description_concat",
            "description_concat_top",
        )
    )
    visible_codes = {
        code
        for text in visible_texts
        for code in document_capacity.CODE_TOKEN_RE.findall(text)
    }
    if visible_codes & prior_item_codes:
        raise ScientificWorldError("A prior-world item code leaked into candidate text")


def _assemble_candidate(
    *,
    policy: Mapping[str, Any],
    mode: str,
    split: str,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    baseline_world: Mapping[str, Any],
    baseline_identity33: Sequence[Mapping[str, Any]],
    view: pure_renderer.RestrictedCandidateView,
    binding: CandidateBinding,
    natural: pure_renderer.NaturalExpressionCandidate,
) -> CandidateObservation:
    if natural.view_sha256 != view.view_sha256:
        raise ScientificWorldError("Natural output is bound to the wrong view")
    value = natural.thaw()
    candidate_rows = {
        str(row["item_handle"]): dict(row) for row in value["items"]
    }
    if (
        len(candidate_rows) != len(value["items"])
        or set(candidate_rows) != set(binding.item_handle_to_uid)
    ):
        raise ScientificWorldError("Natural candidate item keyset drift")

    world = copy.deepcopy(dict(baseline_world))
    public_item_by_uid = {
        str(row["item_uid"]): row for row in world["public"]["items"]
    }
    ast_by_uid = {
        str(row["item_uid"]): row for row in world["private"]["render_asts"]
    }
    uid_to_handle = {uid: handle for handle, uid in binding.item_handle_to_uid.items()}
    seen_override_assets: set[tuple[str, int]] = set()
    used_override_items: set[str] = set()
    clone_target_base_descriptions: dict[str, str] = {}
    for expected_ordinal, override in enumerate(binding.registered_overrides):
        kind = override["override_kind"]
        asset_index = override["asset_index"]
        left_uid = override["item_uid_left"]
        right_uid = override["item_uid_right"]
        key = (kind, asset_index)
        if (
            override["registration_ordinal"] != expected_ordinal
            or key in seen_override_assets
            or left_uid == right_uid
            or left_uid in used_override_items
            or right_uid in used_override_items
            or left_uid not in uid_to_handle
            or right_uid not in uid_to_handle
            or public_item_by_uid[left_uid]["seller_uid"] != override["seller_uid_left"]
            or public_item_by_uid[right_uid]["seller_uid"] != override["seller_uid_right"]
        ):
            raise ScientificWorldError("Registered override lineage drift")
        seen_override_assets.add(key)
        used_override_items.update((left_uid, right_uid))
        left = candidate_rows[uid_to_handle[left_uid]]
        right = candidate_rows[uid_to_handle[right_uid]]
        if kind == "high_semantic_similarity":
            if (
                left["category"] != right["category"]
                or left["product"] != right["product"]
                or left["attribute"] != right["attribute"]
                or left["title_skeleton_index"] == right["title_skeleton_index"]
            ):
                raise ScientificWorldError("High-semantic override changed")
        elif kind == "exact_title_clone":
            if (
                not left["title"]
                or not right["title"]
                or not right["base_description"]
                or ast_by_uid[right_uid]["description_nonempty"] is not True
            ):
                raise ScientificWorldError("Exact-title clone endpoint is unqualified")
            clone_target_base_descriptions[right_uid] = str(right["base_description"])
            right["title"] = left["title"]
        else:
            raise ScientificWorldError("Unknown registered override")

    identity_by_item: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in world["private"]["identity_slots_audit"]:
        identity_by_item[str(row["item_uid"])].append(dict(row))
    noise_slot_to_handle = {
        slot: handle for handle, slot in binding.noise_handle_to_slot_uid.items()
    }
    view_value = view.thaw()
    noise_item_to_handle = {
        str(row["item_handle"]): str(row["noise_handle"])
        for row in view_value["noise_targets"]
    }
    role_to_family = policy["identity_design"]["role_to_template_family"]
    items_by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    noise_records_by_item: dict[str, dict[str, Any]] = {}
    private_uid_literals = {
        str(world["public"]["world"]["world_uid"]),
        *public_item_by_uid,
        *(str(row["seller_uid"]) for row in world["public"]["sellers"]),
        *(
            str(row["controller_uid"])
            for row in world["private"]["controller_membership"]
        ),
        *(
            str(row["canonical_pair_uid"])
            for row in world["public"]["complete_model_pair_endpoints"]
        ),
    }
    for handle, item_uid in binding.item_handle_to_uid.items():
        candidate = candidate_rows[handle]
        public_item = public_item_by_uid[item_uid]
        ast = ast_by_uid[item_uid]
        for field in (
            "category",
            "product",
            "attribute",
            "delivery",
            "service",
            "title_skeleton_index",
            "description_skeleton_index",
        ):
            ast[field] = candidate[field]
        public_item["category"] = candidate["category"]
        public_item["title"] = candidate["title"]
        slots = []
        for source_slot in sorted(
            identity_by_item[item_uid],
            key=lambda row: str(row["slot_uid"]).encode("utf-8"),
        ):
            family = role_to_family[str(source_slot["planned_role"])]
            expected_clause = text_renderer.identity_clause(
                template_family=str(family),
                identity_type=str(source_slot["identity_type"]),
                normalized_value=str(source_slot["raw_surface"]),
                template=template,
            )
            if expected_clause.count(str(source_slot["raw_surface"])) != 1:
                raise ScientificWorldError("Identity clause no longer round-trips")
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
        item_state = {
            "world_uid": str(public_item["world_uid"]),
            "seller_uid": str(public_item["seller_uid"]),
            "item_uid": item_uid,
            "time_bucket": int(public_item["time_bucket"]),
            "title_nonempty": bool(ast["title_nonempty"]),
            "description_nonempty": bool(ast["description_nonempty"]),
            "base_description": str(candidate["base_description"]),
            "noise_clause": str(candidate["noise_clause"]),
            "identity_slots": slots,
        }
        items_by_seller[str(public_item["seller_uid"])].append(item_state)
        noise_handle = noise_item_to_handle.get(handle)
        if noise_handle is not None:
            slot_uid = binding.noise_handle_to_slot_uid[noise_handle]
            if slot_uid != str(ast["noise_slot_uid"]):
                raise ScientificWorldError("Noise target changed")
            noise_records_by_item[item_uid] = {
                "noise_slot_uid": slot_uid,
                "raw_surface": str(candidate["noise_clause"]),
            }
        elif candidate["noise_clause"] or ast["noise_slot_uid"]:
            raise ScientificWorldError("Noise lineage is incomplete")
        visible_without_identity = (
            str(candidate["title"])
            + str(candidate["base_description"])
            + str(candidate["noise_clause"])
        )
        if any(uid and uid in visible_without_identity for uid in private_uid_literals):
            raise ScientificWorldError("Private UID leaked into candidate text")

    identity_audit, identity_edit, noise_audit = world_builder._render_identity_slots(
        policy=policy,
        template=template,
        fixture=fixture,
        items_by_seller=items_by_seller,
        noise_records_by_item=noise_records_by_item,
    )
    descriptions = {
        str(item["item_uid"]): str(item["description"])
        for rows in items_by_seller.values()
        for item in rows
    }
    if set(descriptions) != set(public_item_by_uid):
        raise ScientificWorldError("Candidate descriptions do not cover all items")
    for item_uid, description in descriptions.items():
        public_item_by_uid[item_uid]["description"] = description
    if any(
        not descriptions[item_uid] or not base_description
        for item_uid, base_description in clone_target_base_descriptions.items()
    ):
        raise ScientificWorldError("Exact-title clone target lost its description")
    world["private"]["identity_slots_audit"] = identity_audit
    world["private"]["identity_slots_edit"] = identity_edit
    world["private"]["noise_slots_audit"] = noise_audit

    structural_sha = common.canonical_sha256(_structural_parent_projection(world))
    if structural_sha != binding.structural_parent_sha256:
        raise ScientificWorldError("Natural candidate changed its structural parent")
    profiles, provenance, identity33, redacted = _build_profiles_and_identity33(
        policy=policy,
        mode=mode,
        split=split,
        template=template,
        world=world,
        candidate_only_attributes=CANDIDATE_ONLY_ATTRIBUTES,
    )
    provenance_sha = common.canonical_sha256(provenance)
    identity33_sha = common.canonical_sha256(identity33)
    provenance_source_multiset_sha = _profile_provenance_source_multiset_sha256(
        provenance
    )
    if (
        identity33_sha != binding.baseline_identity33_sha256
        or common.canonical_json_bytes(identity33)
        != common.canonical_json_bytes(baseline_identity33)
    ):
        raise ScientificWorldError("Natural candidate changed Identity33")
    item_hashes = tuple(
        collision.item_document_hash(
            title=str(row["title"]), description=str(row["description"])
        )
        for row in redacted
    )
    seller_hashes = tuple(collision.seller_document_hash(row) for row in profiles)
    item_codes, capacity_audit = _audit_document_capacity(
        world=world,
        redacted_items=redacted,
        seller_profiles=profiles,
    )
    return CandidateObservation(
        world=_canonical_clone(world),
        redacted_items=redacted,
        seller_profiles=profiles,
        identity33=identity33,
        profile_provenance_sha256=provenance_sha,
        profile_provenance_source_multiset_sha256=(
            provenance_source_multiset_sha
        ),
        item_document_hashes=item_hashes,
        seller_document_hashes=seller_hashes,
        item_codes=item_codes,
        document_capacity_audit=capacity_audit,
        natural_output_sha256=natural.output_sha256,
    )


def _collision_categories(
    *,
    item_hashes: Sequence[str],
    seller_hashes: Sequence[str],
    historical_item_hashes: frozenset[str],
    historical_seller_hashes: frozenset[str],
    current_item_hashes: set[str],
    current_seller_hashes: set[str],
) -> tuple[str, ...]:
    item_set = set(item_hashes)
    seller_set = set(seller_hashes)
    hits = {
        "same_world_item_document": len(item_set) != len(item_hashes),
        "same_world_seller_document": len(seller_set) != len(seller_hashes),
        "historical_item_document": bool(item_set & historical_item_hashes),
        "historical_seller_document": bool(seller_set & historical_seller_hashes),
        "current_dataset_item_document": bool(item_set & current_item_hashes),
        "current_dataset_seller_document": bool(seller_set & current_seller_hashes),
    }
    return tuple(name for name in COLLISION_CATEGORIES if hits[name])


def _build_private_truth(
    world: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Project labels only after document selection has irrevocably completed."""

    membership = _sorted_rows(
        world["private"]["controller_membership"], "controller_uid", "seller_uid"
    )
    seller_controller = {
        str(row["seller_uid"]): str(row["controller_uid"]) for row in membership
    }
    if len(seller_controller) != 28:
        raise ScientificWorldError("Private controller membership is incomplete")
    pair_labels = []
    positive_count = 0
    for endpoint in _sorted_rows(
        world["public"]["complete_model_pair_endpoints"], "canonical_pair_uid"
    ):
        left = str(endpoint["seller_uid_left"])
        right = str(endpoint["seller_uid_right"])
        label = int(seller_controller[left] == seller_controller[right])
        positive_count += label
        pair_labels.append(
            {
                "canonical_pair_uid": str(endpoint["canonical_pair_uid"]),
                "world_uid": str(endpoint["world_uid"]),
                "label": label,
            }
        )
    if len(pair_labels) != 378 or positive_count != 20:
        raise ScientificWorldError("Private pair truth cardinality drift")
    controller_members: defaultdict[str, list[str]] = defaultdict(list)
    for seller_uid, controller_uid in seller_controller.items():
        controller_members[controller_uid].append(seller_uid)
    qrels = []
    world_uid = str(world["public"]["world"]["world_uid"])
    for seller_uid in common.utf8_sort(seller_controller):
        relevant = common.utf8_sort(
            value
            for value in controller_members[seller_controller[seller_uid]]
            if value != seller_uid
        )
        if len(relevant) not in {1, 2}:
            raise ScientificWorldError("Query relevant-set size drift")
        qrels.append(
            {
                "world_uid": world_uid,
                "query_uid": common.query_uid(world_uid, seller_uid),
                "query_seller_uid": seller_uid,
                "relevant_seller_uids": relevant,
            }
        )
    return tuple(membership), tuple(pair_labels), tuple(qrels)


def build_scientific_world(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    style_profile: Mapping[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    structure_key_hex: str,
    document_variation_key: bytes,
    anonymous_handle_key: bytes,
    historical_item_hashes: frozenset[str],
    historical_seller_hashes: frozenset[str],
    historical_identity_hashes: frozenset[str],
    current_item_hashes: set[str],
    current_seller_hashes: set[str],
    current_identity_hashes: set[str],
    current_item_codes: set[str],
    candidate_limit: int = CANDIDATE_LIMIT,
    identity_maximum_counter: int = 128,
) -> AcceptedScientificWorld:
    """Build one world; caller-owned registries mutate only after acceptance."""

    if candidate_limit != CANDIDATE_LIMIT:
        raise ScientificWorldError("Candidate limit must remain frozen at 32")
    if current_identity_hashes & set(historical_identity_hashes):
        raise ScientificWorldError(
            "Current identity registry intersects historical exclusions"
        )
    split = str(world_record["split"])
    baseline = world_builder.build_world(
        policy=_canonical_clone(policy),
        template=_canonical_clone(template),
        fixture=_canonical_clone(fixture),
        style_profile=_canonical_clone(style_profile),
        mode=mode,
        world_record=_canonical_clone(world_record),
        structure_key_hex=structure_key_hex,
    )
    _qualify_exact_title_clone_endpoints(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        structure_key_hex=structure_key_hex,
        world=baseline,
    )
    trial_identity = set(current_identity_hashes)
    remapped, allocation_receipt = identity_remap.remap_world_identity_values(
        baseline,
        template=template,
        key_hex=str(policy["randomness"][mode]["identity_value_key_hex"]),
        historical_forbidden=historical_identity_hashes,
        allocated_in_trial=trial_identity,
        maximum_counter=identity_maximum_counter,
    )
    identity_delta = tuple(common.utf8_sort(trial_identity - current_identity_hashes))
    asset_hashes = tuple(
        common.utf8_sort(
            identity_values.value_hash(str(row["identity_value"]))
            for row in remapped["private"]["identity_assets"]
        )
    )
    if (
        not identity_delta
        or identity_delta != asset_hashes
        or set(identity_delta) & historical_identity_hashes
        or set(identity_delta) & current_identity_hashes
    ):
        raise ScientificWorldError("Identity allocation delta did not close")

    _profiles, _provenance, identity33, _redacted = _build_profiles_and_identity33(
        policy=policy,
        mode=mode,
        split=split,
        template=template,
        world=remapped,
    )
    capacity_parent, capacity_receipt = document_capacity.apply_capacity_parent(
        policy=policy,
        mode=mode,
        world_record=world_record,
        document_variation_key=document_variation_key,
        world=remapped,
    )
    structural_sha = common.canonical_sha256(
        _structural_parent_projection(capacity_parent)
    )
    identity33_sha = common.canonical_sha256(identity33)
    parent_codes = {
        str(row["code"]) for row in capacity_parent["private"]["render_asts"]
    }
    if len(parent_codes) != len(capacity_parent["private"]["render_asts"]):
        raise ScientificWorldError("V9 parent contains duplicate item codes")
    view, binding = _build_restricted_view(
        policy=policy,
        mode=mode,
        split=split,
        template=template,
        fixture=fixture,
        world=capacity_parent,
        anonymous_handle_key=anonymous_handle_key,
        structural_parent_sha256=structural_sha,
        baseline_identity33_sha256=identity33_sha,
        capacity_receipt=capacity_receipt,
    )
    rejection_counts = {name: 0 for name in COLLISION_CATEGORIES}
    accepted: CandidateObservation | None = None
    accepted_index: int | None = None
    candidate_zero_lineage_reference: str | None = None
    world_uid = str(world_record["world_uid"])
    for candidate_index in range(candidate_limit):
        candidate_key = collision.derive_candidate_key(
            document_variation_key,
            split=split,
            world_uid=world_uid,
            candidate_index=candidate_index,
        )
        natural = pure_renderer.render_candidate_natural_expressions(
            restricted_view=view,
            candidate_key=candidate_key,
        )
        observation = _assemble_candidate(
            policy=policy,
            mode=mode,
            split=split,
            template=template,
            fixture=fixture,
            baseline_world=capacity_parent,
            baseline_identity33=identity33,
            view=view,
            binding=binding,
            natural=natural,
        )
        if tuple(observation.item_codes) != tuple(sorted(parent_codes)):
            raise ScientificWorldError("Candidate item-code registry drift")
        if candidate_index == 0:
            candidate_zero_lineage_reference = (
                observation.profile_provenance_source_multiset_sha256
            )
            if (
                current_item_hashes & set(historical_item_hashes)
                or current_seller_hashes & set(historical_seller_hashes)
            ):
                raise ScientificWorldError(
                    "Current document registries intersect historical exclusions"
                )
            if parent_codes & current_item_codes:
                raise ScientificWorldError("V9 item-code registry collision")
        elif (
            candidate_zero_lineage_reference is None
            or observation.profile_provenance_source_multiset_sha256
            != candidate_zero_lineage_reference
        ):
            raise ScientificWorldError(
                "Candidate changed the candidate-zero lineage reference"
            )
        _assert_prior_item_codes_absent(
            observation=observation,
            prior_item_codes=current_item_codes,
        )
        categories = _collision_categories(
            item_hashes=observation.item_document_hashes,
            seller_hashes=observation.seller_document_hashes,
            historical_item_hashes=historical_item_hashes,
            historical_seller_hashes=historical_seller_hashes,
            current_item_hashes=current_item_hashes,
            current_seller_hashes=current_seller_hashes,
        )
        if categories:
            for category in categories:
                rejection_counts[category] += 1
            continue
        accepted = observation
        accepted_index = candidate_index
        break
    if (
        accepted is None
        or accepted_index is None
        or candidate_zero_lineage_reference is None
    ):
        raise ScientificWorldError("All 32 exact-document candidates collided")

    if (
        len(accepted.item_document_hashes)
        != len(set(accepted.item_document_hashes))
        or len(accepted.seller_document_hashes)
        != len(set(accepted.seller_document_hashes))
    ):
        raise ScientificWorldError("Accepted document multiplicity drift")
    candidate_template_path = common.repo_path(CANDIDATE_TEMPLATE_RELATIVE_PATH)
    if common.sha256_file(candidate_template_path) != CANDIDATE_TEMPLATE_SHA256:
        raise ScientificWorldError("Candidate-only template hash drift before views")
    processing_template = common.load_json(candidate_template_path)
    processing_policy = _canonical_clone(policy)
    processing_policy["template_library"]["path"] = CANDIDATE_TEMPLATE_RELATIVE_PATH
    processing_policy["template_library"]["sha256"] = CANDIDATE_TEMPLATE_SHA256
    safe_library = _candidate_safe_library(
        policy=policy,
        template=template,
        fixture=fixture,
        split=split,
    )
    effective_style_rows = stage_parent._effective_style_rows(
        policy=policy,
        template=template,
        mode=mode,
        world=accepted.world,
    )
    effective_styles = {
        str(row["seller_uid"]): dict(row["style_factors"])
        for row in effective_style_rows
    }
    private_projection = accepted.world["private"]
    materialized_views = channel_materializer.materialize_label_free_channel_views(
        processing_policy=processing_policy,
        profile_policy=policy,
        mode=mode,
        split=split,
        processing_template=processing_template,
        safe_library=safe_library,
        fixture=fixture,
        world_uid=world_uid,
        public_sellers=accepted.world["public"]["sellers"],
        public_items=accepted.world["public"]["items"],
        complete_model_pair_endpoints=accepted.world["public"][
            "complete_model_pair_endpoints"
        ],
        render_asts=private_projection["render_asts"],
        identity_slots_audit=private_projection["identity_slots_audit"],
        noise_slots_audit=private_projection["noise_slots_audit"],
        override_audit=private_projection["override_audit"],
        effective_styles=effective_styles,
        full_redacted_items=accepted.redacted_items,
        full_seller_profiles=accepted.seller_profiles,
    )
    controller_membership, pair_labels, qrels = _build_private_truth(accepted.world)
    current_identity_hashes.update(identity_delta)
    current_item_codes.update(accepted.item_codes)
    current_item_hashes.update(accepted.item_document_hashes)
    current_seller_hashes.update(accepted.seller_document_hashes)
    return AcceptedScientificWorld(
        split=split,
        split_ordinal=int(world_record["split_ordinal"]),
        world_uid=world_uid,
        candidate_index=accepted_index,
        candidates_examined=accepted_index + 1,
        rejection_counts=rejection_counts,
        world=accepted.world,
        redacted_items=accepted.redacted_items,
        seller_profiles=accepted.seller_profiles,
        masked_redacted_items=materialized_views.masked_redacted_items,
        neutral_redacted_items=materialized_views.neutral_redacted_items,
        masked_seller_profiles=materialized_views.masked_seller_profiles,
        neutral_seller_profiles=materialized_views.neutral_seller_profiles,
        public_code_probe_input=materialized_views.public_code_probe_input,
        text_probe_eligibility_input=(
            materialized_views.text_probe_eligibility_input
        ),
        channel_structure_audit=materialized_views.channel_structure_audit,
        identity33=accepted.identity33,
        controller_membership=controller_membership,
        pair_labels=pair_labels,
        qrels=qrels,
        identity_allocation_receipt=_canonical_clone(allocation_receipt),
        identity_registry_delta=identity_delta,
        code_registry_delta=accepted.item_codes,
        item_registry_delta=tuple(sorted(accepted.item_document_hashes)),
        seller_registry_delta=tuple(sorted(accepted.seller_document_hashes)),
        structural_parent_sha256=structural_sha,
        candidate_zero_lineage_reference_sha256=(
            candidate_zero_lineage_reference
        ),
        document_capacity_receipt=_canonical_clone(capacity_receipt),
        document_capacity_audit={
            **_canonical_clone(accepted.document_capacity_audit),
            "prior_item_code_registry_count": len(current_item_codes)
            - len(accepted.item_codes),
            "prior_item_code_intersection_zero": True,
        },
        profile_provenance_sha256=accepted.profile_provenance_sha256,
        identity33_sha256=common.canonical_sha256(accepted.identity33),
        natural_output_sha256=accepted.natural_output_sha256,
    )
