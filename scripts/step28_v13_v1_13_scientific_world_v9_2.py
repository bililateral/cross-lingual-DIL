#!/usr/bin/env python3
"""V9.2 call layer adding one pre-truth style-deranged full surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_counterfactual_text_v9_2 as counterfactual
import step28_v13_v1_13_scientific_world_v9 as v9


VERSION = "2026-08-23-step28-v13-v1-13-scientific-world-v9-2"
PERSISTED_STRUCTURE_VERSION = (
    "2026-08-23-step28-v13-v1-13-quality-channel-structure-v9-2"
)
M1_REPEAT_IDS = ("r01", "r02", "r03", "r04", "r05")
M1_MAPPING_DOMAIN = "step28-v13-v1.13-m1"


@dataclass(frozen=True)
class AcceptedScientificWorldV92:
    base: v9.AcceptedScientificWorld
    counterfactual_redacted_items: tuple[dict[str, Any], ...]
    counterfactual_seller_profiles: tuple[dict[str, Any], ...]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def build_m1_mapping_commitments(
    endpoints: Sequence[Mapping[str, Any]], *, world_uid: str
) -> tuple[dict[str, str], ...]:
    """Commit the five public-ID-only endpoint-disjoint M1 permutations."""

    rows = tuple(endpoints)
    if len(rows) != 378:
        raise v9.ScientificWorldError("M1 commitment requires all 378 pairs")
    pair_by_endpoints: dict[tuple[str, str], tuple[str, str]] = {}
    seller_uids: set[str] = set()
    for row in rows:
        if str(row.get("world_uid", "")) != world_uid:
            raise v9.ScientificWorldError("M1 endpoint world binding drift")
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        ordered = tuple(sorted((left, right), key=lambda value: value.encode("utf-8")))
        if left == right or ordered in pair_by_endpoints:
            raise v9.ScientificWorldError("M1 endpoint universe is not a simple graph")
        pair_uid_hex = (ordered[0].encode("utf-8") + b"\x00" + ordered[1].encode("utf-8")).hex()
        pair_by_endpoints[ordered] = (str(row["canonical_pair_uid"]), pair_uid_hex)
        seller_uids.update(ordered)
    sellers = tuple(sorted(seller_uids, key=lambda value: value.encode("utf-8")))
    if len(sellers) != 28 or len(pair_by_endpoints) != 28 * 27 // 2:
        raise v9.ScientificWorldError("M1 endpoint universe is not K28")

    factor_edges: list[tuple[tuple[str, str], ...]] = []
    for factor_index in range(27):
        edges = [(sellers[27], sellers[factor_index])]
        for offset in range(1, 14):
            edges.append(
                (
                    sellers[(factor_index + offset) % 27],
                    sellers[(factor_index - offset) % 27],
                )
            )
        normalized = tuple(
            tuple(sorted(edge, key=lambda value: value.encode("utf-8")))
            for edge in edges
        )
        if len(normalized) != 14 or len(set(normalized)) != 14:
            raise v9.ScientificWorldError("M1 one-factor construction drift")
        factor_edges.append(normalized)
    if len({edge for factor in factor_edges for edge in factor}) != 378:
        raise v9.ScientificWorldError("M1 one-factorization does not cover K28")

    commitments: list[dict[str, str]] = []
    for repeat_id in M1_REPEAT_IDS:
        mapping: list[dict[str, str]] = []
        for factor_index, factor in enumerate(factor_edges):
            ranked = sorted(
                factor,
                key=lambda edge: (
                    hashlib.sha256(
                        common.canonical_json_bytes(
                            [
                                M1_MAPPING_DOMAIN,
                                repeat_id,
                                world_uid,
                                factor_index,
                                pair_by_endpoints[edge][1],
                            ]
                        )
                    ).digest(),
                    pair_by_endpoints[edge][1].encode("utf-8"),
                ),
            )
            for index, target_edge in enumerate(ranked):
                source_edge = ranked[(index + 1) % len(ranked)]
                if set(target_edge) & set(source_edge):
                    raise v9.ScientificWorldError(
                        "M1 source and target endpoints overlap"
                    )
                mapping.append(
                    {
                        "target_pair_uid": pair_by_endpoints[target_edge][0],
                        "source_pair_uid": pair_by_endpoints[source_edge][0],
                    }
                )
        mapping.sort(key=lambda row: row["target_pair_uid"].encode("utf-8"))
        if (
            len(mapping) != 378
            or len({row["target_pair_uid"] for row in mapping}) != 378
            or len({row["source_pair_uid"] for row in mapping}) != 378
            or any(row["target_pair_uid"] == row["source_pair_uid"] for row in mapping)
        ):
            raise v9.ScientificWorldError("M1 mapping bijection closure drift")
        commitments.append(
            {
                "repeat_id": repeat_id,
                "mapping_sha256": common.canonical_sha256(mapping),
            }
        )
    if len({row["mapping_sha256"] for row in commitments}) != 5:
        raise v9.ScientificWorldError("Five M1 mapping commitments are not distinct")
    return tuple(commitments)


def _v9_2_structure_audit(
    *,
    base_audit: Mapping[str, Any],
    counterfactual_surface: counterfactual.CounterfactualFullSurface,
    original_identity33: tuple[dict[str, Any], ...],
    text_probe_eligibility_input: tuple[dict[str, Any], ...],
    identity_mechanism_sha256: str,
    m1_mapping_commitments: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    audit = v9._canonical_clone(dict(base_audit))
    original_version = str(audit.pop("version"))
    audit = {
        "version": PERSISTED_STRUCTURE_VERSION,
        "base_v9_structure_version": original_version,
        **audit,
        "counterfactual_full_item_sha256": common.canonical_sha256(
            counterfactual_surface.redacted_items
        ),
        "counterfactual_full_profile_sha256": (
            v9.channel_materializer._persisted_profile_sha256(
                counterfactual_surface.seller_profiles
            )
        ),
        "counterfactual_replay": v9._canonical_clone(
            counterfactual_surface.audit
        ),
        "shared_identity33_sha256": common.canonical_sha256(original_identity33),
        "shared_text_probe_eligibility_sha256": common.canonical_sha256(
            text_probe_eligibility_input
        ),
        "shared_identity_mechanism_sha256": identity_mechanism_sha256,
        "m1_mapping_commitments": v9._canonical_clone(m1_mapping_commitments),
        "m1_mapping_commitment_bundle_sha256": common.canonical_sha256(
            m1_mapping_commitments
        ),
        "model_input_file_count": 8,
        "original_author_model_input_file_count": 6,
        "counterfactual_model_input_file_count": 2,
        "labels_or_retrieval_truth_materialized_before_audit": False,
    }
    audit["v9_2_extension_sha256"] = common.canonical_sha256(audit)
    return audit


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
    candidate_limit: int = v9.CANDIDATE_LIMIT,
    identity_maximum_counter: int = 128,
) -> AcceptedScientificWorldV92:
    """Build one V9.2 world while withholding pair truth until all eight inputs exist."""

    if candidate_limit != v9.CANDIDATE_LIMIT:
        raise v9.ScientificWorldError("Candidate limit must remain frozen at 32")
    if current_identity_hashes & set(historical_identity_hashes):
        raise v9.ScientificWorldError(
            "Current identity registry intersects historical exclusions"
        )
    split = str(world_record["split"])
    baseline = v9.world_builder.build_world(
        policy=v9._canonical_clone(policy),
        template=v9._canonical_clone(template),
        fixture=v9._canonical_clone(fixture),
        style_profile=v9._canonical_clone(style_profile),
        mode=mode,
        world_record=v9._canonical_clone(world_record),
        structure_key_hex=structure_key_hex,
    )
    v9._qualify_exact_title_clone_endpoints(
        policy=policy,
        template=template,
        mode=mode,
        split=split,
        structure_key_hex=structure_key_hex,
        world=baseline,
    )
    trial_identity = set(current_identity_hashes)
    remapped, allocation_receipt = v9.identity_remap.remap_world_identity_values(
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
            v9.identity_values.value_hash(str(row["identity_value"]))
            for row in remapped["private"]["identity_assets"]
        )
    )
    if (
        not identity_delta
        or identity_delta != asset_hashes
        or set(identity_delta) & historical_identity_hashes
        or set(identity_delta) & current_identity_hashes
    ):
        raise v9.ScientificWorldError("Identity allocation delta did not close")

    _profiles, _provenance, identity33, _redacted = (
        v9._build_profiles_and_identity33(
            policy=policy,
            mode=mode,
            split=split,
            template=template,
            world=remapped,
        )
    )
    capacity_parent, capacity_receipt = v9.document_capacity.apply_capacity_parent(
        policy=policy,
        mode=mode,
        world_record=world_record,
        document_variation_key=document_variation_key,
        world=remapped,
    )
    structural_sha = common.canonical_sha256(
        v9._structural_parent_projection(capacity_parent)
    )
    identity33_sha = common.canonical_sha256(identity33)
    parent_codes = {
        str(row["code"]) for row in capacity_parent["private"]["render_asts"]
    }
    if len(parent_codes) != len(capacity_parent["private"]["render_asts"]):
        raise v9.ScientificWorldError("V9.2 parent contains duplicate item codes")
    view, binding = v9._build_restricted_view(
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
    rejection_counts = {name: 0 for name in v9.COLLISION_CATEGORIES}
    accepted: v9.CandidateObservation | None = None
    accepted_index: int | None = None
    candidate_zero_lineage_reference: str | None = None
    world_uid = str(world_record["world_uid"])
    for candidate_index in range(candidate_limit):
        candidate_key = v9.collision.derive_candidate_key(
            document_variation_key,
            split=split,
            world_uid=world_uid,
            candidate_index=candidate_index,
        )
        natural = v9.pure_renderer.render_candidate_natural_expressions(
            restricted_view=view,
            candidate_key=candidate_key,
        )
        observation = v9._assemble_candidate(
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
            raise v9.ScientificWorldError("Candidate item-code registry drift")
        if candidate_index == 0:
            candidate_zero_lineage_reference = (
                observation.profile_provenance_source_multiset_sha256
            )
            if (
                current_item_hashes & set(historical_item_hashes)
                or current_seller_hashes & set(historical_seller_hashes)
            ):
                raise v9.ScientificWorldError(
                    "Current document registries intersect historical exclusions"
                )
            if parent_codes & current_item_codes:
                raise v9.ScientificWorldError("V9.2 item-code registry collision")
        elif (
            candidate_zero_lineage_reference is None
            or observation.profile_provenance_source_multiset_sha256
            != candidate_zero_lineage_reference
        ):
            raise v9.ScientificWorldError(
                "Candidate changed the candidate-zero lineage reference"
            )
        v9._assert_prior_item_codes_absent(
            observation=observation,
            prior_item_codes=current_item_codes,
        )
        categories = v9._collision_categories(
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
        raise v9.ScientificWorldError("All 32 exact-document candidates collided")
    if (
        len(accepted.item_document_hashes)
        != len(set(accepted.item_document_hashes))
        or len(accepted.seller_document_hashes)
        != len(set(accepted.seller_document_hashes))
    ):
        raise v9.ScientificWorldError("Accepted document multiplicity drift")

    candidate_template_path = common.repo_path(v9.CANDIDATE_TEMPLATE_RELATIVE_PATH)
    if common.sha256_file(candidate_template_path) != v9.CANDIDATE_TEMPLATE_SHA256:
        raise v9.ScientificWorldError("Candidate-only template hash drift before views")
    processing_template = common.load_json(candidate_template_path)
    processing_policy = v9._canonical_clone(policy)
    processing_policy["template_library"]["path"] = (
        v9.CANDIDATE_TEMPLATE_RELATIVE_PATH
    )
    processing_policy["template_library"]["sha256"] = v9.CANDIDATE_TEMPLATE_SHA256
    safe_library = v9._candidate_safe_library(
        policy=policy,
        template=template,
        fixture=fixture,
        split=split,
    )
    effective_style_rows = v9.stage_parent._effective_style_rows(
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
    materialized_views = v9.channel_materializer.materialize_label_free_channel_views(
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
    # V9.2's only semantic insertion: create the fourth surface before the
    # first controller-membership, pair-label, or qrels projection.
    counterfactual_surface = (
        counterfactual.materialize_style_deranged_full_surface(
            profile_policy=policy,
            mode=mode,
            split=split,
            base_template=template,
            safe_library=safe_library,
            fixture=fixture,
            world_uid=world_uid,
            candidate_key=v9.collision.derive_candidate_key(
                document_variation_key,
                split=split,
                world_uid=world_uid,
                candidate_index=accepted_index,
            ),
            public_world=accepted.world["public"]["world"],
            public_sellers=accepted.world["public"]["sellers"],
            public_items=accepted.world["public"]["items"],
            original_redacted_items=accepted.redacted_items,
            original_seller_profiles=accepted.seller_profiles,
            complete_model_pair_endpoints=accepted.world["public"][
                "complete_model_pair_endpoints"
            ],
            render_asts=private_projection["render_asts"],
            identity_slots_audit=private_projection["identity_slots_audit"],
            noise_slots_audit=private_projection["noise_slots_audit"],
            override_audit=private_projection["override_audit"],
            effective_styles=effective_styles,
            baseline_identity33=accepted.identity33,
        )
    )
    structure_audit = _v9_2_structure_audit(
        base_audit=materialized_views.channel_structure_audit,
        counterfactual_surface=counterfactual_surface,
        original_identity33=accepted.identity33,
        text_probe_eligibility_input=materialized_views.text_probe_eligibility_input,
        identity_mechanism_sha256=common.canonical_sha256(
            private_projection["mechanism_assignments"]
        ),
        m1_mapping_commitments=build_m1_mapping_commitments(
            accepted.world["public"]["complete_model_pair_endpoints"],
            world_uid=world_uid,
        ),
    )
    controller_membership, pair_labels, qrels = v9._build_private_truth(accepted.world)

    current_identity_hashes.update(identity_delta)
    current_item_codes.update(accepted.item_codes)
    current_item_hashes.update(accepted.item_document_hashes)
    current_seller_hashes.update(accepted.seller_document_hashes)
    base = v9.AcceptedScientificWorld(
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
        text_probe_eligibility_input=materialized_views.text_probe_eligibility_input,
        channel_structure_audit=structure_audit,
        identity33=accepted.identity33,
        controller_membership=controller_membership,
        pair_labels=pair_labels,
        qrels=qrels,
        identity_allocation_receipt=v9._canonical_clone(allocation_receipt),
        identity_registry_delta=identity_delta,
        code_registry_delta=accepted.item_codes,
        item_registry_delta=tuple(sorted(accepted.item_document_hashes)),
        seller_registry_delta=tuple(sorted(accepted.seller_document_hashes)),
        structural_parent_sha256=structural_sha,
        candidate_zero_lineage_reference_sha256=candidate_zero_lineage_reference,
        document_capacity_receipt=v9._canonical_clone(capacity_receipt),
        document_capacity_audit={
            **v9._canonical_clone(accepted.document_capacity_audit),
            "prior_item_code_registry_count": len(current_item_codes)
            - len(accepted.item_codes),
            "prior_item_code_intersection_zero": True,
        },
        profile_provenance_sha256=accepted.profile_provenance_sha256,
        identity33_sha256=common.canonical_sha256(accepted.identity33),
        natural_output_sha256=accepted.natural_output_sha256,
    )
    return AcceptedScientificWorldV92(
        base=base,
        counterfactual_redacted_items=counterfactual_surface.redacted_items,
        counterfactual_seller_profiles=counterfactual_surface.seller_profiles,
    )
