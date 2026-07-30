#!/usr/bin/env python3
"""Generate immutable Step 28-v13 development-smoke dataset partitions."""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_candidate_sampling as candidate_sampling_mod
import step28_v13_feature_derangement as feature_derangement_mod
import step28_v13_history_features as history_features_mod
import step28_v13_independent_dgp_comparator as independent_comparator
import step28_v13_independent_private_dgp_replay as independent_replay
import step28_v13_integrity_receipts as integrity_receipts_mod
import step28_v13_placebo_support as placebo_support_mod
import step28_v13_producer_dgp_projection as producer_projection_mod
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_mod
import step28_v13_run_parser_template_fixture as parser_fixture_runner
import step28_v13_safe_slots as safe_slots_mod
import step28_v13_smoke_private_regeneration as smoke_regeneration
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder


SPLITS = ("train", "development", "audit_a", "audit_b")
DEFAULT_RELEASE_NAME = "dataset_smoke_v3"


def _validate_fixture_result(
    fixture: Mapping[str, Any],
    *,
    fixture_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Bind generation to the exact successful exhaustive parser fixture."""

    contract = fixture["full_render_context_contract"]
    output_spec = contract["output_manifest"]
    output_path = common.repo_path(str(output_spec["path"]))
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Missing exhaustive parser/template fixture result: {output_path}"
        )
    result = common.load_json(output_path)
    expected_dependencies = (
        parser_fixture_runner._validate_dependencies(fixture)
    )
    expected_gates = {
        "all_split_style_product_attribute_coverage": True,
        "all_title_modifiers_covered": True,
        "base_and_guard_preservation": True,
        "exact_parser_rows_and_flags": True,
        "must_ignore_preservation": True,
        "production_redactor_identity_removal": True,
        "title_only_zero_parser_rows": True,
        "zero_unexpected_parser_rows": True,
    }
    expected_family_counts = {
        "adjacent_ordered_roles": 625,
        "maximum_eight_slots": 704,
        "single_role_full_render": 140800,
        "title_only": 90112,
    }
    if (
        result.get("version")
        != "2026-07-29-step28-v13-parser-template-fixture-result-v3"
        or result.get("status") != output_spec["required_status"]
        or result.get("scientific_metrics_produced") is not False
        or expected_dependencies["fixture"]
        != common.sha256_file(fixture_path)
        or int(result.get("case_count", -1))
        != int(contract["expected_case_count"])
        or int(result.get("expected_full_case_count", -1))
        != int(contract["expected_case_count"])
        or result.get("case_outcome_manifest_sha256")
        != output_spec["expected_case_outcome_manifest_sha256"]
        or result.get("runner_sha256") != expected_dependencies["runner"]
        or int(result.get("effective_style_count", -1))
        != int(contract["expected_reachable_effective_style_count"])
        or result.get("family_counts") != expected_family_counts
        or result.get("gates") != expected_gates
    ):
        raise common.ContractError(
            "Exhaustive parser/template fixture result is not release-qualified"
        )
    dependency_hashes = result.get("dependency_hashes")
    if dependency_hashes != expected_dependencies:
        raise common.ContractError(
            "Exhaustive parser/template fixture dependency binding drift"
        )
    return output_path, result


def _load_release_inputs(
    policy: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    template, fixture = common.validate_policy_release_documents(
        policy,
        mode=mode,
    )
    fixture_path = common.repo_path(
        str(
            policy["identity_design"][
                "role_template_parser_flag_fixture"
            ]["path"]
        )
    )
    _validate_fixture_result(
        fixture,
        fixture_path=fixture_path,
    )
    style_spec = policy["style_reference_boundary"]["generator_release_inputs"][
        "profile"
    ]
    style_path = common.verify_file_pin(
        style_spec, label="synthetic style reference"
    )
    style_profile = common.load_json(style_path)
    common.validate_independent_replay_public_domains(
        policy,
        template=template,
        style_profile=style_profile,
    )
    return template, fixture, style_profile


def _extend_with_world_uid(
    world_uid: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if "world_uid" in row:
            raise common.ContractError("Cannot prepend duplicate world_uid")
        output.append({"world_uid": world_uid, **dict(row)})
    return output


def build_split_payload(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    style_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one complete split in memory before any artifact is written."""

    if mode != "development_smoke":
        raise common.ContractError(
            "This combined generator is development-smoke only; formal "
            "generation requires the split-private custody entrypoint"
        )
    if split not in SPLITS:
        raise common.ContractError(f"Unknown split: {split}")
    integrity_receipts_mod.validate_deployment_contract(policy)
    records = [
        row
        for row in structure.build_mode_world_pool(policy, mode=mode)
        if row["split"] == split
    ]
    expected_world_count = int(policy["modes"][mode]["world_counts"][split])
    if len(records) != expected_world_count:
        raise common.ContractError("Split world-count drift")
    structure_key = common.structure_key_for_split(
        policy, mode=mode, split=split
    )
    public_candidate_policy = (
        candidate_sampling_mod.build_public_candidate_policy(
            policy, mode=mode, split=split
        )
    )
    candidate_key_hex = str(
        policy["randomness"][mode]["candidate_key_hex"]
    )
    candidate_integrity_context = (
        integrity_receipts_mod.build_candidate_integrity_context(
            policy,
            candidate_policy=public_candidate_policy,
            mode=mode,
            split=split,
        )
    )

    payload: dict[str, list[dict[str, Any]]] = {
        "worlds": [],
        "sellers": [],
        "items": [],
        "complete_model_pair_endpoints": [],
        "candidate_pairs": [],
        "candidate_sampling_audit": [],
        "seller_profiles": [],
        "redacted_items": [],
        "history_safe_occurrences": [],
        "history_item_index": [],
        "history_projection_attestations": [],
        "identity33_all_pairs": [],
        "parsed_identity_occurrences": [],
        "identity_slots_audit": [],
        "identity_slots_edit": [],
        "noise_slots_audit": [],
        "render_asts": [],
        "redaction_diagnostics": [],
        "rewire_safe_identity_slots": [],
        "rewire_nuisance_ledger": [],
        "controller_membership": [],
        "controller_style_groups": [],
        "mechanism_assignments": [],
        "identity_assets": [],
        "positive_targets": [],
        "negative_flags": [],
        "override_audit": [],
        "solver_audit": [],
        "producer_typed_dgp_projections": [],
        "independent_typed_dgp_replay_ledgers": [],
        "world_generation_audit": [],
        "redaction_registry_audit": [],
        "dgp_replay_seller_uid_pool": [],
        "dgp_replay_all_item_uid_pool": [],
        "dgp_replay_nonempty_title_item_uid_pool": [],
        "dgp_replay_nonempty_description_item_uid_pool": [],
    }
    world_digests: list[dict[str, str]] = []
    for record in records:
        world_uid = str(record["world_uid"])
        world = world_builder.build_world(
            policy=dict(policy),
            template=dict(template),
            fixture=dict(fixture),
            style_profile=dict(style_profile),
            mode=mode,
            world_record=record,
            structure_key_hex=structure_key,
        )
        observed_uid_pools = independent_comparator.build_observed_uid_pools(
            world_uid=world_uid,
            sellers=world["public"]["sellers"],
            items=world["public"]["items"],
        )
        independent_expected = independent_replay.replay_typed_dgp(
            policy,
            mode=mode,
            split=split,
            world_uid=world_uid,
            structure_key_hex=structure_key,
            **observed_uid_pools,
        )
        producer_projection = producer_projection_mod.project_world(
            world=world,
            mode=mode,
            split=split,
        )
        independent_replay_audit = independent_comparator.compare_typed_dgp(
            expected_replay=independent_expected,
            producer_projection=producer_projection,
        )
        payload["producer_typed_dgp_projections"].append(
            producer_projection
        )
        payload["independent_typed_dgp_replay_ledgers"].append(
            independent_expected
        )
        payload["dgp_replay_seller_uid_pool"].extend(
            {
                "world_uid": world_uid,
                "seller_uid": seller_uid,
            }
            for seller_uid in observed_uid_pools["observed_seller_uids"]
        )
        for source_name, target_name in (
            (
                "observed_all_item_uid_rows",
                "dgp_replay_all_item_uid_pool",
            ),
            (
                "observed_nonempty_title_item_uid_rows",
                "dgp_replay_nonempty_title_item_uid_pool",
            ),
            (
                "observed_nonempty_description_item_uid_rows",
                "dgp_replay_nonempty_description_item_uid_pool",
            ),
        ):
            payload[target_name].extend(
                dict(row) for row in observed_uid_pools[source_name]
            )
        producer_regeneration_audit = (
            smoke_regeneration.validate_producer_regeneration_match(
                policy,
                mode=mode,
                split=split,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                world=world,
            )
        )
        processed = production.process_world(
            policy,
            mode=mode,
            split=split,
            template=template,
            world=world,
        )
        history_item_index = [
            {
                "world_uid": str(row["world_uid"]),
                "seller_uid": str(row["seller_uid"]),
                "item_uid": str(row["item_uid"]),
                "time_bucket": int(row["time_bucket"]),
            }
            for row in world["public"]["items"]
        ]
        history_item_index.sort(
            key=lambda row: (
                row["world_uid"].encode("utf-8"),
                row["seller_uid"].encode("utf-8"),
                row["item_uid"].encode("utf-8"),
            )
        )
        profiles, profile_audit = profiles_mod.build_world_profiles(
            policy,
            mode=mode,
            split=split,
            sellers=world["public"]["sellers"],
            items=processed["public"]["profile_safe_items"],
        )
        profile_audit = {
            **profile_audit,
            "input_boundary": "post_v4_synthetic_public_prefix_canonicalization",
            "profile_safe_items_sha256": common.canonical_sha256(
                processed["public"]["profile_safe_items"]
            ),
            "raw_identity_bearing_profiles_persisted": False,
        }
        (
            candidate_pairs,
            candidate_sampling_audit,
            candidate_generation_audit,
        ) = candidate_sampling_mod.build_world_c40(
            public_candidate_policy,
            candidate_key_hex=candidate_key_hex,
            mode=mode,
            split=split,
            sellers=world["public"]["sellers"],
            raw_observed_items=world["public"]["items"],
            complete_pair_endpoints=world["public"][
                "complete_model_pair_endpoints"
            ],
        )
        if (
            candidate_generation_audit[
                "labels_or_oracle_or_model_scores_read"
            ]
            is not False
            or candidate_generation_audit[
                "ephemeral_step4_raw_evidence_persisted"
            ]
            is not False
        ):
            raise common.ContractError("C40 label/raw-evidence boundary failed")
        history_projection_attestation = (
            production.build_history_projection_attestation(
                policy,
                mode=mode,
                split=split,
                world_uid=world_uid,
                sellers=world["public"]["sellers"],
                items=world["public"]["items"],
                history_safe_occurrences=processed["public"][
                    "history_safe_occurrences"
                ],
                history_item_index=history_item_index,
                parsed_rows=processed["private"][
                    "parsed_identity_occurrences"
                ],
                identity_slots_audit=world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=world["private"][
                    "noise_slots_audit"
                ],
                render_asts=world["private"]["render_asts"],
            )
        )
        identity33_rows, identity33_audit = (
            history_features_mod.build_identity33_all_pairs(
                policy,
                mode=mode,
                split=split,
                history_safe_occurrences=processed["public"][
                    "history_safe_occurrences"
                ],
                history_item_index=history_item_index,
                projection_attestations=[
                    history_projection_attestation
                ],
                complete_model_pair_endpoints=world["public"][
                    "complete_model_pair_endpoints"
                ],
            )
        )
        safe_slots, nuisance_ledger, safe_slot_audit = (
            safe_slots_mod.project_safe_slots(
                policy,
                mode=mode,
                split=split,
                sellers=world["public"]["sellers"],
                items=world["public"]["items"],
                parsed_rows=processed["private"][
                    "parsed_identity_occurrences"
                ],
                identity_slots_edit=world["private"]["identity_slots_edit"],
            )
        )
        private_slot_by_uid = {
            str(row["slot_uid"]): row
            for row in world["private"]["identity_slots_audit"]
        }
        if len(private_slot_by_uid) != len(
            world["private"]["identity_slots_audit"]
        ) or {
            str(row["slot_uid"]) for row in safe_slots
        } != set(private_slot_by_uid):
            raise common.ContractError("Safe-slot/private slot keyset drift")
        if any(
            str(row["identity_uid"])
            != str(private_slot_by_uid[str(row["slot_uid"])]["identity_uid"])
            or str(row["bundle_uid"])
            != str(private_slot_by_uid[str(row["slot_uid"])]["bundle_uid"])
            for row in safe_slots
        ):
            raise common.ContractError(
                "Independently derived safe-slot UID differs from private trace"
            )
        public = world["public"]
        private = world["private"]
        payload["worlds"].append(dict(public["world"]))
        payload["sellers"].extend(dict(row) for row in public["sellers"])
        payload["items"].extend(dict(row) for row in public["items"])
        payload["complete_model_pair_endpoints"].extend(
            dict(row) for row in public["complete_model_pair_endpoints"]
        )
        payload["candidate_pairs"].extend(
            dict(row) for row in candidate_pairs
        )
        payload["candidate_sampling_audit"].extend(
            dict(row) for row in candidate_sampling_audit
        )
        payload["seller_profiles"].extend(
            {"world_uid": world_uid, **profile} for profile in profiles
        )
        payload["redacted_items"].extend(
            dict(row) for row in processed["public"]["redacted_items"]
        )
        payload["history_safe_occurrences"].extend(
            dict(row)
            for row in processed["public"]["history_safe_occurrences"]
        )
        payload["history_item_index"].extend(
            dict(row) for row in history_item_index
        )
        payload["history_projection_attestations"].append(
            history_projection_attestation
        )
        payload["identity33_all_pairs"].extend(
            dict(row) for row in identity33_rows
        )
        payload["parsed_identity_occurrences"].extend(
            dict(row)
            for row in processed["private"]["parsed_identity_occurrences"]
        )
        for name in (
            "identity_slots_audit",
            "identity_slots_edit",
            "noise_slots_audit",
            "render_asts",
            "controller_membership",
            "controller_style_groups",
            "mechanism_assignments",
        ):
            payload[name].extend(dict(row) for row in private[name])
        payload["redaction_diagnostics"].extend(
            dict(row)
            for row in processed["private"]["redaction_diagnostics"]
        )
        payload["rewire_safe_identity_slots"].extend(safe_slots)
        payload["rewire_nuisance_ledger"].extend(nuisance_ledger)
        payload["identity_assets"].extend(
            _extend_with_world_uid(world_uid, private["identity_assets"])
        )
        payload["positive_targets"].extend(
            _extend_with_world_uid(world_uid, private["positive_targets"])
        )
        payload["negative_flags"].extend(
            {
                "world_uid": world_uid,
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "flag": str(row["flag"]),
                "asset_index": int(row["asset_index"]),
            }
            for row in private["negative_flags"]
        )
        payload["override_audit"].extend(
            _extend_with_world_uid(world_uid, private["override_audit"])
        )
        payload["solver_audit"].append(dict(private["solver_audit"]))
        payload["redaction_registry_audit"].append(
            {
                "world_uid": world_uid,
                "registry_hashes": processed["private"][
                    "redaction_registry_hashes"
                ],
                "collision_audit": processed["private"][
                    "redaction_registry_collision_audit"
                ],
            }
        )
        generation_audit = {
            "world_uid": world_uid,
            "split": split,
            "mode_global_ordinal": int(record["mode_global_ordinal"]),
            "split_ordinal": int(record["split_ordinal"]),
            "profile_audit": profile_audit,
            "parser_audit": processed["private"][
                "parser_structural_audit"
            ],
            "redaction_audit": processed["private"][
                "redaction_structural_audit"
            ],
            "safe_slot_audit": safe_slot_audit,
            "identity33_audit": identity33_audit,
            "independent_typed_dgp_replay_audit": (
                independent_replay_audit
            ),
            "producer_regeneration_audit": producer_regeneration_audit,
        }
        payload["world_generation_audit"].append(generation_audit)
        world_digests.append(
            {
                "world_uid": world_uid,
                "observed_sha256": common.canonical_sha256(
                    {
                        "world": public["world"],
                        "sellers": public["sellers"],
                        "items": public["items"],
                        "complete_model_pair_endpoints": public[
                            "complete_model_pair_endpoints"
                        ],
                        "candidate_pairs": candidate_pairs,
                        "seller_profiles": profiles,
                        "redacted_items": processed["public"][
                            "redacted_items"
                        ],
                        "profile_safe_items": processed["public"][
                            "profile_safe_items"
                        ],
                        "history_safe_occurrences": processed["public"][
                            "history_safe_occurrences"
                        ],
                        "history_item_index": history_item_index,
                        "history_projection_attestation": (
                            history_projection_attestation
                        ),
                        "identity33_all_pairs": identity33_rows,
                    }
                ),
                "private_trace_sha256": common.canonical_sha256(
                    {
                        **private,
                        **processed["private"],
                        "candidate_sampling_audit": (
                            candidate_sampling_audit
                        ),
                        "profile_audit": profile_audit,
                    }
                ),
            }
        )

    payload["candidate_sampling_audit"].sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        )
    )
    candidate_rank = {
        (
            str(row["world_uid"]),
            str(row["canonical_pair_uid"]),
        ): int(row["selected_rank"])
        for row in payload["candidate_sampling_audit"]
        if str(row["selected_bool"]) == "true"
    }
    payload["candidate_pairs"].sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            candidate_rank[
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                )
            ],
        )
    )
    payload["rewire_nuisance_ledger"].sort(
        key=lambda row: str(row["identity_uid"]).encode("utf-8")
    )
    if len(
        {str(row["identity_uid"]) for row in payload["rewire_nuisance_ledger"]}
    ) != len(payload["rewire_nuisance_ledger"]):
        raise common.ContractError("Split rewire nuisance ledger has duplicate identities")
    _validate_split_payload(
        policy,
        split=split,
        expected_world_count=expected_world_count,
        payload=payload,
    )
    placebos = (
        feature_derangement_mod.build_all_feature_derangements(
            policy,
            mode=mode,
            split=split,
            m2_identity33_all_pairs=payload["identity33_all_pairs"],
            candidate_pairs=payload["candidate_pairs"],
            complete_pair_endpoints=payload[
                "complete_model_pair_endpoints"
            ],
        )
        if split == "train"
        else []
    )
    support_comparability_preflight: dict[str, Any] | None = None
    if split == "train":
        support_comparability_preflight = (
            placebo_support_mod.run_support_comparability_preflight(
                policy,
                mode=mode,
                split=split,
                m2_identity33_all_pairs=payload[
                    "identity33_all_pairs"
                ],
                candidate_pairs=payload["candidate_pairs"],
                complete_pair_endpoints=payload[
                    "complete_model_pair_endpoints"
                ],
                placebos=placebos,
            )
        )
        if not support_comparability_preflight[
            "all_five_primary_validity_pass"
        ]:
            failed = [
                {
                    "rewire_seed_id": row["rewire_seed_id"],
                    "failed_gates": [
                        name
                        for name, passed in row["primary_c40"]["gates"].items()
                        if not passed
                    ],
                }
                for row in support_comparability_preflight["seed_results"]
                if not row["primary_validity_pass"]
            ]
            raise common.ContractError(
                "Placebo support-comparability preflight failed: "
                f"{failed}"
            )
    aggregate_integrity_receipts = {
        "candidate_integrity": (
            integrity_receipts_mod.build_candidate_integrity_receipt(
                candidate_integrity_context,
                candidate_policy=public_candidate_policy,
                candidate_key_hex=candidate_key_hex,
                mode=mode,
                split=split,
                worlds=payload["worlds"],
                sellers=payload["sellers"],
                raw_observed_items=payload["items"],
                complete_pair_endpoints=payload[
                    "complete_model_pair_endpoints"
                ],
                candidate_pairs=payload["candidate_pairs"],
                candidate_sampling_audit=payload[
                    "candidate_sampling_audit"
                ],
            )
        ),
        "independent_dgp_comparison": (
            integrity_receipts_mod.build_independent_dgp_comparison_receipt(
                policy,
                mode=mode,
                split=split,
                worlds=payload["worlds"],
                per_world_comparison_receipts=[
                    row["independent_typed_dgp_replay_audit"]
                    for row in payload["world_generation_audit"]
                ],
                producer_typed_dgp_projections=payload[
                    "producer_typed_dgp_projections"
                ],
                independent_replay_ledgers=payload[
                    "independent_typed_dgp_replay_ledgers"
                ],
            )
        ),
        "render_integrity": (
            integrity_receipts_mod.build_render_integrity_receipt(
                policy,
                mode=mode,
                split=split,
                template=template,
                worlds=payload["worlds"],
                sellers=payload["sellers"],
                items=payload["items"],
                redacted_items=payload["redacted_items"],
                parsed_identity_occurrences=payload[
                    "parsed_identity_occurrences"
                ],
                identity_slots_audit=payload[
                    "identity_slots_audit"
                ],
                noise_slots_audit=payload["noise_slots_audit"],
                render_asts=payload["render_asts"],
                override_audit=payload["override_audit"],
            )
        ),
    }
    if split == "train":
        if support_comparability_preflight is None:
            raise common.ContractError(
                "Train M1 receipt lacks support preflight"
            )
        aggregate_integrity_receipts[
            "m1_derangement_integrity"
        ] = integrity_receipts_mod.build_m1_derangement_integrity_receipt(
            policy,
            mode=mode,
            split=split,
            m2_identity33_all_pairs=payload["identity33_all_pairs"],
            candidate_pairs=payload["candidate_pairs"],
            complete_pair_endpoints=payload[
                "complete_model_pair_endpoints"
            ],
            placebos=placebos,
            support_preflight=support_comparability_preflight,
        )
    aggregate_parent_projections = (
        integrity_receipts_mod.build_development_parent_projections(
            policy,
            split=split,
            receipts=aggregate_integrity_receipts,
        )
    )
    structural_audit_receipt = (
        integrity_receipts_mod.build_structural_audit_receipt(
            policy,
            mode=mode,
            split=split,
            receipts=aggregate_integrity_receipts,
            parent_projections=aggregate_parent_projections,
        )
    )
    return {
        "tables": payload,
        "placebos": placebos,
        "support_comparability_preflight": (
            support_comparability_preflight
        ),
        "world_digests": sorted(
            world_digests, key=lambda row: row["world_uid"].encode("utf-8")
        ),
        "aggregate_integrity_receipts": aggregate_integrity_receipts,
        "aggregate_parent_projections": aggregate_parent_projections,
        "structural_audit_receipt": structural_audit_receipt,
    }


def _validate_split_payload(
    policy: Mapping[str, Any],
    *,
    split: str,
    expected_world_count: int,
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    worlds = payload["worlds"]
    sellers = payload["sellers"]
    items = payload["items"]
    pairs = payload["complete_model_pair_endpoints"]
    candidates = payload["candidate_pairs"]
    candidate_audit = payload["candidate_sampling_audit"]
    profiles = payload["seller_profiles"]
    parsed = payload["parsed_identity_occurrences"]
    history_rows = payload["history_safe_occurrences"]
    history_item_index = payload["history_item_index"]
    history_attestations = payload["history_projection_attestations"]
    identity33_rows = payload["identity33_all_pairs"]
    replay_sellers = payload["dgp_replay_seller_uid_pool"]
    replay_all_items = payload["dgp_replay_all_item_uid_pool"]
    replay_title_items = payload[
        "dgp_replay_nonempty_title_item_uid_pool"
    ]
    replay_description_items = payload[
        "dgp_replay_nonempty_description_item_uid_pool"
    ]
    if (
        len(worlds) != expected_world_count
        or len(sellers) != 28 * expected_world_count
        or len(profiles) != 28 * expected_world_count
        or len(pairs) != 378 * expected_world_count
        or len(candidates) != 40 * expected_world_count
        or len(candidate_audit) != 378 * expected_world_count
        or len(payload["redacted_items"]) != len(items)
        or len(history_item_index) != len(items)
        or len(history_rows) != len(parsed)
        or len(identity33_rows) != len(pairs)
        or len(history_attestations) != expected_world_count
        or len(payload["rewire_safe_identity_slots"]) != len(parsed)
        or len(replay_sellers) != len(sellers)
        or len(replay_all_items) != len(items)
        or len(payload["solver_audit"]) != expected_world_count
        or len(payload["producer_typed_dgp_projections"])
        != expected_world_count
        or len(payload["independent_typed_dgp_replay_ledgers"])
        != expected_world_count
    ):
        raise common.ContractError(f"{split} aggregate count gate failed")
    world_uids = {str(row["world_uid"]) for row in worlds}
    if len(world_uids) != expected_world_count:
        raise common.ContractError("Split world UID uniqueness failed")
    if any(str(row["world_uid"]) not in world_uids for row in sellers):
        raise common.ContractError("Seller world foreign key failed")
    seller_uids = {str(row["seller_uid"]) for row in sellers}
    if len(seller_uids) != len(sellers):
        raise common.ContractError("Seller UID uniqueness failed")
    item_uids = {str(row["item_uid"]) for row in items}
    if len(item_uids) != len(items):
        raise common.ContractError("Item UID uniqueness failed")
    pair_uids = {str(row["canonical_pair_uid"]) for row in pairs}
    if len(pair_uids) != len(pairs):
        raise common.ContractError("Complete pair UID uniqueness failed")
    pair_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in pairs
    }
    replay_seller_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]))
        for row in replay_sellers
    }
    observed_seller_keys = {
        (str(row["world_uid"]), str(row["seller_uid"])) for row in sellers
    }
    if (
        replay_seller_keys != observed_seller_keys
        or len(replay_seller_keys) != len(replay_sellers)
    ):
        raise common.ContractError("Independent replay seller UID pool drift")
    replay_all_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in replay_all_items
    }
    observed_all_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in items
    }
    replay_title_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in replay_title_items
    }
    replay_description_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in replay_description_items
    }
    expected_title_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in items
        if str(row["title"])
    }
    expected_description_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in items
        if str(row["description"])
    }
    if (
        replay_all_keys != observed_all_keys
        or len(replay_all_keys) != len(replay_all_items)
        or replay_title_keys != expected_title_keys
        or len(replay_title_keys) != len(replay_title_items)
        or replay_description_keys != expected_description_keys
        or len(replay_description_keys) != len(replay_description_items)
    ):
        raise common.ContractError("Independent replay item UID pool drift")
    per_world_pairs = Counter(str(row["world_uid"]) for row in pairs)
    if any(per_world_pairs[world_uid] != 378 for world_uid in world_uids):
        raise common.ContractError("A world does not contain all 378 pairs")
    candidate_schema = policy["candidate_design"][
        "public_safe_projection_columns"
    ]
    candidate_audit_schema = policy["candidate_design"][
        "sampling_audit_projection_columns"
    ]
    if any(list(row) != candidate_schema for row in candidates):
        raise common.ContractError("C40 safe projection schema/order drift")
    if any(list(row) != candidate_audit_schema for row in candidate_audit):
        raise common.ContractError("C40 sampling audit schema/order drift")
    candidate_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidates
    }
    audit_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_audit
    }
    selected_audit_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_audit
        if str(row["selected_bool"]) == "true"
    }
    if (
        len(candidate_keys) != len(candidates)
        or len(audit_keys) != len(candidate_audit)
        or candidate_keys != selected_audit_keys
        or not audit_keys.issubset(pair_keys)
    ):
        raise common.ContractError("C40 split keyset closure failed")
    per_world_candidates = Counter(
        str(row["world_uid"]) for row in candidates
    )
    per_world_audit = Counter(
        str(row["world_uid"]) for row in candidate_audit
    )
    for world_uid in world_uids:
        ranks = sorted(
            int(row["selected_rank"])
            for row in candidate_audit
            if str(row["world_uid"]) == world_uid
            and str(row["selected_bool"]) == "true"
        )
        if (
            per_world_candidates[world_uid] != 40
            or per_world_audit[world_uid] != 378
            or ranks != list(range(1, 41))
        ):
            raise common.ContractError("C40 per-world count/rank gate failed")
    identity33_schema = [
        "canonical_pair_uid",
        "world_uid",
        *policy["history_features"]["feature_names"],
    ]
    if any(list(row) != identity33_schema for row in identity33_rows):
        raise common.ContractError("Identity33 aggregate schema/order drift")
    identity33_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in identity33_rows
    }
    if identity33_keys != pair_keys or len(identity33_keys) != len(identity33_rows):
        raise common.ContractError("Identity33 aggregate all-pair keyset drift")
    history_schema = policy["relational_integrity"]["observed_core_schemas"][
        "history_safe_occurrences.csv"
    ]
    if any(set(row) != set(history_schema) for row in history_rows):
        raise common.ContractError("History-safe aggregate schema drift")
    history_item_schema = policy["relational_integrity"][
        "observed_core_schemas"
    ]["history_item_index.csv"]
    if any(list(row) != history_item_schema for row in history_item_index):
        raise common.ContractError("History item-index aggregate schema drift")
    if {
        str(row["item_uid"]) for row in history_item_index
    } != item_uids:
        raise common.ContractError("History item-index aggregate keyset drift")
    if {
        str(row.get("world_uid", "")) for row in history_attestations
    } != world_uids:
        raise common.ContractError(
            "History projection attestation world keyset drift"
        )


def _write_table_set(
    policy: Mapping[str, Any],
    *,
    stage: Path,
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    observed = stage / "observed"
    oracle = stage / "oracle"
    schema = policy["relational_integrity"]["observed_core_schemas"]
    profile_spec = common.load_json(
        common.verify_file_pin(
            policy["frozen_inputs"]["step3_profile_schema"],
            label="Step3 seller-profile schema",
        )
    )
    profile_fields = ["world_uid", *profile_spec["profile_fields"]]
    files: list[tuple[Path, str]] = []

    def write_csv(
        relative: Path,
        name: str,
        fieldnames: Sequence[str],
        role: str,
    ) -> None:
        rows = [dict(row) for row in payload[name]]
        common.write_csv(stage / relative, rows, fieldnames)
        files.append((stage / relative, role))

    def write_jsonl(relative: Path, name: str, role: str) -> None:
        rows = [dict(row) for row in payload[name]]
        common.write_jsonl(stage / relative, rows)
        files.append((stage / relative, role))

    write_csv(Path("observed/worlds.csv"), "worlds", schema["worlds.csv"], "observed")
    write_csv(Path("observed/sellers.csv"), "sellers", schema["sellers.csv"], "observed")
    write_jsonl(Path("observed/items.jsonl"), "items", "observed_raw")
    write_csv(
        Path("observed/complete_model_pair_endpoints.csv"),
        "complete_model_pair_endpoints",
        policy["relational_integrity"]["pair_projection_contract"][
            "complete_model_pair_endpoints_schema"
        ],
        "observed_model_endpoint",
    )
    write_csv(
        Path("observed/candidate_pairs.csv"),
        "candidate_pairs",
        policy["candidate_design"]["public_safe_projection_columns"],
        "observed_candidate_endpoint",
    )
    for row in payload["seller_profiles"]:
        if list(row) != profile_fields:
            raise common.ContractError("Persisted seller-profile schema/order drift")
    write_jsonl(
        Path("observed/seller_profiles.jsonl"),
        "seller_profiles",
        "observed_profile_intermediate",
    )
    write_jsonl(
        Path("observed/redacted_items.jsonl"),
        "redacted_items",
        "observed_redacted",
    )
    write_csv(
        Path("observed/history_safe_occurrences.csv"),
        "history_safe_occurrences",
        schema["history_safe_occurrences.csv"],
        "observed_history_safe",
    )
    write_csv(
        Path("observed/history_item_index.csv"),
        "history_item_index",
        schema["history_item_index.csv"],
        "observed_history_item_index",
    )
    write_jsonl(
        Path("observed/history_projection_attestations.jsonl"),
        "history_projection_attestations",
        "observed_history_projection_attestation",
    )
    write_csv(
        Path("observed/identity33_all_pairs.csv"),
        "identity33_all_pairs",
        [
            "canonical_pair_uid",
            "world_uid",
            *policy["history_features"]["feature_names"],
        ],
        "observed_model_identity33",
    )

    write_csv(
        Path(
            "structural_audit/independent_replay_inputs/"
            "seller_uid_pool.csv"
        ),
        "dgp_replay_seller_uid_pool",
        ("world_uid", "seller_uid"),
        "private_replay_uid_input",
    )
    write_csv(
        Path(
            "structural_audit/independent_replay_inputs/"
            "all_item_uid_pool.csv"
        ),
        "dgp_replay_all_item_uid_pool",
        ("world_uid", "seller_uid", "item_uid"),
        "private_replay_uid_input",
    )
    write_csv(
        Path(
            "structural_audit/independent_replay_inputs/"
            "nonempty_title_item_uid_pool.csv"
        ),
        "dgp_replay_nonempty_title_item_uid_pool",
        ("world_uid", "seller_uid", "item_uid"),
        "private_replay_uid_input",
    )
    write_csv(
        Path(
            "structural_audit/independent_replay_inputs/"
            "nonempty_description_item_uid_pool.csv"
        ),
        "dgp_replay_nonempty_description_item_uid_pool",
        ("world_uid", "seller_uid", "item_uid"),
        "private_replay_uid_input",
    )

    write_csv(
        Path(
            "structural_audit/"
            "parsed_identity_occurrences.structural_audit_private.csv"
        ),
        "parsed_identity_occurrences",
        schema["parsed_identity_occurrences.structural_audit_private.csv"],
        "private_structural",
    )
    write_csv(
        Path(
            "candidate_integrity_private/"
            "candidate_sampling_audit.csv"
        ),
        "candidate_sampling_audit",
        policy["candidate_design"]["sampling_audit_projection_columns"],
        "private_candidate_sampling_audit",
    )
    write_csv(
        Path("structural_audit/rewire_safe_identity_slots.csv"),
        "rewire_safe_identity_slots",
        policy["placebo"]["rewire_safe_slot_schema"],
        "private_rewire_safe",
    )
    write_csv(
        Path("structural_audit/rewire_nuisance_ledger.csv"),
        "rewire_nuisance_ledger",
        ("identity_uid", "nuisance_class"),
        "private_rewire_safe",
    )
    write_csv(
        Path("structural_audit/renderer_identity_slots.audit.csv"),
        "identity_slots_audit",
        policy["relational_integrity"][
            "renderer_identity_slots_audit_schema"
        ],
        "private_structural",
    )
    write_csv(
        Path("structural_audit/renderer_identity_slots.edit.csv"),
        "identity_slots_edit",
        policy["relational_integrity"][
            "renderer_identity_slots_edit_schema"
        ],
        "private_structural",
    )
    write_csv(
        Path("structural_audit/renderer_noise_slots.audit.csv"),
        "noise_slots_audit",
        policy["relational_integrity"][
            "renderer_noise_slots_audit_schema"
        ],
        "private_structural",
    )
    write_jsonl(
        Path("structural_audit/render_asts.jsonl"),
        "render_asts",
        "private_structural",
    )
    write_jsonl(
        Path("structural_audit/redaction_diagnostics.jsonl"),
        "redaction_diagnostics",
        "private_structural",
    )
    write_jsonl(
        Path("structural_audit/world_generation_audit.jsonl"),
        "world_generation_audit",
        "private_structural",
    )
    write_jsonl(
        Path("structural_audit/redaction_registry_audit.jsonl"),
        "redaction_registry_audit",
        "private_structural",
    )

    write_csv(
        Path("oracle/controller_membership.csv"),
        "controller_membership",
        ("world_uid", "controller_uid", "seller_uid"),
        "private_oracle",
    )
    write_csv(
        Path("oracle/controller_style_groups.csv"),
        "controller_style_groups",
        ("world_uid", "controller_uid", "style_id"),
        "private_oracle",
    )
    write_csv(
        Path("oracle/mechanism_assignments.csv"),
        "mechanism_assignments",
        ("world_uid", "controller_uid", "mechanism", "mechanism_slot_uid"),
        "private_oracle",
    )
    write_jsonl(
        Path("oracle/identity_assets.jsonl"),
        "identity_assets",
        "private_oracle",
    )
    write_csv(
        Path("oracle/positive_targets.csv"),
        "positive_targets",
        (
            "world_uid",
            "controller_uid",
            "mechanism",
            "mechanism_slot_uid",
            "seller_uid_left",
            "seller_uid_right",
            "canonical_pair_uid",
        ),
        "private_oracle",
    )
    write_csv(
        Path("oracle/negative_flags.csv"),
        "negative_flags",
        ("world_uid", "canonical_pair_uid", "flag", "asset_index"),
        "private_oracle",
    )
    write_csv(
        Path("oracle/registered_override_audit.csv"),
        "override_audit",
        (
            "world_uid",
            "override_kind",
            "asset_index",
            "canonical_pair_uid",
            "seller_uid_left",
            "seller_uid_right",
            "item_uid_left",
            "item_uid_right",
        ),
        "private_oracle",
    )
    write_jsonl(
        Path("oracle/solver_audit.jsonl"),
        "solver_audit",
        "private_oracle",
    )
    write_jsonl(
        Path("oracle/producer_typed_dgp_projection.private.jsonl"),
        "producer_typed_dgp_projections",
        "private_producer_typed_dgp_projection",
    )
    projection_relative = Path(
        "oracle/producer_typed_dgp_projection.private.jsonl"
    )
    projection_path = stage / projection_relative
    projections = payload["producer_typed_dgp_projections"]
    projection_modes = {str(row["mode"]) for row in projections}
    projection_splits = {str(row["split"]) for row in projections}
    if len(projection_modes) != 1 or len(projection_splits) != 1:
        raise common.ContractError(
            "Producer typed projection mode/split closure failed"
        )
    projection_mode = next(iter(projection_modes))
    projection_split = next(iter(projection_splits))
    projection_world_uids = sorted(
        (str(row["world_uid"]) for row in projections),
        key=lambda value: value.encode("utf-8"),
    )
    expected_projection_world_uids = (
        independent_replay.registered_world_uids_for_split(
            policy,
            mode=projection_mode,
            split=projection_split,
        )
    )
    if projection_world_uids != expected_projection_world_uids:
        raise common.ContractError(
            "Producer typed projection complete world set drift"
        )
    projection_manifest = {
        "version": (
            "2026-07-28-step28-v13-producer-typed-dgp-"
            "projection-manifest-v1-draft"
        ),
        "mode": projection_mode,
        "split": projection_split,
        "evidence_level": (
            "DEVELOPMENT_PRODUCER_PRIVATE_PROJECTION_"
            "NOT_FORMAL_CUSTODY_SEAL"
        ),
        "formal_custody_seal": False,
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY_PATH),
        "world_count": len(projection_world_uids),
        "registered_split_world_count": len(
            expected_projection_world_uids
        ),
        "complete_registered_world_set_exact": True,
        "registered_world_uids_sha256": common.canonical_sha256(
            expected_projection_world_uids
        ),
        "projection_file": common.artifact_record(
            projection_path,
            role="private_producer_typed_dgp_projection",
            root=stage,
        ),
        "source_record": {
            "role": "producer_typed_dgp_projector",
            "path_basename": Path(
                producer_projection_mod.__file__
            ).name,
            "sha256": common.sha256_file(
                Path(producer_projection_mod.__file__).resolve()
            ),
        },
    }
    projection_manifest["canonical_self_hash"] = (
        common.canonical_sha256(projection_manifest)
    )
    projection_manifest_path = (
        stage
        / "oracle"
        / "producer_typed_dgp_projection_manifest.private.json"
    )
    common.write_json(projection_manifest_path, projection_manifest)
    files.append(
        (
            projection_manifest_path,
            "private_producer_typed_dgp_projection_manifest",
        )
    )
    return [
        common.artifact_record(path, role=role, root=stage)
        for path, role in sorted(
            files, key=lambda value: value[0].relative_to(stage).as_posix()
        )
    ]


def _producer_hashes() -> dict[str, str]:
    names = (
        "step28_v13_common.py",
        "step28_v13_structure.py",
        "step28_v13_nonidentity.py",
        "step28_v13_identity_values.py",
        "step28_v13_identity_plan.py",
        "step28_v13_text_renderer.py",
        "step28_v13_world_builder.py",
        "step28_v13_independent_private_dgp_replay.py",
        "step28_v13_independent_dgp_comparator.py",
        "step28_v13_producer_dgp_projection.py",
        "step28_v13_run_independent_dgp_replay.py",
        "step28_v13_compare_independent_dgp_replay.py",
        "step28_v13_profiles.py",
        "step28_v13_production_chain.py",
        "step28_v13_safe_slots.py",
        "step28_v13_candidate_sampling.py",
        "step28_v13_feature_derangement.py",
        "step28_v13_integrity_receipts.py",
        "step28_v13_placebo_rewire.py",
        "step28_v13_placebo_support.py",
        "step28_v13_smoke_private_regeneration.py",
        "step28_v13_history_features.py",
        "step28_v13_generate_dataset.py",
    )
    return {
        name: common.sha256_file(common.ROOT / "scripts" / name)
        for name in names
    }


def _write_placebo_set(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    stage: Path,
    placebos: Sequence[Mapping[str, Any]],
    support_preflight: Mapping[str, Any] | None,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    m1_integrity_receipt: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if mode != "development_smoke" or split not in SPLITS:
        raise common.ContractError(
            "Placebo persistence mode/split boundary drift"
        )
    if split != "train":
        if (
            placebos
            or support_preflight is not None
            or m1_integrity_receipt is not None
        ):
            raise common.ContractError(
                "Non-train split contains M1 persistence inputs"
            )
        return []
    if not placebos:
        raise common.ContractError(
            "Train split is missing all M1 persistence inputs"
        )
    if len(placebos) != int(policy["placebo"]["replicates"]):
        raise common.ContractError("Placebo output replicate count drift")
    if (
        not isinstance(support_preflight, Mapping)
        or not isinstance(m1_integrity_receipt, Mapping)
    ):
        raise common.ContractError(
            "Train M1 support or aggregate receipt is absent"
        )
    recomputed_support = (
        placebo_support_mod.run_support_comparability_preflight(
            policy,
            mode=mode,
            split=split,
            m2_identity33_all_pairs=m2_identity33_all_pairs,
            candidate_pairs=candidate_pairs,
            complete_pair_endpoints=complete_pair_endpoints,
            placebos=placebos,
        )
    )
    if dict(support_preflight) != recomputed_support:
        raise common.ContractError(
            "Placebo persistence support preflight replay mismatch"
        )
    recomputed_m1_receipt = (
        integrity_receipts_mod.build_m1_derangement_integrity_receipt(
            policy,
            mode=mode,
            split=split,
            m2_identity33_all_pairs=m2_identity33_all_pairs,
            candidate_pairs=candidate_pairs,
            complete_pair_endpoints=complete_pair_endpoints,
            placebos=placebos,
            support_preflight=recomputed_support,
        )
    )
    if dict(m1_integrity_receipt) != recomputed_m1_receipt:
        raise common.ContractError(
            "Placebo persistence M1 aggregate receipt replay mismatch"
        )
    integrity_receipts_mod.validate_aggregate_receipt(
        policy,
        role="m1_derangement_integrity",
        receipt=m1_integrity_receipt,
    )
    if (
        m1_integrity_receipt["aggregate_content_hashes"][
            "support_preflight_sha256"
        ]
        != common.canonical_sha256(recomputed_support)
    ):
        raise common.ContractError(
            "Placebo persistence support/M1 receipt hash mismatch"
        )
    files: list[tuple[Path, str]] = []

    def write_csv(
        relative: Path,
        rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str],
        role: str,
    ) -> None:
        path = stage / relative
        common.write_csv(path, [dict(row) for row in rows], fieldnames)
        files.append((path, role))

    seen_seed_ids: set[str] = set()
    expected_seed_ids = [
        "rws_" + common.sha256_bytes(bytes.fromhex(seed_hex))
        for seed_hex in policy["randomness"]["development_smoke"][
            "rewire_key_hexes"
        ]
    ]
    required_output_keys = {
        "rewire_seed_id",
        "identity33_all_pairs",
        "feature_derangement_mapping",
        "joint_vector_multiset_exact_by_world_and_universe",
        "endpoint_disjoint_bijection_exact",
        "labels_or_controller_inputs_read",
        "candidate_trigger_or_audit_inputs_read",
        "canonical_self_hash",
    }
    if (
        len(expected_seed_ids) != int(policy["placebo"]["replicates"])
        or len(set(expected_seed_ids)) != len(expected_seed_ids)
    ):
        raise common.ContractError(
            "Placebo persistence registered seed set drift"
        )
    identity33_schema = [
        "canonical_pair_uid",
        "world_uid",
        *policy["history_features"]["feature_names"],
    ]
    mapping_schema = policy["placebo"][
        "feature_derangement_mapping_schema"
    ]
    for output in placebos:
        if not isinstance(output, Mapping) or set(output) != required_output_keys:
            raise common.ContractError(
                "Placebo persistence output envelope drift"
            )
        seed_id = str(output["rewire_seed_id"])
        expected_self_hash = common.canonical_sha256(
            {
                key: value
                for key, value in output.items()
                if key != "canonical_self_hash"
            }
        )
        identity33_rows = output["identity33_all_pairs"]
        mapping_rows = output["feature_derangement_mapping"]
        if (
            not re.fullmatch(r"rws_[0-9a-f]{64}", seed_id)
            or seed_id not in expected_seed_ids
            or seed_id in seen_seed_ids
            or output["labels_or_controller_inputs_read"] is not False
            or output["candidate_trigger_or_audit_inputs_read"] is not False
            or output[
                "joint_vector_multiset_exact_by_world_and_universe"
            ]
            is not True
            or output["endpoint_disjoint_bijection_exact"] is not True
            or output["canonical_self_hash"] != expected_self_hash
            or not isinstance(identity33_rows, list)
            or len(identity33_rows) != 3780
            or any(list(row) != identity33_schema for row in identity33_rows)
            or len(
                {
                    (
                        str(row["world_uid"]),
                        str(row["canonical_pair_uid"]),
                    )
                    for row in identity33_rows
                }
            )
            != 3780
            or not isinstance(mapping_rows, list)
            or len(mapping_rows) != 3780
            or any(list(row) != mapping_schema for row in mapping_rows)
            or len(
                {
                    (
                        str(row["world_uid"]),
                        str(row["destination_pair_uid"]),
                    )
                    for row in mapping_rows
                }
            )
            != 3780
            or len(
                {
                    (
                        str(row["world_uid"]),
                        str(row["universe"]),
                        str(row["source_pair_uid"]),
                    )
                    for row in mapping_rows
                }
            )
            != 3780
            or any(
                str(row["rewire_seed_id"]) != seed_id
                or row["endpoint_disjoint_bool"] is not True
                for row in mapping_rows
            )
        ):
            raise common.ContractError("Placebo output seed/boundary drift")
        seen_seed_ids.add(seed_id)
        public_root = Path("placebo") / seed_id
        private_root = Path("placebo_integrity_private") / seed_id
        write_csv(
            public_root / "identity33_all_pairs.csv",
            identity33_rows,
            identity33_schema,
            "observed_placebo_identity33",
        )
        write_csv(
            private_root / "feature_derangement_mapping.csv",
            mapping_rows,
            mapping_schema,
            "private_m1_feature_derangement_mapping",
        )
        receipt = {
            "version": (
                "2026-07-28-step28-v13-development-m1-"
                "derangement-private-receipt-v1-draft"
            ),
            "evidence_level": "DEVELOPMENT_SELF_HASH_NOT_FORMAL_CUSTODY",
            "rewire_seed_id": seed_id,
            "identity33_row_count": len(output["identity33_all_pairs"]),
            "mapping_row_count": len(
                output["feature_derangement_mapping"]
            ),
            "joint_vector_multiset_exact_by_world_and_universe": True,
            "endpoint_disjoint_bijection_exact": True,
            "labels_or_controller_inputs_read": False,
            "candidate_trigger_or_audit_inputs_read": False,
            "output_canonical_self_hash": str(
                output["canonical_self_hash"]
            ),
            "formal_use_forbidden": True,
        }
        receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
        receipt_path = (
            stage / private_root / "derangement_receipt.private.json"
        )
        common.write_json(receipt_path, receipt)
        files.append((receipt_path, "private_m1_derangement_receipt"))
    if seen_seed_ids != set(expected_seed_ids):
        raise common.ContractError(
            "Placebo persistence registered seed set is incomplete"
        )
    support_seed_results = (
        support_preflight.get("seed_results")
        if isinstance(support_preflight, Mapping)
        else None
    )
    if (
        support_preflight is None
        or not isinstance(support_preflight, Mapping)
        or set(support_preflight)
        != {
            "version",
            "evidence_level",
            "mode",
            "split",
            "feature_count",
            "world_count",
            "primary_pair_count_per_source",
            "secondary_pair_count_per_source",
            "m2_identity33_sha256",
            "shared_m2_c40_rms_scale_sha256",
            "seed_results",
            "all_five_primary_validity_pass",
            "labels_or_controller_inputs_read",
            "candidate_trigger_or_audit_inputs_read",
            "formal_use_forbidden",
            "canonical_self_hash",
        }
        or support_preflight.get("canonical_self_hash")
        != common.canonical_sha256(
            {
                key: value
                for key, value in support_preflight.items()
                if key != "canonical_self_hash"
            }
        )
        or support_preflight.get("all_five_primary_validity_pass")
        is not True
        or support_preflight.get("labels_or_controller_inputs_read")
        is not False
        or support_preflight.get(
            "candidate_trigger_or_audit_inputs_read"
        )
        is not False
        or not isinstance(support_seed_results, list)
        or any(
            not isinstance(row, Mapping)
            for row in support_seed_results
        )
        or [
            str(row.get("rewire_seed_id", ""))
            for row in support_seed_results
        ]
        != expected_seed_ids
        or any(
            row.get("primary_validity_pass") is not True
            for row in support_seed_results
        )
    ):
        raise common.ContractError(
            "Placebo support preflight is absent or failed at persistence"
        )
    support_path = (
        stage
        / "placebo_integrity_private"
        / "support_comparability_preflight.json"
    )
    common.write_json(support_path, dict(support_preflight))
    files.append((support_path, "private_m1_support_preflight"))
    return [
        common.artifact_record(path, role=role, root=stage)
        for path, role in sorted(
            files, key=lambda value: value[0].relative_to(stage).as_posix()
        )
    ]


def _write_aggregate_integrity_set(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    stage: Path,
    receipts: Mapping[str, Mapping[str, Any]],
    parent_projections: Sequence[Mapping[str, Any]],
    structural_audit_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Persist only scalar/hash development receipts, never joinable rows."""

    expected_roles = integrity_receipts_mod.expected_receipt_roles(
        policy, split=split
    )
    if set(receipts) != set(expected_roles):
        raise common.ContractError(
            "Aggregate integrity persistence role set drift"
        )
    expected_structural = (
        integrity_receipts_mod.build_structural_audit_receipt(
            policy,
            mode=mode,
            split=split,
            receipts=receipts,
            parent_projections=parent_projections,
        )
    )
    if dict(structural_audit_receipt) != expected_structural:
        raise common.ContractError(
            "Structural audit receipt changed before persistence"
        )
    files: list[tuple[Path, str]] = []
    root = stage / "aggregate_integrity"
    for role in expected_roles:
        receipt = dict(receipts[role])
        integrity_receipts_mod.validate_aggregate_receipt(
            policy, role=role, receipt=receipt
        )
        path = root / f"{role}.receipt.json"
        common.write_json(path, receipt)
        if common.sha256_file(path) != (
            integrity_receipts_mod.pretty_json_sha256(receipt)
        ):
            raise common.ContractError(
                "Persisted aggregate receipt byte hash drift"
            )
        files.append((path, f"development_aggregate_{role}_receipt"))
    projection_path = root / "parent_projections.development.json"
    common.write_json(projection_path, list(parent_projections))
    files.append(
        (projection_path, "development_self_hash_parent_projections")
    )
    structural_path = root / "structural_audit.receipt.json"
    common.write_json(structural_path, dict(structural_audit_receipt))
    files.append(
        (structural_path, "development_receipt_only_structural_audit")
    )
    return [
        common.artifact_record(path, role=role, root=stage)
        for path, role in sorted(
            files, key=lambda value: value[0].relative_to(stage).as_posix()
        )
    ]


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on Linux; formal generation is Linux-only."""

    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_release_tree(stage: Path) -> None:
    """Flush every closed artifact, then each directory bottom-up."""

    root = common.filesystem_path(stage)
    files: list[tuple[str, str]] = []
    directories: list[tuple[str, str]] = [("", root)]
    for current, directory_names, file_names in os.walk(root):
        directory_names.sort()
        file_names.sort()
        current_relative = os.path.relpath(current, root)
        if current_relative == ".":
            current_relative = ""
        for name in directory_names:
            filesystem_value = os.path.join(current, name)
            relative_value = os.path.join(current_relative, name).replace(
                os.sep, "/"
            )
            directories.append((relative_value, filesystem_value))
        for name in file_names:
            filesystem_value = os.path.join(current, name)
            relative_value = os.path.join(current_relative, name).replace(
                os.sep, "/"
            )
            files.append((relative_value, filesystem_value))
    for _relative, filesystem_value in sorted(
        files,
        key=lambda value: value[0].encode("utf-8"),
    ):
        # Windows rejects fsync on a read-only CRT descriptor.  Opening the
        # already-written staging file in update mode changes no bytes but
        # provides a flushable descriptor on both Windows and Linux.
        with open(filesystem_value, "r+b") as handle:
            os.fsync(handle.fileno())
    for _relative, filesystem_value in sorted(
        directories,
        key=lambda value: (
            len(value[0].split("/")) if value[0] else 0,
            value[0].encode("utf-8"),
        ),
        reverse=True,
    ):
        _fsync_directory(Path(filesystem_value))


def _cleanup_release_stage(stage: Path, *, parent: Path) -> None:
    """Delete only a verified private staging directory after failure."""

    resolved_stage = stage.resolve(strict=False)
    resolved_parent = parent.resolve(strict=True)
    if (
        resolved_stage.parent != resolved_parent
        or not resolved_stage.name.startswith(".staging-")
    ):
        raise common.ContractError(
            "Refusing to clean an unverified dataset staging path"
        )
    stage_filesystem_path = common.filesystem_path(stage)
    if os.path.exists(stage_filesystem_path):
        shutil.rmtree(stage_filesystem_path)


def _validated_release_parent(
    policy: Mapping[str, Any],
    *,
    mode: str,
    release_name: str,
) -> Path:
    """Create one release parent and reject links, junctions, and escapes."""

    root = common.mode_output_root(policy, mode)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve(strict=True)
    expected_parent = resolved_root / release_name
    parent = root / release_name
    parent_value = common.filesystem_path(parent)
    if os.path.lexists(parent_value):
        if not os.path.isdir(parent_value):
            raise common.ContractError(
                "Dataset release parent exists but is not a directory"
            )
        resolved_parent = parent.resolve(strict=True)
        if parent.is_symlink() or resolved_parent != expected_parent:
            raise common.ContractError(
                "Dataset release parent may not be a symlink, junction, or escape"
            )
    else:
        os.mkdir(parent_value)
        resolved_parent = parent.resolve(strict=True)
        if resolved_parent != expected_parent:
            raise common.ContractError(
                "Created dataset release parent resolved outside its root"
            )
    return expected_parent


def write_split_release(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    release_name: str,
    result: Mapping[str, Any],
) -> Path:
    if mode != "development_smoke":
        raise common.ContractError(
            "This combined writer is development-smoke only; formal "
            "artifacts require the split-private immutable sealer"
        )
    if split not in SPLITS:
        raise common.ContractError("Unknown dataset release split")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", release_name):
        raise common.ContractError("Invalid dataset release directory name")
    expected_result_keys = {
        "tables",
        "placebos",
        "support_comparability_preflight",
        "world_digests",
        "aggregate_integrity_receipts",
        "aggregate_parent_projections",
        "structural_audit_receipt",
    }
    if not isinstance(result, Mapping) or set(result) != expected_result_keys:
        raise common.ContractError(
            "Dataset release result envelope drift"
        )
    snapshot = copy.deepcopy(dict(result))
    expected_world_count = int(
        policy["modes"][mode]["world_counts"][split]
    )
    _validate_split_payload(
        policy,
        split=split,
        expected_world_count=expected_world_count,
        payload=snapshot["tables"],
    )
    template, fixture, style_profile = _load_release_inputs(
        policy, mode=mode
    )
    regenerated = build_split_payload(
        policy,
        mode=mode,
        split=split,
        template=template,
        fixture=fixture,
        style_profile=style_profile,
    )
    if common.canonical_json_bytes(snapshot) != common.canonical_json_bytes(
        regenerated
    ):
        raise common.ContractError(
            "Dataset release snapshot deterministic regeneration mismatch"
        )
    # Publish only the fresh deterministic snapshot, never caller-owned
    # mutable objects that could change after validation.
    result = regenerated
    parent = _validated_release_parent(
        policy,
        mode=mode,
        release_name=release_name,
    )
    target = parent / split
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing split release: {target}"
        )
    stage = parent / f".staging-{split}-{uuid.uuid4().hex}"
    if stage.exists():
        raise common.ContractError("Unexpected dataset staging collision")
    stage.mkdir()
    published = False
    try:
        files = _write_table_set(
            policy, stage=stage, payload=result["tables"]
        )
        files.extend(
            _write_placebo_set(
                policy,
                mode=mode,
                split=split,
                stage=stage,
                placebos=result.get("placebos", []),
                support_preflight=result.get(
                    "support_comparability_preflight"
                ),
                m2_identity33_all_pairs=result["tables"][
                    "identity33_all_pairs"
                ],
                candidate_pairs=result["tables"]["candidate_pairs"],
                complete_pair_endpoints=result["tables"][
                    "complete_model_pair_endpoints"
                ],
                m1_integrity_receipt=result[
                    "aggregate_integrity_receipts"
                ].get("m1_derangement_integrity"),
            )
        )
        files.extend(
            _write_aggregate_integrity_set(
                policy,
                mode=mode,
                split=split,
                stage=stage,
                receipts=result["aggregate_integrity_receipts"],
                parent_projections=result[
                    "aggregate_parent_projections"
                ],
                structural_audit_receipt=result[
                    "structural_audit_receipt"
                ],
            )
        )
        files.sort(key=lambda row: str(row["path"]).encode("utf-8"))
        policy_path = common.DEFAULT_POLICY_PATH
        contract_path = common.repo_path(str(policy["contract"]["path"]))
        template_path = common.repo_path(
            str(policy["template_library"]["path"])
        )
        fixture_path = common.repo_path(
            str(
                policy["identity_design"][
                    "role_template_parser_flag_fixture"
                ]["path"]
            )
        )
        manifest: dict[str, Any] = {
            "version": (
                "2026-07-29-step28-v13-split-dataset-manifest-v5-draft"
            ),
            "status": "DEVELOPMENT_SMOKE_PASS_NOT_SCIENTIFIC_EVIDENCE",
            "mode": mode,
            "split": split,
            "run_id": policy["modes"][mode]["run_id"],
            "release_name": release_name,
            "scientific_metrics_produced": False,
            "policy_sha256": common.sha256_file(policy_path),
            "contract_sha256": common.sha256_file(contract_path),
            "template_sha256": common.sha256_file(template_path),
            "fixture_sha256": common.sha256_file(fixture_path),
            "producer_sha256": _producer_hashes(),
            "split_payload_digest_sha256": common.canonical_sha256(
                result
            ),
            "world_count": len(result["tables"]["worlds"]),
            "seller_count": len(result["tables"]["sellers"]),
            "item_count": len(result["tables"]["items"]),
            "complete_pair_count": len(
                result["tables"]["complete_model_pair_endpoints"]
            ),
            "candidate_pair_count": len(
                result["tables"]["candidate_pairs"]
            ),
            "candidate_sampling_audit_count": len(
                result["tables"]["candidate_sampling_audit"]
            ),
            "placebo_replicate_count": len(result.get("placebos", [])),
            "aggregate_integrity_receipt_count": len(
                result["aggregate_integrity_receipts"]
            ),
            "aggregate_parent_projection_count": len(
                result["aggregate_parent_projections"]
            ),
            "structural_audit_receipt_sha256": result[
                "structural_audit_receipt"
            ]["canonical_self_hash"],
            "m1_support_comparability_pass": (
                result.get("support_comparability_preflight") is not None
                and result["support_comparability_preflight"][
                    "all_five_primary_validity_pass"
                ]
                is True
            )
            if split == "train"
            else None,
            "parsed_identity_occurrence_count": len(
                result["tables"]["parsed_identity_occurrences"]
            ),
            "history_safe_occurrence_count": len(
                result["tables"]["history_safe_occurrences"]
            ),
            "history_item_index_count": len(
                result["tables"]["history_item_index"]
            ),
            "history_projection_attestation_count": len(
                result["tables"]["history_projection_attestations"]
            ),
            "identity33_all_pair_count": len(
                result["tables"]["identity33_all_pairs"]
            ),
            "world_digests": result["world_digests"],
            "files": files,
            "parent_manifests": [],
            "formal_use_forbidden": True,
        }
        manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
        common.write_json(stage / "split_manifest.json", manifest)
        _fsync_release_tree(stage)
        common.atomic_rename_no_replace(stage, target)
        published = True
        try:
            _fsync_directory(parent)
        except OSError as error:
            raise common.ContractError(
                "Dataset split output was published but parent directory "
                f"fsync failed: {target}"
            ) from error
    except BaseException:
        if not published:
            _cleanup_release_stage(stage, parent=parent)
        raise
    return target


def _materialized_split_parent_record(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    release_name: str,
    parent: Path,
    expected_payload_digest: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate one published child before binding it into the parent."""

    root = parent / split
    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise common.ContractError(
            f"Complete release is missing split manifest: {split}"
        )
    manifest = common.load_json(manifest_path)
    expected_self_hash = common.canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "canonical_self_hash"
        }
    )
    current_parent_hashes = {
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY_PATH),
        "contract_sha256": common.sha256_file(
            common.repo_path(str(policy["contract"]["path"]))
        ),
        "template_sha256": common.sha256_file(
            common.repo_path(str(policy["template_library"]["path"]))
        ),
        "fixture_sha256": common.sha256_file(
            common.repo_path(
                str(
                    policy["identity_design"][
                        "role_template_parser_flag_fixture"
                    ]["path"]
                )
            )
        ),
    }
    current_producer_hashes = _producer_hashes()
    if (
        manifest.get("version")
        != "2026-07-29-step28-v13-split-dataset-manifest-v5-draft"
        or manifest.get("status")
        != "DEVELOPMENT_SMOKE_PASS_NOT_SCIENTIFIC_EVIDENCE"
        or manifest.get("canonical_self_hash") != expected_self_hash
        or manifest.get("mode") != mode
        or manifest.get("split") != split
        or manifest.get("run_id") != policy["modes"][mode]["run_id"]
        or manifest.get("release_name") != release_name
        or manifest.get("scientific_metrics_produced") is not False
        or manifest.get("formal_use_forbidden") is not True
        or manifest.get("parent_manifests") != []
        or manifest.get("producer_sha256") != current_producer_hashes
        or manifest.get("split_payload_digest_sha256")
        != expected_payload_digest
        or any(
            manifest.get(key) != value
            for key, value in current_parent_hashes.items()
        )
    ):
        raise common.ContractError(
            f"Published split manifest is not parent-eligible: {split}"
        )
    file_rows = manifest.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise common.ContractError("Published split file manifest is empty")
    registered: dict[str, Mapping[str, Any]] = {}
    for row in file_rows:
        relative = str(row["path"])
        if relative in registered:
            raise common.ContractError(
                "Published split manifest contains a duplicate path"
            )
        registered[relative] = row
        path = root / Path(relative)
        stat_result = os.stat(common.filesystem_path(path))
        if (
            stat_result.st_size != int(row["size_bytes"])
            or common.sha256_file(path) != row["sha256"]
        ):
            raise common.ContractError(
                f"Published split artifact drift: {split}/{relative}"
            )
    actual: set[str] = set()
    root_value = common.filesystem_path(root)
    for current, directory_names, file_names in os.walk(root_value):
        directory_names.sort()
        file_names.sort()
        for name in file_names:
            actual.add(
                os.path.relpath(
                    os.path.join(current, name),
                    root_value,
                ).replace(os.sep, "/")
            )
    if actual != {*registered, "split_manifest.json"}:
        raise common.ContractError(
            f"Published split physical file set drift: {split}"
        )
    return (
        {
            "role": f"split_{split}",
            "file_sha256": common.sha256_file(manifest_path),
            "content_sha256": expected_self_hash,
        },
        manifest,
    )


def write_release_manifest(
    policy: Mapping[str, Any],
    *,
    mode: str,
    release_name: str,
    split_payload_digests: Mapping[str, str],
) -> Path:
    """Bind all four immutable split releases and the exhaustive fixture."""

    release_contract = policy["development_complete_release"]
    required_splits = list(release_contract["required_split_order"])
    if (
        release_name != release_contract["release_name"]
        or required_splits != list(SPLITS)
        or list(split_payload_digests) != required_splits
    ):
        raise common.ContractError(
            "Complete release name or split order differs from policy"
        )
    parent = _validated_release_parent(
        policy,
        mode=mode,
        release_name=release_name,
    )
    manifest_path = parent / str(
        release_contract["manifest_filename"]
    )
    if os.path.lexists(common.filesystem_path(manifest_path)):
        raise FileExistsError(
            f"Refusing to overwrite complete release manifest: {manifest_path}"
        )
    top_level_entries = {
        entry.name
        for entry in os.scandir(common.filesystem_path(parent))
    }
    if top_level_entries != set(SPLITS):
        raise common.ContractError(
            "Complete release parent has missing or unexpected entries"
        )
    parent_records: list[dict[str, str]] = []
    split_records: list[dict[str, Any]] = []
    for split in SPLITS:
        parent_record, split_manifest = (
            _materialized_split_parent_record(
                policy,
                mode=mode,
                split=split,
                release_name=release_name,
                parent=parent,
                expected_payload_digest=str(
                    split_payload_digests[split]
                ),
            )
        )
        parent_records.append(parent_record)
        split_records.append(
            {
                "split": split,
                "manifest_path": f"{split}/split_manifest.json",
                "manifest_file_sha256": parent_record["file_sha256"],
                "manifest_content_sha256": parent_record[
                    "content_sha256"
                ],
                "split_payload_digest_sha256": str(
                    split_payload_digests[split]
                ),
                "world_count": int(split_manifest["world_count"]),
                "seller_count": int(split_manifest["seller_count"]),
                "item_count": int(split_manifest["item_count"]),
                "candidate_pair_count": int(
                    split_manifest["candidate_pair_count"]
                ),
            }
        )
    parent_records.sort(key=lambda row: row["role"].encode("utf-8"))
    fixture_path = common.repo_path(
        str(
            policy["identity_design"][
                "role_template_parser_flag_fixture"
            ]["path"]
        )
    )
    fixture = common.load_json(fixture_path)
    fixture_result_path, fixture_result = _validate_fixture_result(
        fixture,
        fixture_path=fixture_path,
    )
    manifest: dict[str, Any] = {
        "version": (
            release_contract["manifest_version"]
        ),
        "status": release_contract["required_status"],
        "mode": mode,
        "run_id": policy["modes"][mode]["run_id"],
        "release_name": release_name,
        "complete_split_order": list(SPLITS),
        "scientific_metrics_produced": False,
        "formal_use_forbidden": True,
        "m0_exact_mount_allowlist_per_split": list(
            release_contract["m0_exact_mount_allowlist_per_split"]
        ),
        "m0_observed_directory_mount_forbidden": bool(
            release_contract["m0_observed_directory_mount_forbidden"]
        ),
        "policy_sha256": common.sha256_file(common.DEFAULT_POLICY_PATH),
        "contract_sha256": common.sha256_file(
            common.repo_path(str(policy["contract"]["path"]))
        ),
        "template_sha256": common.sha256_file(
            common.repo_path(str(policy["template_library"]["path"]))
        ),
        "fixture_sha256": common.sha256_file(fixture_path),
        "producer_sha256": _producer_hashes(),
        "preflight_fixture_result": {
            "repository_path": fixture_result_path.relative_to(
                common.ROOT
            ).as_posix(),
            "file_sha256": common.sha256_file(fixture_result_path),
            "size_bytes": os.stat(
                common.filesystem_path(fixture_result_path)
            ).st_size,
            "status": fixture_result["status"],
            "case_count": int(fixture_result["case_count"]),
            "case_outcome_manifest_sha256": fixture_result[
                "case_outcome_manifest_sha256"
            ],
            "runner_sha256": fixture_result["runner_sha256"],
        },
        "parent_manifests": parent_records,
        "splits": split_records,
        "files": [],
    }
    manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
    common.write_json(manifest_path, manifest)
    try:
        _fsync_directory(parent)
    except OSError as error:
        raise common.ContractError(
            "Complete release manifest was published but parent directory "
            "durability is unknown; verify the immutable manifest and "
            "repeat only the parent fsync recovery step"
        ) from error
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_policy_argument(parser)
    common.add_mode_argument(parser)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--release-name", default=DEFAULT_RELEASE_NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.policy).resolve() != common.DEFAULT_POLICY_PATH.resolve():
        raise common.ContractError(
            "Development aggregate receipts bind the exact default policy "
            "file; alternate policy paths are forbidden"
        )
    policy = common.load_policy(args.policy, mode=args.mode)
    integrity_receipts_mod.validate_deployment_contract(policy)
    if args.mode == "formal":
        raise common.ContractError(
            "Formal generation requires the not-yet-released custody launcher"
        )
    template, fixture, style_profile = _load_release_inputs(
        policy,
        mode=args.mode,
    )
    if args.validate_config_only:
        print("Step28-v13 smoke dataset generator configuration is valid")
        return
    selected = SPLITS if args.split == "all" else (args.split,)
    split_payload_digests: dict[str, str] = {}
    for split in selected:
        result = build_split_payload(
            policy,
            mode=args.mode,
            split=split,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
        )
        digest = common.canonical_sha256(result)
        split_payload_digests[split] = digest
        if args.dry_run:
            print(
                f"Step28-v13 {split} dry-run PASS: "
                f"worlds={len(result['tables']['worlds'])} "
                f"items={len(result['tables']['items'])} "
                f"digest={digest}"
            )
            continue
        target = write_split_release(
            policy,
            mode=args.mode,
            split=split,
            release_name=args.release_name,
            result=result,
        )
        print(
            f"Step28-v13 {split} dataset written: {target} digest={digest}"
        )
    if not args.dry_run and selected == SPLITS:
        release_manifest = write_release_manifest(
            policy,
            mode=args.mode,
            release_name=args.release_name,
            split_payload_digests=split_payload_digests,
        )
        print(
            "Step28-v13 complete release manifest written: "
            f"{release_manifest}"
        )


if __name__ == "__main__":
    main()
