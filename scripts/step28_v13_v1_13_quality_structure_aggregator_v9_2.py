#!/usr/bin/env python3
"""V9.2 label-free structure gates layered on the frozen V9 aggregator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_quality_gate_registry_v9_2 as registry
import step28_v13_v1_13_quality_structure_aggregator_v9 as v9
import step28_v13_v1_13_scientific_world_v9_2 as scientific_world
from step28_v13_v1_13_style_derangement import build_style_source_derangement


VERSION = "2026-08-23-step28-v13-v1-13-quality-structure-aggregator-v9-2"
HEX_DIGITS = frozenset("0123456789abcdef")
REPLAY_INVARIANT_NAMES = (
    "non_effective_style_ast",
    "public_item_non_text",
    "model_item_key_and_empty_pattern",
    "pair_endpoint_order",
    "identity33",
    "identity_slot_core",
    "noise_slot_core",
    "override_audit",
    "clone_endpoint_and_direction",
    "seller_profile_keyset",
)
REPLAY_FIELDS = (
    "version",
    "world_uid",
    "mapping",
    "candidate_key_sha256",
    "forbidden_capability_mounted",
    "double_replay",
    "invariants",
    "model_inputs",
    "style_structure",
    "labels_or_retrieval_truth_read",
    "quality_result_read_count",
    "canonical_self_hash",
)
COUNTERFACTUAL_FORBIDDEN_CAPABILITIES = (
    "controller_membership",
    "pair_labels",
    "qrels",
    "audit_a_truth",
    "audit_b_truth",
    "quality_results",
)
MODEL_SURFACES = (
    "surface_full",
    "surface_code_masked",
    "surface_code_neutralized",
    "surface_style_deranged_full",
)
SURFACE_HASH_FIELDS = {
    "surface_full": ("full_item_sha256", "full_profile_sha256"),
    "surface_code_masked": ("masked_item_sha256", "masked_profile_sha256"),
    "surface_code_neutralized": (
        "neutral_item_sha256",
        "neutral_profile_sha256",
    ),
    "surface_style_deranged_full": (
        "counterfactual_full_item_sha256",
        "counterfactual_full_profile_sha256",
    ),
}
EXTENSION_FIELDS = (
    "version",
    "base_v9_structure_version",
    *(field for field in v9.STRUCTURE_AUDIT_FIELDS if field != "version"),
    "counterfactual_full_item_sha256",
    "counterfactual_full_profile_sha256",
    "counterfactual_replay",
    "shared_identity33_sha256",
    "shared_text_probe_eligibility_sha256",
    "shared_identity_mechanism_sha256",
    "m1_mapping_commitments",
    "m1_mapping_commitment_bundle_sha256",
    "model_input_file_count",
    "original_author_model_input_file_count",
    "counterfactual_model_input_file_count",
    "labels_or_retrieval_truth_materialized_before_audit",
    "v9_2_extension_sha256",
)


class QualityStructureAggregationV92Error(v9.QualityStructureAggregationError):
    """Raised when the V9.2 structure extension is incomplete or inconsistent."""


def _required_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise QualityStructureAggregationV92Error(f"{name} must be lowercase SHA-256")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityStructureAggregationV92Error(
            f"{name} must be a nonnegative integer"
        )
    return value


def _base_v9_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if set(row) != set(EXTENSION_FIELDS):
        raise QualityStructureAggregationV92Error(
            "V9.2 structure row exact schema drift"
        )
    output = {
        field: row[field]
        for field in v9.STRUCTURE_AUDIT_FIELDS
        if field != "version"
    }
    output = {"version": row["base_v9_structure_version"], **output}
    return output


def _validate_equal_commitment(value: object, *, name: str) -> int:
    if not isinstance(value, Mapping) or set(value) != {
        "source_sha256",
        "counterfactual_sha256",
        "equal",
    }:
        raise QualityStructureAggregationV92Error(
            f"{name} equality commitment schema drift"
        )
    source = _required_sha256(value["source_sha256"], name=f"{name} source")
    counterfactual = _required_sha256(
        value["counterfactual_sha256"], name=f"{name} counterfactual"
    )
    expected = source == counterfactual
    if value["equal"] is not expected:
        raise QualityStructureAggregationV92Error(
            f"{name} equality boolean disagrees with its hashes"
        )
    return int(not expected)


def _validate_counterfactual_replay(
    replay: object,
    *,
    split: str,
    world_uid: str,
    seller_uids: Sequence[str],
) -> dict[str, int]:
    if not isinstance(replay, Mapping):
        raise QualityStructureAggregationV92Error("Counterfactual replay is absent")
    if set(replay) != set(REPLAY_FIELDS):
        raise QualityStructureAggregationV92Error(
            "Counterfactual replay exact schema drift"
        )
    replay_without_hash = dict(replay)
    supplied_self_hash = replay_without_hash.pop("canonical_self_hash", None)
    if supplied_self_hash != common.canonical_sha256(replay_without_hash):
        raise QualityStructureAggregationV92Error(
            "Counterfactual replay self-hash drift"
        )
    if str(replay.get("world_uid", "")) != world_uid:
        raise QualityStructureAggregationV92Error(
            "Counterfactual replay world binding drift"
        )
    _required_sha256(replay.get("candidate_key_sha256"), name="candidate key")
    mapping = replay.get("mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != {
        "attempt",
        "seller_set_sha256",
        "mapping_sha256",
        "target_source_pairs",
        "fixed_point_count",
    }:
        raise QualityStructureAggregationV92Error(
            "Counterfactual mapping schema drift"
        )
    expected_mapping = build_style_source_derangement(
        split=split,
        world_uid=world_uid,
        seller_uids=seller_uids,
    )
    expected_mapping_payload = {
        "attempt": expected_mapping.attempt,
        "seller_set_sha256": expected_mapping.seller_set_sha256,
        "mapping_sha256": expected_mapping.mapping_sha256,
        "target_source_pairs": [list(row) for row in expected_mapping.target_source_pairs],
        "fixed_point_count": 0,
    }
    if dict(mapping) != expected_mapping_payload:
        raise QualityStructureAggregationV92Error(
            "Persisted style mapping is not the public-ID-only frozen mapping"
        )
    forbidden = replay.get("forbidden_capability_mounted")
    if (
        not isinstance(forbidden, Mapping)
        or set(forbidden) != set(COUNTERFACTUAL_FORBIDDEN_CAPABILITIES)
        or any(type(value) is not bool for value in forbidden.values())
    ):
        raise QualityStructureAggregationV92Error(
            "Counterfactual capability receipt schema drift"
        )
    double_replay = replay.get("double_replay")
    if not isinstance(double_replay, Mapping) or set(double_replay) != {
        "independent_production_replay_count",
        "canonical_commitment_sha256",
        "byte_identical",
    }:
        raise QualityStructureAggregationV92Error(
            "Counterfactual double-replay receipt schema drift"
        )
    _required_sha256(
        double_replay["canonical_commitment_sha256"],
        name="counterfactual replay commitment",
    )
    invariants = replay.get("invariants")
    if not isinstance(invariants, Mapping) or set(invariants) != set(
        REPLAY_INVARIANT_NAMES
    ):
        raise QualityStructureAggregationV92Error(
            "Counterfactual invariant registry drift"
        )
    invariant_mismatches = sum(
        _validate_equal_commitment(invariants[name], name=name)
        for name in REPLAY_INVARIANT_NAMES
    )
    model_inputs = replay.get("model_inputs")
    if not isinstance(model_inputs, Mapping) or set(model_inputs) != {
        "original_full_items_sha256",
        "original_full_profiles_sha256",
        "counterfactual_full_items_sha256",
        "counterfactual_full_profiles_sha256",
    }:
        raise QualityStructureAggregationV92Error(
            "Counterfactual model-input commitment schema drift"
        )
    for name, value in model_inputs.items():
        _required_sha256(value, name=name)
    style = replay.get("style_structure")
    expected_style_fields = {
        "minimum_distinct_style_factor_tuples_required",
        "observed_distinct_style_factor_tuple_count",
        "minimum_visible_carrier_fields_per_seller_required",
        "minimum_observed_visible_carrier_fields_per_seller",
        "visible_carrier_fields_by_seller_sha256",
        "renderer_style_item_read_count",
        "renderer_style_field_read_count",
        "renderer_style_factor_read_count",
        "minimum_actual_style_factor_reads_per_seller",
        "renderer_style_factor_reads_by_seller_sha256",
        "renderer_style_read_audit_sha256",
        "effective_style_uid_changed_seller_count",
        "effective_style_factor_tuple_changed_seller_count",
        "visible_change_seller_count",
        "title_change_count",
        "description_change_count",
        "visible_change_seller_set_sha256",
    }
    if not isinstance(style, Mapping) or set(style) != expected_style_fields:
        raise QualityStructureAggregationV92Error(
            "Counterfactual style-structure receipt schema drift"
        )
    for name in (
        "visible_carrier_fields_by_seller_sha256",
        "renderer_style_factor_reads_by_seller_sha256",
        "renderer_style_read_audit_sha256",
        "visible_change_seller_set_sha256",
    ):
        _required_sha256(style[name], name=name)
    integer_style = {
        name: _nonnegative_int(value, name=name)
        for name, value in style.items()
        if name not in {
            "visible_carrier_fields_by_seller_sha256",
            "renderer_style_factor_reads_by_seller_sha256",
            "renderer_style_read_audit_sha256",
            "visible_change_seller_set_sha256",
        }
    }
    if (
        integer_style["minimum_distinct_style_factor_tuples_required"] != 2
        or integer_style["minimum_visible_carrier_fields_per_seller_required"] != 1
    ):
        raise QualityStructureAggregationV92Error(
            "Counterfactual style threshold commitment drift"
        )
    labels_read = replay.get("labels_or_retrieval_truth_read")
    quality_reads = _nonnegative_int(
        replay.get("quality_result_read_count"), name="quality-result read count"
    )
    if type(labels_read) is not bool:
        raise QualityStructureAggregationV92Error(
            "Counterfactual truth-read receipt type drift"
        )
    return {
        "mapping_count_mismatch": 0,
        "fixed_point_count": int(mapping["fixed_point_count"]),
        "replay_count_mismatch": int(
            double_replay["independent_production_replay_count"] != 2
        ),
        "replay_byte_mismatch": int(double_replay["byte_identical"] is not True),
        "invariant_mismatch_count": invariant_mismatches,
        "minimum_distinct_style_factor_tuple_count": integer_style[
            "observed_distinct_style_factor_tuple_count"
        ],
        "minimum_visible_carrier_fields_per_seller": integer_style[
            "minimum_observed_visible_carrier_fields_per_seller"
        ],
        "minimum_actual_style_factor_reads_per_seller": integer_style[
            "minimum_actual_style_factor_reads_per_seller"
        ],
        "forbidden_capability_mounted_count": sum(
            int(value) for value in forbidden.values()
        ),
        "truth_or_retrieval_read_count": int(labels_read),
        "quality_result_read_count": quality_reads,
        "original_full_items_sha256": model_inputs["original_full_items_sha256"],
        "original_full_profiles_sha256": model_inputs[
            "original_full_profiles_sha256"
        ],
        "counterfactual_full_items_sha256": model_inputs[
            "counterfactual_full_items_sha256"
        ],
        "counterfactual_full_profiles_sha256": model_inputs[
            "counterfactual_full_profiles_sha256"
        ],
        "identity33_source_sha256": invariants["identity33"]["source_sha256"],
    }


def _aggregate(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    eligibility_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    model_surface_rows_by_split: Mapping[
        str,
        Mapping[
            str,
            tuple[
                Sequence[Mapping[str, Any]],
                Sequence[Mapping[str, Any]],
            ],
        ],
    ],
    expected_world_counts: Mapping[str, int],
    expected_sellers_per_world: int,
    maximum_position_deviation: float,
    enforce_position_margin: bool,
    claim_boundary: str,
) -> dict[str, Any]:
    if registry.GATE_REGISTRY_SHA256 != hashlib.sha256(
        registry.GATE_REGISTRY_BYTES
    ).hexdigest():
        raise QualityStructureAggregationV92Error("Gate-registry self-binding drift")
    if (
        set(eligibility_rows_by_split) != set(v9.SPLITS)
        or set(model_surface_rows_by_split) != set(v9.SPLITS)
        or any(
            tuple(model_surface_rows_by_split[split]) != MODEL_SURFACES
            for split in v9.SPLITS
        )
    ):
        raise QualityStructureAggregationV92Error(
            "V9.2 persisted model/eligibility split universe drift"
        )
    base_rows_by_split: dict[str, tuple[dict[str, Any], ...]] = {}
    for split in v9.SPLITS:
        rows = tuple(structure_rows_by_split.get(split, ()))
        base_rows_by_split[split] = tuple(_base_v9_row(row) for row in rows)
    base = v9._aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=base_rows_by_split,
        expected_world_counts=expected_world_counts,
        expected_sellers_per_world=expected_sellers_per_world,
        maximum_position_deviation=maximum_position_deviation,
        enforce_position_margin=enforce_position_margin,
        claim_boundary=claim_boundary,
    )

    metric_values: dict[str, float | int] = {
        "train_code_character_position_maximum_deviation": float(
            base["split_receipts"]["train"][
                "code_character_position_maximum_absolute_deviation"
            ]
        ),
        "development_code_character_position_maximum_deviation": float(
            base["split_receipts"]["development"][
                "code_character_position_maximum_absolute_deviation"
            ]
        ),
    }
    for metric in registry.ZERO_TOLERANCE_STRUCTURE_METRICS:
        metric_values[metric] = int(base["zero_tolerance_counts"].get(metric, 0))

    aggregate = {
        "model_input_file_count_mismatch_world_count": 0,
        "style_derangement_mapping_count_mismatch_world_count": 0,
        "minimum_distinct_style_factor_tuple_count": math.inf,
        "minimum_visible_carrier_fields_per_seller": math.inf,
        "minimum_actual_style_factor_reads_per_seller": math.inf,
        "style_derangement_fixed_point_count": 0,
        "independent_production_replay_count_mismatch_world_count": 0,
        "independent_production_replay_byte_mismatch_world_count": 0,
        "cross_branch_invariant_mismatch_count": 0,
        "identity_mechanism_commitment_missing_world_count": 0,
        "shared_text_eligibility_commitment_mismatch_world_count": 0,
        "m1_mapping_commitment_count_mismatch_world_count": 0,
        "m1_distinct_mapping_commitment_count_mismatch_world_count": 0,
        "counterfactual_forbidden_capability_mounted_count": 0,
        "counterfactual_truth_or_retrieval_read_count": 0,
        "counterfactual_quality_result_read_count": 0,
        "persisted_model_input_hash_mismatch_count": 0,
    }
    world_receipts: list[dict[str, Any]] = []
    for split in v9.SPLITS:
        public_by_world: dict[str, list[Mapping[str, Any]]] = {}
        for row in public_rows_by_split[split]:
            public_by_world.setdefault(str(row["world_uid"]), []).append(row)
        seller_world = {
            str(row["seller_uid"]): str(row["world_uid"])
            for row in public_rows_by_split[split]
        }
        eligibility_by_world: dict[str, list[Mapping[str, Any]]] = {}
        for row in eligibility_rows_by_split[split]:
            if not isinstance(row, Mapping) or set(row) != {
                "world_uid",
                "canonical_pair_uid",
                "text_probe_eligible",
            } or type(row["text_probe_eligible"]) is not bool:
                raise QualityStructureAggregationV92Error(
                    "Persisted text eligibility schema drift"
                )
            eligibility_by_world.setdefault(str(row["world_uid"]), []).append(row)
        surface_world_rows: dict[
            str, dict[str, tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]]
        ] = {}
        for surface in MODEL_SURFACES:
            items, profiles = model_surface_rows_by_split[split][surface]
            grouped_items: dict[str, list[Mapping[str, Any]]] = {}
            grouped_profiles: dict[str, list[Mapping[str, Any]]] = {}
            for item in items:
                world = str(item.get("world_uid", ""))
                grouped_items.setdefault(world, []).append(item)
            for profile in profiles:
                seller_uid = str(profile.get("seller_uid", ""))
                world = seller_world.get(seller_uid)
                if world is None:
                    raise QualityStructureAggregationV92Error(
                        "Persisted model profile seller is outside public world rows"
                    )
                grouped_profiles.setdefault(world, []).append(profile)
            if set(grouped_items) != set(public_by_world) or set(
                grouped_profiles
            ) != set(public_by_world):
                raise QualityStructureAggregationV92Error(
                    "Persisted model surface world universe drift"
                )
            surface_world_rows[surface] = {
                world_uid: (
                    sorted(
                        grouped_items[world_uid],
                        key=lambda value: str(value["item_uid"]).encode("utf-8"),
                    ),
                    sorted(
                        grouped_profiles[world_uid],
                        key=lambda value: str(value["seller_uid"]).encode("utf-8"),
                    ),
                )
                for world_uid in public_by_world
            }
        for row in structure_rows_by_split[split]:
            row_without_extension_hash = dict(row)
            extension_sha256 = row_without_extension_hash.pop(
                "v9_2_extension_sha256", None
            )
            if extension_sha256 != common.canonical_sha256(row_without_extension_hash):
                raise QualityStructureAggregationV92Error(
                    "V9.2 structure extension self-hash drift"
                )
            world_uid = str(row["world_uid"])
            public_rows = public_by_world.get(world_uid, [])
            seller_uids = tuple(str(value["seller_uid"]) for value in public_rows)
            replay = _validate_counterfactual_replay(
                row["counterfactual_replay"],
                split=split,
                world_uid=world_uid,
                seller_uids=seller_uids,
            )
            aggregate["model_input_file_count_mismatch_world_count"] += int(
                row["model_input_file_count"] != 8
                or row["original_author_model_input_file_count"] != 6
                or row["counterfactual_model_input_file_count"] != 2
            )
            aggregate["style_derangement_mapping_count_mismatch_world_count"] += replay[
                "mapping_count_mismatch"
            ]
            aggregate["minimum_distinct_style_factor_tuple_count"] = min(
                aggregate["minimum_distinct_style_factor_tuple_count"],
                replay["minimum_distinct_style_factor_tuple_count"],
            )
            aggregate["minimum_visible_carrier_fields_per_seller"] = min(
                aggregate["minimum_visible_carrier_fields_per_seller"],
                replay["minimum_visible_carrier_fields_per_seller"],
            )
            aggregate["minimum_actual_style_factor_reads_per_seller"] = min(
                aggregate["minimum_actual_style_factor_reads_per_seller"],
                replay["minimum_actual_style_factor_reads_per_seller"],
            )
            aggregate["style_derangement_fixed_point_count"] += replay[
                "fixed_point_count"
            ]
            aggregate[
                "independent_production_replay_count_mismatch_world_count"
            ] += replay["replay_count_mismatch"]
            aggregate[
                "independent_production_replay_byte_mismatch_world_count"
            ] += replay["replay_byte_mismatch"]
            aggregate["cross_branch_invariant_mismatch_count"] += replay[
                "invariant_mismatch_count"
            ]
            aggregate["counterfactual_forbidden_capability_mounted_count"] += replay[
                "forbidden_capability_mounted_count"
            ]
            aggregate["counterfactual_truth_or_retrieval_read_count"] += replay[
                "truth_or_retrieval_read_count"
            ]
            aggregate["counterfactual_quality_result_read_count"] += replay[
                "quality_result_read_count"
            ]
            mechanism_sha = row.get("shared_identity_mechanism_sha256")
            aggregate["identity_mechanism_commitment_missing_world_count"] += int(
                not isinstance(mechanism_sha, str)
                or len(mechanism_sha) != 64
                or any(character not in HEX_DIGITS for character in mechanism_sha)
            )
            eligibility_sha = _required_sha256(
                row["shared_text_probe_eligibility_sha256"],
                name="shared text eligibility",
            )
            actual_eligibility_rows = eligibility_by_world.get(world_uid)
            if actual_eligibility_rows is None:
                raise QualityStructureAggregationV92Error(
                    "Persisted text eligibility world is absent"
                )
            aggregate[
                "shared_text_eligibility_commitment_mismatch_world_count"
            ] += int(
                eligibility_sha
                != common.canonical_sha256(actual_eligibility_rows)
            )
            commitments = row.get("m1_mapping_commitments")
            if not isinstance(commitments, list):
                raise QualityStructureAggregationV92Error(
                    "M1 mapping commitments must be a list"
                )
            repeat_ids = [
                value.get("repeat_id") if isinstance(value, Mapping) else None
                for value in commitments
            ]
            hashes = [
                value.get("mapping_sha256") if isinstance(value, Mapping) else None
                for value in commitments
            ]
            aggregate["m1_mapping_commitment_count_mismatch_world_count"] += int(
                repeat_ids != ["r01", "r02", "r03", "r04", "r05"]
            )
            valid_hashes = all(
                isinstance(value, str)
                and len(value) == 64
                and not any(character not in HEX_DIGITS for character in value)
                for value in hashes
            )
            aggregate[
                "m1_distinct_mapping_commitment_count_mismatch_world_count"
            ] += int(not valid_hashes or len(set(hashes)) != 5)
            expected_commitments = scientific_world.build_m1_mapping_commitments(
                [
                    {
                        "world_uid": world_uid,
                        "seller_uid_left": pair_uid.split("||", 1)[0],
                        "seller_uid_right": pair_uid.split("||", 1)[1],
                        "canonical_pair_uid": pair_uid,
                    }
                    for pair_uid in sorted(
                        (
                            f"{left}||{right}"
                            for index, left in enumerate(
                                sorted(seller_uids, key=lambda value: value.encode("utf-8"))
                            )
                            for right in sorted(
                                seller_uids, key=lambda value: value.encode("utf-8")
                            )[index + 1 :]
                        ),
                        key=lambda value: value.encode("utf-8"),
                    )
                ],
                world_uid=world_uid,
            )
            if commitments != list(expected_commitments):
                raise QualityStructureAggregationV92Error(
                    "M1 mapping commitments do not match public endpoint IDs"
                )
            if row.get("m1_mapping_commitment_bundle_sha256") != common.canonical_sha256(
                commitments
            ):
                raise QualityStructureAggregationV92Error(
                    "M1 mapping commitment bundle hash drift"
                )
            if row.get("labels_or_retrieval_truth_materialized_before_audit") is not False:
                raise QualityStructureAggregationV92Error(
                    "Truth was materialized before V9.2 structure receipt"
                )
            persisted_internal_mismatch = sum(
                (
                    row["full_item_sha256"]
                    != replay["original_full_items_sha256"],
                    row["full_profile_sha256"]
                    != replay["original_full_profiles_sha256"],
                    row["counterfactual_full_item_sha256"]
                    != replay["counterfactual_full_items_sha256"],
                    row["counterfactual_full_profile_sha256"]
                    != replay["counterfactual_full_profiles_sha256"],
                    row["shared_identity33_sha256"]
                    != replay["identity33_source_sha256"],
                )
            )
            persisted_actual_mismatch = 0
            for surface in MODEL_SURFACES:
                item_field, profile_field = SURFACE_HASH_FIELDS[surface]
                actual_items, actual_profiles = surface_world_rows[surface][world_uid]
                persisted_actual_mismatch += int(
                    common.canonical_sha256(actual_items) != row[item_field]
                )
                persisted_actual_mismatch += int(
                    common.canonical_sha256(actual_profiles) != row[profile_field]
                )
            aggregate["persisted_model_input_hash_mismatch_count"] += int(
                persisted_internal_mismatch + persisted_actual_mismatch
            )
            world_receipts.append(
                {
                    "split": split,
                    "world_uid": world_uid,
                    "mapping_sha256": row["counterfactual_replay"]["mapping"][
                        "mapping_sha256"
                    ],
                    "persisted_internal_hash_mismatch_count": int(
                        persisted_internal_mismatch
                    ),
                    "shared_text_probe_eligibility_sha256": eligibility_sha,
                }
            )
    if not world_receipts:
        raise QualityStructureAggregationV92Error("No V9.2 structure worlds supplied")
    metric_values.update({name: int(value) for name, value in aggregate.items()})
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": "STRUCTURE_CALCULATION_COMPLETE_NO_STANDALONE_QUALIFICATION",
        "claim_boundary": claim_boundary,
        "gate_registry_sha256": registry.GATE_REGISTRY_SHA256,
        "metric_values": metric_values,
        "pending_structure_metrics": [
            metric
            for metric, _threshold, _comparison in registry.V9_2_STRUCTURE_METRICS
            if metric not in metric_values
        ],
        "base_v9_structure_receipt": base,
        "world_receipt_count": len(world_receipts),
        "world_receipt_bundle_sha256": common.canonical_sha256(world_receipts),
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def aggregate_fixture_structure(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    eligibility_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    model_surface_rows_by_split: Mapping[
        str,
        Mapping[
            str,
            tuple[
                Sequence[Mapping[str, Any]],
                Sequence[Mapping[str, Any]],
            ],
        ],
    ],
    expected_world_counts: Mapping[str, int],
    expected_sellers_per_world: int,
) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3
        for value in expected_world_counts.values()
    ):
        raise QualityStructureAggregationV92Error("Fixture world boundary widened")
    return _aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=structure_rows_by_split,
        eligibility_rows_by_split=eligibility_rows_by_split,
        model_surface_rows_by_split=model_surface_rows_by_split,
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
    eligibility_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    model_surface_rows_by_split: Mapping[
        str,
        Mapping[
            str,
            tuple[
                Sequence[Mapping[str, Any]],
                Sequence[Mapping[str, Any]],
            ],
        ],
    ],
    policy: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate all label-free gates under a separate one-shot authority."""

    capabilities = run_authorization.get("capabilities", {})
    if (
        capabilities.get("quality_audit_run") is not True
        or capabilities.get("metric_generation") is not True
        or capabilities.get("audit_a_b_truth_open") is not False
        or capabilities.get("formal_500_by_4") is not False
        or capabilities.get("model_training") is not False
    ):
        raise QualityStructureAggregationV92Error(
            "V9.2 formal structure authority is absent or over-broad"
        )
    return _aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=structure_rows_by_split,
        eligibility_rows_by_split=eligibility_rows_by_split,
        model_surface_rows_by_split=model_surface_rows_by_split,
        expected_world_counts=policy["design_scale"]["world_counts"],
        expected_sellers_per_world=policy["design_scale"][
            "seller_count_per_world"
        ],
        maximum_position_deviation=policy["quality_gates"][
            "code_character_position_maximum_absolute_deviation_from_one_sixteenth"
        ],
        enforce_position_margin=True,
        claim_boundary="V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
    )
