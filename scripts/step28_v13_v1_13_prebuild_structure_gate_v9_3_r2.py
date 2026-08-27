#!/usr/bin/env python3
"""Run the one pre-text V9.3-R2 structure-shortcut qualification gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_build_bounded_registered_negative_plan_v9_3_r2 as plan_builder
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as signatures_module
import step28_v13_v1_13_method_dataset_builder_v9_3 as dataset_builder
import step28_v13_v1_13_method_policy_v9_3 as method_policy_module
import step28_v13_v1_13_method_world_v9_3 as method_world
import step28_v13_v1_13_quality_probe_core_v9_3 as probe_core
import step28_v13_v1_13_registered_negative_plan_v9_3 as plan_contract
import step28_v13_v1_13_structure_matrix_v9_3 as structure_matrix
import step28_v13_v1_13_world_builder_v9_3 as world_builder


VERSION = (
    "2026-08-27-step28-v13-v1-13-prebuild-structure-gate-"
    "v9-3-r2-user-accepted-residual-22"
)
PASS_STATUS = "PASS_PREBUILD_STRUCTURE_GATE_NOT_METHOD_ROOT_OR_TRAINING_QUALIFIED"
FAIL_STATUS = "DATASET_INVALIDATED_PREBUILD_STRUCTURE_GATE"
FORMAL_OUTPUT_RELATIVE = Path(
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "registered_negative_structure_gate_r2_20260827"
)
PLAN_ROOT_RELATIVE = plan_builder.FORMAL_OUTPUT_RELATIVE
AUTHORITY_RELATIVE = Path(
    "private_custody/step28_v13_v1_13_v9_3_r2_method_random_authority.json"
)
SPLITS = ("train", "development")
ROWS_PER_WORLD = 378
WORLDS_PER_SPLIT = 500
ROWS_PER_SPLIT = ROWS_PER_WORLD * WORLDS_PER_SPLIT
COUNT_FAMILIES = (
    "pair_seller",
    "pair_noise",
    "directed_seller",
    "directed_noise",
    "role_seller",
    "role_noise",
    "endpoint_seller",
    "endpoint_noise",
    "role_triad",
    "role_size_seller",
    "role_size_noise",
)
EXPECTED_COUNT_FAMILY_PROJECTION_MAP_SHA256 = (
    "8b49db61cc9195bb0d1aaaa11c041d66e007a04ee7d21c0e56671e151268c261"
)
PLAN_ROOT_FILE_NAMES = (
    "construction_receipt.json",
    "development_registered_negative_plan.json",
    "development_residual_disclosure.json",
    "train_registered_negative_plan.json",
    "train_residual_disclosure.json",
)


class PrebuildStructureGateError(common.ContractError):
    """Raised when the frozen pre-text structure gate cannot close."""


def _expanded_feature_names(
    *, view: str, prefixes: Sequence[str], exact: Sequence[str] = ()
) -> list[str]:
    names = (
        structure_matrix.seller_matrix_feature_names()
        if view == "seller_slot"
        else structure_matrix.noise_matrix_feature_names()
        if view == "noise_visible"
        else ()
    )
    if not names:
        raise PrebuildStructureGateError("Unknown structure coverage view")
    selected = [
        name
        for name in names
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise PrebuildStructureGateError(
            f"Empty structure coverage selection: {view}"
        )
    return selected


def count_family_coverage() -> dict[str, list[dict[str, Any]]]:
    """Freeze the 11 count-family projections into the two existing views."""

    seller_pair = {
        "view": "seller_slot",
        "feature_names": _expanded_feature_names(
            view="seller_slot",
            prefixes=("seller_pair_", "seller_slot_"),
        ),
    }
    noise_pair = {
        "view": "noise_visible",
        "feature_names": _expanded_feature_names(
            view="noise_visible",
            prefixes=("noise_pair_", "noise_slot_"),
        ),
    }
    role = {
        "view": "noise_visible",
        "feature_names": _expanded_feature_names(
            view="noise_visible",
            prefixes=("role_", "registered_treatment_"),
            exact=(
                "registered_endpoint_count",
                "registered_clone_endpoint_count",
                "registered_semantic_endpoint_count",
            ),
        ),
    }
    controller_size = {
        "view": "noise_visible",
        "feature_names": _expanded_feature_names(
            view="noise_visible",
            prefixes=("controller_size_", "role_"),
        ),
    }
    endpoint = {
        "view": "noise_visible",
        "feature_names": _expanded_feature_names(
            view="noise_visible",
            prefixes=("registered_treatment_",),
            exact=(
                "registered_endpoint_count",
                "registered_clone_endpoint_count",
                "registered_semantic_endpoint_count",
            ),
        ),
    }

    def combine(*projections: Mapping[str, Any]) -> list[dict[str, Any]]:
        by_view: dict[str, set[str]] = {}
        for projection in projections:
            by_view.setdefault(str(projection["view"]), set()).update(
                str(name) for name in projection["feature_names"]
            )
        view_order = ("seller_slot", "noise_visible")
        output: list[dict[str, Any]] = []
        for view in view_order:
            if view not in by_view:
                continue
            canonical_names = (
                structure_matrix.seller_matrix_feature_names()
                if view == "seller_slot"
                else structure_matrix.noise_matrix_feature_names()
            )
            output.append(
                {
                    "view": view,
                    "feature_names": [
                        name for name in canonical_names if name in by_view[view]
                    ],
                }
            )
        return output

    return {
        "pair_seller": combine(seller_pair),
        "pair_noise": combine(noise_pair),
        "directed_seller": combine(seller_pair, role),
        "directed_noise": combine(noise_pair, role),
        "role_seller": combine(seller_pair, role),
        "role_noise": combine(noise_pair, role),
        "endpoint_seller": combine(seller_pair, endpoint),
        "endpoint_noise": combine(noise_pair, endpoint),
        "role_triad": combine(controller_size),
        "role_size_seller": combine(seller_pair, role, controller_size),
        "role_size_noise": combine(noise_pair, role, controller_size),
    }


def validate_count_family_coverage(
    coverage: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if tuple(coverage) != COUNT_FAMILIES:
        raise PrebuildStructureGateError("Count-family coverage order drift")
    if (
        common.canonical_sha256(coverage)
        != EXPECTED_COUNT_FAMILY_PROJECTION_MAP_SHA256
    ):
        raise PrebuildStructureGateError(
            "Preregistered count-family projection map drift"
        )
    view_names = {
        "seller_slot": set(structure_matrix.seller_matrix_feature_names()),
        "noise_visible": set(structure_matrix.noise_matrix_feature_names()),
    }
    covered_names: dict[str, set[str]] = {view: set() for view in view_names}
    for family in COUNT_FAMILIES:
        projections = coverage[family]
        if not isinstance(projections, list) or not projections:
            raise PrebuildStructureGateError(
                f"Count-family coverage absent: {family}"
            )
        observed_views: set[str] = set()
        for projection in projections:
            if set(projection) != {"view", "feature_names"}:
                raise PrebuildStructureGateError("Coverage projection schema drift")
            view = str(projection["view"])
            names = projection["feature_names"]
            if (
                view not in view_names
                or view in observed_views
                or not isinstance(names, list)
                or not names
                or len(names) != len(set(names))
                or not set(names).issubset(view_names[view])
            ):
                raise PrebuildStructureGateError(
                    f"Count-family feature coverage drift: {family}"
                )
            observed_views.add(view)
            covered_names[view].update(names)
    payload = {
        "count_family_order": list(COUNT_FAMILIES),
        "count_family_count": len(COUNT_FAMILIES),
        "coverage": deepcopy(dict(coverage)),
        "coverage_semantics": (
            "frozen_view_projection_diagnostic_not_cellwise_or_"
            "cross_view_interaction_complete"
        ),
        "seller_slot_and_noise_visible_models_remain_separate": True,
        "cross_view_interactions_tested": False,
        "theoretical_5_324_cell_balance_certified": False,
        "covered_feature_name_counts": {
            view: len(names) for view, names in covered_names.items()
        },
    }
    payload["canonical_sha256"] = common.canonical_sha256(payload)
    return payload


class _FrozenMatricesCapability:
    __slots__ = ("commitments", "_consumed", "_token")

    def __init__(self, token: object, commitments: Mapping[str, Any]) -> None:
        if token is not _CAPABILITY_TOKEN:
            raise PrebuildStructureGateError(
                "Frozen-matrix capability cannot be constructed directly"
            )
        self.commitments = deepcopy(dict(commitments))
        self._consumed = False
        self._token = token


_CAPABILITY_TOKEN = object()


def _verified_matrix_commitments(
    matrices: Mapping[str, Mapping[str, probe_core.FrozenMatrix]],
) -> dict[str, dict[str, Any]]:
    if tuple(matrices) != SPLITS:
        raise PrebuildStructureGateError("Frozen structure split order drift")
    commitments: dict[str, Any] = {}
    for split in SPLITS:
        if tuple(matrices[split]) != ("seller_slot", "noise_visible"):
            raise PrebuildStructureGateError("Frozen structure view order drift")
        seller = matrices[split]["seller_slot"]
        noise = matrices[split]["noise_visible"]
        if (
            seller.row_keys != noise.row_keys
            or len(seller.row_keys) != ROWS_PER_SPLIT
            or seller.values.flags.writeable
            or noise.values.flags.writeable
        ):
            raise PrebuildStructureGateError(
                "Frozen structure matrix closure drift"
            )
        commitments[split] = {}
        for view in ("seller_slot", "noise_visible"):
            matrix = matrices[split][view]
            observed = {
                "version": probe_core.VERSION,
                "view": view,
                "shape": list(matrix.values.shape),
                "dtype": "little-endian float64",
                "row_keys_sha256": common.canonical_sha256(
                    [list(key) for key in matrix.row_keys]
                ),
                "column_names_sha256": common.canonical_sha256(
                    list(matrix.column_names)
                ),
                "matrix_raw_f8_c_sha256": probe_core._matrix_sha256(
                    matrix.values
                ),
            }
            if matrix.commitment != observed:
                raise PrebuildStructureGateError(
                    f"Frozen structure matrix commitment drift: {split}/{view}"
                )
            commitments[split][view] = deepcopy(observed)
    return commitments


def _issue_frozen_matrices_capability(
    matrices: Mapping[str, Mapping[str, probe_core.FrozenMatrix]],
) -> _FrozenMatricesCapability:
    commitments = _verified_matrix_commitments(matrices)
    return _FrozenMatricesCapability(_CAPABILITY_TOKEN, commitments)


def _pair_labels_for_world(
    *,
    policy: Mapping[str, Any],
    mode: str,
    world_record: Mapping[str, Any],
    structure_key_hex: str,
    schedule_world: Mapping[str, Any],
) -> dict[str, int]:
    world_uid = str(world_record["world_uid"])
    world_ordinal = int(world_record["split_ordinal"])
    base = structure.build_world_membership(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
    )
    membership, _noise = world_builder._planned_membership(
        base=base,
        schedule_world=schedule_world,
        world_ordinal=world_ordinal,
    )
    sellers = tuple(common.utf8_sort(membership["seller_uids"]))
    controller = membership["seller_to_controller"]
    output = {
        common.canonical_pair_uid(left, right): int(
            controller[left] == controller[right]
        )
        for left_index, left in enumerate(sellers)
        for right in sellers[left_index + 1 :]
    }
    if len(output) != ROWS_PER_WORLD or sum(output.values()) != 20:
        raise PrebuildStructureGateError("Abstract pair-label law drift")
    return output


def materialize_labels_once(
    capability: _FrozenMatricesCapability,
    *,
    matrices: Mapping[str, Mapping[str, probe_core.FrozenMatrix]],
    policy: Mapping[str, Any],
    mode: str,
    world_records: Mapping[str, Sequence[Mapping[str, Any]]],
    schedules: Mapping[str, Mapping[str, Any]],
    access_counts: dict[str, int],
) -> dict[str, np.ndarray]:
    if (
        not isinstance(capability, _FrozenMatricesCapability)
        or capability._token is not _CAPABILITY_TOKEN
    ):
        raise PrebuildStructureGateError(
            "Labels requested before all structure matrices were frozen"
        )
    if capability._consumed:
        raise PrebuildStructureGateError(
            "Frozen-matrix label capability was already consumed"
        )
    capability._consumed = True
    observed_commitments = _verified_matrix_commitments(matrices)
    if observed_commitments != capability.commitments:
        raise PrebuildStructureGateError("Frozen-matrix capability drift")
    if access_counts != {"train": 0, "development": 0, "audit_a": 0, "audit_b": 0}:
        raise PrebuildStructureGateError("Pair-label access counter drift")
    labels: dict[str, np.ndarray] = {}
    for split in SPLITS:
        records = world_records[split]
        if len(records) != WORLDS_PER_SPLIT:
            raise PrebuildStructureGateError("World-record cardinality drift")
        world_label_maps: dict[str, dict[str, int]] = {}
        structure_key = common.structure_key_for_split(
            policy, mode=mode, split=split
        )
        for record in records:
            ordinal = int(record["split_ordinal"])
            world_label_maps[str(record["world_uid"])] = _pair_labels_for_world(
                policy=policy,
                mode=mode,
                world_record=record,
                structure_key_hex=structure_key,
                schedule_world=schedules[split]["worlds"][ordinal],
            )
        row_keys = matrices[split]["seller_slot"].row_keys
        values = np.fromiter(
            (
                world_label_maps[world_uid][pair_uid]
                for world_uid, pair_uid in row_keys
            ),
            dtype=np.int8,
            count=ROWS_PER_SPLIT,
        )
        if (
            values.shape != (ROWS_PER_SPLIT,)
            or int(values.sum()) != WORLDS_PER_SPLIT * 20
        ):
            raise PrebuildStructureGateError("Materialized label totals drift")
        values.setflags(write=False)
        labels[split] = values
        access_counts[split] += 1
    if access_counts != {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0}:
        raise PrebuildStructureGateError("Final pair-label access counter drift")
    return labels


def _build_frozen_split(
    *,
    split: str,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    world_records: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    joint_signatures: Mapping[str, Any],
) -> dict[str, probe_core.FrozenMatrix]:
    seller_names = structure_matrix.seller_matrix_feature_names()
    noise_names = structure_matrix.noise_matrix_feature_names()
    seller_values = np.empty(
        (ROWS_PER_SPLIT, len(seller_names)), dtype=np.dtype("<f8"), order="C"
    )
    noise_values = np.empty(
        (ROWS_PER_SPLIT, len(noise_names)), dtype=np.dtype("<f8"), order="C"
    )
    row_keys: list[tuple[str, str]] = []
    structure_key = common.structure_key_for_split(
        policy, mode=mode, split=split
    )
    for world_index, record in enumerate(world_records):
        blueprint = world_builder.build_structure_blueprint(
            policy=dict(policy),
            template=dict(template),
            mode=mode,
            world_record=record,
            structure_key_hex=structure_key,
            balanced_schedule=schedule,
            registered_negative_plan=plan,
            joint_signatures=joint_signatures,
            candidate_index=0,
        )
        if blueprint["audit"]["natural_text_field_count"] != 0:
            raise PrebuildStructureGateError("Prebuild blueprint produced text")
        seller_rows, noise_rows = method_world._structure_rows(
            policy=policy,
            template=template,
            split=split,
            world=blueprint,
            candidate_index=0,
        )
        start = world_index * ROWS_PER_WORLD
        stop = start + ROWS_PER_WORLD
        seller_values[start:stop] = structure_matrix.seller_matrix(seller_rows)
        noise_values[start:stop] = structure_matrix.noise_matrix(noise_rows)
        keys = [
            (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            for row in seller_rows
        ]
        if keys != [
            (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            for row in noise_rows
        ]:
            raise PrebuildStructureGateError("Structure view row-key drift")
        row_keys.extend(keys)
        if (world_index + 1) % 25 == 0 or world_index + 1 == len(world_records):
            print(
                plan_contract.canonical_json_bytes(
                    {
                        "event": "prebuild_structure_worlds_frozen",
                        "split": split,
                        "completed": world_index + 1,
                        "total": len(world_records),
                    }
                ).decode("utf-8"),
                flush=True,
            )
    if len(row_keys) != ROWS_PER_SPLIT or len(row_keys) != len(set(row_keys)):
        raise PrebuildStructureGateError("Structure row-key closure drift")
    return {
        "seller_slot": probe_core.freeze_matrix(
            view="seller_slot",
            values=seller_values,
            row_keys=row_keys,
            column_names=seller_names,
            take_ownership=True,
        ),
        "noise_visible": probe_core.freeze_matrix(
            view="noise_visible",
            values=noise_values,
            row_keys=row_keys,
            column_names=noise_names,
            take_ownership=True,
        ),
    }


def validate_probe_result_contract(
    result: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    average_precision_baseline: float,
) -> dict[str, Any]:
    expected_result_fields = {
        "single_feature_maximum_symmetric_roc_auc_by_view",
        "model_results",
        "maximum_symmetric_roc_auc",
        "maximum_average_precision_uplift",
        "bootstrap",
    }
    expected_models = (
        "seller_slot::logistic_l2",
        "seller_slot::hist_gradient_boosting_depth2",
        "noise_visible::logistic_l2",
        "noise_visible::hist_gradient_boosting_depth2",
    )
    singles = result.get("single_feature_maximum_symmetric_roc_auc_by_view")
    models = result.get("model_results")
    bootstrap = result.get("bootstrap")
    if (
        set(result) != expected_result_fields
        or not isinstance(singles, Mapping)
        or tuple(singles) != ("seller_slot", "noise_visible")
        or not isinstance(models, Mapping)
        or tuple(models) != expected_models
        or not isinstance(bootstrap, Mapping)
    ):
        raise PrebuildStructureGateError("Frozen probe result schema drift")
    finite_values: list[float] = [
        float(result["maximum_symmetric_roc_auc"]),
        float(result["maximum_average_precision_uplift"]),
        *(float(value) for value in singles.values()),
    ]
    for model_name in expected_models:
        row = models[model_name]
        if not isinstance(row, Mapping) or set(row) != {
            "symmetric_roc_auc",
            "average_precision",
            "score_vector_sha256",
        }:
            raise PrebuildStructureGateError("Frozen probe model result drift")
        score_sha256 = row["score_vector_sha256"]
        if (
            not isinstance(score_sha256, str)
            or len(score_sha256) != 64
            or any(value not in "0123456789abcdef" for value in score_sha256)
        ):
            raise PrebuildStructureGateError("Frozen probe score hash drift")
        finite_values.extend(
            (float(row["symmetric_roc_auc"]), float(row["average_precision"]))
        )
    if (
        not math.isclose(
            float(result["maximum_symmetric_roc_auc"]),
            max(float(row["symmetric_roc_auc"]) for row in models.values()),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(result["maximum_average_precision_uplift"]),
            max(
                float(row["average_precision"]) - average_precision_baseline
                for row in models.values()
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise PrebuildStructureGateError("Frozen probe family maximum drift")
    expected_bootstrap_fields = {
        "replicates",
        "world_count",
        "score_family_size",
        "draws_raw_i8_c_sha256",
        "family_max_symmetric_auc_vector_sha256",
        "family_max_average_precision_uplift_vector_sha256",
        "symmetric_auc_95_upper",
        "average_precision_uplift_95_upper",
    }
    bootstrap_policy = policy["bootstrap"]
    if (
        set(bootstrap) != expected_bootstrap_fields
        or bootstrap["replicates"] != 9_999
        or bootstrap["replicates"] != bootstrap_policy["replicates"]
        or bootstrap["world_count"] != 500
        or bootstrap["world_count"]
        != bootstrap_policy["development_world_count"]
        or bootstrap["score_family_size"] != 4
        or bootstrap["draws_raw_i8_c_sha256"]
        != bootstrap_policy["draws_raw_i8_c_sha256"]
    ):
        raise PrebuildStructureGateError("Frozen probe bootstrap contract drift")
    for field in (
        "draws_raw_i8_c_sha256",
        "family_max_symmetric_auc_vector_sha256",
        "family_max_average_precision_uplift_vector_sha256",
    ):
        value = bootstrap[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PrebuildStructureGateError("Frozen bootstrap hash drift")
    finite_values.extend(
        (
            float(bootstrap["symmetric_auc_95_upper"]),
            float(bootstrap["average_precision_uplift_95_upper"]),
            float(average_precision_baseline),
        )
    )
    if (
        average_precision_baseline != 20 / 378
        or not all(math.isfinite(value) for value in finite_values)
    ):
        raise PrebuildStructureGateError("Frozen probe finite-value drift")
    return {
        "view_order": ["seller_slot", "noise_visible"],
        "model_order": list(expected_models),
        "model_count": 4,
        "matrix_concatenation_used": False,
        "average_precision_baseline": average_precision_baseline,
        "bootstrap_replicates": bootstrap["replicates"],
        "bootstrap_world_count": bootstrap["world_count"],
        "bootstrap_score_family_size": bootstrap["score_family_size"],
        "bootstrap_draws_raw_i8_c_sha256": bootstrap[
            "draws_raw_i8_c_sha256"
        ],
    }


def evaluate_gate(
    *,
    policy: Mapping[str, Any],
    effective_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    schedules: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    joint_signatures: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = validate_count_family_coverage(count_family_coverage())
    records = tuple(
        structure.build_mode_world_pool(effective_policy, mode=dataset_builder.MODE)
    )
    world_records = {
        split: tuple(row for row in records if row["split"] == split)
        for split in SPLITS
    }
    if any(len(world_records[split]) != WORLDS_PER_SPLIT for split in SPLITS):
        raise PrebuildStructureGateError("Train/development world pool drift")
    matrices = {
        split: _build_frozen_split(
            split=split,
            policy=effective_policy,
            template=template,
            mode=dataset_builder.MODE,
            world_records=world_records[split],
            schedule=schedules[split],
            plan=plans[split],
            joint_signatures=joint_signatures,
        )
        for split in SPLITS
    }
    capability = _issue_frozen_matrices_capability(matrices)
    access_counts = {"train": 0, "development": 0, "audit_a": 0, "audit_b": 0}
    labels = materialize_labels_once(
        capability,
        matrices=matrices,
        policy=effective_policy,
        mode=dataset_builder.MODE,
        world_records=world_records,
        schedules=schedules,
        access_counts=access_counts,
    )
    if _verified_matrix_commitments(matrices) != capability.commitments:
        raise PrebuildStructureGateError(
            "Frozen matrices changed during label materialization"
        )
    average_precision_baseline = 20 / 378
    result = probe_core.evaluate_family(
        train=matrices["train"],
        development=matrices["development"],
        train_labels=labels["train"],
        development_labels=labels["development"],
        policy=policy,
        average_precision_baseline=average_precision_baseline,
        bootstrap=True,
    )
    if _verified_matrix_commitments(matrices) != capability.commitments:
        raise PrebuildStructureGateError(
            "Frozen matrices changed during probe evaluation"
        )
    probe_contract_audit = validate_probe_result_contract(
        result,
        policy=policy,
        average_precision_baseline=average_precision_baseline,
    )
    gates = policy["quality_gates"]
    observations = {
        "single_feature_maximum_symmetric_roc_auc": max(
            result["single_feature_maximum_symmetric_roc_auc_by_view"].values()
        ),
        "family_maximum_symmetric_roc_auc": result[
            "maximum_symmetric_roc_auc"
        ],
        "family_maximum_average_precision_uplift": result[
            "maximum_average_precision_uplift"
        ],
        "bootstrap_95_upper_symmetric_roc_auc": result["bootstrap"][
            "symmetric_auc_95_upper"
        ],
        "bootstrap_95_upper_average_precision_uplift": result["bootstrap"][
            "average_precision_uplift_95_upper"
        ],
    }
    thresholds = {
        "single_feature_maximum_symmetric_roc_auc": gates[
            "maximum_single_feature_symmetric_roc_auc"
        ],
        "family_maximum_symmetric_roc_auc": gates[
            "maximum_family_symmetric_roc_auc"
        ],
        "family_maximum_average_precision_uplift": gates[
            "maximum_family_average_precision_uplift"
        ],
        "bootstrap_95_upper_symmetric_roc_auc": gates[
            "bootstrap_95_upper_symmetric_roc_auc"
        ],
        "bootstrap_95_upper_average_precision_uplift": gates[
            "bootstrap_95_upper_average_precision_uplift"
        ],
    }
    if not all(
        math.isfinite(float(value))
        for value in (*observations.values(), *thresholds.values())
    ):
        raise PrebuildStructureGateError(
            "Structure-gate observation or threshold is nonfinite"
        )
    gate_rows = [
        {
            "gate": name,
            "observed": float(observations[name]),
            "maximum": float(thresholds[name]),
            "passed": float(observations[name]) <= float(thresholds[name]),
        }
        for name in thresholds
    ]
    passed = all(row["passed"] for row in gate_rows)
    payload: dict[str, Any] = {
        "version": VERSION,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "claim_boundary": (
            "frozen_registered_structure_probes_did_not_detect_a_shortcut_above_threshold"
            if passed
            else "v9_3_r2_invalidated_before_text_or_method_root"
        ),
        "count_family_coverage": coverage,
        "matrix_commitments": capability.commitments,
        "label_access_counts": access_counts,
        "label_commitments": {
            split: {
                "row_count": len(labels[split]),
                "positive_count": int(labels[split].sum()),
                "raw_i1_sha256": hashlib.sha256(
                    labels[split].tobytes(order="C")
                ).hexdigest(),
            }
            for split in SPLITS
        },
        "probe_result": result,
        "probe_contract_audit": probe_contract_audit,
        "hard_gates": gate_rows,
        "natural_text_generated": False,
        "identity_assets_generated": False,
        "method_root_generated": False,
        "audit_a_b_truth_read_count": 0,
        "m0_m1_m2_m3_training_authorized": False,
    }
    payload["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
        payload
    )
    return payload


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def validate_published_plan_root(
    *,
    plan_root: Path,
    schedule_root: Path,
    signature_path: Path,
    schedules: Mapping[str, Mapping[str, Any]],
    joint_signatures: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen and bind all five files of the published R2 plan root."""

    repository_root = Path(__file__).resolve().parents[1]
    if plan_root.resolve() != (repository_root / PLAN_ROOT_RELATIVE).resolve():
        raise PrebuildStructureGateError("Published R2 plan-root path drift")
    if not plan_root.is_dir():
        raise PrebuildStructureGateError("Published R2 plan root is absent")
    observed_names = tuple(
        sorted(
            (path.name for path in plan_root.iterdir() if path.is_file()),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if observed_names != PLAN_ROOT_FILE_NAMES or any(
        path.is_dir() for path in plan_root.iterdir()
    ):
        raise PrebuildStructureGateError("Published R2 plan file-set drift")

    payloads = {
        name: plan_contract.load_json(plan_root / name)
        for name in PLAN_ROOT_FILE_NAMES
    }
    receipt = payloads["construction_receipt.json"]
    required_receipt_fields = {
        "version",
        "status",
        "inputs",
        "source_files",
        "public_design_seeds",
        "train_replay",
        "development_replay",
        "plan_pair_audit",
        "published_files",
        "canonical_self_sha256",
    }
    if (
        set(receipt) != required_receipt_fields
        or receipt["version"] != plan_builder.VERSION
        or receipt["status"] != plan_builder.PLAN_STATUS
        or receipt["public_design_seeds"]
        != plan_builder.constructor.PUBLIC_DESIGN_SEEDS
        or receipt["canonical_self_sha256"]
        != plan_contract.canonical_self_sha256(receipt)
    ):
        raise PrebuildStructureGateError("Published R2 construction receipt drift")

    invocation = plan_builder.validate_formal_invocation(
        output_directory=plan_root,
        train_schedule_path=schedule_root / "train_balanced_schedule.json",
        development_schedule_path=(
            schedule_root / "development_balanced_schedule.json"
        ),
        joint_signature_path=signature_path,
    )
    expected_sources = plan_builder.expected_source_files(repository_root)
    if (
        receipt["inputs"] != invocation["inputs"]
        or receipt["source_files"] != expected_sources
    ):
        raise PrebuildStructureGateError("Published R2 provenance drift")
    plan_builder.validate_source_files(
        receipt["source_files"], repository_root=repository_root
    )

    plans = {
        split: payloads[f"{split}_registered_negative_plan.json"]
        for split in SPLITS
    }
    disclosures = {
        split: payloads[f"{split}_residual_disclosure.json"]
        for split in SPLITS
    }
    replays = {
        split: plan_builder.validate_split_bundle(
            plan=plans[split],
            disclosure=disclosures[split],
            schedule=schedules[split],
            joint_signatures=joint_signatures,
            expected_inputs=receipt["inputs"],
            expected_sources=receipt["source_files"],
        )
        for split in SPLITS
    }
    if (
        receipt["train_replay"] != replays["train"]
        or receipt["development_replay"] != replays["development"]
    ):
        raise PrebuildStructureGateError("Published R2 replay receipt drift")
    pair_audit = plan_contract.validate_train_development_plan_pair(
        plans["train"],
        plans["development"],
        schedules["train"],
        schedules["development"],
        joint_signatures,
        expected_version=plan_contract.BOUNDED_RESIDUAL_VERSION,
        require_exact_balance=False,
        plan_success_status=plan_contract.BOUNDED_RESIDUAL_STATUS,
        pair_success_status=plan_builder.PAIR_STATUS,
    )
    if receipt["plan_pair_audit"] != pair_audit:
        raise PrebuildStructureGateError("Published R2 plan-pair receipt drift")

    expected_published_names = {
        f"{split}_{suffix}.json"
        for split in SPLITS
        for suffix in (
            "registered_negative_plan",
            "residual_disclosure",
        )
    }
    if set(receipt["published_files"]) != expected_published_names:
        raise PrebuildStructureGateError("Published R2 file receipt drift")
    for name in expected_published_names:
        path = plan_root / name
        if receipt["published_files"][name] != {
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }:
            raise PrebuildStructureGateError(
                f"Published R2 payload hash drift: {name}"
            )

    return {
        "plans": plans,
        "disclosures": disclosures,
        "receipt": receipt,
        "replays": replays,
        "plan_pair_audit": pair_audit,
        "root_files": {
            name: _file_record(plan_root / name, repository_root)
            for name in PLAN_ROOT_FILE_NAMES
        },
    }


def _formal_input_records(
    *,
    repository_root: Path,
    policy: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_path: Path,
    schedule_root: Path,
    signature_path: Path,
    plan_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = policy["frozen_inputs"]
    parent_policy_path = common.repo_path(str(frozen["parent_policy"]))
    parent_policy = common.load_json(parent_policy_path)
    style_profile_path = common.verify_file_pin(
        parent_policy["style_reference_boundary"]["generator_release_inputs"][
            "profile"
        ],
        label="V9.3-R2 prebuild structure style profile",
    )
    runtime_sources = {
        "prebuild_structure_gate": Path(__file__),
        "plan_builder": Path(plan_builder.__file__),
        "plan_validator": Path(plan_contract.__file__),
        "dataset_builder": Path(dataset_builder.__file__),
        "world_builder": Path(world_builder.__file__),
        "method_world": Path(method_world.__file__),
        "structure_matrix": Path(structure_matrix.__file__),
        "probe_core": Path(probe_core.__file__),
        "balanced_schedule_validator": Path(balanced.__file__),
        "joint_signature_validator": Path(signatures_module.__file__),
    }
    return {
        "method_policy": _file_record(
            common.repo_path(method_policy_module.POLICY_PATH), repository_root
        ),
        "method_policy_canonical_self_sha256": policy["canonical_self_sha256"],
        "parent_policy": _file_record(
            parent_policy_path, repository_root
        ),
        "text_template": _file_record(
            common.repo_path(str(frozen["text_template"])), repository_root
        ),
        "style_profile": _file_record(style_profile_path, repository_root),
        "runtime_source_files": {
            name: _file_record(path, repository_root)
            for name, path in runtime_sources.items()
        },
        "authority_commitment": {
            "path": AUTHORITY_RELATIVE.as_posix(),
            "file_sha256": common.sha256_file(authority_path),
            "canonical_self_sha256": authority["canonical_self_sha256"],
            "key_values_recorded": False,
        },
        "balanced_schedule_files": {
            split: _file_record(
                schedule_root / f"{split}_balanced_schedule.json",
                repository_root,
            )
            for split in SPLITS
        },
        "published_plan_root_files": deepcopy(plan_bundle["root_files"]),
        "joint_signatures": _file_record(signature_path, repository_root),
    }


def run_formal(
    *, output_directory: Path, authority_path: Path
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output_directory.resolve() != (root / FORMAL_OUTPUT_RELATIVE).resolve():
        raise PrebuildStructureGateError("Formal structure-gate output path drift")
    if authority_path.resolve() != (root / AUTHORITY_RELATIVE).resolve():
        raise PrebuildStructureGateError("Formal structure-gate authority path drift")
    if output_directory.exists():
        raise PrebuildStructureGateError("Formal structure-gate output exists")
    building = output_directory.with_name(output_directory.name + ".building")
    if building.exists():
        raise PrebuildStructureGateError("Stale structure-gate building path exists")
    policy = method_policy_module.load_policy()
    if (
        policy.get("status")
        != "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED"
    ):
        raise PrebuildStructureGateError(
            "Structure gate is forbidden before method-policy file pins freeze"
        )
    authority = dataset_builder._validate_authority(common.load_json(authority_path))
    if authority["method_policy_canonical_self_sha256"] != policy[
        "canonical_self_sha256"
    ]:
        raise PrebuildStructureGateError("Structure-gate authority/policy drift")
    base = common.load_json(common.repo_path(str(policy["frozen_inputs"]["parent_policy"])))
    effective = dataset_builder._effective_policy(base, authority["keys"])
    template = common.load_json(common.repo_path(str(policy["frozen_inputs"]["text_template"])))
    schedule_root = common.repo_path(str(policy["frozen_inputs"]["balanced_schedule_root"]))
    plan_root = common.repo_path(str(policy["frozen_inputs"]["registered_negative_plan_root"]))
    signature_path = common.repo_path(str(policy["frozen_inputs"]["joint_noise_signature"]))
    schedules = {
        split: common.load_json(schedule_root / f"{split}_balanced_schedule.json")
        for split in SPLITS
    }
    joint_signatures = common.load_json(signature_path)
    balanced.validate_train_development_pair(
        schedules["train"], schedules["development"]
    )
    signatures_module.validate_payload(joint_signatures)
    plan_bundle = validate_published_plan_root(
        plan_root=plan_root,
        schedule_root=schedule_root,
        signature_path=signature_path,
        schedules=schedules,
        joint_signatures=joint_signatures,
    )
    plans = plan_bundle["plans"]
    opening_inputs = _formal_input_records(
        repository_root=root,
        policy=policy,
        authority=authority,
        authority_path=authority_path,
        schedule_root=schedule_root,
        signature_path=signature_path,
        plan_bundle=plan_bundle,
    )
    result = evaluate_gate(
        policy=policy,
        effective_policy=effective,
        template=template,
        schedules=schedules,
        plans=plans,
        joint_signatures=joint_signatures,
    )
    result["plan_root_validation"] = {
        "construction_receipt_canonical_self_sha256": plan_bundle["receipt"][
            "canonical_self_sha256"
        ],
        "plan_pair_audit_sha256": plan_contract.canonical_sha256(
            plan_bundle["plan_pair_audit"]
        ),
        "split_replays": plan_bundle["replays"],
        "residual_disclosure_commitments": {
            split: {
                "canonical_self_sha256": plan_bundle["disclosures"][split][
                    "canonical_self_sha256"
                ],
                "constraint_cells_sha256": plan_bundle["disclosures"][split][
                    "state_audit"
                ]["constraint_cells_sha256"],
                "l1_bound_violation": plan_bundle["disclosures"][split][
                    "state_audit"
                ]["l1_bound_violation"],
                "squared_objective": plan_bundle["disclosures"][split][
                    "state_audit"
                ]["squared_objective"],
                "violated_cell_count": plan_bundle["disclosures"][split][
                    "state_audit"
                ]["violated_cell_count"],
                "role_eligibility_authority": plan_bundle["disclosures"][split][
                    "role_eligibility_authority"
                ],
            }
            for split in SPLITS
        },
    }
    closing_policy = method_policy_module.load_policy()
    closing_authority = dataset_builder._validate_authority(
        common.load_json(authority_path)
    )
    closing_plan_bundle = validate_published_plan_root(
        plan_root=plan_root,
        schedule_root=schedule_root,
        signature_path=signature_path,
        schedules={
            split: common.load_json(
                schedule_root / f"{split}_balanced_schedule.json"
            )
            for split in SPLITS
        },
        joint_signatures=common.load_json(signature_path),
    )
    closing_inputs = _formal_input_records(
        repository_root=root,
        policy=closing_policy,
        authority=closing_authority,
        authority_path=authority_path,
        schedule_root=schedule_root,
        signature_path=signature_path,
        plan_bundle=closing_plan_bundle,
    )
    if (
        closing_policy != policy
        or closing_authority != authority
        or closing_inputs != opening_inputs
        or closing_plan_bundle["replays"] != plan_bundle["replays"]
        or closing_plan_bundle["plan_pair_audit"]
        != plan_bundle["plan_pair_audit"]
    ):
        raise PrebuildStructureGateError(
            "Structure-gate input provenance changed during evaluation"
        )
    result["inputs"] = opening_inputs
    result["canonical_self_sha256"] = plan_contract.canonical_self_sha256(result)
    try:
        building.mkdir(parents=True, exist_ok=False)
        result_path = building / "structure_gate_result.json"
        with result_path.open("xb") as stream:
            stream.write(plan_contract.canonical_json_bytes(result) + b"\n")
        reopened = plan_contract.load_json(result_path)
        if (
            reopened != result
            or reopened["canonical_self_sha256"]
            != plan_contract.canonical_self_sha256(reopened)
        ):
            raise PrebuildStructureGateError("Structure-gate persistence drift")
        building.replace(output_directory)
        if (
            {path.name for path in output_directory.iterdir() if path.is_file()}
            != {"structure_gate_result.json"}
            or any(path.is_dir() for path in output_directory.iterdir())
        ):
            raise PrebuildStructureGateError(
                "Structure-gate final file-set drift"
            )
        final_policy = method_policy_module.load_policy()
        final_authority = dataset_builder._validate_authority(
            common.load_json(authority_path)
        )
        final_plan_bundle = validate_published_plan_root(
            plan_root=plan_root,
            schedule_root=schedule_root,
            signature_path=signature_path,
            schedules={
                split: common.load_json(
                    schedule_root / f"{split}_balanced_schedule.json"
                )
                for split in SPLITS
            },
            joint_signatures=common.load_json(signature_path),
        )
        final_inputs = _formal_input_records(
            repository_root=root,
            policy=final_policy,
            authority=final_authority,
            authority_path=authority_path,
            schedule_root=schedule_root,
            signature_path=signature_path,
            plan_bundle=final_plan_bundle,
        )
        final_result = plan_contract.load_json(
            output_directory / "structure_gate_result.json"
        )
        if (
            final_policy != policy
            or final_authority != authority
            or final_inputs != opening_inputs
            or final_plan_bundle["replays"] != plan_bundle["replays"]
            or final_plan_bundle["plan_pair_audit"]
            != plan_bundle["plan_pair_audit"]
            or final_result != result
        ):
            raise PrebuildStructureGateError(
                "Structure-gate final persistence or provenance drift"
            )
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        if output_directory.exists():
            shutil.rmtree(output_directory)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=common.repo_path(FORMAL_OUTPUT_RELATIVE.as_posix())
    )
    parser.add_argument(
        "--authority", type=Path, default=common.repo_path(AUTHORITY_RELATIVE.as_posix())
    )
    args = parser.parse_args()
    result = run_formal(
        output_directory=args.output_directory.resolve(),
        authority_path=args.authority.resolve(),
    )
    print(plan_contract.canonical_json_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
