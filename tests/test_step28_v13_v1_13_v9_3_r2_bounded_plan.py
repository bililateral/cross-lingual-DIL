#!/usr/bin/env python3
"""Contracts for the V9.3-R2 user-accepted-residual plan successor."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_build_bounded_registered_negative_plan_v9_3_r2 as r2
import step28_v13_v1_13_build_residual_checkpoint_v9_3 as checkpoint
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as constructor
import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_method_world_v9_3 as method_world
import step28_v13_v1_13_prebuild_structure_gate_v9_3_r2 as structure_gate
import step28_v13_v1_13_quality_probe_core_v9_3 as probe_core
import step28_v13_v1_13_registered_negative_plan_v9_3 as plan_contract
import step28_v13_v1_13_world_builder_v9_3 as world_builder


PREFLIGHT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "design_preflight_v2_20260825"
)
JOINT_SIGNATURE = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "joint_noise_signature_preflight_v2_20260826.json"
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


class BoundedPlanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = read_json(PREFLIGHT / "train_balanced_schedule.json")
        cls.development = read_json(
            PREFLIGHT / "development_balanced_schedule.json"
        )
        cls.signatures = read_json(JOINT_SIGNATURE)

    def test_formal_paths_and_label_free_inputs_are_exactly_pinned(self) -> None:
        self.assertEqual(constructor.LOCAL_REPAIR_COARSE_MILP_THRESHOLD, 22)
        self.assertEqual(
            r2.PLAN_VERSION,
            "2026-08-27-step28-v13-v1-13-registered-negative-plan-"
            "v9-3-r2-user-accepted-residual-22",
        )
        self.assertEqual(
            r2.PLAN_STATUS,
            "PASS_DETERMINISTIC_USER_ACCEPTED_RESIDUAL_22_"
            "PENDING_STRUCTURE_GATE",
        )
        self.assertEqual(r2.TRAIN_EXPECTED_L1, 20)
        self.assertEqual(r2.DEVELOPMENT_EXPECTED_L1, 22)
        self.assertEqual(r2.DEVELOPMENT_EXPECTED_OBJECTIVE, 22)
        self.assertEqual(r2.DEVELOPMENT_EXPECTED_VIOLATED_CELL_COUNT, 22)
        output = ROOT / r2.FORMAL_OUTPUT_RELATIVE
        audit = r2.validate_formal_invocation(
            output_directory=output,
            train_schedule_path=PREFLIGHT / "train_balanced_schedule.json",
            development_schedule_path=(
                PREFLIGHT / "development_balanced_schedule.json"
            ),
            joint_signature_path=JOINT_SIGNATURE,
        )
        self.assertEqual(
            audit["status"],
            "PASS_V9_3_R2_FORMAL_INVOCATION_ONLY_NO_PLAN_RUN",
        )
        self.assertEqual(
            set(audit["inputs"]),
            {"train_schedule", "development_schedule", "joint_signatures"},
        )
        with self.assertRaisesRegex(r2.BoundedPlanBuildError, "output path"):
            r2.validate_formal_invocation(
                output_directory=output.with_name("wrong_r2_path"),
                train_schedule_path=PREFLIGHT / "train_balanced_schedule.json",
                development_schedule_path=(
                    PREFLIGHT / "development_balanced_schedule.json"
                ),
                joint_signature_path=JOINT_SIGNATURE,
            )

    def test_source_authority_is_tamper_evident(self) -> None:
        records = r2.expected_source_files(ROOT)
        r2.validate_source_files(records, repository_root=ROOT)
        tampered = copy.deepcopy(records)
        tampered["plan_validator"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(r2.BoundedPlanBuildError, "source-file drift"):
            r2.validate_source_files(tampered, repository_root=ROOT)

    def test_four_named_role_predicates_share_one_position_authority(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        self.assertEqual(search.role_eligible.shape, (500, 12, 28))
        self.assertEqual(
            constructor.ROLE_ELIGIBILITY_PREDICATE_NAMES,
            plan_contract.ROLE_ELIGIBILITY_PREDICATE_NAMES,
        )
        row = search.assignments[0].copy()
        self.assertTrue(search._valid_row(0, row))
        selected_at_zero = int(row[0])
        search.role_eligible[0, 1, selected_at_zero] = False
        self.assertTrue(search._valid_row(0, row))
        search.role_eligible[0, 0, selected_at_zero] = False
        self.assertFalse(search._valid_row(0, row))

    def test_constructor_and_validator_role_eligibility_replay_identically(self) -> None:
        signatures = [
            row["signature"] for row in self.signatures["noise_slot_multiset"]
        ]
        for signature in signatures:
            for treatment, role in constructor.ROLE_NAMES:
                self.assertEqual(
                    constructor._role_eligible_logical_item_ordinals(
                        treatment=treatment,
                        role=role,
                        signature=signature,
                    ),
                    plan_contract._role_eligible_logical_item_ordinals(
                        treatment=treatment,
                        role=role,
                        signature=signature,
                    ),
                )

    def test_full_disclosure_covers_all_5324_cells_in_frozen_order(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        cells = search._constraint_cells()
        l1 = sum(
            search._cell_violation(
                int(search.arrays[family][index]), lower, upper
            )
            for family, index, lower, upper in cells
        )
        audit = checkpoint.audit_search_state(
            search,
            expected_l1=l1,
            expected_objective=search.objective,
        )
        self.assertEqual(len(audit["constraint_cells"]), 5_324)
        self.assertEqual(
            [row["cell_ordinal"] for row in audit["constraint_cells"]],
            list(range(5_324)),
        )
        self.assertEqual(
            audit["constraint_cells_sha256"],
            plan_contract.canonical_sha256(audit["constraint_cells"]),
        )
        self.assertEqual(
            sum(row["absolute_violation"] for row in audit["constraint_cells"]),
            audit["l1_bound_violation"],
        )
        self.assertEqual(
            sum(audit["family_l1_bound_violation"].values()),
            audit["l1_bound_violation"],
        )

    def test_bounded_validator_preserves_mechanical_gates(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        plan = constructor.materialize_plan(
            split="train",
            assignments=search.assignments,
            schedule=self.train,
            joint_signatures=self.signatures,
            plan_version=r2.PLAN_VERSION,
        )
        audit = plan_contract.validate_plan(
            plan,
            self.train,
            self.signatures,
            expected_version=r2.PLAN_VERSION,
            require_exact_balance=False,
            success_status=r2.PLAN_STATUS,
        )
        self.assertFalse(audit["exact_balance_required"])
        self.assertEqual(audit["status"], r2.PLAN_STATUS)
        self.assertEqual(
            audit["role_eligibility_predicates"],
            list(constructor.ROLE_ELIGIBILITY_PREDICATE_NAMES),
        )
        self.assertEqual(
            r2.assignments_from_plan(plan).tolist(),
            search.assignments.tolist(),
        )
        tampered = copy.deepcopy(plan)
        endpoints = tampered["worlds"][0]["assignments"][0]["endpoints"]
        endpoints[1]["seller_slot"] = endpoints[0]["seller_slot"]
        tampered["canonical_self_sha256"] = plan_contract.canonical_self_sha256(
            tampered
        )
        with self.assertRaises(plan_contract.RegisteredNegativePlanError):
            plan_contract.validate_plan(
                tampered,
                self.train,
                self.signatures,
                expected_version=r2.PLAN_VERSION,
                require_exact_balance=False,
                success_status=r2.PLAN_STATUS,
            )

    def test_initial_random_state_cannot_masquerade_as_bounded_terminal(self) -> None:
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        with self.assertRaisesRegex(r2.BoundedPlanBuildError, "frozen L1"):
            r2.audit_bounded_state(
                search,
                split="train",
                expected_l1=r2.TRAIN_EXPECTED_L1,
                expected_objective=r2.TRAIN_EXPECTED_OBJECTIVE,
                expected_violated_cell_count=(
                    r2.TRAIN_EXPECTED_VIOLATED_CELL_COUNT
                ),
            )

    def test_structure_blueprint_precedes_text_and_truth(self) -> None:
        policy = read_json(
            ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json"
        )
        template = read_json(ROOT / str(policy["template_library"]["path"]))
        mode = "development_smoke"
        world_record = next(
            row
            for row in structure.build_mode_world_pool(policy, mode=mode)
            if row["split"] == "train" and row["split_ordinal"] == 0
        )
        search = constructor.JointSearch(
            self.train, self.signatures, split="train"
        )
        plan = constructor.materialize_plan(
            split="train",
            assignments=search.assignments,
            schedule=self.train,
            joint_signatures=self.signatures,
            plan_version=r2.PLAN_VERSION,
        )
        blueprint = world_builder.build_structure_blueprint(
            policy=policy,
            template=template,
            mode=mode,
            world_record=world_record,
            structure_key_hex=common.structure_key_for_split(
                policy, mode=mode, split="train"
            ),
            balanced_schedule=self.train,
            registered_negative_plan=plan,
            joint_signatures=self.signatures,
            candidate_index=0,
        )
        self.assertEqual(blueprint["audit"]["natural_text_field_count"], 0)
        self.assertEqual(blueprint["audit"]["identity_asset_count"], 0)
        self.assertEqual(blueprint["audit"]["pair_label_count"], 0)
        self.assertNotIn("items", blueprint["public"])
        forbidden_fields = {
            "title",
            "description",
            "base_description",
            "natural_variation_phrase",
            "pair_label",
            "qrels",
            "identity_assets",
        }

        def walk(value: object) -> None:
            if isinstance(value, dict):
                self.assertFalse(forbidden_fields & set(value))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(blueprint)
        seller_rows, noise_rows = method_world._structure_rows(
            policy=policy,
            template=template,
            split="train",
            world=blueprint,
            candidate_index=0,
        )
        self.assertEqual(len(seller_rows), 378)
        self.assertEqual(len(noise_rows), 378)

    def test_all_11_count_families_have_frozen_feature_coverage(self) -> None:
        coverage = structure_gate.count_family_coverage()
        audit = structure_gate.validate_count_family_coverage(coverage)
        self.assertEqual(tuple(coverage), structure_gate.COUNT_FAMILIES)
        self.assertEqual(
            common.canonical_sha256(coverage),
            structure_gate.EXPECTED_COUNT_FAMILY_PROJECTION_MAP_SHA256,
        )
        self.assertEqual(audit["count_family_count"], 11)
        self.assertFalse(audit["cross_view_interactions_tested"])
        self.assertFalse(audit["theoretical_5_324_cell_balance_certified"])
        self.assertTrue(
            audit["seller_slot_and_noise_visible_models_remain_separate"]
        )
        self.assertGreater(
            audit["covered_feature_name_counts"]["seller_slot"], 0
        )
        self.assertGreater(
            audit["covered_feature_name_counts"]["noise_visible"], 0
        )
        tampered = copy.deepcopy(coverage)
        tampered["pair_seller"][0]["feature_names"].append(
            "not_a_frozen_feature"
        )
        with self.assertRaises(structure_gate.PrebuildStructureGateError):
            structure_gate.validate_count_family_coverage(tampered)
        reordered = copy.deepcopy(coverage)
        reordered["pair_seller"][0]["feature_names"].reverse()
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError,
            "Preregistered",
        ):
            structure_gate.validate_count_family_coverage(reordered)

    def test_labels_cannot_be_requested_without_frozen_matrix_capability(self) -> None:
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError,
            "cannot be constructed directly",
        ):
            structure_gate._FrozenMatricesCapability(object(), {})
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError,
            "before all structure matrices",
        ):
            structure_gate.materialize_labels_once(
                object(),
                matrices={},
                policy={},
                mode="development_smoke",
                world_records={},
                schedules={},
                access_counts={
                    "train": 0,
                    "development": 0,
                    "audit_a": 0,
                    "audit_b": 0,
                },
            )
        consumed = structure_gate._FrozenMatricesCapability(
            structure_gate._CAPABILITY_TOKEN, {}
        )
        consumed._consumed = True
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError,
            "already consumed",
        ):
            structure_gate.materialize_labels_once(
                consumed,
                matrices={},
                policy={},
                mode="development_smoke",
                world_records={},
                schedules={},
                access_counts={
                    "train": 0,
                    "development": 0,
                    "audit_a": 0,
                    "audit_b": 0,
                },
            )

    def test_structure_gate_wrong_formal_path_fails_before_authority_read(self) -> None:
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError, "output path"
        ):
            structure_gate.run_formal(
                output_directory=ROOT / "wrong_structure_gate_path",
                authority_path=ROOT / structure_gate.AUTHORITY_RELATIVE,
            )

    def test_structure_gate_rejects_partial_published_plan_root(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28_v9_3_r2_partial_root_", dir=ROOT
        ) as temporary:
            plan_root = Path(temporary)
            (plan_root / "construction_receipt.json").write_text(
                "{}\n", encoding="utf-8"
            )
            original = structure_gate.PLAN_ROOT_RELATIVE
            structure_gate.PLAN_ROOT_RELATIVE = plan_root.relative_to(ROOT)
            try:
                with self.assertRaisesRegex(
                    structure_gate.PrebuildStructureGateError,
                    "file-set",
                ):
                    structure_gate.validate_published_plan_root(
                        plan_root=plan_root,
                        schedule_root=PREFLIGHT,
                        signature_path=JOINT_SIGNATURE,
                        schedules={
                            "train": self.train,
                            "development": self.development,
                        },
                        joint_signatures=self.signatures,
                    )
            finally:
                structure_gate.PLAN_ROOT_RELATIVE = original

    def test_matrix_capability_recomputes_frozen_matrix_bytes(self) -> None:
        row_keys = (("world", "pair"),)
        seller = probe_core.freeze_matrix(
            view="seller_slot",
            values=np.asarray([[1.0]], dtype=np.dtype("<f8")),
            row_keys=row_keys,
            column_names=("seller",),
        )
        noise = probe_core.freeze_matrix(
            view="noise_visible",
            values=np.asarray([[1.0]], dtype=np.dtype("<f8")),
            row_keys=row_keys,
            column_names=("noise",),
        )
        matrices = {
            "train": {"seller_slot": seller, "noise_visible": noise},
            "development": {"seller_slot": seller, "noise_visible": noise},
        }
        original_rows = structure_gate.ROWS_PER_SPLIT
        structure_gate.ROWS_PER_SPLIT = 1
        try:
            structure_gate._issue_frozen_matrices_capability(matrices)
            seller.values.setflags(write=True)
            seller.values[0, 0] = 2.0
            seller.values.setflags(write=False)
            with self.assertRaisesRegex(
                structure_gate.PrebuildStructureGateError,
                "commitment drift",
            ):
                structure_gate._issue_frozen_matrices_capability(matrices)
        finally:
            structure_gate.ROWS_PER_SPLIT = original_rows

    def test_probe_result_schema_is_fail_closed(self) -> None:
        policy = read_json(
            ROOT
            / "schema"
            / "step28_v13_v1_13_method_qualification_policy_v9_3.json"
        )
        baseline = 20 / 378
        model_names = (
            "seller_slot::logistic_l2",
            "seller_slot::hist_gradient_boosting_depth2",
            "noise_visible::logistic_l2",
            "noise_visible::hist_gradient_boosting_depth2",
        )
        aucs = (0.51, 0.52, 0.515, 0.525)
        aps = tuple(baseline + value for value in (0.001, 0.002, 0.003, 0.004))
        models = {
            name: {
                "symmetric_roc_auc": auc,
                "average_precision": ap,
                "score_vector_sha256": "a" * 64,
            }
            for name, auc, ap in zip(model_names, aucs, aps, strict=True)
        }
        result = {
            "single_feature_maximum_symmetric_roc_auc_by_view": {
                "seller_slot": 0.51,
                "noise_visible": 0.515,
            },
            "model_results": models,
            "maximum_symmetric_roc_auc": max(aucs),
            "maximum_average_precision_uplift": max(aps) - baseline,
            "bootstrap": {
                "replicates": 9999,
                "world_count": 500,
                "score_family_size": 4,
                "draws_raw_i8_c_sha256": policy["bootstrap"][
                    "draws_raw_i8_c_sha256"
                ],
                "family_max_symmetric_auc_vector_sha256": "b" * 64,
                "family_max_average_precision_uplift_vector_sha256": "c" * 64,
                "symmetric_auc_95_upper": 0.529,
                "average_precision_uplift_95_upper": 0.014,
            },
        }
        audit = structure_gate.validate_probe_result_contract(
            result,
            policy=policy,
            average_precision_baseline=baseline,
        )
        self.assertEqual(audit["model_count"], 4)
        self.assertFalse(audit["matrix_concatenation_used"])

        missing_model = copy.deepcopy(result)
        missing_model["model_results"].pop(model_names[-1])
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError, "schema"
        ):
            structure_gate.validate_probe_result_contract(
                missing_model,
                policy=policy,
                average_precision_baseline=baseline,
            )
        wrong_bootstrap = copy.deepcopy(result)
        wrong_bootstrap["bootstrap"]["replicates"] = 9998
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError, "bootstrap"
        ):
            structure_gate.validate_probe_result_contract(
                wrong_bootstrap,
                policy=policy,
                average_precision_baseline=baseline,
            )
        nonfinite = copy.deepcopy(result)
        nonfinite["bootstrap"]["symmetric_auc_95_upper"] = float("-inf")
        with self.assertRaisesRegex(
            structure_gate.PrebuildStructureGateError, "finite"
        ):
            structure_gate.validate_probe_result_contract(
                nonfinite,
                policy=policy,
                average_precision_baseline=baseline,
            )

    def test_gate_order_is_four_matrices_then_labels_then_models(self) -> None:
        policy = read_json(
            ROOT
            / "schema"
            / "step28_v13_v1_13_method_qualification_policy_v9_3.json"
        )
        events: list[str] = []
        records = (
            {
                "world_uid": "train-world",
                "mode_global_ordinal": 0,
                "split": "train",
                "split_ordinal": 0,
            },
            {
                "world_uid": "development-world",
                "mode_global_ordinal": 1,
                "split": "development",
                "split_ordinal": 0,
            },
        )

        def fake_build_frozen_split(*, split: str, **_kwargs: object) -> dict:
            events.append(f"matrices:{split}")
            row_keys = ((f"{split}-world", "pair"),)
            return {
                "seller_slot": probe_core.freeze_matrix(
                    view="seller_slot",
                    values=np.asarray([[1.0]], dtype=np.dtype("<f8")),
                    row_keys=row_keys,
                    column_names=("seller",),
                ),
                "noise_visible": probe_core.freeze_matrix(
                    view="noise_visible",
                    values=np.asarray([[1.0]], dtype=np.dtype("<f8")),
                    row_keys=row_keys,
                    column_names=("noise",),
                ),
            }

        def fake_labels(
            capability: object,
            *,
            access_counts: dict[str, int],
            **_kwargs: object,
        ) -> dict:
            self.assertEqual(
                events, ["matrices:train", "matrices:development"]
            )
            events.append("labels")
            capability._consumed = True
            access_counts.update(
                {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0}
            )
            return {
                "train": np.asarray([0], dtype=np.int8),
                "development": np.asarray([0], dtype=np.int8),
            }

        baseline = 20 / 378
        model_names = (
            "seller_slot::logistic_l2",
            "seller_slot::hist_gradient_boosting_depth2",
            "noise_visible::logistic_l2",
            "noise_visible::hist_gradient_boosting_depth2",
        )

        def fake_probe(**_kwargs: object) -> dict:
            self.assertEqual(
                events,
                ["matrices:train", "matrices:development", "labels"],
            )
            events.append("models")
            model_rows = {
                name: {
                    "symmetric_roc_auc": 0.5,
                    "average_precision": baseline,
                    "score_vector_sha256": "a" * 64,
                }
                for name in model_names
            }
            return {
                "single_feature_maximum_symmetric_roc_auc_by_view": {
                    "seller_slot": 0.5,
                    "noise_visible": 0.5,
                },
                "model_results": model_rows,
                "maximum_symmetric_roc_auc": 0.5,
                "maximum_average_precision_uplift": 0.0,
                "bootstrap": {
                    "replicates": 9999,
                    "world_count": 500,
                    "score_family_size": 4,
                    "draws_raw_i8_c_sha256": policy["bootstrap"][
                        "draws_raw_i8_c_sha256"
                    ],
                    "family_max_symmetric_auc_vector_sha256": "b" * 64,
                    "family_max_average_precision_uplift_vector_sha256": "c" * 64,
                    "symmetric_auc_95_upper": 0.5,
                    "average_precision_uplift_95_upper": 0.0,
                },
            }

        original_worlds = structure_gate.WORLDS_PER_SPLIT
        original_rows = structure_gate.ROWS_PER_SPLIT
        structure_gate.WORLDS_PER_SPLIT = 1
        structure_gate.ROWS_PER_SPLIT = 1
        try:
            with (
                mock.patch.object(
                    structure_gate.structure,
                    "build_mode_world_pool",
                    return_value=records,
                ),
                mock.patch.object(
                    structure_gate,
                    "_build_frozen_split",
                    side_effect=fake_build_frozen_split,
                ),
                mock.patch.object(
                    structure_gate,
                    "materialize_labels_once",
                    side_effect=fake_labels,
                ),
                mock.patch.object(
                    structure_gate.probe_core,
                    "evaluate_family",
                    side_effect=fake_probe,
                ),
            ):
                result = structure_gate.evaluate_gate(
                    policy=policy,
                    effective_policy={},
                    template={},
                    schedules={"train": {}, "development": {}},
                    plans={"train": {}, "development": {}},
                    joint_signatures={},
                )
        finally:
            structure_gate.WORLDS_PER_SPLIT = original_worlds
            structure_gate.ROWS_PER_SPLIT = original_rows
        self.assertEqual(
            events,
            ["matrices:train", "matrices:development", "labels", "models"],
        )
        self.assertTrue(result["scientific_pass"])
        self.assertEqual(result["label_access_counts"]["audit_a"], 0)
        self.assertEqual(result["label_access_counts"]["audit_b"], 0)


if __name__ == "__main__":
    unittest.main()
