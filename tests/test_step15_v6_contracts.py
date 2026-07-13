from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step15_train_incremental_hard_negative as step15  # noqa: E402
import step15_build_v6_inductive_pair_features as inductive  # noqa: E402
import step15_v6_source_only_lr_baseline as source_lr  # noqa: E402
import step9_run_few_shot_adaptation as step9  # noqa: E402


class Step15V6ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "schema" / "step15_v6_paper_hardening_policy.json").open(
            "r", encoding="utf-8"
        ) as handle:
            cls.policy = json.load(handle)

    def test_strict_clean_feature_lineage(self) -> None:
        features = self.policy["feature_sets"]["strict_clean_30d"]
        self.assertEqual(len(features), 30)
        self.assertEqual(len(set(features)), 30)
        self.assertFalse(set(features) & set(self.policy["forbidden_strict_clean_features"]))

    def test_normalized_retrieval_ablation_is_explicit(self) -> None:
        strict = set(self.policy["feature_sets"]["strict_clean_30d"])
        retrieval = set(self.policy["feature_sets"]["normalized_retrieval_33d"])
        self.assertEqual(
            retrieval - strict,
            {
                "candidate_rule_count_non_identifier",
                "sparse_lexical_similarity_raw",
                "structural_support_score_raw",
            },
        )
        domain_features = set(
            self.policy["feature_preprocessing"]["normalized_retrieval_33d"][
                "domain_standardized_features"
            ]
        )
        self.assertEqual(
            domain_features,
            {"sparse_lexical_similarity_raw", "structural_support_score_raw"},
        )

    def test_domain_standardizer_rejects_unknown_domain(self) -> None:
        rows = [
            {"step15_pool": "en", "x": "1"},
            {"step15_pool": "zh", "x": "3"},
        ]
        bundle = step15.fit_standardizer_bundle(
            rows,
            ["x"],
            {"domain_standardized_features": ["x"], "allowed_domains": ["en", "zh"]},
        )
        with self.assertRaisesRegex(ValueError, "Unknown domain"):
            step15.apply_standardizer_bundle(
                [{"step15_pool": "unexpected", "x": "2"}],
                ["x"],
                bundle,
            )

    def test_component_weights_reduce_repeated_component_mass(self) -> None:
        rows = [
            {"pair_uid": "a", "step15_pool": "zh", "split_component_id": "large"},
            {"pair_uid": "b", "step15_pool": "zh", "split_component_id": "large"},
            {"pair_uid": "c", "step15_pool": "zh", "split_component_id": "large"},
            {"pair_uid": "d", "step15_pool": "zh", "split_component_id": "single"},
        ]
        adjusted, diagnostics = step15.apply_component_inverse_sqrt_weights(
            np.ones(4, dtype=float), rows
        )
        self.assertLess(adjusted[0], adjusted[3])
        self.assertAlmostEqual(float(np.mean(adjusted)), 1.0, places=12)
        self.assertEqual(diagnostics["largest_component_edge_count"], 3)

    def test_class_balance_equalizes_incoming_effective_weight_mass(self) -> None:
        weights = np.asarray([0.2, 0.8, 2.0, 4.0], dtype=float)
        labels = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float)
        adjusted, diagnostics = step15.apply_class_balance_multipliers(weights, labels)
        self.assertAlmostEqual(float(adjusted[labels == 1.0].sum()), float(adjusted[labels == 0.0].sum()))
        self.assertEqual(
            diagnostics["method"],
            "equalize_effective_weight_mass_after_component_and_row_quality_weights",
        )

    def test_auxiliary_head_inherits_identity_effective_weights(self) -> None:
        identity_weights = np.asarray([0.2, 0.8, 2.0, 4.0], dtype=float)
        evidence_labels = np.asarray([0, 0, 1, -1], dtype=int)
        adjusted, diagnostics = step15.evidence_weights(
            evidence_labels,
            3,
            identity_weights,
            class_balance=True,
        )
        self.assertEqual(adjusted[3], 0.0)
        self.assertAlmostEqual(float(adjusted[evidence_labels == 0].sum()), 1.5)
        self.assertAlmostEqual(float(adjusted[evidence_labels == 1].sum()), 1.5)
        self.assertEqual(
            diagnostics["method"],
            "identity_effective_weight_chain_then_evidence_class_balance",
        )

    def test_warm_start_phase_plan_rejects_missing_prefix(self) -> None:
        phase_by_id = {phase["phase_id"]: phase for phase in self.policy["curriculum_phases"]}
        cfg = self.policy["experiments"]["step15_v6_m3_warm_start_curriculum"]
        with self.assertRaisesRegex(ValueError, "complete configured prefix"):
            step15.resolve_experiment_phase_plan(
                "step15_v6_m3_warm_start_curriculum",
                cfg,
                ["phase3_add_contact_url_noise"],
                phase_by_id,
            )

    def test_partial_warm_start_prefix_never_exposes_test(self) -> None:
        experiment = "step15_v6_m3_warm_start_curriculum"
        self.assertFalse(
            step15.should_evaluate_test_endpoint(
                experiment,
                "phase1_add_semantic_topic_negative",
                ["phase0_identity_anchor", "phase1_add_semantic_topic_negative"],
                self.policy,
            )
        )
        self.assertTrue(
            step15.should_evaluate_test_endpoint(
                experiment,
                "phase3_add_contact_url_noise",
                self.policy["experiments"][experiment]["phase_ids"],
                self.policy,
            )
        )

    def test_m5_candidates_are_valid_only_before_selection(self) -> None:
        experiment = "step15_v6_m5_aux_evidence_lambda_0p1"
        self.assertFalse(
            step15.should_evaluate_test_endpoint(
                experiment,
                "phase3_add_contact_url_noise",
                self.policy["experiments"][experiment]["phase_ids"],
                self.policy,
            )
        )
        self.assertEqual(
            self.policy["validation_only_model_selection"]["m5_auxiliary_loss_weight"][
                "tie_break_order"
            ][0],
            experiment,
        )

    def test_selected_m5_test_is_materialized_from_frozen_artifact_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            artifact_path = directory / "artifact.json"
            valid_path = directory / "valid.csv"
            test_template = str(directory / "{experiment_name}_{phase_id}_{seed}.test.csv")
            bundle = {
                "feature_names": ["x"],
                "domain_standardized_features": [],
                "global_means": [0.0],
                "global_stds": [1.0],
                "sha256": "fixture",
            }
            step15.atomic_write_json(
                artifact_path,
                {
                    "feature_names": ["x"],
                    "standardizer_bundle": bundle,
                    "frozen_zh_valid_threshold": 0.5,
                    "params": {
                        "w1": [[1.0]],
                        "b1": [0.0],
                        "wi": [2.0],
                        "bi": [0.0],
                        "we": [[0.0]],
                        "be": [0.0],
                    },
                },
            )
            step15.atomic_write_csv(
                valid_path,
                [{"pair_uid": "valid", "threshold": 0.5}],
                ["pair_uid", "threshold"],
            )
            run = {
                "experiment_name": "m5a",
                "phase_id": "phase3",
                "seed": 1,
                "output_paths": {
                    "artifact": str(artifact_path.relative_to(ROOT)),
                    "zh_valid_predictions": str(valid_path.relative_to(ROOT)),
                    "zh_test_predictions": None,
                },
            }
            rows = [
                {
                    "pair_uid": "positive",
                    "data_bucket": "zh_target_strict",
                    "step15_pool": "zh_target_strict",
                    "split_name": "test",
                    "review_label": "positive",
                    "usable_for_supervision": "1",
                    "usable_for_core_transfer": "1",
                    "split_component_id": "p",
                    "x": "2",
                },
                {
                    "pair_uid": "negative",
                    "data_bucket": "zh_target_strict",
                    "step15_pool": "zh_target_strict",
                    "split_name": "test",
                    "review_label": "negative",
                    "usable_for_supervision": "1",
                    "usable_for_core_transfer": "1",
                    "split_component_id": "n",
                    "x": "-2",
                },
            ]
            policy = {
                "training": {"default_seeds": [1]},
                "splits": {"target_test": "test"},
                "validation_only_model_selection": {
                    "m5": {"candidate_experiments": ["m5a"], "phase_id": "phase3"}
                },
                "outputs": {"zh_test_predictions_template": test_template},
            }
            updated = step15.materialize_validation_selected_test_predictions(
                [run],
                {"m5": {"selected_experiment": "m5a"}},
                policy,
                {"zh_target_strict": rows},
            )[0]
            self.assertEqual(
                updated["zh_test_evaluation_role"],
                "validation_selected_frozen_artifact_preregistered_endpoint_only",
            )
            self.assertFalse(updated["test_materialization"]["training_reused"])
            self.assertTrue(step15.resolve_path(updated["output_paths"]["zh_test_predictions"]).exists())
            self.assertEqual(updated["zh_test_metrics"]["accuracy"], 1.0)
    def test_inductive_reference_uses_only_training_sellers(self) -> None:
        profiles = {
            "train_a": {
                "source_market_raw": "m",
                "signature_titles": [{"value": "shared train title"}],
                "signature_description_segments": [],
                "item_count": 1,
            },
            "train_b": {
                "source_market_raw": "m",
                "signature_titles": [{"value": "other train title"}],
                "signature_description_segments": [],
                "item_count": 3,
            },
            "test_only": {
                "source_market_raw": "m",
                "signature_titles": [{"value": "shared train title"}],
                "signature_description_segments": [],
                "item_count": 100,
            },
        }
        reference = inductive.fit_reference(
            profiles,
            {"train_a", "train_b"},
            {"item_count": "item_count"},
            {
                "minimum_signature_value_length": 1,
                "boilerplate_seller_share_threshold": 0.75,
                "rare_seller_share_threshold": 0.5,
            },
        )
        self.assertEqual(reference["train_seller_count"], 2)
        self.assertEqual(reference["title_df"]["shared train title"], 1)
        self.assertNotIn(100.0, reference["domain_numeric_values"]["item_count"])

    def test_v6_policy_points_to_isolated_inductive_features(self) -> None:
        lineage = self.policy["inductive_feature_lineage"]
        self.assertEqual(lineage["reference_scope"], "frozen_train_sellers_only")
        for pool in self.policy["pools"].values():
            self.assertTrue(pool["pair_features"].startswith("reports/step15_v6/features/"))

    def test_linux_runner_never_rebuilds_canonical_inputs(self) -> None:
        runner = (ROOT / "scripts" / "run_step15_v6_linux_20260711.sh").read_text(
            encoding="utf-8"
        )
        for script_name in (
            "step4_build_silver_candidates.py",
            "step7_build_pair_feature_preview.py",
            "step7_refresh_nonsemantic_pair_features.py",
            "step15_build_evidence_type_labels.py",
        ):
            self.assertNotIn(f'"$PYTHON_BIN" scripts/{script_name}', runner)

    def test_curriculum_and_mixup_have_matched_budget_controls(self) -> None:
        for experiment_name in (
            "step15_v6_m0_all_at_once_binary",
            "step15_v6_m1_evidence_weighted",
            "step15_v6_m2_domain_balanced",
        ):
            self.assertEqual(
                self.policy["experiments"][experiment_name]["fixed_update_budget_per_phase"],
                1000,
            )
        m2b = self.policy["experiments"]["step15_v6_m2b_matched_budget_full_data_replay"]
        m3 = self.policy["experiments"]["step15_v6_m3_warm_start_curriculum"]
        self.assertEqual(m2b["phase_ids"], m3["phase_ids"])
        self.assertEqual(m2b["fixed_update_budget_per_phase"], m3["fixed_update_budget_per_phase"])
        self.assertIn("training_evidence_types_override", m2b)
        m4 = self.policy["experiments"]["step15_v6_m4_trusted_positive_mixup"]
        m4c = self.policy["experiments"]["step15_v6_m4c_matched_continuation_no_mixup"]
        self.assertEqual(m4["phase_ids"], m4c["phase_ids"])
        self.assertEqual(m4["fixed_update_budget_per_phase"], m4c["fixed_update_budget_per_phase"])
        self.assertTrue(m4c["disable_positive_mixup"])
        self.assertEqual(
            self.policy["experiments"]["step15_v6_m2_domain_balanced"][
                "fixed_update_budget_per_phase"
            ],
            len(m3["phase_ids"]) * m3["fixed_update_budget_per_phase"],
        )

    def test_source_only_prediction_contract_exposes_experiment_name(self) -> None:
        rows = [
            {
                "pair_uid": "pair",
                "split_component_id": "component",
                "review_label": "positive",
            }
        ]
        prediction = source_lr.prediction_rows(
            rows,
            np.asarray([0.75]),
            0.5,
            "source_model_seed_1",
        )[0]
        self.assertEqual(prediction["experiment_name"], "source_model_seed_1")
        self.assertEqual(prediction["model_id"], "source_model_seed_1")
        source_cfg = self.policy["source_only_lr_baseline"]
        self.assertFalse(source_cfg["target_identity_labels_used_for_training"])
        self.assertFalse(source_cfg["strict_zero_shot_without_target_covariates"])

    def test_prediction_writers_preserve_sub_micro_ranking_differences(self) -> None:
        rows = [
            {
                "pair_uid": "positive",
                "data_bucket": "zh_target_strict",
                "split_name": "valid",
                "review_label": "positive",
                "review_stratum": "semantic_structural",
                "source_seller_raw_left": "seller-a",
                "source_seller_raw_right": "seller-b",
            },
            {
                "pair_uid": "negative",
                "data_bucket": "zh_target_strict",
                "split_name": "valid",
                "review_label": "negative",
                "review_stratum": "semantic_structural",
                "source_seller_raw_left": "seller-c",
                "source_seller_raw_right": "seller-d",
            },
        ]
        probabilities = np.asarray([0.5000004, 0.5000003], dtype=float)

        for prediction_builder in (
            lambda: step15.prediction_rows(rows, probabilities, 0.5, "step15"),
            lambda: step15.step7.prediction_rows(rows, probabilities, 0.5, "step7"),
            lambda: source_lr.prediction_rows(rows, probabilities, 0.5, "source"),
        ):
            predictions = prediction_builder()
            persisted_scores = np.asarray(
                [float(row["prob_positive"]) for row in predictions],
                dtype=float,
            )
            self.assertGreater(persisted_scores[0], persisted_scores[1])
            self.assertEqual(
                step15.step7.roc_auc_score(np.asarray([1.0, 0.0]), persisted_scores),
                1.0,
            )

    def test_output_paths_are_isolated(self) -> None:
        for key, value in self.policy["outputs"].items():
            if key == "summary_write_mode":
                continue
            self.assertTrue(value.startswith("reports/step15_v6/"), value)

    def test_atomic_writers_create_isolated_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "new" / "predictions" / "rows.csv"
            json_path = root / "new" / "artifacts" / "artifact.json"
            step15.atomic_write_csv(csv_path, [{"x": 1}], ["x"])
            step15.atomic_write_json(json_path, {"x": 1})
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

    def test_internal_test_is_endpoint_only(self) -> None:
        self.assertEqual(
            self.policy["evaluation_boundary"]["zh_test_evaluation_schedule"],
            "final_preregistered_endpoint_only",
        )

    def test_summary_merges_only_same_input_manifest(self) -> None:
        self.assertEqual(
            self.policy["outputs"]["summary_write_mode"],
            "merge_by_experiment_phase_seed_same_input_manifest_only",
        )

    def test_step9_output_root_isolates_every_template(self) -> None:
        policy = {
            "output_templates": {
                "summary": "reports/step9_few_shot_summary.json",
                "prediction": "reports/step9_{experiment_name}_{seed}.csv",
            }
        }
        step9.redirect_output_templates(policy, "reports/step15_v6/baselines/step9")
        self.assertEqual(
            policy["runtime_output_root"], "reports/step15_v6/baselines/step9"
        )
        self.assertTrue(
            all(
                value.startswith("reports/step15_v6/baselines/step9/")
                for value in policy["output_templates"].values()
            )
        )

    def test_candidate_universe_verification_is_frozen_and_passed(self) -> None:
        verification = json.loads(
            (
                ROOT
                / "reports"
                / "step15_v6"
                / "manifests"
                / "step4_candidate_universe_verification.windows.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(verification["mismatches"], {})
        self.assertEqual(
            {
                pool: record["pair_count"]
                for pool, record in verification["pools"].items()
            },
            {
                "en_content_train_pool": 6683,
                "zh_target_strict": 3857,
                "zh_target_aux": 580,
            },
        )


if __name__ == "__main__":
    unittest.main()
