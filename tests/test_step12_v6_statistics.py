from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step12_v6_statistical_robustness_audit as step12  # noqa: E402


class Step12V6StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )

    def test_component_swap_never_splits_a_component(self) -> None:
        candidate = np.asarray([0.9, 0.8, 0.3, 0.2])
        baseline = np.asarray([0.1, 0.2, 0.7, 0.8])
        groups = [np.asarray([0, 1]), np.asarray([2, 3])]
        swapped_candidate, swapped_baseline = step12.component_swapped_scores(
            candidate,
            baseline,
            groups,
            np.asarray([True, False]),
        )
        np.testing.assert_array_equal(swapped_candidate[:2], baseline[:2])
        np.testing.assert_array_equal(swapped_baseline[:2], candidate[:2])
        np.testing.assert_array_equal(swapped_candidate[2:], candidate[2:])
        np.testing.assert_array_equal(swapped_baseline[2:], baseline[2:])

    def test_grouped_randomization_p_value_uses_plus_one_two_sided_formula(self) -> None:
        y_true = np.asarray([1.0, 0.0, 1.0, 0.0])
        candidate_scores = np.asarray([0.9, 0.1, 0.8, 0.2])
        baseline_scores = np.asarray([0.1, 0.9, 0.2, 0.8])
        candidate = {
            "scores": candidate_scores,
            "threshold": 0.5,
        }
        baseline = {
            "scores": baseline_scores,
            "threshold": 0.5,
        }
        result = step12.paired_component_randomization_test(
            candidate,
            baseline,
            y_true,
            [np.asarray([0, 1]), np.asarray([2, 3])],
            "average_precision",
            127,
            19,
        )
        self.assertEqual(result["valid_permutations"], 127)
        self.assertAlmostEqual(
            result["p_value"],
            (1 + result["extreme_count"]) / 128,
        )
        self.assertGreaterEqual(result["p_value"], 1 / 128)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_identical_models_have_randomization_p_value_one(self) -> None:
        y_true = np.asarray([1.0, 0.0, 1.0, 0.0])
        scores = np.asarray([0.9, 0.1, 0.8, 0.2])
        model = {"scores": scores, "threshold": 0.5}
        result = step12.paired_component_randomization_test(
            model,
            model,
            y_true,
            [np.asarray([0, 1]), np.asarray([2, 3])],
            "average_precision",
            31,
            7,
        )
        self.assertEqual(result["p_value"], 1.0)

    def test_holm_uses_only_permutation_p_values(self) -> None:
        rows = [
            {
                "analysis_mode": step12.PRIMARY_ANALYSIS_MODE,
                "evaluation_scope": "all_test",
                "metric": "average_precision",
                "p_value_method": step12.PERMUTATION_P_VALUE_METHOD,
                "permutation_p_value_raw": 0.01,
            },
            {
                "analysis_mode": step12.SUPPLEMENTAL_ANALYSIS_MODE,
                "evaluation_scope": "all_test",
                "metric": "average_precision",
                "p_value_method": "not_computed_for_supplemental_bootstrap",
                "permutation_p_value_raw": 0.0001,
            },
        ]
        step12.holm_adjust(rows)
        self.assertEqual(rows[0]["p_value_holm"], 0.01)
        self.assertEqual(rows[0]["p_value_holm_source"], "permutation_p_value_raw")
        self.assertIsNone(rows[1]["p_value_holm"])

    def test_holm_family_combines_all_test_and_strict_soft_scope(self) -> None:
        rows = [
            {
                "analysis_mode": step12.PRIMARY_ANALYSIS_MODE,
                "evaluation_scope": "all_test",
                "metric": "average_precision",
                "p_value_method": step12.PERMUTATION_P_VALUE_METHOD,
                "permutation_p_value_raw": 0.01,
            },
            {
                "analysis_mode": step12.PRIMARY_ANALYSIS_MODE,
                "evaluation_scope": "strict_plus_soft_primary_positive_vs_all_negative",
                "metric": "average_precision",
                "p_value_method": step12.PERMUTATION_P_VALUE_METHOD,
                "permutation_p_value_raw": 0.02,
            },
        ]
        step12.holm_adjust(rows)
        expected_family = f"{step12.PRIMARY_ANALYSIS_MODE}|average_precision"
        self.assertEqual(rows[0]["holm_family"], expected_family)
        self.assertEqual(rows[1]["holm_family"], expected_family)
        self.assertEqual(rows[0]["p_value_holm"], 0.02)
        self.assertEqual(rows[1]["p_value_holm"], 0.02)

    def test_step9_selection_is_validation_only_with_preregistered_tie_break(self) -> None:
        config = self.policy["validation_selected_aliases"][
            "step9_strongest_clean_validation_selected"
        ]
        validation_values = {
            model_id: 0.75 for model_id in config["candidate_model_ids"]
        }
        selected, record = step12.select_by_validation_metric(
            validation_values,
            config["simplicity_tie_break_order"],
            config["tie_tolerance"],
        )
        self.assertEqual(selected, "step9_e5_lr_l2_100pct_seed_mean")
        self.assertEqual(record["selection_split"], "zh_valid")
        self.assertFalse(record["test_metrics_used_for_selection"])
        validation_values["step9_labse_lr_l2_100pct_seed_mean"] = 0.8
        selected, _ = step12.select_by_validation_metric(
            validation_values,
            config["simplicity_tie_break_order"],
            config["tie_tolerance"],
        )
        self.assertEqual(selected, "step9_labse_lr_l2_100pct_seed_mean")

    def test_m5_and_final_ties_prefer_simpler_preregistered_models(self) -> None:
        aliases = self.policy["validation_selected_aliases"]
        for alias_name, expected in (
            ("step15_v6_m5_selected", "step15_v6_m5_lambda_0p1"),
            ("step15_v6_final_selected", "step15_v6_m3"),
        ):
            config = aliases[alias_name]
            selected, _ = step12.select_by_validation_metric(
                {model_id: 0.7 for model_id in config["candidate_model_ids"]},
                config["simplicity_tie_break_order"],
                config["tie_tolerance"],
            )
            self.assertEqual(selected, expected)

    def test_fail_closed_output_contract_and_validate_only_policy(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            directory = Path(temporary_dir)
            policy = {
                "outputs": {
                    "summary_json": str(directory / "summary.json"),
                    "metrics_csv": str(directory / "metrics.csv"),
                }
            }
            step12.assert_output_targets_absent(policy)
            (directory / "metrics.csv").write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                step12.assert_output_targets_absent(policy)
        self.assertEqual(
            self.policy["fixed_test"]["role"],
            "internal_development_test_not_prospective_final_holdout",
        )

    def test_m5_policy_requires_only_validation_selected_test_predictions(self) -> None:
        config = self.policy["validation_selected_aliases"]["step15_v6_m5_selected"]
        self.assertEqual(
            config["test_prediction_availability"],
            "validation_selected_candidate_only",
        )
        self.assertEqual(
            config["simplicity_tie_break_order"],
            ["step15_v6_m5_lambda_0p1", "step15_v6_m5_lambda_0p3"],
        )

    def test_unselected_m5_validation_candidate_does_not_require_test_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            directory = Path(temporary_dir)
            experiment = "m5_candidate"
            phase = "phase3"
            seed = 1
            valid_path = directory / f"{experiment}_{phase}_{seed}.valid.csv"
            valid_rows = [
                {
                    "pair_uid": f"pair-{idx}",
                    "review_label": "positive" if idx < 2 else "negative",
                    "split_component_id": f"component-{idx}",
                }
                for idx in range(4)
            ]
            scores = [0.9, 0.8, 0.2, 0.1]
            with valid_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pair_uid",
                        "y_true",
                        "review_label",
                        "split_component_id",
                        "prob_positive",
                        "threshold",
                        "experiment_name",
                    ],
                )
                writer.writeheader()
                for row, score in zip(valid_rows, scores, strict=True):
                    writer.writerow(
                        {
                            **row,
                            "y_true": 1 if row["review_label"] == "positive" else 0,
                            "prob_positive": score,
                            "threshold": 0.5,
                            "experiment_name": f"{experiment}_{phase}_seed_{seed}",
                        }
                    )
            relative = str(valid_path.relative_to(ROOT))
            y_valid = step12.labels_from_rows(valid_rows)
            metrics = step12.step7.evaluate_probabilities(
                y_valid, np.asarray(scores, dtype=float), 0.5
            )
            candidate = step12.load_step15_validation_candidate(
                {
                    "model_id": "candidate",
                    "role": "validation_only",
                    "experiment": experiment,
                    "phase": phase,
                },
                [seed],
                {
                    "valid": str(directory / "{experiment}_{phase}_{seed}.valid.csv"),
                    "test": str(directory / "does-not-exist.test.csv"),
                },
                valid_rows,
                y_valid,
                {
                    (experiment, phase, seed): {
                        "output_paths": {"zh_valid_predictions": relative},
                        "zh_valid_metrics": metrics,
                    }
                },
                {relative: {"path": relative}},
                set(),
                json.loads(
                    (ROOT / "schema" / "step15_v6_paper_hardening_policy.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            self.assertEqual(candidate["model_id"], "candidate")
            self.assertEqual(candidate["source_paths"], [relative])
            self.assertFalse((directory / "does-not-exist.test.csv").exists())


if __name__ == "__main__":
    unittest.main()
