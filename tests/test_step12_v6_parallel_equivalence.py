from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step12_v6_statistical_robustness_audit as step12  # noqa: E402


class Step12V6ParallelEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step12_v6_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        cls.y_true = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        cls.test_rows = [
            {"pair_uid": f"pair-{idx}", "split_component_id": f"component-{idx // 2}"}
            for idx in range(len(cls.y_true))
        ]
        cls.groups = step12.component_groups(cls.test_rows)

    @staticmethod
    def model(scores: np.ndarray, role: str) -> dict:
        return {
            "role": role,
            "seed_ids": [1, 2],
            "seed_scores": np.asarray(scores, dtype=float),
            "scores": np.asarray(scores, dtype=float).mean(axis=0),
            "threshold": 0.5000003456789012,
            "threshold_source": "unit_test_validation",
            "valid_average_precision": 0.75,
        }

    def test_metric_specific_fast_path_matches_full_evaluation_exactly(self) -> None:
        scores = np.asarray([0.8, 0.8, 0.7, 0.7, 0.4, 0.4, 0.2, 0.2])
        full = step12.step7.evaluate_probabilities(self.y_true, scores, 0.5)
        for metric in (
            "roc_auc",
            "average_precision",
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "tp",
            "tn",
            "fp",
            "fn",
        ):
            expected = step12.evaluation_metric_value(full, metric)
            observed = step12.metric_value(metric, self.y_true, scores, 0.5)
            self.assertEqual(None if expected is None else float(expected), observed)

    def test_model_rows_are_identical_between_serial_and_process_pool(self) -> None:
        model_a = self.model(
            np.asarray(
                [
                    [0.91, 0.20, 0.78, 0.32, 0.70, 0.35, 0.65, 0.30],
                    [0.86, 0.25, 0.82, 0.28, 0.74, 0.31, 0.60, 0.34],
                ]
            ),
            "candidate",
        )
        model_b = self.model(
            np.asarray(
                [
                    [0.71, 0.30, 0.68, 0.42, 0.60, 0.45, 0.55, 0.40],
                    [0.76, 0.35, 0.62, 0.38, 0.64, 0.41, 0.50, 0.44],
                ]
            ),
            "baseline",
        )
        models = {"candidate": model_a, "baseline": model_b}
        serial = step12.model_metric_rows(
            models,
            self.test_rows,
            self.y_true,
            self.groups,
            copy.deepcopy(self.policy),
            47,
            worker_count=1,
        )
        parallel = step12.model_metric_rows(
            models,
            self.test_rows,
            self.y_true,
            self.groups,
            copy.deepcopy(self.policy),
            47,
            worker_count=2,
        )
        self.assertEqual(serial, parallel)

    def test_slice_rows_are_identical_between_serial_and_process_pool(self) -> None:
        models = {
            "candidate": self.model(
                np.asarray(
                    [
                        [0.91, 0.20, 0.78, 0.32, 0.70, 0.35, 0.65, 0.30],
                        [0.86, 0.25, 0.82, 0.28, 0.74, 0.31, 0.60, 0.34],
                    ]
                ),
                "candidate",
            ),
            "baseline": self.model(
                np.asarray(
                    [
                        [0.71, 0.30, 0.68, 0.42, 0.60, 0.45, 0.55, 0.40],
                        [0.76, 0.35, 0.62, 0.38, 0.64, 0.41, 0.50, 0.44],
                    ]
                ),
                "baseline",
            ),
        }
        masks = {
            "all": np.ones(len(self.y_true), dtype=bool),
            "first_six": np.asarray([True, True, True, True, True, True, False, False]),
        }
        serial = step12.evidence_slice_rows(
            models,
            self.test_rows,
            self.y_true,
            masks,
            copy.deepcopy(self.policy),
            43,
            worker_count=1,
        )
        parallel = step12.evidence_slice_rows(
            models,
            self.test_rows,
            self.y_true,
            masks,
            copy.deepcopy(self.policy),
            43,
            worker_count=2,
        )
        self.assertEqual(serial, parallel)

    def test_comparison_scope_rows_are_schedule_independent(self) -> None:
        candidate = self.model(
            np.asarray(
                [
                    [0.91, 0.20, 0.78, 0.32, 0.70, 0.35, 0.65, 0.30],
                    [0.86, 0.25, 0.82, 0.28, 0.74, 0.31, 0.60, 0.34],
                ]
            ),
            "candidate",
        )
        baseline = self.model(
            np.asarray(
                [
                    [0.71, 0.30, 0.68, 0.42, 0.60, 0.45, 0.55, 0.40],
                    [0.76, 0.35, 0.62, 0.38, 0.64, 0.41, 0.50, 0.44],
                ]
            ),
            "baseline",
        )
        task = (
            {
                "comparison_id": "candidate_vs_baseline",
                "candidate": "candidate",
                "baseline": "baseline",
            },
            "all_test",
            candidate,
            baseline,
            self.y_true,
            self.groups,
            ["average_precision", "roc_auc", "pr_auc"],
            37,
            20260711,
            0.95,
            41,
            20260712,
        )
        serial = step12.ordered_process_map(step12._comparison_scope_task, [task, task], 1)
        parallel = step12.ordered_process_map(step12._comparison_scope_task, [task, task], 2)
        self.assertEqual(serial, parallel)

    def test_policy_freezes_parallelism_as_execution_only(self) -> None:
        execution = self.policy["parallel_execution"]
        self.assertEqual(execution["backend"], "process_pool")
        self.assertEqual(execution["max_workers"], 24)
        self.assertEqual(execution["native_threads_per_worker"], 1)
        self.assertTrue(execution["deterministic_task_order"])


if __name__ == "__main__":
    unittest.main()
