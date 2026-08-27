from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_method_policy_v9_3 as method_policy
import step28_v13_v1_13_quality_probe_core_v9_3 as core
import step28_v13_v1_13_quality_probe_validator_v9 as prior_validated_core


class QualityProbeCoreV93Contracts(unittest.TestCase):
    def test_freeze_matrix_commits_exact_bytes_rows_and_columns(self) -> None:
        values = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        frozen = core.freeze_matrix(
            view="fixture",
            values=values,
            row_keys=(("w0", "p0"), ("w0", "p1")),
            column_names=("a", "b"),
        )
        self.assertEqual(frozen.commitment["shape"], [2, 2])
        self.assertFalse(frozen.values.flags.writeable)
        values[0, 0] = 9.0
        self.assertEqual(frozen.values[0, 0], 0.0)

    def test_freeze_matrix_rejects_duplicate_rows_and_nonfinite_values(self) -> None:
        with self.assertRaises(core.QualityProbeCoreV93Error):
            core.freeze_matrix(
                view="fixture",
                values=np.asarray([[0.0], [np.nan]]),
                row_keys=(("w", "p"), ("w", "p")),
                column_names=("a",),
            )

    def test_fixed_models_fit_train_and_score_development_only(self) -> None:
        policy = method_policy.load_policy()
        train_rows = 120
        development_rows = 80
        train_y = np.asarray([index % 2 for index in range(train_rows)], dtype=np.int8)
        development_y = np.asarray(
            [index % 2 for index in range(development_rows)], dtype=np.int8
        )
        train_x = np.column_stack((train_y, np.arange(train_rows) % 7)).astype(
            np.float64
        )
        development_x = np.column_stack(
            (development_y, np.arange(development_rows) % 7)
        ).astype(np.float64)
        train = core.freeze_matrix(
            view="fixture",
            values=train_x,
            row_keys=tuple((f"tw{index // 20}", f"tp{index}") for index in range(train_rows)),
            column_names=("signal", "noise"),
        )
        development = core.freeze_matrix(
            view="fixture",
            values=development_x,
            row_keys=tuple(
                (f"dw{index // 20}", f"dp{index}")
                for index in range(development_rows)
            ),
            column_names=("signal", "noise"),
        )
        result = core.evaluate_family(
            train={"fixture": train},
            development={"fixture": development},
            train_labels=train_y,
            development_labels=development_y,
            policy=policy,
            average_precision_baseline=0.5,
            bootstrap=False,
        )
        self.assertEqual(
            result["single_feature_maximum_symmetric_roc_auc_by_view"]["fixture"],
            1.0,
        )
        self.assertEqual(set(result["model_results"]), {
            "fixture::logistic_l2",
            "fixture::hist_gradient_boosting_depth2",
        })
        self.assertIsNone(result["bootstrap"])

    def test_bootstrap_is_byte_and_metric_equivalent_to_validated_v9_core(self) -> None:
        worlds = tuple(f"w{index}" for index in range(5))
        row_worlds = tuple(world for world in worlds for _ in range(4))
        labels = np.asarray([0, 0, 1, 1] * 5, dtype=np.int8)
        scores = {
            "a": np.asarray([0.1, 0.2, 0.8, 0.9] * 5, dtype=np.float64),
            "b": np.asarray([0.4, 0.4, 0.6, 0.6] * 5, dtype=np.float64),
        }
        draws = core.generate_bootstrap_draws(
            replicates=31, world_count=5, seed=12345
        )
        expected_draws = prior_validated_core.generate_bootstrap_draws(
            replicates=31, world_count=5, seed=12345
        )
        self.assertTrue(np.array_equal(draws, expected_draws))
        actual = core._bootstrap_family_upper(
            labels=labels,
            row_world_uids=row_worlds,
            ordered_world_uids=worlds,
            score_family=scores,
            baseline=0.5,
            draws=draws,
            batch_size=16,
        )
        expected = prior_validated_core._bootstrap_family_upper(
            labels=labels,
            row_world_uids=row_worlds,
            ordered_world_uids=worlds,
            score_family=scores,
            baseline=0.5,
            draws=draws,
            batch_size=16,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
