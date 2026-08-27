from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_probe_core_v9_4 as core
import step28_v13_v1_13_quality_probe_validator_v9 as validated_core


def policy(*, replicates: int = 31, world_count: int = 5, seed: int = 12345):
    draws = core.generate_bootstrap_draws(
        replicates=replicates, world_count=world_count, seed=seed
    )
    return {
        "probe_models": {
            "logistic_l2": {
                "preprocessing": (
                    "StandardScaler_fit_on_train_only_then_transform_"
                    "train_and_development"
                ),
                "standard_scaler": {
                    "copy": True,
                    "with_mean": True,
                    "with_std": True,
                },
                "C": 1.0,
                "class_weight": None,
                "dual": False,
                "fit_intercept": True,
                "intercept_scaling": 1,
                "l1_ratio": None,
                "max_iter": 10000,
                "multi_class": "deprecated",
                "n_jobs": None,
                "penalty": "l2",
                "random_state": 793820367,
                "solver": "lbfgs",
                "tol": 1e-10,
                "verbose": 0,
                "warm_start": False,
            },
            "hist_gradient_boosting_depth2": {
                "class": "sklearn.ensemble.HistGradientBoostingClassifier",
                "preprocessing": "raw_unstandardized_float64",
                "categorical_features": "from_dtype",
                "class_weight": None,
                "early_stopping": False,
                "interaction_cst": None,
                "l2_regularization": 1.0,
                "learning_rate": 0.03,
                "loss": "log_loss",
                "max_bins": 255,
                "max_depth": 2,
                "max_features": 1.0,
                "max_iter": 200,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "monotonic_cst": None,
                "n_iter_no_change": 10,
                "random_state": 793820367,
                "scoring": "loss",
                "tol": 1e-7,
                "validation_fraction": 0.1,
                "verbose": 0,
                "warm_start": False,
            },
        },
        "bootstrap": {
            "replicates": replicates,
            "development_world_count": world_count,
            "seed": seed,
            "draws_raw_i8_c_sha256": __import__("hashlib").sha256(
                draws.tobytes(order="C")
            ).hexdigest(),
            "streaming_batch_size": 16,
        },
    }


class QualityProbeCoreV94Contracts(unittest.TestCase):
    def test_freeze_matrix_commits_copy_and_is_read_only(self) -> None:
        values = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        frozen = core.freeze_matrix(
            view="model_visible_14",
            values=values,
            row_keys=(("w0", "p0"), ("w0", "p1")),
            column_names=("a", "b"),
        )
        values[0, 0] = 9.0
        self.assertEqual(frozen.values[0, 0], 0.0)
        self.assertFalse(frozen.values.flags.writeable)
        with self.assertRaises(ValueError):
            frozen.values.setflags(write=True)
        self.assertEqual(frozen.commitment["shape"], (2, 2))
        with self.assertRaises(TypeError):
            frozen.commitment["matrix_raw_f8_c_sha256"] = "0" * 64
        core.verify_frozen_matrix(frozen)

    def test_fixed_models_fit_train_and_score_development(self) -> None:
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
            view="model_visible_14",
            values=train_x,
            row_keys=tuple(
                (f"tw{index // 20}", f"tp{index}") for index in range(train_rows)
            ),
            column_names=("signal", "noise"),
        )
        development = core.freeze_matrix(
            view="model_visible_14",
            values=development_x,
            row_keys=tuple(
                (f"dw{index // 20}", f"dp{index}")
                for index in range(development_rows)
            ),
            column_names=("signal", "noise"),
        )
        result = core._evaluate_family(
            train={"model_visible_14": train},
            development={"model_visible_14": development},
            train_labels=train_y,
            development_labels=development_y,
            train_label_row_keys=train.row_keys,
            development_label_row_keys=development.row_keys,
            policy=policy(world_count=4),
            average_precision_baseline=0.5,
            bootstrap=False,
        )
        self.assertEqual(
            result["single_feature_maximum_symmetric_roc_auc_by_view"][
                "model_visible_14"
            ],
            1.0,
        )
        self.assertEqual(
            set(result["model_results"]),
            {
                "model_visible_14::logistic_l2",
                "model_visible_14::hist_gradient_boosting_depth2",
            },
        )

    def test_bootstrap_matches_prior_validated_numerical_core(self) -> None:
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
        expected_draws = validated_core.generate_bootstrap_draws(
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
        expected = validated_core._bootstrap_family_upper(
            labels=labels,
            row_world_uids=row_worlds,
            ordered_world_uids=worlds,
            score_family=scores,
            baseline=0.5,
            draws=draws,
            batch_size=16,
        )
        self.assertEqual(actual, expected)

    def test_bootstrap_sparse_branch_matches_independent_expansion(self) -> None:
        worlds = tuple(f"w{index}" for index in range(5))
        row_worlds = tuple(world for world in worlds for _ in range(10))
        labels = np.asarray(([0] * 8 + [1] * 2) * 5, dtype=np.int8)
        scores = {
            "forward": np.linspace(0.01, 0.99, len(labels), dtype=np.float64),
            "reverse": np.linspace(0.99, 0.01, len(labels), dtype=np.float64),
        }
        draws = core.generate_bootstrap_draws(
            replicates=31, world_count=5, seed=12345
        )
        actual = core._bootstrap_family_upper(
            labels=labels,
            row_world_uids=row_worlds,
            ordered_world_uids=worlds,
            score_family=scores,
            baseline=0.2,
            draws=draws,
            batch_size=16,
        )
        replicate_auc: list[float] = []
        replicate_ap: list[float] = []
        row_worlds_array = np.asarray(row_worlds)
        for draw in draws:
            selected_worlds = [worlds[index] for index in draw]
            selected_rows = np.concatenate(
                [np.flatnonzero(row_worlds_array == world) for world in selected_worlds]
            )
            auc_values: list[float] = []
            ap_values: list[float] = []
            for score in scores.values():
                auc = float(roc_auc_score(labels[selected_rows], score[selected_rows]))
                auc_values.append(max(auc, 1.0 - auc))
                ap_values.append(
                    float(
                        average_precision_score(
                            labels[selected_rows], score[selected_rows]
                        )
                        - 0.2
                    )
                )
            replicate_auc.append(max(auc_values))
            replicate_ap.append(max(ap_values))
        self.assertAlmostEqual(
            actual["symmetric_auc_95_upper"],
            float(np.quantile(replicate_auc, 0.95, method="linear")),
            places=15,
        )
        self.assertAlmostEqual(
            actual["average_precision_uplift_95_upper"],
            float(np.quantile(replicate_ap, 0.95, method="linear")),
            places=15,
        )

    def test_bootstrap_dense_tie_branch_matches_independent_expansion(self) -> None:
        worlds = tuple(f"w{index}" for index in range(5))
        row_worlds = tuple(world for world in worlds for _ in range(10))
        labels = np.asarray(([0] * 8 + [1] * 2) * 5, dtype=np.int8)
        scores = {
            "four_ties": np.asarray([0.1, 0.2, 0.3, 0.4, 0.1] * 10),
            "two_ties": np.asarray([0.25] * 25 + [0.75] * 25),
        }
        draws = core.generate_bootstrap_draws(
            replicates=31, world_count=5, seed=12345
        )
        actual = core._bootstrap_family_upper(
            labels=labels,
            row_world_uids=row_worlds,
            ordered_world_uids=worlds,
            score_family=scores,
            baseline=0.2,
            draws=draws,
            batch_size=16,
        )
        row_worlds_array = np.asarray(row_worlds)
        replicate_auc: list[float] = []
        replicate_ap: list[float] = []
        for draw in draws:
            selected_rows = np.concatenate([
                np.flatnonzero(row_worlds_array == worlds[index])
                for index in draw
            ])
            auc_values: list[float] = []
            ap_values: list[float] = []
            for score in scores.values():
                auc = float(roc_auc_score(labels[selected_rows], score[selected_rows]))
                auc_values.append(max(auc, 1.0 - auc))
                ap_values.append(float(
                    average_precision_score(
                        labels[selected_rows], score[selected_rows]
                    ) - 0.2
                ))
            replicate_auc.append(max(auc_values))
            replicate_ap.append(max(ap_values))
        self.assertAlmostEqual(
            actual["symmetric_auc_95_upper"],
            float(np.quantile(replicate_auc, 0.95, method="linear")),
            places=15,
        )
        self.assertAlmostEqual(
            actual["average_precision_uplift_95_upper"],
            float(np.quantile(replicate_ap, 0.95, method="linear")),
            places=15,
        )

    def test_keys_labels_and_split_are_fail_closed(self) -> None:
        values = np.asarray([[0.0], [1.0]], dtype=np.float64)
        with self.assertRaisesRegex(core.QualityProbeCoreV94Error, "row-key type"):
            core.freeze_matrix(
                view="model_visible_14",
                values=values,
                row_keys=((1, "p0"), (1, "p1")),
                column_names=("a",),
            )
        frozen = core.freeze_matrix(
            view="model_visible_14",
            values=values,
            row_keys=(("w0", "p0"), ("w0", "p1")),
            column_names=("a",),
        )
        with self.assertRaisesRegex(core.QualityProbeCoreV94Error, "world overlap"):
            core._evaluate_family(
                train={"model_visible_14": frozen},
                development={"model_visible_14": frozen},
                train_labels=np.asarray([0, 1], dtype=np.int8),
                development_labels=np.asarray([0, 1], dtype=np.int8),
                train_label_row_keys=frozen.row_keys,
                development_label_row_keys=frozen.row_keys,
                policy=policy(world_count=1),
                average_precision_baseline=0.5,
                bootstrap=False,
            )
        with self.assertRaisesRegex(core.QualityProbeCoreV94Error, "dtype drift"):
            core._validate_labels(
                np.asarray([0.1, 0.9], dtype=np.float64), 2, label="test"
            )
        frozen_labels = core._validate_labels(
            np.asarray([0, 1], dtype=np.int8), 2, label="test"
        )
        with self.assertRaises(ValueError):
            frozen_labels.setflags(write=True)

    def test_label_keys_and_prevalence_must_match(self) -> None:
        train = core.freeze_matrix(
            view="model_visible_14",
            values=np.asarray([[0.0], [1.0]], dtype=np.float64),
            row_keys=(("train", "p0"), ("train", "p1")),
            column_names=("a",),
        )
        development = core.freeze_matrix(
            view="model_visible_14",
            values=np.asarray([[0.0], [1.0]], dtype=np.float64),
            row_keys=(("development", "p0"), ("development", "p1")),
            column_names=("a",),
        )
        labels = np.asarray([0, 1], dtype=np.int8)
        with self.assertRaisesRegex(core.QualityProbeCoreV94Error, "row-key join"):
            core._evaluate_family(
                train={"model_visible_14": train},
                development={"model_visible_14": development},
                train_labels=labels,
                development_labels=labels,
                train_label_row_keys=tuple(reversed(train.row_keys)),
                development_label_row_keys=development.row_keys,
                policy=policy(world_count=1),
                average_precision_baseline=0.5,
                bootstrap=False,
            )
        with self.assertRaisesRegex(core.QualityProbeCoreV94Error, "prevalence"):
            core._evaluate_family(
                train={"model_visible_14": train},
                development={"model_visible_14": development},
                train_labels=labels,
                development_labels=labels,
                train_label_row_keys=train.row_keys,
                development_label_row_keys=development.row_keys,
                policy=policy(world_count=1),
                average_precision_baseline=0.25,
                bootstrap=False,
            )

    def test_formal_bootstrap_draw_commitment_is_unchanged(self) -> None:
        draws = core.generate_bootstrap_draws(
            replicates=9999,
            world_count=500,
            seed=281320260810,
        )
        digest = __import__("hashlib").sha256(draws.tobytes(order="C")).hexdigest()
        self.assertEqual(
            digest,
            "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661",
        )


if __name__ == "__main__":
    unittest.main()
