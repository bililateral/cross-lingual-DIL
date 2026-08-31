from __future__ import annotations

import hashlib
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_confirmatory_evaluator_v3 as evaluator
import step28_v13_v1_13_v9_4_1_blind_stage_protocol_v3 as blind
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


def one_world_k28() -> tuple[list[str], list[str], np.ndarray]:
    sellers = [f"seller_{index:02d}" for index in range(28)]
    groups = []
    offset = 0
    for size in [2] * 8 + [3] * 4:
        groups.append(set(sellers[offset : offset + size]))
        offset += size
    left = []
    right = []
    relevance = []
    for index, first in enumerate(sellers):
        for second in sellers[index + 1 :]:
            left.append(first)
            right.append(second)
            relevance.append(int(any(first in group and second in group for group in groups)))
    return left, right, np.asarray(relevance, dtype=np.int8)


class ModelTrainingV3Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = core.load_policy()

    def test_policy_is_exact_and_all_formal_capabilities_are_false(self) -> None:
        self.assertEqual(
            self.policy["canonical_self_hash"],
            "5e26b6c5fd6fea068c5a766ddb94302248f4db4d3ec3ea8d180f60334fa8b4bb",
        )
        self.assertEqual(set(self.policy["authorization_state"].values()), {False})
        with self.assertRaises(core.ModelTrainingV3Error):
            core.require_formal_execution_authorization(self.policy)
        with self.assertRaises(core.ModelTrainingV3Error):
            evaluator.require_split_specific_formal_authorization()

    def test_model_metric_registry_and_blind_order_are_complete(self) -> None:
        self.assertEqual(tuple(self.policy["implementation_scope"]["formal_report_models"]), core.MODEL_IDS)
        metrics = self.policy["metric_contract"]
        self.assertIn("average_precision", metrics["pooled_classification_metrics"])
        self.assertIn("trapezoidal_pr_auc", metrics["pooled_classification_metrics"])
        self.assertIn("map", metrics["retrieval_metrics"])
        self.assertIn("mrr", metrics["retrieval_metrics"])
        order = self.policy["stage_order"]
        self.assertLess(
            order.index("current_regression_commit_push_independent_review"),
            order.index("one_time_public_projection_authorization_and_freeze"),
        )
        self.assertLess(order.index("freeze_audit_a_blind_predictions_without_truth"), order.index("one_time_audit_a_truth_authorization_and_evaluation"))
        self.assertLess(order.index("freeze_audit_b_blind_predictions_only_after_pinned_audit_a_pass"), order.index("one_time_audit_b_truth_authorization_and_evaluation"))

    def test_blind_builder_has_no_truth_interface_and_stage_order_is_one_way(self) -> None:
        blind.validate_truth_free_signature()
        blind.validate_next_stage(
            "V3_IMPLEMENTATION_FROZEN_NO_FORMAL_EXECUTION",
            "CURRENT_REGRESSION_COMMIT_PUSH_INDEPENDENT_REVIEW_FROZEN",
        )
        with self.assertRaises(core.ModelTrainingV3Error):
            blind.validate_next_stage(
                "V3_IMPLEMENTATION_FROZEN_NO_FORMAL_EXECUTION",
                "PUBLIC_PROJECTION_FROZEN_NO_TRUTH",
            )
        blind.validate_next_stage(
            "TRAIN_DEVELOPMENT_MODELS_AND_THRESHOLDS_FROZEN",
            "AUDIT_A_BLIND_PREDICTIONS_FROZEN_NO_TRUTH",
        )
        with self.assertRaises(core.ModelTrainingV3Error):
            blind.validate_next_stage(
                "TRAIN_DEVELOPMENT_MODELS_AND_THRESHOLDS_FROZEN",
                "AUDIT_A_EVALUATION_FROZEN_PASSED",
            )
        signature = inspect.signature(blind.build_blind_prediction_payload)
        for forbidden in ("labels", "truth", "qrels", "relevance", "controller", "membership"):
            self.assertNotIn(forbidden, signature.parameters)

    def test_blind_payload_hashes_probabilities_and_writes_no_formal_artifact(self) -> None:
        predictions = {
            model_id: np.asarray([0.1, 0.5, 0.9], dtype="<f8")
            for model_id in core.MODEL_IDS
        }
        thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
        payload = blind.build_blind_prediction_payload(
            split="audit_a",
            predictions=predictions,
            thresholds=thresholds,
            row_key_sha256="1" * 64,
            training_parent_sha256="2" * 64,
        )
        self.assertFalse(payload["audit_truth_read"])
        self.assertFalse(payload["formal_artifact_written"])
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual(payload["canonical_self_hash"], core.canonical_sha256({key: value for key, value in payload.items() if key != "canonical_self_hash"}))

    def test_formal_training_layout_rejects_nonformal_fixture_before_fit(self) -> None:
        with self.assertRaises(core.ModelTrainingV3Error):
            core._validate_formal_training_rows(
                np.asarray([1, 0], dtype=np.int8), ["world_fixture", "world_fixture"]
            )

    def test_ap_pr_auc_and_roc_use_equal_score_groups(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype="<f8")
        first = core.score_curve_metrics(labels, scores)
        permutation = np.asarray([0, 2, 1, 3])
        second = core.score_curve_metrics(labels[permutation], scores[permutation])
        self.assertEqual(first, second)
        self.assertNotEqual(first["average_precision"], first["trapezoidal_pr_auc"])
        self.assertGreaterEqual(first["roc_auc"], 0.0)
        self.assertLessEqual(first["roc_auc"], 1.0)

    def test_threshold_tie_uses_higher_numeric_threshold(self) -> None:
        labels = np.asarray([0, 0], dtype=np.int8)
        scores = np.asarray([0.2, 0.8], dtype="<f8")
        worlds = np.asarray([0, 0], dtype=np.int64)
        self.assertEqual(core.select_development_threshold(labels, scores, worlds), np.inf)

    def test_retrieval_metrics_require_and_use_complete_k28(self) -> None:
        left, right, relevance = one_world_k28()
        scores = np.where(relevance == 1, 0.9, 0.1).astype("<f8")
        report = core.retrieval_report(scores, np.zeros(378, dtype=np.int64), left, right, relevance)
        self.assertEqual(report["aggregate"]["map"], 1.0)
        self.assertEqual(report["aggregate"]["mrr"], 1.0)
        self.assertEqual(report["aggregate"]["recall_at_10"], 1.0)
        with self.assertRaises(core.ModelTrainingV3Error):
            core.retrieval_report(scores[:-1], np.zeros(377, dtype=np.int64), left[:-1], right[:-1], relevance[:-1])

    def test_m1_maps_are_replayed_from_world_and_endpoint_rows(self) -> None:
        left, right, _ = one_world_k28()
        world_uids = ["world_fixture"] * 378
        pair_to_index = {
            core.common_v1.canonical_pair_endpoints(a, b): index
            for index, (a, b) in enumerate(zip(left, right, strict=True))
        }
        sellers = sorted(set(left) | set(right), key=lambda value: value.encode("utf-8"))
        mappings = {}
        for repeat_id in core.M1_REPEAT_IDS:
            exact = core.common_v1.build_m1_mapping("world_fixture", sellers, repeat_id)
            mapping = np.empty(378, dtype=np.int64)
            for target, source in exact.items():
                mapping[pair_to_index[target]] = pair_to_index[source]
            mappings[f"m1_{repeat_id}"] = mapping
        observed = core._validate_m1_source_indices(mappings, world_uids, left, right)
        self.assertEqual(set(observed), set(core.M1_IDS))
        changed = {key: value.copy() for key, value in mappings.items()}
        changed["m1_r01"][[0, 1]] = changed["m1_r01"][[1, 0]]
        with self.assertRaises(core.ModelTrainingV3Error):
            core._validate_m1_source_indices(changed, world_uids, left, right)

    def test_bootstrap_pooled_ap_matches_literal_world_repetition(self) -> None:
        labels = np.asarray([1, 0, 0, 1] * 3, dtype=np.int8)
        scores = np.asarray(
            [0.9, 0.7, 0.4, 0.2, 0.8, 0.7, 0.3, 0.1, 0.95, 0.6, 0.4, 0.05],
            dtype="<f8",
        )
        worlds = np.repeat(np.arange(3), 4)
        indices = np.asarray([[0, 0, 2], [1, 2, 2], [0, 1, 2]], dtype="<i8")
        observed = core.bootstrap_pooled_score_metrics(labels, scores, worlds, indices, batch_size=2)
        expected = []
        for draw in indices:
            row_indices = np.concatenate([np.flatnonzero(worlds == world) for world in draw])
            expected.append(core.score_curve_metrics(labels[row_indices], scores[row_indices])["average_precision"])
        np.testing.assert_allclose(observed["average_precision"], expected, rtol=0.0, atol=1e-15)

    def test_actual_formal_bootstrap_matrix_is_rebuilt_and_rehashed(self) -> None:
        matrix = core.build_bootstrap_indices(self.policy, "development")
        self.assertEqual(matrix.shape, (9999, 500))
        self.assertEqual(
            hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
            "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661",
        )
        changed = matrix.copy()
        changed[0, 0] = (changed[0, 0] + 1) % 500
        with self.assertRaises(core.ModelTrainingV3Error):
            core.validate_bootstrap_indices(self.policy, "development", changed)

    def test_zero_beta_and_inactive_rows_exactly_preserve_p0(self) -> None:
        p0 = np.asarray([np.nextafter(0.0, 1.0), 0.2, 0.8, np.nextafter(1.0, 0.0)], dtype="<f8")
        identity = np.zeros((4, 33), dtype="<f8")
        identity[1, 0] = 1.0
        scale = np.ones(33, dtype="<f8")
        mu = np.zeros(33, dtype="<f8")
        artifact = {"scale": scale, "mu": mu, "beta": np.zeros(33, dtype="<f8")}
        observed = core.predict_residual_model(artifact, p0, identity)
        self.assertEqual(observed.tobytes(), p0.tobytes())

    def test_m1_m2_family_estimates_transform_only_from_correct_rows(self) -> None:
        rows_per_world = 6
        world_uids = [f"w{world:03d}" for world in range(500) for _ in range(rows_per_world)]
        row_count = len(world_uids)
        p0 = np.full(row_count, 0.4, dtype="<f8")
        labels = np.tile(np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8), 500)
        identity = np.zeros((row_count, 33), dtype="<f8")
        identity[:, 0] = np.arange(1, row_count + 1)
        mappings = {}
        for shift, model_id in enumerate(core.M1_IDS, start=1):
            mapping = np.empty(row_count, dtype=np.int64)
            for world in range(500):
                start = world * rows_per_world
                local = np.arange(start, start + rows_per_world)
                mapping[local] = np.roll(local, -shift)
            mappings[model_id] = mapping
        transform_inputs = []

        def fake_transform(matrix, worlds):
            transform_inputs.append(np.asarray(matrix)[:, 0].copy())
            return np.ones(33, dtype="<f8"), np.zeros(33, dtype="<f8")

        def fake_fit(base, model_identity, y, scale, mu, l2):
            return {
                "beta": np.zeros(33, dtype="<f8"),
                "scale": scale,
                "mu": mu,
            }

        def fake_predict(artifact, base, model_identity):
            return np.asarray(base, dtype="<f8").copy()

        with mock.patch.object(core.common_v1, "fit_identity_transform", side_effect=fake_transform), mock.patch.object(
            core, "fit_residual_with_frozen_transform", side_effect=fake_fit
        ), mock.patch.object(core, "predict_residual_model", side_effect=fake_predict):
            result = core.fit_m1_m2_family(
                p0,
                identity,
                labels,
                world_uids,
                mappings,
                enforce_runtime=False,
            )
        self.assertEqual(result["selected_l2"], 100.0)
        self.assertEqual(len(transform_inputs), 46)
        for observed in transform_inputs:
            self.assertTrue(np.all(np.diff(observed) > 0.0))

    def test_residual_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(7)
        p0 = rng.uniform(0.05, 0.95, 12).astype("<f8")
        phi = rng.normal(size=(12, 4)).astype("<f8")
        labels = rng.integers(0, 2, size=12, dtype=np.int8)
        beta = rng.normal(scale=0.1, size=4).astype("<f8")
        objective, gradient = core.residual_objective_and_gradient(beta, p0, phi, labels, 0.3)
        self.assertTrue(np.isfinite(objective))
        numerical = []
        epsilon = 1e-6
        for column in range(4):
            plus = beta.copy()
            minus = beta.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            plus_value = core.residual_objective_and_gradient(plus, p0, phi, labels, 0.3)[0]
            minus_value = core.residual_objective_and_gradient(minus, p0, phi, labels, 0.3)[0]
            numerical.append((plus_value - minus_value) / (2 * epsilon))
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)

    def test_m3_exact_grid_tie_selects_first_entry(self) -> None:
        class DummyModel:
            def fit(self, matrix, labels):
                return self

            def predict_proba(self, matrix):
                probability = np.full(len(matrix), 0.25, dtype="<f8")
                return np.column_stack((1.0 - probability, probability))

        worlds = [f"w{index:03d}" for index in range(500) for _ in range(2)]
        matrix = np.arange(3000, dtype="<f8").reshape(1000, 3)
        labels = np.tile(np.asarray([0, 1], dtype=np.int8), 500)
        with mock.patch.object(core, "_lightgbm_classifier", return_value=DummyModel()):
            result = core.fit_m3_one(matrix, labels, worlds, enforce_runtime=False)
        self.assertEqual(result["selected_grid_index"], 0)
        self.assertEqual(result["selected_grid"], core.M3_GRID[0])

    def test_evaluator_accepts_raw_inputs_only_and_reports_all_metrics(self) -> None:
        left, right, relevance = one_world_k28()
        base_scores = np.linspace(0.01, 0.99, 378, dtype="<f8")
        predictions = {
            model_id: np.ascontiguousarray(base_scores + index * 1e-8, dtype="<f8")
            for index, model_id in enumerate(core.MODEL_IDS)
        }
        thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
        result = evaluator.evaluate_split_from_raw_inputs(
            policy=self.policy,
            split="development",
            predictions=predictions,
            thresholds=thresholds,
            world_ordinals=np.zeros(378, dtype=np.int64),
            seller_uid_left=left,
            seller_uid_right=right,
            labels=relevance,
            retrieval_relevance=relevance,
            actual_bootstrap_indices=np.zeros((5, 1), dtype="<i8"),
            enforce_formal_layout=False,
        )
        self.assertEqual(set(result["model_points"]), set(core.MODEL_IDS))
        m0 = result["model_points"]["m0"]
        self.assertIn("average_precision", m0["pooled"])
        self.assertIn("trapezoidal_pr_auc", m0["pooled"])
        self.assertIn("map", m0["retrieval"])
        self.assertIn("mrr", m0["retrieval"])
        signature = inspect.signature(evaluator.evaluate_split_from_raw_inputs)
        self.assertNotIn("bootstrap_ap", signature.parameters)
        self.assertNotIn("precomputed_metric_series", signature.parameters)
        self.assertFalse(result["formal_conclusion_authorized"])
        self.assertEqual(
            result["gate"]["status"],
            "PASSED_DEVELOPMENT_M1_M0_EQUIVALENCE_GATE",
        )
        self.assertFalse(result["gate"]["audit_truth_opened_or_authorized_by_this_result"])

    def test_evaluator_rejects_qrels_that_disagree_with_labels(self) -> None:
        left, right, relevance = one_world_k28()
        predictions = {model_id: np.full(378, 0.5, dtype="<f8") for model_id in core.MODEL_IDS}
        thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
        changed = relevance.copy()
        changed[0] = 1 - changed[0]
        with self.assertRaises(core.ModelTrainingV3Error):
            evaluator.evaluate_split_from_raw_inputs(
                policy=self.policy,
                split="development",
                predictions=predictions,
                thresholds=thresholds,
                world_ordinals=np.zeros(378, dtype=np.int64),
                seller_uid_left=left,
                seller_uid_right=right,
                labels=relevance,
                retrieval_relevance=changed,
                actual_bootstrap_indices=np.zeros((2, 1), dtype="<i8"),
                enforce_formal_layout=False,
            )

    def test_formal_audit_requires_exact_frozen_development_thresholds(self) -> None:
        predecessor = core.common_v1.load_policy()
        thresholds = {model_id: 0.5 for model_id in core.MODEL_IDS}
        thresholds["m0"] = float(predecessor["frozen_models"]["m0"]["threshold"])
        thresholds["c0"] = float(predecessor["frozen_models"]["c0"]["threshold"])
        predictions = {
            model_id: np.asarray([0.25, 0.75], dtype="<f8")
            for model_id in core.MODEL_IDS
        }
        kwargs = {
            "policy": self.policy,
            "split": "audit_a",
            "predictions": predictions,
            "thresholds": thresholds,
            "world_ordinals": np.asarray([0, 0], dtype=np.int64),
            "seller_uid_left": ["a", "a"],
            "seller_uid_right": ["b", "c"],
            "labels": np.asarray([0, 1], dtype=np.int8),
            "retrieval_relevance": np.asarray([0, 1], dtype=np.int8),
            "actual_bootstrap_indices": np.zeros((1, 1), dtype="<i8"),
        }
        with mock.patch.object(core, "validate_formal_split_layout"), mock.patch.object(
            core, "validate_bootstrap_indices", return_value=np.zeros((1, 1), dtype="<i8")
        ):
            with self.assertRaisesRegex(
                core.ModelTrainingV3Error,
                "complete frozen development threshold registry",
            ):
                evaluator.evaluate_split_from_raw_inputs(**kwargs)
            frozen = dict(thresholds)
            frozen["m2"] = 0.6
            with self.assertRaisesRegex(
                core.ModelTrainingV3Error,
                "Audit threshold differs from frozen development threshold: m2",
            ):
                evaluator.evaluate_split_from_raw_inputs(
                    **kwargs,
                    frozen_development_thresholds=frozen,
                )

    def test_simultaneous_confirmatory_gate_uses_six_named_comparisons(self) -> None:
        def point(ap):
            return {
                "pooled": {"average_precision": ap},
                "threshold": {"world_equal_confusion": {}},
                "retrieval": {},
            }

        model_points = {"m0": point(0.10), "m2": point(0.20)}
        model_series = {
            "m0": {"average_precision": np.full(20, 0.10, dtype="<f8")},
            "m2": {"average_precision": np.full(20, 0.20, dtype="<f8")},
        }
        for model_id in core.M1_IDS:
            model_points[model_id] = point(0.10)
            model_series[model_id] = {
                "average_precision": np.full(20, 0.10, dtype="<f8")
            }
        result = evaluator._confirmatory_gate("audit_a", model_series, model_points)
        self.assertEqual(
            result["comparison_order"],
            [
                "m2_minus_mean_five_individual_m1",
                "m2_minus_m1_r01",
                "m2_minus_m1_r02",
                "m2_minus_m1_r03",
                "m2_minus_m1_r04",
                "m2_minus_m1_r05",
            ],
        )
        self.assertTrue(result["numerical_gate_passed"])
        np.testing.assert_allclose(result["simultaneous_lower_bounds"], 0.10)
        self.assertFalse(result["formal_conclusion_authorized"])


if __name__ == "__main__":
    unittest.main()
