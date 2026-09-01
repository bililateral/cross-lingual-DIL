from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_transfer_claim_controls_v4 as controls


class TransferClaimControlsV4Tests(unittest.TestCase):
    def test_policy_freezes_the_scientific_boundary(self) -> None:
        policy = controls.load_policy()
        self.assertEqual(
            policy["canonical_self_hash"], controls.canonical_self_hash(policy)
        )
        self.assertFalse(
            policy["truth_boundary"]["audit_a_labels_or_qrels_allowed"]
        )
        self.assertFalse(
            policy["truth_boundary"]["audit_b_labels_or_qrels_allowed"]
        )
        self.assertEqual(policy["fit"]["l2"], 0.01)
        self.assertFalse(policy["fit"]["intercept_penalized"])
        self.assertEqual(
            policy["primary_development_diagnostic"]["estimand"],
            "pooled_average_precision_t1_minus_i0",
        )

    def test_objective_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(20260901)
        phi = rng.normal(size=(37, 4)).astype("<f8")
        labels = (rng.random(37) < 0.27).astype(np.int8)
        offset = rng.normal(scale=0.2, size=37).astype("<f8")
        theta = rng.normal(scale=0.1, size=5).astype("<f8")
        objective, gradient = controls.objective_and_gradient(
            theta, phi, labels, 0.03, offset
        )
        self.assertTrue(math.isfinite(objective))
        epsilon = 1e-6
        numerical = np.empty_like(theta)
        for index in range(len(theta)):
            left = theta.copy()
            right = theta.copy()
            left[index] -= epsilon
            right[index] += epsilon
            left_value = controls.objective_and_gradient(
                left, phi, labels, 0.03, offset
            )[0]
            right_value = controls.objective_and_gradient(
                right, phi, labels, 0.03, offset
            )[0]
            numerical[index] = (right_value - left_value) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)

    def test_identity_only_fits_the_class_prevalence_intercept(self) -> None:
        phi = np.zeros((40, 3), dtype="<f8")
        labels = np.asarray([1] * 8 + [0] * 32, dtype=np.int8)
        artifact = controls.fit_control(phi, labels, 0.01, offset=None)
        probabilities = controls.predict_control(artifact, phi, offset=None)
        np.testing.assert_allclose(probabilities, 0.2, rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(artifact["theta"][1:], 0.0, rtol=0.0, atol=1e-12)

    def test_source_on_and_source_off_have_identical_trainable_dimension(self) -> None:
        phi = np.asarray(
            [[-1.0, 0.5], [0.0, -0.5], [1.0, 0.5], [2.0, -0.5]],
            dtype="<f8",
        )
        theta = np.asarray([-2.0, 0.1, -0.2], dtype="<f8")
        artifact = {"theta": theta}
        off = controls.predict_control(artifact, phi, offset=None)
        offset = np.asarray([-0.4, -0.2, 0.2, 0.4], dtype="<f8")
        on = controls.predict_control(artifact, phi, offset=offset)
        self.assertEqual(theta.shape, (phi.shape[1] + 1,))
        self.assertFalse(np.array_equal(off, on))

    def test_existing_formal_output_is_not_the_v4_output(self) -> None:
        policy = controls.load_policy()
        self.assertNotEqual(
            policy["output_root"],
            "reports/step28_model_experiment/v9_4_1_train_development_v2_20260901",
        )
        source = Path(controls.__file__).read_text(encoding="utf-8")
        self.assertNotIn("audit_a/pair_labels.csv", source)
        self.assertNotIn("audit_b/pair_labels.csv", source)

    def test_summary_uses_python_boolean_literals(self) -> None:
        source = Path(controls.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"audit_predictions_created": false', source)


if __name__ == "__main__":
    unittest.main()
