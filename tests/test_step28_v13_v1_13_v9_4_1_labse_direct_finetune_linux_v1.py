from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import step28_v13_v1_13_v9_4_1_labse_direct_finetune_linux_v1 as direct


class LabseDirectFinetuneLinuxV1Tests(unittest.TestCase):
    def test_contract_validation_does_not_authorize_or_read_formal_truth(self) -> None:
        result = direct.validate_contract()
        self.assertFalse(result["implementation_policy_embedded_authorization"])
        self.assertTrue(result["separate_execution_authorization_present"])
        self.assertTrue(result["formal_gpu_training_authorized"])
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)

    def test_formal_run_requires_separate_authorization_before_gpu_or_labels(self) -> None:
        policy = {"canonical_self_hash": "frozen-policy"}
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with mock.patch.object(direct, "AUTHORIZATION_PATH", missing), mock.patch.object(
                direct.common, "load_policy", return_value=policy
            ), mock.patch.object(direct, "require_gpu_runtime") as gpu, mock.patch.object(
                direct.controls, "_load_inputs"
            ) as supervision:
                with self.assertRaisesRegex(
                    direct.DirectFinetuneError, "implementation-only"
                ):
                    direct.run()
                gpu.assert_not_called()
                supervision.assert_not_called()

    def test_current_authorization_is_narrow_and_valid(self) -> None:
        policy = direct.common.load_policy()
        authorization = direct.load_execution_authorization(policy)
        self.assertTrue(authorization["formal_gpu_training_authorized"])
        self.assertFalse(authorization["audit_a_prediction_authorized"])
        self.assertFalse(authorization["audit_a_truth_authorized"])
        self.assertFalse(authorization["audit_b_prediction_authorized"])
        self.assertFalse(authorization["audit_b_truth_authorized"])

    def test_runtime_smoke_has_no_formal_supervision_loader(self) -> None:
        source = inspect.getsource(direct.smoke_runtime)
        self.assertNotIn("_load_inputs", source)
        self.assertNotIn("private_custody", source)
        self.assertIn('"formal_train_labels_read": 0', source)
        self.assertIn('"audit_a_truth_reads": 0', source)

    def test_complete_world_index_rejects_partial_worlds(self) -> None:
        complete = direct._world_row_indices(
            {"world_uids": np.asarray(["w0"] * 378, dtype=str)}
        )
        self.assertEqual(complete["w0"].shape, (378,))
        with self.assertRaisesRegex(direct.DirectFinetuneError, "complete K28"):
            direct._world_row_indices(
                {"world_uids": np.asarray(["w0"] * 377, dtype=str)}
            )

    def test_numeric_standardizer_is_train_derived_and_handles_constant_columns(self) -> None:
        mean, scale = direct._standardizer(
            np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float64)
        )
        np.testing.assert_array_equal(mean, np.asarray([1.0, 3.0]))
        np.testing.assert_array_equal(scale, np.asarray([1.0, 1.0]))

    def test_direct_models_share_semantics_and_differ_only_by_identity_block(self) -> None:
        policy = direct.common.load_policy()
        models = policy["models"]
        self.assertEqual(
            models["ft_base"]["semantic_features"],
            models["ft_joint"]["semantic_features"],
        )
        self.assertFalse(models["ft_base"]["identity_features"])
        self.assertEqual(
            models["ft_joint"]["identity_features"],
            "identity33_train_subset_transform",
        )


if __name__ == "__main__":
    unittest.main()
