from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_same_budget_cpu_controls_v1 as budget


class SameBudgetCpuControlsV1Tests(unittest.TestCase):
    def test_contract_validation_does_not_read_truth(self) -> None:
        result = budget.validate_contract()
        self.assertEqual(result["audit_a_truth_reads"], 0)
        self.assertEqual(result["audit_b_truth_reads"], 0)

    def test_world_mask_selects_complete_k28_worlds(self) -> None:
        rows = ["w0"] * 378 + ["w1"] * 378 + ["w2"] * 378
        mask = budget._mask_for_worlds(rows, ("w0", "w2"))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(int(np.sum(mask)), 756)
        self.assertTrue(np.all(mask[:378]))
        self.assertFalse(np.any(mask[378:756]))
        self.assertTrue(np.all(mask[756:]))

    def test_output_does_not_overwrite_v3_or_v4(self) -> None:
        value = budget.OUTPUT_ROOT.as_posix()
        self.assertNotIn("train_development_v2_20260901", value)
        self.assertNotIn("transfer_claim_controls_v4_20260901", value)


if __name__ == "__main__":
    unittest.main()
