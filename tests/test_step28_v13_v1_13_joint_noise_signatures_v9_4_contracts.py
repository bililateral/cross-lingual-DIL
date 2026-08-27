from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_joint_noise_signatures_v9_4 as signatures_v94


class JointNoiseSignaturesV94Contracts(unittest.TestCase):
    def test_signature_padding_truncation_and_presence_masks_are_exact(self) -> None:
        singleton = signatures_v94._signature([(2, True, True)])
        self.assertEqual(singleton, {
            "item_count": 2,
            "title_present_mask": "11",
            "description_present_mask": "11",
            "joint_empty_mask": "00",
        })
        long = signatures_v94._signature([
            (index, index % 2 == 0, index % 3 == 0) for index in range(2, 14)
        ])
        self.assertEqual(long["item_count"], 8)
        self.assertEqual(len(long["joint_empty_mask"]), 8)

    def test_largest_remainder_is_exactly_28_slots(self) -> None:
        signature = {
            "item_count": 2,
            "title_present_mask": "11",
            "description_present_mask": "11",
            "joint_empty_mask": "00",
        }
        key = signatures_v94._signature_key(signature)
        slots = signatures_v94._largest_remainder(
            Counter({key: 648}),
            {key: signature},
        )
        self.assertEqual(len(slots), 28)
        self.assertTrue(all(slot == signature for slot in slots))

    def test_contract_is_label_and_model_free(self) -> None:
        contract = signatures_v94.contract_payload()
        self.assertFalse(contract["reads_pair_truth"])
        self.assertFalse(contract["reads_model_output"])
        self.assertEqual(contract["slot_count"], 28)


if __name__ == "__main__":
    unittest.main()
