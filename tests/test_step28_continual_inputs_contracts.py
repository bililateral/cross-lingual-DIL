"""Scientific alignment and arrival boundaries for the continual input audit."""
from __future__ import annotations

import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_inputs as inputs


class ContinualInputContracts(unittest.TestCase):
    def world(self):
        sellers = [f"seller_{i:02d}" for i in range(28)]
        world = {"world_uid": "world", "split": "train", "world_ordinal": 0,
                 "seller_count": 28, "item_count": 99, "pair_count": 378}
        rows = [dict(zip(inputs.ROW_FIELDS, ("train", "0", "world", f"{a}||{b}", a, b)))
                for a, b in combinations(sellers, 2)]
        return sellers, world, rows

    def test_complete_world_and_duplicate_edge_rejection(self):
        sellers, world, rows = self.world()
        inputs.validate_world_rows(rows, world, sellers, "train", 0)
        rows[-1] = rows[0]
        with self.assertRaises(ValueError):
            inputs.validate_world_rows(rows, world, sellers, "train", 0)

    def test_same_count_wrong_split_or_world_rejected(self):
        for field, value in (("split", "development"), ("world_uid", "future_world"),
                             ("world_ordinal", "1"), ("canonical_pair_uid", "wrong")):
            with self.subTest(field=field):
                sellers, world, rows = self.world()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    inputs.validate_world_rows(rows, world, sellers, "train", 0)

    def test_arrivals_are_complete_disjoint_replayable(self):
        first, second = inputs.random_arrivals(), inputs.random_arrivals()
        self.assertEqual(first, second)
        self.assertFalse(first["distinct_domains_claimed"])
        for order in first["orders"]:
            stages = order["train_world_ordinals_by_stage"]
            self.assertEqual([len(stage) for stage in stages], [100] * 5)
            flat = sum(stages, [])
            self.assertEqual(sorted(flat), list(range(500)))
        self.assertEqual(len({tuple(sum(o["train_world_ordinals_by_stage"], []))
                              for o in first["orders"]}), 3)

    def test_no_partial_world_arrival(self):
        with self.assertRaises(ValueError):
            inputs.random_arrivals(501, 5)

    def test_descriptive_variance_fraction_against_hand_calculation(self):
        values = np.asarray([[0., 3.], [2., 3.], [4., 3.], [6., 3.]])
        # Total SS=20; two equally sized group means 1 and 5 give between SS=16.
        np.testing.assert_allclose(inputs.between_group_fraction(values, ["a", "a", "b", "b"]),
                                   [.8, 0.], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
