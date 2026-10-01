"""Hand-computed curves and supervision-alignment failure cases for the local audit."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_saved_model_audit as audit


class SavedModelAuditContracts(unittest.TestCase):
    def test_tied_scores_match_hand_computation(self):
        point = audit.curve_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.5, 0.5, 0.1]))
        self.assertAlmostEqual(point["ap"], 5 / 6)
        self.assertAlmostEqual(point["trapezoidal_pr_auc"], 11 / 12)
        self.assertAlmostEqual(point["roc_auc"], 7 / 8)

    def test_all_scores_tied_do_not_use_input_order(self):
        for labels in ([1, 1, 0, 0], [0, 1, 0, 1]):
            point = audit.curve_metrics(np.array(labels), np.ones(4))
            self.assertEqual(point, {"ap": 0.5, "trapezoidal_pr_auc": 0.75, "roc_auc": 0.5})

    def test_single_class_and_empty_curves_are_explicitly_undefined(self):
        for labels in ([], [0, 0], [1, 1]):
            self.assertTrue(all(value is None for value in audit.curve_metrics(
                np.array(labels), np.ones(len(labels))).values()))

    def test_invalid_metric_inputs_fail(self):
        for labels, scores in (([1, 0], [0.1]), ([1, 0], [np.nan, 0.1]), ([2, 0], [0.1, 0.2])):
            with self.assertRaises(ValueError):
                audit.curve_metrics(np.array(labels), np.array(scores))

    def test_conditional_world_denominator_is_reported(self):
        point = audit.summarize_subset(np.array([1, 0, 0, 0]), np.array([0.8, 0.2, 0.6, 0.3]),
                                       [np.array([0, 1]), np.array([2, 3])], np.ones(4, dtype=bool))
        self.assertEqual(point["eligible_worlds"], 1)
        self.assertEqual(point["total_worlds"], 2)
        self.assertEqual(point["world_equal_both_classes_only"]["ap"], 1)
        self.assertEqual(point["negatives"], 3)

    def read_fixture(self, split="development", swap=False, extra=False, duplicate=False):
        with tempfile.TemporaryDirectory() as directory:
            rows, labels = Path(directory) / "rows.csv", Path(directory) / "labels.csv"
            keys = ["a||b", "a||c"] if not duplicate else ["a||b", "a||b"]
            with rows.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["split", "world_ordinal", "world_uid", "canonical_pair_uid",
                                 "seller_uid_left", "seller_uid_right"])
                for key in keys:
                    writer.writerow([split, 0, "world", key, *key.split("||")])
            with labels.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["canonical_pair_uid", "world_uid", "label"])
                for index, key in enumerate(reversed(keys) if swap else keys):
                    writer.writerow([key, "world", index])
                if extra:
                    writer.writerow(["d||e", "world", 0])
            return audit.aligned_development_labels(rows, labels)

    def test_correct_alignment(self):
        labels, worlds = self.read_fixture()
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(worlds, ["world", "world"])

    def test_audit_split_rows_rejected(self):
        for split in ("audit_a", "audit_b", "train"):
            with self.assertRaises(ValueError):
                self.read_fixture(split=split)

    def test_label_order_and_count_mismatch_rejected(self):
        for kwargs in ({"swap": True}, {"extra": True}, {"duplicate": True}):
            with self.assertRaises(ValueError):
                self.read_fixture(**kwargs)


if __name__ == "__main__":
    unittest.main()
