"""Independent boundary, metric and actual tree-update checks for the specialist."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_specialist as subject


class SpecialistContracts(unittest.TestCase):
    def test_public_mask_is_union_of_two_strong_features(self):
        names = ["other", "verified_direct_token_count_log1p", "strong_rotation_path_count_log1p"]
        values = np.array([[99., 0, 0], [0, 1, 0], [0, 0, 2], [0, 1, 2]])
        np.testing.assert_array_equal(subject.no_strong_mask(values, names), [True, False, False, False])
        with self.assertRaises(ValueError):
            subject.no_strong_mask(np.array([[np.nan, 0, 0]]), names)

    def test_shared_medians_do_not_use_development(self):
        train = np.array([[1., np.nan], [3., 10.], [np.nan, 20.], [9., 30.]])
        dev = np.array([[np.nan, np.nan], [100000., -100000.]])
        a, b, medians = subject.shared_imputation(train, dev)
        np.testing.assert_array_equal(medians, [3., 20.])
        np.testing.assert_array_equal(a[2], [3., 20.])
        np.testing.assert_array_equal(b[0], medians)
        self.assertTrue(np.isnan(train[0, 1]))
        with self.assertRaises(ValueError):
            subject.shared_imputation(np.array([[np.nan]]), np.array([[1.]]))

    def test_label_alignment_and_split_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"split": "train", "world_ordinal": "0", "world_uid": "w",
                     "canonical_pair_uid": "a||b", "seller_uid_left": "a", "seller_uid_right": "b"},
                    {"split": "train", "world_ordinal": "0", "world_uid": "w",
                     "canonical_pair_uid": "a||c", "seller_uid_left": "a", "seller_uid_right": "c"}]
            rp, lp = root / "rows.csv", root / "labels.csv"
            def write(path, fields, values):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader(); writer.writerows(values)
            write(rp, subject.ROW_FIELDS, rows)
            read = subject.read_rows(rp, "train")
            truth = [{"canonical_pair_uid": r["canonical_pair_uid"], "world_uid": "w", "label": str(i)}
                     for i, r in enumerate(rows)]
            fields = ["canonical_pair_uid", "world_uid", "label"]
            write(lp, fields, truth)
            np.testing.assert_array_equal(subject.read_labels(lp, read), [0, 1])
            for bad in (list(reversed(truth)), truth[:1],
                        [{**truth[0], "label": "2"}, truth[1]]):
                write(lp, fields, bad)
                with self.assertRaises(ValueError):
                    subject.read_labels(lp, read)
            write(rp, subject.ROW_FIELDS, rows + rows[:1])
            with self.assertRaises(ValueError):
                subject.read_rows(rp, "train")
            for split in ("audit_a", "audit_b", "test"):
                with self.assertRaises(ValueError):
                    subject.read_rows(root / "must_not_be_opened.csv", split)

    def test_account_and_world_split_isolation(self):
        a = [{"world_uid": "w1", "seller_uid_left": "a", "seller_uid_right": "b"}]
        b = [{"world_uid": "w2", "seller_uid_left": "c", "seller_uid_right": "d"}]
        subject.disjoint_splits(a, b)
        for bad in ({**b[0], "world_uid": "w1"}, {**b[0], "seller_uid_left": "a"}):
            with self.assertRaises(ValueError):
                subject.disjoint_splits(a, [bad])

    def test_confusion_and_tied_operating_point(self):
        c = subject.confusion(np.array([1, 0, 1, 0]), np.array([True, True, False, False]))
        for key in ("precision", "recall", "f1", "specificity", "balanced_accuracy"):
            self.assertEqual(c[key], .5)
        self.assertEqual(c["mcc"], 0)
        y = np.array([1, 0, 0, 1, 0, 0])
        raw = np.array([.9, .8, .8, .7, .6, .1])
        # One of four negatives is affordable, but the two equal .8 negatives cannot be split.
        p = subject.empirical_operating_point(y, raw, .25)
        self.assertEqual((p["tp"], p["fp"], p["recall"], p["threshold"]), (1, 0, .5, .9))
        constant = subject.empirical_operating_point(np.array([0, 1]), np.ones(2), .01)
        self.assertTrue(constant["always_negative"])
        self.assertEqual(constant["recall"], 0)

    def test_exact_paired_world_bootstrap(self):
        delta = np.array([.1, .3])
        draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        r = subject.paired_summary(delta, draws)
        self.assertAlmostEqual(r["difference"], .2)
        np.testing.assert_allclose(r["percentile_95_interval"], [.1075, .2925], rtol=0, atol=1e-15)
        same = subject.paired_summary(np.array([.2, .2]), draws)
        np.testing.assert_allclose(same["percentile_95_interval"], [.2, .2])
        with self.assertRaises(ValueError):
            subject.paired_summary(delta, np.array([[0, 2]]))

    def test_retrieval_denominators_and_uid_ties(self):
        r = subject.query_ranking(np.array([0, 1, 1]))
        self.assertAlmostEqual(r["map"], 7 / 12)
        self.assertEqual(r["mrr"], .5)
        self.assertEqual(r["recall_at_1"], 0)
        self.assertEqual(r["recall_at_3"], 1)
        self.assertAlmostEqual(r["ndcg_at_3"], (1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))
        rows = [{"world_uid": "w", "seller_uid_left": a, "seller_uid_right": b}
                for a, b in [("a", "b"), ("a", "c"), ("b", "c")]]
        # Only a-b is relevant. Equal scores sort candidate UID ascending: both eligible queries rank it first.
        out = subject.retrieval(rows, np.array([1, 0, 0]), np.ones(3), np.ones(3, dtype=bool))
        self.assertEqual(out["eligible_queries"], 2)
        self.assertEqual(out["queries_with_candidates"], 3)
        self.assertEqual(out["world_equal"]["map"], 1)

    def test_actual_newton_tree_restriction_and_disk_replay(self):
        parameters = dict(objective="binary", n_estimators=1, num_leaves=2, max_depth=1,
                          learning_rate=.2, min_child_samples=1, min_child_weight=0,
                          reg_lambda=1., reg_alpha=0., boost_from_average=False,
                          verbosity=-1, deterministic=True, force_col_wise=True, num_threads=1)
        x = np.array([[0.]] * 4 + [[1.]] * 4 + [[2.], [3.]])
        y = np.array([0] * 4 + [1] * 4 + [0, 1], dtype=np.int8)
        selected = np.array([True] * 8 + [False] * 2)
        model = subject.fit_model(x, y, selected, parameters)
        scores = subject.predictions(model, x)
        # At p=.5, each leaf has sum Hessian=1 and |sum gradient|=2:
        # learning_rate * (-sum gradient)/(sum Hessian + lambda) = +/- .2.
        np.testing.assert_allclose(scores[:8, 0], [-.2] * 4 + [.2] * 4, rtol=0, atol=1e-14)
        np.testing.assert_allclose(scores[:8, 1], 1 / (1 + np.exp(-scores[:8, 0])), atol=1e-15)
        altered = y.copy(); altered[~selected] = 1 - altered[~selected]
        other = subject.fit_model(x, altered, selected, parameters)
        np.testing.assert_array_equal(subject.predictions(other, x), scores)
        self.assertLess(subject.probability_loss(y[selected], scores[selected, 1]), math.log(2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.txt"
            path.write_text(model.model_to_string(), encoding="utf-8")
            loaded = lgb.Booster(model_file=str(path))
            order = np.array([7, 0, 5, 2])
            np.testing.assert_array_equal(subject.predictions(loaded, x[order]), scores[order])


if __name__ == "__main__":
    unittest.main()
