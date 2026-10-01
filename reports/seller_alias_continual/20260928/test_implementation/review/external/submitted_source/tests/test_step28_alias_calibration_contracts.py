"""Hand-checkable probability, ranking, supervision and saved-output contracts."""
from __future__ import annotations

import copy
import csv
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_alias_calibration as method
import step28_alias_calibration_run as runner


def handmade_truth() -> np.ndarray:
    controllers = [c for c in range(8) for _ in range(2)] + [c for c in range(8, 12) for _ in range(3)]
    return np.array([int(controllers[i] == controllers[j]) for i, j in itertools.combinations(range(28), 2)], dtype=np.uint8)


def handwritten_scores(groups: int, offset: int) -> np.ndarray:
    # Continuous, overlapping scores; no fitted or reverse-engineered formal truth.
    rng = np.random.default_rng(1700 + offset)
    return (rng.normal(-2.4, 1.1, (groups, 378)) + .7 * handmade_truth()).astype(np.float32)


def handmade_bundle() -> tuple[dict, dict, np.ndarray]:
    partition = {"calibration": [{"group_uid": f"cal{i}", "domain": "ABC"[i // 12]} for i in range(36)],
                 "development": [{"group_uid": f"val{i}", "domain": "ABC"[i % 3]} for i in range(60)]}
    saved = {}
    for index, run_id in enumerate(method.RUNS):
        score = {"calibration": handwritten_scores(36, index), "development": handwritten_scores(60, 20 + index)}
        matrices = {role: method.metrics.group_metrics(np.tile(handmade_truth(), (len(v), 1)), v)[0]
                    for role, v in score.items()}
        saved[run_id] = {"scores": score, "old_metrics": matrices, "threshold": 0.2,
                         "origin": {"handmade_model": run_id, "formal_model_loaded": False}}
    return saved, partition, np.tile(handmade_truth(), (36, 1))


def write_pairs(root: Path, duplicate: bool = False) -> tuple[Path, dict]:
    path = root / "pairs.csv"
    rows = []
    for uid in ("g0", "g1"):
        sellers = [f"{uid}s{i:02}" for i in range(28)]
        rows.extend({"group_uid": uid, "seller_uid_left": a, "seller_uid_right": b, "label": str(y)}
                    for (a, b), y in zip(itertools.combinations(sellers, 2), handmade_truth(), strict=True))
    if duplicate:
        rows[-1] = dict(rows[-2])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_uid", "seller_uid_left", "seller_uid_right", "label"])
        writer.writeheader()
        writer.writerows(reversed(rows))
    return path, method.data.record(path, root)


def passing_summary() -> dict:
    return {"metrics": {name: {"mean": .02 if name in ("map", "recall_at_5") else 0.,
                                "per_seed": [.02] * 3 if name in ("map", "recall_at_5") else [0.] * 3,
                                "conditional_95pct_interval": [.01, .03]}
                        for name in method.metrics.COLUMNS}}


class CalibrationContracts(unittest.TestCase):
    pipeline_evidence: dict = {}

    def test_scalar_nll_gradient_and_finite_difference(self):
        x = np.array([-2.0, -0.4, .2, 1.7, 2.9])
        y = np.array([0., 0., 1., 0., 0.])
        theta = np.array([.8, -.6])
        actual, grad = method.loss_gradient(theta, x, y)
        values = [theta[0] * value + theta[1] for value in x]
        expected = sum(math.log1p(math.exp(z)) - label * z for z, label in zip(values, y)) / 5
        independent = np.array([sum((1 / (1 + math.exp(-z)) - label) * value
                                   for z, label, value in zip(values, y, x)) / 5,
                                sum(1 / (1 + math.exp(-z)) - label for z, label in zip(values, y)) / 5])
        self.assertAlmostEqual(actual, expected, places=14)
        np.testing.assert_allclose(grad, independent, atol=1e-14, rtol=0)
        for i in range(2):
            direction = np.eye(2)[i] * 1e-6
            numeric = (method.loss_gradient(theta + direction, x, y)[0]
                       - method.loss_gradient(theta - direction, x, y)[0]) / 2e-6
            self.assertAlmostEqual(grad[i], numeric, places=8)

    def test_known_optimum_real_optimizer(self):
        target_logits = np.repeat([-math.log(3), math.log(3)], 4)
        raw = target_logits / 2 - .7
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        result = method.fit(raw, y, role="calibration")
        self.assertEqual(result["status"], "PASS_CALIBRATION_FIT")
        np.testing.assert_allclose([result["a"], result["b"]], [2., 1.4], atol=2e-6, rtol=0)
        self.assertLess(result["final_nll"], result["initial_nll"])

    def test_identity_is_feasible_and_preserved_at_known_optimum(self):
        x = np.repeat([-math.log(3), math.log(3)], 4)
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        result = method.fit(x, y, role="calibration")
        self.assertEqual(result["status"], "PASS_CALIBRATION_FIT")
        np.testing.assert_allclose([result["a"], result["b"]], [1., 0.], atol=1e-12, rtol=0)

    def test_negative_association_reaches_positive_lower_bound(self):
        x = np.repeat([-1., 1.], 4)
        y = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        result = method.fit(x, y, role="calibration")
        self.assertEqual(result["status"], "PASS_CALIBRATION_FIT")
        self.assertAlmostEqual(result["a"], .001, places=12)
        self.assertLessEqual(result["projected_gradient_max"], 1e-6)

    def test_nonfinite_inputs_and_wrong_roles_rejected(self):
        for role in ("fit", "development", "heldout", "owners"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                method.fit(np.array([-1., 1.]), np.array([0, 1]), role=role)
        for x, y in (([np.nan, 1.], [0, 1]), ([0., 1.], [0, 2]), ([0., 1.], [1, 1]), ([0.], [0, 1])):
            with self.assertRaises(ValueError):
                method.fit(np.array(x), np.array(y), role="calibration")

    def test_positive_order_and_numerical_tie_collapse(self):
        raw = np.array([[2., -1., 2., .1], [.1, .1, -.2, -9.]])
        changed = method.transform(raw, {"a": .3, "b": -.7})
        self.assertTrue(method.preserve_order(raw, changed)["exact_order_and_ties_preserved"])
        with self.assertRaises(ValueError):
            method.transform(raw, {"a": -.2, "b": 0.})
        with self.assertRaises(ValueError):
            method.preserve_order(np.array([[0., 1e-300]]), np.array([[100., 100.]]))

    def test_probability_changes_while_all_ranking_metrics_stay(self):
        y = handmade_truth()[None, :]
        raw = np.where(y, .2, -.8)
        changed = method.transform(raw, {"a": 2., "b": -1.})
        left, before = method.metrics.group_metrics(y, raw)
        right, after = method.metrics.group_metrics(y, changed)
        for name in (*method.metrics.CURVE_KEYS, *method.metrics.RETRIEVAL_KEYS):
            self.assertEqual(left[0, method.metrics.COLUMNS.index(name)], right[0, method.metrics.COLUMNS.index(name)])
        self.assertEqual(before[0]["tp"], 20)
        self.assertEqual(after[0]["tp"], 0)
        self.assertNotEqual(left[0, 4], right[0, 4])

    def test_unclipped_fit_loss_extreme_logits_is_finite(self):
        x, y = np.array([-1000., 1000.]), np.array([1., 0.])
        value, gradient = method.loss_gradient(np.array([1., 0.]), x, y)
        self.assertEqual(value, 1000.)
        np.testing.assert_array_equal(gradient, [1000., 0.])

    def test_csv_identity_alignment_uses_only_selected_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, record = write_pairs(root)
            values, identities = runner.read_labels(path, record, {"g0", "g1"}, ["g1"], split="train")
            np.testing.assert_array_equal(values, handmade_truth()[None, :])
            self.assertEqual(set(identities["selected_sellers"]), {"g1"})
            self.assertEqual(identities["parsed_groups"], 2)
            self.assertEqual(identities["fitted_groups"], 1)

    def test_duplicate_pairs_or_heldout_role_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, record = write_pairs(root, duplicate=True)
            with self.assertRaises(ValueError):
                runner.read_labels(path, record, {"g0", "g1"}, ["g1"], split="train")
            with self.assertRaises(ValueError):
                runner.read_labels(path, record, {"g0", "g1"}, ["g1"], split="heldout")

    def test_failed_parse_consumes_once_and_cannot_reenter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset").mkdir()
            method.data.write_json(root / "dataset/manifest.json", {"files": {"train/supervision/pairs.csv": {}}})
            config = {"data_root": "dataset", "data_manifest_sha256": method.data.sha256(root / "dataset/manifest.json")}
            part = {"calibration": [{"group_uid": "g1"}], "fit": [{"group_uid": "g0"}]}
            with mock.patch.object(method.data, "ROOT", root), mock.patch.object(runner, "read_labels", side_effect=ValueError("injected")) as reader:
                with self.assertRaises(ValueError):
                    runner.parse_once(root, config, part, "train")
                with self.assertRaises(FileExistsError):
                    runner.parse_once(root, config, part, "train")
                self.assertEqual(reader.call_count, 1)
            self.assertEqual(method.data.read_json(root / "train_access.json")["parse_attempts"], 1)

    def test_seventeen_guards_and_raw_reference_protection(self):
        primary, raw = passing_summary(), passing_summary()
        self.assertEqual(len(method.acceptance(primary, raw)["checks"]), 17)
        self.assertTrue(method.acceptance(primary, raw)["passed"])
        mutations = [("map", "mean", .005, "map_minimum_observed_gain"),
                     ("map", "conditional_95pct_interval", [0., .03], "map_interval_above_zero"),
                     ("map", "per_seed", [.02, 0., .02], "map_improves_each_seed"),
                     ("recall_at_5", "mean", 0., "recall_at_5_mean_strictly_improves"),
                     ("recall_at_5", "per_seed", [0., .02, .02], "recall_at_5_s0_strictly_improves")]
        for name, sign in method.ranking.GUARDS.items():
            mutations.extend([(name, "mean", -sign * .001, name + "_mean_non_degradation"),
                              (name, "per_seed", [-sign * .001, 0., 0.], name + "_s0_non_degradation")])
        for name, field, value, failed in mutations:
            altered = copy.deepcopy(primary)
            altered["metrics"][name][field] = value
            self.assertEqual(method.acceptance(altered, raw)["failed"], [failed])
        for name in ("brier", "log_loss"):
            for field, value, suffix in (("mean", .001, "mean"), ("per_seed", [.001, 0., 0.], "s0")):
                altered = copy.deepcopy(raw)
                altered["metrics"][name][field] = value
                self.assertEqual(method.acceptance(primary, altered)["failed"], [name + "_" + suffix + "_against_raw_a"])

    def test_worsened_calibrated_reference_cannot_manufacture_success(self):
        primary, raw = passing_summary(), passing_summary()
        for name in ("brier", "log_loss"):
            primary["metrics"][name].update(mean=-.001, per_seed=[-.001] * 3)
            raw["metrics"][name].update(mean=.001, per_seed=[.001] * 3)
        result = method.acceptance(primary, raw)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failed"]), 4)

    def test_complete_real_six_fit_pipeline_reload_and_result_recovery(self):
        saved, partition, truth = handmade_bundle()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            fitted = runner.fit_all(out, saved, truth, partition)
            arrays = runner.restored_scores(out, saved, partition)
            # Byte corruption of an actual saved scalar map must fail before any valid parse.
            path = out / "s0_d/fit.json"
            original = path.read_bytes()
            path.write_bytes(original.replace(b'"a":', b'"x":', 1))
            with self.assertRaises(ValueError):
                runner.restored_scores(out, saved, partition)
            path.write_bytes(original)
            valid = np.tile(handmade_truth(), (60, 1))
            runner.collect(out, valid, arrays, saved, partition)
            with mock.patch.object(runner, "sources", return_value=[]):
                with self.assertRaises(ValueError):
                    runner.finalize(out)
            with mock.patch.object(method, "summarize", side_effect=RuntimeError("post-collection fault")):
                with self.assertRaises(RuntimeError):
                    runner.finalize(out)
            collected = method.data.read_json(out / "evaluation/collected.json")
            self.assertEqual(len(collected["points"]), 12)
            for entry in collected["points"].values():
                method.data.verify(out / "evaluation" / entry["file"]["path"], entry["file"])
            with mock.patch.object(runner, "parse_once", side_effect=AssertionError("must not reparse")), \
                 mock.patch.object(method, "fit", side_effect=AssertionError("must not refit")):
                result = runner.finalize(out)
            self.assertEqual(result["maximum_raw_metric_difference"], 0.)
            self.assertEqual(len(result["acceptance"]["checks"]), 17)
            type(self).pipeline_evidence = {
                "description": "Six actual SciPy fits on generated handmade scores; no model or formal dataset",
                "actual_fits": {run_id: method.data.read_json(out / run_id / "fit.json") for run_id in method.RUNS},
                "fitted_records": fitted, "full_score_roundtrip": True,
                "twelve_metrics_saved_before_injected_statistics_failure": True,
                "finalize_recovered_without_labels_or_refitting": True,
                "maximum_raw_metric_difference": result["maximum_raw_metric_difference"],
                "acceptance": result["acceptance"]}

    def test_incomplete_scores_and_incomplete_metrics_cannot_proceed(self):
        with self.assertRaises(ValueError):
            method.summarize({}, ["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            method.data.write_json(out / "fitted.json", {"status": "SIX_CALIBRATORS_SAVED_BEFORE_VALID", "runs": {}})
            with self.assertRaises(ValueError):
                runner.restored_scores(out, {}, {})

    def test_formal_authorization_required_before_loading_saved_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            method.data.write_json(root / "authorization.json", {"status": "CPU_ONLY"})
            with mock.patch.object(runner.platform, "system", return_value="Linux"), \
                 mock.patch.object(runner.os, "sched_getaffinity", return_value={0}, create=True), \
                 mock.patch.object(runner, "sources", return_value=[]), \
                 mock.patch.object(runner, "historical", side_effect=AssertionError("inputs must not open")):
                with self.assertRaises(ValueError):
                    runner.execute(root / "job", root / "authorization.json")


if __name__ == "__main__":
    unittest.main()
