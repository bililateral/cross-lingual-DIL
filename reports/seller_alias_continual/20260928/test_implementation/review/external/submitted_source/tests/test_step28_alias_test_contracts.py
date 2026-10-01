"""Handwritten test-split, paired inference, one-parse and statistics cases."""
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
import step28_alias_test as method
import step28_alias_test_run as runner

data = method.data


def handmade_truth() -> np.ndarray:
    controllers = [c for c in range(8) for _ in range(2)] + [c for c in range(8, 12) for _ in range(3)]
    return np.array([int(controllers[i] == controllers[j]) for i, j in itertools.combinations(range(28), 2)], dtype=np.uint8)


def fixture(root: Path) -> tuple[dict, list, dict, dict, np.ndarray, Path]:
    dataset = root / "data"
    (dataset / "heldout/supervision").mkdir(parents=True)
    metadata, groups, item_lines, pair_rows = [], [], [], []
    for split, count in (("train", 60), ("development", 20), ("heldout", 40)):
        for domain in "ABC":
            for index in range(count):
                uid = f"{split}_{domain}{index:02}"
                metadata.append(dict(domain=domain, split=split, group_uid=uid, group_index=index, accounts=28, items=56))
                if split != "heldout":
                    continue
                sellers = tuple(f"{uid}_s{i:02}" for i in range(28))
                items = tuple(tuple((f"{s}_i{j}", "手写商品标题", "手写商品描述") for j in range(2)) for s in sellers)
                groups.append(data.Group(uid, sellers, items))
                for s, rows in zip(sellers, items, strict=True):
                    for item, title, description in rows:
                        item_lines.append(data.json_bytes(dict(group_uid=uid, seller_uid=s, item_uid=item, title=title, description=description)))
                pair_rows.extend(dict(group_uid=uid, seller_uid_left=u, seller_uid_right=v, label=int(y))
                                 for (u, v), y in zip(itertools.combinations(sellers, 2), handmade_truth(), strict=True))
    with (dataset / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["domain", "split", "group_uid", "group_index", "accounts", "items"])
        writer.writeheader()
        writer.writerows(reversed(metadata))
    (dataset / "heldout/items.jsonl").write_bytes(b"".join(reversed(item_lines)))
    with (dataset / "heldout/supervision/pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group_uid", "seller_uid_left", "seller_uid_right", "label"])
        writer.writeheader()
        writer.writerows(reversed(pair_rows))
    p = copy.deepcopy(runner.contract())
    p["data"]["root"] = "data"
    names = ("groups.csv", "heldout/items.jsonl", "heldout/supervision/pairs.csv")
    p["data"]["inputs"] = {name: data.record(dataset / name, root) for name in names}
    manifest = {"study": p["data"]["study"], "root_seed": p["data"]["root_seed"],
                "files": {name: {k: rec[k] for k in ("bytes", "sha256")} for name, rec in p["data"]["inputs"].items()}}
    data.write_json(dataset / "manifest.json", manifest)
    data.write_json(dataset / "validation.json", {"status": "PASS_GENERATION_CONTRACT_NOT_MODEL_QUALIFICATION"})
    p["data"]["manifest"] = data.record(dataset / "manifest.json", root)
    p["data"]["validation"] = data.record(dataset / "validation.json", root)
    truth = np.tile(handmade_truth(), (120, 1))
    raw = np.random.default_rng(83).normal(-1, 1, (120, 378)).astype(np.float32)
    info = {"A": {"map": {"a": .7, "b": -.2}, "threshold": .1},
            "C": {"map": {"a": 1.1, "b": .05}, "threshold": .3}}
    out = root / "job"
    out.mkdir()
    blind = {"policy_sha256": data.sha256(runner.POLICY), "group_ids": [g.uid for g in groups],
             "domains": ["ABC"[i // 40] for i in range(120)], "models": {}}
    for name, values in (("A", raw), ("C", (raw + truth).astype(np.float32))):
        p["models"][name]["payload"] = {"handmade": name}
        p["models"][name]["calibration"] = {"handmade_map": name}
        blind["models"][name] = runner.save_scores(out, name, values, info[name]["map"],
            {"model": p["models"][name]["payload"], "map": p["models"][name]["calibration"]})
    data.write_json(out / "blind.json", blind)
    return p, groups, info, blind, truth, out


def passing() -> dict:
    fields = {m: {"mean": 0., "conditional_95pct_interval": [0., 0.]} for m in method.metrics.COLUMNS}
    fields["map"] = {"mean": .02, "conditional_95pct_interval": [.01, .03]}
    fields["recall_at_5"]["mean"] = .01
    return {"C_cal_minus_A_cal": copy.deepcopy(fields), "C_cal_minus_A_raw": copy.deepcopy(fields)}


class TestContracts(unittest.TestCase):
    pipeline_evidence: dict = {}

    def test_contract_nine_roles_and_no_training(self):
        p = runner.contract()
        self.assertEqual(len(p["criteria"]), 9)
        self.assertEqual(p["inference"]["optimizer_updates"], 0)
        self.assertEqual(p["access"]["label_parses"], dict(train=0, development=0, heldout=1, owners=0))

    def test_public_reads_only_heldout_and_stable_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, *_ = fixture(root)
            actual, metadata = runner.public_inputs(p, root)
            self.assertEqual(actual, groups)
            self.assertEqual([r["domain"] for r in metadata], list("A" * 40 + "B" * 40 + "C" * 40))
            self.assertFalse((root / "data/train/items.jsonl").exists())
            self.assertTrue(all(g.labels is None for g in actual))

    def test_forbidden_input_paths_rejected(self):
        for name in ("train/items.jsonl", "development/supervision/pairs.csv", "heldout/supervision/owners.csv", "../groups.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.allowed_input({}, name)

    def test_public_missing_group_and_extra_feature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, *_ = fixture(root)
            path = root / "data/heldout/items.jsonl"
            lines = path.read_bytes().splitlines(keepends=True)
            original_lines = list(lines)
            row = __import__("json").loads(lines[0])
            row["controller_uid"] = "forbidden"
            lines[0] = data.json_bytes(row)
            path.write_bytes(b"".join(lines))
            p["data"]["inputs"]["heldout/items.jsonl"] = data.record(path, root)
            manifest_path = root / "data/manifest.json"
            manifest = data.read_json(manifest_path)
            manifest["files"]["heldout/items.jsonl"] = {k: p["data"]["inputs"]["heldout/items.jsonl"][k] for k in ("bytes", "sha256")}
            data.write_json(manifest_path, manifest)
            p["data"]["manifest"] = data.record(manifest_path, root)
            with self.assertRaisesRegex(ValueError, "schema"):
                runner.public_inputs(p, root)
            missing_uid = __import__("json").loads(original_lines[0])["group_uid"]
            path.write_bytes(b"".join(line for line in original_lines
                                      if __import__("json").loads(line)["group_uid"] != missing_uid))
            p["data"]["inputs"]["heldout/items.jsonl"] = data.record(path, root)
            manifest["files"]["heldout/items.jsonl"] = {k: p["data"]["inputs"]["heldout/items.jsonl"][k] for k in ("bytes", "sha256")}
            data.write_json(manifest_path, manifest)
            p["data"]["manifest"] = data.record(manifest_path, root)
            with self.assertRaisesRegex(ValueError, "Missing test group"):
                runner.public_inputs(p, root)

    def test_reverse_supervision_alignment_and_single_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, truth, out = fixture(root)
            actual, alignment = runner.parse_once(out, p, groups, blind, info, root)
            np.testing.assert_array_equal(actual, truth)
            self.assertEqual((alignment["pairs"], alignment["positive_pairs"], alignment["queries"]), (45360, 2400, 3360))
            with self.assertRaises(FileExistsError):
                runner.parse_once(out, p, groups, blind, info, root)

    def test_duplicate_pair_rejected_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, _, out = fixture(root)
            path = root / "data/heldout/supervision/pairs.csv"
            lines = path.read_bytes().splitlines(keepends=True)
            lines[-1] = lines[-2]
            path.write_bytes(b"".join(lines))
            p["data"]["inputs"]["heldout/supervision/pairs.csv"] = data.record(path, root)
            with self.assertRaisesRegex(ValueError, "uniqueness"):
                runner.parse_once(out, p, groups, blind, info, root)
            self.assertEqual(data.read_json(out / "heldout_access.json")["parse_attempts"], 1)

    def test_both_models_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, _, out = fixture(root)
            del blind["models"]["C"]
            data.write_json(out / "blind.json", blind)
            with self.assertRaisesRegex(ValueError, "Both fixed"):
                runner.parse_once(out, p, groups, blind, info, root)
            self.assertFalse((out / "heldout_access.json").exists())

    def test_score_corruption_rejected_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, _, out = fixture(root)
            np.save(out / "C_cal_scores.npy", np.zeros((120, 378)))
            with self.assertRaises(ValueError):
                runner.parse_once(out, p, groups, blind, info, root)
            self.assertFalse((out / "heldout_access.json").exists())

    def test_wrong_model_map_and_group_binding_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, _, out = fixture(root)
            bad = copy.deepcopy(blind)
            bad["models"]["C"]["origin"]["map"] = p["models"]["A"]["calibration"]
            with self.assertRaisesRegex(ValueError, "binding"):
                runner.restore_scores(out, bad, p, info)
            blind["group_ids"] = list(reversed(blind["group_ids"]))
            data.write_json(out / "blind.json", blind)
            with self.assertRaisesRegex(ValueError, "aligned"):
                runner.parse_once(out, p, groups, blind, info, root)
            self.assertFalse((out / "heldout_access.json").exists())

    def test_nine_conditions_fail_individually_zero_tolerance(self):
        self.assertTrue(method.acceptance(passing())["passed"])
        for key, comp, metric, statistic, operator, bound in method.CRITERIA:
            values = passing()
            failed = bound if operator == ">" else np.nextafter(bound, -np.inf if operator == ">=" else np.inf)
            if statistic == "lower":
                values[comp][metric]["conditional_95pct_interval"][0] = failed
            else:
                values[comp][metric]["mean"] = failed
            self.assertEqual(method.acceptance(values)["failed"], [key])

    def test_bad_calibrated_control_cannot_hide_raw_regression(self):
        values = passing()
        for m in ("brier", "log_loss"):
            values["C_cal_minus_A_cal"][m]["mean"] = -.01
            values["C_cal_minus_A_raw"][m]["mean"] = .001
        self.assertEqual(method.acceptance(values)["failed"], ["T8", "T9"])

    def test_stratified_group_bootstrap_independent_frequency_reference(self):
        domains = list("A" * 40 + "B" * 40 + "C" * 40)
        candidate = np.zeros((120, 22))
        candidate[:, 12] = np.arange(120, dtype=float) / 120
        draws = method.bootstrap_draws()
        result = method.compare(candidate, np.zeros_like(candidate), domains, draws)["map"]
        frequencies = np.zeros((5000, 120), dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(20260928))
        for b in range(5000):
            for d in range(3):
                for g in rng.integers(0, 40, 40):
                    frequencies[b, d * 40 + g] += 1
        scalar = frequencies @ candidate[:, 12] / 120
        values = sorted(scalar.tolist())
        interval = []
        for q in (.025, .975):
            h = 4999 * q
            lo = math.floor(h)
            interval.append(values[lo] + (values[math.ceil(h)] - values[lo]) * (h - lo))
        np.testing.assert_allclose(result["conditional_95pct_interval"], interval, atol=2e-15, rtol=0)
        self.assertAlmostEqual(result["mean"], math.fsum(candidate[:, 12]) / 120, places=14)
        self.assertTrue(np.all(frequencies[:, :40].sum(1) == 40))

    def test_ties_two_positives_and_probability_saturation(self):
        y = handmade_truth()[None, :]
        z = np.zeros((1, 378))
        actual = method.metrics.retrieval(y, z, 28)[0]
        adj = np.zeros((28, 28), dtype=int)
        for v, (a, b) in zip(y[0], itertools.combinations(range(28), 2), strict=True):
            adj[a, b] = adj[b, a] = v
        maps, recalls = [], []
        for q in range(28):
            relevant = [int(adj[q, j]) for j in range(28) if j != q]
            maps.append(sum(sum(relevant[:i]) / i for i, value in enumerate(relevant, 1) if value) / sum(relevant))
            recalls.append(sum(relevant[:5]) / sum(relevant))
        self.assertAlmostEqual(actual[0], sum(maps) / 28, places=14)
        self.assertAlmostEqual(actual[4], sum(recalls) / 28, places=14)
        raw = np.array([[1000., 1001., 1001.]])
        cal = method.calibration.transform(raw, {"a": .8, "b": .4})
        self.assertTrue(method.calibration.preserve_order(raw, cal)["exact_order_and_ties_preserved"])
        self.assertTrue(np.all(np.exp(-np.logaddexp(0., -cal)) == 1.))

    def test_four_matrices_saved_before_stats_and_label_free_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, groups, info, blind, _, out = fixture(root)
            truth, _ = runner.parse_once(out, p, groups, blind, info, root)
            scores = runner.restore_scores(out, blind, p, info)
            runner.collect(out, truth, scores, blind, info, [])
            with mock.patch.object(method, "summarize", side_effect=RuntimeError("injected statistics fault")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    runner.finalize(out, [])
            self.assertEqual(len(list((out / "evaluation").glob("*_metrics.npy"))), 4)
            self.assertFalse((out / "evaluation/evaluation.json").exists())
            with mock.patch.object(runner, "parse_once", side_effect=AssertionError("no reparse")), mock.patch.object(runner, "inference_one", side_effect=AssertionError("no reinference")):
                result = runner.finalize(out, [])
            self.assertEqual(result["status"], runner.COMPLETE)
            self.pipeline_evidence.update(complete_matrices=4, matrix_shape=[120, 22], shared_heldout_parse=1,
                                          restored_score_arrays=4, actual_distinct_maps={k: v["map"] for k, v in info.items()},
                                          stats_failure_preserved=True, saved_only_recovery=True,
                                          observed_handmade_acceptance=result["acceptance"])

    def test_actual_tiny_checkpoint_restore_in_inference_wrapper(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = torch.nn.Linear(1, 1)
            with torch.no_grad():
                original.weight.fill_(2.)
                original.bias.fill_(.5)
            metadata = {"run_id": "s0_d", "epoch": 6, "handmade": True}
            payload = runner.core.save_state(root / "tiny.pt", original, None, metadata)
            model_hash = runner.core.state_digest(original.state_dict())
            p = {"models": {"A": {"payload": {**payload, "path": "tiny.pt", "payload_state_sha256": payload["state_sha256"], "model_parameters_sha256": model_hash}}}}
            info = {"A": {"config": {}, "point": {"metadata": metadata}}}
            def tiny_scores(model, groups, config, arm, check):
                model.eval()
                with torch.inference_mode():
                    return model(torch.ones(120 * 378, 1)).numpy().reshape(120, 378)
            with mock.patch.object(data, "ROOT", root), mock.patch.object(runner.base, "load_model", side_effect=lambda *a: torch.nn.Linear(1, 1)), mock.patch.object(runner.base, "score", side_effect=tiny_scores):
                values, actual = runner.inference_one(p, "A", info, [None] * 120, lambda: None)
            np.testing.assert_array_equal(values, np.full((120, 378), 2.5, dtype=np.float32))
            self.assertEqual(actual["parameters_sha256"], model_hash)
            self.pipeline_evidence["actual_tiny_restore_forward"] = True
            self.pipeline_evidence["native_pretrained_weights_loaded"] = 0


if __name__ == "__main__":
    unittest.main()
