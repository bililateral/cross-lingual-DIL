"""Linux adapter checks using handmade labels and opaque miniature files only."""
from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_sensitivity_evaluate as adapter
from test_step28_continual_sensitivity_contracts import fixture

frozen, data, pilot = adapter.frozen, adapter.data, adapter.pilot


class LinuxEvaluationContracts(unittest.TestCase):
    def test_exact_original_evaluation_and_single_hand_label_parse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, _, groups, meta, checked, old, scores, arrays, old_config = fixture(root)
            reference = copy.deepcopy(arrays)
            for name in reference:
                if "_stage" in name:
                    reference[name] += .1
            attach = pilot.attach_labels
            with ExitStack() as stack:
                stack.enter_context(patch.object(pilot, "contract", return_value=old_config))
                stack.enter_context(patch.object(pilot, "public_inputs", return_value=(groups, meta, checked)))
                stack.enter_context(patch.object(frozen, "origin", return_value=(old, scores, arrays)))
                stack.enter_context(patch.object(adapter.distill, "reference_baseline", return_value=arrays))
                stack.enter_context(patch.object(frozen, "reference_distillation", return_value=reference))
                with patch.object(frozen.platform, "system", return_value="Windows"):
                    expected = frozen.evaluate(out, root / "original")
                with patch.object(adapter.platform, "system", return_value="Linux"), \
                        patch.object(adapter, "verify_supervision", return_value={}), \
                        patch.object(adapter, "verify_outer_job", return_value={}), \
                        patch.object(pilot, "attach_labels", wraps=attach) as labels:
                    actual = adapter.evaluate(out, root / "linux")
                self.assertEqual(labels.call_count, 1)
                self.assertEqual(actual, expected)
                for path in (root / "original").rglob("*_metrics.npy"):
                    self.assertEqual(path.read_bytes(), (root / "linux" / path.relative_to(root / "original")).read_bytes())
                arrays["ABC_shared"][0, 0] += 1e-8
                with patch.object(adapter.platform, "system", return_value="Linux"), \
                        patch.object(adapter, "verify_supervision", return_value={}), \
                        patch.object(adapter, "verify_outer_job", return_value={}), \
                        patch.object(pilot, "attach_labels", wraps=attach) as labels:
                    collected = adapter.evaluate(out, root / "collected", collect_only=True)
                self.assertEqual(labels.call_count, 1)
                self.assertEqual(collected["status"], "METRICS_COLLECTED_SHARED_RECONCILIATION_REQUIRED")
                self.assertEqual(len(list((root / "collected").rglob("*_metrics.npy"))), 18)
                self.assertFalse((root / "collected/evaluation.json").exists())
                differences = data.read_json(root / "collected/shared_differences.json")
                self.assertFalse(differences["weaker_replay/ABC_shared"]["exact"])
                self.assertAlmostEqual(differences["weaker_replay/ABC_shared"]["maximum_absolute_difference"], 1e-8)
                self.assertTrue(all(not arm["comparisons"] for arm in collected["arms"].values()))
                with patch.object(adapter.platform, "system", return_value="Linux"), \
                        patch.object(adapter, "verify_outer_job", return_value={}), \
                        patch.object(adapter, "verify_supervision", side_effect=FileNotFoundError("handmade")), \
                        patch.object(pilot, "attach_labels") as labels:
                    with self.assertRaises(FileNotFoundError):
                        adapter.evaluate(out, root / "missing")
                    labels.assert_not_called()
                    self.assertFalse((root / "missing").exists())

    def test_supervision_presence_size_and_hash_before_parser(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(data, "ROOT", Path(temp)):
            root = Path(temp) / "data"
            path = root / "development/supervision/pairs.csv"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"handmade opaque fixture")
            data.write_json(root / "manifest.json", {"files": {
                "development/supervision/pairs.csv": data.record(path, root)}})
            config = {"data_root": "data"}
            self.assertEqual(adapter.verify_supervision(config)["bytes"], 23)
            path.write_bytes(b"Handmade opaque fixture")
            with self.assertRaises(ValueError):
                adapter.verify_supervision(config)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                adapter.verify_supervision(config)

    def test_outer_success_timing_and_six_actual_model_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            out = job / "run"
            out.mkdir()
            for name, value in {"exit_status.txt": "0", "started.txt": "2026-09-23T22:00:00+08:00",
                                "finished.txt": "2026-09-24T00:00:00+08:00", "resource_usage.log": "hand"}.items():
                (job / name).write_text(value)
            record = {"arms": {}}
            for arm in frozen.ARMS:
                directory = out / arm
                directory.mkdir()
                models = {}
                for order in ("ABC", "BCA", "CAB"):
                    path = directory / (order + ".pt")
                    path.write_bytes(b"handmade model fixture")
                    models[order] = data.record(path, directory)
                data.write_json(directory / "manifest.json", {"models": models})
                record["arms"][arm] = {"manifest": data.record(directory / "manifest.json", out)}
            config = {"runtime": {"maximum_gpu_stage_seconds": 18000}}
            self.assertEqual(len(adapter.verify_outer_job(out, config, record)["models"]), 6)
            (job / "exit_status.txt").write_text("1")
            with self.assertRaises(ValueError):
                adapter.verify_outer_job(out, config, record)
            (job / "exit_status.txt").write_text("0")
            path.write_bytes(b"Handmade model fixture")
            with self.assertRaises(ValueError):
                adapter.verify_outer_job(out, config, record)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                adapter.verify_outer_job(out, config, record)


if __name__ == "__main__":
    unittest.main()
