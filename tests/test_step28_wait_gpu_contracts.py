"""Queue-control fixtures only; no project labels, models or GPU computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_wait_gpu as waiter


class QueueContracts(unittest.TestCase):
    def test_idle_gpu_does_not_launch_before_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with mock.patch.object(waiter, "resources", return_value={"ready": True}), \
                    mock.patch.object(waiter.time, "sleep", side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    waiter.wait_ready(path, "base_models", path, 32 * 1024**3)
            state = waiter.read(path / "wait_status.json")
            self.assertEqual(state["status"], "WAITING_BASE_REVIEW")
            self.assertFalse(state["ready"])
            self.assertEqual(state["consecutive_ready"], 0)

    def test_two_consecutive_eligible_checks_and_review_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"path": "fixture"}]
            with mock.patch.object(waiter, "approved", return_value=rows), \
                    mock.patch.object(waiter, "resources", side_effect=[{"ready": x} for x in (True, False, True, True)]) as probe, \
                    mock.patch.object(waiter, "verify_sources") as verify, \
                    mock.patch.object(waiter.time, "sleep") as sleep:
                self.assertEqual(waiter.wait_ready(Path(tmp), "base_models", Path(tmp), 32 * 1024**3), rows)
                self.assertEqual(sleep.call_count, 3)
                self.assertEqual(probe.call_args.args, (32 * 1024**3,))
                verify.assert_called_once_with(rows)

    def test_unfinished_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            waiter.write(path / "base_review_ready.json", {"status": "REVIEW_RUNNING"})
            with self.assertRaises(ValueError):
                waiter.approved(path, "base_models", path)

    def test_changed_source_stops_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.txt").write_text("changed")
            with mock.patch.object(waiter, "ROOT", root), mock.patch.object(waiter.subprocess, "Popen") as spawn:
                with self.assertRaises(RuntimeError):
                    waiter.launch(root / "reports/run", "fixture.sh", [{"path": "source.txt", "bytes": 7, "sha256": "wrong"}])
                spawn.assert_not_called()

    def test_actual_harmless_bash_spawn_has_one_attempt_and_explicit_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "fixture.sh"
            script.write_text('mkdir "$1"\nprintf "%s" "$CUDA_VISIBLE_DEVICES" > "$1/fixture_gpu.txt"\n')
            rows = [{"path": "fixture.sh", "bytes": script.stat().st_size,
                     "sha256": hashlib.sha256(script.read_bytes()).hexdigest()}]
            base = root / "reports/run"
            with mock.patch.object(waiter, "ROOT", root):
                process = waiter.launch(base, "fixture.sh", rows)
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual((base / "job/fixture_gpu.txt").read_text(), "0")
                with self.assertRaises(RuntimeError):
                    waiter.launch(base, "fixture.sh", rows)

    def complete_fixture(self, base: Path) -> list:
        run = base / "job/run"
        run.mkdir(parents=True)
        (base / "job/exit_status.txt").write_text("0")
        rows = [{"path": "handcrafted_source"}]
        waiter.write(run / "manifest.json", {"status": waiter.BASE_COMPLETE, "physical_updates": 2592,
                     "arms": dict.fromkeys(("labse", "multilingual_e5_large", "bge_m3")), "source_files": rows})
        digest = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
        waiter.write(run / "completion.json", {"status": waiter.BASE_COMPLETE, "manifest_sha256": digest})
        waiter.write(run / "model_verification.json", {"status": "SIX_MODELS_FRESH_SIZE_SHA_VERIFIED",
                     "manifest_sha256": digest, "files": ["fixture"] * 6})
        return rows

    def test_base_success_gate_and_failure_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self.complete_fixture(base)
            waiter.check_base_completion(base, rows)
            waiter.write(base / "job/run/failure.json", {"status": "failed"})
            with self.assertRaises(RuntimeError):
                waiter.check_base_completion(base, rows)

    def test_incomplete_base_cannot_start_sensitivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self.complete_fixture(base)
            waiter.write(base / "job/run/completion.json", {"status": waiter.BASE_COMPLETE, "manifest_sha256": "wrong"})
            with self.assertRaises(RuntimeError):
                waiter.check_base_completion(base, rows)

    def fixture_queue(self, root: Path) -> tuple[Path, Path, Path]:
        queue, base, sensitive = (root / "reports" / p for p in ("queue", "base", "sensitivity"))
        queue.mkdir(parents=True)
        sensitive.mkdir()
        waiter.write(queue / "queue.json", {"base_models": "reports/base", "weight_sensitivity": "reports/sensitivity"})
        return queue, base, sensitive

    def test_order_completion_then_second_resource_wait_then_self_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue, base, sensitive = self.fixture_queue(root)
            script = root / "scripts/step28_wait_gpu.py"
            script.parent.mkdir()
            script.write_text("disposable queue fixture")
            (sensitive / "job/run").mkdir(parents=True)
            (sensitive / "job/run/startup.json").write_text("{}")
            first = mock.Mock(returncode=0, pid=1)
            first.poll.return_value = 0
            second = mock.Mock(pid=2)
            second.poll.return_value = None
            events = []
            def wait(q, stage, s, disk):
                events.append(("wait", stage, disk))
                return []
            def launch(path, name, rows):
                events.append(("launch", path.name))
                return first if path == base else second
            with mock.patch.object(waiter, "ROOT", root), mock.patch.object(waiter, "__file__", str(script)), \
                    mock.patch.object(waiter, "wait_ready", side_effect=wait), \
                    mock.patch.object(waiter, "launch", side_effect=launch), \
                    mock.patch.object(waiter, "check_base_completion", side_effect=lambda *args: events.append(("complete",))):
                waiter.run_queue(queue)
            self.assertEqual(events, [("wait", "base_models", 32*1024**3), ("launch", "base"), ("complete",),
                                      ("wait", "weight_sensitivity", 24*1024**3), ("launch", "sensitivity")])
            self.assertFalse(script.exists())
            self.assertFalse(waiter.read(queue / "completion.json")["sensitivity_training_complete"])

    def test_base_exit_failure_stops_queue_without_second_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue, _, _ = self.fixture_queue(root)
            process = mock.Mock(returncode=1)
            process.poll.return_value = 1
            with mock.patch.object(waiter, "ROOT", root), mock.patch.object(waiter, "wait_ready", return_value=[]) as ready, \
                    mock.patch.object(waiter, "launch", return_value=process) as spawn:
                with self.assertRaises(RuntimeError):
                    waiter.run_queue(queue)
                self.assertEqual(spawn.call_count, 1)
                self.assertEqual(ready.call_count, 1)

    def test_queue_cannot_be_automatically_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue, _, _ = self.fixture_queue(root)
            (queue / "queue_start.json").write_text("{}")
            with mock.patch.object(waiter, "ROOT", root), mock.patch.object(waiter, "wait_ready") as ready:
                with self.assertRaises(FileExistsError):
                    waiter.run_queue(queue)
                ready.assert_not_called()


if __name__ == "__main__":
    unittest.main()
