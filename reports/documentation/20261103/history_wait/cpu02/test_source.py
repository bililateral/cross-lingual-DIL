"""Delta checks only: handwritten receipts and ordinary CPU child processes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import step28_history_wait as wait
import step28_history_optuna_control as control


def record(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=wait.hashlib.sha256(path.read_bytes()).hexdigest())


class WaitTests(unittest.TestCase):
    def test_predecessor_wait_failure_and_full_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings = dict(state=str(root))
            for key in ("forever_gate", "vendor_gate"):
                campaign = root/key
                campaign.mkdir()
                path = root/(key+".json")
                wait.write(path, dict(campaign=str(campaign)))
                bindings[key] = record(path)
            path = root/"process.json"
            wait.write(path, dict(owner="fake_owner", **bindings))
            bindings["process"] = record(path)
            live = mock.Mock(pid=999999, info=dict(pid=999999, username="fake_owner",
                status="sleeping", cmdline=[bindings["vendor_gate"]["path"]]))
            validate = mock.Mock()
            with mock.patch.object(wait.psutil, "process_iter", return_value=[live]):
                self.assertFalse(wait.predecessor(bindings, validate)["ready"])
                wait.write(root/"completion.json", dict(status="BOTH_FORMAL_BATCHES_COMPLETE"))
                self.assertEqual(wait.predecessor(bindings, validate)["reason"], "PREDECESSOR_EXIT_PENDING")
                validate.assert_not_called()
            with mock.patch.object(wait.psutil, "process_iter", return_value=[]):
                self.assertTrue(wait.predecessor(bindings, validate)["ready"])
                validate.assert_called_once_with(bindings)
                (root/"completion.json").unlink()
                with self.assertRaisesRegex(RuntimeError, "without complete"):
                    wait.predecessor(bindings, validate)
                wait.write(root/"failure.json", dict(error="original failure"))
                with self.assertRaisesRegex(RuntimeError, "chain failed"):
                    wait.predecessor(bindings, validate)
            available = mock.Mock(return_value=dict(ready=False))
            with mock.patch.object(wait, "predecessor", return_value=dict(ready=False)):
                self.assertEqual(wait.poll(bindings, {}, validate, available)["status"], "WAITING")
                available.assert_not_called()
            with mock.patch.object(wait, "predecessor", return_value=dict(ready=True)):
                self.assertEqual(wait.poll(bindings, {}, validate, available)["status"], "WAITING")
                available.return_value = dict(ready=True)
                self.assertEqual(wait.poll(bindings, {}, validate, available)["status"], "READY")

    def test_actual_dispatch_hook_deletes_only_after_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root/"formal"
            trial = campaign/"trial_00"
            trial.mkdir(parents=True)
            wait.write(trial/"batch.json", dict(sources=[]))
            state = root/"state"
            state.mkdir()
            active = root/"active_history_wait.py"
            active.write_text("retained test listener")
            source = record(active)
            self.assertFalse(wait.observe_update(campaign, active, source, state))
            calls = []
            def observe():
                calls.append(wait.observe_update(campaign, active, source, state))
            allowed = dict(workers=3, campaign=str(campaign), limits=control.run.LIMITS,
                           dispatch_index=str(trial))
            ledger = control.Ledger(root, dict(training=0, confirmation=0, auxiliary=0), control.run.BUDGETS)
            row = dict(id="toy", gpu=False, bucket="training", seconds=120)
            command = lambda row, job, deadline: [sys.executable, "-c",
                'import json,time;print(json.dumps(dict(event="updates",run_id="toy",stage=1,completed=1)),flush=True);time.sleep(.2)']
            with mock.patch.object(control.queue, "check_running"):
                result = control.dispatch(trial, [row], command, lambda row, job: None,
                    allowed, ledger, poll_seconds=.03, observe=observe)
            self.assertTrue(result["completed"][0]["success"])
            self.assertIn(True, calls)
            self.assertFalse(active.exists())
            self.assertTrue((state/"first_update.json").exists())
            self.assertFalse(wait.observe_update(campaign, active, source, state))

    def test_production_wait_entry_waits_then_calls_formal_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root/"active_history_wait.py"
            active.write_bytes(Path(wait.__file__).read_bytes())
            spec = importlib.util.spec_from_file_location("temporary_active_wait", active)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            workspace = root/"workspace"
            workspace.mkdir()
            template = root/"gate.json"
            wait.write(template, dict(workspace=str(workspace), campaign=str(workspace/"formal"),
                predecessor={}, sources=[], prior_used_seconds=dict(auxiliary=135)))
            evidence = root/"review.txt"
            evidence.write_text("handwritten evidence")
            config_path = root/"config.json"
            wait.write(config_path, dict(approved=True, review_closed=True, state=str(root/"state"),
                workspace=str(workspace), gate_template=record(template), listener_sources=[record(active)],
                review_evidence=record(evidence), listener=str(active), prior_auxiliary_seconds=135))
            observations = [dict(status="WAITING"), dict(status="WAITING"), dict(status="READY")]
            def execute(path, *, observe):
                gate = wait.read(path)
                self.assertGreaterEqual(gate["prior_used_seconds"]["auxiliary"], 135)
                campaign = Path(gate["campaign"])
                trial = campaign/"trial_00"
                trial.mkdir(parents=True)
                wait.write(trial/"batch.json", dict(sources=[]))
                job = trial/"toy"
                job.mkdir()
                (job/"console.txt").write_text(json.dumps(dict(event="updates", stage=1, completed=1, run_id="toy"))+"\n")
                self.assertTrue(observe())
            original_cwd = Path.cwd()
            try:
                with mock.patch.object(module, "poll", side_effect=observations), mock.patch.object(module.time, "sleep") as sleep, mock.patch.object(control.run, "gate"), mock.patch.object(control, "execute", side_effect=execute) as launch:
                    module.execute(config_path)
                    self.assertEqual(sleep.call_count, 2)
                    self.assertEqual(launch.call_count, 1)
                    with self.assertRaises(FileExistsError):
                        module.execute(config_path)
                self.assertFalse(active.exists())
                self.assertTrue((root/"state/completion.json").exists())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
