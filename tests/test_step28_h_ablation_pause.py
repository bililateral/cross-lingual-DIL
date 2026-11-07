"""Linux CPU-only evidence for dispatch gating and the narrow controller bridge."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCES = Path(sys.argv.pop(1))
FROZEN = Path(sys.argv.pop(1))
sys.path.insert(0, str(FROZEN/"scripts"))
dispatch = load(SOURCES/"step28_history_optuna_control.py", "pause_dispatch_under_test")
bridge = load(SOURCES/"step28_h_ablation_pause.py", "pause_bridge_under_test")
control = load(SOURCES/"step28_h_ablation_control.py", "pause_controller_under_test")


class Ledger:
    def __init__(self):
        self.used = dict(training=0., auxiliary=0.)
        self.caps = dict(training=1000., auxiliary=1000.)
        self.events = []
    def control(self):
        pass
    def finish(self, bucket, identity, started, released):
        self.used[bucket] += released-started
        self.events.append(identity)
    def pause(self, seconds, *, active):
        time.sleep(seconds)


class PauseTests(unittest.TestCase):
    def run_dispatch(self, root, rows, flag, *, prior=None, during=None):
        ledger = Ledger()
        allowed = dict(workers=3, sources=["frozen-science"], workspace=str(root), campaign=str(root))
        errors, result = [], []
        def command(row, job, deadline):
            return [sys.executable, "-c", "import time,sys;time.sleep(.12);sys.exit(int(sys.argv[1]))", str(row.get("exit", 0))]
        def worker():
            try:
                with patch.object(dispatch.queue, "check_running", lambda *a: None):
                    result.append(dispatch.dispatch(root, rows, command, lambda *a: None, allowed,
                        ledger, poll_seconds=.02, dispatch_paused=flag.is_set, continuation=prior))
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            if during:
                during(thread)
            thread.join(5)
        finally:
            flag.clear()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        return ledger, result, errors

    @staticmethod
    def wait_until(predicate):
        end = time.monotonic()+4
        while not predicate():
            if time.monotonic() > end:
                raise AssertionError("Condition not observed")
            time.sleep(.01)

    def test_initial_pause_then_resume_cpu_followup(self):
        with tempfile.TemporaryDirectory() as d:
            root, flag = Path(d), threading.Event()
            flag.set()
            def during(thread):
                self.wait_until(lambda: (root/"dispatch.json").exists())
                time.sleep(.1)
                self.assertFalse((root/"generation").exists())
                self.assertEqual(bridge.read(root/"dispatch.json")["status"], "PAUSED_IDLE")
                flag.clear()
            ledger, result, errors = self.run_dispatch(root,
                [dict(id="generation", bucket="auxiliary", seconds=10, shutdown_reserve=0, gpu=False)], flag, during=during)
            self.assertFalse(errors)
            self.assertEqual(ledger.events, ["generation"])
            self.assertEqual(result[0]["status"], "ALL_COMPLETE_RELEASED")

    def test_running_job_drains_and_last_job_waits(self):
        with tempfile.TemporaryDirectory() as d:
            root, flag = Path(d), threading.Event()
            def during(thread):
                self.wait_until(lambda: (root/"only/process.json").exists())
                flag.set()
                self.wait_until(lambda: (root/"only/dispatch_receipt.json").exists())
                self.wait_until(lambda: bridge.read(root/"dispatch.json")["status"] == "PAUSED_IDLE")
                self.assertTrue(thread.is_alive())
                self.assertTrue(bridge.read(root/"only/dispatch_receipt.json")["success"])
                time.sleep(.1)
                flag.clear()
            ledger, result, errors = self.run_dispatch(root,
                [dict(id="only", bucket="training", seconds=10, shutdown_reserve=0, gpu=False)], flag, during=during)
            self.assertFalse(errors)
            self.assertEqual(ledger.events, ["only"])

    def test_three_slots_finish_without_taking_pending(self):
        with tempfile.TemporaryDirectory() as d:
            root, flag = Path(d), threading.Event()
            rows = [dict(id=f"job{i}", bucket="training", seconds=10, shutdown_reserve=0, gpu=False) for i in range(5)]
            def during(thread):
                self.wait_until(lambda: all((root/f"job{i}/process.json").exists() for i in range(3)))
                flag.set()
                self.wait_until(lambda: all((root/f"job{i}/dispatch_receipt.json").exists() for i in range(3)))
                self.assertFalse((root/"job3").exists())
                self.assertFalse((root/"job4").exists())
                flag.clear()
            ledger, result, errors = self.run_dispatch(root, rows, flag, during=during)
            self.assertFalse(errors)
            self.assertCountEqual(ledger.events, [r["id"] for r in rows])
            self.assertEqual(len(result[0]["completed"]), 5)

    def test_continue_preserves_completed_and_retired_slots(self):
        with tempfile.TemporaryDirectory() as d:
            root, flag = Path(d), threading.Event()
            rows = [dict(id=f"job{i}", bucket="training", seconds=10, shutdown_reserve=0, gpu=False) for i in range(4)]
            prior = dict(sources=["frozen-science"], active={}, retired_slots=[0], pending=["job2", "job3"],
                completed=[dict(id="job0", index=0, slot=0, success=False, released=True),
                           dict(id="job1", index=1, slot=1, success=True, released=True)])
            ledger, result, errors = self.run_dispatch(root, rows, flag, prior=prior)
            self.assertEqual(len(errors), 1)  # Original failed attempt still blocks whole-batch success.
            self.assertIn("Incomplete batch", str(errors[0]))
            self.assertFalse((root/"job0").exists())
            self.assertFalse((root/"job1").exists())
            saved = bridge.read(root/"dispatch.json")
            self.assertEqual(saved["retired_slots"], [0])
            self.assertEqual([r["slot"] for r in saved["completed"][2:]], [1, 2])
            self.assertCountEqual(ledger.events, ["job2", "job3"])

    def test_continue_rejects_changed_queue(self):
        with tempfile.TemporaryDirectory() as d:
            root, flag = Path(d), threading.Event()
            prior = dict(sources=["frozen-science"], active={}, retired_slots=[], pending=["wrong"], completed=[])
            _, _, errors = self.run_dispatch(root,
                [dict(id="expected", bucket="training", seconds=10, gpu=False)], flag, prior=prior)
            self.assertIsInstance(errors[0], ValueError)

    def test_ledger_does_not_reset_or_double_final_tail(self):
        with tempfile.TemporaryDirectory() as d:
            budget = dict(caps=control.run.BUDGETS,
                used_seconds=dict(training=123., confirmation=7., auxiliary=80.),
                events=[dict(id="final_tail", bucket="auxiliary", conservative_seconds=30)])
            ledger = bridge.restored_ledger(control, Path(d), budget, 4)
            self.assertEqual(ledger.used, dict(training=123., confirmation=7., auxiliary=84.))
            self.assertEqual(sum(e["id"] == "final_tail" for e in ledger.events), 1)
            self.assertEqual(budget["used_seconds"]["auxiliary"], 80.)

    def test_held_child_exit_status_and_release(self):
        with tempfile.TemporaryDirectory() as d:
            child_file = Path(d)/"child.json"
            program = ("import subprocess,sys,time,json;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.8);raise SystemExit(7)'],start_new_session=True);"
                "open(sys.argv[1],'w').write(json.dumps(p.pid));time.sleep(20)")
            parent = subprocess.Popen([sys.executable, "-c", program, str(child_file)], start_new_session=True)
            try:
                self.wait_until(child_file.exists)
                pid = bridge.read(child_file)
                expected = bridge.identity(pid)
                os.kill(parent.pid, signal.SIGSTOP)
                held = bridge.HeldChild(expected, parent.pid)
                self.assertEqual(held.wait(4), 7)
                self.assertFalse(dispatch.release(held))
                self.assertEqual(held.returncode, 7)
            finally:
                parent.kill()
                parent.wait()


if __name__ == "__main__":
    unittest.main(verbosity=2)
