"""Native-entry control checks on CPU; simulated workers are not BGE evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import torch

import step28_defaults_native as n


def approval(job):
    return dict(status="APPROVED_ORIGINAL_DEFAULTS_NATIVE", job=str(job.resolve()),
        methods=list(n.METHODS), limits=n.LIMITS, sources=n.sources(), max_update_calls=6,
        handwritten_only=True, sharing="idle_only", total_seconds=120)


class NativeEntryTests(unittest.TestCase):
    def temporary(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ["DEFAULTS_TEST_ROOT"])
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_authorization_resources_and_new_worker_arguments(self):
        root = self.temporary()
        job, auth = root/"native", root/"authorization.json"
        row = approval(job)
        n.m.data.write_json(auth, row)
        self.assertEqual(n.authorization(auth, job), row)
        for key, value in (("status", "CPU_ONLY"), ("max_update_calls", 7), ("total_seconds", float("nan")),
                           ("sources", []), ("sharing", "any_process")):
            # Deliberately malformed external JSON, including NaN. The normal
            # evidence writer correctly rejects NaN before the parser under test.
            auth.write_text(json.dumps({**row, key: value}), encoding="utf-8")
            with self.assertRaises(ValueError):
                n.authorization(auth, job)
        n.m.data.write_json(auth, row)
        observed = dict(gpu="RTX 5090, 32607, 11256, 90", compute_apps="",
                        host_available_bytes=128*n.GIB, disk_available_bytes=128*n.GIB)
        with mock.patch.object(n.verified, "resources", return_value=observed):
            self.assertEqual(n.admission("idle_only")["free_gpu_bytes"], 11256*2**20)
            observed["compute_apps"] = "123, 7000"
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                n.admission("idle_only")
            observed.update(compute_apps="", gpu="RTX 5090, 32607, 10000, 90")
            with self.assertRaisesRegex(RuntimeError, "insufficient"):
                n.admission("idle_only")
        child = job/"H_default"
        child.mkdir(parents=True)
        with mock.patch.object(n.verified, "native", return_value={}) as native:
            n.worker(child, "H_default", time.monotonic()+60, auth)
            args, options = native.call_args
            self.assertEqual(args[1].arm, "G_half")
            self.assertIsNone(args[3])
            self.assertIs(options["implementation"], n.m)
            self.assertIs(options["compare_legacy"], False)
            self.assertEqual(options["allocator_bytes"], int(9.25*n.GIB))
            self.assertIs(options["budget_factory"], n.Budget)
        # The shared native function itself reaches the new config after applying
        # the requested allocator; stop before any model or handcrafted BGE input.
        with mock.patch.object(torch, "set_num_threads"), mock.patch.object(torch, "set_num_interop_threads"), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=types.SimpleNamespace(total_memory=32*n.GIB)), \
                mock.patch.object(torch.cuda, "set_per_process_memory_fraction") as allocator, \
                mock.patch.object(n.m, "config", side_effect=RuntimeError("original_native_config")), \
                mock.patch.object(n.m.parent, "configure_numerics"):
            with self.assertRaisesRegex(RuntimeError, "original_native_config"):
                n.verified.native(child, n.m.Spec("H_default", "ABC", 4), time.monotonic()+60, None,
                    implementation=n.m, allocator_bytes=n.LIMITS["allocator_bytes"], compare_legacy=False,
                    source_files=[], budget_factory=mock.Mock())
            allocator.assert_called_once_with(9.25/32)
        n.m.data.write_json(Path(os.environ["DEFAULTS_TEST_ROOT"])/"native_wiring.json",
            dict(authorization_rejects_bad_scope=True, resources_reject_low_or_occupied=True,
                 original_adapter_and_bounded_allocator_selected=True, gpu_calls_simulated=True,
                 actual_bge_loads=0, actual_native_updates=0))

    def test_sequential_deadline_and_stop_without_retry(self):
        evidence = []
        for failure in (None, "nonzero", "timeout", "invalid_result"):
            root = self.temporary()
            job, auth = root/"native", root/"authorization.json"
            row = approval(job)
            n.m.data.write_json(auth, row)
            calls = []
            def simulated(command, **options):
                name = command[command.index("--worker")+1]
                child = Path(command[command.index("--job")+1])
                deadline = float(command[command.index("--deadline")+1])
                calls.append((name, deadline))
                self.assertGreater(options["timeout"], 0)
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(command, options["timeout"])
                if failure == "nonzero":
                    return types.SimpleNamespace(returncode=1)
                result = dict(status="PASS_NATIVE_HANDWRITTEN", spec=n.m.Spec(name, "ABC", 4).identity(),
                    source_files=row["sources"], attempted_calls=3, actual_calls=3,
                    actual_next_update_equal=True, inference_equal=True, outer_checkpoint_equals_legacy=False)
                if failure == "invalid_result":
                    result["actual_calls"] = 0
                n.m.data.write_json(child/"result.json", result)
                return types.SimpleNamespace(returncode=0)
            with mock.patch.object(n, "admission", return_value={"simulated": True}), \
                    mock.patch.object(n.subprocess, "run", side_effect=simulated):
                if failure is None:
                    result = n.controller(job, auth)
                    self.assertEqual(result["status"], "COMPLETE_TWO_ORIGINAL_NATIVE_CHECKS")
                    self.assertEqual([c[0] for c in calls], list(n.METHODS))
                    self.assertEqual(calls[0][1], calls[1][1])
                else:
                    with self.assertRaises((RuntimeError, ValueError)):
                        n.controller(job, auth)
                    self.assertEqual(len(calls), 1)
                    result = n.m.data.read_json(job/"status.json")
                    self.assertEqual(result["status"], "FAILED_STOP_NO_RETRY")
                with self.assertRaisesRegex(ValueError, "overwritten"):
                    n.controller(job, auth)
                before = {str(p.relative_to(job)): p.read_bytes() for p in job.rglob("*") if p.is_file()}
                with mock.patch.object(sys, "argv", ["native", "--job", str(job), "--authorization", str(auth)]):
                    with self.assertRaisesRegex(ValueError, "overwritten"):
                        n.main()
                self.assertEqual(before, {str(p.relative_to(job)): p.read_bytes()
                                         for p in job.rglob("*") if p.is_file()})
            self.assertGreater(result["charged_seconds"], 0)
            evidence.append(dict(scenario=failure or "success", launched=len(calls),
                                 charged_seconds=result["charged_seconds"], simulated_workers=True,
                                 rejected_existing_cli_job_unchanged=True))
        n.m.data.write_json(Path(os.environ["DEFAULTS_TEST_ROOT"])/"native_control.json", evidence)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
