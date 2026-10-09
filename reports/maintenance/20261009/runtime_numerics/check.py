"""Linux CPU maintenance check only; no formal data, pretrained model or GPU work."""
import io
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

started = time.monotonic()
root = Path(__file__).resolve().parents[4]
out = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "tests"), str(root / "scripts")]
os.environ["RECORD_REPLAY_TEST_ROOT"] = str(out)
import torch
import test_step28_record_replay as tests

assert os.name == "posix" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert len(os.sched_getaffinity(0)) == 1
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
numerics = tests.method.parent.configure_numerics()
suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.NumericsTests)
for name in ("test_chunked_vjp_every_parameter_and_actual_adam_match_direct_autograd",
             "test_augmented_constraint_changes_source_gradient_in_original_nullspace",
             "test_full_checkpoint_memory_rng_restore_reproduces_next_real_update"):
    suite.addTest(tests.RecordReplayTests(name))
with (out / "unittest.txt").open("w", encoding="utf-8") as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)

# Removing the real entry call must be detected, even if a receipt-shaped dict is returned.
from unittest import mock
with mock.patch.object(tests.method.parent, "configure_numerics", return_value=numerics):
    stream = io.StringIO()
    mutant = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.TestSuite([tests.NumericsTests(name) for name in (
            "test_formal_entry_applies_and_records_before_training",
            "test_native_entry_applies_before_model_loading")]))
(out / "missing_operation_counterexample.txt").write_text(stream.getvalue(), encoding="utf-8")
passed = result.wasSuccessful() and not result.skipped and len(mutant.failures) == 2 and not mutant.errors
report = {"status": "PASS" if passed else "FAIL", "python": sys.version,
          "torch": torch.__version__, "cuda_build": torch.version.cuda,
          "cuda_initialized": torch.cuda.is_initialized(), "numerics": numerics,
          "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
          "skipped": len(result.skipped), "missing_operation_detected": len(mutant.failures),
          "elapsed_seconds": time.monotonic() - started,
          "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "formal_inputs": False, "formal_labels": False, "pretrained_model_loads": 0,
          "qualification": "CPU maintenance only; not a new GPU/formal-run qualification"}
(out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if passed else 1)
