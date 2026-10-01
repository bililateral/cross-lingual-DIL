"""Linux single-CPU test contracts with handwritten inputs, no formal payloads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

import step28_alias_test_run as runner


def run(out: Path) -> dict:
    if platform.system() != "Linux" or len(os.sched_getaffinity(0)) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Existing Linux,oneCPU,no visibleGPU required")
    out.mkdir(parents=True, exist_ok=False)
    started, snapshot = time.monotonic(), runner.sources()
    sys.path.insert(0, str(runner.data.ROOT / "tests"))
    import test_step28_alias_test_contracts as contracts
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(contracts))
    evidence = contracts.TestContracts.pipeline_evidence
    runner.data.write_json(out / "handmade.json", evidence)
    success = result.wasSuccessful() and not result.skipped and evidence.get("complete_matrices") == 4 and evidence.get("actual_tiny_restore_forward") is True
    report = {"status": "PASS_TEST_HANDMADE_AUDIT" if success else "FAILED_TEST_HANDMADE_AUDIT",
              "source_files": snapshot, "policy_sha256": runner.data.sha256(runner.POLICY),
              "contracts": {"run": result.testsRun, "passed": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
                            "errors": len(result.errors), "failures": len(result.failures), "skipped": len(result.skipped)},
              "handmade": runner.data.record(out / "handmade.json", out), "seconds": time.monotonic() - started,
              "formal_label_reads": 0, "formal_text_reads": 0, "formal_score_reads": 0,
              "native_model_loads": 0, "encoder_updates": 0, "gpu_used": False,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                              "threads": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}},
              "scope": "Actual handmade CSV/pipeline/bootstrap and tiny tensor checkpoint inference; unchanged BGE native evidence reused,not formal test performance."}
    if runner.sources() != snapshot:
        raise ValueError("Sources changed during CPU verification")
    runner.data.write_json(out / "audit.json", report)
    if report["seconds"] > 3600 or sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > 268435456:
        raise RuntimeError("CPU evidence budget exceeded")
    if not success:
        raise RuntimeError("Test contract verification failed; preserve actual logs")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out.resolve())
    print(runner.data.json_bytes({k: result[k] for k in ("status", "contracts", "seconds")}).decode())
