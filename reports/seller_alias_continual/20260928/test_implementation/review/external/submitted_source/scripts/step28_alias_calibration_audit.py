"""Linux single-CPU actual scalar fits and handmade contracts; no formal labels."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np
import scipy

import step28_alias_calibration_run as runner


def run(out: Path) -> dict:
    if platform.system() != "Linux" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Linux and one CPU affinity required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_files = runner.sources()
    sys.path.insert(0, str(runner.data.ROOT / "tests"))
    import test_step28_alias_calibration_contracts as contracts
    suite = unittest.defaultTestLoader.loadTestsFromModule(contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    native = contracts.CalibrationContracts.pipeline_evidence
    runner.data.write_json(out / "handmade.json", native)
    success = result.wasSuccessful() and not result.skipped and len(native.get("actual_fits", {})) == 6
    report = {"status": "PASS_CALIBRATION_HANDMADE_AUDIT" if success else "FAILED_CALIBRATION_HANDMADE_AUDIT",
              "source_files": source_files, "policy_sha256": runner.data.sha256(runner.POLICY),
              "contracts": {"run": result.testsRun, "passed": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
                            "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped)},
              "handmade": runner.data.record(out / "handmade.json", out),
              "native_scalar_fits_in_pipeline": len(native.get("actual_fits", {})),
              "formal_label_reads": 0, "formal_text_reads": 0, "formal_score_arrays_read": 0,
              "model_loads": 0, "encoder_updates": 0, "gpu_used": False,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                              "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
              "seconds": time.monotonic() - started,
              "scope": "Handmade probability fitting, CSV alignment, ranking invariance, full saved pipeline and failure boundaries; not formal performance."}
    if runner.sources() != source_files:
        raise ValueError("Audit sources changed during execution")
    runner.data.write_json(out / "audit.json", report)
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if time.monotonic() - started > 1800 or size > 134217728:
        raise RuntimeError("CPU audit budget exceeded")
    if not success:
        raise RuntimeError("Handmade calibration verification failed; inspect actual logs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out)
    print(runner.data.json_bytes({"status": report["status"], "contracts": report["contracts"],
                                  "seconds": report["seconds"]}).decode())


if __name__ == "__main__":
    main()
