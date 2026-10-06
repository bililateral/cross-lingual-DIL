"""Supervise this revision's single authorized handwritten CPU verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.monotonic()

import psutil


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence"
    output.mkdir(exist_ok=False)
    process = psutil.Process()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(process.cpu_affinity()) != 1
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))):
        raise ValueError("Requires hidden GPU and one CPU/thread, set before Python starts")
    peak, reason = 0, None
    with (output / "unittest.log").open("wb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-m", "unittest", "discover",
                                  "-s", "tests", "-p", "test_step28_relation_revision.py", "-v"],
                                 cwd=root, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            total_rss = process.memory_info().rss
            for descendant in process.children(recursive=True):
                try:
                    total_rss += descendant.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, total_rss)
            size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            if time.monotonic()-STARTED > 585 or peak > 2*2**30 or size > 15*2**20:
                reason = "time, RSS, or evidence resource limit"
                child.kill()
                break
            time.sleep(.1)
        code = child.wait()
    record = {"status": "PASS" if code == 0 and reason is None else "FAIL",
              "exit_code": code, "limit_failure": reason,
              "elapsed_seconds": time.monotonic()-STARTED,
              "peak_process_tree_rss_bytes": peak, "affinity": process.cpu_affinity(),
              "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "python": sys.version, "formal_data_access": False,
              "native_bge_loaded": False, "retry": False}
    (output / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    raise SystemExit(0 if record["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
