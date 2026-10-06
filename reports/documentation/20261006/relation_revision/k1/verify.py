"""One authorized K1 CPU run; no formal inputs, GPU, or automatic retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

start = time.monotonic()
root = Path(__file__).resolve().parent
output = root / "evidence"
output.mkdir(exist_ok=False)
process = psutil.Process()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert len(process.cpu_affinity()) == 1
assert all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
peak, reason = 0, None
with (output / "unittest.log").open("wb") as log:
    child = subprocess.Popen([sys.executable, "-B", "-m", "unittest",
        "test_step28_relation_revision.RevisionTests.test_serial_kernel_matches_joint_and_both_history_paths_are_live", "-v"],
        cwd=root / "tests", stdout=log, stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            rss = process.memory_info().rss
            for p in process.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            if time.monotonic()-start > 110 or peak > 2*2**30 or size > 3*2**20:
                reason = "time/RSS/evidence limit"
                break
            time.sleep(.1)
    finally:
        if child.poll() is None:
            child.kill()
        code = child.wait()
record = {"status": "PASS" if code == 0 and reason is None else "FAIL",
    "exit_code": code, "limit_failure": reason, "elapsed_seconds": time.monotonic()-start,
    "peak_process_tree_rss_bytes": peak, "affinity": process.cpu_affinity(),
    "python": sys.version, "formal_data_access": False, "native_bge_loaded": False,
    "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "retry": False}
(output / "result.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(json.dumps(record), flush=True)
raise SystemExit(0 if record["status"] == "PASS" else 1)
