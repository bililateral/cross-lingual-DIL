"""One bounded handwritten CPU attempt; total 300s includes failed attempts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--prior", type=int, required=True)
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.platform != "linux" or not 0 <= args.prior < 300 or args.job.exists():
        raise ValueError("Linux, fresh attempt and remaining approved CPU budget required")
    manifest = json.loads((root/"cpu_sources.json").read_text())
    for row in manifest:
        content = (root/row["path"]).read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Synchronized source identity differs")
    args.job.mkdir(parents=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        VENDORLINK_TEST_ROOT=str(args.job), PYTHONPATH=str(root/"scripts")+":"+str(root/"tests"),
        TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
    command = [sys.executable, "-B", str(root/"tests/test_step29_vendorlink.py"), *args.tests]
    started, peak, reason = time.monotonic(), 0, None
    with (args.job/"console.txt").open("w") as output:
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while child.poll() is None:
            try:
                process = psutil.Process(child.pid)
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
                peak = max(peak, rss)
                size = sum(p.stat().st_size for p in args.job.rglob("*") if p.is_file())
                if time.monotonic()-started >= 300-args.prior-1:
                    reason = "CPU_TIME_LIMIT_TAIL_RESERVED"
                elif rss > 4*2**30 or size > 64*2**20:
                    reason = "CPU_MEMORY_OR_OUTPUT_LIMIT"
                if reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
        code = child.wait()
    elapsed = time.monotonic()-started
    row = dict(exit_code=code, limit_reason=reason, wall_seconds=elapsed, charged_seconds=math.ceil(elapsed),
        cumulative_seconds=args.prior+math.ceil(elapsed), peak_rss_bytes=peak, python=sys.executable,
        sources=manifest, formal_data_reads=0, pretrained_model_loads=0, gpu=False)
    (args.job/"receipt.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in row.items() if k != "sources"}), flush=True)
    raise SystemExit(code if code else (1 if reason else 0))


if __name__ == "__main__":
    main()
