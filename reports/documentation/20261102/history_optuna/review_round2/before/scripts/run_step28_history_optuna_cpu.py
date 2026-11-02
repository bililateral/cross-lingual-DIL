"""Bounded Linux-only delta checks; source receipt includes unsuccessful attempts."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

STARTED = time.monotonic()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-seconds", type=int, required=True)
    args = parser.parse_args()
    if (sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or args.output.exists() or not 0 <= args.prior_seconds < 1200):
        raise ValueError("Fresh, bounded Linux CPU qualification only")
    import step28_history_optuna_run as run
    args.output.mkdir()
    source_files = run.sources()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(run.data.ROOT/"scripts")+":"+str(run.data.ROOT/"tests"),
        "HISTORY_TEST_ROOT": str(args.output)}
    command = [sys.executable, "-B", str(run.data.ROOT/"tests/test_step28_history_optuna.py"), "-v"]
    peak, stopped, child = 0, None, None
    with (args.output/"stdout.txt").open("x") as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            owned = run.inherited.owned_processes(child.pid)
            rss = run.inherited.resident_bytes(owned)
            peak = max(peak, rss)
            if time.monotonic()-STARTED >= 1200-args.prior_seconds-30 or rss > 4*2**30:
                stopped = "CPU_TIME_OR_RSS_LIMIT"
                for process in owned.values():
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, live = psutil.wait_procs(list(owned.values()), timeout=5)
                for process in live:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.wait(timeout=10)
                break
            time.sleep(.1)
    same_source = run.sources() == source_files
    import optuna
    import torch
    elapsed = time.monotonic()-STARTED
    charged = math.ceil(elapsed)
    status = "PASS" if child.returncode == 0 and not stopped and same_source and args.prior_seconds+charged <= 1200 else "FAIL"
    run.data.write_json(args.output/"receipt.json", dict(status=status, sources=source_files, command=command,
        executable=sys.executable, server_epoch=time.time(), actual_seconds=elapsed, charged_seconds=charged,
        prior_seconds=args.prior_seconds, total_seconds=args.prior_seconds+charged, limit_seconds=1200,
        peak_rss_bytes=peak, exit_code=child.returncode, stopped=stopped, sources_unchanged=same_source,
        torch_version=torch.__version__, optuna_version=optuna.__version__, cuda_visible_devices="",
        scope="Handwritten CPU data/TinyEncoder; real Optuna and subprocesses. No native model/formal data/GPU/confirmation generation."))
    print(status, args.prior_seconds+charged, flush=True)
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
