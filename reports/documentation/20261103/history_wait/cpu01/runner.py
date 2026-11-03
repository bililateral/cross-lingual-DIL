"""One bounded 60-second Linux CPU delta attempt, without formal corpus links."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

START = time.monotonic()
BASE = Path(__file__).resolve().parent
WORK = BASE/"workspace"
OUT = BASE/"cpu01"
OUT.mkdir()
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
       "PYTHONPATH": str(WORK/"scripts")+":"+str(WORK/"tests")}
command = [sys.executable, "-B", str(WORK/"tests/test_step28_history_wait.py"), "-v"]
peak, stopped = 0, None
with (OUT/"stdout.txt").open("x") as log:
    child = subprocess.Popen(command, env=env, cwd=WORK, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    while child.poll() is None:
        try:
            members = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
            rss = sum(p.memory_info().rss for p in members if p.is_running())
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            pass
        if time.monotonic()-START > 50 or peak > 4*2**30:
            stopped = "CPU_TIME_OR_RSS_LIMIT"
            os.killpg(child.pid, signal.SIGKILL)
            break
        time.sleep(.05)
    child.wait()
sys.path.insert(0, str(WORK/"scripts"))
import step28_history_optuna_run as run
sources = run.sources()
delta = [run.data.record(WORK/name, WORK) for name in
         ("scripts/step28_history_wait.py", "tests/test_step28_history_wait.py")]
elapsed = time.monotonic()-START
status = "PASS" if child.returncode == 0 and stopped is None and elapsed < 58 else "FAIL"
run.data.write_json(OUT/"receipt.json", dict(status=status, sources=sources, delta_sources=delta,
    actual_seconds=elapsed, charged_seconds=math.ceil(elapsed)+2, conservative_tail_seconds=2,
    allowance_seconds=60, peak_rss_bytes=peak, exit_code=child.returncode, stopped=stopped,
    command=command, server_epoch=time.time(), gpu_calls=0, formal_data_reads=0,
    scope="Handwritten whole-chain receipts, mocked resource boundary, production dispatch with ordinary CPU child and real update callback; actual waiter execute with handwritten gate and substituted formal action. No model/data/GPU execution."))
print(status, math.ceil(elapsed)+2, flush=True)
raise SystemExit(0 if status == "PASS" else 1)
