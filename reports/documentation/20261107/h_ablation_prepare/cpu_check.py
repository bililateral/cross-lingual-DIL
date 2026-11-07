"""One bounded CPU qualification invocation; no formal/GPU entry."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("tests", nargs="*")
args = parser.parse_args()
if sys.platform != "linux" or sys.executable != "/home/yongpeng/miniconda3/envs/py310/bin/python":
    raise RuntimeError("Existing Linux py310 only")
start = time.monotonic()
prior = sum(json.loads(p.read_text())["charged_seconds"] for p in args.out.parent.glob("cpu*/receipt.json"))
remaining = 1200-prior
if remaining <= 10: raise TimeoutError("CPU qualification allowance exhausted")
args.out.mkdir()
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
       "PYTHONPATH": str(args.workspace/"scripts")+":"+str(args.workspace/"tests"),
       "ABLATION_TEST_ROOT": str(args.out)}
command = [sys.executable, str(args.workspace/"tests/test_step28_h_ablation.py"), "-v", *args.tests]
with (args.out/"stdout.txt").open("x") as stream:
    try:
        completed = subprocess.run(command, env=env, cwd=args.workspace, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=remaining-5)
        code, error = completed.returncode, None
    except BaseException as e:
        code, error = -1, repr(e)
elapsed = time.monotonic()-start
sys.path.insert(0, str(args.workspace/"scripts"))
import step28_h_ablation_run as run
test = args.workspace/"tests/test_step28_h_ablation.py"
result = dict(status="PASS" if code == 0 else "FAIL", returncode=code, error=error,
    charged_seconds=math.ceil(elapsed), elapsed_seconds=elapsed, prior_seconds=prior,
    cumulative_seconds=prior+math.ceil(elapsed), limit_seconds=1200, sources=run.sources(),
    test_source=dict(path=str(test), bytes=test.stat().st_size, sha256=hashlib.sha256(test.read_bytes()).hexdigest()),
    command=command, cpu_threads=1, gpu=False, formal_data=False, formal_training=False)
(args.out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("sources","test_source","command")}), flush=True)
raise SystemExit(code)
