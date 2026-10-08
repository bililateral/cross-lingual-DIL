"""Preserve exact command, stdout, stderr, exit status, and elapsed wall time."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

folder = Path(sys.argv[1]).resolve()
folder.mkdir(parents=True, exist_ok=False)
command = sys.argv[2:]
started = time.time()
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with (folder / "stdout.txt").open("w", encoding="utf-8") as stdout, (folder / "stderr.txt").open("w", encoding="utf-8") as stderr:
    result = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=False)
record = {"command": command, "cwd": os.getcwd(), "start_unix_seconds": started,
          "elapsed_wall_seconds": time.time()-started, "exit_code": result.returncode,
          "scope": "Current webpage workspace; attachment evidence and handmade examples only; not project native verification."}
(folder / "execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False))
print((folder / "stdout.txt").read_text())
print((folder / "stderr.txt").read_text())
sys.exit(result.returncode)
