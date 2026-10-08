"""Record a single review-only command, including failures, without retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

record_dir = Path(sys.argv[1])
label = sys.argv[2]
command = sys.argv[3:]
record_dir.mkdir(parents=True, exist_ok=True)
record = record_dir / (label + ".execution.json")
assert not record.exists(), "Do not overwrite an earlier attempt"
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
start_utc = datetime.now(timezone.utc).isoformat()
start = time.perf_counter()
stdout_path = record_dir / (label + ".stdout.txt")
stderr_path = record_dir / (label + ".stderr.txt")
with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
    try:
        proc = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=60)
        code, timeout = proc.returncode, False
    except subprocess.TimeoutExpired:
        code, timeout = 124, True
payload = {"label": label, "command": command, "working_directory": str(Path.cwd()),
           "started_at_utc": start_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
           "wall_seconds": time.perf_counter() - start, "exit_code": code, "timeout": timeout,
           "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
           "review_environment": {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
record.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
print(stdout_path.read_text(encoding="utf-8", errors="replace"))
if stderr_path.stat().st_size:
    print(stderr_path.read_text(encoding="utf-8", errors="replace"))
sys.exit(code)
