"""Bounded Linux-only, handwritten orchestration checks for this deployment."""
from pathlib import Path
import ast
import json
import math
import os
import signal
import subprocess
import sys
import time
import psutil

root = Path(__file__).resolve().parent
workspace = root/'workspace'
sys.path.insert(0, str(workspace/'scripts'))
import step28_comparison_finish as c

pending = [workspace/'scripts/step28_comparison_finish.py']
seen = set()
while pending:
    path = pending.pop()
    if path in seen:
        continue
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                 [node.module] if isinstance(node, ast.ImportFrom) else [])
        pending.extend(workspace/'scripts'/(name+'.py') for name in names
                       if name and name.startswith('step28_'))
science = c.run.sources()
seen -= {workspace/r['path'] for r in science}
seen.update(workspace/p for p in ('tests/test_step28_comparison_finish.py',
                                  'run_step28_comparison_linux_20261031.sh'))
orchestration = [c.data.record(p, workspace) for p in sorted(seen)]
job = root/'cpu01'
job.mkdir()
command = [sys.executable, '-B', '-m', 'unittest', '-v', 'test_step28_comparison_finish',
           'test_step28_continual_comparison.ComparisonTests.test_full_gate_accepts_only_pinned_reused_completion']
env = dict(os.environ, PYTHONPATH=str(workspace/'scripts')+':'+str(workspace/'tests'),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
start = time.monotonic()
peak, reason = 0, None
with (job/'console.txt').open('w') as log:
    child = subprocess.Popen(command, cwd=workspace, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    while child.poll() is None:
        try:
            process = psutil.Process(child.pid)
            peak = max(peak, sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]))
            if time.monotonic()-start >= 180 or peak > 4*2**30:
                reason = 'TIME_OR_RSS_LIMIT'
                os.killpg(child.pid, signal.SIGKILL)
                break
        except psutil.NoSuchProcess:
            pass
        time.sleep(.1)
    code = child.wait()
wall = time.monotonic()-start
row = dict(success=code == 0, exit_code=code, limit_reason=reason,
           wall_seconds=wall, charged_seconds=math.ceil(wall),
           cumulative_seconds=104+math.ceil(wall), peak_rss_bytes=peak,
           sources=science, orchestration_sources=orchestration,
           runner=c.data.record(Path(__file__), root), command=command,
           formal_data_reads=0, model_loads=0, gpu=False)
c.shared.write(job/'receipt.json', row)
print(json.dumps({k:v for k,v in row.items() if k not in ('sources','orchestration_sources')}, ensure_ascii=False))
raise SystemExit(code)
