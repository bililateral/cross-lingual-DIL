"""Run an explicitly supplied local review command and retain exact raw evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--id', required=True)
p.add_argument('--cwd', required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
root = Path(__file__).resolve().parents[1]
logs = root / 'execution' / a.id
logs.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
overrides = {
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
    'MPLCONFIGDIR': str(root / 'temporary' / 'matplotlib'),
}
env.update(overrides)
start = time.perf_counter()
record = {
    'scope': 'Isolated reviewer workspace only; not project Linux execution',
    'command_argv': command, 'cwd': a.cwd, 'environment_overrides': overrides,
    'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'launcher_python': sys.version, 'platform': platform.platform(),
    'timeout_seconds': 60,
}
with (logs / 'stdout.txt').open('wb') as stdout, (logs / 'stderr.txt').open('wb') as stderr:
    try:
        result = subprocess.run(command, cwd=a.cwd, env=env, stdout=stdout,
                                stderr=stderr, timeout=60, check=False)
        record['exit_code'] = result.returncode
    except subprocess.TimeoutExpired:
        record['exit_code'] = None
        record['failure'] = 'TimeoutExpired after 60 seconds'
record['elapsed_seconds'] = time.perf_counter() - start
record['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
record['child_user_seconds'] = usage.ru_utime
record['child_system_seconds'] = usage.ru_stime
record['child_peak_rss_kib_linux'] = usage.ru_maxrss
for name in ('stdout.txt', 'stderr.txt'):
    b = (logs / name).read_bytes()
    record[name] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
(logs / 'execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False, indent=2))
for name in ('stdout.txt', 'stderr.txt'):
    print(name + ':')
    print((logs / name).read_text(errors='replace'))
sys.exit(record['exit_code'] if record['exit_code'] is not None else 124)
