#!/usr/bin/env python3
"""Retain exact execution identity, stdout/stderr, failures and wall times."""
import os, sys, json, time, hashlib, subprocess
from datetime import datetime,timezone
from pathlib import Path
p=Path(__file__).resolve().parent
logdir=p/'attempt01'
logdir.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
ledger={'scope':'Reviewer-isolated handwritten mathematics. Never a project Linux/native qualification.',
        'automatic_retries':0,'runs':[]}
def identity(path):
    return {'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for file in ('make_inputs.py','math_checks.py'):
    command=[sys.executable,str(p/file)]
    before=[identity(p/file)]
    if file=='math_checks.py':before.append(identity(p/'inputs.json'))
    start=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
    r=subprocess.run(command,cwd=p,env=env,capture_output=True,timeout=40,check=False)
    elapsed=time.perf_counter()-timer
    (logdir/(file+'.stdout.txt')).write_bytes(r.stdout)
    (logdir/(file+'.stderr.txt')).write_bytes(r.stderr)
    ledger['runs'].append({'command':command,'start_utc':start,'elapsed_seconds':elapsed,
            'exit_code':r.returncode,'inputs':before,
            'stdout':identity(logdir/(file+'.stdout.txt')),'stderr':identity(logdir/(file+'.stderr.txt'))})
    (logdir/'execution.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(r.stdout.decode('utf-8'));print(r.stderr.decode('utf-8'),file=sys.stderr)
    if r.returncode:sys.exit(r.returncode)
