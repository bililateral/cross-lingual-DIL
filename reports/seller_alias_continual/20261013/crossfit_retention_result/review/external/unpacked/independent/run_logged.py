#!/usr/bin/env python3
"""Run only this review's saved-result reference and preserve each attempt."""
from pathlib import Path
import datetime, hashlib, json, os, platform, subprocess, sys, time

base=Path(__file__).resolve().parents[1]
attempts=base/'independent'/'attempts'
attempts.mkdir(exist_ok=True)
attempt=attempts/f'run{len(list(attempts.glob("run*")))+1:02}'
attempt.mkdir(exist_ok=False)
source=base/'independent'/'recompute.py'
(attempt/'recompute_snapshot.py').write_bytes(source.read_bytes())
cmd=[sys.executable,'-u',str(source),'--input',str(base/'input'),'--output',str(attempt/'output')]
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    env[key]='1'
env['CUDA_VISIBLE_DEVICES']=''
started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with (attempt/'stdout.log').open('w') as out,(attempt/'stderr.log').open('w') as err:
    result=subprocess.run(cmd,cwd=base,env=env,stdout=out,stderr=err,check=False)
record={'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'command_argv':cmd,'cwd':str(base),'exit_code':result.returncode,'wall_seconds':time.perf_counter()-t,
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'platform':platform.platform(),
 'env_overrides':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
 'scope':'web workspace; independent saved-result reference only; no torch or project code execution'}
(attempt/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
print((attempt/'stdout.log').read_text())
print((attempt/'stderr.log').read_text())
sys.exit(result.returncode)
