"""Replay only the web-side attachment/math checks; no project modules or training."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audit/outputs';OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
records=[]
for name in ('01_identity.py','02_mathematics.py','03_source_and_mode.py'):
    started=time.monotonic()
    p=subprocess.run([sys.executable,str(ROOT/'audit/scripts'/name)],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
    log=OUT/(name.removesuffix('.py')+'.replay.stdout.txt');log.write_text(p.stdout)
    records.append({'script':'audit/scripts/'+name,'exit_code':p.returncode,'wall_seconds':time.monotonic()-started,'log':str(log.relative_to(ROOT))})
    (OUT/'replay_execution.json').write_text(json.dumps({'kind':'web_only_replay','project_execution':False,'runs':records},ensure_ascii=False,indent=2)+'\n')
    if p.returncode:raise SystemExit(p.returncode)
print(json.dumps(records,ensure_ascii=False,indent=2))
