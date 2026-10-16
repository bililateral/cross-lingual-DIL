"""Reproducible bounded reviewer-only launcher; capture even failures, no auto retry."""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, platform, subprocess, sys, time
here=pathlib.Path(__file__).resolve().parent
script=here/(sys.argv[1] if len(sys.argv)>1 else 'reviewer_probes.py')
name=script.stem
out=here/(name+'_execution')
out.mkdir(exist_ok=False)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
affinity=None
if hasattr(os,'sched_getaffinity'):
    affinity=min(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{affinity})
receipt={'scope':'Isolated ChatGPT reviewer CPU, NOT project Linux/native qualification; no project budget charge',
 'command':[sys.executable,str(script)],'python':sys.version,'platform':platform.platform(),
 'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
 'cpu_affinity':affinity,'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':44}
t=time.monotonic()
with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
    try:
        r=subprocess.run(receipt['command'],cwd=here,env=env,stdout=stdout,stderr=stderr,timeout=44,check=False)
        receipt['exit_code']=r.returncode
    except subprocess.TimeoutExpired:
        receipt['exit_code']=None;receipt['timeout']=True
receipt['elapsed_seconds']=time.monotonic()-t
receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'execution.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print((out/'stdout.txt').read_text())
print((out/'stderr.txt').read_text())
