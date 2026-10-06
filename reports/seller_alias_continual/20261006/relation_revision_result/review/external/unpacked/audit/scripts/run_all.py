"""Replay the five independent audits. Requires existing NumPy and CPU Torch.
Does not install packages, connect to the project Linux host, train, load models,
parse formal labels, or invoke project execution/verification entry points.
"""
from pathlib import Path
import subprocess,sys,os,time,json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audit/outputs'; OUT.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
names=['01_identity.py','02_saved_statistics.py','03_execution_evidence.py','04_diagnostics.py','05_report_crosscheck.py']
records=[]
for name in names:
    start=time.monotonic(); log=OUT/(name.removesuffix('.py')+'.stdout.txt')
    with log.open('w') as stream:
        rc=subprocess.run([sys.executable,str(ROOT/'audit/scripts'/name)],stdout=stream,stderr=subprocess.STDOUT,env=env).returncode
    records.append({'script':name,'exit_code':rc,'wall_seconds':time.monotonic()-start,'log':str(log.relative_to(ROOT))})
    (OUT/'replay_run.json').write_text(json.dumps({'scope':'Web local independent saved-result audits; no project Linux or model execution.','runs':records},ensure_ascii=False,indent=2)+'\n')
    print(name,rc,flush=True)
    if rc: sys.exit(rc)
