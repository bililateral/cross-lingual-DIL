"""Run a local audit command and persist exact argv/times/env/exit/stdout/stderr."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, pathlib, platform, subprocess, sys, time
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd',default='/mnt/data/er_review_input');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
out=pathlib.Path('/mnt/data/er_review_evidence/logs')/a.name;out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
 proc=subprocess.run(cmd,cwd=a.cwd,env=env,stdout=so,stderr=se,check=False)
meta={'command':cmd,'cwd':a.cwd,'started_at_utc':start,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'exit_code':proc.returncode,'python':sys.version,'platform':platform.platform(),'parent_allowed_cpu':sorted(os.sched_getaffinity(0)),'threads':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},'origin':'CURRENT_WEB_CONTAINER_NOT_PROJECT_LINUX'}
(out/'execution.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(meta,ensure_ascii=False,indent=2));print('---stdout---');print((out/'stdout.log').read_text());print('---stderr---');print((out/'stderr.log').read_text());sys.exit(proc.returncode)
