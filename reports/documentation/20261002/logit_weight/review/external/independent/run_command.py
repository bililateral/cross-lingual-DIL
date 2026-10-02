"""Run a bounded local review command, preserving raw stdout/stderr and metadata."""
from __future__ import annotations
import datetime, hashlib, json, os, pathlib, shlex, subprocess, sys, time
ROOT=pathlib.Path('/mnt/data/logit_audit')
name, *command=sys.argv[1:]
if not name or not command: raise SystemExit('name and command required')
out=ROOT/'evidence'/name
out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env.update(PYTHONPATH='scripts:tests', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
start=datetime.datetime.now(datetime.timezone.utc); t=time.monotonic()
meta={'command_argv':command,'command_shell_display':shlex.join(command),'cwd':str(ROOT/'submission'),'start_utc':start.isoformat(),'environment_overrides':{k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES']}}
try:
 with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
  p=subprocess.run(command,cwd=ROOT/'submission',env=env,stdout=stdout,stderr=stderr,timeout=180)
 meta.update(exit_code=p.returncode,timed_out=False)
except subprocess.TimeoutExpired:
 meta.update(exit_code=None,timed_out=True)
except Exception as e:
 meta.update(exit_code=None,runner_exception=repr(e))
meta.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-t)
for f in ['stdout.txt','stderr.txt']:
 b=(out/f).read_bytes();meta[f]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(out/'command.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(meta,ensure_ascii=False,indent=2))
print('--- stdout ---'); print((out/'stdout.txt').read_text(errors='replace'))
print('--- stderr ---'); print((out/'stderr.txt').read_text(errors='replace'))
sys.exit(meta.get('exit_code') or (124 if meta.get('timed_out') else 0))
