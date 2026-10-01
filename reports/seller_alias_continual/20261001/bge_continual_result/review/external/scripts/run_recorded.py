#!/usr/bin/env python3
"""Execute an explicit command, preserving raw stdout/stderr and metadata."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys, time
p=argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--cwd',default='/mnt/data'); p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
cmd=a.command
if cmd and cmd[0]=='--': cmd=cmd[1:]
base=pathlib.Path(__file__).resolve().parents[1]; logs=base/'logs'; logs.mkdir(exist_ok=True)
assert cmd, 'explicit command required'
assert not (logs/(a.name+'.json')).exists(), 'log name already used; preserve prior attempts'
t0=time.perf_counter(); start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (logs/(a.name+'.stdout')).open('wb') as out,(logs/(a.name+'.stderr')).open('wb') as err:
 try:
  r=subprocess.run(cmd,cwd=a.cwd,stdout=out,stderr=err,check=False,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  code=r.returncode
 except BaseException as e:
  err.write((repr(e)+'\n').encode()); code=127
meta={'command':cmd,'cwd':a.cwd,'started_at_utc':start,'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-t0,'exit_code':code}
for suffix in ['stdout','stderr']:
 b=(logs/(a.name+'.'+suffix)).read_bytes();meta[suffix]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(logs/(a.name+'.exit')).write_text(str(code)+'\n');(logs/(a.name+'.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(meta,indent=2,ensure_ascii=False));print('--- stdout ---');print((logs/(a.name+'.stdout')).read_text(errors='replace'));print('--- stderr ---');print((logs/(a.name+'.stderr')).read_text(errors='replace'))
sys.exit(code)
