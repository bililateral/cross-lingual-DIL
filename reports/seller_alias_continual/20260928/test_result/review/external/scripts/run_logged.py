#!/usr/bin/env python3
"""Run a permitted local audit command; retain unedited process streams and status."""
import argparse,json,os,pathlib,subprocess,sys,time,datetime,platform,hashlib
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd',required=True);p.add_argument('--logs',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
out=pathlib.Path(a.logs)/a.name;out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update({x:'1' for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')});env['PYTHONDONTWRITEBYTECODE']='1'
allowed=os.sched_getaffinity(0);cpu=min(allowed);os.sched_setaffinity(0,{cpu})
started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
(out/'command.json').write_text(json.dumps({'argv':cmd,'cwd':a.cwd,'thread_environment':{x:env[x] for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},'cpu_affinity':[cpu],'started_utc':started,'python':sys.version,'platform':platform.platform()},ensure_ascii=False,indent=2)+'\n')
with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
 r=subprocess.run(cmd,cwd=a.cwd,env=env,stdout=so,stderr=se,check=False)
meta={'exit_code':r.returncode,'elapsed_seconds':time.perf_counter()-t,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'status.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'exit_code.txt').write_text(str(r.returncode)+'\n')
print(json.dumps({'log_directory':str(out),**meta},ensure_ascii=False));print((out/'stdout.log').read_text(errors='replace'));print((out/'stderr.log').read_text(errors='replace'),file=sys.stderr)
sys.exit(r.returncode)
