"""Run one explicitly supplied command, preserving streams, command, environment and exit."""
import argparse, datetime, json, os, pathlib, platform, subprocess, sys, time
p=argparse.ArgumentParser(); p.add_argument('--cwd',required=True);p.add_argument('--log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
out=pathlib.Path(a.log);out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update({k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')});env['PYTHONDONTWRITEBYTECODE']='1'
aff=sorted(os.sched_getaffinity(0)); cpu=aff[0]
def setup(): os.sched_setaffinity(0,{cpu})
t0=time.time(); start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (out/'stdout.log').open('wb') as so,(out/'stderr.log').open('wb') as se:
 r=subprocess.run(cmd,cwd=a.cwd,env=env,stdout=so,stderr=se,preexec_fn=setup)
meta={'command':cmd,'cwd':a.cwd,'started_utc':start,'elapsed_seconds':time.time()-t0,'exit_code':r.returncode,'cpu_affinity':[cpu],'thread_environment':{k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'python':sys.version,'platform':platform.platform()}
(out/'execution.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'exit_status.txt').write_text(str(r.returncode)+'\n')
print(json.dumps(meta,ensure_ascii=False));print((out/'stdout.log').read_text());print((out/'stderr.log').read_text(),file=sys.stderr)
sys.exit(r.returncode)
