"""Run an explicitly allowed local audit command, keeping unedited stdout/stderr."""
from pathlib import Path
import argparse,datetime,json,os,platform,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd',default='/mnt/data/bge_continual_external_audit/source');p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.cmd[1:] if a.cmd and a.cmd[0]=='--' else a.cmd
if not cmd: raise ValueError('empty command')
root=Path('/mnt/data/bge_continual_external_audit'); log=root/'logs'/a.name
if log.with_suffix('.json').exists(): raise FileExistsError(log)
env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',NUMEXPR_NUM_THREADS='1')
aff=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,{aff[0]})
start=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
with open(str(log)+'.stdout','wb') as out,open(str(log)+'.stderr','wb') as err:
 proc=subprocess.run(cmd,cwd=a.cwd,env=env,stdout=out,stderr=err)
record={'name':a.name,'command':cmd,'cwd':a.cwd,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'exit_code':proc.returncode,'affinity':sorted(os.sched_getaffinity(0)),'environment':{k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','NUMEXPR_NUM_THREADS')},'python_runner':sys.version,'platform':platform.platform()}
Path(str(log)+'.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');Path(str(log)+'.exit').write_text(str(proc.returncode)+'\n'); print(json.dumps(record,ensure_ascii=False,indent=2));sys.exit(proc.returncode)
