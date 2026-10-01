"""Run an audit command in the existing environment, pinned to one available CPU."""
import os,sys,json,subprocess,time,datetime,platform,pathlib
out=pathlib.Path('/mnt/data/ranking_result_reaudit_20260928/evidence')
name=sys.argv[1]; cmd=sys.argv[2:]; env=os.environ.copy()
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
aff=sorted(os.sched_getaffinity(0));cpu=aff[0]
os.sched_setaffinity(0,{cpu}); start=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
with (out/f'{name}.stdout.log').open('w') as stdout,(out/f'{name}.stderr.log').open('w') as stderr:
 p=subprocess.run(['/usr/bin/time','-v','-o',str(out/f'{name}.resource.log'),*cmd],cwd='/mnt/data/ranking_result_reaudit_20260928/submitted',env=env,stdout=stdout,stderr=stderr)
info={'command':cmd,'start_utc':start,'finish_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'exit_code':p.returncode,'cpu_affinity':[cpu],'python':platform.python_version(),'platform':platform.platform(),'thread_environment':{k:env[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},'new_dependencies_installed':False}
(out/f'{name}.execution.json').write_text(json.dumps(info,indent=2)+'\n');(out/f'{name}.exit.txt').write_text(str(p.returncode)+'\n')
print(json.dumps(info,indent=2));sys.exit(p.returncode)
