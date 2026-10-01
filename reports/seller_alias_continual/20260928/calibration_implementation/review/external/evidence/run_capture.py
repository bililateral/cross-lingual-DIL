from pathlib import Path
import os,sys,subprocess,json,time,datetime,platform,importlib.metadata
E=Path(__file__).resolve().parent; SRC=E.parent/'submission'
name=sys.argv[1]; command=sys.argv[2:]
out=E/name; out.mkdir(exist_ok=False)
cpu=min(os.sched_getaffinity(0)); env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
full=['/usr/bin/time','-v','-o',str(out/'resource_usage.log'),'taskset','-c',str(cpu),*command]
rec={'command':full,'cwd':str(SRC),'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE')},'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy')},'cpu_affinity':[cpu],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'command.json').write_text(json.dumps(rec,indent=2)); start=time.monotonic()
with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
 r=subprocess.run(full,cwd=SRC,env=env,stdout=so,stderr=se,timeout=120)
rec.update(exit_code=r.returncode,wall_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'execution.json').write_text(json.dumps(rec,indent=2)); (out/'exit_status.txt').write_text(str(r.returncode)+'\n')
print(json.dumps(rec,indent=2)); print((out/'stdout.log').read_text()); print((out/'stderr.log').read_text()); sys.exit(r.returncode)
