"""Capture allowed single-CPU handmade audit subprocesses without changing source."""
from pathlib import Path
import os,sys,subprocess,json,time,platform,hashlib,datetime
BASE=Path('/mnt/data/test_pretest_audit_20260928')
name=sys.argv[1]; cmd=sys.argv[2:]; cpu=min(os.sched_getaffinity(0))
out=BASE/'evidence/logs'/name; out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
actual=['taskset','-c',str(cpu),*cmd]; start=time.monotonic()
record={'command':cmd,'actual_command':actual,'cwd':str(BASE/'source'),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment_overrides':{k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE')},'cpu':cpu}
(out/'command.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
 result=subprocess.run(actual,cwd=BASE/'source',env=env,stdout=stdout,stderr=stderr)
record.update(exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)); (out/'exit_status.txt').write_text(str(result.returncode)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2)); print((out/'stdout.log').read_text()); print((out/'stderr.log').read_text())
sys.exit(result.returncode)
