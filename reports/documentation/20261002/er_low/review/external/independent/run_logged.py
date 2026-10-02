#!/usr/bin/env python3
"""Run explicit local commands; retain separate streams and non-overwriting receipts."""
import argparse,datetime,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd',default='/mnt/data/er_low_source');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
root=pathlib.Path('/mnt/data/er_low_audit/logs');root.mkdir(exist_ok=True)
assert cmd and not (root/(a.name+'.json')).exists(), 'Do not overwrite a prior run'
env=os.environ.copy();env.update(PYTHONPATH='scripts:tests:/mnt/data/er_low_audit/independent',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
cpu=sorted(os.sched_getaffinity(0))[0];start=datetime.datetime.now(datetime.timezone.utc);mono=time.monotonic()
with (root/(a.name+'.stdout.log')).open('wb') as out,(root/(a.name+'.stderr.log')).open('wb') as err:
 r=subprocess.run(cmd,cwd=a.cwd,env=env,stdout=out,stderr=err,preexec_fn=lambda:os.sched_setaffinity(0,{cpu}))
end=datetime.datetime.now(datetime.timezone.utc)
rec=dict(name=a.name,command=cmd,cwd=a.cwd,start_utc=start.isoformat(),end_utc=end.isoformat(),elapsed_seconds=time.monotonic()-mono,exit_code=r.returncode,cpu_affinity=[cpu],environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')})
(root/(a.name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2));print('STDOUT\n'+(root/(a.name+'.stdout.log')).read_text(errors='replace'));print('STDERR\n'+(root/(a.name+'.stderr.log')).read_text(errors='replace'));raise SystemExit(r.returncode)
