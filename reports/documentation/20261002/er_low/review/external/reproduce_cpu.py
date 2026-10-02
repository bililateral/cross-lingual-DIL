#!/usr/bin/env python3
"""Reproduce only the completed local CPU scope, using a new destination.
Requires the user's own installed compatible Python/NumPy/Torch; installs nothing.
Never calls native BGE, remote services, or the formal execute entry.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,pathlib,subprocess,sys,zipfile

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=pathlib.Path,required=True);a=p.parse_args()
    base=pathlib.Path(__file__).resolve().parent;out=a.workdir.resolve()
    if out.exists():p.error('Use a new work directory; evidence will not be overwritten.')
    original=base/'original/er_low_review.zip';b=original.read_bytes()
    if len(b)!=2024711 or hashlib.sha256(b).hexdigest()!='bb6d99cd2cce7b72230a8f84df50e96c74454a3a4a498898bc993aa9e8549aef':p.error('Original submitted ZIP differs.')
    out.mkdir(parents=True);source=out/'source';source.mkdir()
    with zipfile.ZipFile(original) as archive:
        for name in archive.namelist():
            path=(source/name).resolve()
            if not path.is_relative_to(source):raise ValueError('Invalid archive member')
        archive.extractall(source)
    inventory=json.loads((source/'source_inventory.json').read_text())
    freeze=json.loads((source/'reports/documentation/20261002/er_low/freeze.json').read_text())
    for row in [*inventory['files'],*freeze['source_files']]:
        payload=(source/row['path']).read_bytes()
        if len(payload)!=row['bytes'] or hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Frozen file differs: '+row['path'])
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join(map(str,(source/'scripts',source/'tests',base/'independent'))),ER_AUDIT_EVIDENCE=str(out/'evidence'),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    commands=[[sys.executable,'-B','-m','unittest',name,'-v'] for name in ('test_step28_er_weight_contracts','test_independent_gradient','test_independent_statistics_v2','test_independent_paths_v2')]
    commands += [[sys.executable,'-B','scripts/'+name,'--help'] for name in ('step28_er_weight_run.py','step28_er_weight_check.py')]
    commands += [['bash','-n','scripts/'+name] for name in ('run_step28_er_check_linux_20261001.sh','run_step28_er_weight_linux_20261001.sh')]
    logs=out/'logs';logs.mkdir();failed=False
    for i,cmd in enumerate(commands):
        start=datetime.datetime.now(datetime.timezone.utc)
        with (logs/f'{i:02}_stdout.log').open('wb') as stdout,(logs/f'{i:02}_stderr.log').open('wb') as stderr:
            result=subprocess.run(cmd,cwd=source,env=env,stdout=stdout,stderr=stderr)
        end=datetime.datetime.now(datetime.timezone.utc);record=dict(command=cmd,start_utc=start.isoformat(),end_utc=end.isoformat(),exit_code=result.returncode)
        (logs/f'{i:02}_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));failed|=result.returncode!=0
    raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
