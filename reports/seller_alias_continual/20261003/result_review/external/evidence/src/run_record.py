#!/usr/bin/env python3
"""Execute one review command and retain exact streams and provenance."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--cwd', required=True)
    ap.add_argument('--evidence', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--source', action='append', default=[])
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        ap.error('command required')
    run = Path(a.evidence) / 'runs' / a.label
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    settings = {k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
    settings.update({'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false'})
    env.update(settings)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    selected_cpu = affinity[0] if affinity else None
    sources=[]
    for arg in a.source + command:
        q=Path(arg)
        if not q.is_absolute():
            q=Path(a.cwd)/q
        if q.suffix == '.py' and q.is_file():
            q=q.resolve()
            if str(q) in [s['path'] for s in sources]:
                continue
            target=run/'source_snapshots'/f'{len(sources):02d}_{q.name}'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(q,target)
            sources.append({'path':str(q),'snapshot':str(target.relative_to(run)),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    meta={'label':a.label,'argv':command,'shell_command':shlex.join(command),'cwd':str(Path(a.cwd).resolve()),'wrapper_argv':sys.argv,'environment_overrides':settings,'inherited_environment_values':'not recorded (may include secrets)','allowed_cpus_before':affinity,'selected_cpu':selected_cpu,'sources':sources,'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    (run/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    t=time.monotonic()
    try:
        def pin():
            if selected_cpu is not None:
                os.sched_setaffinity(0,{selected_cpu})
        with (run/'stdout.txt').open('wb') as out, (run/'stderr.txt').open('wb') as err:
            p=subprocess.run(command,cwd=a.cwd,env=env,stdout=out,stderr=err,preexec_fn=pin if os.name=='posix' else None)
        code=p.returncode
    except BaseException as exc:
        code=127
        (run/'stderr.txt').write_text(repr(exc)+'\n')
        (run/'stdout.txt').touch(exist_ok=True)
    meta.update({'ended_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'exit_code':code})
    meta['streams']={f:{'bytes':(run/f).stat().st_size,'sha256':hashlib.sha256((run/f).read_bytes()).hexdigest()} for f in ('stdout.txt','stderr.txt')}
    (run/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run':str(run),'exit_code':code,'elapsed_seconds':meta['elapsed_seconds']},ensure_ascii=False))
    for stream in ('stdout.txt','stderr.txt'):
        print(f'[{stream}]')
        print((run/stream).read_text(errors='replace'),end='')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
