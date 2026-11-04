"""Bounded incremental CPU qualification, existing Linux py310 only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--job', type=Path, required=True)
p.add_argument('--prior', type=int, required=True)
p.add_argument('--test', choices=('test_step29_english_abc.py', 'test_step29_english_gpu_wait.py'),
               default='test_step29_english_abc.py')
args = p.parse_args()
if sys.platform != 'linux' or args.job.exists() or not 0 <= args.prior < 1800:
    raise ValueError('Fresh Linux attempt and remaining CPU allowance required')
manifest = json.loads((args.workspace/'source_manifest.json').read_text())
for row in manifest['files']:
    content = (args.workspace/row['path']).read_bytes()
    if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Actual source differs: '+row['path'])
args.job.mkdir()
env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
       'TOKENIZERS_PARALLELISM':'false', 'CUDA_VISIBLE_DEVICES':'', 'TRANSFORMERS_OFFLINE':'1',
       'HF_HUB_OFFLINE':'1', 'PYTHONDONTWRITEBYTECODE':'1', 'TMPDIR':str(args.job),
       'ABC_TEST_ROOT':str(args.job), 'PYTHONPATH':str(args.workspace/'scripts')+':'+str(args.workspace/'tests')}
command = [sys.executable, '-B', str(args.workspace/'tests'/args.test)]
started, peak, reason = time.monotonic(), 0, None
with (args.job/'console.txt').open('w') as output:
    child = subprocess.Popen(command,cwd=args.workspace,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        try:
            process=psutil.Process(child.pid)
            rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)])
            peak=max(peak,rss)
            if time.monotonic()-started >= 1800-args.prior-2: reason='CPU_TIME_LIMIT'
            if rss > 4*2**30: reason='CPU_RSS_LIMIT'
            if reason:
                os.killpg(child.pid,signal.SIGKILL);break
        except psutil.NoSuchProcess:
            pass
        time.sleep(.1)
    code=child.wait()
elapsed=time.monotonic()-started
value=dict(exit_code=code,limit_reason=reason,wall_seconds=elapsed,charged_seconds=math.ceil(elapsed),
           cumulative_seconds=args.prior+math.ceil(elapsed),peak_rss_bytes=peak,sources=manifest,
           formal_data_reads=0,pretrained_model_loads=0,gpu=False)
(args.job/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k!='sources'}),flush=True)
raise SystemExit(code if code else (1 if reason else 0))
