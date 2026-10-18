"""Linux CPU wallclock wrapper; all attempts count against the approved 300 s."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

started = float(os.environ['QUALIFICATION_STARTED_EPOCH'])
root, prior = Path(sys.argv[1]).resolve(), float(sys.argv[2])
if os.name != 'posix' or root.exists() or prior < 0 or prior >= 300:
    raise ValueError('Linux, independent attempt and remaining cumulative budget required')
root.mkdir(parents=True)
command = ['taskset','-c','0','/home/yongpeng/miniconda3/envs/py310/bin/python',
           'scripts/step28_positive_ranking_verify.py','cpu','--output',str(root/'result'),
           '--prior-seconds',str(prior),*sys.argv[3:]]
receipt = dict(command=command, started_epoch=started, prior_charged_seconds=prior,
               deadline_epoch=started+300-prior, scope='Handwritten CPU only; no formal inputs/model links')
with (root/'console.txt').open('w',encoding='utf-8') as output:
    process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['pid']=process.pid
    try:
        process.wait(timeout=max(.01,started+300-prior-time.time()-5))
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        receipt['timeout']=True
receipt.update(exit_code=process.returncode,elapsed_seconds=time.time()-started)
receipt['charged_seconds']=math.ceil(receipt['elapsed_seconds'])
receipt['cumulative_charged_seconds']=prior+receipt['charged_seconds']
(root/'wallclock.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
sys.exit(process.returncode)
