"""One fixed CPU evaluation, charged to the approved 7200-second ledger."""
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import time

PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'


def launch(root, script=None):
    root = Path(root)
    ledger_path = root/'budget.json'
    ledger = json.loads(ledger_path.read_text())
    used = sum(x['seconds'] for x in ledger['entries'])
    if (ledger['limit_seconds'] != 7200 or used != ledger['used_seconds'] or ledger.get('active')
            or any(x['id'] == 'evaluation' for x in ledger['entries']) or (root/'formal').exists()):
        raise ValueError('Budget identity, unfinished attempt or prior evaluation forbids launch')
    # Leave room for saved-result checking, export and custody. All work uses
    # the same ledger; this is a limit, not a prediction of actual runtime.
    allowance = min(5400, 7200-used-600)
    if allowance <= 0: raise ValueError('Insufficient approved remaining CPU budget')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1',
           'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
           'PYTHONPATH':str(root/'workspace/scripts')}
    script = script or root/'workspace/scripts/step28_defaults_evaluate.py'
    command = ['nice','-n','15',PYTHON,'-B',str(script),'--manifest',str(root/'input_manifest.json'),
               '--review-ready',str(root/'review_ready.json'),'--output',str(root/'formal'),'--seconds',str(allowance)]
    started = time.monotonic()
    ledger['active'] = dict(id='evaluation', allowance=allowance, epoch=time.time(), command=command)
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    code = None; error = None
    try:
        with (root/'formal.stdout.txt').open('x') as log:
            p = subprocess.run(command, cwd=root/'workspace', env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=allowance)
        code = p.returncode
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        charge = math.ceil(time.monotonic()-started)
        ledger['active'] = None
        ledger['entries'].append(dict(id='evaluation',seconds=charge,exit_code=code,error=error,
                                      success=(code==0 and error is None)))
        ledger['used_seconds'] = used+charge
        ledger['labels_opened'] = (root/'formal/evaluation_access.json').exists()
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    if code: raise RuntimeError('Evaluation failed; no automatic retry or repeated label opening')


if __name__ == '__main__':
    launch(Path(sys.argv[1]))
