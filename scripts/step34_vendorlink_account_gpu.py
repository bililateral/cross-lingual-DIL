"""One finite actual-BERT probe for the new largest H training shape."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import psutil
import torch

from step28_formal_queue import GpuMemory
from step29_vendorlink_methods import make_optimizer, update_h
from step34_vendorlink_account_data import Check, PYTHON, pinned, write
from step34_vendorlink_account_policy import settings, LIMITS
from step34_vendorlink_account_run import make_model, groups_for


def probe(binding, root, seconds):
    check_base = Check(root, seconds)
    nv = GpuMemory()
    peak = 0
    def check():
        nonlocal peak
        check_base()
        peak = max(peak, nv.snapshot()['processes'].get(os.getpid(), 0))
        if peak > LIMITS['process_gpu_bytes']:
            raise RuntimeError('Whole GPU PID exceeds approved 14 GiB')
    status, error, logs = 'FAILED', None, []
    try:
        if nv.snapshot()['free'] < LIMITS['free_gpu_bytes']:
            raise RuntimeError('GPU admission unavailable')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
        for row in binding['files']:
            pinned(row)
        task = dict(seed=41, method='H_full', settings=settings('H1', 'H_full'))
        model = make_model(binding, task)
        accounts = [dict(account_uid=f'fixture{i:02d}', records=[dict(uid=f'{i:02d}-{j:02d}',
                    title='test '*260, description='token '*260) for j in range(12)]) for i in range(11)]
        current, history = groups_for(accounts, 41, 'AlphaBay', 0)[:2]
        teacher = model.teacher(history, check)
        optimizer = make_optimizer(model)
        before = model.head[0].weight.detach().clone()
        for step in range(2):
            logs.append(update_h(model, optimizer, current, history, teacher, 100+step, check))
        torch.cuda.synchronize()
        check()
        if torch.equal(before, model.head[0].weight) or logs[1]['history']['distillation'] <= 0:
            raise ValueError('Missing actual update/birth-score objective')
        status = 'PASS'
    except BaseException as e:
        error = repr(e)
        raise
    finally:
        elapsed = time.monotonic()-check_base.started
        write(root/'gpu_shape.json', dict(status=status, error=error, logs=logs, seconds=elapsed,
              charged_seconds=math.ceil(elapsed), bucket='gpu', peak_gpu_pid_bytes=peak,
              peak_rss_bytes=check_base.peak_rss, peak_allocated=torch.cuda.max_memory_allocated(),
              advertisements=44, record_pairs=946, updates_completed=len(logs), candidate='H1',
              alpha=.07049361507739538, beta=.767438161798796, real_data_read=False))
        nv.close()


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Linux existing py310 only')
    if (args.root/'gpu_shape.json').exists():
        raise ValueError('No automatic retry of the same probe')
    probe(json.loads(args.binding.read_text()), args.root, 600)
