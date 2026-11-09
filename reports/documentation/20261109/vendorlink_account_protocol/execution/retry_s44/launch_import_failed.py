"""Run the user-authorized single retry through the unchanged reviewed Runner."""
import json
import os
from pathlib import Path
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
sys.path.insert(0, str(HERE / 'workspace/scripts'))

from step34_vendorlink_account_control import permit, validate_training
from step34_vendorlink_account_data import pinned, read, record, write
from step34_vendorlink_account_policy import PROTOCOL
from step30_evolution_control import Runner
from step30_evolution_io import Ledger


def main():
    # No experiment/source changes, checkpoint recovery, scoring, or queue replay.
    claim = HERE / 'claim'
    with claim.open('x') as output:
        output.write(str(os.getpid()))
    original_launch = read(ORIGINAL / 'launch.json')
    gate_path = HERE / 'gate.json'
    g = permit(gate_path)
    while psutil.pid_exists(original_launch['pid']):
        p = psutil.Process(original_launch['pid'])
        if (p.create_time() != original_launch['created']
                or p.username() != original_launch['owner']
                or p.cmdline() != original_launch['command']
                or p.cwd() != original_launch['cwd']):
            raise RuntimeError('Original controller identity changed; no retry launch')
        write(HERE / 'state.json', dict(status='WAIT_ORIGINAL_ACCOUNTING', pid=os.getpid()))
        time.sleep(5)
    original_dispatch = read(ORIGINAL / 'campaign/dispatch.json')
    if original_dispatch['active'] or original_dispatch['status'] != 'INCOMPLETE':
        raise RuntimeError('Original active jobs have not been released')
    original_budget = read(ORIGINAL / 'campaign/budget.json')
    root = Path(g['campaign'])
    batch = read(root / 'P/batch.json')
    task = next(t for t in batch['tasks'] if t['id'] == 'P_VendorLink_SEQ_s44')
    if task != next(t for t in read(ORIGINAL / 'campaign/P/batch.json')['tasks'] if t['id'] == task['id']):
        raise RuntimeError('Retry task/configuration differs')
    write(HERE / 'inherited_budget.json', dict(source=record(ORIGINAL / 'campaign/budget.json'),
          used_seconds=original_budget['used_seconds'], reason='All original success/failure costs retained'))
    # Continue the cumulative ledger without charging its original tail twice.
    ledger = Ledger.__new__(Ledger)
    ledger.root, ledger.caps = root, dict(original_budget['caps'])
    ledger.used = dict(original_budget['used_seconds'])
    ledger.events = list(original_budget['events'])
    ledger.cpu_tests = original_budget['cpu_test_seconds']
    ledger.charge('auxiliary', 'single_retry_preparation', 120, True)
    target = root / 'P/training' / task['id']
    identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=record(gate_path))

    def completed(job):
        result = read(target / 'run/completion.json')
        validate_training(task, result, pinned(g['manifest']), identity)
        latest = target / 'run/latest.pt'
        old = record(latest)
        latest.unlink()
        write(target / 'latest_retired.json', old)

    runner = Runner(g, gate_path, root, ledger,
                    worker_entry=HERE / 'workspace/scripts/step34_vendorlink_account_control.py',
                    caps=ledger.caps, limits=g['limits'], halt_dispatch_on_failure=True)
    write(HERE / 'state.json', dict(status='SINGLE_RETRY_DISPATCH', pid=os.getpid()))
    try:
        runner.run_many([dict(action='train', target=target, bucket='training', seconds=14400,
                             extra=['--phase', 'P', '--task', task['id']])], 1, completed)
    except BaseException as error:
        write(HERE / 'state.json', dict(status='SINGLE_RETRY_FAILED', error=repr(error), automatic_retry=False))
        raise
    write(HERE / 'state.json', dict(status='SINGLE_RETRY_COMPLETE', task=task['id'],
                                  automatic_next_phase=False, automatic_retry=False))


if __name__ == '__main__':
    main()
