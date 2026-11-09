"""One authorized H45 job; bounded incremental accounting beside the live V44."""
import math
import os
from pathlib import Path
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'workspace/scripts'))
from step34_vendorlink_account_control import permit, read, validate_training
from step34_vendorlink_account_data import pinned, record, write
from step34_vendorlink_account_policy import PROTOCOL
from step30_evolution_control import Runner
from step30_evolution_io import Ledger


def main():
    with (HERE/'claim').open('x') as output:
        output.write(str(os.getpid()))
    gate = HERE/'gate.json'
    g = permit(gate)
    authorization = read(pinned(g['parallel_authorization']))
    quota = read(pinned(g['incremental_quota']))
    assert quota['secondary_caps'] == dict(training=28800, gpu=0, auxiliary=1800)
    assert quota['global_caps'] == dict(training=178200, gpu=21600, auxiliary=16200)
    assert sum(quota['global_caps'].values()) == 216000
    assert quota['primary_training_base']+14400+28800 <= quota['global_caps']['training']
    assert 14400+1800 == quota['global_caps']['auxiliary']
    root = Path(g['campaign'])
    batch = read(root/'P/batch.json')
    task = next(t for t in batch['tasks'] if t['id'] == 'P_H_SEQ_s45')
    original = HERE.parent/'campaign/P/training/P_H_SEQ_s45'
    assert not (original/'process.json').exists() and not (original/'run').exists()
    ledger = Ledger.__new__(Ledger)
    ledger.root, ledger.caps = root, quota['secondary_caps']
    ledger.used = dict(training=0, gpu=0, auxiliary=0)
    ledger.events, ledger.cpu_tests = [], 0
    # Incremental only: do not copy cumulative history or charge its tail twice.
    ledger.charge('auxiliary', 'H45_preparation', time.time()-authorization['preparation_started_epoch'], True)
    target = root/'P/training'/task['id']
    identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=record(gate))

    def completed(job):
        result = read(target/'run/completion.json')
        validate_training(task, result, pinned(g['manifest']), identity)
        latest = target/'run/latest.pt'
        old = record(latest)
        latest.unlink()
        write(target/'latest_retired.json', old)

    runner = Runner(g, gate, root, ledger,
                    worker_entry=HERE/'workspace/scripts/step34_vendorlink_account_control.py',
                    caps=ledger.caps, limits=g['limits'], halt_dispatch_on_failure=True)
    error = None
    write(HERE/'state.json', dict(status='H45_DISPATCH', automatic_next_phase=False))
    try:
        runner.run_many([dict(action='train', target=target, bucket='training', seconds=28800,
                             extra=['--phase', 'P', '--task', task['id']])], 1, completed)
    except BaseException as exc:
        error = repr(exc)
    primary = quota['primary_controller']
    wait_active = 0.
    while psutil.pid_exists(primary['pid']):
        tick = time.monotonic()
        try:
            p = psutil.Process(primary['pid'])
        except psutil.NoSuchProcess:
            break
        if (p.create_time() != primary['created'] or p.username() != primary['owner']
                or p.cmdline() != primary['command'] or p.cwd() != primary['cwd']):
            error = error or 'Primary identity changed; no speculative merge'
            break
        if (ledger.used['auxiliary']+math.ceil(wait_active)+60 >= ledger.caps['auxiliary']
                or time.monotonic() > quota['primary_deadline']+60):
            break
        wait_active += time.monotonic()-tick
        time.sleep(30)
    ledger.charge('auxiliary', 'primary_settlement_check', wait_active, True)
    primary_root = Path(quota['primary_campaign'])
    dispatch = read(primary_root/'dispatch.json')
    merged = dict(status='PENDING_PRIMARY_SETTLEMENT', primary=str(primary_root/'budget.json'),
                  secondary=record(root/'budget.json'), global_caps=quota['global_caps'])
    if not psutil.pid_exists(primary['pid']) and not dispatch['active']:
        settled = read(primary_root/'budget.json')
        used = {k: settled['used_seconds'][k]+ledger.used[k] for k in ledger.used}
        assert all(used[k] <= quota['global_caps'][k] for k in used)
        merged.update(status='SETTLED', primary=record(primary_root/'budget.json'), used_seconds=used,
                      cpu_test_seconds=settled['cpu_test_seconds']+ledger.cpu_tests)
    write(HERE/'combined_budget.json', merged)
    write(HERE/'state.json', dict(status='H45_FAILED' if error else 'H45_COMPLETE', error=error,
          automatic_retry=False, automatic_next_phase=False, accounting=merged['status']))


if __name__ == '__main__':
    main()
