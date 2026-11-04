"""One-shot, fast NVML waiter for three serial handwritten GPU probes.

It does not reserve GPU memory, suspend another process, or guarantee priority.
The earlier failed probe stays intact; this separately authorized attempt carries
its costs forward. No formal datasets or formal experiment launcher are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

import step28_formal_queue as queue

THRESHOLD = 15 * 2**30
POLL_SECONDS = .1
POOLINGS = ('mean_last4_sum', 'mean_last', 'cls')


class Auxiliary:
    def __init__(self, prior: int):
        self.prior, self.used, self.tick = prior, 0., time.monotonic()

    def settle(self):
        now = time.monotonic()
        self.used += now-self.tick
        self.tick = now
        if self.prior + math.ceil(self.used) >= 14400:
            raise TimeoutError('Cumulative auxiliary allowance exhausted')

    def sleep(self):
        self.settle()
        time.sleep(POLL_SECONDS)
        self.tick = time.monotonic()  # Passive waiting is not compute time.


def wait_ready(monitor, budget, waiting):
    """No CUDA allocation and no subprocess-per-poll; below 15 GiB keeps waiting."""
    while True:
        snapshot = monitor.snapshot()
        if snapshot['free'] >= THRESHOLD:
            return snapshot
        waiting(snapshot)
        budget.sleep()


def observe_update(permit: dict, job: Path, pid: int) -> None:
    """Delete only the bound active copy, after the probe's real update receipt."""
    root = Path(permit['root']).parent
    receipt = root/'state/first_update.json'
    if receipt.exists():
        return
    log = job/'updates.json'
    if not log.exists():
        return
    value = json.loads(log.read_text())
    if value.get('pid') != pid:
        raise ValueError('First-update PID does not match the owned probe')
    events = [event for event in value.get('intervals', [])
              if event.get('logs', {}).get('optimizer_updates') == 1
              and event.get('end', 0) > event.get('begin', 0)]
    if not events:
        return
    listener = Path(permit['listener']['path'])
    if (listener.resolve() != (root/'active_gpu_wait.py').resolve() or listener.is_symlink()
            or listener.stat().st_uid != os.getuid()):
        raise ValueError('Only the owned active listener copy may self-delete')
    queue.verify(permit['listener'])
    evidence = dict(epoch=time.time(), pid=pid, log=str(log), update=events[0],
                    listener=permit['listener'], listener_deleted=False)
    queue.write(receipt, evidence)
    listener.unlink()
    queue.write(receipt, {**evidence, 'listener_deleted':True})


def execute(permit: dict, monitor, command=None) -> dict:
    root = Path(permit['root']).parent
    state = root/'state'
    state.mkdir()  # Durable one-shot claim; a failed/completed attempt cannot restart.
    Path(permit['root']).mkdir()
    budget = Auxiliary(permit['prior_auxiliary_seconds'])
    ledger = dict(status='PREPARING', prior_gpu_seconds=permit['prior_gpu_seconds'],
                  new_gpu_seconds=0, completed=[], threshold_bytes=THRESHOLD,
                  poll_seconds=POLL_SECONDS, priority_guaranteed=False,
                  formal_experiment=False, active=None)
    child, output = None, None
    started = None
    last_saved = 0.

    def save():
        budget.settle()
        ledger['auxiliary_seconds'] = budget.used
        ledger['cumulative_gpu_seconds'] = ledger['prior_gpu_seconds']+ledger['new_gpu_seconds']
        ledger['observed_epoch'] = time.time()
        queue.write(state/'status.json', ledger)

    def waiting(snapshot):
        nonlocal last_saved
        ledger.update(status='WAIT_RESOURCE', free_gpu_bytes=snapshot['free'])
        if time.monotonic()-last_saved >= 10:
            save()
            last_saved = time.monotonic()

    def charge_child():
        nonlocal child, output, started
        if child is not None:
            queue.stop(child)
            elapsed = time.monotonic()-started
            ledger['new_gpu_seconds'] += math.ceil(elapsed)
            ledger['last_worker_seconds'] = elapsed
            ledger['active'] = None
            output.close()
            child, output, started = None, None, None

    try:
        for row in permit['sources']:
            queue.verify(row)
        for row in permit['model_files']:
            queue.verify(row)  # Verify before waiting; probe also verifies at launch.
        save()
        for index, pooling in enumerate(POOLINGS):
            if ledger['prior_gpu_seconds']+ledger['new_gpu_seconds']+300 > 1800:
                raise TimeoutError('Remaining GPU allowance cannot reserve one probe')
            while True:
                observed = wait_ready(monitor, budget, waiting)
                detected = time.time()
                if psutil.virtual_memory().available < 32*2**30:
                    raise RuntimeError('Insufficient system RAM for the approved probe')
                # Recheck immediately before launch; resource loss means wait, not fail.
                fresh = monitor.snapshot()
                if fresh['free'] >= THRESHOLD:
                    break
                waiting(fresh)
                budget.sleep()
            job = Path(permit['root'])/f'probe{index:02d}_{pooling}'
            output = (root/f'{job.name}.stdout.txt').open('x')
            started = time.monotonic()
            deadline = started+300
            if command is None:
                argv = [sys.executable, '-B', str(Path(permit['workspace'])/'gpu_probe.py'),
                        '--permit', permit['permit_path'], '--job', str(job),
                        '--pooling', pooling, '--deadline', str(deadline)]
            else:
                argv = command(pooling, job, deadline)
            env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                   'TOKENIZERS_PARALLELISM':'false', 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                   'CUDA_VISIBLE_DEVICES':'0', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                   'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':str(Path(permit['workspace'])/'scripts')}
            child = subprocess.Popen(argv, env=env, cwd=permit['workspace'], stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            ledger.update(status='RUNNING_SINGLE', active=dict(pid=child.pid, created=process.create_time(),
                job=str(job), pooling=pooling), detected_epoch=detected, launched_epoch=time.time(),
                detected_free_bytes=observed['free'], launch_free_bytes=fresh['free'])
            save()
            peak = 0
            while child.poll() is None:
                observe_update(permit, job, child.pid)
                snapshot = monitor.snapshot()
                peak = max(peak, snapshot['processes'].get(child.pid, 0))
                if peak > permit['single_process_gpu_bytes']:
                    raise RuntimeError('Owned probe exceeded its GPU limit')
                # queue.stop may wait 5 s after TERM plus 10 s after KILL.
                if time.monotonic() >= deadline-20:
                    raise TimeoutError('Owned probe reached shutdown reserve')
                budget.sleep()
            code = child.returncode
            observe_update(permit, job, child.pid)  # A short probe may exit between polls.
            charge_child()
            if code != 0:
                raise RuntimeError(f'GPU probe failed with exit {code}; no automatic retry')
            result = json.loads((job/'result.json').read_text())
            if (result['status'] != 'HANDWRITTEN_GPU_COMPLETE' or result['updates'] != 2
                    or not all(result['changed'].values()) or result['formal_data_reads'] != 0
                    or result['pooling'] != pooling):
                raise RuntimeError('Incomplete actual GPU probe evidence')
            ledger['completed'].append(dict(pooling=pooling, job=str(job), nvml_peak=peak, result=result))
            save()
        ledger['status'] = 'SINGLE_GPU_CHECKS_COMPLETE'
        save()
        return ledger
    except BaseException as error:
        failed = ledger.get('active')
        try:
            charge_child()
        except BaseException as cleanup_error:
            ledger['cleanup_error'] = repr(cleanup_error)
        # The first update may land after the last observation but before a
        # monitor/deadline error. Recheck after stopping, without hiding that error.
        if failed is not None:
            try:
                observe_update(permit, Path(failed['job']), failed['pid'])
            except BaseException as observation_error:
                ledger['first_update_observation_error'] = repr(observation_error)
        ledger.update(status='FAILED_NO_RETRY', error=repr(error))
        # Preserve the original error even when the auxiliary budget is exhausted.
        ledger['auxiliary_seconds'] = budget.used + time.monotonic()-budget.tick
        ledger['cumulative_gpu_seconds'] = ledger['prior_gpu_seconds']+ledger['new_gpu_seconds']
        queue.write(state/'status.json', ledger)
        raise
    finally:
        if output is not None:
            output.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permit', type=Path, required=True)
    args = parser.parse_args()
    permit = json.loads(args.permit.read_text())
    if (sys.platform != 'linux' or permit.get('operation') != 'english_abc_handwritten_gpu'
            or permit.get('approved') is not True or permit.get('review_closed') is not True
            or permit.get('formal_experiment') is not False or permit.get('total_worker_seconds') != 1800
            or permit.get('prior_gpu_seconds') != 14 or permit.get('threshold_bytes') != THRESHOLD
            or permit.get('poll_seconds') != POLL_SECONDS or permit.get('workers') != 1
            or permit.get('poolings') != list(POOLINGS) or permit.get('automatic_retry') is not False
            or permit.get('allocator_cap_bytes') != 14*2**30
            or permit.get('single_process_gpu_bytes') != int(14.75*2**30)
            or not 0 <= permit.get('prior_auxiliary_seconds', -1) < 14400):
        raise ValueError('Reviewed, explicit, bounded single-GPU waiting permission required')
    queue.verify(permit['review_evidence'])
    listener = queue.verify(permit['listener'])
    source = next(row for row in permit['sources']
                  if row['path'] == str(Path(permit['workspace'])/'scripts/step29_english_gpu_wait.py'))
    if (Path(__file__).resolve() != listener.resolve()
            or any(source[k] != permit['listener'][k] for k in ('bytes', 'sha256'))):
        raise ValueError('Actual active copy must match the retained pinned source')
    permit['permit_path'] = str(args.permit.resolve())
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    monitor = queue.GpuMemory()
    try:
        execute(permit, monitor)
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
