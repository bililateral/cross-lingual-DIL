"""One fixed 18-run queue and the already reviewed comparison evaluation entry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

import step28_continual_comparison_run as run
import step28_continual_five_control as prior

data, m = run.data, run.m


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    data.write_json(temporary, value)
    os.replace(temporary, path)


class Ledger:
    def __init__(self, path: Path, limits: dict, prior_used: dict, clock=time.monotonic):
        self.path, self.limits, self.clock = path, limits, clock
        self.used = {key: float(prior_used.get(key, 0.)) for key in limits}
        self.active, self.completed, self.failed = {}, [], False
        self.write_carry = 0.
        if any(value < 0 or value > limits[key] for key, value in self.used.items()):
            raise ValueError('Invalid inherited comparison charges')

    def snapshot(self):
        return dict(limits=self.limits, used=self.used, active=self.active,
                    completed=self.completed, failed=self.failed,
                    cumulative_seconds=sum(self.used.values()))

    def save(self):
        write(self.path, self.snapshot())

    def available(self, bucket):
        return self.limits[bucket]-self.used[bucket]-sum(
            row['allowance'] for row in self.active.values() if row['bucket'] == bucket)

    def reserve(self, identity, bucket, allowance, remaining=0.):
        if (self.failed or identity in self.active
                or any(row['id'] == identity for row in self.completed)
                or allowance <= 0 or allowance+remaining > self.available(bucket)):
            raise ValueError('Duplicate attempt or insufficient whole-queue reservation')
        now = self.clock()
        row = dict(id=identity, bucket=bucket, allowance=allowance, started=now, deadline=now+allowance)
        self.active[identity] = row
        self.save()
        return row

    def settle(self, identity, success):
        row = self.active.pop(identity)
        actual = self.clock()-row['started']
        self.used[row['bucket']] += actual
        self.completed.append({**row, 'actual_seconds': actual, 'success': bool(success)})
        self.failed |= not success or actual > row['allowance'] or self.used[row['bucket']] > self.limits[row['bucket']]
        self.save()
        if self.failed:
            raise RuntimeError('Failed/over-budget attempt retained; no retry')

    def charge(self, bucket, seconds):
        tick = self.clock()
        if bucket == 'control':
            seconds += self.write_carry
            self.write_carry = 0.
        self.used[bucket] += seconds
        self.failed |= self.available(bucket) < 0
        try:
            self.save()
        finally:
            if bucket == 'control':
                self.write_carry += self.clock()-tick
        if self.failed:
            raise RuntimeError('Cumulative budget stopped this campaign')


def validate_gate(path: Path) -> dict:
    gate = run.gate(path, 'train')
    if (gate['budget_seconds'] != dict(training=800*3600, confirmation=18*3600,
            cpu=1200, native=3000, generation=600, control=3600, analysis=1800, preparation=4200)
            or sum(gate['budget_seconds'].values()) != 822*3600
            or gate['maximum_workers'] != 3
            or gate['per_run_seconds'] != dict(L2R_text_fixed=72*3600, ER_GCR_group_fixed=12*3600)
            or gate['approved_actions'] != ['train', 'generate', 'confirmation', 'analyze']
            or not gate['train_data'] or not gate['new_confirmation_data']):
        raise ValueError('Approved fixed 822-hour scope differs')
    required = {'scripts/step28_comparison_control.py', 'tests/test_step28_comparison_control.py',
                'run_step28_comparison_linux_20261026.sh'}
    if {row['path'] for row in gate['orchestration_sources']} != required:
        raise ValueError('Incomplete orchestration source binding')
    for row in gate['orchestration_sources']:
        data.verify(data.ROOT/row['path'], row)
    ready_path = Path(gate['review_ready'])
    ready = data.read_json(ready_path)
    if (ready['status'] != 'REVIEW_CLOSED_AND_CPU_PASSED'
            or ready['orchestration_sources'] != gate['orchestration_sources']):
        raise ValueError('Orchestration disposition has not closed')
    receipt = data.read_json(data.verify(Path(ready['cpu_receipt']['path']), ready['cpu_receipt']))
    if receipt['exit_code'] != 0 or receipt['orchestration_sources'] != gate['orchestration_sources']:
        raise ValueError('CPU receipt does not cover these exact sources')
    return gate


def reference_check(gate):
    preflight = data.read_json(data.verify(Path(gate['reference_preflight']['path']), gate['reference_preflight']))
    if preflight['failures'] or len(preflight['runs']) != 18 or len(preflight['endpoints']) != 54:
        raise ValueError('Missing pinned H/S endpoints')
    data.verify(Path(gate['reference_gate']['path']), gate['reference_gate'])
    for row in preflight['runs']:
        path = Path(gate['reference_campaign'])/row['id']/'run/completion.json'
        data.verify(path, dict(bytes=row['completion_bytes'], sha256=row['completion_sha256']))
    for row in preflight['endpoints']:
        data.verify(Path(row['path']), row)


def resources(campaign, startup=False):
    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())*2**20
    apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip()
    import shutil
    row = dict(epoch=time.time(), free_gpu_bytes=free, cuda_pids=apps,
               host_available_bytes=psutil.virtual_memory().available,
               disk_free_bytes=shutil.disk_usage(campaign).free)
    row['ready'] = (free >= (30 if startup else 10)*2**30
        and (not startup or not apps)
        and row['host_available_bytes'] >= (208 if startup else 64)*2**30
        and row['disk_free_bytes'] >= (192 if startup else 76)*2**30)
    write(campaign/'admission_status.json', {**row, 'startup': startup})
    return row['ready']


def campaign_size(campaign, limits):
    files = [(p.suffix, p.stat().st_size) for p in campaign.rglob('*') if p.is_file() and not p.is_symlink()]
    if (sum(size for _, size in files) > limits['disk_bytes']
            or sum(size for suffix, size in files if suffix != '.pt') > limits['small_bytes']
            or prior.run.resident_bytes(prior.run.owned_processes(os.getpid())) > 192*2**30):
        raise RuntimeError('Campaign disk/small-evidence/RSS allowance exceeded')


def dispatch(queue, campaign, ledger, bucket, command, *, max_workers=3,
             admit=None, check=lambda: None, observe=lambda: None, poll_seconds=30.):
    """Reviewed refill structure; dispatch metadata is outside worker-owned job dirs."""
    active, index, tick = {}, 0, None
    cores = psutil.Process().cpu_affinity()[:max_workers]
    if len(cores) != max_workers:
        raise RuntimeError('Distinct worker CPU cores unavailable')
    try:
        while active or index < len(queue):
            tick = time.monotonic()
            for identity, (process, log, _) in list(active.items()):
                if process.poll() is None and time.monotonic() >= ledger.active[identity]['deadline']-60:
                    tick += prior.terminate_owned(process)
                    raise TimeoutError('Owned child stopped within reserved termination tail')
                if process.poll() is not None:
                    log.close()
                    del active[identity]
                    ledger.settle(identity, process.returncode == 0)
            check()
            observe()
            while len(active) < max_workers and index < len(queue):
                if admit and not admit(len(active)):
                    break
                row = queue[index]
                remaining = sum(item['allowance'] for item in queue[index+1:])
                reservation = ledger.reserve(row['id'], bucket, row['allowance'], remaining)
                slot = next(v for v in range(max_workers) if v not in [a[2] for a in active.values()])
                metadata = campaign/'dispatch'/row['id']
                metadata.mkdir(parents=True)
                job = campaign/row['id']  # The unchanged train/evaluate entry owns mkdir.
                line = ['taskset', '-c', str(cores[slot]), *command(row, job, reservation['deadline'], metadata)]
                write(metadata/'dispatch.json', dict(reservation=reservation, command=line))
                log = (metadata/'console.txt').open('w', encoding='utf-8')
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
                try:
                    process = subprocess.Popen(line, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    ledger.settle(row['id'], False)
                    raise
                active[row['id']] = (process, log, slot)
                write(metadata/'process.json', dict(pid=process.pid, owner=psutil.Process(process.pid).username(),
                                                   created=psutil.Process(process.pid).create_time()))
                index += 1
            ledger.charge('control', time.monotonic()-tick)
            tick = None
            if active or index < len(queue):
                next_stop = min((ledger.active[k]['deadline']-60 for k in active),
                                default=time.monotonic()+poll_seconds)
                time.sleep(max(0., min(poll_seconds, next_stop-time.monotonic())))
    finally:
        cleanup_tick, waited = time.monotonic(), 0.
        if tick is not None:
            try:
                ledger.charge('control', time.monotonic()-tick)
            except RuntimeError:
                pass
        for identity, (process, log, _) in active.items():
            try:
                waited += prior.terminate_owned(process)
            finally:
                log.close()
        for identity in list(ledger.active):
            try:
                ledger.settle(identity, False)
            except RuntimeError:
                pass
        ledger.charge('control', time.monotonic()-cleanup_tick-waited)


def execute(gate_path):
    gate = validate_gate(gate_path)
    campaign = Path(gate['campaign'])
    campaign.mkdir()  # Exclusive campaign creation also rejects duplicate waiters.
    ledger = Ledger(campaign/'budget.json', gate['budget_seconds'], gate['prior_used_seconds'])
    ledger.save()
    try:
        tick = time.monotonic()
        try:
            reference_check(gate)
        finally:
            ledger.charge('preparation', time.monotonic()-tick)
        while True:
            tick = time.monotonic()
            try:
                ready = resources(campaign, startup=True)
            finally:
                ledger.charge('control', time.monotonic()-tick)
            if ready:
                break
            write(campaign/'status.json', dict(status='WAITING_FOR_STARTUP_RESOURCES', epoch=time.time()))
            time.sleep(60)
        write(campaign/'status.json', dict(status='TRAINING', epoch=time.time(), pid=os.getpid()))
        launcher = Path(gate['launcher'])
        def observe():
            if not launcher.exists():
                return
            for path in (campaign/'dispatch').glob('*/console.txt') if (campaign/'dispatch').exists() else []:
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get('event') == 'updates' and event.get('completed', 0) > 0:
                            launcher.unlink()
                            write(campaign/'first_update.json', dict(event=event, log=str(path),
                                  launcher_deleted=True, epoch=time.time()))
                            return
        queue = [dict(id=s.run_id, spec=s.identity(), allowance=gate['per_run_seconds'][s.name])
                 for s in m.specifications(new_only=True)]
        def train_command(row, job, deadline, metadata):
            write(metadata/'spec.json', row['spec'])
            return [sys.executable, '-B', str(Path(run.__file__)), '--job', str(job),
                    '--spec', str(metadata/'spec.json'), '--gate', str(gate_path), '--deadline', str(deadline)]
        dispatch(queue, campaign, ledger, 'training', train_command,
                 admit=lambda active: resources(campaign), check=lambda: campaign_size(campaign, gate['limits']),
                 observe=observe)
        import step28_continual_comparison_evaluate as evaluation
        tick = time.monotonic()
        try:
            records = evaluation.full_gate(campaign, gate)
        finally:
            ledger.charge('preparation', time.monotonic()-tick)
        write(campaign/'training_complete.json', dict(status='ALL_18_NEW_AND_54_REFERENCE_ENDPOINTS_COMPLETE',
              new_run_ids=[s.run_id for s in m.specifications(new_only=True)], all_runs=len(records),
              sources=gate['sources'], budget=ledger.snapshot()))
        for action, identity, bucket in [('generate', 'generation', 'generation'),
                                         ('confirmation', 'confirmation', 'confirmation'),
                                         ('analyze', 'analysis', 'analysis')]:
            def post_command(row, job, deadline, metadata):
                return [sys.executable, '-B', str(Path(evaluation.__file__)), action,
                        '--campaign', str(campaign), '--gate', str(gate_path), '--deadline', str(deadline)]
            dispatch([dict(id=identity, allowance=ledger.available(bucket))], campaign, ledger,
                     bucket, post_command, max_workers=1,
                     admit=(lambda active: resources(campaign)) if action == 'confirmation' else None,
                     check=lambda: campaign_size(campaign, gate['limits']))
        write(campaign/'completion.json', dict(status='ALL_COMPARISON_TRAINING_CONFIRMATION_ANALYSIS_COMPLETE',
                                               sources=gate['sources'], budget=ledger.snapshot()))
    except BaseException as error:
        write(campaign/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error),
              traceback=traceback.format_exc(), budget=ledger.snapshot()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    execute(args.gate.resolve())
