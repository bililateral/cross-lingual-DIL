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
        self.budget_exceeded = False
        self.write_carry = 0.
        if any(value < 0 or value > limits[key] for key, value in self.used.items()):
            raise ValueError('Invalid inherited comparison charges')

    def snapshot(self):
        return dict(limits=self.limits, used=self.used, active=self.active,
                    completed=self.completed, failed=self.failed, budget_exceeded=self.budget_exceeded,
                    cumulative_seconds=sum(self.used.values()))

    def save(self):
        write(self.path, self.snapshot())

    def available(self, bucket):
        return self.limits[bucket]-self.used[bucket]-sum(
            row['allowance'] for row in self.active.values() if row['bucket'] == bucket)

    def reserve(self, identity, bucket, allowance, remaining=0., *, allow_failed_campaign=False):
        if (self.budget_exceeded or (self.failed and not allow_failed_campaign) or identity in self.active
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
        result = {**row, 'actual_seconds': actual,
                  'success': bool(success and actual <= row['allowance']),
                  'allowance_exceeded': actual > row['allowance']}
        self.completed.append(result)
        self.budget_exceeded |= self.available(row['bucket']) < 0
        self.failed |= not result['success'] or self.budget_exceeded
        self.save()
        if self.budget_exceeded:
            raise RuntimeError('Cumulative bucket/reservation exceeded')
        return result

    def charge(self, bucket, seconds):
        tick = self.clock()
        if bucket == 'control':
            seconds += self.write_carry
            self.write_carry = 0.
        self.used[bucket] += seconds
        self.budget_exceeded |= self.available(bucket) < 0
        self.failed |= self.budget_exceeded
        try:
            self.save()
        finally:
            if bucket == 'control':
                self.write_carry += self.clock()-tick
        if self.budget_exceeded:
            raise RuntimeError(f'Cumulative {bucket} budget exceeded')


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
    launcher_name = 'run_step28_comparison_linux_20261030.sh' if gate.get('recovery') else 'run_step28_comparison_linux_20261026.sh'
    required = {'scripts/step28_comparison_control.py', 'tests/test_step28_comparison_control.py', launcher_name}
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


def recovery_attempts(gate):
    """One explicit recovery; old failures stay in their original receipt and total fees."""
    recovery = gate['recovery']
    old_gate = data.read_json(data.verify(Path(recovery['original_gate']['path']), recovery['original_gate']))
    failure = data.read_json(data.verify(Path(recovery['original_failure']['path']), recovery['original_failure']))
    rows = failure['budget']['completed']
    succeeded = {r['id'] for r in rows if r['success']}
    failed = {r['id'] for r in rows if not r['success']}
    restart = {f'CAB_s{s}_L2R_text_fixed' for s in (4, 5, 6)}
    pending = ['CAB_s6_L2R_text_fixed', 'CAB_s4_L2R_text_fixed',
               'CAB_s5_L2R_text_fixed', 'CAB_s6_ER_GCR_group_fixed']
    if (failure['status'] != 'STOP_NO_RETRY' or failure['budget']['active']
            or len(rows) != 17 or len({r['id'] for r in rows}) != 17
            or failed != restart or len(succeeded) != 14
            or set(recovery['reused']) != succeeded or recovery['pending_ids'] != pending
            or succeeded | set(pending) != {s.run_id for s in m.specifications(new_only=True)}
            or recovery['failure_policy'] != 'retire_failed_slot_continue_healthy'
            or gate['budget_seconds'] != old_gate['budget_seconds']
            or any(gate['prior_used_seconds'][k] < v for k, v in failure['budget']['used'].items())
            or gate['prior_used_seconds']['training'] != failure['budget']['used']['training']):
        raise ValueError('Recovery identities, inherited charges or fixed scope differ')
    old_sources = {r['path']: r for r in old_gate['sources']}
    new_sources = {r['path']: r for r in gate['sources']}
    allowed_changes = {'scripts/step28_continual_comparison_run.py',
                       'scripts/step28_continual_comparison_evaluate.py',
                       'tests/test_step28_continual_comparison.py'}
    if (old_sources.keys() != new_sources.keys()
            or any(new_sources[k] != v for k, v in old_sources.items() if k not in allowed_changes)):
        raise ValueError('Recovery changed frozen learning dependencies')
    return old_gate, failure, pending


def adopt_recovery(campaign, gate, ledger):
    old_gate, failure, pending = recovery_attempts(gate)
    old = Path(old_gate['campaign'])
    def link_tree(source, destination):
        import shutil
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Unexpected symlink in retained trajectory')
        shutil.copytree(source, destination, copy_function=os.link)
    for identity, receipt in gate['recovery']['reused'].items():
        root = old/identity/'run'
        row = data.read_json(data.verify(root/'completion.json', receipt))
        if (row['status'] != run.COMPLETE or row['updates'] != 864
                or row['source_files'] != old_gate['sources']
                or m.Spec.from_dict(row['spec']).run_id != identity
                or set(row['points']) != {'1', '2', '3'}
                or row['access'] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
            raise ValueError('Retained trajectory is incomplete or foreign')
        for number, point in row['points'].items():
            if (point['adam_step'] != int(number)*288 or not point['full_restore_verified']
                    or not point['inference_restore_verified']):
                raise ValueError('Retained endpoint has no complete restoration evidence')
            for key in ('model', 'mapping', 'memory', 'calibration_scores'):
                data.verify(root/point[key]['path'], point[key])
        link_tree(old/identity, campaign/identity)
    (campaign/'prior_failed').mkdir()
    for identity in (r['id'] for r in failure['budget']['completed'] if not r['success']):
        link_tree(old/identity, campaign/'prior_failed'/identity)
    ledger.completed = [dict(r) for r in failure['budget']['completed'] if r['success']]
    ledger.save()
    write(campaign/'recovery.json', dict(original_gate=gate['recovery']['original_gate'],
          original_failure=gate['recovery']['original_failure'], prior_attempts=failure['budget']['completed'],
          reused_ids=sorted(gate['recovery']['reused']), pending_ids=pending,
          note='Prior failed fees are inherited; only this explicitly authorized new attempt is runnable.'))
    return pending


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


def resources(campaign, startup=False, active_workers=None):
    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())*2**20
    apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], text=True).strip()
    import shutil
    row = dict(epoch=time.time(), free_gpu_bytes=free, cuda_pids=apps,
               host_available_bytes=psutil.virtual_memory().available,
               disk_free_bytes=shutil.disk_usage(campaign).free)
    row['ready'] = (free >= (30 if startup else 10)*2**30
        and (not startup or not apps)
        and row['host_available_bytes'] >= (208 if startup else 64)*2**30
        and row['disk_free_bytes'] >= (192 if startup else 76)*2**30)
    if active_workers is not None:
        def applications(text):
            rows = []
            for line in text.splitlines():
                pid, size = (part.strip() for part in line.split(','))
                rows.append((int(pid), int(size)*2**20 if size.isdecimal() else 0))
            return rows
        owned = set(prior.run.owned_processes(os.getpid()))
        second_free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                          '--format=csv,noheader,nounits'], text=True).strip())*2**20
        second_apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_memory',
                      '--format=csv,noheader,nounits'], text=True).strip()
        own = lambda text: {pid: size for pid, size in applications(text) if pid in owned}
        stable = free == second_free and own(apps) == own(second_apps)
        row.update(slot_admission(min(free, second_free), applications(second_apps), owned, active_workers))
        row.update(stable_gpu_observation=stable, second_free_gpu_bytes=second_free, second_cuda_pids=second_apps)
        row['ready'] &= row['slot_ready'] and stable
    write(campaign/'admission_status.json', {**row, 'startup': startup})
    return row['ready']


def slot_admission(free, applications, owned, active):
    allocated = sum(min(size, int(7.75*2**30)) for pid, size in applications if pid in owned)
    unmaterialized = max(0, active*int(7.75*2**30)-allocated)
    required = 10*2**30+unmaterialized
    return dict(unmaterialized_own_reserve_bytes=unmaterialized, required_free_gpu_bytes=required,
                slot_ready=free >= required)


def campaign_size(campaign, limits):
    files = [(p.suffix, p.stat().st_size) for p in campaign.rglob('*') if p.is_file() and not p.is_symlink()]
    if (sum(size for _, size in files) > limits['disk_bytes']
            or sum(size for suffix, size in files if suffix != '.pt') > limits['small_bytes']
            or prior.run.resident_bytes(prior.run.owned_processes(os.getpid())) > 192*2**30):
        raise RuntimeError('Campaign disk/small-evidence/RSS allowance exceeded')


def dispatch(queue, campaign, ledger, bucket, command, *, max_workers=3,
             admit=None, check=lambda: None, observe=lambda: None, poll_seconds=30.,
             retire_failed_slots=False):
    """On a child failure, stop admission and drain already running siblings."""
    active, index, tick, failures = {}, 0, None, []
    slots = set(range(max_workers))
    cores = psutil.Process().cpu_affinity()[:max_workers]
    if len(cores) != max_workers:
        raise RuntimeError('Distinct worker CPU cores unavailable')
    try:
        while active or (index < len(queue) and slots and (not ledger.failed or retire_failed_slots)):
            tick = time.monotonic()
            for identity, (process, log, slot) in list(active.items()):
                timed_out = False
                if process.poll() is None and time.monotonic() >= ledger.active[identity]['deadline']-60:
                    tick += prior.terminate_owned(process)
                    timed_out = True
                if process.poll() is not None:
                    log.close()
                    del active[identity]
                    settled = ledger.settle(identity, process.returncode == 0 and not timed_out)
                    if not settled['success']:
                        failure = dict(id=identity, returncode=process.returncode,
                                       deadline_stopped=timed_out, allowance_exceeded=settled['allowance_exceeded'],
                                       worker_failure_path=str(campaign/identity/'failure.json'))
                        failures.append(failure)
                        if retire_failed_slots:
                            slots.remove(slot)
                        write(campaign/'dispatch'/identity/'failure.json', failure)
            check()
            observe()
            if failures:
                write(campaign/'status.json', dict(status='CONTINUING_HEALTHY_SLOTS' if retire_failed_slots else 'DRAINING_ACTIVE_WORKERS', epoch=time.time(),
                      failed_workers=failures, active=list(active),
                      available_slots=sorted(slots), not_dispatched=[row['id'] for row in queue[index:]]))
            while (not ledger.failed or retire_failed_slots) and len(active) < len(slots) and index < len(queue):
                if admit and not admit(len(active)):
                    break
                row = queue[index]
                remaining = sum(item['allowance'] for item in queue[index+1:])
                reservation = ledger.reserve(row['id'], bucket, row['allowance'], remaining,
                                             allow_failed_campaign=retire_failed_slots)
                slot = next(v for v in sorted(slots) if v not in [a[2] for a in active.values()])
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
                except OSError as error:
                    log.close()
                    ledger.settle(row['id'], False)
                    failure = dict(id=row['id'], error=repr(error), returncode=None,
                                   deadline_stopped=False, worker_started=False)
                    failures.append(failure)
                    if retire_failed_slots:
                        slots.remove(slot)
                    write(metadata/'failure.json', failure)
                    index += 1
                    break
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
            if active or (index < len(queue) and slots and (not ledger.failed or retire_failed_slots)):
                next_stop = min((ledger.active[k]['deadline']-60 for k in active),
                                default=time.monotonic()+poll_seconds)
                time.sleep(max(0., min(poll_seconds, next_stop-time.monotonic())))
    finally:
        handling_error = sys.exc_info()[0] is not None
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
        try:
            ledger.charge('control', time.monotonic()-cleanup_tick-waited)
        except RuntimeError:
            if not handling_error:
                raise
    if failures:
        write(campaign/'status.json', dict(status='FAILED_WORKERS_DRAINED_NO_RETRY', epoch=time.time(),
              failed_workers=failures, active=[], not_dispatched=[row['id'] for row in queue[index:]]))
        raise RuntimeError('Worker failure retained after active siblings finished: '
                           + ', '.join(row['id'] for row in failures))


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
            pending = adopt_recovery(campaign, gate, ledger) if gate.get('recovery') else None
        finally:
            ledger.charge('preparation', time.monotonic()-tick)
        while True:
            tick = time.monotonic()
            try:
                ready = resources(campaign, active_workers=0) if gate.get('recovery') else resources(campaign, startup=True)
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
        specs = {s.run_id: s for s in m.specifications(new_only=True)}
        queue = [dict(id=s.run_id, spec=s.identity(), allowance=gate['per_run_seconds'][s.name])
                 for s in (specs[k] for k in (pending or specs))]
        def train_command(row, job, deadline, metadata):
            write(metadata/'spec.json', row['spec'])
            return [sys.executable, '-B', str(Path(run.__file__)), '--job', str(job),
                    '--spec', str(metadata/'spec.json'), '--gate', str(gate_path), '--deadline', str(deadline)]
        dispatch(queue, campaign, ledger, 'training', train_command,
                 admit=lambda active: resources(campaign, active_workers=active) if gate.get('recovery') else resources(campaign),
                 check=lambda: campaign_size(campaign, gate['limits']), observe=observe,
                 retire_failed_slots=bool(gate.get('recovery')))
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
