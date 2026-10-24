"""One authorized recovery of the fixed 45-run campaign; frozen learners unchanged."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback

import psutil

FAILED_IDS = ('BCA_s5_H_pos_tuned', 'BCA_s6_SEQ_ref', 'BCA_s6_ER0.1_ref')
OLD_FAILURE_SHA = 'a1fe6ff1e42ec393868edfd145b6292cb12e444f20d716613c80c33df2db4d45'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def split_attempts(failure, specifications):
    rows = failure['budget']['completed']
    expected = [s.run_id for s in specifications]
    ids = [r['id'] for r in rows]
    success = [r['id'] for r in rows if r['success']]
    failed = [r['id'] for r in rows if not r['success']]
    if (failure['status'] != 'STOP_NO_RETRY' or failure['budget']['active']
            or len(ids) != len(set(ids)) or len(success) != 24 or tuple(failed) != FAILED_IDS
            or set(ids) != set(expected[:27])):
        raise ValueError('Recovery requires the observed 24-success/3-failure fixed prefix')
    pending = [s for s in specifications if s.run_id not in success]
    if len(pending) != 21 or [s.run_id for s in pending[:3]] != list(FAILED_IDS):
        raise ValueError('Recovery queue differs')
    return success, pending


def hardlink_tree(source, destination):
    """Regular hardlinks remain visible to the original rglob storage accounting."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_dir() or destination.exists():
        raise ValueError('Fresh destination and real source directory required')
    destination.mkdir()
    for directory, dirs, files in os.walk(source, followlinks=False):
        relative = Path(directory).relative_to(source)
        for name in dirs:
            original = Path(directory)/name
            if original.is_symlink():
                raise ValueError('Unexpected source directory symlink')
            (destination/relative/name).mkdir()
        for name in files:
            original = Path(directory)/name
            if original.is_symlink() or not original.is_file():
                raise ValueError('Unexpected nonregular source file')
            target = destination/relative/name
            os.link(original, target)
            if not os.path.samefile(original, target):
                raise ValueError('Hardlink identity differs')


def inherited_ledger(ctl, path, failure, cpu_seconds, wait_seconds):
    ledger = ctl.Ledger(path)
    ledger.used.update(copy.deepcopy(failure['budget']['used']))
    ledger.used['cpu'] += cpu_seconds
    ledger.used['control'] += wait_seconds
    # The old failed attempts remain in the immutable prior receipt and used totals.
    ledger.completed = [copy.deepcopy(r) for r in failure['budget']['completed'] if r['success']]
    if any(ledger.used[k] >= ctl.LIMITS[k] for k in ledger.used):
        raise ValueError('No remaining approved bucket')
    ledger.save()
    return ledger


def verify_reused(run, root, identity, gate):
    row = read(root/identity/'run/completion.json')
    if (row['status'] != run.COMPLETE or row['updates'] != 864
            or row['source_files'] != gate['sources']
            or run.m.Spec.from_dict(row['spec']).run_id != identity
            or set(row['points']) != {'1', '2', '3'}):
        raise ValueError('Existing successful trajectory is not reusable')
    for stage, point in row['points'].items():
        if not point['full_restore_verified'] or not point['inference_restore_verified']:
            raise ValueError('Existing endpoint lacks original restore evidence')
        records = [point['model'], point['mapping'], row['training'][stage], *point['scores'].values()]
        if point['memory'] is not None:
            records.append(point['memory'])
        for record in records:
            # Hash/size only; never parse blind predictions or open labels here.
            run.data.verify(root/identity/'run'/record['path'], record)
    return row


def validate_ready(gate, gate_path):
    ready_path = Path(gate['review_ready'])
    if not ready_path.exists():
        return None
    ready = read(ready_path)
    if (ready['status'] != 'REVIEW_CLOSED_AND_CPU_PASSED'
            or ready['authorization_sha256'] != sha(gate_path)):
        raise ValueError('Recovery review does not bind this authorization')
    required = {str(Path(__file__).resolve()),
                str(Path(gate_path).parent/'scripts/step28_continual_refill.py'),
                str(Path(gate_path).parent/'tests/test_step28_continual_recovery.py'),
                str(Path(gate_path).parent/gate['launcher_name'])}
    if {r['path'] for r in ready['sources']} != required or len(ready['sources']) != len(required):
        raise ValueError('Recovery source coverage differs')
    for record in ready['sources']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Reviewed recovery source changed')
    receipt = read(ready['cpu_receipt'])
    if (sha(ready['cpu_receipt']) != ready['cpu_receipt_sha256'] or receipt['exit_code'] != 0
            or receipt['recovery_cumulative_seconds'] > 180 or receipt['formal_data_reads'] != 0
            or receipt['campaign_cumulative_cpu_seconds'] > 1200):
        raise ValueError('Bounded handwritten CPU evidence absent')
    tested = {r['path']: r['sha256'] for r in receipt['sources']}
    if any(tested.get(r['path']) != r['sha256'] for r in ready['sources']):
        raise ValueError('Reviewed sources differ from CPU evidence')
    return ready


def resources():
    values = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    import shutil
    return dict(epoch=time.time(), free_gpu_bytes=int(values)*2**20, compute_apps=apps,
                host_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(Path.cwd()).free)


def resource_ready(row):
    return (row['free_gpu_bytes'] >= 30*2**30 and not row['compute_apps']
            and row['host_available_bytes'] >= 208*2**30 and row['disk_free_bytes'] >= 288*2**30)


def qualify_cpu(gate_path, job):
    gate, root = read(gate_path), gate_path.parent
    previous = [read(p) for p in (root/'qualification').glob('cpu*/receipt.json')]
    prior = gate.get('prior_recovery_cpu_seconds', 0) + sum(p['charged_seconds'] for p in previous)
    limit = min(180-prior, 1200-gate['prior_cpu_seconds']-prior)
    if limit <= 0 or job.exists() or job.parent != root/'qualification':
        raise ValueError('Fresh bounded CPU attempt required')
    job.mkdir(parents=True)
    source_paths = [Path(__file__).resolve(), root/'scripts/step28_continual_refill.py',
                    root/'tests/test_step28_continual_recovery.py',
                    root/gate['launcher_name'], gate_path]
    sources = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in source_paths]
    original = read(gate['original_gate'])
    for record in original['sources']:
        path = Path(gate['workspace'])/record['path']
        if sha(path) != record['sha256']:
            raise ValueError('Frozen source differs before CPU check')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               RECOVERY_FROZEN_WORKSPACE=gate['workspace'], RECOVERY_TEST_ROOT=str(job))
    command = [sys.executable, '-B', str(root/'tests/test_step28_continual_recovery.py'), '-v']
    started, peak, reason = time.monotonic(), 0, None
    with (job/'console.txt').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while child.poll() is None:
            try:
                p = psutil.Process(child.pid)
                peak = max(peak, sum(q.memory_info().rss for q in [p,*p.children(recursive=True)] if q.is_running()))
                used = sum(p.stat().st_size for p in job.rglob('*') if p.is_file())
                if time.monotonic()-started >= limit:
                    reason = 'CPU_TIME_LIMIT'
                elif peak > 4*2**30 or used > 256*2**20:
                    reason = 'CPU_RSS_OR_OUTPUT_LIMIT'
                if reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
        code = child.wait()
    elapsed = time.monotonic()-started
    receipt = dict(exit_code=code, reason=reason, wall_seconds=elapsed, charged_seconds=math.ceil(elapsed),
                   recovery_cumulative_seconds=prior+math.ceil(elapsed),
                   campaign_cumulative_cpu_seconds=gate['prior_cpu_seconds']+prior+math.ceil(elapsed),
                   peak_rss_bytes=peak, sources=sources, original_sources=original['sources'],
                   command=command, formal_data_reads=0, pretrained_model_loads=0, gpu=False)
    write(job/'receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('sources','original_sources','command')}))
    if code:
        raise SystemExit(code)


def wait_for_recovery(gate_path, launcher):
    gate, root = read(gate_path), Path(gate_path).parent
    if gate['status'] != 'AUTHORIZED_WAIT_BASED_RECOVERY':
        raise ValueError('Wait-based recovery not authorized')
    lock = root/'waiter_started.json'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), owner=psutil.Process().username(),
                       created=psutil.Process().create_time(), gate_sha256=sha(gate_path)), stream)
    charged = gate.get('prior_wait_control_seconds', 0.)
    try:
        while True:
            tick = time.monotonic()
            try:
                row, ready = resources(), validate_ready(gate, gate_path)
                launch = ready is not None and resource_ready(row)
                observed_cost = time.monotonic()-tick
                if gate['prior_control_seconds']+charged+observed_cost >= 1740:
                    raise RuntimeError('Remaining control allowance reserved for shutdown')
                write(root/'waiter_status.json', dict(status='READY' if launch else 'WAITING',
                      review_ready=ready is not None, resource_ready=resource_ready(row), resource=row,
                      control_seconds=charged+observed_cost, sleep_seconds=60))
            finally:
                # Exactly one charge per check, including exceptions; sleep is outside.
                charged += time.monotonic()-tick
            if launch:
                break
            time.sleep(60)
        with (root/'controller.console.txt').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'run',
                    '--gate', str(gate_path), '--wait-seconds', str(charged)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(root/'controller_process.json', dict(pid=child.pid, started_epoch=time.time()))
        # Only a newly dispatched trajectory can satisfy actual-update self-removal.
        after_launch = 0.
        while child.poll() is None:
            tick = time.monotonic()
            for identity in gate['pending_ids']:
                console = Path(gate['campaign'])/identity/'console.txt'
                if console.exists():
                    for line in console.read_text(encoding='utf-8').splitlines():
                        if line.startswith('{'):
                            item = json.loads(line)
                            if item.get('event') == 'updates' and item.get('completed', 0) >= 1:
                                launcher = Path(launcher).resolve()
                                if launcher.parent != root.resolve() or launcher.name != gate['launcher_name']:
                                    raise ValueError('Unexpected self-removal target')
                                launcher.unlink()
                                write(root/'waiter_status.json', dict(status='TRAINING_UPDATED_WAITER_REMOVED',
                                      controller_pid=child.pid, run_id=identity,
                                      first_update=item['completed'], control_seconds=charged,
                                      post_launch_control_seconds=after_launch+time.monotonic()-tick,
                                      post_launch_accounted_in_controller_30_second_tail=True))
                                return
            after_launch += time.monotonic()-tick
            if after_launch >= 25:
                raise RuntimeError('Post-launch observer work reached its prepaid tail; controller remains running')
            time.sleep(10)
        raise RuntimeError(f'Recovery controller exited before first real update: {child.returncode}')
    except BaseException:
        write(root/'waiter_failure.json', dict(traceback=traceback.format_exc(), control_seconds=charged))
        raise


def execute(gate_path, wait_seconds):
    started = time.monotonic()
    gate = read(gate_path)
    ready = validate_ready(gate, gate_path)
    if gate['status'] != 'AUTHORIZED_WAIT_BASED_RECOVERY' or ready is None:
        raise ValueError('Recovery implementation has not closed review')
    sys.path.insert(0, str(Path(gate['workspace'])/'scripts'))
    import step28_continual_five_control as ctl
    run, m = ctl.run, ctl.m
    original_gate = read(gate['original_gate'])
    old = Path(gate['original_campaign'])
    if sha(old/'failure.json') != OLD_FAILURE_SHA or original_gate['sources'] != run.sources():
        raise ValueError('Original failure or frozen computation sources changed')
    failure = read(old/'failure.json')
    success, pending = split_attempts(failure, m.specifications())
    if gate['pending_ids'] != [s.run_id for s in pending]:
        raise ValueError('Authorized pending queue changed')
    campaign = Path(gate['campaign'])
    if campaign.exists():
        raise ValueError('A recovery attempt already exists; no retry')
    preflight = ctl.resource_admission(original_gate, 0, campaign.parent, startup=True)
    campaign.mkdir()
    cpu = read(ready['cpu_receipt'])['recovery_cumulative_seconds'] - gate.get('inherited_recovery_cpu_seconds', 0)
    ledger = inherited_ledger(ctl, campaign/'budget.json', failure, cpu, wait_seconds)
    preparing = True
    try:
        write(campaign/'recovery.json', dict(status='AUTHORIZED_NEW_ATTEMPT',
              original_campaign=str(old), original_failure_sha256=OLD_FAILURE_SHA,
              prior_attempts=failure['budget']['completed'], reused_ids=success,
              pending_ids=[s.run_id for s in pending], authorization_sha256=sha(gate_path),
              reviewed_recovery=ready, preflight=preflight))
        for identity in success:
            verify_reused(run, old, identity, original_gate)
            hardlink_tree(old/identity, campaign/identity)
        prior_failed = campaign/'prior_failed'
        prior_failed.mkdir()
        for identity in FAILED_IDS:
            # Count retained failed payloads in the same original disk budget.
            hardlink_tree(old/identity, prior_failed/identity)
        if (old/'prior_failed').exists():
            hardlink_tree(old/'prior_failed', prior_failed/'earlier')
        preparing = False
        ledger.charge('control', time.monotonic()-started+30.)
        queue = [dict(id=s.run_id, spec=s.identity(), allowance=86400.,
                      projected_seconds=original_gate['per_run_projected_seconds'][s.name]) for s in pending]
        def command(row, job, deadline):
            run.data.write_json(job/'spec.json', row['spec'])
            return [sys.executable, '-B', str(Path(run.__file__)), '--job', str(job),
                    '--spec', str(job/'spec.json'), '--deadline', str(deadline), '--gate', gate['original_gate']]
        import step28_continual_refill as refill
        receipts = refill.dispatch(ctl, queue, campaign, ledger, 'training', command,
                                  admit=lambda active: refill.slot_available(original_gate, active, campaign))
        tick = time.monotonic()
        records = [read(campaign/s.run_id/'run/completion.json') for s in m.specifications()]
        ctl.joint_opening(records, original_gate['sources'])
        result = dict(status='ALL_45_BLIND_COMPLETE', run_ids=[s.run_id for s in m.specifications()],
                      receipts=ledger.completed, sources=original_gate['sources'], budget=ledger.snapshot(),
                      recovery='recovery.json')
        write(campaign/'training_complete.json', result)
        ledger.charge('control', time.monotonic()-tick)
        import step28_continual_five_evaluate as evaluator
        for identity, action, bucket in (('old_dev','old-dev','evaluation'), ('generation','generate','generation'),
                                         ('confirmation','confirmation','confirmation'), ('analysis','analyze','analysis')):
            def post_command(row, job, deadline):
                return [sys.executable, '-B', str(Path(evaluator.__file__)), action, '--campaign', str(campaign),
                        '--job', str(job), '--gate', gate['original_gate'], '--deadline', str(deadline)]
            ctl.dispatch([dict(id=identity, allowance=ledger.available(bucket))], campaign, ledger, bucket,
                         post_command, max_workers=1)
        write(campaign/'completion.json', {**result, 'status':'ALL_45_AND_FROZEN_CONFIRMATION_ANALYSIS_COMPLETE',
                                          'budget':ledger.snapshot()})
    except BaseException as error:
        original_traceback = traceback.format_exc()
        accounting_error = None
        if preparing:
            preparing = False
            try:
                ledger.charge('control', time.monotonic()-started+30.)
            except BaseException as charge_error:
                accounting_error = repr(charge_error)
        write(campaign/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error),
              traceback=original_traceback, accounting_error=accounting_error, budget=ledger.snapshot()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('wait', 'run', 'cpu'))
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--self-delete', type=Path)
    parser.add_argument('--wait-seconds', type=float, default=0.)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.action == 'wait':
        wait_for_recovery(args.gate.resolve(), args.self_delete)
    elif args.action == 'cpu':
        qualify_cpu(args.gate.resolve(), args.job.resolve())
    else:
        execute(args.gate.resolve(), args.wait_seconds)
