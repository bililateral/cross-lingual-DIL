"""One explicitly authorized ER retry, using the reviewed freeze/drain bridge."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

TARGET = 'ABC_s6_ER'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pinned(row):
    if record(row['path']) != row:
        raise ValueError('Bound file changed: '+row['path'])
    return Path(row['path'])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retry_continuation(snapshot, ordered_ids):
    """Resolve one authorized failed attempt, not the history of other failures."""
    value = copy.deepcopy(snapshot)
    failed = [r for r in value['completed'] if not r['success']]
    if (value['active'] or len(failed) != 1 or failed[0]['id'] != TARGET
            or failed[0]['slot'] != 0 or not failed[0]['released']
            or value['retired_slots'] != [0]
            or ordered_ids[failed[0]['index']] != TARGET):
        raise ValueError('Only the released ABC_s6_ER failure in slot0 is authorized')
    indices = [r['index'] for r in value['completed']]
    if (len(indices) != len(set(indices))
            or any(ordered_ids[r['index']] != r['id'] for r in value['completed'])
            or value['pending'] != [name for i, name in enumerate(ordered_ids) if i not in indices]):
        raise ValueError('Fixed queue or completed identity differs')
    value['completed'] = [r for r in value['completed'] if r['success']]
    done = {r['index'] for r in value['completed']}
    value['pending'] = [name for i, name in enumerate(ordered_ids) if i not in done]
    value['retired_slots'] = []
    value['status'] = 'AUTHORIZED_ONE_RETRY_PENDING'
    return value, failed[0]


def beta_replacement_gate(old, request_path, controller_identity, state, prior_auxiliary):
    """Same beta science/sources/caps; bind its real replacement supervisor."""
    value = copy.deepcopy(old)
    if prior_auxiliary < old['prior_seconds']['auxiliary']:
        raise ValueError('Waiting costs may not be reset')
    value['dependency'].update(request=record(request_path), controller=controller_identity, state=str(state))
    value['prior_seconds']['auxiliary'] = prior_auxiliary
    return value


def check_allowance(budget, expected_caps, preparation, active_count):
    if (budget['caps'] != expected_caps or not 0 <= preparation <= 1800
            or budget['used_seconds']['auxiliary']+preparation+120 >= budget['caps']['auxiliary']):
        raise TimeoutError('Insufficient unchanged auxiliary allowance before touching processes')
    if budget['used_seconds']['training']+(active_count+1)*21600 > budget['caps']['training']:
        raise TimeoutError('Cannot reserve active workers and one complete retry')


def stop_exact(bridge, identity, sig):
    process = bridge.same_process(identity)
    process.send_signal(sig)
    end = time.monotonic()+10
    while psutil.pid_exists(process.pid):
        if psutil.Process(process.pid).status() == psutil.STATUS_ZOMBIE:
            return
        if time.monotonic() >= end:
            raise RuntimeError('Owned predecessor did not exit')
        time.sleep(.02)


def replace_beta(request, request_path, bridge, state):
    beta = request['beta']
    root = Path(beta['root'])
    old_gate = read(pinned(beta['gate']))
    old = bridge.same_process(beta['identity'])
    if old.children(recursive=True) or (root/'campaign').exists():
        raise ValueError('Only an idle, childless beta waiter may be handed over')
    wait = root/'wait'
    archived = root/'wait_before_chinese_retry_s6'
    gate_path = root/'gate_after_chinese_retry_s6.json'
    if archived.exists() or gate_path.exists():
        raise ValueError('No second waiting handover')
    started = time.monotonic()
    stop_exact(bridge, beta['identity'], signal.SIGTERM)
    status = read(wait/'status.json')
    if (status['status'] != 'WAITING_CHINESE_COMPLETION'
            or status['identity'] != read(pinned(beta['launch']))['identity']):
        raise ValueError('Waiting status identity differs')
    # Preserve the entire original wait attempt and its claim; no scientific job ran.
    wait.rename(archived)
    me = {**bridge.identity(os.getpid()),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    prior = old_gate['prior_seconds']['auxiliary']+math.ceil(status['active_seconds'])+1
    # Finite deployment reservation, inside the unchanged beta auxiliary bucket.
    prior += 90
    new_gate = beta_replacement_gate(old_gate, request_path, me, state, prior)
    if prior+150 >= new_gate['caps']['auxiliary']:
        raise TimeoutError('Beta handover has insufficient unchanged auxiliary allowance')
    write(gate_path, new_gate)
    command = [sys.executable, '-B', '-u', str(pinned(beta['entry'])), '--gate', str(gate_path)]
    with (state/'beta_console.txt').open('x') as log:
        child = subprocess.Popen(command, cwd=old_gate['workspace'], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    end = time.monotonic()+60
    while not (wait/'status.json').exists():
        if child.poll() is not None or time.monotonic() >= end:
            raise RuntimeError('Replacement beta waiter did not acknowledge; do not retry')
        time.sleep(.1)
    current = read(wait/'status.json')
    if current['status'] != 'WAITING_CHINESE_COMPLETION' or current['identity']['pid'] != child.pid:
        raise RuntimeError('Unexpected replacement waiter state')
    write(state/'beta_handover.json', dict(old_launch=beta['launch'], old_wait=str(archived),
        old_status=status, gate=record(gate_path), identity=current['identity'],
        prior_auxiliary_seconds=prior, old_wait_tail_conservative_seconds=1,
        handover_seconds_conservative=90, handover_elapsed_seconds=time.monotonic()-started,
        scientific_sources_unchanged=True, training_started=False))


def execute(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    if (request.get('approved') is not True or request.get('review_closed') is not True
            or request.get('retry_task') != TARGET or request.get('retry_count') != 1):
        raise ValueError('Explicit reviewed one-retry authorization required')
    for row in [request['review'], request['authorization'], *request['updated_sources']]:
        pinned(row)
    state = request_path.parent/'state'
    state.mkdir(exist_ok=True)
    with (state/'claim.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state/'controller.json').exists() or (state/'frozen.json').exists():
            raise ValueError('This handover has already been attempted')
        workspace, campaign = Path(request['workspace']), Path(request['campaign'])
        sys.path.insert(0, str(workspace/'scripts'))
        bridge = load(pinned(request['bridge']), 'reviewed_bridge')
        control = load(pinned(request['updated_control']), 'reviewed_control')
        dispatch = load(pinned(request['updated_dispatch']), 'reviewed_dispatch')
        allowed = control.run.gate(Path(request['original_gate']), 'train')
        if str(workspace) != allowed['workspace'] or str(campaign) != allowed['campaign']:
            raise ValueError('Original campaign differs')
        budget_now = read(campaign/'budget.json')
        ids = [s.run_id for s in control.m.specifications()]
        snapshot = read(campaign/'training/dispatch.json')
        check_allowance(budget_now, control.run.BUDGETS, request['preparation_seconds'], len(snapshot['active']))
        probe = copy.deepcopy(snapshot)
        # The actual drain retains these workers; this only validates the one failed task now.
        probe['pending'] = [n for n in ids if n not in {r['id'] for r in probe['completed']}]
        probe['active'] = {}
        retry_continuation(probe, ids)
        if (campaign/'pause_dispatch').exists():
            raise ValueError('A new pause request must not be overridden')
        beta_proc = bridge.same_process(request['beta']['identity'])
        if beta_proc.children(recursive=True) or (Path(request['beta']['root'])/'campaign').exists():
            raise ValueError('Beta is no longer an idle waiter')
        snapshot, budget, active = bridge.freeze(control, request, campaign, state)
        try:
            write(state/'controller.json', dict(identity=bridge.identity(os.getpid()),
                  phase='DRAINING_BEFORE_EXPLICIT_RETRY', request=str(request_path)))
        except BaseException:
            bridge.same_process(request['old_controller']).send_signal(signal.SIGCONT)
            raise
        drained = False
        try:
            continuation = bridge.drain(control, request, campaign, state, snapshot, budget, active)
            drained = True
            effective, failed = retry_continuation(continuation['dispatch'], ids)
            job = campaign/'training'/TARGET
            if (job.is_symlink() or job.resolve() != job.absolute()
                    or job.stat().st_uid != os.getuid() or not (job/'failure.json').exists()):
                raise ValueError('Failed job ownership/path differs')
            if read(job/'dispatch_receipt.json') != failed or (job/'run/completion.json').exists():
                raise ValueError('Only the failed, incomplete attempt may be archived')
            archive = campaign/'attempts'/TARGET/'attempt0'
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise ValueError('Failed attempt already archived; no automatic retry')
            files = [record(p) for p in sorted(job.rglob('*')) if p.is_file()]
            job.rename(archive)
            for row in files:
                new_path = archive/Path(row['path']).relative_to(job)
                if {**record(new_path), 'path': row['path']} != row:
                    raise RuntimeError('Failed attempt changed during archival')
            write(state/'attempt_resolution.json', dict(task=TARGET, slot=0, retry_count=1,
                failed_receipt=failed, archived_at=str(archive), original_path=str(job),
                original_files=files, new_attempt_path=str(job), effective_queue=effective,
                failed_fees_retained=True, authorization=request['authorization']))
            continuation['dispatch'] = effective
            continuation['controller_update'] = dict(request=str(request_path),
                updated_sources=request['updated_sources'], review=request['review'])
            write(state/'continuation.json', continuation)
            # The old beta waiter sees the still-live stopped predecessor until its own handover.
            replace_beta(request, request_path, bridge, state)
            stop_exact(bridge, request['old_controller'], signal.SIGKILL)
            write(state/'controller.json', dict(identity=bridge.identity(os.getpid()),
                  phase='PAUSED_CONTINUATION', request=str(request_path)))
            control.execute(Path(request['original_gate']), continuation=continuation,
                            dispatch_impl=dispatch.dispatch,
                            dispatch_paused=lambda: (campaign/'pause_dispatch').exists())
            write(state/'completion.json', dict(status='CONTROLLED_CAMPAIGN_COMPLETE',
                  epoch=time.time(), retry_task=TARGET, retry_count=1))
        except BaseException as error:
            write(state/'failure.json', dict(error=repr(error), no_automatic_retry=True))
            if not (campaign/'failure.json').exists():
                write(campaign/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error),
                      used_seconds=read(campaign/'budget.json')['used_seconds']))
            if drained:
                try:
                    stop_exact(bridge, request['old_controller'], signal.SIGKILL)
                except (psutil.NoSuchProcess, FileNotFoundError):
                    pass
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    execute(parser.parse_args().request)
