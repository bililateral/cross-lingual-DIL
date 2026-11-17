"""One approved beta batch, after the complete Chinese campaign and resources."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time

import psutil

from step30_evolution_control import kernel_identity, resource_observation
from step35_vendorlink_account_control import campaign, permit, read
from step35_vendorlink_account_io import pinned, record, write
from step35_vendorlink_account_policy import BETA_PROTOCOL


def dependency_observation(g):
    """Read operational metadata only: never Chinese scores, labels or owners."""
    dep = g['dependency']
    root, state = Path(dep['campaign']), Path(dep['state'])
    request = read(pinned(dep['request']))
    original = read(pinned(dep['gate']))
    if root != Path(request['campaign']) or dep['gate']['path'] != request['original_gate']:
        raise ValueError('Chinese dependency binding differs')
    for name in ('failure.json', 'completion.json'):
        if name == 'failure.json' and any((p/name).exists() for p in (root, state)):
            raise RuntimeError('Chinese dependency failed; no automatic beta launch')
    identity = dep['controller']
    try:
        process = psutil.Process(identity['pid'])
        same = (kernel_identity(process.pid) == {k: identity[k] for k in ('start_ticks', 'boot_id')}
                and process.username() == identity['owner'] and process.cwd() == identity['cwd']
                and process.cmdline() == identity['command'])
        live = same and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        live, same = False, False
    complete = root/'completion.json'
    if not complete.exists():
        if not live:
            raise RuntimeError('Chinese supervisor exited or changed before complete closure')
        return dict(ready=False, status='WAITING_CHINESE_COMPLETION', controller=identity['pid'])
    value = read(complete)
    update = value.get('controller_update', {})
    expected_sources = request['updated_sources']
    if (value.get('status') != 'COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED'
            or value.get('new_trajectories') != 63 or value.get('full_reused') != 9
            or value.get('sources') != original['sources']
            or update.get('request') != dep['request']['path']
            or update.get('updated_sources') != expected_sources):
        raise ValueError('Chinese whole-campaign completion identity differs')
    if not (state/'completion.json').exists():
        if not live:
            raise RuntimeError('Chinese supervisor exited before its final closure receipt; no automatic beta launch')
        return dict(ready=False, status='WAITING_CHINESE_SUPERVISOR_CLOSURE')
    if read(state/'completion.json').get('status') != 'CONTROLLED_CAMPAIGN_COMPLETE':
        raise ValueError('Chinese supervisor closure differs')
    related = []
    for proc in psutil.process_iter(['pid', 'username', 'cmdline', 'status']):
        if proc.info['username'] != identity['owner'] or proc.info['status'] == psutil.STATUS_ZOMBIE:
            continue
        command = proc.info['cmdline'] or []
        if any(str(request['workspace']) in arg or dep['request']['path'] == arg for arg in command):
            related.append(proc.pid)
    if related:
        return dict(ready=False, status='WAITING_CHINESE_PROCESS_RELEASE', pids=related)
    return dict(ready=True, status='CHINESE_COMPLETE_AND_RELEASED', completion=record(complete),
                supervisor_completion=record(state/'completion.json'))


def wait_then_run(g, gate_path, started):
    if g['protocol'] != BETA_PROTOCOL:
        raise ValueError('Only the approved single beta batch may use this waiter')
    root = Path(g['experiment_root'])
    wait = root/'wait'
    if Path(g['campaign']).exists():
        raise ValueError('Existing campaign may not be recreated or restarted')
    # An existing attempt is never silently resumed, even if its process exited.
    wait.mkdir()
    with (wait/'claim.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pinned(g['waiter'])
        me = psutil.Process()
        identity = dict(pid=me.pid, owner=me.username(), command=me.cmdline(), cwd=me.cwd(),
                        **kernel_identity(me.pid))
        write(wait/'controller.json', dict(identity=identity, gate=record(gate_path), automatic_retry=False))
        active_seconds = time.monotonic()-started
        tick = None
        try:
            while True:
                tick = time.monotonic()
                dependency = dependency_observation(g)
                observation = (resource_observation(g, root, True, limits=g['limits'])
                               if dependency['ready'] else None)
                ready = dependency['ready'] and observation['ready']
                status = 'READY' if ready else (dependency['status'] if not dependency['ready'] else 'WAITING_RESOURCES')
                # Sum measured active segments, then round once; sleeping is free.
                spent = active_seconds+time.monotonic()-tick
                charged = math.ceil(spent)
                if g['prior_seconds']['auxiliary']+charged+150 >= g['caps']['auxiliary']:
                    raise TimeoutError('Wait monitoring reached the auxiliary launch/cleanup reserve')
                write(wait/'status.json', dict(status=status, observed_epoch=time.time(), identity=identity,
                    dependency=dependency, resources=observation, active_seconds=spent,
                    auxiliary_used_seconds=g['prior_seconds']['auxiliary']+charged,
                    sleep_charged=False, automatic_retry=False))
                active_seconds += time.monotonic()-tick
                tick = None
                if ready:
                    write(wait/'ready.json', dict(status='READY', gate=record(gate_path), dependency=dependency,
                          resources=observation, charged_seconds=math.ceil(active_seconds), identity=identity))
                    break
                time.sleep(120)
        except BaseException as error:
            # Persist all measured overhead; neither restart nor scientific stop.
            if tick is not None:
                active_seconds += time.monotonic()-tick
            write(wait/'failure.json', dict(error=repr(error), automatic_retry=False,
                  charged_seconds=math.ceil(active_seconds), prior_seconds=g['prior_seconds']))
            raise
        # Same process and lock; campaign never runs alongside a second waiter.
        campaign(g, gate_path)


def main(started=None):
    started = time.monotonic() if started is None else started
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    g = permit(args.gate)
    wait_then_run(g, args.gate, started)
