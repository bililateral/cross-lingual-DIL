"""One ABC successor, after the whole H_pos_history campaign and resource admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def verify(row):
    path = Path(row['path'])
    if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Waiter binding differs: '+str(path))
    return path


def related(binding):
    """Match owner, exact process identity and the predecessor workspace/arguments."""
    expected = binding['process']
    tokens = {binding['gate']['path'], binding['config'], binding['listener']}
    found = []
    for process in psutil.process_iter(['pid', 'username', 'cmdline', 'cwd', 'status', 'create_time']):
        try:
            info = process.info
            if process.pid == os.getpid() or info['status'] == psutil.STATUS_ZOMBIE:
                continue
            if (info['pid'] == expected['pid'] and info['create_time'] == expected['created']
                    and (info['username'] != expected['owner'] or info['cmdline'] != expected['command']
                         or info['cwd'] != expected['cwd'])):
                raise RuntimeError('Pinned predecessor process changed identity')
            if (info['username'] == expected['owner']
                    and (info['cwd'] == expected['cwd'] or tokens.intersection(info['cmdline'] or []))):
                found.append(info['pid'])
        except psutil.NoSuchProcess:
            pass
    return sorted(found)


def dispatch_complete(path, ids, sources):
    value = read(path)
    rows = value['completed']
    if (value['status'] != 'ALL_COMPLETE_RELEASED' or value['sources'] != sources
            or value['pending'] or value['active'] or value['retired_slots']
            or len(rows) != len(ids) or {r['id'] for r in rows} != set(ids)
            or any(r['success'] is not True or r['released'] is not True or r['error'] is not None
                   or not math.isfinite(r['actual_seconds']) or r['actual_seconds'] < 0 for r in rows)):
        raise ValueError('Incomplete predecessor action: '+str(path))


def history_complete(binding):
    """Completion/dispatch metadata only; never load scores, SQLite, labels or weights."""
    gate = read(verify(binding['gate']))
    root = Path(gate['campaign'])
    completion = read(root/'completion.json')
    if completion['sources'] != gate['sources']:
        raise ValueError('Predecessor completion sources differ')
    for number in range(12):
        folder = root/f'trial_{number:02}'
        dispatch_complete(folder/'dispatch.json',
                          [f'search_{number:02}_{order}_s7_H_pos_history' for order in ('ABC','BCA','CAB')], gate['sources'])
        dispatch_complete(folder/'evaluation_dispatch/dispatch.json', ['evaluation'], gate['sources'])
    selected = completion['selected']
    if selected['number'] not in range(12):
        raise ValueError('Invalid predecessor selected trial')
    anchor = selected['parameters'] == dict(history_weight=.1, teacher_weight=.5)
    if anchor:
        if completion['status'] != 'COMPLETE_ANCHOR_SELECTED_NO_NEW_CONFIRMATION':
            raise ValueError('Anchor completion status differs')
    else:
        if completion['status'] != 'COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED':
            raise ValueError('Non-anchor completion status differs')
        dispatch_complete(root/'replication/dispatch.json',
                          [f'replication_{selected["number"]:02}_{order}_s{seed}_H_pos_history'
                           for order in ('ABC','BCA','CAB') for seed in (4,5,6)], gate['sources'])
        for folder, job in ((root/'replication', 'evaluation'), (root, 'generation'),
                            (root, 'confirmation'), (root, 'analysis')):
            dispatch_complete(folder/(job+'_dispatch')/'dispatch.json', [job], gate['sources'])
    budget = read(root/'budget.json')
    if (budget['used_seconds'] != completion['used_seconds']
            or set(completion['used_seconds']) != {'training','confirmation','auxiliary'}
            or any(not math.isfinite(v) or v < gate['prior_used_seconds'][k] or v >= cap
                   for k, cap in dict(training=972000,confirmation=21600,auxiliary=14400).items()
                   for v in [completion['used_seconds'][k]])):
        raise ValueError('Predecessor accounting incomplete')
    state = Path(binding['state'])
    if read(state/'completion.json')['status'] != 'FORMAL_CONTROLLER_COMPLETED':
        raise ValueError('Predecessor wrapper did not complete')
    return completion['status']


def predecessor(binding, processes=related):
    gate = read(verify(binding['gate']))
    root, state = Path(gate['campaign']), Path(binding['state'])
    if (root/'failure.json').exists() or (state/'failure.json').exists():
        raise RuntimeError('H_pos_history failed; successor will not start')
    alive = processes(binding)
    if alive:
        return dict(ready=False, reason='PREDECESSOR_STILL_RUNNING', pids=alive)
    if not (root/'completion.json').exists() or not (state/'completion.json').exists():
        raise RuntimeError('H_pos_history exited without whole-campaign completion')
    status = history_complete(binding)
    return dict(ready=True, reason='WHOLE_HISTORY_COMPLETE_AND_EXITED', pids=[], completion=status)


def retire(config, reason, update=None):
    listener, state = Path(config['listener']), Path(config['state'])
    if not listener.exists():
        return
    verify(config['active_source'])
    write(state/'retirement.json', dict(status='PLANNED', reason=reason, update=update))
    listener.unlink()
    write(state/'retirement.json', dict(status='DELETED', reason=reason, update=update, epoch=time.time()))


def observe_update(config, campaign):
    point = Path(campaign)/'first_update.json'
    if point.exists() and Path(config['listener']).exists():
        value = read(point)
        job = Path(value['job']).resolve()
        if not job.is_relative_to(Path(campaign).resolve()):
            raise ValueError('First update outside campaign')
        with (job/'run/updates.jsonl').open() as stream:
            update = json.loads(stream.readline())
        if (value['update'] != update or update.get('stage') != 1
                or update.get('total_updates') != 1 or update.get('optimizer_updates') != 1):
            raise ValueError('First optimizer update not corroborated by actual worker log')
        retire(config, 'FIRST_REAL_OPTIMIZER_UPDATE', value)


def execute(config_path):
    config = read(config_path)
    if config.get('approved') is not True or config.get('review_closed') is not True:
        raise ValueError('Approved and closed wait deployment required')
    state = Path(config['state'])
    state.mkdir()  # Unique claim, including failed attempts. No automatic restart.
    own = psutil.Process()
    write(state/'process.json', dict(pid=own.pid, owner=own.username(), created=own.create_time(),
                                   command=own.cmdline(), cwd=own.cwd()))
    used = config['prior_auxiliary_seconds']
    tick, handed_over = time.monotonic(), False
    monitor = None
    try:
        gate = read(verify(config['gate_template']))
        verify(config['authorization'])
        verify(config['review_evidence'])
        verify(config['active_source'])
        if (Path(__file__).resolve() != Path(config['listener']).resolve()
                or gate['authorization'] != config['authorization']
                or gate['prior_used_seconds']['auxiliary'] != used
                or Path(gate['campaign']).exists()):
            raise ValueError('Exact fresh listener, authorization and fees required')
        os.chdir(gate['workspace'])
        sys.path.insert(0, str(Path(gate['workspace'])/'scripts'))
        import step29_english_control as control
        # Delegate to the pinned production entry in this process. The self-deleting
        # wrapper is separately bound above, not in every future worker's source gate.
        sys.argv = [control.__file__, '--gate', str(config['gate_template']['path'])]
        control.formal.gate(config['gate_template']['path'])
        while True:
            previous = predecessor(config['predecessor'])
            available = None
            if previous['ready']:
                if monitor is None:
                    monitor = control.queue.GpuMemory()
                available = control.queue.resources(gate, Path(gate['workspace']), monitor.snapshot())
            ready = bool(available and available['ready'])
            write(state/'waiting.json', dict(epoch=time.time(), status='READY' if ready else 'WAITING',
                                            predecessor=previous, resources=available))
            now = time.monotonic()
            used += now-tick
            tick = now
            write(state/'budget.json', dict(auxiliary_used_seconds=used, idle_sleep_charged=False, limit=14400))
            if not math.isfinite(used) or used >= 14400-150:
                raise TimeoutError('Waiter auxiliary budget exhausted')
            if ready:
                break
            begin_sleep = time.monotonic()
            time.sleep(5)
            tick += time.monotonic()-begin_sleep
        monitor.close(); monitor = None
        # Activation/save/import costs are charged through the same auxiliary bucket.
        gate['prior_used_seconds']['auxiliary'] = used + time.monotonic()-now
        activated = state/'formal_gate.json'
        write(activated, gate)
        write(state/'start.json', dict(epoch=time.time(), gate=str(activated), pid=os.getpid(),
                                      prior_auxiliary_seconds=gate['prior_used_seconds']['auxiliary']))
        handed_over = True
        control.execute(activated, observe=lambda: observe_update(config, gate['campaign']))
        result = read(Path(gate['campaign'])/'completion.json')
        if Path(config['listener']).exists():
            if result['status'] != 'STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING':
                raise RuntimeError('Controller completed without a corroborated training update')
            retire(config, 'PRECHECK_ENDED_WITHOUT_TRAINING')
        write(state/'completion.json', dict(epoch=time.time(), status='ABC_CONTROLLER_COMPLETED',
                                           campaign_status=result['status']))
    except BaseException as error:
        if not handed_over:
            used += max(0, time.monotonic()-tick)
            write(state/'budget.json', dict(auxiliary_used_seconds=used, idle_sleep_charged=False, limit=14400))
        write(state/'failure.json', dict(epoch=time.time(), status='STOP_NO_RETRY', error=repr(error),
                                        handed_over=handed_over))
        raise
    finally:
        if monitor is not None:
            monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    execute(parser.parse_args().config.resolve())
