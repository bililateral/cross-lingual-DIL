"""Single-slot dispatch shared by the VendorLink and FOREVER formal batches.

Only the two callers build commands and run identities. No retry, concurrent
slots, model selection, or evaluation access is granted by this helper.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(row: dict) -> Path:
    path = Path(row['path'])
    actual = record(path)
    if any(actual[k] != row[k] for k in ('bytes', 'sha256')):
        raise ValueError('Pinned file identity differs: ' + str(path))
    return path


def gate(path: Path, experiment: str, action: str) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if (sys.platform != 'linux' or value.get('experiment') != experiment
            or value.get('approved') is not True or value.get('review_closed') is not True
            or action not in value.get('actions', []) or value.get('workers') != 1
            or not value.get('sources') or not value.get('authorization')):
        raise ValueError('Approved, reviewed, source-bound single-slot scope required')
    verify(value['authorization'])
    workspace = Path(value['workspace']).resolve()
    for row in value['sources']:
        source = verify(row)
        if not source.resolve().is_relative_to(workspace):
            raise ValueError('Source outside the isolated workspace')
    required = {str(Path(__file__).resolve()), str(Path(sys.argv[0]).resolve())}
    if not required.issubset({str(Path(r['path']).resolve()) for r in value['sources']}):
        raise ValueError('Actual entry/queue not covered by sources')
    return value


def resources(allowed: dict, campaign: Path) -> dict:
    import shutil
    free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 2**20
    limits = allowed['limits']
    used = sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file() and not p.is_symlink())
    ram, disk = psutil.virtual_memory().available, shutil.disk_usage(campaign).free
    return dict(free_gpu_bytes=free, free_ram_bytes=ram, free_disk_bytes=disk, campaign_bytes=used,
        ready=(free >= limits['free_gpu_bytes'] and ram >= limits['rss_bytes']
               and disk >= max(0, limits['disk_bytes'] - used) + limits['free_reserve_bytes']))


def stop(process) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def check_running(process, allowed, campaign):
    """Also cover preparation/statistics loops which do not call a model budget."""
    import shutil
    limits = allowed['limits']
    child = psutil.Process(process.pid)
    rss = sum(p.memory_info().rss for p in [psutil.Process(), child, *child.children(recursive=True)])
    used = sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file() and not p.is_symlink())
    if (rss > limits['rss_bytes'] or used > limits['disk_bytes']
            or shutil.disk_usage(campaign).free < limits['free_reserve_bytes']):
        raise RuntimeError('Actual worker/parent resource cap exceeded')


def dispatch(allowed: dict, rows: list[dict], command, validate, followups=(), preflight=None, started=None) -> dict:
    """Reserve all remaining runs; charge actual outer elapsed including failures."""
    campaign = Path(allowed['campaign'])
    campaign.mkdir()  # An existing/failed batch cannot silently restart.
    used = dict(allowed['prior_used_seconds'])
    limits = allowed['budget_seconds']
    tasks = [*rows, *followups]
    if (len({r['id'] for r in rows}) != len(rows) or not rows
            or any(used.get(k, 0) < 0 or used.get(k, 0) > v for k, v in limits.items())
            or len({r['id'] for r in tasks}) != len(tasks)
            or any(sum(r['seconds'] for r in tasks if r.get('bucket', 'training') == bucket)
                   + used.get(bucket, 0) > limit for bucket, limit in limits.items())):
        raise ValueError('Whole fixed queue does not fit the approved budget')
    result = dict(status='WAITING_OR_TRAINING', used=used, limits=limits, completed=[], active=None)
    process = None
    control_tick = time.monotonic() if started is None else started
    def charge_control():
        nonlocal control_tick
        now = time.monotonic()
        used['control'] = used.get('control', 0) + now - control_tick
        control_tick = now
        if used['control'] >= limits['control']:
            raise TimeoutError('Controller allowance exhausted')
        write(campaign/'budget.json', result)
    def pause(seconds):
        nonlocal control_tick
        charge_control()
        started = time.monotonic()
        time.sleep(seconds)
        control_tick += time.monotonic() - started  # Sleeping is not controller work.
    try:
        if preflight is not None:
            preparation_started = time.monotonic()
            success = False
            try:
                preflight()
                success = True
            finally:
                elapsed = time.monotonic() - preparation_started
                control_tick += elapsed
                used['preparation'] = used.get('preparation', 0) + elapsed
                write(campaign/'preflight.json', dict(actual_seconds=elapsed, success=success))
                if used['preparation'] > limits['preparation']:
                    raise TimeoutError('Preflight exceeds preparation allowance')
        for index, row in enumerate(tasks):
            bucket = row.get('bucket', 'training')
            while row.get('gpu', True):
                observation = resources(allowed, campaign)
                with (campaign/'admission.jsonl').open('a', encoding='utf-8') as out:
                    out.write(json.dumps(dict(time=time.time(), run_id=row['id'], **observation)) + '\n')
                if observation['ready']:
                    break
                pause(30)
            job = campaign/row['id']
            job.mkdir()
            started = time.monotonic()
            result['active'] = dict(id=row['id'], bucket=bucket, started_epoch=time.time(), seconds=row['seconds'])
            write(campaign/'budget.json', result)
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'CUDA_VISIBLE_DEVICES': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            success = False
            with (job/'console.txt').open('x', encoding='utf-8') as log:
                try:
                    argv = ['taskset', '-c', str(psutil.Process().cpu_affinity()[0]),
                            *command(row, job, started + row['seconds'])]
                    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                        env=env, cwd=allowed['workspace'], start_new_session=True)
                    write(job/'process.json', dict(pid=process.pid, command=argv,
                        created=psutil.Process(process.pid).create_time(), owner=psutil.Process(process.pid).username()))
                    while process.poll() is None:
                        try:
                            check_running(process, allowed, campaign)
                        except (psutil.NoSuchProcess, FileNotFoundError):
                            pass  # A child exit or an atomic rename is rechecked next poll.
                        if time.monotonic() >= started + row['seconds'] - 60:
                            raise TimeoutError('Run reached its shutdown reserve')
                        pause(5)
                    if process.returncode != 0:
                        raise RuntimeError('Worker failed; no automatic retry or next dispatch')
                    validate(row, job)
                    success = True
                finally:
                    if process is not None:
                        stop(process)
                    elapsed = time.monotonic() - started
                    used[bucket] = used.get(bucket, 0) + elapsed
                    result['completed'].append(dict(id=row['id'], bucket=bucket, actual_seconds=elapsed,
                        success=success and elapsed <= row['seconds']))
                    result['active'] = None
                    process = None
                    charge_control()
                    if elapsed > row['seconds'] or used[bucket] > limits[bucket]:
                        raise TimeoutError('Actual elapsed exceeds the allowance; charges are not clipped')
            if index == len(rows) - 1:
                result['status'] = 'ALL_TRAINING_COMPLETE_EVALUATION_CLOSED'
                write(campaign/'training_complete.json', result)
        if followups:
            result['status'] = 'ALL_FORMAL_ACTIONS_COMPLETE'
            write(campaign/'completion.json', result)
        charge_control()
        return result
    except BaseException as error:
        result.update(status='STOP_NO_AUTOMATIC_RETRY', error=repr(error))
        write(campaign/'failure.json', result)
        raise
