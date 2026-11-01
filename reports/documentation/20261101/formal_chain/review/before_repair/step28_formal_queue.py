"""Bounded formal dispatch: FOREVER slots retire independently; followups are serial."""
from __future__ import annotations

import hashlib
import ctypes
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
            or action not in value.get('actions', [])
            or value.get('workers') not in ((1, 3) if experiment == 'FOREVER' else (1,))
            or not value.get('sources') or not value.get('authorization')):
        raise ValueError('Approved, reviewed, source-bound formal scope required')
    if value['workers'] == 3 and (
            value.get('worker_failure_policy') != 'retire_failed_slot_continue_healthy'
            or value['limits'].get('allocator_bytes') != 9*2**30
            or value['limits'].get('process_gpu_bytes') != 10*2**30
            or value['limits']['free_gpu_bytes'] != 10*2**30):
        raise ValueError('Three-slot FOREVER needs the approved 9+1 GiB limits')
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


class GpuMemory:
    """Read existing NVML directly; no CUDA allocation or per-poll subprocess."""
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    class Process(ctypes.Structure):
        _fields_ = [('pid', ctypes.c_uint), ('used', ctypes.c_ulonglong),
                    ('gpu_instance', ctypes.c_uint), ('compute_instance', ctypes.c_uint)]

    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f'NVML memory observation failed: {code}')

    def snapshot(self):
        memory = self.Memory()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)))
        count = ctypes.c_uint(256)
        processes = (self.Process * count.value)()
        self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
            self.handle, ctypes.byref(count), processes))
        occupied = {p.pid: p.used for p in processes[:count.value]}
        if any(v == 2**64-1 for v in occupied.values()):
            raise RuntimeError('Process GPU usage unavailable; cannot enforce admission')
        return dict(free=memory.free, total=memory.total, processes=occupied)

    def close(self):
        self.check(self.lib.nvmlShutdown())


def admission(snapshot, owned_pids, slot_bytes):
    pending = sum(max(0, slot_bytes-snapshot['processes'].get(pid, 0)) for pid in owned_pids)
    return dict(free_gpu_bytes=snapshot['free'], pending_gpu_bytes=pending,
                required_gpu_bytes=slot_bytes+pending, ready=snapshot['free'] >= slot_bytes+pending)


def resources(allowed: dict, campaign: Path, snapshot=None, owned_pids=()) -> dict:
    import shutil
    limits = allowed['limits']
    if snapshot is None:
        free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 2**20
        snapshot = dict(free=free, processes={})
    gpu = admission(snapshot, owned_pids, limits['free_gpu_bytes'])
    used = campaign_size(campaign)
    ram, disk = psutil.virtual_memory().available, shutil.disk_usage(campaign).free
    own_rss = 0
    for p in psutil.Process().children(recursive=True):
        try:
            own_rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return dict(**{k: v for k, v in gpu.items() if k != 'ready'},
        free_ram_bytes=ram, free_disk_bytes=disk, campaign_bytes=used,
        ready=(gpu['ready'] and ram >= max(0, limits['rss_bytes']-own_rss)
               and disk >= max(0, limits['disk_bytes'] - used) + limits['free_reserve_bytes']))


def stop(process) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def campaign_size(campaign):
    total = 0
    for path in campaign.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Atomic checkpoints and completed recovery retirement can race a scan.
    return total


def check_running(process, allowed, campaign):
    """Also cover preparation/statistics loops which do not call a model budget."""
    import shutil
    limits = allowed['limits']
    parent = psutil.Process()
    rss = 0
    for p in [parent, *parent.children(recursive=True)]:
        try:
            rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    used = campaign_size(campaign)
    if (rss > limits['rss_bytes'] or used > limits['disk_bytes']
            or shutil.disk_usage(campaign).free < limits['free_reserve_bytes']):
        raise RuntimeError('Actual worker/parent resource cap exceeded')


def dispatch(allowed: dict, rows: list[dict], command, validate, followups=(), preflight=None,
             started=None, poll_seconds=5.) -> dict:
    """Reserve the fixed queue; charge each child's full elapsed, including failure."""
    campaign = Path(allowed['campaign'])
    campaign.mkdir()  # A durable one-shot claim; never restart a failed batch.
    used, limits = dict(allowed['prior_used_seconds']), allowed['budget_seconds']
    tasks = [*rows, *followups]
    if (not rows or len({r['id'] for r in tasks}) != len(tasks)
            or any(v < 0 or v > limits[k] for k, v in used.items())
            or any(sum(r['seconds'] for r in tasks if r.get('bucket', 'training') == k)
                   + used.get(k, 0) > cap for k, cap in limits.items())):
        raise ValueError('Whole fixed queue does not fit the approved budget')
    workers = allowed.get('workers', 1)
    isolate = allowed.get('worker_failure_policy') == 'retire_failed_slot_continue_healthy'
    result = dict(status='WAITING_OR_TRAINING', used=used, limits=limits, completed=[],
                  active={}, retired_slots=[], sources=allowed.get('sources', []))
    active, monitor = {}, None
    control_tick = time.monotonic() if started is None else started
    def charge_control():
        nonlocal control_tick
        now = time.monotonic()
        used['control'] = used.get('control', 0) + now-control_tick
        control_tick = now
        write(campaign/'budget.json', result)
        if used['control'] >= limits['control']:
            raise TimeoutError('Controller allowance exhausted')
    def pause(seconds):
        nonlocal control_tick
        charge_control()
        begin = time.monotonic()
        time.sleep(seconds)
        control_tick += time.monotonic()-begin
    def finish(slot, error=None):
        item = active.pop(slot)
        process, row, job = item['process'], item['row'], item['job']
        try:
            stop(process)
            if error is None:
                if process.returncode != 0:
                    raise RuntimeError(f'Worker failed with exit {process.returncode}')
                validate(row, job)
        except Exception as exc:
            error = repr(exc)
        finally:
            item['log'].close()
        elapsed = time.monotonic()-item['started']
        bucket = row.get('bucket', 'training')
        used[bucket] = used.get(bucket, 0)+elapsed
        if elapsed > row['seconds']:
            error = error or 'Trajectory actual elapsed exceeds its allowance'
        receipt = dict(id=row['id'], slot=slot, bucket=bucket, actual_seconds=elapsed,
                       success=error is None, error=error)
        result['completed'].append(receipt)
        result['active'].pop(str(slot), None)
        write(job/'dispatch_receipt.json', receipt)
        if error and slot not in result['retired_slots']:
            result['retired_slots'].append(slot)
        charge_control()
        if used[bucket] > limits[bucket]:
            raise TimeoutError('Actual cumulative allowance exceeded; fees are not clipped')
        return error
    try:
        if allowed.get('predecessor'):
            from step28_formal_wait import predecessor_state
            while True:
                previous = predecessor_state(allowed['predecessor'])
                write(campaign/'waiting.json', dict(epoch=time.time(), status='WAITING_PREDECESSOR',
                                                    predecessor=previous))
                if previous['ready']:
                    break
                pause(60)
        if preflight is not None:
            preparation_started = time.monotonic()
            success = False
            try:
                preflight()
                success = True
            finally:
                elapsed = time.monotonic()-preparation_started
                control_tick += elapsed
                used['preparation'] = used.get('preparation', 0)+elapsed
                write(campaign/'preflight.json', dict(actual_seconds=elapsed, success=success))
                if used['preparation'] > limits['preparation']:
                    raise TimeoutError('Preflight exceeds preparation allowance')
        monitor = GpuMemory()
        cores = psutil.Process().cpu_affinity()
        if len(cores) < workers:
            raise RuntimeError('Not enough distinct permitted CPU cores')
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'CUDA_VISIBLE_DEVICES': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        # Full training succeeds before any followup is even admitted.
        for phase, phase_rows in [('training', rows), *[(r['id'], [r]) for r in followups]]:
            pending = list(phase_rows)
            width = workers if phase == 'training' else 1
            while pending or active:
                snapshot = monitor.snapshot()
                check_running(None, allowed, campaign)  # Shared RAM/disk fault stops this batch.
                if allowed.get('listener'):
                    from step28_formal_wait import observe_first_update
                    observe_first_update(allowed, campaign)
                for slot, item in list(active.items()):
                    process, row = item['process'], item['row']
                    error = None
                    cap = allowed['limits'].get('process_gpu_bytes')
                    if cap and snapshot['processes'].get(process.pid, 0) > cap:
                        error = 'PROCESS_GPU_LIMIT_EXCEEDED'
                        write(item['job']/'gpu_limit.json', dict(epoch=time.time(), pid=process.pid,
                            observed_bytes=snapshot['processes'][process.pid], limit_bytes=cap))
                    elif time.monotonic() >= item['started']+row['seconds']-60:
                        error = 'Trajectory shutdown reserve reached'
                    if error or process.poll() is not None:
                        error = finish(slot, error)
                        if error and (not isolate or phase != 'training'):
                            raise RuntimeError('Worker failed; no automatic retry or next dispatch: '+error)
                for slot in range(width):
                    if not pending or slot in active or slot in result['retired_slots']:
                        continue
                    row = pending[0]
                    if row.get('gpu', True):
                        # Refresh after exits and before each admission; pending reservations
                        # include workers that have not initialized CUDA yet.
                        snapshot = monitor.snapshot()
                        observation = resources(allowed, campaign, snapshot,
                                                [v['process'].pid for v in active.values()])
                        write(campaign/'admission.json', dict(epoch=time.time(), id=row['id'], **observation))
                        if not observation['ready']:
                            break
                    charge_control()  # Never launch after exhausting control balance.
                    job = campaign/row['id']
                    job.mkdir()
                    begin = time.monotonic()
                    log = (job/'console.txt').open('x', encoding='utf-8')
                    try:
                        argv = ['taskset', '-c', str(cores[slot]),
                                *command(row, job, begin+row['seconds'])]
                        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=allowed['workspace'], start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active[slot] = dict(process=process, row=row, job=job, started=begin, log=log)
                    result['active'][str(slot)] = dict(id=row['id'], bucket=row.get('bucket', 'training'),
                        pid=process.pid, started_epoch=time.time(), seconds=row['seconds'])
                    child = psutil.Process(process.pid)
                    write(job/'process.json', dict(pid=process.pid, command=argv,
                        created=child.create_time(), owner=child.username(), cwd=allowed['workspace'], slot=slot))
                    pending.pop(0)
                    charge_control()
                if pending and not active and all(s in result['retired_slots'] for s in range(width)):
                    result['not_dispatched'] = [r['id'] for r in pending]
                    raise RuntimeError('All slots retired; no automatic retry')
                if pending or active:
                    pause(poll_seconds)
            if phase == 'training':
                if len(result['completed']) != len(rows) or not all(r['success'] for r in result['completed']):
                    raise RuntimeError('Incomplete training batch; evaluation remains closed')
                charge_control()
                result['status'] = 'ALL_TRAINING_COMPLETE_EVALUATION_CLOSED'
                write(campaign/'training_complete.json', result)
        monitor.close()
        monitor = None
        charge_control()
        if followups:
            result['status'] = 'ALL_FORMAL_ACTIONS_COMPLETE'
            write(campaign/'completion.json', result)
        return result
    except BaseException as error:
        # Drain only our own children; preserve the original failure even if cleanup fails.
        cleanup = []
        for slot in list(active):
            try:
                finish(slot, 'Shared controller stop: '+repr(error))
            except BaseException as exc:
                cleanup.append(repr(exc))
        result.update(status='STOP_NO_AUTOMATIC_RETRY', error=repr(error), cleanup_errors=cleanup)
        try:
            charge_control()
        except BaseException as exc:
            cleanup.append(repr(exc))
        write(campaign/'failure.json', result)
        raise
    finally:
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                write(campaign/'monitor_cleanup.json', dict(error=repr(exc)))
