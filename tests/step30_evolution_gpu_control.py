"""One bounded GPU qualification process; preserves existing job commitments."""
from __future__ import annotations

import argparse
import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pynvml as nv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--permit', type=Path, required=True)
    args = parser.parse_args()
    import json
    allowed = json.loads(args.permit.read_text())
    sys.path.insert(0, allowed['source_root'])
    from step30_evolution_io import PYTHON, directory_bytes, read, record, verify, write
    from step30_evolution_control import release
    import shutil

    root = Path(allowed['root'])
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or allowed['operation'] != 'Evolution_single_worker_gpu_qualification'
            or allowed['approved'] is not True or allowed['workers'] != 1
            or allowed['automatic_retry'] is not False or allowed['gpu_cap_seconds'] != 900):
        raise ValueError('Approved bounded one-worker qualification only')
    for row in [allowed['authorization'], allowed['entry'], allowed['control_entry'],
                allowed['cpu_receipt'], allowed['starting_budget'], *allowed['sources']]:
        verify(row)
    if verify(allowed['control_entry']).resolve() != Path(__file__).resolve():
        raise ValueError('Actual qualification monitor source differs')
    if read(verify(allowed['cpu_receipt']))['status'] != 'PASS':
        raise ValueError('Handwritten fixture validation required')
    budget = read(verify(allowed['starting_budget']))
    if budget['used_seconds']['gpu'] + 900 > 14400:
        raise ValueError('Cannot reserve the complete GPU attempt')
    if (root/'gpu01').exists() or (root/'gpu_launch.json').exists():
        raise ValueError('One GPU attempt; no automatic retry')
    nv.nvmlInit()
    device = nv.nvmlDeviceGetHandleByIndex(0)
    memory = nv.nvmlDeviceGetMemoryInfo(device)
    observed = {p.pid: int(p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(device)}
    h_root = Path('/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_pause')
    h_request = read(h_root/'request.json')
    h_campaign = Path(h_request['campaign'])
    h_dispatch = read(h_campaign/'training/dispatch.json')
    h_gate = read(Path(h_request['original_gate']))
    if not (h_campaign/'pause_dispatch').exists() or h_dispatch['status'] not in ('PAUSED_DRAINING', 'PAUSED_IDLE'):
        raise ValueError('H must retain its controlled no-dispatch state')
    commitments, unreflected = [], 0
    for row in h_dispatch['active'].values():
        try:
            p = psutil.Process(row['pid'])
            if (p.username() != 'yongpeng' or p.cwd() != h_request['workspace']
                    or str(h_campaign/'training'/row['id']) not in p.cmdline()):
                raise ValueError('H active worker identity differs')
            reserved = h_gate['limits']['process_gpu_bytes']
            remaining = max(0, reserved-observed.get(p.pid, 0))
            unreflected += remaining
            commitments.append(dict(pid=p.pid, owner=p.username(), created=p.create_time(),
                command=p.cmdline(), reserved_bytes=reserved, unreflected_bytes=remaining))
        except psutil.NoSuchProcess:
            commitments.append(dict(pid=row['pid'], exited=True))
    if set(observed) - {r['pid'] for r in commitments if not r.get('exited')}:
        raise ValueError('Unknown GPU process requires its own commitment check')
    admission = dict(observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        gpu_total_bytes=memory.total, gpu_free_bytes=memory.free, unreflected_bytes=unreflected,
        other_commitments=commitments, ram_available=psutil.virtual_memory().available,
        disk_free=shutil.disk_usage(root).free, experiment_bytes=directory_bytes(Path(allowed['experiment_root'])),
        h_dispatch_status=h_dispatch['status'], permit=record(args.permit))
    ready = (memory.free-unreflected >= 17*2**30 and admission['ram_available'] >= 40*2**30
             and admission['disk_free'] >= 128*2**30 and admission['experiment_bytes']+3*2**30 <= 64*2**30)
    write(root/'gpu_admission.json', dict(**admission, ready=ready))
    if not ready:
        nv.nvmlShutdown()
        raise RuntimeError('Resource gate closed; no automatic wait or retry')

    process, error, ok, parent_active = None, None, False, 0.
    peak_gpu, peak_rss = 0, 0
    started = time.monotonic()
    deadline = started+900
    command = ['taskset', '-c', '46,47', PYTHON, '-B', '-u', allowed['entry']['path'],
               '--permit', str(args.permit), '--output', str(root/'gpu01'), '--deadline', str(deadline)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        with (root/'gpu_console.log').open('x') as log:
            process = subprocess.Popen(command, cwd=allowed['source_root'], env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, umask=0o077)
            child = psutil.Process(process.pid)
            write(root/'gpu_launch.json', dict(pid=child.pid, owner=child.username(), created=child.create_time(),
                command=command, cwd=child.cwd(), permit=record(args.permit), monitor=record(Path(__file__))))
            last_save = 0.
            while process.poll() is None:
                active_start = time.monotonic()
                try:
                    peak_rss = max(peak_rss, child.memory_info().rss)
                    current = {p.pid: int(p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(device)}
                    peak_gpu = max(peak_gpu, current.get(child.pid, 0))
                    if (time.monotonic() >= deadline-15 or peak_rss > 32*2**30
                            or peak_gpu > allowed['process_gpu_bytes']):
                        raise RuntimeError('Qualification lifecycle, RSS or whole-process GPU limit')
                    if time.monotonic()-last_save >= 2:
                        write(root/'gpu_resource_progress.json', dict(pid=child.pid,
                            elapsed=time.monotonic()-started, sampled_peak_gpu_bytes=peak_gpu,
                            sampled_peak_rss_bytes=peak_rss))
                        last_save = time.monotonic()
                except psutil.NoSuchProcess:
                    pass
                finally:
                    parent_active += time.monotonic()-active_start
                time.sleep(.1)
            if process.returncode != 0 or read(root/'gpu01/receipt.json')['status'] != 'PASS':
                raise RuntimeError('GPU qualification child failed; no retry')
            ok = True
    except BaseException as exc:
        error = repr(exc)
    finally:
        try:
            if process is not None:
                release(process)
        except BaseException as exc:
            ok, error = False, repr(exc)
        elapsed = time.monotonic()-started
        if elapsed > 900:
            ok, error = False, error or 'Complete lifecycle exceeded 900 seconds'
        gpu_charge, parent_charge = math.ceil(elapsed), math.ceil(parent_active)
        receipt = dict(success=ok, error=error, exit_code=None if process is None else process.returncode,
            lifecycle_seconds=elapsed, charged_gpu_seconds=gpu_charge, parent_active_seconds=parent_active,
            charged_auxiliary_seconds=parent_charge, sampled_peak_gpu_bytes=peak_gpu,
            sampled_peak_rss_bytes=peak_rss, released=True, automatic_retry=False, permit=record(args.permit))
        write(root/'gpu_lifecycle.json', receipt)
        budget['used_seconds']['gpu'] += gpu_charge
        budget['used_seconds']['auxiliary'] += parent_charge
        budget['events'].append(dict(operation='single_gpu01', success=ok, receipt=record(root/'gpu_lifecycle.json'),
            gpu_seconds=gpu_charge, auxiliary_seconds=parent_charge))
        write(root/'budget.json', budget)
        nv.nvmlShutdown()
    print(json.dumps(receipt), flush=True)
    if not ok:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
