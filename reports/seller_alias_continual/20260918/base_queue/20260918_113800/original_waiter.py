"""One-use resource waiter for the already approved sensitivity run."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/yongpeng/cross-lingual')
INTERVAL = 60


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resources() -> dict:
    gpu = subprocess.run(['nvidia-smi', '-i', '0',
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15)
    jobs = subprocess.run(['nvidia-smi', '-i', '0',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15)
    if gpu.returncode or jobs.returncode:
        raise RuntimeError('GPU resource query failed')
    free = int(gpu.stdout.strip())
    pids = [int(x.strip()) for x in jobs.stdout.splitlines() if x.strip()]
    memory = next(int(x.split()[1]) * 1024 for x in
        Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    disk = shutil.disk_usage(ROOT).free
    return {'gpu_free_mib': free, 'compute_pids': pids,
        'host_free_bytes': memory, 'disk_free_bytes': disk,
        'ready': not pids and free >= 24576 and memory >= 16 * 1024**3 and disk >= 24 * 1024**3}


def launch(base: Path, snapshot: list) -> None:
    for row in snapshot:
        path = ROOT / row['path']
        if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Approved source changed: ' + row['path'])
    if (base / 'job').exists():
        raise RuntimeError('Run directory already exists; no restart')
    # One launch attempt for this run, including failed startup; never retry it.
    with (base / 'launch.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'LAUNCH_ATTEMPT', 'time': datetime.datetime.now().astimezone().isoformat()}, f)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    with (base / 'launcher.log').open('ab') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'scripts/run_step28_sensitivity_linux_20260916.sh'),
            str(base / 'job')], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'status': 'LAUNCHER_STARTED', 'pid': process.pid,
        'time': datetime.datetime.now().astimezone().isoformat(), 'gpu': 0}
    write(base / 'launch.json', receipt)
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError('Launcher exited before entry startup: ' + str(process.returncode))
        if (base / 'job/run/startup.json').is_file():
            script = Path(__file__).resolve()
            if script != ROOT / 'scripts/step28_wait_gpu.py':
                raise RuntimeError('Unexpected self-delete path')
            script.unlink()
            receipt.update(status='ENTRY_STARTED_WAITER_REMOVED', waiter_removed=not script.exists(),
                checks_and_training_completion='See job/train.log and job/run; not inferred from startup')
            write(base / 'launch.json', receipt)
            return
        time.sleep(1)
    raise RuntimeError('Startup not observed within 120 seconds; do not relaunch; inspect recorded PID')


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    if not base.is_relative_to(ROOT / 'reports') or not base.is_dir():
        raise ValueError('Use the prepared project run directory')
    snapshot = json.loads((base / 'approved_sources.json').read_text())
    consecutive = 0
    while True:
        try:
            state = resources()
        except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            state = {'ready': False, 'error': str(error)}
        consecutive = consecutive + 1 if state['ready'] else 0
        state.update(time=datetime.datetime.now().astimezone().isoformat(), consecutive_ready=consecutive)
        write(base / 'wait_status.json', state)
        if consecutive >= 2:
            launch(base, snapshot)
            return
        time.sleep(INTERVAL)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        base = Path(sys.argv[1]).resolve()
        if base.is_relative_to(ROOT / 'reports') and base.is_dir():
            write(base / 'wait_failure.json', {'error': str(error), 'automatic_restart': False})
        raise
