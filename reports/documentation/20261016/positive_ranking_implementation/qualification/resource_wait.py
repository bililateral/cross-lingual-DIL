"""One authorized native attempt, then one conditional formal run; no retries."""
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import psutil

ROOT = Path.cwd().resolve()
Q = ROOT / 'reports/documentation/20261016/positive_ranking_implementation/qualification'
LAUNCH = ROOT / 'reports/launch'
PYTHON = '/home/yongpeng/miniconda3/envs/py310/bin/python'
NATIVE = ROOT / 'reports/qualification/native'
STATUS = LAUNCH / 'status.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def record(path):
    payload = Path(path).read_bytes()
    return dict(path=Path(path).relative_to(ROOT).as_posix(), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

def status(phase, **values):
    write(STATUS, dict(phase=phase, time=datetime.datetime.now().astimezone().isoformat(), pid=os.getpid(), **values))

def wait_resources(phase):
    while True:
        process = psutil.Process()
        ancestors = {process.pid, *(p.pid for p in process.parents())}
        jobs = [p.info for p in psutil.process_iter(['pid', 'username', 'cmdline'])
                if p.pid not in ancestors and any('step28' in x or 'resource_wait.py' in x for x in (p.info['cmdline'] or []))]
        gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=index,uuid,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
        apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True).strip()
        values = dict(gpu=gpu, compute_apps=apps, project_jobs=jobs,
                      host_available_bytes=psutil.virtual_memory().available, disk_free_bytes=shutil.disk_usage(ROOT).free)
        status(phase, **values)
        if (not apps and not jobs and int(gpu.split(',')[-2]) >= 30*1024
                and values['host_available_bytes'] >= 72*2**30 and values['disk_free_bytes'] >= 16*2**30):
            return values
        time.sleep(60)

def native_once(gate):
    assert not NATIVE.exists() and not (LAUNCH/'native_wallclock.json').exists()
    resources = wait_resources('WAIT_NATIVE_RESOURCES')
    for row in gate['source_files']:
        assert record(ROOT/row['path']) == row
    started = time.time()
    env = dict(os.environ, QUALIFICATION_STARTED_EPOCH=str(started))
    command = [PYTHON, 'scripts/step28_positive_ranking_verify.py', 'gpu', '--output', str(NATIVE), '--gate', str(Q/'native_gate.json')]
    with (LAUNCH/'native.console.txt').open('w') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        receipt = dict(started_epoch=started, deadline_epoch=started+1200, pid=child.pid, command=command, resources=resources)
        write(LAUNCH/'native_start.json', receipt)
        status('NATIVE_RUNNING', child_pid=child.pid, started_epoch=started, deadline_epoch=started+1200)
        try:
            child.wait(timeout=max(.01, started+1195-time.time()))
        except subprocess.TimeoutExpired:
            receipt['timeout'] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    elapsed = time.time()-started
    receipt.update(exit_code=child.returncode, elapsed_seconds=elapsed, charged_seconds=math.ceil(elapsed))
    write(LAUNCH/'native_wallclock.json', receipt)
    assert child.returncode == 0 and receipt['charged_seconds'] <= 1200
    result = read(NATIVE/'result.json')
    assert result['status'] == 'PASS_HANDWRITTEN_ONLY' and result['mode'] == 'gpu'
    assert result['source_files'] == gate['source_files'] and result['actual_updates'] == 12
    assert result['cumulative_charged_seconds'] <= 1200 and not result['formal_inputs'] and not result['formal_labels']
    assert result['peak_rss_bytes'] <= 64*2**30 and result['peak_work_bytes'] <= 10*2**30 and result['max_reserved_bytes'] <= 28*2**30
    assert set(result['arms']) == {'S_pos', 'H_pos'}
    for arm, value in result['arms'].items():
        assert value['actual_updates'] == len(value['updates']) == 6 and value['charged_seconds'] <= 600
        assert value['maximum_records'] == 224 and value['channel_tokens'] == 256
        assert value['metadata'] == gate['candidates'][arm] and value['initial_state'] == result['arms']['S_pos']['initial_state']
        assert all(value[k] is True for k in ('both_phases_encoder_head_effect', 'pooling_restores_verified', 'next_update_bitwise_equal', 'rank_precision_verified'))
        assert value['equality_counts']['tensors'] > 0 and value['equality_counts']['elements'] > 0
        assert len(value['table_precision']) == 5
        assert [t['role'] for t in value['table_precision']] == ['current', 'current', 'history', 'current', 'history']
        for phase in ('current', 'paired'):
            for component in ('encoder', 'head'):
                effect = value['effects'][phase][component]
                assert effect['finite'] and effect['active_grad_parameters'] > 0 and effect['adam']['finite']
                assert all(effect[k] > 0 for k in ('pre_clip_gradient_max_delta', 'post_clip_gradient_max_delta', 'parameter_max_delta'))
                if phase == 'paired':
                    assert all(effect['isolated_branch_preclip_deltas'][role] > 0 for role in ('current', 'history'))
        assert read(NATIVE/arm/'temporary_checkpoint_identity.json')['removed'] is True
        assert not (NATIVE/arm/'work').exists()
    continuation = result['arms']['H_pos']['resources']
    assert continuation['allowed_context_pid'] == child.pid
    assert continuation['live_allocated_bytes'] == continuation['reserved_bytes'] == 0
    assert result['arms']['S_pos']['resources']['compute_apps'] == ''
    measurements = [v['measurements'] for v in result['arms'].values()]
    seconds = sum(v['projection']['projected_total_seconds'] for v in result['arms'].values())
    peak = math.ceil(1.1*(max(m['W_F'] for m in measurements)+sum(2*m['W_I'] for m in measurements)+2*2**30))
    projection = dict(projected_total_seconds=seconds, projected_peak_output_bytes=peak, within_budget=seconds <= 57600 and peak <= 16*2**30)
    write(LAUNCH/'native_acceptance.json', dict(status='SAVED_NATIVE_BEHAVIOR_AND_BUDGET_CHECKED', gpu=record(NATIVE/'result.json'), outer_wallclock=record(LAUNCH/'native_wallclock.json'), projection=projection,
          scope='Checks actual native outputs and preserved behavioral evidence. Formal production entry independently validates source/gates and recomputes its existing projection before any formal input. No new scientific experiment.'))
    assert projection['within_budget']
    return result

def formal_once(gate):
    links = read(LAUNCH/'formal_links.json')
    for row in links:
        target, source = ROOT/row['path'], Path(row['source'])
        assert source.is_file()
        if row.get('sha256'):
            payload = source.read_bytes()
            assert len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists() and not target.is_symlink()
        target.symlink_to(source)
    write(LAUNCH/'formal_links_created.json', dict(files=links, note='Only seven authorized formal files and52 opened baseline files; no old Memory/heldout/owners links.'))
    policy = read(ROOT/'schema/step28_positive_ranking_policy.json')
    formal = dict(gate, kind='FORMAL', job='reports/job', runtime=policy['runtime'], supervision=policy['supervision'], gpu=record(NATIVE/'result.json'))
    write(Q/'formal_gate.json', formal)
    resources = wait_resources('WAIT_FORMAL_RESOURCES')
    assert not (ROOT/'reports/job').exists() and not (LAUNCH/'formal_start.json').exists()
    command = ['bash', 'scripts/run_step28_positive_ranking_linux_20261018.sh', '0', str(Q/'formal_gate.json'), 'reports/job']
    with (LAUNCH/'formal.dispatch.console.txt').open('w') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(LAUNCH/'formal_start.json', dict(time=datetime.datetime.now().astimezone().isoformat(), pid=child.pid, command=command, resources=resources, gate=record(Q/'formal_gate.json')))
    status('FORMAL_STARTED_WAIT_FIRST_UPDATE', child_pid=child.pid)
    while child.poll() is None:
        log = ROOT/'reports/job.console.txt'
        if log.exists():
            for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('event') == 'updates' and event.get('point') == 'A_S_pos' and event.get('completed', 0) >= 1:
                    identity = record(Path(__file__).resolve())
                    Path(__file__).unlink()
                    status('FORMAL_RUNNING_FIRST_UPDATE_CONFIRMED', child_pid=child.pid, first_update=event, listener_source_removed=identity)
                    return
        time.sleep(10)
    raise RuntimeError('Formal wrapper exited before first observed update; no retry, inspect original logs')

if __name__ == '__main__':
    assert os.name == 'posix' and len(psutil.Process().cpu_affinity()) == 1
    LAUNCH.mkdir(parents=True, exist_ok=True)
    lock = (LAUNCH/'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATUS.exists(), 'Existing listener lifecycle cannot be restarted'
    try:
        gate = read(Q/'native_gate.json')
        native_once(gate)
        formal_once(gate)
    except BaseException:
        status('STOP_NO_RETRY', traceback=traceback.format_exc())
        raise
