"""Wait for the pinned comparison campaign, then train the 18 original S/H runs."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

import step28_continual_defaults_run as run
import step28_comparison_control as shared

data, m = run.data, run.method
GIB = 2**30
SLOT_BYTES = int(10.25*GIB)
ALLOCATOR_BYTES = int(9.25*GIB)
BUDGET = dict(training=108*3600, control=2*3600)
PER_RUN_SECONDS = 6*3600
PREDECESSOR_COMPLETE = 'ALL_COMPARISON_TRAINING_CONFIRMATION_ANALYSIS_COMPLETE'
COMPLETE = 'ALL_18_ORIGINAL_RUNS_AND_54_ENDPOINTS_BLIND_COMPLETE'


def sources():
    rows = run.sources(('scripts/step28_defaults_control.py', 'tests/test_step28_defaults_control.py'))
    rows.append(data.record(data.ROOT/'run_step28_defaults_linux_20261026.sh', data.ROOT))
    return sorted(rows, key=lambda row: row['path'])


def validate_gate(path: Path, active: Path) -> dict:
    gate = data.read_json(path)
    if (gate.get('status') != 'APPROVED_IMPLEMENTATION_AND_RESOURCES'
            or gate.get('scope') != 'original_s_h_three_stage'
            or gate.get('train_data') is not True or gate.get('approved_actions') != ['train_blind']
            or gate.get('evaluation_labels') is not False or gate.get('new_confirmation_data') is not False
            or gate.get('runs') != [s.identity() for s in m.specifications()]
            or gate.get('sources') != run.sources() or gate.get('orchestration_sources') != sources()
            or gate.get('budget_seconds') != BUDGET or gate.get('per_run_seconds') != PER_RUN_SECONDS
            or gate.get('maximum_workers') != 3 or gate.get('allocator_bytes') != ALLOCATOR_BYTES
            or gate.get('limits') != dict(disk_bytes=128*GIB, small_bytes=2*GIB)
            or gate.get('listener') != str(active.resolve())
            or active.name != 'active_defaults_wait.py'):
        raise ValueError('Exact original-S/H blind-training scope or sources differ')
    if active.read_bytes() != (data.ROOT/'scripts/step28_defaults_control.py').read_bytes():
        raise ValueError('Expendable listener differs from retained source')
    ready = data.read_json(data.verify(Path(gate['review_ready']['path']), gate['review_ready']))
    if (ready.get('status') != 'REVIEW_CLOSED_AND_CPU_PASSED'
            or ready.get('orchestration_sources') != gate['orchestration_sources']):
        raise ValueError('The incremental orchestration disposition is not closed')
    cpu = data.read_json(data.verify(Path(ready['cpu_receipt']['path']), ready['cpu_receipt']))
    if cpu.get('exit_code') != 0 or any(r not in cpu['sources'] for r in sources()):
        raise ValueError('CPU evidence does not cover this orchestration')
    native = data.read_json(data.verify(Path(gate['native_result']['path']), gate['native_result']))
    if (native.get('status') != 'PASS_NATIVE_HANDWRITTEN' or native.get('actual_calls') != 3
            or not native.get('actual_next_update_equal') or not native.get('inference_equal')
            or any(r not in native['source_files'] for r in run.sources())):
        raise ValueError('Applicable original-method native evidence missing')
    if gate['prior_used_seconds'].get('training', 0) != 0:
        raise ValueError('This fresh campaign cannot conceal previous training attempts')
    validate_prior(gate, cpu)
    return gate


def validate_prior(gate: dict, cpu: dict) -> None:
    prior = gate['prior_used_seconds']['control']
    parts = gate['prior_breakdown_seconds']
    if (parts.get('cpu_all_attempts') != cpu['cumulative_seconds']
            or parts['cpu_all_attempts'] < 56 or parts.get('native_all_attempts') != 171
            or parts.get('preparation_conservative_charge', 0) < 120
            or not all(math.isfinite(v) and v >= 0 for v in parts.values())
            or prior != sum(parts.values()) or not 347 <= prior <= BUDGET['control']-60):
        raise ValueError('Inherited CPU/native/preparation charges or finalization reserve missing')


def predecessor_processes(predecessor: dict, launch: dict) -> list[dict]:
    """Identify this campaign, not every CUDA process on the shared host."""
    workspace, campaign = predecessor['workspace'], predecessor['campaign']
    gate_path = predecessor['gate']['path']
    entries = {'step28_comparison_control.py', 'step28_continual_comparison_run.py',
               'step28_continual_comparison_evaluate.py'}
    alive = []
    for process in psutil.process_iter(['pid', 'username', 'create_time', 'cmdline', 'status']):
        try:
            info = process.info
            if info['username'] != launch['owner'] or info['status'] == psutil.STATUS_ZOMBIE:
                continue
            command = info['cmdline'] or []
            original = (info['pid'] == launch['pid']
                        and abs(info['create_time']-launch['created']) < .01)
            relevant = (gate_path in command or any(x.startswith(campaign+'/') for x in command)
                        or any(Path(x).name in entries for x in command))
            if not original and not relevant:
                continue
            cwd = process.cwd()
            if original and cwd != workspace:
                raise RuntimeError('Pinned controller identity changed before its exit')
            if original or (relevant and cwd == workspace):
                alive.append(dict(pid=info['pid'], owner=info['username'], created=info['create_time'],
                                  cwd=cwd, command=command))
        except psutil.NoSuchProcess:
            continue
    return alive


def predecessor_state(predecessor: dict) -> dict:
    old_gate = data.read_json(data.verify(Path(predecessor['gate']['path']), predecessor['gate']))
    launch = data.read_json(data.verify(Path(predecessor['launch']['path']), predecessor['launch']))
    campaign = Path(predecessor['campaign'])
    if (old_gate['campaign'] != str(campaign) or old_gate['workspace'] != predecessor['workspace']
            or launch['cwd'] != predecessor['workspace'] or launch['formal_gate'] != predecessor['gate']):
        raise ValueError('Predecessor gate/launch/campaign identity differs')
    if (campaign/'failure.json').exists():
        raise RuntimeError('Predecessor failed; do not treat GPU release as success')
    alive = predecessor_processes(predecessor, launch)
    completion = campaign/'completion.json'
    if not completion.exists():
        if not alive:
            raise RuntimeError('Predecessor exited without complete final receipt; no automatic launch')
        return dict(ready=False, reason='PREDECESSOR_STILL_RUNNING', processes=alive)
    result = data.read_json(completion)
    expected = {f"{r['order']}_s{r['seed']}_{r['name']}" for r in old_gate['new_runs']}
    expected.update(('generation', 'confirmation', 'analysis'))
    budget = result['budget']
    completed = budget['completed']
    if (result.get('status') != PREDECESSOR_COMPLETE or result.get('sources') != old_gate['sources']
            or budget['failed'] or budget['active'] or len(completed) != len(expected)
            or {r['id'] for r in completed} != expected or not all(r['success'] for r in completed)
            or any(not math.isfinite(v) or v < 0 or v > budget['limits'][k]
                   for k, v in budget['used'].items())):
        raise ValueError('Predecessor final receipt does not prove all training and post-work complete')
    return dict(ready=not alive, reason='PREDECESSOR_EXIT_PENDING' if alive else 'PREDECESSOR_COMPLETE',
                processes=alive, completion=data.record(completion, campaign))


def gpu_admission(free: int, applications: list[tuple[int, int]], owned: set[int], active: int) -> dict:
    # One GPU process per unchanged training worker; no worker borrows another slot's reserve.
    own = {pid: size for pid, size in applications if pid in owned}
    allocated = sum(min(size, SLOT_BYTES) for size in own.values())
    unmaterialized = max(0, active*SLOT_BYTES-allocated)
    required = SLOT_BYTES+unmaterialized
    return dict(free_gpu_bytes=free, own_allocated_bytes=allocated,
                unmaterialized_own_reserve_bytes=unmaterialized, required_free_gpu_bytes=required,
                ready=len(own) <= active and free >= required)


def gpu_snapshot(deadline: float) -> tuple[int, list[tuple[int, int]]]:
    def query(field: str) -> str:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Bounded GPU admission snapshot expired')
        return subprocess.check_output(['nvidia-smi', '--id=0', field,
            '--format=csv,noheader,nounits'], text=True, timeout=min(10., remaining)).strip()
    free = int(query('--query-gpu=memory.free'))*2**20
    text = query('--query-compute-apps=pid,used_memory')
    applications = []
    for line in text.splitlines():
        pid, size = (v.strip() for v in line.split(','))
        # Unknown own use is conservatively left reserved. Foreign CUDA is not a veto.
        applications.append((int(pid), int(size)*2**20 if size.isdecimal() else 0))
    return free, applications


def resources(campaign: Path, active: int, *, startup: bool = False) -> dict:
    import shutil
    own = set(shared.prior.run.owned_processes(os.getpid()))
    # Two bounded observations must agree; growth or release defers this refill.
    # All four queries share the previous 20-second admission allowance, with no retry loop.
    deadline = time.monotonic()+20.
    first, second = gpu_snapshot(deadline), gpu_snapshot(deadline)
    own_use = lambda sample: {pid: size for pid, size in sample[1] if pid in own}
    stable = first[0] == second[0] and own_use(first) == own_use(second)
    row = gpu_admission(min(first[0], second[0]), second[1], own, active)
    row.update(epoch=time.time(), cuda_processes=second[1], samples=[first, second],
               stable_gpu_observation=stable, active_workers=active, startup=startup,
               host_available_bytes=psutil.virtual_memory().available,
               disk_free_bytes=shutil.disk_usage(campaign).free)
    row['ready'] &= (stable and row['host_available_bytes'] >= (208 if startup else 64)*GIB
                     and row['disk_free_bytes'] >= (192 if startup else 76)*GIB)
    shared.write(campaign/'resource_status.json', row)
    return row


def first_update(campaign: Path, active: Path, run_ids: set[str]) -> dict | None:
    if not active.exists():
        return None
    for log in sorted((campaign/'dispatch').glob('*/console.txt')):
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = __import__('json').loads(line)
                except ValueError:
                    continue
                if (event.get('event') == 'updates' and event.get('run_id') in run_ids
                        and event.get('stage') in (1, 2, 3) and event.get('completed', 0) > 0):
                    active.unlink()
                    record = dict(epoch=time.time(), event=event, log=str(log), listener_deleted=True)
                    shared.write(campaign/'first_update.json', record)
                    return record
    return None


def training_command(row: dict, job: Path, deadline: float, metadata: Path, gate_path: Path) -> list[str]:
    shared.write(metadata/'spec.json', row['spec'])
    return [sys.executable, '-B', str(Path(run.__file__)), '--job', str(job),
            '--spec', str(metadata/'spec.json'), '--gate', str(gate_path), '--deadline', str(deadline)]


def complete_blind(campaign: Path, gate: dict) -> list[dict]:
    records = []
    for spec in m.specifications():
        path = campaign/spec.run_id/'run/completion.json'
        row = data.read_json(path)
        if (row.get('status') != run.inherited.COMPLETE or row.get('spec') != spec.identity()
                or row.get('source_files') != gate['sources'] or row.get('updates') != 864
                or row.get('gradient_group_presentations') != 1440
                or row.get('access') != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
                or set(row.get('points', {})) != {'1', '2', '3'}):
            raise ValueError('Original-method blind completion is incomplete')
        for stage, point in row['points'].items():
            if (point.get('adam_step') != int(stage)*288 or point.get('continual_spec') != spec.identity()
                    or not point.get('full_restore_verified') or not point.get('inference_restore_verified')):
                raise ValueError('Missing native endpoint/restore completion')
            if not {'raw', 'stage-cal', 'first-cal'} <= point.get('scores', {}).keys():
                raise ValueError('Blind endpoint scores missing')
            for payload in (point['model'], point['mapping'], *point['scores'].values()):
                candidate = path.parent/payload['path']
                if candidate.stat().st_size != payload['bytes']:
                    raise ValueError('Blind endpoint absent or incomplete')
        records.append(dict(id=spec.run_id, completion=data.record(path, campaign)))
    return records


def execute(gate_path: Path, active: Path) -> None:
    gate = validate_gate(gate_path, active)
    campaign = Path(gate['campaign'])
    campaign.mkdir()  # One durable claim; rejected old directories are never modified.
    ledger = shared.Ledger(campaign/'budget.json', BUDGET, gate['prior_used_seconds'])
    try:
        # Covers import/gate/claim and final receipt overhead conservatively.
        ledger.charge('control', 60.)
        while True:
            tick = time.monotonic()
            try:
                previous = predecessor_state(gate['predecessor'])
                available = resources(campaign, 0, startup=True) if previous['ready'] else None
                shared.write(campaign/'waiting.json', dict(epoch=time.time(), predecessor=previous,
                    resources=available, status='READY' if available and available['ready'] else 'WAITING'))
            finally:
                ledger.charge('control', time.monotonic()-tick)
            if available and available['ready']:
                break
            time.sleep(60)
        # Recheck exact sources and predecessor immediately before dispatch.
        tick = time.monotonic()
        try:
            validate_gate(gate_path, active)
            if not predecessor_state(gate['predecessor'])['ready']:
                raise RuntimeError('Predecessor changed before dispatch')
        finally:
            ledger.charge('control', time.monotonic()-tick)
        queue = [dict(id=s.run_id, spec=s.identity(), allowance=PER_RUN_SECONDS) for s in m.specifications()]
        run_ids = {r['id'] for r in queue}
        shared.write(campaign/'start.json', dict(epoch=time.time(), pid=os.getpid(), sources=gate['sources']))
        shared.dispatch(queue, campaign, ledger, 'training',
            lambda row, job, deadline, metadata: training_command(row, job, deadline, metadata, gate_path),
            admit=lambda count: resources(campaign, count, startup=not (ledger.active or ledger.completed))['ready'],
            check=lambda: shared.campaign_size(campaign, gate['limits']),
            observe=lambda: first_update(campaign, active, run_ids))
        tick = time.monotonic()
        try:
            records = complete_blind(campaign, gate)
            first_update(campaign, active, run_ids)
        finally:
            ledger.charge('control', time.monotonic()-tick)
        shared.write(campaign/'completion.json', dict(status=COMPLETE, runs=records, sources=gate['sources'],
            endpoints=54, evaluation_labels_opened=False, new_confirmation_generated=False,
            budget=ledger.snapshot()))
    except BaseException as error:
        shared.write(campaign/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error),
            traceback=traceback.format_exc(), budget=ledger.snapshot()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    if args.workspace.resolve() != data.ROOT:
        parser.error('Use the pinned independent workspace')
    os.chdir(data.ROOT)
    execute(args.gate.resolve(), Path(__file__).resolve())
