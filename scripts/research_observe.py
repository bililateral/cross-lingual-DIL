"""Read-only metadata snapshot; no project imports, model loads or effect reads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

PYTHON = Path('/home/yongpeng/miniconda3/envs/py310/bin/python')


def read_object(path):
    """A parsed JSON object, never a Path or a file binding."""
    for attempt in range(2):
        try:
            value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError(f'Expected JSON object: {path}')
            return value
        except json.JSONDecodeError:
            if attempt:
                raise ValueError(f'Concurrent or invalid JSON; repeat observation: {path}') from None


def identity(expected):
    wanted = dict(pid=expected['pid'], owner=expected.get('owner', expected.get('username')),
                  command=expected.get('command', expected.get('cmdline')), cwd=expected['cwd'])
    created = expected.get('created', expected.get('create_time'))
    if created is not None:
        wanted['created'] = created
    if 'start_ticks' in expected or 'boot_id' in expected:
        wanted.update(start_ticks=expected.get('start_ticks'), boot_id=expected.get('boot_id'))
    if None in wanted.values() or ('created' not in wanted and 'start_ticks' not in wanted):
        raise ValueError('Identity requires pid, owner, command, cwd and creation time or start_ticks/boot_id')
    try:
        p = psutil.Process(wanted['pid'])
        with p.oneshot():
            actual = dict(pid=p.pid, owner=p.username(), created=p.create_time(),
                          command=p.cmdline(), cwd=p.cwd())
            if 'start_ticks' in wanted:
                fields = Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ', 1)[1].split()
                actual.update(start_ticks=int(fields[19]),
                              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        matched = all(actual[k] == v for k, v in wanted.items())
        result = dict(exists=True, matches=matched, expected=wanted)
        if matched:
            result['actual'] = actual
        return result
    except (psutil.NoSuchProcess, FileNotFoundError):
        return dict(exists=False, matches=False, expected=wanted)
    except psutil.AccessDenied:
        return dict(exists=True, matches=None, expected=wanted, error='Access denied')


def argument(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def observe(root, gate_path=None, identity_receipt=None):
    root = Path(root).resolve(strict=True)
    if (gate_path is None) != (identity_receipt is None):
        raise ValueError('Supply both the effective gate and its identity receipt')
    gate_path = Path(gate_path or root / 'gate.json').resolve(strict=True)
    receipt_path = Path(identity_receipt or root / 'launch.json').resolve(strict=True)
    if not gate_path.is_relative_to(root):
        raise ValueError('Gate must be inside the explicitly selected root')
    launch = read_object(receipt_path)
    binding = launch['gate']
    if ('path' in binding or identity_receipt is not None) and binding.get('path') != str(gate_path):
        raise ValueError('Identity receipt names a different gate')
    raw_gate = gate_path.read_bytes()
    if len(raw_gate) != binding['bytes'] or hashlib.sha256(raw_gate).hexdigest() != binding['sha256']:
        raise ValueError('Gate size/SHA differs from identity receipt')
    gate = json.loads(raw_gate)
    campaign = Path(gate['campaign']).resolve()
    if not campaign.is_relative_to(root):
        raise ValueError('Campaign must be inside the explicitly selected root')
    controller = identity(launch.get('identity', launch))
    dispatch_path = campaign / 'dispatch.json'
    dispatch_present = dispatch_path.exists()
    dispatch = read_object(dispatch_path) if dispatch_present else {}
    budget = read_object(campaign / 'budget.json') if (campaign / 'budget.json').exists() else {}
    active, running, commitments = [], {}, {}
    for slot, source in dispatch.get('active', {}).items():
        proc = identity(source)
        row = dict(slot=slot, process=proc)
        if proc['matches']:
            command = proc['actual']['command']
            action = argument(command, '--action')
            bucket = {'train': 'training', 'score': 'gpu', 'application': 'gpu', 'analyze': 'auxiliary'}.get(action)
            elapsed = max(0, time.time() - proc['actual']['created'])
            deadline = argument(command, '--deadline')
            remaining = max(0, float(deadline) - time.monotonic()) if deadline else None
            row.update(action=action, bucket=bucket, elapsed_estimate_seconds=round(elapsed, 3),
                       deadline_remaining_seconds=round(remaining, 3) if remaining is not None else None)
            if bucket:
                running[bucket] = running.get(bucket, 0) + elapsed
                if remaining is not None:
                    commitments[bucket] = commitments.get(bucket, 0) + elapsed + remaining
        active.append(row)
    training = []
    for folder in sorted(campaign.glob('*/training/*')):
        if not folder.is_dir():
            continue
        workload = folder / 'run/workload.json'
        count = read_object(workload)['current_updates'] if workload.exists() else None
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f'Invalid workload update count: {workload}')
        receipt = folder / 'dispatch_receipt.json'
        saved = read_object(receipt) if receipt.exists() else {}
        training.append(dict(task=folder.name, phase=folder.parent.parent.name,
                             counters=dict(updates=count, total_updates=count) if count is not None else None,
                             counters_source='run/workload.json (epoch or completion)',
                             updates_mtime_utc=datetime.fromtimestamp(workload.stat().st_mtime, timezone.utc).isoformat()
                             if count is not None else None,
                             training_completion_record_exists=(folder / 'run/completion.json').exists(),
                             wrapper_validation_success=saved.get('success'),
                             worker_released=saved.get('released'),
                             failure_record_exists=(folder / 'worker_failure.json').exists()))
    settled = budget.get('used_seconds', {})
    return dict(observed_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                controller=controller, identity_receipt=str(receipt_path), gate_path=str(gate_path),
                launch_gate_matches=True, dispatch_record_exists=dispatch_present,
                dispatch={key: dispatch.get(key) for key in ('status', 'pending', 'retired', 'budget_stop')},
                dispatch_failure_count=len(dispatch['failures']) if 'failures' in dispatch else None,
                controller_error_present=bool(dispatch['controller_error']) if 'controller_error' in dispatch else None,
                active=active if 'active' in dispatch else None, training=training,
                campaign_failure_record_exists=(campaign / 'failure.json').exists(),
                campaign_completion_record_exists=(campaign / 'completion.json').exists(),
                scientific_decisions={phase: dict(analysis_record_exists=(campaign / phase / 'analysis/result/results.json').exists(),
                                                 decision_record_exists=(campaign / 'decisions' / f'{phase}.json').exists(),
                                                 status='NOT_READ') for phase in ('D', 'P', 'M', 'C')},
                budget=dict(settled_seconds=settled, caps_seconds=budget.get('caps'),
                            active_lifecycle_estimate_seconds={k: round(v, 3) for k, v in running.items()},
                            active_full_commitment_estimate_seconds={k: round(v, 3) for k, v in commitments.items()},
                            note='Active lifecycle and full commitment overlap; never add both. Estimates use matching process creation/deadline; reservations are not expenditure. Monitoring not estimated; no ledger writes.'),
                consistency='Sequential live snapshot, not an atomic campaign transaction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Explicit execution root containing launch.json and gate.json')
    parser.add_argument('--gate', type=Path, help='Effective gate selected by the handoff')
    parser.add_argument('--identity-receipt', type=Path, help='Named launch or handover receipt binding that gate')
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != PYTHON.resolve():
        parser.error('Run only with the existing Linux py310 Python')
    result = observe(args.root, args.gate, args.identity_receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['controller']['matches'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
