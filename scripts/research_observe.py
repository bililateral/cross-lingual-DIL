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
    aliases = {'owner': 'username', 'created': 'create_time', 'command': 'cmdline'}
    expected = {key: expected.get(key, expected.get(alias)) for key, alias in aliases.items()} | {
        'pid': expected.get('pid'), 'cwd': expected.get('cwd')}
    if any(v is None for v in expected.values()):
        raise ValueError('Identity requires pid, owner, created, command and cwd')
    try:
        p = psutil.Process(expected['pid'])
        with p.oneshot():
            actual = dict(pid=p.pid, owner=p.username(), created=p.create_time(),
                          command=p.cmdline(), cwd=p.cwd())
        return dict(exists=True, matches=actual == expected, expected=expected, actual=actual)
    except psutil.NoSuchProcess:
        return dict(exists=False, matches=False, expected=expected)
    except psutil.AccessDenied:
        return dict(exists=True, matches=None, expected=expected, error='Access denied')


def last_update(path):
    # Only project the last complete line's counters; never export losses/ids.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length - 1024 * 1024))
        lines = stream.read().splitlines()
    for line in reversed(lines[-2:]):
        try:
            row = json.loads(line)
            return {key: row[key] for key in ('stage', 'epoch', 'stage_update', 'updates', 'total_updates')
                    if key in row}
        except (ValueError, UnicodeDecodeError):
            continue
    raise ValueError(f'No complete update in bounded tail: {path}')


def argument(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def observe(root):
    root = Path(root).resolve(strict=True)
    launch, gate = read_object(root / 'launch.json'), read_object(root / 'gate.json')
    campaign = Path(gate['campaign']).resolve()
    if not campaign.is_relative_to(root):
        raise ValueError('Campaign must be inside the explicitly selected root')
    controller = identity(launch)
    binding = launch.get('gate', {})
    gate_matches = (binding.get('bytes') == (root / 'gate.json').stat().st_size and
                    binding.get('sha256') == hashlib.sha256((root / 'gate.json').read_bytes()).hexdigest())
    dispatch = read_object(campaign / 'dispatch.json') if (campaign / 'dispatch.json').exists() else {}
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
        updates = folder / 'run/updates.jsonl'
        receipt = folder / 'dispatch_receipt.json'
        saved = read_object(receipt) if receipt.exists() else {}
        training.append(dict(task=folder.name, phase=folder.parent.parent.name,
                             counters=last_update(updates) if updates.exists() else None,
                             updates_mtime_utc=datetime.fromtimestamp(updates.stat().st_mtime, timezone.utc).isoformat()
                             if updates.exists() else None,
                             training_completion_record_exists=(folder / 'run/completion.json').exists(),
                             wrapper_validation_success=saved.get('success'),
                             worker_released=saved.get('released'),
                             failure_record_exists=(folder / 'worker_failure.json').exists()))
    settled = budget.get('used_seconds', {})
    return dict(observed_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                controller=controller, launch_gate_matches=gate_matches,
                dispatch={key: dispatch.get(key) for key in ('status', 'pending', 'retired', 'budget_stop')},
                dispatch_failure_count=len(dispatch.get('failures', [])),
                controller_error_present=bool(dispatch.get('controller_error')),
                active=active, training=training,
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
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != PYTHON.resolve():
        parser.error('Run only with the existing Linux py310 Python')
    result = observe(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['controller']['matches'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
