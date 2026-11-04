"""One-shot H/S -> FOREVER -> VendorLink chain; no retries or science changes."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import psutil

import step28_formal_queue as queue


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relevant_processes(gate_path, campaign, owner):
    found = []
    for p in psutil.process_iter(['pid', 'username', 'create_time', 'cmdline', 'status']):
        try:
            info = p.info
            if (p.pid == os.getpid() or info['username'] != owner
                    or info['status'] == psutil.STATUS_ZOMBIE):
                continue
            args = info['cmdline'] or []
            if str(gate_path) in args or any(a.startswith(str(campaign)+'/') for a in args):
                found.append(dict(pid=p.pid, owner=owner, created=info['create_time'],
                                  command=args, cwd=p.cwd()))
        except psutil.NoSuchProcess:
            pass
    return found


def valid_budget(budget, expected):
    completed = budget['completed']
    if (budget.get('active') or budget.get('failed') or budget.get('budget_exceeded')
            or len(completed) != len(expected) or {r['id'] for r in completed} != expected
            or not all(r['success'] for r in completed)
            or any(not math.isfinite(v) or v < 0 or v > budget['limits'][k]
                   for k, v in budget['used'].items())):
        raise ValueError('Predecessor is incomplete, failed or over budget')


def predecessor_state(binding):
    gate_path = queue.verify(binding['gate'])
    gate = read(gate_path)
    campaign = Path(gate['campaign'])
    alive = relevant_processes(gate_path, campaign, binding['owner'])
    if (campaign/'failure.json').exists():
        raise RuntimeError('Predecessor failed; downstream remains closed')
    if not (campaign/'completion.json').exists():
        if not alive:
            raise RuntimeError('Predecessor exited without complete success; no launch')
        return dict(ready=False, reason='PREDECESSOR_STILL_RUNNING', processes=alive)
    result = read(campaign/'completion.json')
    if binding['kind'] == 'H/S':
        expected = {f"{r['order']}_s{r['seed']}_{r['name']}" for r in gate['runs']}
        if (len(expected) != 18 or result['status'] != 'ALL_18_ORIGINAL_RUNS_AND_54_ENDPOINTS_BLIND_COMPLETE'
                or result['sources'] != gate['sources'] or result['endpoints'] != 54
                or result['evaluation_labels_opened'] is not False
                or result['new_confirmation_generated'] is not False
                or len(result['runs']) != 18 or {r['id'] for r in result['runs']} != expected):
            raise ValueError('H/S full 18-run / 54-endpoint receipt differs')
        valid_budget(result['budget'], expected)
        # Verify the recorded per-run receipts and endpoint presence without opening scores/labels.
        for item in result['runs']:
            rec = item['completion']
            path = queue.verify({**rec, 'path': str(campaign/rec['path'])})
            run = read(path)
            if (run['source_files'] != gate['sources'] or run['updates'] != 864
                    or set(run['points']) != {'1', '2', '3'}):
                raise ValueError('H/S trajectory metadata incomplete')
            for point in run['points'].values():
                if not point['full_restore_verified'] or not point['inference_restore_verified']:
                    raise ValueError('H/S endpoint restore evidence missing')
                for payload in (point['model'], point['mapping'], *point['scores'].values()):
                    if (path.parent/payload['path']).stat().st_size != payload['bytes']:
                        raise ValueError('H/S endpoint payload absent or incomplete')
    elif binding['kind'] == 'FOREVER':
        expected = {f"{r['order']}_s{r['seed']}_{r['name']}" for r in gate['runs']}
        expected.update(('generation', 'confirmation', 'analysis'))
        if (len(expected) != 12 or result['status'] != 'ALL_FORMAL_ACTIONS_COMPLETE'
                or result['sources'] != gate['sources']):
            raise ValueError('FOREVER full training/confirmation/analysis receipt differs')
        valid_budget(result, expected)
        # The actual training validators and full blind gate already ran in this chain.
        # Runtime validity is required; the scientific primary result may be false.
        if (read(campaign/'confirmation/result.json')['status'] != 'ALL_FOREVER_METRICS_COMPLETE'
                or read(campaign/'analysis/result.json')['status'] != 'FOREVER_FIXED_COMPARISON_COMPLETE'):
            raise ValueError('FOREVER confirmation/analysis completion missing')
    else:
        raise ValueError('Unknown predecessor')
    return dict(ready=not alive, reason='PREDECESSOR_EXIT_PENDING' if alive else 'COMPLETE',
                processes=alive, completion=queue.record(campaign/'completion.json'))


def observe_first_update(allowed, campaign):
    listener = Path(allowed['listener'])
    if not listener.exists():
        return
    for path in sorted(campaign.glob('*/run/updates.jsonl')):
        with path.open(encoding='utf-8') as stream:
            line = stream.readline()
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A writer may not yet have finished its first line.
        if row.get('role') == 'current':
            queue.verify({**allowed['listener_source'], 'path': str(listener)})
            queue.write(campaign/'first_update.json', dict(epoch=time.time(), log=str(path),
                event=row, listener_deleted_after_update=True))
            listener.unlink()
            return


def run(forever_gate, vendor_gate, state, listener):
    started = time.monotonic()
    first, second = read(forever_gate), read(vendor_gate)
    if (first['experiment'] != 'FOREVER' or second['experiment'] != 'VendorLink'
            or first['predecessor']['kind'] != 'H/S'
            or second['predecessor']['kind'] != 'FOREVER'
            or second['predecessor']['gate'] != queue.record(forever_gate)
            or first['listener'] != str(listener)
            or first['workspace'] != second['workspace']):
        raise ValueError('Exact H/S -> FOREVER -> VendorLink chain required')
    queue.verify({**first['listener_source'], 'path': str(listener)})
    state.mkdir()  # Unique durable claim persists after the listener deletes itself.
    p = psutil.Process()
    queue.write(state/'process.json', dict(pid=p.pid, created=p.create_time(), owner=p.username(),
        command=p.cmdline(), cwd=p.cwd(), forever_gate=queue.record(forever_gate), vendor_gate=queue.record(vendor_gate)))
    try:
        for module_name, gate_path, experiment in (
                ('step28_forever_formal', forever_gate, 'FOREVER'),
                ('step29_vendorlink_formal', vendor_gate, 'VendorLink')):
            tick = started if experiment == 'FOREVER' else time.monotonic()
            queue.write(state/'status.json', dict(status='RUNNING_OR_WAITING', experiment=experiment, epoch=time.time()))
            module = importlib.import_module(module_name)
            module.ENTRY_STARTED = tick
            sys.argv = [str(Path(module.__file__).resolve()), 'train', '--gate', str(gate_path)]
            module.main()  # Same controller process; sequential, with no extra polling budget.
        queue.write(state/'completion.json', dict(status='BOTH_FORMAL_BATCHES_COMPLETE', epoch=time.time()))
    except BaseException as error:
        queue.write(state/'failure.json', dict(status='STOP_NO_AUTOMATIC_RETRY',
            error=repr(error), traceback=traceback.format_exc(), epoch=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forever-gate', required=True, type=Path)
    parser.add_argument('--vendor-gate', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    args = parser.parse_args()
    run(args.forever_gate.resolve(), args.vendor_gate.resolve(), args.state.resolve(), Path(__file__).resolve())
