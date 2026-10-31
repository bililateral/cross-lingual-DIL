"""One repaired CAB_s6, then merge the unchanged donor queue and hand off.

The donor keeps its original controller, failures, sources and accounting.
Only the explicitly authorized new attempt is dispatched here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import psutil

import step28_comparison_control as shared
import step28_continual_comparison_evaluate as evaluation
import step28_defaults_control as defaults

run, data, m = shared.run, shared.data, shared.m
TARGET = 'CAB_s6_L2R_text_fixed'
LIMITS = dict(training=800*3600, confirmation=18*3600, cpu=1200, native=3000,
              generation=600, control=3600, analysis=1800, preparation=4200)


def pinned(row):
    return data.read_json(data.verify(Path(row['path']), row))


def donor(gate):
    old = pinned(gate['donor']['gate'])
    launch = pinned(gate['donor']['launch'])
    if (old['campaign'] != gate['donor']['campaign'] or old['workspace'] != launch['cwd']
            or launch['formal_gate'] != gate['donor']['gate']):
        raise ValueError('Donor gate/launch identity differs')
    return old, launch


def validate(gate_path):
    gate = run.gate(gate_path, 'train')
    old, _ = donor(gate)
    data.verify(Path(gate['authorization']['path']), gate['authorization'])
    ids = {s.run_id for s in m.specifications(new_only=True)}
    if (gate.get('single_repair') != TARGET or gate['budget_seconds'] != LIMITS
            or gate['per_run_seconds'] != old['per_run_seconds'] or gate['limits'] != old['limits']
            or gate['maximum_workers'] != 3 or not gate['train_data'] or not gate['new_confirmation_data']
            or set(gate['adoption']) != ids-{TARGET}
            or gate['approved_actions'] != ['train', 'generate', 'confirmation', 'analyze']
            or gate['reference_gate'] != old['reference_gate']
            or gate['reference_campaign'] != old['reference_campaign']):
        raise ValueError('Only one fixed CAB repair and the original complete comparison are allowed')
    for identity, item in gate['adoption'].items():
        expected = str(Path(old['campaign'])/identity/'run/completion.json')
        source = (old['recovery']['original_gate'] if identity in old['recovery']['reused']
                  else gate['donor']['gate'])
        if item != dict(completion=expected, source_gate=source):
            raise ValueError('Unexpected donor/source mapping')
    for row in gate['orchestration_sources']:
        data.verify(data.ROOT/row['path'], row)
    ready = pinned(gate['review_ready'])
    if ready['status'] != 'REVIEW_CLOSED_AND_CPU_PASSED' or ready['sources'] != gate['sources']:
        raise ValueError('Exact repair closure missing')
    cpu = pinned(ready['cpu_receipt'])
    if (not cpu['success'] or cpu['sources'] != gate['sources']
            or cpu['orchestration_sources'] != gate['orchestration_sources']):
        raise ValueError('Orchestration evidence differs')
    return gate


def donor_processes(gate):
    old, launch = donor(gate)
    return defaults.predecessor_processes(dict(workspace=old['workspace'], campaign=old['campaign'],
                                               gate=gate['donor']['gate']), launch)


def check_combined_budget(old, own, pending_gcr=False):
    """Combine actual charges once; reservations are reported separately."""
    reserved = {key: 0. for key in LIMITS}
    for ledger in (old, own):
        if ledger.get('budget_exceeded'):
            raise RuntimeError('A shared budget was exceeded')
        for row in ledger['active'].values():
            reserved[row['bucket']] += row['allowance']
    if pending_gcr:
        reserved['training'] += 12*3600
    used = {key: old['used'][key]+own['used'][key] for key in LIMITS}
    if any(used[key]+reserved[key] > LIMITS[key] for key in LIMITS):
        raise RuntimeError('Combined actual charges/reservations exceed the original budget')
    return dict(used=used, reserved=reserved, limits=LIMITS)


def donor_budget(gate):
    path = Path(gate['donor']['campaign'])
    old = data.read_json(path/'budget.json')
    pending = ('CAB_s6_ER_GCR_group_fixed' not in old['active'] and
               not any(r['id'] == 'CAB_s6_ER_GCR_group_fixed' for r in old['completed']))
    return old, pending


def combined_resources(gate, campaign):
    # Count physical payloads once when adoption hard-links the same endpoints.
    seen, total, small = set(), 0, 0
    for root in (Path(gate['donor']['campaign']), campaign):
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # An active worker retired its temporary checkpoint.
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_size
                small += stat.st_size if path.suffix != '.pt' else 0
                seen.add(key)
    old_processes = donor_processes(gate)
    own = shared.prior.run.owned_processes(os.getpid())
    for row in old_processes:
        try:
            own[row['pid']] = psutil.Process(row['pid'])
        except psutil.NoSuchProcess:
            pass
    rss = shared.prior.run.resident_bytes(own)
    if (total > gate['limits']['disk_bytes'] or small > gate['limits']['small_bytes']
            or rss > 192*2**30):
        raise RuntimeError('Combined campaign storage/RSS limit exceeded')


def inspect_donor(gate):
    path = Path(gate['donor']['campaign'])
    if donor_processes(gate):
        return None
    failure = data.read_json(path/'failure.json')
    budget = failure['budget']
    failed = [r for r in budget['completed'] if not r['success']]
    successful = {r['id'] for r in budget['completed'] if r['success']}
    expected_error = repr(RuntimeError('Worker failure retained after active siblings finished: '+TARGET))
    if (failure['status'] != 'STOP_NO_RETRY' or failure['error'] != expected_error
            or budget['active'] or budget['budget_exceeded']
            or len(failed) != 1 or failed[0]['id'] != TARGET
            or successful != set(gate['adoption']) or len(budget['completed']) != 18):
        raise ValueError('Donor did not finish all 17 required healthy trajectories')
    for forbidden in ('training_complete.json', 'completion.json', 'confirmation', 'analysis'):
        if (path/forbidden).exists():
            raise ValueError('Unexpected donor post-training work')
    return failure


def adopt(campaign, gate, old_failure, ledger):
    """Pin exact successful outputs; never rewrite donor success/failure records."""
    if ledger.failed or ledger.active or len(ledger.completed) != 1 or not ledger.completed[0]['success']:
        raise ValueError('Repaired CAB trajectory did not complete')
    old = old_failure['budget']
    check_combined_budget(old, ledger.snapshot())
    receipts = {}
    for identity, item in gate['adoption'].items():
        completion = Path(item['completion'])
        row = data.read_json(completion)
        source_gate = pinned(item['source_gate'])
        if row['source_files'] != source_gate['sources'] or row['status'] != run.COMPLETE:
            raise ValueError('Donor completion/source mismatch')
        receipts[identity] = dict(donor_completion=str(completion),
                                  completion=data.record(completion, completion.parent))
        source = completion.parent.parent
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Unexpected symlink in a retained trajectory')
        shutil.copytree(source, campaign/identity, copy_function=os.link)
    shared.write(campaign/'adoption.json', dict(runs=receipts))
    shared.write(campaign/'inherited_failure.json', dict(donor=gate['donor'], failure=old_failure,
                 new_attempt_budget=ledger.snapshot(), note='Old failure and all old fees remain valid history.'))
    for key in LIMITS:
        ledger.used[key] += old['used'][key]
    ledger.completed = [r for r in old['completed'] if r['success']]+ledger.completed
    ledger.save()


def execute(gate_path):
    gate = validate(gate_path)
    campaign = Path(gate['campaign'])
    campaign.mkdir()
    # Before adoption this ledger contains new work only; the donor is unchanged.
    ledger = shared.Ledger(campaign/'budget.json', LIMITS, gate['additional_used_seconds'])
    ledger.save()
    merged = False
    try:
        old, pending = donor_budget(gate)
        reservation = dict(ledger.snapshot())
        reservation['active'] = {TARGET: dict(bucket='training', allowance=72*3600)}
        check_combined_budget(old, reservation, pending)
        tick = time.monotonic()
        try:
            shared.reference_check(gate)
        finally:
            ledger.charge('preparation', time.monotonic()-tick)

        def check():
            combined_resources(gate, campaign)
            if not merged:
                old, pending = donor_budget(gate)
                shared.write(campaign/'combined_budget.json',
                             check_combined_budget(old, ledger.snapshot(), pending))

        launcher = Path(gate['launcher'])

        def observe():
            if not launcher.exists():
                return
            log = campaign/'dispatch'/TARGET/'console.txt'
            if log.exists():
                for line in log.read_text().splitlines():
                    if line.startswith('{'):
                        event = json.loads(line)
                        if (event.get('event') == 'updates' and event.get('run_id') == TARGET
                                and event.get('completed', 0) > 0):
                            launcher.unlink()
                            shared.write(campaign/'first_update.json', dict(event=event, launcher_deleted=True))
                            return

        spec = m.Spec('L2R_text_fixed', 'CAB', 6)

        def command(row, job, deadline, metadata):
            shared.write(metadata/'spec.json', spec.identity())
            return [os.sys.executable, '-B', str(Path(run.__file__)), '--job', str(job),
                    '--spec', str(metadata/'spec.json'), '--gate', str(gate_path), '--deadline', str(deadline)]

        shared.write(campaign/'status.json', dict(status='ONE_REPAIRED_CAB_WAIT_OR_TRAIN', epoch=time.time()))
        shared.dispatch([dict(id=TARGET, allowance=72*3600)], campaign, ledger, 'training', command,
                        max_workers=1, admit=lambda count: shared.resources(campaign, active_workers=count),
                        check=check, observe=observe)
        while True:
            tick = time.monotonic()
            try:
                check()
                finished = inspect_donor(gate)
                shared.write(campaign/'status.json', dict(status='WAITING_FOR_17_DONOR_SUCCESSES', epoch=time.time()))
            finally:
                ledger.charge('control', time.monotonic()-tick)
            if finished is not None:
                break
            time.sleep(60)
        tick = time.monotonic()
        try:
            adopt(campaign, gate, finished, ledger)
            merged = True
            records = evaluation.full_gate(campaign, gate)
        finally:
            ledger.charge('preparation', time.monotonic()-tick)
        shared.write(campaign/'training_complete.json', dict(status='ALL_18_NEW_AND_54_REFERENCE_ENDPOINTS_COMPLETE',
                     new_run_ids=[s.run_id for s in m.specifications(new_only=True)], all_runs=len(records),
                     sources=gate['sources'], budget=ledger.snapshot()))
        for action, identity, bucket in [('generate','generation','generation'),
                                         ('confirmation','confirmation','confirmation'),('analyze','analysis','analysis')]:
            def post_command(row, job, deadline, metadata):
                return [os.sys.executable, '-B', str(Path(evaluation.__file__)), action,
                        '--campaign', str(campaign), '--gate', str(gate_path), '--deadline', str(deadline)]
            shared.dispatch([dict(id=identity, allowance=ledger.available(bucket))], campaign, ledger,
                            bucket, post_command, max_workers=1, check=check,
                            admit=(lambda count: shared.resources(campaign, active_workers=count)) if action == 'confirmation' else None)
        shared.write(campaign/'completion.json', dict(status=defaults.PREDECESSOR_COMPLETE,
                     sources=gate['sources'], budget=ledger.snapshot()))
    except BaseException as error:
        shared.write(campaign/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error),
                     traceback=traceback.format_exc(), budget=ledger.snapshot(), inherited_merged=merged))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    execute(parser.parse_args().gate.resolve())
