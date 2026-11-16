"""One explicit postprocessing recovery; no training or generation retries."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pinned(binding):
    if record(binding['path']) != binding:
        raise ValueError('Recovery binding differs: '+binding['path'])
    return Path(binding['path'])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    p = psutil.Process()
    return dict(pid=p.pid, owner=p.username(), command=p.cmdline(), cwd=p.cwd(),
                start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def restore_beta(request, request_path, state):
    beta = request['beta']
    root = Path(beta['root'])
    old, failure = read(pinned(beta['gate'])), read(pinned(beta['failure']))
    if ((root/'campaign').exists()
            or failure['error'] != "RuntimeError('Chinese dependency failed; no automatic beta launch')"):
        raise ValueError('Only the failed, unstarted dependency waiter may be restored')
    prior = failure['prior_seconds']['auxiliary']+failure['charged_seconds']+90
    if prior+150 >= old['caps']['auxiliary']:
        raise TimeoutError('Independent beta allowance is insufficient')
    archive, gate = root/'wait_before_postprocess_resume', root/'gate_after_postprocess_resume.json'
    if archive.exists() or gate.exists():
        raise ValueError('This waiting recovery has already been attempted')
    new = copy.deepcopy(old)
    new['dependency'].update(request=record(request_path), controller=identity(), state=str(state))
    new['prior_seconds']['auxiliary'] = prior
    (root/'wait').rename(archive)
    write(gate, new)
    command = [sys.executable, '-B', '-u', str(pinned(beta['entry'])), '--gate', str(gate)]
    with (state/'beta_console.txt').open('x') as log:
        child = subprocess.Popen(command, cwd=old['workspace'], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    end = time.monotonic()+30
    while not (root/'wait/status.json').exists():
        if child.poll() is not None or time.monotonic() >= end:
            raise RuntimeError('Restored beta waiter did not acknowledge')
        time.sleep(.1)
    current = read(root/'wait/status.json')
    if current['status'] != 'WAITING_CHINESE_COMPLETION' or current['identity']['pid'] != child.pid:
        raise RuntimeError('Restored waiter identity/status differs')
    write(state/'beta_handover.json', dict(identity=current['identity'], gate=record(gate),
        archived_wait=str(archive), prior_auxiliary_seconds=prior, handover_conservative_seconds=90,
        scientific_sources_unchanged=True, training_started=False))


def execute(request_path):
    started = time.monotonic()
    request_path = Path(request_path).resolve()
    request = read(request_path)
    if (not request['approved'] or not request['review_closed']
            or request['actions'] != ['confirmation', 'analyze']
            or request['automatic_retry'] is not False
            or request['budget_caps'] != dict(training=1360800, confirmation=43200, auxiliary=None)):
        raise ValueError('Explicit finite postprocessing recovery required')
    for b in request['updated_sources']+request['prior_bindings']+[
            request['authorization'], request['review'], request['qualification']]:
        pinned(b)
    for p in psutil.process_iter(['pid', 'cmdline', 'status']):
        if p.pid != os.getpid() and p.info['status'] != psutil.STATUS_ZOMBIE:
            if any(request['workspace'] in a or request['beta']['root'] in a
                   or request['prior_request'] == a for a in (p.info['cmdline'] or [])):
                raise RuntimeError(f'Related predecessor remains: {p.pid}')
    root, state = Path(request['campaign']), request_path.parent/'state'
    state.mkdir()
    with (state/'claim.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            dispatch = read(root/'training/dispatch.json')
            if (dispatch['status'] != 'ALL_COMPLETE_RELEASED' or dispatch['active'] or dispatch['pending']
                    or dispatch['retired_slots'] or len(dispatch['completed']) != 63
                    or not all(r['success'] and r['released'] for r in dispatch['completed'])):
                raise ValueError('All 63 training trajectories must already be complete')
            if any((root/n).exists() for n in ('completion.json', 'confirmation', 'analysis',
                                             'confirmation_dispatch', 'analysis_dispatch')):
                raise ValueError('Postprocessing may only start once after this failure')
            generated = read(root/'generation/result.json')
            if (generated['status'] != 'GENERATED_VALIDATED_OWNERS_SEALED' or generated['groups'] != 120
                    or generated['root_seed'] != 2026110601
                    or generated['namespace'] != 'h_history_ablation_confirmation_20261106'
                    or generated['sources'] != dispatch['sources']):
                raise ValueError('The already-generated fixed confirmation batch differs')
            access = read(root/'evaluation_access.json')['events']
            if len(access) != 1 or access[0]['role'] != 'ablation_generation':
                raise ValueError('Unexpected prior confirmation access')
            for name, binding in generated['files'].items():
                p = root/'generation'/name
                if p.is_symlink() or p.stat().st_uid != os.getuid() or p.stat().st_size != binding['bytes']:
                    raise ValueError('Generated artifact metadata differs')
            if (root/'generation/owners.jsonl').stat().st_mode & 0o777:
                raise ValueError('Generated owners must remain sealed')
            previous = state/'previous_segment'
            previous.mkdir()
            for name in ('failure.json', 'budget.json', 'cost_records.json', 'execution_time.json'):
                source, target = root/name, previous/name
                target.write_bytes(source.read_bytes())
                if {**record(target), 'path': str(source)} != record(source):
                    raise ValueError('Failed segment archival differs')
            write(state/'generation_accepted.json', dict(
                result=record(root/'generation/result.json'), original_receipt=record(root/'generation/dispatch_receipt.json'),
                reason='Generation completed and sealed; only final resources JSON failed for positive infinite deadline',
                generated_again=False, payloads_read_by_recovery=False,
                payload_hash_checks_deferred_to_original_authorized_evaluator=True,
                original_exit_and_failure_preserved=True, approved_request=record(request_path)))
            budget = read(root/'budget.json')
            if budget['caps'] != request['budget_caps'] or budget['used_seconds']['confirmation'] != 0:
                raise ValueError('Inherited cumulative budget differs')
            preparation = math.ceil(time.time()-request['preparation_started_epoch'])
            budget['used_seconds']['auxiliary'] += preparation
            budget['events'].append(dict(id='postprocessing_recovery_preparation', bucket='auxiliary',
                conservative_seconds=preparation, cpu_test_seconds=request['cpu_test_seconds'],
                basis='Calendar envelope including diagnosis, recovery preparation and CPU subset; not CPU time'))
            update = dict(request=str(request_path), updated_sources=request['updated_sources'], review=request['review'])
            continuation = dict(dispatch=dispatch, budget=budget, controller_update=update,
                prior_controller=dict(identity=request['prior_controller'], stop_reason='GENERATION_RESOURCE_JSON_INFINITY',
                                      evidence=str(previous)))
            write(state/'continuation.json', continuation)
            (root/'failure.json').unlink()
            write(state/'controller.json', dict(identity=identity(), phase='POSTPROCESSING_ONLY', request=str(request_path)))
            restore_beta(request, request_path, state)
            sys.path.insert(0, str(Path(request['workspace'])/'scripts'))
            control = load(pinned(request['control']), 'postprocess_control')
            dispatcher = load(pinned(request['dispatch']), 'approved_dispatch')
            control.execute(Path(request['original_gate']), entry_path=pinned(request['entry']),
                            request_path=request_path, continuation=continuation, dispatch_impl=dispatcher.dispatch)
            write(state/'completion.json', dict(status='CONTROLLED_CAMPAIGN_COMPLETE', epoch=time.time()))
        except BaseException as error:
            write(state/'failure.json', dict(error=repr(error), automatic_retry=False))
            if not (root/'failure.json').exists():
                write(root/'failure.json', dict(status='STOP_NO_RETRY', error=repr(error)))
            raise


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    execute(parser.parse_args().request)
