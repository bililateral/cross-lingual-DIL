"""Finite adaptive ABC campaign; two persistent slots, one ledger, no retries."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

import step29_english_formal as formal

queue, policy = formal.queue, formal.policy


def release(process, sleep=time.sleep):
    """Release only the process group created for this worker, including descendants."""
    def members():
        found = []
        for p in psutil.process_iter(['pid', 'status', 'username']):
            try:
                if os.getpgid(p.pid) == process.pid and p.status() != psutil.STATUS_ZOMBIE:
                    if p.username() != psutil.Process().username() or p.pid == os.getpid():
                        raise RuntimeError('Foreign process in owned worker group')
                    found.append(p.pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
        return found
    unexpected = process.poll() is not None and bool(members())
    for sig, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        if not members():
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic()+seconds
        while members() and time.monotonic() < end:
            sleep(.02)
    process.wait(timeout=1)
    if members():
        raise RuntimeError('Owned process group did not release')
    return unexpected


class Ledger:
    def __init__(self, root, prior, caps):
        self.root, self.used, self.caps = root, dict(prior), caps
        if set(prior) != set(caps) or any(not math.isfinite(v) or v < 0 or v >= caps[k] for k,v in prior.items()):
            raise ValueError('Cumulative finite fees in all three buckets required')
        self.events = []
        self.tick = time.monotonic()
        self.used['auxiliary'] += 30  # Finite final cleanup/save allowance, inside cap.

    def save(self):
        queue.write(self.root/'budget.json', dict(used_seconds=self.used, caps=self.caps,
                                                final_tail_seconds=30, events=self.events))

    def control(self):
        now = time.monotonic()
        self.used['auxiliary'] += now-self.tick
        self.tick = now
        self.save()
        if any(self.used[k] >= self.caps[k] for k in self.caps):
            raise TimeoutError('Cumulative campaign budget exhausted')

    def pause(self, seconds):
        self.control()
        self.exclude_sleep(seconds)

    def exclude_sleep(self, seconds):
        # Cleanup must remain possible even after a cap: do not call control here.
        begin = time.monotonic()
        try:
            time.sleep(seconds)
        finally:
            self.tick += time.monotonic()-begin

    def finish(self, row, begin, end):
        seconds = math.ceil(end-begin)
        self.used[row['bucket']] += seconds
        self.events.append(dict(id=row['id'], bucket=row['bucket'], actual_seconds=end-begin,
                                charged_seconds=seconds))
        self.save()


def observe_first_update(root, active):
    if (root/'first_update.json').exists():
        return
    for slot, item in active.items():
        path = item['job']/'run/updates.jsonl'
        if not path.exists():
            continue
        with path.open() as stream:
            line = stream.readline()
        if not line.endswith('\n'):
            continue
        value = json.loads(line)
        if value.get('optimizer_updates') == 1 and value.get('stage') == 1 and value.get('total_updates') == 1:
            queue.write(root/'first_update.json', dict(epoch=time.time(), pid=item['process'].pid,
                created=item['created'], slot=slot, job=str(item['job']), update=value))
            return


class Runner:
    def __init__(self, allowed, ledger, monitor, poll_seconds=1., observe=None):
        self.allowed, self.ledger, self.monitor = allowed, ledger, monitor
        self.observe = observe
        self.root, self.poll = Path(allowed['campaign']), poll_seconds
        self.retired = set()  # Never reconstructed/reset at a new adaptive round.

    def dispatch(self, root, rows, command, validate):
        active, pending, receipts = {}, list(rows), []
        cores = psutil.Process().cpu_affinity()
        if len(cores) < self.allowed['workers']:
            raise ValueError('Two permitted CPU cores required')
        training = all(r['bucket'] == 'training' for r in rows)
        def saved(status):
            queue.write(root/'dispatch.json', dict(status=status, retired_slots=sorted(self.retired),
                pending=[r['id'] for r in pending], completed=receipts,
                active={str(s): dict(id=v['row']['id'], pid=v['process'].pid) for s,v in active.items()}))
        def finish(slot, error=None):
            item = active[slot]
            process, row = item['process'], item['row']
            unexpected = release(process, self.ledger.exclude_sleep)
            end = time.monotonic()
            item['log'].close()
            self.ledger.finish(row, item['begin'], end)
            del active[slot]
            if process.returncode or unexpected or end-item['begin'] > row['seconds']:
                error = error or f'Worker failed/overran: exit={process.returncode}, descendants={unexpected}'
            if error is None:
                try:
                    validate(row, item['job'])
                except Exception as exc:
                    error = repr(exc)
            receipt = dict(id=row['id'], slot=slot, success=error is None, error=error,
                           actual_seconds=end-item['begin'], released=True)
            receipts.append(receipt)
            queue.write(item['job']/'dispatch_receipt.json', receipt)
            if error:
                self.retired.add(slot)
            self.ledger.control()
            # Optional observer failure must not settle this released worker twice.
            observe_first_update(self.root, {slot: item})
            if self.observe is not None:
                self.observe()
        try:
            while pending or active:
                self.ledger.control()
                queue.check_running(None, self.allowed, self.root)
                observe_first_update(self.root, active)
                if self.observe is not None:
                    self.observe()
                snapshot = self.monitor.snapshot() if self.monitor else None
                for slot, item in list(active.items()):
                    row, p = item['row'], item['process']
                    error = None
                    if snapshot and snapshot['processes'].get(p.pid, 0) > self.allowed['limits']['process_gpu_bytes']:
                        error = 'PROCESS_GPU_LIMIT_EXCEEDED'
                    if time.monotonic() >= item['begin']+row['seconds']-row.get('shutdown_reserve', 60):
                        error = error or 'SHUTDOWN_RESERVE_REACHED'
                    if error or p.poll() is not None:
                        finish(slot, error)
                for slot in range(self.allowed['workers']):
                    if not pending or slot in active or slot in self.retired or (not training and active):
                        continue
                    row = pending[0]
                    reserved = sum(v['row']['seconds'] for v in active.values() if v['row']['bucket']==row['bucket'])
                    if self.ledger.used[row['bucket']]+reserved+row['seconds'] > self.ledger.caps[row['bucket']]:
                        if active:
                            break  # Wait for actual cost/refund before considering the next task.
                        raise TimeoutError('Cannot reserve the next complete task within remaining budget')
                    if row.get('gpu', True):
                        if self.monitor is None:
                            raise ValueError('Actual NVML admission required')
                        current = self.monitor.snapshot()
                        if self.allowed['limits']['free_gpu_bytes'] > current['total']:
                            raise ValueError('Even one required slot exceeds physical GPU capacity')
                        observation = queue.resources(self.allowed, self.root, current,
                                                      [v['process'].pid for v in active.values()])
                        queue.write(root/'admission.json', dict(epoch=time.time(), id=row['id'], **observation))
                        if not observation['ready']:
                            break
                    self.ledger.control()
                    job = root/row['id']
                    job.mkdir()
                    begin = time.monotonic()
                    output = (job/'console.txt').open('x')
                    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                           'TOKENIZERS_PARALLELISM':'false', 'CUDA_VISIBLE_DEVICES':'0' if row.get('gpu', True) else '',
                           'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}
                    try:
                        argv = ['taskset', '-c', str(cores[slot]), *command(row, job, begin+row['seconds'])]
                        p = subprocess.Popen(argv, cwd=self.allowed['workspace'], env=env, stdout=output,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    except OSError as exc:
                        output.close()
                        end=time.monotonic()
                        self.ledger.finish(row,begin,end)
                        self.retired.add(slot)
                        receipt=dict(id=row['id'],slot=slot,success=False,error=repr(exc),
                                     actual_seconds=end-begin,released=True,launch_failed=True)
                        receipts.append(receipt)
                        queue.write(job/'dispatch_receipt.json',receipt)
                        pending.pop(0)
                        continue
                    child = psutil.Process(p.pid)
                    active[slot] = dict(process=p, row=row, job=job, log=output, begin=begin, created=child.create_time())
                    queue.write(job/'process.json', dict(pid=p.pid, created=child.create_time(), owner=child.username(),
                                                         command=argv, cwd=self.allowed['workspace'], slot=slot))
                    pending.pop(0)
                saved('RUNNING')
                if pending and not active and len(self.retired) == self.allowed['workers']:
                    raise RuntimeError('All slots retired; no retry')
                if pending or active:
                    self.ledger.pause(self.poll)
            complete = len(receipts)==len(rows) and all(r['success'] for r in receipts)
            saved('COMPLETE' if complete else 'INCOMPLETE_NO_DOWNSTREAM')
            if not complete:
                raise RuntimeError('Incomplete round: no evaluation, feedback or dependent candidate')
        except BaseException:
            for slot in list(active):
                try:
                    finish(slot, 'CONTROLLER_STOP_NO_RETRY')
                except BaseException as exc:
                    queue.write(root/f'cleanup_error_{slot}.json', dict(error=repr(exc)))
            saved('STOP_NO_RETRY')
            raise


def pipeline(evaluate_old, round_action, save):
    """The actual finite choice sequence; callbacks run full rounds, not single metrics."""
    old = evaluate_old()
    expected = {f'ABC_s{s}_{m}' for s in (4,5,6) for m in ('H_full','H_ER','H_SEQ','VendorLink_SEQ')}
    if set(old) != expected:
        raise ValueError('Complete old seed 4/5/6 four-system precheck required')
    if not policy.precheck([old[f'ABC_s{s}_VendorLink_SEQ'] for s in (4,5,6)]):
        return dict(status='STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING')
    settings = {name: policy.candidate(name) for name in ('P0','P1','P2')}
    vendor = formal.Settings('VendorLink_SEQ')
    rows = [formal.spec(v,7) for v in settings.values()]+[formal.spec(vendor,7)]
    initial = round_action('representations', rows)
    def checked(values, specs):
        if set(values) != {r['id'] for r in specs}:
            raise ValueError('Missing/extra paired trajectory')
        for value in values.values(): policy.metrics(value)
        return values
    checked(initial, rows)
    metrics7 = {name:initial[formal.spec(v,7)['id']] for name,v in settings.items()}
    v7 = initial[formal.spec(vendor,7)['id']]
    def ordered(names, metrics_by_name, original, reference):
        return sorted((n for n in names if policy.basic_quality(metrics_by_name[n], original)),
            key=lambda n:policy.selection_key(n,settings[n],metrics_by_name[n],original,reference))
    incumbent = ordered(settings,metrics7,metrics7['P0'],v7)[0]
    for name in ('L1','L2','L3','T1','T2'):
        candidate = policy.candidate(name, settings[incumbent])
        duplicate = next((n for n,v in settings.items() if asdict(v)==asdict(candidate)),None)
        if duplicate is None:
            row = formal.spec(candidate,7)
            values = checked(round_action(name,[row]),[row])
            settings[name], metrics7[name] = candidate, values[row['id']]
        save('choice_'+name, dict(parent=incumbent, duplicate=duplicate,
                                  attempted_settings=asdict(candidate)))
        incumbent = ordered(settings,metrics7,metrics7['P0'],v7)[0]
    finalists = [n for n in ordered(settings,metrics7,metrics7['P0'],v7) if n!='P0'][:2]
    save('seed7_selection', dict(names=finalists, incumbent=incumbent, metrics=metrics7))
    if not finalists:
        return dict(status='STOP_NO_ELIGIBLE_SEED8_CANDIDATE')
    rows = [formal.spec(settings[n],8) for n in finalists]+[formal.spec(settings['P0'],8),formal.spec(vendor,8)]
    result8 = checked(round_action('seed8', rows), rows)
    h8, v8 = result8[formal.spec(settings['P0'],8)['id']],result8[formal.spec(vendor,8)['id']]
    passed, means, checks = [], {}, {}
    for name in finalists:
        n8 = result8[formal.spec(settings[name],8)['id']]
        checks[name] = policy.success([metrics7[name],n8],[metrics7['P0'],h8],[v7,v8])
        means[name] = policy.mean([metrics7[name],n8])
        if checks[name]['passed']:passed.append(name)
    save('freeze_checks', checks)
    if not passed:
        return dict(status='STOP_NO_CONFIGURATION_PASSES_FREEZE')
    winner = ordered(passed,means,policy.mean([metrics7['P0'],h8]),policy.mean([v7,v8]))[0]
    save('frozen', dict(name=winner, settings=asdict(settings[winner]), seeds=[7,8], success=checks[winner]))
    systems = dict(H_full=settings[winner],H_ER=replace(settings[winner],method='H_ER'),
                   H_SEQ=replace(settings[winner],method='H_SEQ'), original_H=settings['P0'], VendorLink=vendor)
    rows = [formal.spec(v,s) for s in (9,10,11) for v in systems.values()]
    final = checked(round_action('final',rows),rows)
    paired = {name:[final[formal.spec(v,s)['id']] for s in (9,10,11)] for name,v in systems.items()}
    verdict = policy.success(paired['H_full'],paired['original_H'],paired['VendorLink'],final=True)
    deltas = {name:{k:policy.mean(paired['H_full'])[k]-policy.mean(paired[name])[k]
                    for k in ('B2','B3','C3','A3','D')} for name in ('original_H','VendorLink','H_ER','H_SEQ')}
    return dict(status='COMPLETE_FIXED_DATA_DEVELOPMENT', selected=winner, verdict=verdict,
                paired_seed_order=[9,10,11], paired=paired, mean_differences=deltas,
                interpretation='No independent confirmation; H_ER contribution requires joint interpretation of all differences')


def execute(gate_path, *, observe=None):
    begin = time.monotonic()
    allowed = formal.gate(gate_path)
    root = Path(allowed['campaign'])
    if root.is_symlink() or not root.resolve().is_relative_to(Path(allowed['workspace']).resolve()):
        raise ValueError('Fresh campaign inside isolated workspace required')
    root.mkdir()  # One-shot claim. Never overwrite/restart an earlier attempt.
    ledger = Ledger(root,allowed['prior_used_seconds'],formal.CAPS)
    ledger.tick = begin
    monitor, result = None, None
    seen = set()
    try:
        ledger.control()
        monitor = queue.GpuMemory()
        runner = Runner(allowed, ledger, monitor, observe=observe)
        def round_action(name, specs):
            if any(row['id'] in seen for row in specs) or len({r['id'] for r in specs}) != len(specs):
                raise ValueError('Duplicate effective configuration/method/seed')
            seen.update(r['id'] for r in specs)
            folder = root/name;folder.mkdir()
            batch_path = folder/'batch.json'
            batch = dict(name=name,runs=specs,training_root=str(folder),gate_sha256=formal.run.file_sha256(gate_path))
            queue.write(batch_path,batch)
            def command(row,job,deadline):
                return [sys.executable,'-B',str(Path(formal.__file__).resolve()),row['action'],
                    '--gate',str(gate_path),'--batch',str(batch_path),'--job',str(job),
                    '--deadline',str(deadline),*(['--id',row['id']] if row['action']=='train' else [])]
            if specs:
                tasks=[dict(id=r['id'],spec=r,action='train',bucket='training',seconds=28800) for r in specs]
                runner.dispatch(folder,tasks,command,lambda row,job:formal.validate_completed(row['spec'],job/'run',allowed))
                (folder/'dispatch.json').replace(folder/'training_dispatch.json')
            # Serial evaluator uses the same ledger and surviving slots.
            seconds=math.floor(ledger.caps['gpu']-ledger.used['gpu']-60)
            if seconds <= 60:raise TimeoutError('No evaluation allowance remains')
            eval_specs = [r for r,_ in formal.evaluation_specs(batch,allowed)]
            def evaluate_command(row,job,deadline):
                # Dispatcher and worker agree on a direct child named evaluation.
                return command(row,job,deadline)
            runner.dispatch(folder,[dict(id='evaluation',action='evaluate',bucket='gpu',seconds=seconds)],
                evaluate_command,lambda row,job:formal.validate_evaluation(formal.read(job/'result.json'),eval_specs,name,allowed))
            (folder/'dispatch.json').replace(folder/'evaluation_dispatch.json')
            return formal.validate_evaluation(formal.read(folder/'evaluation/result.json'),eval_specs,name,allowed)
        result=pipeline(lambda:round_action('precheck',[]),round_action,lambda name,value:queue.write(root/(name+'.json'),value))
        ledger.control()
    except BaseException as exc:
        queue.write(root/'failure.json',dict(status='STOP_NO_RETRY',error=repr(exc)))
        raise
    finally:
        try:
            if monitor is not None:monitor.close()
        finally:
            try:
                ledger.control()
            except BaseException as exc:
                previous=formal.read(root/'failure.json') if (root/'failure.json').exists() else {}
                queue.write(root/'failure.json',dict(**previous,cleanup_error=repr(exc)))
                raise
    queue.write(root/'completion.json',dict(**result,used_seconds=ledger.used,sources=allowed['sources']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',type=Path,required=True)
    execute(p.parse_args().gate.resolve())
