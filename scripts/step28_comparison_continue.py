"""Continue the same three live training slots and the one remaining GCR job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback

import psutil

import step28_comparison_control as shared
import step28_continual_comparison_evaluate as evaluation
import step28_defaults_control as defaults
from step28_comparison_finish import LIMITS, TARGET, pinned

data, run, m = shared.data, shared.run, shared.m
PENDING = 'CAB_s6_ER_GCR_group_fixed'
ACTIVE = {TARGET, 'CAB_s4_L2R_text_fixed', 'CAB_s5_L2R_text_fixed'}


def exact_process(record):
    """Missing or reused PIDs are not the recorded worker."""
    try:
        process = psutil.Process(record['pid'])
        if abs(process.create_time()-record['created']) >= .01:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        observed = dict(owner=process.username(),cwd=process.cwd(),command=process.cmdline())
        if any(observed[k] != record[k] for k in observed):
            # /proc may lose cmdline between the first status read and normal exit.
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            if (observed['owner']==record['owner'] and observed['cwd']==record['cwd']
                    and not observed['command']):
                # Linux clears argv before the task becomes a zombie. Keep waiting;
                # an empty argv never proves success or permits another slot refill.
                return process
            raise RuntimeError(f'Recorded process identity changed: pid={record["pid"]}, observed={observed}, '
                               f'expected={ {k:record[k] for k in observed} }, status={process.status()}')
        return process
    except psutil.NoSuchProcess:
        return None


class AttachedWorker:
    """An orphan's exit code is unavailable: validate its real completion instead.

    No synthetic exit code is reported. Failure/missing/foreign completion rejects;
    the unchanged full 108-state gate remains mandatory before any evaluation.
    """
    def __init__(self, record, source_gate):
        self.record, self.source_gate = record, source_gate
        self.pid, self.job = record['pid'], Path(record['job'])
        self.returncode = None
        self.successful = False
        self.completion_check = None

    def poll(self):
        if self.completion_check is not None:
            return 'EXITED'
        if exact_process(self.record) is not None:
            return None
        outcome = dict(pid=self.pid, actual_exit_code=None,
                       basis='Exited recorded worker; independently checked saved completion', success=False)
        try:
            if (self.job/'failure.json').exists():
                raise ValueError('Worker retained a failure')
            path = self.job/'run/completion.json'
            row = data.read_json(path)
            if (row['status'] != run.COMPLETE or row['updates'] != 864
                    or m.Spec.from_dict(row['spec']).run_id != self.record['id']
                    or row['source_files'] != self.source_gate['sources']
                    or set(row['points']) != {'1','2','3'}
                    or row['access'] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
                raise ValueError('Attached completion is incomplete or foreign')
            for stage, point in row['points'].items():
                if (point['adam_step'] != int(stage)*288 or not point['full_restore_verified']
                        or not point['inference_restore_verified']):
                    raise ValueError('Attached stage restoration is incomplete')
            outcome.update(success=True, completion=data.record(path, self.job))
        except (OSError, ValueError, KeyError, TypeError) as error:
            outcome['error'] = repr(error)
        self.successful = outcome['success']
        self.completion_check = outcome
        return 'EXITED'

    def wait(self, timeout):
        process = exact_process(self.record)
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired as error:
                raise subprocess.TimeoutExpired(self.record['command'], timeout) from error
        self.poll()
        return None


def inherited(gate):
    claim = pinned(gate['takeover'])
    old, repair = claim['budgets']['donor'], claim['budgets']['repair']
    successes = [r for r in old['completed'] if r['success']]
    failures = [r for r in old['completed'] if not r['success']]
    active = {**old['active'], **repair['active']}
    if (len(successes) != 14 or len(failures) != 1 or failures[0]['id'] != TARGET
            or set(active) != ACTIVE or repair['completed'] or repair['failed']
            or old['budget_exceeded'] or repair['budget_exceeded']
            or set(claim['workers']) != ACTIVE or gate['pending_ids'] != [PENDING]
            or len({r['slot'] for r in claim['workers'].values()}) != 3):
        raise ValueError('The exact live 14-success/3-active/1-pending takeover differs')
    used = {k: old['used'][k]+repair['used'][k]+gate['additional_used_seconds'].get(k,0.) for k in LIMITS}
    for key in LIMITS:
        reserved = sum(r['allowance'] for r in active.values() if r['bucket']==key)
        if used[key]+reserved+(43200 if key=='training' else 0) > LIMITS[key]:
            raise ValueError('Combined inherited charges and whole queue exceed original budget')
    return claim, used, successes, active


def validate(path):
    gate = run.gate(path, 'train')
    if (gate['budget_seconds'] != LIMITS or gate['maximum_workers'] != 3
            or gate['approved_actions'] != ['train','generate','confirmation','analyze']
            or gate['pending_ids'] != [PENDING]
            or set(gate['adoption']) != {s.run_id for s in m.specifications(new_only=True)}):
        raise ValueError('Original queue/scope differs')
    data.verify(Path(gate['authorization']['path']),gate['authorization'])
    for row in gate['orchestration_sources']:
        data.verify(data.ROOT/row['path'],row)
    ready = pinned(gate['review_ready'])
    cpu = pinned(ready['cpu_receipt'])
    if (ready['status'] != 'REVIEW_CLOSED_AND_CPU_PASSED' or not cpu['success']
            or cpu['sources'] != gate['sources'] or cpu['orchestration_sources'] != gate['orchestration_sources']):
        raise ValueError('Exact continuation evidence is missing')
    old, repair = pinned(gate['donor_gate']), pinned(gate['repair_gate'])
    if (gate['limits'] != old['limits'] or gate['per_run_seconds'] != old['per_run_seconds']
            or gate['reference_gate'] != old['reference_gate']
            or gate['reference_campaign'] != old['reference_campaign']):
        raise ValueError('Scientific resources/reference scope changed')
    for identity, row in gate['adoption'].items():
        owner = repair if identity == TARGET else old
        source = (gate['repair_gate'] if identity == TARGET else
                  old['recovery']['original_gate'] if identity in old['recovery']['reused'] else gate['donor_gate'])
        if row != dict(completion=str(Path(owner['campaign'])/identity/'run/completion.json'), source_gate=source):
            raise ValueError('Fixed result provenance differs')
    inherited(gate)
    return gate


def resources(gate, campaign, workers):
    # Include the independently running adopted workers, not just new descendants.
    cohort = shared.prior.run.owned_processes(os.getpid())
    for record in workers.values():
        process = exact_process(record)
        if process is not None:
            cohort[process.pid] = process
            cohort.update({p.pid:p for p in process.children(recursive=True)})
    seen, total, small = set(), 0, 0
    roots = [campaign, Path(pinned(gate['donor_gate'])['campaign']), Path(pinned(gate['repair_gate'])['campaign'])]
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev,stat.st_ino)
            if key not in seen:
                seen.add(key);total += stat.st_size
                small += stat.st_size if path.suffix != '.pt' else 0
    if (total > gate['limits']['disk_bytes'] or small > gate['limits']['small_bytes']
            or shared.prior.run.resident_bytes(cohort) > 192*2**30):
        raise RuntimeError('Combined original resource limit exceeded')
    return set(cohort)


def adopt(campaign, gate, ledger):
    expected = set(gate['adoption'])
    if (ledger.failed or ledger.active or len(ledger.completed) != 18
            or {r['id'] for r in ledger.completed} != expected
            or not all(r['success'] for r in ledger.completed)):
        raise ValueError('All original 18 trajectories are required')
    receipts = {}
    for identity,item in gate['adoption'].items():
        path = Path(item['completion']); row=data.read_json(path)
        if row['source_files'] != pinned(item['source_gate'])['sources'] or row['status'] != run.COMPLETE:
            raise ValueError('Completion/source identity differs')
        source = path.parent.parent
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Unexpected trajectory symlink')
        receipts[identity] = dict(donor_completion=str(path),completion=data.record(path,path.parent))
        shutil.copytree(source,campaign/identity,copy_function=os.link)
    shared.write(campaign/'adoption.json',dict(runs=receipts))


def execute(gate_path):
    gate=validate(gate_path);claim,used,completed,active=inherited(gate)
    campaign=Path(gate['campaign']);campaign.mkdir()
    ledger=shared.Ledger(campaign/'budget.json',LIMITS,used)
    ledger.completed=completed;ledger.active=active;ledger.save()
    attached={}
    try:
        cores=psutil.Process().cpu_affinity()[:3]
        for identity,record in claim['workers'].items():
            process=exact_process(record)
            if process is None or process.cpu_affinity()!=[cores[record['slot']]] or os.getpgid(process.pid)!=process.pid:
                raise ValueError('Live worker/core/process-group identity differs')
            item=gate['adoption'][identity]
            worker=AttachedWorker(record,pinned(item['source_gate']))
            metadata=campaign/'dispatch'/identity;metadata.mkdir(parents=True)
            shared.write(metadata/'attached.json',dict(process=record,reservation=active[identity]))
            attached[identity]=(worker,open(os.devnull,'w'),record['slot'])
        shared.write(campaign/'ready.json',dict(status='LIVE_WORKERS_VALIDATED_WAITING_FOR_OWNERSHIP',pids=[x[0].pid for x in attached.values()]))
        deadline=time.monotonic()+300
        while not (campaign/'activate.json').exists():
            if time.monotonic()>=deadline:
                raise TimeoutError('No ownership transfer; workers left with original controllers')
            time.sleep(.2)
        activation=data.read_json(campaign/'activate.json')
        if (activation['takeover'] != gate['takeover']
                or any(exact_process(r) is not None for r in claim['controllers'].values())):
            raise ValueError('Original dispatch owners have not exited')
        shared.write(campaign/'inherited_history.json',dict(takeover=gate['takeover'],budgets=claim['budgets'],
                     note='All prior failures/charges retained; active starts/deadlines transferred without restart.'))
        old=pinned(gate['donor_gate']);workspace=Path(old['workspace'])
        row=dict(id=PENDING,allowance=43200,cwd=str(workspace),environment={'PYTHONPATH':str(workspace/'scripts')})
        def command(row, job, deadline, metadata):
            spec=m.Spec('ER_GCR_group_fixed','CAB',6)
            shared.write(metadata/'spec.json',spec.identity())
            return [os.sys.executable,'-B',str(workspace/'scripts/step28_continual_comparison_run.py'),
                    '--job',str(Path(old['campaign'])/PENDING),'--spec',str(metadata/'spec.json'),
                    '--gate',gate['donor_gate']['path'],'--deadline',str(deadline)]
        shared.write(campaign/'status.json',dict(status='THREE_HEALTHY_SLOTS_CONTINUE_ORIGINAL_QUEUE'))
        shared.dispatch([row],campaign,ledger,'training',command,existing=attached,retire_failed_slots=True,
            check=lambda:resources(gate,campaign,claim['workers']),
            admit=lambda count:shared.resources(campaign,active_workers=count,
                                                owned_pids=resources(gate,campaign,claim['workers'])))
        tick=time.monotonic()
        try:
            adopt(campaign,gate,ledger)
            records=evaluation.full_gate(campaign,gate)
        finally:
            ledger.charge('preparation',time.monotonic()-tick)
        shared.write(campaign/'training_complete.json',dict(status='ALL_18_NEW_AND_54_REFERENCE_ENDPOINTS_COMPLETE',
                     new_run_ids=[s.run_id for s in m.specifications(new_only=True)],all_runs=len(records),
                     sources=gate['sources'],budget=ledger.snapshot()))
        for action,bucket in [('generate','generation'),('confirmation','confirmation'),('analyze','analysis')]:
            def post(row,job,deadline,metadata):
                return [os.sys.executable,'-B',str(Path(evaluation.__file__)),action,'--campaign',str(campaign),
                        '--gate',str(gate_path),'--deadline',str(deadline)]
            shared.dispatch([dict(id=bucket,allowance=ledger.available(bucket))],campaign,ledger,bucket,post,
                max_workers=1,check=lambda:resources(gate,campaign,{}),
                admit=(lambda count:shared.resources(campaign,active_workers=count)) if action=='confirmation' else None)
        shared.write(campaign/'completion.json',dict(status=defaults.PREDECESSOR_COMPLETE,sources=gate['sources'],budget=ledger.snapshot()))
    except BaseException as error:
        shared.write(campaign/'failure.json',dict(status='STOP_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),budget=ledger.snapshot()))
        raise
    finally:
        for _,log,_ in attached.values():
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gate',type=Path,required=True)
    execute(parser.parse_args().gate.resolve())
