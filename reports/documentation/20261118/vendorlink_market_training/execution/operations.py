"""Deployment, bounded qualification and observation for this single FIT12 campaign."""
import argparse
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
WS = ROOT/'workspace'
SCRIPTS = WS/'scripts'
EVIDENCE = ROOT/'evidence'
CAMPAIGN = ROOT/'campaign'
GATE = ROOT/'formal_gate.json'
sys.path.insert(0, str(SCRIPTS))
from step28_formal_queue import GpuMemory
from step30_evolution_control import kernel_identity
from step30_evolution_io import PYTHON, directory_bytes, read, record, verify, write

GIB = 2**30
PREVIOUS = ROOT.parents[2]/'20261113/vendorlink_beta_quarter'
PREVIOUS_GATE = PREVIOUS/'gate_after_postprocess_resume.json'
PRIOR_HASH = '7cd1670eaecc47aff18d2c9353c6903b18c80c016aca11822319b498d9ef166c'
PUBLIC_BETA = '0fcc9250927d09607b790700904f222e8ba42e301852415e0ff05b8f34ed9d49'
NAMES = ['step35_market_'+s for s in ('accounting','adapters','worker','queue')]
NAMES += ['step35_vendorlink_account_'+s for s in ('control','evaluate','io','methods','policy','run')]
NAMES += ['step29_vendorlink_'+s for s in ('data','methods','run')]
NAMES += ['step30_evolution_'+s for s in ('control','io','policy')] + ['step28_formal_queue']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def sources():
    return [record(SCRIPTS/(name+'.py')) for name in sorted(NAMES)]


def previous():
    if record(PREVIOUS_GATE)['sha256'] != PRIOR_HASH:
        raise ValueError('Previous signed FIT12 gate identity changed')
    return read(PREVIOUS_GATE)


def process_info(pid):
    p = psutil.Process(pid)
    return dict(pid=pid, owner=p.username(), cwd=p.cwd(), command=p.cmdline(), **kernel_identity(pid))


def live(row):
    try:
        actual=process_info(row['pid'])
        return all(actual[k] == row[k] for k in ('owner','cwd','command','start_ticks','boot_id'))
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
        return False


def resources():
    nv = GpuMemory()
    try:
        gpu = nv.snapshot()
    finally:
        nv.close()
    return dict(gpu_free_bytes=gpu['free'], gpu_compute_pids=sorted(gpu['processes']),
                cpu_cores=len(psutil.Process().cpu_affinity()), ram_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(ROOT).free, experiment_bytes=directory_bytes(ROOT))


def receipts(device):
    return [read(p) for p in sorted(EVIDENCE.glob(device+'[0-9][0-9].json'))]


def qualify(device):
    if GATE.exists() or CAMPAIGN.exists():
        raise ValueError('Post-gate qualification is closed')
    EVIDENCE.mkdir(exist_ok=True)
    prefix = 'cpu' if device == 'cpu' else 'gpu'
    old = receipts(prefix)
    if any(r['passed'] for r in old):
        raise ValueError('Existing passing qualification already covers this entry')
    if prefix == 'gpu' and not any(r['passed'] for r in receipts('cpu')):
        raise ValueError('Synthetic CPU qualification must finish first')
    allowance = (1800-7 if device == 'cpu' else 3600) - sum(r['charged_seconds'] for r in old)
    old_steps = sum(r.get('optimizer_intents',0) for r in old)
    # GPU01 used a pre-forward intent counter and stopped in its very first backward.
    # A single signed correction counts actual optimizer entries; all six seconds remain.
    if prefix=='gpu' and old:
        correction=read(EVIDENCE/'gpu01_disposition.json')
        original=old[0]
        if (record(EVIDENCE/'gpu01.json')['sha256']!=correction['original_sha256']
                or original['label']!='gpu01' or original['optimizer_intents']!=1 or original['passed']
                or 'LookupError: <ContextVar' not in original['synthetic_error_tail']
                or correction['actual_optimizer_entries']!=0):
            raise ValueError('The GPU01 pre-optimizer failure correction does not apply')
        old_steps-=1
    if allowance <= 5 or (prefix == 'gpu' and old_steps >= 12):
        raise ValueError('Qualification allowance exhausted')
    initial = resources()
    if (initial['disk_free_bytes'] < 64*GIB or initial['ram_available_bytes'] < 20*GIB or
            initial['experiment_bytes']+2*GIB > 32*GIB or
            (device == 'cuda' and (initial['gpu_compute_pids'] or initial['gpu_free_bytes'] < 5*GIB))):
        raise ValueError('Shared machine unavailable for native qualification')
    label = f'{prefix}{len(old)+1:02d}'
    dest = EVIDENCE/label
    dest.mkdir(exist_ok=False)
    affinity = psutil.Process().cpu_affinity()[:1 if device=='cpu' else 2]
    cmd = ['taskset','-c',','.join(map(str,affinity)),PYTHON,'-B','-u',str(ROOT/'qualify.py'),
           '--device',device,'--output',str(dest),'--used-steps',str(old_steps)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device=='cpu' else '0', TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS=str(len(affinity)), MKL_NUM_THREADS=str(len(affinity)), OPENBLAS_NUM_THREADS='1')
    started = time.monotonic()
    begin = now()
    nv = GpuMemory() if device=='cuda' else None
    peak, rss, billed_monitor, stop = 0,0,0.,None
    try:
        with (dest/'console.log').open('xb') as log:
            p = subprocess.Popen(cmd, cwd=WS, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            try:
                while p.poll() is None:
                    tick = time.monotonic()
                    try:
                        rss = max(rss,psutil.Process(p.pid).memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    if nv:
                        peak = max(peak,nv.snapshot()['processes'].get(p.pid,0))
                    if tick-started >= allowance-4 or rss>16*GIB or shutil.disk_usage(ROOT).free<32*GIB:
                        stop = 'Qualification time, RSS or disk reserve'
                        break
                    billed_monitor += time.monotonic()-tick
                    time.sleep(.15)
            finally:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid,signal.SIGKILL)
                        p.wait()
    finally:
        if nv:
            nv.close()
    elapsed = time.monotonic()-started
    detail = read(dest/'result.json') if (dest/'result.json').exists() else {}
    intents = read(dest/'step_intents.json')['count'] if (dest/'step_intents.json').exists() else 0
    passed = bool(p.returncode == 0 and detail.get('passed') and not stop and
                  (device=='cpu' or 0 < peak and intents+old_steps<=12))
    obj = dict(label=label,device=device,started=begin,finished=now(),elapsed_seconds=elapsed,
               charged_seconds=math.ceil(elapsed),monitor_seconds=math.ceil(billed_monitor),exit_code=p.returncode,
               pid=p.pid,peak_pid_gpu_bytes=peak,peak_rss_bytes=rss,optimizer_intents=intents,passed=passed,
               reason=stop,affinity=affinity,detail=detail,production_sources=sources(),
               qualification_source=record(ROOT/'qualify.py'),operations_source=record(Path(__file__)))
    if not passed:
        obj['synthetic_error_tail']=(dest/'console.log').read_text(errors='replace')[-3500:]
    write(EVIDENCE/(label+'.json'),obj)
    print(label,passed,elapsed,peak,obj.get('synthetic_error_tail',''),flush=True)
    if not passed:
        raise RuntimeError('Qualification failed; original evidence and cost retained')


def prepare(revision):
    from step35_market_queue import CAPS, PROTOCOL
    from step35_market_worker import market_tasks
    from step35_vendorlink_account_policy import LIMITS
    if GATE.exists() or CAMPAIGN.exists() or not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
        raise ValueError('Fresh gate and exact Windows source commit required')
    decision = read(ROOT/'review/decision.json')
    science = sources()
    if (decision['blocking_findings'] or decision['closed'] is not True or
            decision['source_sha256'] != {Path(r['path']).name:r['sha256'] for r in science}):
        raise ValueError('Independent disposition must bind all production sources')
    for item in decision['review_evidence']:
        verify(item)
    cpu,gpu=receipts('cpu'),receipts('gpu')
    if not cpu or not gpu or not cpu[-1]['passed'] or not gpu[-1]['passed']:
        raise ValueError('Missing native qualifications')
    signed = {Path(r['path']).name:r['sha256'] for r in science}
    for r in (cpu[-1],gpu[-1]):
        tested={Path(x['path']).name:x['sha256'] for x in r['production_sources']}
        if tested!=signed and decision.get('qualified_source_changes',{}).get(r['label']) != r['production_sources']:
            raise ValueError('Source changes since qualification require explicit independent disposition')
    prev=previous()
    oldbeta=record(PREVIOUS/'campaign/M/analysis/result/results.json')
    if oldbeta['sha256'] != PUBLIC_BETA:
        raise ValueError('Historical open beta result identity changed')
    verify(prev['prior_M']['results'])
    verify(prev['manifest'])
    for item in prev['model_binding']['files']:
        verify(item)
    reserved=max(5,math.ceil(1.2*gpu[-1]['peak_pid_gpu_bytes']/GIB))
    tasks=market_tasks()
    write(ROOT/'tasks.json',tasks)
    write(ROOT/'final_sources.json', dict(git=revision, sources=science, deployment=str(WS)))
    repaired=sorted(EVIDENCE.glob('repair[0-9][0-9].json'))
    if not repaired or not read(repaired[-1])['passed'] or read(repaired[-1])['production_sources']!=science:
        raise ValueError('Missing focused qualification for the final production files')
    cpu_sec=7+sum(x['charged_seconds'] for x in cpu)+sum(read(p)['charged_seconds'] for p in repaired)
    prior_events=sorted(EVIDENCE.glob('operation_*.json'))
    aux=cpu_sec+sum(x['monitor_seconds'] for x in cpu+gpu)+sum(read(p)['charged_seconds'] for p in prior_events)
    bindings=[record(EVIDENCE/(r['label']+'.json')) for r in cpu+gpu]+[record(p) for p in repaired]
    limits=dict(LIMITS,workers=3,vendor_reserve_bytes=29*GIB,free_gpu_bytes=reserved*GIB)
    write(GATE,dict(protocol=PROTOCOL,training_authorized=True,review_closed=True,source_git=revision,
          workspace=str(WS),campaign=str(CAMPAIGN),experiment_root=str(ROOT),caps=CAPS,limits=limits,
          sources=science,source_list=record(ROOT/'final_sources.json'),tasks=record(ROOT/'tasks.json'),
          worker_entry=record(SCRIPTS/'step35_market_worker.py'),authorization=record(ROOT/'authorization.zh.md'),
          scope=record(ROOT/'proposal.zh.md'),review=record(ROOT/'review/primary_disposition.zh.md'),
          review_decision=record(ROOT/'review/decision.json'),qualification=bindings+[record(ROOT/'oom_qualification.json')],
          manifest=prev['manifest'],model_binding=prev['model_binding'],other_gpu_commitments=[],
          allow_observed_foreign_gpu_processes=True,cpu_test_seconds=cpu_sec,
          prior_operation_ids=[p.name for p in prior_events],operations_entry=record(Path(__file__)),
          prior_seconds=dict(training=0,auxiliary=aux,gpu=sum(x['charged_seconds'] for x in gpu)),
          historical=dict(original_H_ER=dict(binding=prev['prior_M']['results'],method='H_ER'),
                          original_beta_Full=dict(binding=oldbeta,method='H_full')),
          prepared=now(),confirmation_open=False))
    print('FORMAL_GATE',revision,reserved)


def launch():
    g=read(GATE)
    if CAMPAIGN.exists() or (ROOT/'launch.json').exists():
        raise ValueError('Existing formal campaign or launch receipt')
    for item in [g['authorization'],g['scope'],g['review'],g['review_decision'],g['source_list'],g['tasks'],g['operations_entry'],
                 *g['sources'],*g['qualification']]:
        verify(item)
    # This exclusive claim prevents concurrent launchers creating two controllers.
    with (ROOT/'launch.claim').open('x') as claim:
        claim.write(now())
    cmd=[PYTHON,'-B','-u',str(SCRIPTS/'step35_market_queue.py'),'campaign','--gate',str(GATE)]
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    with (ROOT/'controller.log').open('xb') as f:
        p=subprocess.Popen(cmd,cwd=WS,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                           env=env,start_new_session=True)
    row=process_info(p.pid)
    write(ROOT/'launch.json',dict(started=now(),gate=record(GATE),controller=row))
    print('LAUNCH',p.pid,flush=True)


def observe():
    start=read(ROOT/'launch.json')
    ctl=start['controller']
    dis=read(CAMPAIGN/'dispatch.json') if (CAMPAIGN/'dispatch.json').exists() else {}
    active=[]
    for slot,row in dis.get('active',{}).items():
        valid=live(row)
        cmd=row['command']
        target=Path(cmd[cmd.index('--target')+1])
        if not target.is_relative_to(CAMPAIGN):
            raise ValueError('Wrong campaign worker target')
        w=target/'run/workload.json'
        uptime=float(Path('/proc/uptime').read_text().split()[0])
        elapsed=max(0,uptime-row['start_ticks']/os.sysconf('SC_CLK_TCK')) if valid else None
        active.append(dict(slot=slot,identity=row,identity_matches=valid,open_lifecycle_seconds=elapsed,
                           task=cmd[cmd.index('--task')+1] if '--task' in cmd else None,
                           action=cmd[cmd.index('--action')+1],workload=read(w) if w.exists() else {}))
    budget=CAMPAIGN/'budget.json'
    obj=dict(observed=now(),controller=ctl,controller_identity_matches=live(ctl),active=active,
        formal_trajectories=12, successful_training=len(list(CAMPAIGN.glob('training/*/attempt-*/run/completion.json'))),
        failures=len(list(CAMPAIGN.glob('training/*/attempt-*/worker_failure.json'))),
        training_complete=(CAMPAIGN/'training_complete.json').exists(),scores_complete=(CAMPAIGN/'scoring_complete.json').exists(),
        whole_batch_complete=(CAMPAIGN/'completion.json').exists(),failed=(CAMPAIGN/'failure.json').exists(),
        dispatch_status=dis.get('status'),settled=read(budget)['used_seconds'] if budget.exists() else {},
        resources=resources(),formal_gate=record(GATE))
    write(ROOT/'status_observation.json',obj)
    print(obj['observed'],obj['controller_identity_matches'],[(x['task'],x['identity_matches'],x['workload']) for x in active],
          'failure',obj['failed'],flush=True)
    if not obj['controller_identity_matches'] and not obj['whole_batch_complete']:
        print((ROOT/'controller.log').read_text(errors='replace')[-3000:])


def qualify_repairs():
    history=sorted(EVIDENCE.glob('repair[0-9][0-9].json'))
    if GATE.exists() or any(read(p)['passed'] for p in history):
        raise ValueError('No repeated focused qualification')
    remaining=1800-7-sum(r['charged_seconds'] for r in receipts('cpu'))-sum(read(p)['charged_seconds'] for p in history)
    if remaining<5:
        raise TimeoutError('CPU qualification allowance exhausted')
    label=f'repair{len(history)+1:02d}'
    target=EVIDENCE/label
    target.mkdir(exist_ok=False)
    core=psutil.Process().cpu_affinity()[0]
    started=time.monotonic()
    try:
        result=subprocess.run(['taskset','-c',str(core),PYTHON,'-B',str(ROOT/'review/check_repairs.py'),str(target)],
             cwd=WS,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=remaining-2)
        code,log=result.returncode,result.stdout.decode(errors='replace')[-3000:]
    except subprocess.TimeoutExpired:
        code,log=-1,'Single-core timeout'
    seconds=time.monotonic()-started
    detail=read(target/'result.json') if (target/'result.json').exists() else {}
    passed=code==0 and detail.get('passed') is True
    write(EVIDENCE/(label+'.json'),dict(label=label,passed=passed,charged_seconds=math.ceil(seconds),
          elapsed_seconds=seconds,finished=now(),cpu_core=core,exit_code=code,output=log,detail=detail,
          production_sources=sources(),operations_source=record(Path(__file__)),check_source=record(ROOT/'review/check_repairs.py')))
    print(label,passed,seconds,log)
    if not passed:
        raise RuntimeError('Focused qualification failed')


def settle_closed():
    if (CAMPAIGN/'budget.json').exists() and (ROOT/'launch.json').exists() and not live(read(ROOT/'launch.json')['controller']):
        from step35_market_accounting import MarketLedger
        MarketLedger.settle_closed(CAMPAIGN)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('op',choices=('preflight','cpu','gpu','prepare','launch','observe','account'))
    p.add_argument('--revision','--git',dest='revision')
    p.add_argument('source_revision',nargs='?')
    args=p.parse_args()
    if sys.platform!='linux' or Path(sys.executable).resolve()!=Path(PYTHON).resolve():
        raise RuntimeError('Use the existing Linux py310')
    EVIDENCE.mkdir(exist_ok=True)
    t=time.monotonic()
    if args.op=='cpu' and any(r['passed'] for r in receipts('cpu')) and (ROOT/'review/check_repairs.py').is_file():
        qualify_repairs()
    elif args.op in ('cpu','gpu'):
        qualify('cpu' if args.op=='cpu' else 'cuda')
    else:
        from step35_market_accounting import billed_operation
        with billed_operation(ROOT,args.op,settle_closed):
            if args.op=='prepare':
                prepare(args.revision or args.source_revision)
            elif args.op=='launch':
                launch()
            elif args.op=='observe':
                observe()
            elif args.op=='account' and (CAMPAIGN/'budget.json').exists():
                print(read(CAMPAIGN/'budget.json')['used_seconds'])
            else:
                out=dict(observed=now(),resources=resources(),campaign_exists=CAMPAIGN.exists(),gate_exists=GATE.exists(),
                         model_source=previous()['model_binding']['directory'])
                write(ROOT/'preflight.json',out)
                print(out)
