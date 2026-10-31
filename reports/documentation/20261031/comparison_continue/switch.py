"""One explicitly scoped control-ownership transfer; never restart training."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time
import psutil

ROOT=Path('/home/yongpeng/cross-lingual')
P=ROOT/'reports/documentation/20261031/comparison_continue'
OLD=ROOT/'reports/documentation/20261030/comparison_resume'
REPAIR=ROOT/'reports/documentation/20261031/comparison_finish'
H=REPAIR/'defaults'
W=P/'workspace'
sys.path.insert(0,str(W/'scripts'))
import step28_comparison_continue as c

pin=lambda f:{**c.data.record(f,f.parent),'path':str(f)}
now=lambda:datetime.datetime.now().astimezone().isoformat()
read=c.data.read_json
write=c.shared.write


def process_record(process):
    return dict(pid=process.pid,created=process.create_time(),owner=process.username(),
                cwd=process.cwd(),command=process.cmdline())


def prior_process(directory):
    launch=read(directory/'launch.json');gate=read(directory/'formal_gate.json')
    process=psutil.Process(launch['pid'])
    assert (process.username()==launch['owner'] and abs(process.create_time()-launch['created'])<.01
            and process.cwd()==gate['workspace'] and str(directory/'formal_gate.json') in process.cmdline())
    for row in gate['sources']:
        c.data.verify(Path(gate['workspace'])/row['path'],row)
    return process,gate


def launch(directory,workspace,command,gate):
    env=dict(os.environ,PYTHONPATH=str(Path(workspace)/'scripts'),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    with (directory/'console.txt').open('x') as log:
        child=subprocess.Popen(command,cwd=workspace,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=process_record(psutil.Process(child.pid))
    record.update(started=now(),formal_gate=pin(gate))
    write(directory/'launch.json',record)
    return child


cpu=read(P/'cpu05/receipt.json')
assert cpu['success'] and cpu['sources']==c.run.sources()
for row in cpu['orchestration_sources']:
    c.data.verify(W/row['path'],row)
assert not (P/'formal').exists() and not (P/'launch.json').exists()
old,og=prior_process(OLD);repair,rg=prior_process(REPAIR);h,hg=prior_process(H)
assert [old.pid,repair.pid,h.pid]==[2787646,3145836,3146360]
assert not h.children(recursive=True) and read(H/'formal/waiting.json')['status']=='WAITING'
assert read(H/'formal/budget.json')['used']['training']==0
assert all(not (d/'formal/failure.json').exists() for d in (OLD,REPAIR,H))
workers={}
for identity,base,slot in [('CAB_s4_L2R_text_fixed',OLD,1),('CAB_s5_L2R_text_fixed',OLD,2),(c.TARGET,REPAIR,0)]:
    meta=read(base/'formal/dispatch'/identity/'process.json');q=psutil.Process(meta['pid'])
    assert q.username()==meta['owner'] and abs(q.create_time()-meta['created'])<.01 and q.cpu_affinity()==[slot]
    assert str(base/'formal'/identity) in q.cmdline() and str(base/'formal_gate.json') in q.cmdline()
    assert os.getpgid(q.pid)==q.pid
    workers[identity]={**process_record(q),'id':identity,'slot':slot,'job':str(base/'formal'/identity)}
assert {p.pid for p in old.children()}=={workers[k]['pid'] for k in c.ACTIVE-{c.TARGET}}
assert {p.pid for p in repair.children()}=={workers[c.TARGET]['pid']}

fees=sum(read(P/f'cpu{i:02d}/receipt.json')['charged_seconds'] for i in range(1,6))+read(P/'entry_failure.json')['charged_seconds']
assert fees==cpu['cumulative_seconds']-106
write(P/'review_ready.json',dict(status='REVIEW_CLOSED_AND_CPU_PASSED',cpu_receipt=pin(P/'cpu05/receipt.json'),
    sources=cpu['sources'],orchestration_sources=cpu['orchestration_sources'],authorization=pin(P/'scope.zh.txt'),
    disposition='Primary closure: real orphaned CPU workers prove repaired successful slot refills; failed slot retires and healthy slot refills. Prior shared-dispatch evidence retained. No synthetic exit status; actual complete outputs and original full gate required. All failed CPU attempts charged.'))
paused=[];child=None;committed=False
try:
    controllers={name:process_record(q) for name,q in [('donor',old),('repair',repair)]}
    old_h=process_record(h)
    for q in (old,repair,h):
        q.suspend();paused.append(q)
    deadline=time.monotonic()+5
    while any(q.status()!=psutil.STATUS_STOPPED for q in paused):
        if time.monotonic()>deadline:raise TimeoutError('Original controllers did not pause')
        time.sleep(.05)
    budgets={name:read(base/'formal/budget.json') for name,base in [('donor',OLD),('repair',REPAIR)]}
    hb=read(H/'formal/budget.json')
    assert hb['used']['training']==0 and not hb['active'] and not hb['completed']
    claim=dict(observed=now(),controllers=controllers,workers=workers,budgets=budgets)
    write(P/'takeover.json',claim)
    gate=dict(rg)
    for key in ('single_repair','donor','launcher'):
        gate.pop(key,None)
    gate.update(campaign=str(P/'formal'),workspace=str(W),pending_ids=[c.PENDING],
                sources=cpu['sources'],orchestration_sources=cpu['orchestration_sources'],
                review_ready=pin(P/'review_ready.json'),authorization=pin(P/'scope.zh.txt'),takeover=pin(P/'takeover.json'),
                donor_gate=pin(OLD/'formal_gate.json'),repair_gate=pin(REPAIR/'formal_gate.json'),
                additional_used_seconds=dict(cpu=fees,preparation=120,control=60))
    gate['adoption']={s.run_id:dict(completion=str((REPAIR if s.run_id==c.TARGET else OLD)/'formal'/s.run_id/'run/completion.json'),
        source_gate=pin(REPAIR/'formal_gate.json') if s.run_id==c.TARGET else
        og['recovery']['original_gate'] if s.run_id in og['recovery']['reused'] else pin(OLD/'formal_gate.json'))
        for s in c.m.specifications(new_only=True)}
    write(P/'formal_gate.json',gate);c.validate(P/'formal_gate.json')
    child=launch(P,W,[sys.executable,'-B',str(W/'scripts/step28_comparison_continue.py'),'--gate',str(P/'formal_gate.json')],P/'formal_gate.json')
    deadline=time.monotonic()+45
    while not (P/'formal/ready.json').exists():
        if child.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('New control not ready; restore original owners')
        time.sleep(.1)
    assert set(read(P/'formal/ready.json')['pids'])=={r['pid'] for r in workers.values()}
    assert all(c.exact_process(r) is not None for r in workers.values())
    n=P/'defaults';n.mkdir();active=n/'active_defaults_wait.py'
    active.write_bytes((H/'active_defaults_wait.py').read_bytes())
    ng=dict(hg);parts=dict(hg['prior_breakdown_seconds'])
    parts['waiting_since_last_switch']=parts.get('waiting_since_last_switch',0)+hb['used']['control']-hg['prior_used_seconds']['control']
    parts['preparation_conservative_charge']+=120
    assert abs(sum(parts.values())-(hb['used']['control']+120))<1e-6
    ng.update(campaign=str(n/'formal'),listener=str(active),prior_used_seconds=dict(training=0,control=sum(parts.values())),
        prior_breakdown_seconds=parts,authorization=pin(P/'scope.zh.txt'),
        predecessor=dict(campaign=str(P/'formal'),workspace=str(W),gate=pin(P/'formal_gate.json'),launch=pin(P/'launch.json')))
    write(n/'formal_gate.json',ng)
    check="import sys;from pathlib import Path;import step28_defaults_control as c;c.validate_gate(Path(sys.argv[1]),Path(sys.argv[2]))"
    subprocess.run([sys.executable,'-B','-c',check,str(n/'formal_gate.json'),str(active)],cwd=ng['workspace'],check=True,timeout=20,
                   env=dict(os.environ,PYTHONPATH=str(Path(ng['workspace'])/'scripts'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    # Single ownership boundary: parents only. The three worker process groups remain alive.
    committed=True
    for q in (old,repair,h):
        q.kill();q.wait(timeout=10)
    write(P/'formal/activate.json',dict(takeover=gate['takeover'],activated=now()))
    command=[sys.executable,'-B',str(active),'--workspace',ng['workspace'],'--gate',str(n/'formal_gate.json')]
    listener=launch(n,ng['workspace'],command,n/'formal_gate.json')
    write(P/'switch.json',dict(completed=now(),original_controllers=controllers,original_h_s=old_h,
        h_s_old_final_budget=hb,new_h_s_prior=ng['prior_used_seconds'],workers_before=workers,
        workers_after={k:process_record(c.exact_process(r)) for k,r in workers.items()},
        new_controller=read(P/'launch.json'),new_listener=read(n/'launch.json'),cpu_cumulative=cpu['cumulative_seconds']))
    print(json.dumps(dict(status='CONTROL_TRANSFERRED_WITHOUT_TRAINING_RESTART',controller=child.pid,listener=listener.pid,
                         unchanged_workers={k:r['pid'] for k,r in workers.items()},cpu_total=cpu['cumulative_seconds']),ensure_ascii=False),flush=True)
except BaseException as error:
    write(P/'switch_failure.json',dict(error=repr(error),committed=committed,observed=now()))
    if not committed:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=10)
        for q in paused:
            if q.is_running():q.resume()
    raise
