"""Single-core synthetic qualification of the conditional start and first-update retirement."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_benchmark_wait as w
import step28_history_benchmark as b
from step30_evolution_io import read, record, write


def require_error(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError("Expected "+kind.__name__)


def ident(proc):
    p = psutil.Process(proc.pid)
    return dict(pid=p.pid, owner=p.username(), cwd=p.cwd(), command=p.cmdline(), **w.kernel_identity(p.pid))


def qualify(root):
    en, ch = root/"vendorlink_market_training/execution", root/"h_history_benchmark"
    ws, campaign = ch/"workspace", ch/"campaign"
    (ws/"scripts").mkdir(parents=True)
    script = ws/"scripts/once_wait.py"
    script.write_text("# disposable synthetic listener")
    gate = ch/"formal_gate.json"
    write(gate, {"synthetic":True})
    eroot, ews, egate = en/"campaign", en/"workspace", en/"formal_gate.json"
    ews.mkdir(parents=True)
    tasks = [{"id":str(i)} for i in range(12)]
    write(en/"tasks.json",tasks)
    caps=dict(training=100,gpu=100,auxiliary=100)
    write(egate,dict(workspace=str(ews),campaign=str(eroot),protocol="vendorlink-fit12-market-adapters",
                    tasks=record(en/"tasks.json"),caps=caps))
    cmd = [sys.executable,"-c","import time;time.sleep(45)","campaign","--gate",str(egate)]
    started = subprocess.Popen(cmd,cwd=ews)
    try:
        write(en/"launch.json",dict(gate=record(egate),controller=ident(started)))
        g = dict(listener=record(script),predecessor=record(en/"launch.json"),workspace=str(ws),
                 campaign=str(campaign),other_gpu_commitments=[],runs=b.plan(), budget_seconds=dict(control=3600),
                 prior_used_seconds=dict(control=78),limits=dict(free_ram_bytes=1,free_gpu_bytes=100,
                 free_reserve_bytes=1,save_reserve_bytes=1,disk_bytes=2**28))
        assert w.predecessor(g)["status"]=="WAIT_VENDORLINK"
        write(eroot/"training_complete.json",dict(tasks=tasks,winners={r["id"]:{} for r in tasks}))
        assert w.predecessor(g)["status"]=="WAIT_VENDORLINK"
        write(eroot/"scoring_complete.json",dict(tasks=tasks,completions=[
              dict(path=str(eroot/"scoring"/r["id"]/"result/completion.json")) for r in tasks]))
        assert w.predecessor(g)["status"]=="WAIT_VENDORLINK"
        write(eroot/"dispatch.json",dict(status="COMPLETE",active={}))
        write(eroot/"completion.json",dict(status="COMPLETE",dev_opening="whole_batch_once",confirm_open=False,
              decision=dict(path=str(eroot/"analysis/result/decision.json"))))
        assert w.predecessor(g)["status"]=="WAIT_VENDORLINK_RELEASE"
        peer=subprocess.Popen(cmd,cwd=ews)
        try:
            started.terminate(); started.wait(5)
            assert w.predecessor(g)["related"]==[peer.pid]
        finally:
            peer.terminate(); peer.wait(5)
        ledger=dict(caps=caps,used_seconds={k:7 for k in caps},operation_ids=[],events=[
               dict(operation=op,bucket="auxiliary",success=True) for op in ("analysis","dispatch_monitor")])
        write(eroot/"budget.json",ledger)
        require_error(ValueError,lambda:w.predecessor(g))
        ledger["events"].append(dict(operation="controller_setup_or_handoff",bucket="auxiliary",success=True))
        ledger["used_seconds"]["auxiliary"]=101
        write(eroot/"budget.json",ledger)
        require_error(ValueError,lambda:w.predecessor(g))
        ledger["used_seconds"]["auxiliary"]=8
        write(eroot/"budget.json",ledger)
        assert w.predecessor(g)["ready"]
        write(eroot/"failure.json",{"error":"handwritten failure"})
        require_error(RuntimeError,lambda:w.predecessor(g))
        (eroot/"failure.json").unlink()
    finally:
        if started.poll() is None:
            started.terminate(); started.wait(5)
    require_error(FileNotFoundError,lambda:w.verify_handoff(g,gate))
    mem = {"free":99,"processes":{}}
    class Memory:
        def snapshot(self): return dict(mem)
        def close(self): pass
    import step28_formal_queue as hardware
    claimed=[]
    identity=dict(w.process_identity(),command=[sys.executable,str(script),"--gate",str(gate)])
    def launch(allowed, path):
        claimed.append(w.verify_handoff(allowed,path))
    with (patch.object(w,"process_identity",return_value=identity),patch.object(hardware,"GpuMemory",Memory),
          patch.object(w,"training_inputs",return_value=[{"linked":True}]),
          patch.object(b,"read_gate",return_value=g),patch.object(b,"start_campaign",launch)):
        def free(_):
            assert read(ch/"wait/status.json")["status"]=="WAIT_RESOURCES" and not claimed
            mem["free"]=100
        w.wait_then_start(g,gate,time.monotonic(),sleep=free)
        assert len(claimed)==1 and read(ch/"wait/ready.json")["resources"]["gpu_free"]==100
        require_error(FileExistsError,lambda:w.wait_then_start(g,gate,time.monotonic()))
    require_error(ValueError,lambda:w.verify_handoff(g,gate))
    alt=ch/"changed";alt.mkdir()
    other=alt/"formal_gate.json";write(other,dict(synthetic=True))
    mem["free"]=99
    with patch.object(hardware,"GpuMemory",Memory), patch.object(b,"start_campaign",launch):
        def change(_):
            other.write_text("{}")
            mem["free"]=100
        require_error(ValueError,lambda:w.wait_then_start(dict(g,campaign=str(alt/"campaign")),other,time.monotonic(),sleep=change))
    assert (alt/"wait/failure.json").is_file() and not (alt/"wait/ready.json").exists() and len(claimed)==1

    row=next(r for r in g["runs"] if r["name"]==b.forever.NAME)
    target=campaign/"training"/row["id"]/"attempt-1"
    target.mkdir(parents=True)
    write(target/"worker_claim.json",dict(id=row["id"]))
    ran=b.native(row["name"],row["order"],row["seed"]).run_id
    child_code="""
import importlib,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,sys.argv[3]);import step28_forever_formal as native
from types import SimpleNamespace
x=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([x],lr=.1)
def emit(n,identity):
    print(json.dumps(dict(event='updates',run_id=identity,stage=1,completed=n,counts=dict(current=n))),flush=True)
emit(0,sys.argv[2]);emit(1,'different-run');emit(1,None)
d=Path(sys.argv[1]);(d/'waiting').touch()
while not (d/'go').exists():time.sleep(.02)
(x*x).backward();opt.step();assert x.item()<1
native.report_current_update(SimpleNamespace(run_id=sys.argv[2]),1,dict(current=1,replay=0,fixture=0),'current')
(d/'updated').touch();time.sleep(35)
"""
    def wait_file(p):
        end=time.monotonic()+20
        while not p.is_file() and time.monotonic()<end:time.sleep(.025)
        assert p.is_file(),p
    with (target/"console.txt").open("w") as out:
        p=subprocess.Popen([sys.executable,"-B","-c",child_code,str(target),ran,str(b.SCRIPTS)],cwd=ws,stdout=out)
        try:
            wait_file(target/"waiting")
            active={0:dict(job=dict(action="worker",target=target,task_id=row["id"],attempt=1),identity=ident(p))}
            write(target/"process.json",dict(active[0]["identity"],task_id=row["id"],attempt=1))
            write(ch/"wait/controller.json",dict(identity=w.process_identity(),gate=record(gate)))
            w.retire(g,campaign,active);assert script.is_file() and not (campaign/"waiter_deleted.json").exists()
            (target/"go").touch();wait_file(target/"updated")
            with patch.object(Path,"unlink",side_effect=InterruptedError("after intent")):
                require_error(InterruptedError,lambda:w.retire(g,campaign,active))
            assert script.is_file() and (ch/"wait/first_update.json").is_file()
            saved=script.with_suffix(".original")
            script.rename(saved);script.write_bytes(saved.read_bytes())
            require_error(ValueError,lambda:w.retire(g,campaign,{}));assert script.is_file()
            script.unlink();script.symlink_to(saved)
            require_error(ValueError,lambda:w.retire(g,campaign,{}));assert script.is_symlink()
            script.unlink();saved.rename(script)
            with patch.object(w,"write",side_effect=OSError("after unlink")):
                w.retire(g,campaign,active)
            assert not script.exists() and not (campaign/"waiter_deleted.json").exists()
            w.retire(g,campaign,{});w.retire(g,campaign,{})
            rec=read(campaign/"waiter_deleted.json")
            assert rec["removed"] and rec["source"]==g["listener"] and rec["first_observed_updates"]==1
            assert rec["disposition"]=="already_absent_on_reconciliation" and rec["native_run"]==ran
            assert rec["worker"]["pid"]==p.pid and not script.exists() and p.poll() is None
        finally:
            p.terminate();p.wait(5)
    qualify_resource_drop(root,g,row,Memory,mem)
    op_file=b.SCRIPTS.parent.parent/"start/operations.py"
    module=importlib.util.spec_from_file_location("benchmark_operations",op_file)
    deployment=importlib.util.module_from_spec(module);module.loader.exec_module(deployment)
    decision=dict(experiment=b.EXPERIMENT,review_closed=True,manifest_sha256="m",
                  qualification_sha256="q",fix_qualification_sha256="f")
    def accepts(body):
        deployment.accepted_review(body,"m","q","f")
    def doc(values):
        return "<!-- "+json.dumps(values)+" -->\n# Primary review"
    accepts(doc(decision))
    for bad in (False,"false","true"):
        require_error(ValueError,lambda:accepts(doc(dict(decision,review_closed=bad))))
    require_error(ValueError,lambda:accepts("START_REVIEW_CLOSED=false"))
    require_error(ValueError,lambda:accepts(doc(dict(decision,manifest_sha256="previous"))))
    return dict(predecessor_running_and_partial_held=True,terminal_failure_held=True,release_of_related_pid_required=True,
                final_accounting_required=True,over_budget_predecessor_rejected=True,
                resource_99_held_100_admitted=True,unique_launch=len(claimed),direct_launch_rejected=True,
                changed_gate_held=True,pid_verified=True,native_format_real_cpu_updates=1,wrong_run_and_zero_held=True,
                deletion_after_update=True,forever_native_first_update_identified=True,
                wrong_source_and_symlink_preserved=True,journal_reconciled_without_worker=True,
                negative_or_stale_review_rejected=True,one_core_resource_drop_and_readmission=True,
                charged_monitor_seconds=claimed[0])


def qualify_resource_drop(root, previous, row, memory_class, mem):
    import step28_formal_queue as hardware
    ch=root/"drop/benchmark"
    ws=ch/"workspace"; campaign=ch/"campaign"; gate=ch/"formal_gate.json"
    (ws/"scripts").mkdir(parents=True)
    script=ws/"scripts/once_wait.py";script.write_text("# synthetic")
    write(gate,{"synthetic":True})
    worker=ws/"tiny_worker.py"
    worker.write_text("""import json,sys,time
from pathlib import Path
def arg(k):return sys.argv[sys.argv.index(k)+1]
root=Path(arg('--gate')).parent
target=root/'campaign/training'/arg('--id')/('attempt-'+arg('--attempt'))
(target/'worker_claim.json').write_text(json.dumps(dict(id=arg('--id'))))
(target/'run').mkdir();(target/'run/completion.json').write_text('{}')
time.sleep(.3)
""")
    g=dict(previous,runs=[row],workspace=str(ws),campaign=str(campaign),listener=record(script),workers=3,
           experiment=b.EXPERIMENT,sources=[],authorization=record(gate),analysis_plan=record(gate),
           cpu_test_seconds=0,seconds_per_method={row["name"]:120},post_seconds={},
           budget_seconds=dict(training=800,confirmation=900,generation=900,analysis=900,control=900),
           prior_used_seconds=dict(training=0,confirmation=0,generation=0,analysis=0,control=0),
           limits=dict(previous["limits"],rss_bytes=4*2**30,worker_rss_bytes=2**30,process_gpu_bytes=200))
    identity=dict(w.process_identity(),command=[sys.executable,str(script),"--gate",str(gate)])
    real_sleep=time.sleep
    holds=[]
    def sleep(interval):
        if interval==30:
            state=read(campaign/"dispatch.json")
            assert state["status"]=="WAITING_RESOURCES" and not state["active"] and state["pending"]
            attempt=campaign/"training"/row["id"]/"attempt-1"
            assert read(attempt/"admission.json")["ready"] is False
            assert not (attempt/"process.json").exists() and not (attempt/"console.txt").exists()
            holds.append(True);mem["free"]=100
        real_sleep(.03)
    class FallingMemory(memory_class):
        def __init__(self):
            if campaign.exists():
                mem["free"]=99
    mem["free"]=100
    with (patch.object(w,"process_identity",return_value=identity),patch.object(hardware,"GpuMemory",FallingMemory),
          patch.object(b,"read_gate",return_value=g),patch.object(w,"training_inputs",return_value=[{"linked":True}]),
          patch.object(b,"plan",return_value=[row]),patch.object(b,"__file__",str(worker)),
          patch.object(b,"POST_STATUS",{}),patch.object(b,"validate_training",return_value=True),
          patch.object(time,"sleep",side_effect=sleep)):
        w.wait_then_start(g,gate,time.monotonic())
    won=read(campaign/"training_complete.json")["winners"]
    assert holds==[True] and set(won)=={row["id"]} and won[row["id"]]["attempt"]==1
    assert not (campaign/"training"/row["id"]/"attempt-2").exists()
    assert len([e for e in read(campaign/"budget.json")["events"] if e["bucket"]=="training"])==1
    assert read(campaign/"completion.json")["status"]=="ALL_FORMAL_ACTIONS_COMPLETE"
    assert script.exists() and not (campaign/"waiter_deleted.json").exists()


if __name__=="__main__":
    assert sys.platform=="linux" and Path(sys.executable).resolve()==Path(w.PYTHON).resolve()
    assert len(os.sched_getaffinity(0))==1 and os.environ["CUDA_VISIBLE_DEVICES"]==""
    with tempfile.TemporaryDirectory() as d:
        result=qualify(Path(d))
    print(json.dumps(dict(passed=True,evidence=result,official_data_reads=0,official_training_tasks=0,gpu_kernels=0)))
