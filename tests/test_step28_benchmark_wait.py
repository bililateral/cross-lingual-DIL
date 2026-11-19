"""Single-core synthetic qualification of the conditional start and first-update retirement."""
import json
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
    write(egate,dict(workspace=str(ews),campaign=str(eroot),protocol="vendorlink-fit12-market-adapters",tasks=record(en/"tasks.json")))
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

    row=g["runs"][0]
    target=campaign/"training"/row["id"]/"attempt-1"
    target.mkdir(parents=True)
    write(target/"worker_claim.json",dict(id=row["id"]))
    ran=b.native(row["name"],row["order"],row["seed"]).run_id
    child_code="""
import json,sys,time
from pathlib import Path
import torch
x=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([x],lr=.1)
def event(n,identity):print(json.dumps(dict(event='updates',run_id=identity,stage=1,completed=n)),flush=True)
event(0,sys.argv[2]);event(1,'different-run')
d=Path(sys.argv[1]);(d/'waiting').touch()
while not (d/'go').exists():time.sleep(.02)
(x*x).backward();opt.step();assert x.item()<1
event(1,sys.argv[2]);(d/'updated').touch();time.sleep(35)
"""
    def wait_file(p):
        end=time.monotonic()+20
        while not p.is_file() and time.monotonic()<end:time.sleep(.025)
        assert p.is_file(),p
    with (target/"console.txt").open("w") as out:
        p=subprocess.Popen([sys.executable,"-B","-c",child_code,str(target),ran],cwd=ws,stdout=out)
        try:
            wait_file(target/"waiting")
            active={0:dict(job=dict(action="worker",target=target,task_id=row["id"],attempt=1),identity=ident(p))}
            w.retire(g,campaign,active);assert script.is_file() and not (campaign/"waiter_deleted.json").exists()
            (target/"go").touch();wait_file(target/"updated")
            w.retire(g,campaign,active);w.retire(g,campaign,active)
            rec=read(campaign/"waiter_deleted.json")
            assert rec["removed"] and rec["source"]==g["listener"] and rec["first_observed_updates"]==1
            assert rec["worker"]["pid"]==p.pid and not script.exists() and p.poll() is None
        finally:
            p.terminate();p.wait(5)
    return dict(predecessor_running_and_partial_held=True,terminal_failure_held=True,release_of_related_pid_required=True,
                resource_99_held_100_admitted=True,unique_launch=len(claimed),direct_launch_rejected=True,
                changed_gate_held=True,pid_verified=True,native_format_real_cpu_updates=1,wrong_run_and_zero_held=True,
                deletion_after_update=True,charged_monitor_seconds=claimed[0])


if __name__=="__main__":
    assert sys.platform=="linux" and Path(sys.executable).resolve()==Path(w.PYTHON).resolve()
    assert len(os.sched_getaffinity(0))==1 and os.environ["CUDA_VISIBLE_DEVICES"]==""
    with tempfile.TemporaryDirectory() as d:
        result=qualify(Path(d))
    print(json.dumps(dict(passed=True,evidence=result,official_data_reads=0,official_training_tasks=0,gpu_kernels=0)))
