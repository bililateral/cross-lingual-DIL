"""Chinese dispatch with real Linux children, injected OOM and handwritten NVML."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]/"scripts"))
TASKS = ("ABC_s4_H_trial3", "ABC_s4_ER0.1_ref", "ABC_s4_LOGIT0.1_ref",
         "ABC_s4_L2R_text_fixed", "ABC_s4_ER_GCR_group_fixed")
PROTOCOL = "chinese_trial3_six"


def read(path):
    return json.loads(path.read_bytes())


def wait(condition):
    end = time.monotonic()+12
    while time.monotonic() < end:
        value = condition()
        if value:
            return value
        time.sleep(.025)
    raise TimeoutError(f"Timed out at fixture line {condition.__code__.co_firstlineno}")


def child(args):
    from step30_evolution_io import write
    root, name = args.gate.parent, args.id
    config = read(args.gate)
    case = config["case"]
    if args.action != "worker":
        actions = list(config["post_status"])
        previous = actions.index(name)-1
        assert (root/"campaign"/"training_complete.json").is_file()
        if previous >= 0:
            assert (root/"campaign"/actions[previous]/"result.json").is_file()
        write(root/"campaign"/name/"result.json", dict(status=config["post_status"][name], synthetic=True))
        return 0
    target = root/"campaign"/"training"/name/f"attempt-{args.attempt}"
    wait(lambda: (target/"process.json").exists())
    process = read(target/"process.json")
    assert process["pid"] == os.getpid() and process["task_id"] == name and process["attempt"] == args.attempt
    assert process["command"][-2:] == ["--attempt", str(args.attempt)]
    (target/"running").touch()
    if name == TASKS[0] and args.attempt == 1:
        wait(lambda: (root/"fail").exists())
        if case != "interrupt":
            write(target/"worker_failure.json", dict(kind="OTHER" if case == "other" else "CUDA_OOM"))
            return 1
    elif name == TASKS[0]:
        wait(lambda: (root/"retry_done").exists())
        if case == "repeat":
            write(target/"worker_failure.json", dict(kind="CUDA_OOM"))
            return 1
    elif name == TASKS[1]:
        wait(lambda: (root/"first_healthy_done").exists())
    else:
        wait(lambda: (root/"peers_done").exists())
    write(target/"run"/"completion.json", dict(task=name, seed=4, native_attempt=args.attempt, synthetic=True))
    return 0


def controller(gate):
    import psutil
    import step28_formal_queue as hardware
    import step28_history_benchmark_dispatch as batch
    import step30_evolution_control as loop
    from step30_evolution_io import record, write
    base, case = gate.parent, read(gate)["case"]
    def terminate(signum, frame):
        raise KeyboardInterrupt("Terminate and release synthetic children")
    signal.signal(signal.SIGTERM, terminate)
    root = base/"campaign"
    width = 1 if case == "budget" else 3
    tasks = TASKS[:1] if width == 1 else TASKS
    size = 2**20
    class Memory:
        def snapshot(self):
            held = base/"old_gpu_pid"
            processes = {int(held.read_text()):size} if held.exists() else {}
            if processes:
                (base/"saw_old_gpu_pid").touch()
            return dict(free=0 if (base/"low_gpu").exists() else 32*size, processes=processes)
        def close(self):
            pass
    limits = dict(free_gpu_bytes=size, worker_rss_bytes=512*size, free_ram_bytes=8*size,
                  rss_bytes=2048*size, save_reserve_bytes=size, free_reserve_bytes=8*size, disk_bytes=64*size)
    caps = dict(training=97 if case == "budget" else 900, control=350, confirmation=150, generation=150, analysis=150)
    sources = [record(HERE)]
    statuses = read(gate)["post_status"]
    allowed = dict(workspace=str(base), campaign=str(root), experiment=PROTOCOL, workers=width, limits=limits,
                   other_gpu_commitments=[], sources=sources, authorization=record(gate), analysis_plan=record(gate),
                   prior_used_seconds={k: 7 if k=="training" else 0 for k in caps}, budget_seconds=caps,
                   post_seconds={k:90 for k in statuses}, cpu_test_seconds=0)
    def accepted(job, row):
        if case == "interrupt" and row["id"] == TASKS[0]:
            raise KeyboardInterrupt("handwritten parent interrupt during endpoint verification")
        assert read(job/"run"/"completion.json")["task"] == row["id"]
    # Every process is really pinned to the same single authorized CPU.
    core = min(os.sched_getaffinity(0))
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda s: time.sleep(min(.035,s)))
    with (patch.object(psutil.Process, "cpu_affinity", return_value=[core]*width),
          patch.object(hardware, "GpuMemory", Memory), patch.object(loop, "time", clock)):
        try:
            batch.campaign(allowed, gate, [dict(id=t, seconds=90) for t in tasks], HERE, accepted, statuses)
        except RuntimeError as exc:
            write(root/"fixture_stop.json", dict(error=repr(exc)))


def exercise(base, case):
    import psutil
    import step28_history_benchmark_dispatch as batch
    from step30_evolution_io import record, write
    base.mkdir()
    gate, root = base/"fixture.json", base/"campaign"
    from step28_history_benchmark import POST_STATUS
    write(gate, dict(case=case, post_status=POST_STATUS))
    def attempt(task, n=1):
        return root/"training"/task/f"attempt-{n}"
    def state():
        p=root/"dispatch.json"
        return read(p) if p.exists() else {}
    def request(n):
        return batch.request_recovery(root, TASKS[0], n, task_ids=TASKS, protocol=PROTOCOL)
    def failed(n):
        return next((s for s in state().get("oom_slots",[]) if s["task"]==TASKS[0] and s["attempt"]==n),None)
    evidence = dict(case=case)
    with (base/"parent.log").open("w") as output:
        proc=subprocess.Popen([sys.executable,"-B",str(HERE),"--supervise",str(gate)],stdout=output,stderr=output)
        try:
            initial = TASKS[:1] if case=="budget" else TASKS[:3]
            wait(lambda: all((attempt(t)/"running").exists() for t in initial))
            (base/"fail").touch()
            if case in ("handoff","repeat","budget"):
                first = wait(lambda: failed(1))
                first_hash=record(attempt(TASKS[0])/"dispatch_receipt.json")
                if case!="budget":
                    (base/"first_healthy_done").touch()
                    wait(lambda: (attempt(TASKS[3])/"running").exists())
                    peers={t:read(attempt(t)/"process.json")["pid"] for t in (TASKS[2],TASKS[3])}
                    assert not (attempt(TASKS[0],2)/"process.json").exists()
                    (base/"old_gpu_pid").write_text(str(first["pid"]))
                    (base/"low_gpu").touch()
                    wait(lambda: (base/"saw_old_gpu_pid").exists())
                assert request(1)==request(1)
                if case!="budget":
                    time.sleep(.12)
                    assert not (attempt(TASKS[0],2)/"admission.json").exists()
                    (base/"old_gpu_pid").unlink()
                    admission=attempt(TASKS[0],2)/"admission.json"
                    wait(lambda: admission.exists() and not read(admission)["ready"])
                    assert not (attempt(TASKS[0],2)/"process.json").exists()
                    (base/"low_gpu").unlink()
                    wait(lambda: (attempt(TASKS[0],2)/"running").exists())
                    second=read(attempt(TASKS[0],2)/"process.json")
                    assert second["pid"]!=first["pid"] and second["slot"]==first["slot"]
                    assert all(psutil.pid_exists(pid) for pid in peers.values())
                    (base/"retry_done").touch()
                    if case=="repeat":
                        wait(lambda: failed(2))
                        assert request(2)==request(2)
                    wait(lambda: (attempt(TASKS[4])/"running").exists())
                    assert read(attempt(TASKS[4])/"process.json")["slot"]==first["slot"]
                    assert all(psutil.pid_exists(pid) for pid in peers.values())
                    assert first_hash==record(attempt(TASKS[0])/"dispatch_receipt.json")
                    evidence.update(original_pid=first["pid"],retry_pid=second["pid"],healthy_pids=peers,
                          healthy_next=TASKS[3],reopened_next=TASKS[4],held_while_old_pid_visible=True,
                          held_below_reservation=True,duplicate_request_idempotent=True,first_receipt_unchanged=True)
            elif case=="other":
                wait(lambda: state().get("budget_stop"))
                try:
                    request(1)
                except ValueError:
                    pass
                else:
                    raise AssertionError("non-OOM request accepted")
                assert not attempt(TASKS[3]).exists()
                (base/"first_healthy_done").touch()
            if case!="interrupt":
                (base/"peers_done").touch()
            proc.wait(timeout=14)
            assert proc.returncode==0,(base/"parent.log").read_text()
            receipts=[read(p) for p in (root/"training").glob("*/*/dispatch_receipt.json")]
            log=read(root/"budget.json")
            events={e["operation"]:e for e in log["events"] if e["bucket"]=="training"}
            assert len(events)==len(receipts) and all(r["released"] for r in receipts)
            for r in receipts:
                e=events[f"{r['task_id']}/attempt-{r['attempt']}"]
                assert (e["actual_seconds"],e["charged_seconds"],e["success"])==(r["seconds"],math.ceil(r["seconds"]),r["success"])
                assert not psutil.pid_exists(r["process"]["pid"])
            assert log["used_seconds"]["training"]==7+sum(e["charged_seconds"] for e in events.values())
            manifest=root/"training_complete.json"
            assert manifest.exists()==(case=="handoff") and not attempt(TASKS[0],3).exists()
            if case=="handoff":
                wins=read(manifest)["winners"]
                assert set(wins)==set(TASKS) and len(events)==6
                assert all(batch.winner_directory(root,t,v)==attempt(t,v["attempt"]) for t,v in wins.items())
                assert wins[TASKS[0]]["attempt"]==2
            elif case=="repeat":
                assert len(events)==6 and sum(not e["success"] for e in events.values())==2
            elif case=="budget":
                assert len(events)==1 and state()["budget_stop"] and not (attempt(TASKS[0],2)/"process.json").exists()
            elif case=="other":
                assert len(events)==3 and sum(e["success"] for e in events.values())==2
            elif case=="interrupt":
                assert len(events)==3 and not any(e["success"] for e in events.values())
                assert (attempt(TASKS[0])/"shared_failure.json").exists()
            posts = [p for p in POST_STATUS if (root/p/"result.json").is_file()]
            assert posts == (list(POST_STATUS) if case=="handoff" else [])
            assert (root/"completion.json").exists()==(case=="handoff")
            assert (root/"failure.json").exists()==(case!="handoff")
            evidence.update(attempts=len(receipts),inherited_seconds=7,synthetic_post_actions=posts,
                settled_training_seconds=log["used_seconds"]["training"],status=state()["status"],
                winners=len(read(manifest)["winners"]) if manifest.exists() else 0)
        finally:
            for flag in ("fail","first_healthy_done","retry_done","peers_done"):
                (base/flag).touch(exist_ok=True)
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=12)
    return evidence


if __name__=="__main__":
    if sys.argv[1:2] and sys.argv[1] in ("worker", "generation", "confirmation", "analysis"):
        parser=argparse.ArgumentParser()
        parser.add_argument("action"); parser.add_argument("--gate",type=Path)
        parser.add_argument("--id"); parser.add_argument("--deadline",type=float); parser.add_argument("--attempt",type=int)
        raise SystemExit(child(parser.parse_args()))
    if sys.argv[1:2]==["--supervise"]:
        controller(Path(sys.argv[2]))
    else:
        import torch
        import step28_history_benchmark_dispatch as batch
        from step28_forever_formal import Budget
        assert sys.platform=="linux" and Path(sys.executable).resolve()==Path(batch.PYTHON).resolve()
        assert not torch.cuda.is_available() and len(os.sched_getaffinity(0))==1
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for name,error,kind in [("cuda",torch.cuda.OutOfMemoryError("CUDA out of memory"),"CUDA_OOM"),
                                   ("cpu",MemoryError("CPU allocation failed"),"OTHER"),
                                   ("numeric",FloatingPointError("nonfinite gradient"),"OTHER")]:
                batch.record_worker_failure(base/name,"worker",error,"synthetic")
                assert read(base/name/"worker_failure.json")["kind"]==kind
            limits = dict(rss_bytes=4*2**30, disk_bytes=8*2**20, small_bytes=8*2**20,
                          free_reserve_bytes=1, allocator_bytes=2**20)
            for name, change in (("local_rss", {"rss_bytes":1}), ("shared_disk", {"disk_bytes":1})):
                try:
                    Budget(base, time.monotonic()+100, {**limits, **change}).check(reserve=32)
                except (RuntimeError, OSError) as error:
                    batch.record_worker_failure(base/name, "worker", error, "handwritten resource limit")
                else:
                    raise AssertionError("Native resource protection failed")
                assert (base/name/"shared_failure.json").is_file() == (name=="shared_disk")
            cases=[exercise(base/c,c) for c in ("handoff","repeat","other","budget","interrupt")]
        print(json.dumps(dict(passed=True,cases=cases,local_rss_and_shared_disk_distinguished=True,
                              logical_slots_up_to=3,physical_cpu_cores=1,
                              gpu_kernels=0,formal_data_reads=0,model_updates=0)))
