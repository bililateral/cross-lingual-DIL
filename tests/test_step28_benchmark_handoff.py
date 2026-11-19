"""Chinese dispatch with real Linux children, injected OOM and handwritten NVML."""
import argparse
from contextlib import ExitStack
import errno
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
        if case == "other":
            from step28_history_benchmark_dispatch import record_worker_failure
            record_worker_failure(target, "worker", ValueError("method-local synthetic failure"), "handwritten")
            return 1
        if case == "source":
            # The production parent bound this file before this child was started.
            from step30_evolution_io import verify
            (root/"shared_source").write_text("changed")
            verify(config["shared_source"])
        if case == "bad_json":
            (target/"worker_failure.json").write_text("{")
            return 1
        if case != "interrupt":
            write(target/"worker_failure.json", dict(kind="CUDA_OOM"))
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
            foreign = base/"unknown_gpu_pid"
            if foreign.exists():
                processes[int(foreign.read_text())] = size
            return dict(free=0 if (base/"low_gpu").exists() else 32*size, processes=processes)
        def close(self):
            pass
    limits = dict(free_gpu_bytes=size, worker_rss_bytes=512*size, free_ram_bytes=8*size,
                  rss_bytes=2048*size, save_reserve_bytes=size, free_reserve_bytes=8*size, disk_bytes=64*size)
    caps = dict(training=97 if case == "budget" else 900, control=350, confirmation=150, generation=150, analysis=150)
    sources = [record(HERE), read(gate)["shared_source"]]
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
    real_write, real_release, real_command = loop.write, loop.release, batch.ChineseRunner.command
    def write_with_full_disk(path, value):
        if path.name == "dispatch_receipt.json" and TASKS[0] in path.parts and not (base/"write_fault").exists():
            (base/"write_fault").touch()
            raise OSError(errno.ENOSPC, "handwritten first receipt write")
        real_write(path, value)
    def release_with_interrupt(proc):
        if proc.poll() is None and not (base/"release_fault").exists():
            (base/"release_fault").write_text(str(proc.pid))
            raise InterruptedError("handwritten first live release")
        real_release(proc)
    def interrupt_third(runner, job, slot, deadline):
        if slot == 2:
            wait(lambda: all((root/"training"/t/"attempt-1"/"running").exists() for t in TASKS[:2]))
            raise KeyboardInterrupt("handwritten third-slot launch interrupt")
        return real_command(runner, job, slot, deadline)
    with ExitStack() as patches:
        patches.enter_context(patch.object(psutil.Process, "cpu_affinity", return_value=[core]*width))
        patches.enter_context(patch.object(hardware, "GpuMemory", Memory))
        patches.enter_context(patch.object(loop, "time", clock))
        if case == "receipt_io":
            patches.enter_context(patch.object(loop, "write", write_with_full_disk))
        if case == "release_once":
            patches.enter_context(patch.object(loop, "release", release_with_interrupt))
        if case == "launch_interrupt":
            patches.enter_context(patch.object(batch.ChineseRunner, "command", interrupt_third))
        try:
            batch.campaign(allowed, gate, [dict(id=t, seconds=90) for t in tasks], HERE, accepted, statuses)
        except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as exc:
            write(root/"fixture_stop.json", dict(error=repr(exc)))


def exercise(base, case):
    import psutil
    import step28_history_benchmark_dispatch as batch
    from step30_evolution_io import record, write
    base.mkdir()
    gate, root = base/"fixture.json", base/"campaign"
    from step28_history_benchmark import POST_STATUS
    (base/"shared_source").write_text("fixed")
    write(gate, dict(case=case, post_status=POST_STATUS, shared_source=record(base/"shared_source")))
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
            if case != "launch_interrupt":
                initial = TASKS[:1] if case=="budget" else TASKS[:3]
                wait(lambda: all((attempt(t)/"running").exists() for t in initial))
                if case in ("sigterm","release_once"):
                    proc.send_signal(signal.SIGTERM)
                else:
                    (base/"fail").touch()
            if case in ("handoff","repeat","budget","tamper"):
                first = wait(lambda: failed(1))
                first_hash=record(attempt(TASKS[0])/"dispatch_receipt.json")
                if case!="budget":
                    (base/"old_gpu_pid").write_text(str(first["pid"]))
                    wait(lambda: (base/"saw_old_gpu_pid").exists())
                    if case=="handoff":
                        (base/"unknown_gpu_pid").write_text(str(os.getpid()))
                    (base/"first_healthy_done").touch()
                    if case=="handoff":
                        admission=attempt(TASKS[3])/"admission.json"
                        wait(lambda: admission.exists() and os.getpid() in read(admission).get("unknown_pids",[]))
                        (base/"unknown_gpu_pid").unlink()
                    wait(lambda: (attempt(TASKS[3])/"running").exists())
                    assert first["pid"] in read(attempt(TASKS[3])/"admission.json")["released_pids_still_in_nvml"]
                    peers={t:read(attempt(t)/"process.json")["pid"] for t in (TASKS[2],TASKS[3])}
                    assert not (attempt(TASKS[0],2)/"process.json").exists()
                    (base/"low_gpu").touch()
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
                    if case=="tamper":
                        # Same path, task, slot and parsed contents; only the bytes change.
                        with (attempt(TASKS[0])/"dispatch_receipt.json").open("a") as out:
                            out.write(" ")
                    (base/"retry_done").touch()
                    if case=="repeat":
                        wait(lambda: failed(2))
                        assert request(2)==request(2)
                    if case!="tamper":
                        wait(lambda: (attempt(TASKS[4])/"running").exists())
                        assert read(attempt(TASKS[4])/"process.json")["slot"]==first["slot"]
                        assert all(psutil.pid_exists(pid) for pid in peers.values())
                        assert first_hash==record(attempt(TASKS[0])/"dispatch_receipt.json")
                        evidence.update(original_pid=first["pid"],retry_pid=second["pid"],healthy_pids=peers,
                              healthy_next_while_old_pid_visible=TASKS[3],reopened_next=TASKS[4],held_while_old_pid_visible=True,
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
            if case in ("handoff","repeat","other","budget"):
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
                if r["process"]:
                    assert not psutil.pid_exists(r["process"]["pid"])
                else:
                    assert case=="launch_interrupt" and r["task_id"]==TASKS[2]
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
                assert not (attempt(TASKS[0])/"shared_failure.json").exists()
            elif case=="interrupt":
                assert len(events)==3 and not any(e["success"] for e in events.values())
                assert (attempt(TASKS[0])/"shared_failure.json").exists()
            elif case=="tamper":
                assert len(events)==5 and read(attempt(TASKS[0],2)/"dispatch_receipt.json")["success"] is False
                assert "identity mismatch" in read(root/"failure.json")["error"] and not attempt(TASKS[4]).exists()
            else:
                assert len(events)==3 and not any(e["success"] for e in events.values())
                primary=read(root/"failure.json")["error"]
                expected={"source":"identity mismatch","bad_json":"JSONDecodeError",
                          "receipt_io":"first receipt write","launch_interrupt":"third-slot launch interrupt",
                          "sigterm":"SIGTERM","release_once":"SIGTERM"}[case]
                assert expected in primary, primary
                if case=="release_once":
                    assert (base/"release_fault").is_file() and state()["active"]=={}
                    assert any("first live release" in str(r.get("settlement_errors")) for r in receipts)
                if case=="source":
                    assert "shared_source" in read(attempt(TASKS[0])/"shared_failure.json")["error"]
                if case=="receipt_io":
                    assert (base/"write_fault").is_file() and any(r.get("settlement_errors") for r in receipts)
                evidence.update(original_error=primary)
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


def crossing_control_budget(base, original_failure):
    import step28_history_benchmark_dispatch as batch
    from step30_evolution_io import record, write
    base.mkdir()
    gate = base/"fixture.json"
    write(gate, {"synthetic":True})
    caps = dict(training=800, confirmation=150, generation=150, analysis=150, control=160)
    allowed = dict(workspace=str(base), campaign=str(base/"campaign"), experiment=PROTOCOL, workers=1,
                   limits=dict(worker_rss_bytes=2**28, rss_bytes=2**29, free_reserve_bytes=1),
                   sources=[record(gate)], authorization=record(gate), analysis_plan=record(gate),
                   cpu_test_seconds=0, budget_seconds=caps, prior_used_seconds={k:0 for k in caps}, post_seconds={})
    def phase(runner, *args):
        runner.handoff_since = time.monotonic()-180
        if original_failure:
            raise ValueError("handwritten primary phase failure")
        runner.settle_handoff()
    with patch.object(batch, "training_phase", phase):
        try:
            batch.campaign(allowed, gate, [], HERE, None, {})
        except (TimeoutError, ValueError) as error:
            assert ("primary phase failure" if original_failure else "Cumulative budget exceeded") in str(error)
        else:
            raise AssertionError("Control cap passed")
    root = base/"campaign"
    log, failure = read(root/"budget.json"), read(root/"failure.json")
    charges = [e for e in log["events"] if e["operation"]=="controller_setup_or_handoff"]
    assert len(charges)==1 and charges[0]["charged_seconds"] in (180,181)
    assert log["used_seconds"]["auxiliary"]==30+charges[0]["charged_seconds"] and not (root/"completion.json").exists()
    assert bool(failure["cleanup_errors"])==original_failure
    return dict(primary=failure["error"],cleanup=failure["cleanup_errors"],charge_count=len(charges),
                charged_seconds=charges[0]["charged_seconds"])


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
            cases=[exercise(base/c,c) for c in ("handoff","repeat","other","budget","interrupt","source",
                   "bad_json","receipt_io","launch_interrupt","sigterm","release_once","tamper")]
            crossing=[crossing_control_budget(base/str(original),original) for original in (False,True)]
        print(json.dumps(dict(passed=True,cases=cases,local_rss_and_shared_disk_distinguished=True,
                              control_cap=crossing,logical_slots_up_to=3,physical_cpu_cores=1,
                              gpu_kernels=0,formal_data_reads=0,model_updates=0)))
