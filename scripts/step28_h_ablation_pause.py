"""Reviewed, one-off controller handover; scientific workers and frozen sources stay intact."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def verify(row):
    path = Path(row["path"])
    payload = path.read_bytes()
    if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"Source identity mismatch: {path}")
    return path


def proc_stat(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")")+2:].split()
    return dict(state=fields[0], parent=int(fields[1]), start_ticks=int(fields[19]),
                exit_status=int(fields[49]))


def identity(pid):
    process = psutil.Process(pid)
    return dict(pid=pid, start_ticks=proc_stat(pid)["start_ticks"], owner=process.username(),
                command=process.cmdline(), cwd=process.cwd())


def same_process(expected):
    if identity(expected["pid"]) != expected or expected["owner"] != psutil.Process().username():
        raise ValueError("Owned process identity changed")
    return psutil.Process(expected["pid"])


class HeldChild:
    """Read exit status while the original, stopped parent still owns the zombie."""
    def __init__(self, expected, parent):
        self.pid, self.expected, self.parent = expected["pid"], expected, parent
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        value = proc_stat(self.pid)
        if value["start_ticks"] != self.expected["start_ticks"] or value["parent"] != self.parent:
            raise RuntimeError("Held child identity/parent changed; exit status cannot be assumed")
        if value["state"] == "Z":
            self.returncode = os.waitstatus_to_exitcode(value["exit_status"])
        return self.returncode

    def wait(self, timeout):
        end = time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic() >= end:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.02)
        return self.returncode


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restored_ledger(controller, campaign, budget, extra_seconds):
    if budget["caps"] != controller.run.BUDGETS or extra_seconds < 0:
        raise ValueError("The existing budget must be preserved")
    ledger = controller.Ledger.__new__(controller.Ledger)
    ledger.root, ledger.caps = campaign, dict(budget["caps"])
    ledger.used, ledger.events = dict(budget["used_seconds"]), list(budget["events"])
    ledger.used["auxiliary"] += extra_seconds
    ledger.events.append(dict(id="pause_preparation", bucket="auxiliary", conservative_seconds=extra_seconds))
    ledger.tick = time.monotonic()
    ledger.check()
    return ledger


def freeze(controller, request, campaign, state):
    """Stop only the old parent, then require a consistent persisted dispatch boundary."""
    parent = same_process(request["old_controller"])
    snapshot = read(campaign/"training/dispatch.json")
    expected = {v["pid"]: identity(v["pid"]) for v in snapshot["active"].values()}
    parent.send_signal(signal.SIGSTOP)
    try:
        end = time.monotonic()+3
        while parent.status() != psutil.STATUS_STOPPED:
            if time.monotonic() >= end:
                raise RuntimeError("Original controller did not enter STOPPED")
            time.sleep(.02)
        current = read(campaign/"training/dispatch.json")
        if current != snapshot:
            raise RuntimeError("Dispatch changed during handover preparation")
        children = {p.pid for p in parent.children()}
        if children != set(expected):
            raise RuntimeError("A child was launched/reaped outside the saved dispatch boundary")
        rows = []
        for slot, item in snapshot["active"].items():
            process = same_process(expected[item["pid"]])
            rec = read(campaign/"training"/item["id"]/"process.json")
            argv = process.cmdline()
            if (argv != rec["command"][3:] or rec["slot"] != int(slot)
                    or process.ppid() != parent.pid or process.cwd() != request["workspace"]):
                raise RuntimeError("Worker command, parent, slot or cwd differs")
            deadline = float(argv[argv.index("--deadline")+1])
            rows.append(dict(slot=int(slot), id=item["id"], identity=expected[process.pid],
                             started=deadline-21600, deadline=deadline))
        original_budget = read(campaign/"budget.json")
        if (campaign/"failure.json").exists() or (campaign/"completion.json").exists():
            raise RuntimeError("Closed campaign cannot be handed over")
        write(state/"frozen.json", dict(dispatch=snapshot, budget=original_budget, active=rows,
              old_controller=request["old_controller"], frozen_monotonic=time.monotonic(), epoch=time.time()))
        return snapshot, original_budget, rows
    except BaseException:
        parent.send_signal(signal.SIGCONT)
        raise


def drain(controller, request, campaign, state, snapshot, budget, rows):
    """Keep original workers monitored, validate their exits, and never dispatch new work."""
    old, queue, run = controller.inherited, controller.queue, controller.run
    ledger = restored_ledger(controller, campaign, budget, request["preparation_seconds"])
    allowed = run.gate(Path(request["original_gate"]), "train")
    allowed["_gate_sha256"] = controller.data.sha256(Path(request["original_gate"]))
    specs = {s.run_id: s for s in controller.m.specifications()}
    active = {r["slot"]: {**r, "process": HeldChild(r["identity"], request["old_controller"]["pid"])} for r in rows}
    monitor = controller.ObservedGpuMemory(queue.GpuMemory(), campaign, resume=True)
    def save():
        snapshot["active"] = {str(k): dict(id=v["id"], pid=v["process"].pid) for k, v in active.items()}
        snapshot["status"] = "PAUSED_DRAINING" if active else "PAUSED_IDLE"
        write(campaign/"training/dispatch.json", snapshot)
        ledger.save()
    save()
    try:
        while active:
            same_process(request["old_controller"])
            if proc_stat(request["old_controller"]["pid"])["state"] != "T":
                raise RuntimeError("Original dispatcher was resumed outside the handover")
            ledger.control()
            queue.check_running(None, allowed, campaign)
            family = [psutil.Process(), psutil.Process(request["old_controller"]["pid"])]
            family += [q for r in active.values() for q in [psutil.Process(r["process"].pid),
                        *psutil.Process(r["process"].pid).children(recursive=True)]]
            if sum(p.memory_info().rss for p in {p.pid: p for p in family}.values()) > run.LIMITS["rss_bytes"]:
                raise RuntimeError("Combined original workers/controller RSS limit exceeded")
            gpu = monitor.inner.snapshot()
            for slot, item in list(active.items()):
                process, error = item["process"], None
                used = gpu["processes"].get(process.pid)
                if used is not None:
                    key = f"{process.pid}@{psutil.Process(process.pid).create_time()}"
                    prior = monitor.peaks.get(key, {}).get("observed_peak_gpu_bytes", 0)
                    if used > prior:
                        monitor.peaks[key] = dict(pid=process.pid, created=psutil.Process(process.pid).create_time(),
                                                 observed_peak_gpu_bytes=used)
                        monitor.save()
                    if used > run.LIMITS["process_gpu_bytes"]:
                        error = "PROCESS_GPU_LIMIT_EXCEEDED"
                if time.monotonic() >= item["deadline"]-60:
                    error = error or "SHUTDOWN_RESERVE_REACHED"
                if error is None and process.poll() is None:
                    continue
                unexpected = old.release(process)
                ended = time.monotonic()
                ledger.finish("training", item["id"], item["started"], ended)
                if process.returncode or unexpected:
                    error = error or f"Worker exit={process.returncode}; unexpected descendants={unexpected}"
                if ended-item["started"] > 21600:
                    error = error or "Actual task wall time exceeds its cap"
                if error is None:
                    try:
                        run.validate_completion(campaign/"training"/item["id"]/"run", specs[item["id"]], allowed)
                    except Exception as exc:
                        error = repr(exc)
                index = next(i for i, s in enumerate(controller.m.specifications()) if s.run_id == item["id"])
                receipt = dict(id=item["id"], index=index, slot=slot, success=error is None, error=error,
                               released=True, actual_seconds=ended-item["started"], bucket="training")
                snapshot["completed"].append(receipt)
                write(campaign/"training"/item["id"]/"dispatch_receipt.json", receipt)
                if error:
                    snapshot["retired_slots"] = sorted(set(snapshot["retired_slots"]) | {slot})
                active.pop(slot)
                save()
            if active:
                ledger.pause(2, active=True)
        ledger.control()
        save()
        return dict(dispatch=snapshot, budget=read(campaign/"budget.json"),
                    prior_controller=dict(identity=request["old_controller"],
                    original_launch=request["original_launch"],
                    ended_epoch=time.time(), stop_reason="USER_REQUESTED_DRAIN_AND_CONTROL_HANDOVER"))
    except BaseException as error:
        # Never leave live workers without a supervisor on bridge failure.
        for slot, item in list(active.items()):
            old.release(item["process"])
            ledger.finish("training", item["id"], item["started"], time.monotonic())
        write(campaign/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error),
              used_seconds=ledger.used, handover_failure=True))
        write(state/"failure.json", dict(error=repr(error), phase="DRAIN", no_automatic_retry=True))
        raise
    finally:
        monitor.inner.close()


def takeover(request_path):
    request = read(request_path)
    if request.get("approved") is not True:
        raise ValueError("Reviewed handover approval required")
    for row in [request["review"], *request["updated_sources"]]:
        verify(row)
    state = Path(request_path).parent/"state"
    state.mkdir(exist_ok=True)
    lock = (state/"claim.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (state/"frozen.json").exists() or (state/"controller.json").exists():
        raise ValueError("No second takeover or automatic recovery of a failed handover")
    workspace = Path(request["workspace"])
    sys.path.insert(0, str(workspace/"scripts"))
    controller = load(verify(request["updated_control"]), "updated_ablation_controller")
    dispatch = load(verify(request["updated_dispatch"]), "updated_ablation_dispatch")
    campaign = Path(request["campaign"])
    gate = Path(request["original_gate"])
    allowed = controller.run.gate(gate, "train")
    if Path(allowed["campaign"]) != campaign or Path(allowed["workspace"]) != workspace:
        raise ValueError("Original campaign/workspace identity differs")
    controller.predecessors(allowed)
    # Default to paused; only an explicit later resume removes this marker.
    (campaign/"pause_dispatch").touch(exist_ok=False)
    snapshot, budget, rows = freeze(controller, request, campaign, state)
    write(state/"controller.json", dict(identity=identity(os.getpid()), phase="DRAINING", request=str(request_path)))
    try:
        continuation = drain(controller, request, campaign, state, snapshot, budget, rows)
        write(state/"continuation.json", continuation)
        # All original child groups have released. Only the old parent is retired.
        same_process(request["old_controller"]).send_signal(signal.SIGKILL)
        end = time.monotonic()+5
        while psutil.pid_exists(request["old_controller"]["pid"]):
            if psutil.Process(request["old_controller"]["pid"]).status() == psutil.STATUS_ZOMBIE:
                break
            if time.monotonic() >= end:
                raise RuntimeError("Old controller did not retire")
            time.sleep(.02)
        write(state/"controller.json", dict(identity=identity(os.getpid()), phase="PAUSED_CONTINUATION", request=str(request_path)))
        controller.execute(gate, continuation=continuation, dispatch_impl=dispatch.dispatch,
                           dispatch_paused=lambda: (campaign/"pause_dispatch").exists())
        write(state/"completion.json", dict(status="CONTROLLED_CAMPAIGN_COMPLETE", epoch=time.time()))
    except BaseException as error:
        write(state/"failure.json", dict(error=repr(error), no_automatic_retry=True))
        raise


def change(request_path, paused):
    request = read(request_path)
    state = Path(request_path).parent/"state"
    same_process(read(state/"controller.json")["identity"])
    marker = Path(request["campaign"])/"pause_dispatch"
    if paused:
        marker.touch(exist_ok=True)
    else:
        marker.unlink(missing_ok=True)
    write(state/"last_request.json", dict(paused=paused, epoch=time.time(), by=psutil.Process().username()))
    print(json.dumps(dict(requested_paused=paused, acknowledgement="Read the active dispatch status; one in-flight launch may precede acknowledgement")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("takeover", "pause", "resume"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "takeover":
        takeover(args.request)
    else:
        change(args.request, args.action == "pause")
