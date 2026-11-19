"""Start the single Chinese benchmark after the bound FIT12 campaign releases its resources."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import psutil

from step30_evolution_control import kernel_identity, resource_observation
from step30_evolution_io import PYTHON, read, record, verify, write


def process_identity():
    p = psutil.Process()
    return dict(pid=p.pid, owner=p.username(), cwd=p.cwd(), command=p.cmdline(), **kernel_identity(p.pid))


def matches(identity):
    try:
        p = psutil.Process(identity["pid"])
        return (p.status() != psutil.STATUS_ZOMBIE and
                kernel_identity(p.pid) == {k: identity[k] for k in ("boot_id", "start_ticks")} and
                (p.username(), p.cwd(), p.cmdline()) == (identity["owner"], identity["cwd"], identity["command"]))
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def predecessor(g):
    # Inspect only the operating receipts. The scientific decision is never opened.
    launch = read(verify(g["predecessor"]))
    gate_path = verify(launch["gate"])
    gate = read(gate_path)
    root = gate_path.parent/"campaign"
    parent = launch["controller"]
    if (gate.get("protocol") != "vendorlink-fit12-market-adapters" or Path(gate["campaign"]) != root or
            Path(gate["workspace"]) != gate_path.parent/"workspace" or
            Path(parent["cwd"]) != Path(gate["workspace"]) or
            parent["command"][-3:] != ["campaign","--gate",str(gate_path)]):
        raise ValueError("Different VendorLink predecessor")
    live = matches(parent)
    if (root/"failure.json").exists():
        raise RuntimeError("VendorLink has an unresolved terminal failure")
    if not (root/"completion.json").is_file():
        if not live:
            raise RuntimeError("VendorLink controller exited without batch completion")
        return dict(ready=False, status="WAIT_VENDORLINK", controller=parent["pid"])
    final = read(root/"completion.json")
    state = read(root/"dispatch.json")
    tasks = read(verify(gate["tasks"]))
    training, scoring = read(root/"training_complete.json"), read(root/"scoring_complete.json")
    if (final["status"] != "COMPLETE" or final["dev_opening"] != "whole_batch_once" or final["confirm_open"] is not False
            or final["decision"]["path"] != str(root/"analysis/result/decision.json") or len(tasks) != 12
            or training["tasks"] != tasks or set(training["winners"]) != {t["id"] for t in tasks}
            or scoring["tasks"] != tasks or [x["path"] for x in scoring["completions"]] !=
            [str(root/"scoring"/t["id"]/"result/completion.json") for t in tasks]
            or state["status"] != "COMPLETE" or state["active"] or state.get("cleanup_errors")):
        raise ValueError("VendorLink whole-batch receipts disagree")
    if live:
        return dict(ready=False, status="WAIT_VENDORLINK_RELEASE", controller=parent["pid"])
    related = []
    for p in psutil.process_iter(["username","status"]):
        if p.info["username"] != parent["owner"] or p.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        try:
            if str(gate_path) in p.cmdline():
                related.append(p.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    if related:
        return dict(ready=False, status="WAIT_VENDORLINK_RELEASE", related=related)
    return dict(ready=True, status="VENDORLINK_COMPLETE_RELEASED", completion=record(root/"completion.json"))


def resources(g, wait):
    limits = g["limits"]
    obs = resource_observation(dict(g, experiment_root=str(wait)), wait, True, limits=dict(
        limits, free_disk_reserve=limits["free_reserve_bytes"]))
    obs["available_cores"] = len(psutil.Process().cpu_affinity())
    obs["ready"] = bool(obs["ready"] and obs["available_cores"] >= 1)
    return obs


def verify_handoff(g, gate):
    wait = gate.parent/"wait"
    ready, armed = read(wait/"ready.json"), read(wait/"controller.json")
    if (ready["gate"] != record(gate) or ready["identity"] != process_identity() or
            ready["identity"] != armed["identity"] or ready["gate"] != armed["gate"] or
            ready["identity"]["command"][-3:] != [g["listener"]["path"], "--gate", str(gate)] or
            not ready["dependency"]["ready"] or not ready["resources"]["ready"]):
        raise ValueError("Only the live resource waiter can start this authorized campaign")
    return ready["charged_control_seconds"]


def first_update(g, active):
    import step28_history_benchmark as bench
    rows = {row["id"]:row for row in g["runs"]}
    for item in active.values():
        job = item["job"]
        if job["action"] != "worker":
            continue
        path = job["target"]
        claim = path/"worker_claim.json"
        if not claim.is_file():
            continue
        row = rows[job["task_id"]]
        expected = bench.native(row["name"], row["order"], row["seed"]).run_id
        with (path/"console.txt").open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or not line.startswith(b"{") or b'"updates"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if event.get("event") != "updates" or event.get("stage") != 1:
                    continue
                count = (event.get("counts",{}).get("current") if row["name"] == bench.forever.NAME
                         else event.get("completed") if event.get("run_id") == expected else None)
                if (type(count) is int and count > 0 and matches(item["identity"])
                        and read(claim)["id"] == job["task_id"]):
                    return dict(task=row["id"], attempt=job["attempt"], stage=1, first_observed_updates=count,
                                worker=item["identity"], console=str(path/"console.txt"))


def retire(g, root, active):
    if (root/"waiter_deleted.json").is_file():
        return
    event = first_update(g, active)
    if event is None:
        return
    script = verify(g["listener"])
    if script != Path(g["workspace"])/"scripts/once_wait.py" or script.is_symlink():
        raise ValueError("Wrong disposable listener")
    proof = dict(source=g["listener"], controller=process_identity(), observed=time.time(), **event)
    write(root.parent/"wait/first_update.json", proof)
    script.unlink()
    write(root/"waiter_deleted.json", dict(**proof, removed=True))


def wait_then_start(g, gate, start, *, sleep=time.sleep):
    import step28_history_benchmark as bench
    wait = gate.parent/"wait"
    if Path(g["campaign"]).exists():
        raise FileExistsError("Existing Chinese campaign")
    wait.mkdir(mode=0o700)  # Atomic, durable single-use claim, including unsuccessful waits.
    identity, gate_binding = process_identity(), record(gate)
    charged = 0.
    tick = start
    ready = False
    try:
        write(wait/"controller.json", dict(identity=identity, gate=gate_binding))
        while not ready:
            dep = predecessor(g)
            capacity = resources(g, wait) if dep["ready"] else None
            ready = bool(dep["ready"] and capacity["ready"])
            if ready:
                verify(gate_binding)
                verify(g["listener"])
                if bench.read_gate(gate) != g:
                    raise ValueError("The authorized deployment changed while waiting")
            seconds = math.ceil(charged+time.monotonic()-tick)
            if g["prior_used_seconds"]["control"]+seconds+120 >= g["budget_seconds"]["control"]:
                raise TimeoutError("Waiter reached the controller cleanup reserve")
            status = dict(status="READY" if ready else dep["status"] if not dep["ready"] else "WAIT_RESOURCES",
                          observed=time.time(), gate=gate_binding, identity=identity, dependency=dep,
                          resources=capacity, charged_control_seconds=seconds)
            write(wait/"status.json", status)
            charged += time.monotonic()-tick
            tick = None
            if ready:
                write(wait/"ready.json", dict(status, charged_control_seconds=math.ceil(charged)))
            else:
                sleep(60)
                tick = time.monotonic()
    except BaseException as error:
        write(wait/"failure.json", dict(error=repr(error), automatic_restart=False,
              charged_control_seconds=math.ceil(charged+(time.monotonic()-tick if tick is not None else 0))))
        raise
    try:
        bench.start_campaign(g, gate)
    except BaseException as error:
        write(wait/"failure.json", dict(error=repr(error), automatic_restart=False,
              charged_control_seconds=math.ceil(charged), ready_receipt=str(wait/"ready.json"),
              campaign_ledger=str(Path(g["campaign"])/"budget.json")))
        raise


def main(start):
    import step28_history_benchmark as bench
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True, type=Path)
    gate = p.parse_args().gate.resolve()
    g = bench.read_gate(gate)
    if (Path(sys.executable).resolve() != Path(PYTHON).resolve() or
            Path(sys.argv[0]).resolve() != verify(g["listener"]) or
            Path.cwd() != Path(g["workspace"]) or not g["predecessor"]):
        raise ValueError("Run the bound one-shot listener in the Chinese workspace")
    os.umask(0o077)
    def terminate(sig, frame):
        raise InterruptedError("Resource listener received SIGTERM")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        wait_then_start(g, gate, start)
    finally:
        signal.signal(signal.SIGTERM, previous)
