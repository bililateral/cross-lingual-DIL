"""Start the single Chinese benchmark after the bound FIT12 campaign releases its resources."""
import argparse
import hashlib
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
    # This controller writes completion before its final accounting call. Its
    # last three internal charges must show analysis, monitor, then handoff.
    budget = read(root/"budget.json")
    internal = [e for e in budget["events"] if e["operation"] not in budget["operation_ids"]]
    if (budget["caps"] != gate["caps"] or set(budget["used_seconds"]) != set(gate["caps"]) or
            not all(0 <= budget["used_seconds"][k] <= limit for k,limit in gate["caps"].items()) or
            [(e["operation"],e["bucket"],e["success"]) for e in internal[-3:]] !=
            [(op,"auxiliary",True) for op in ("analysis","dispatch_monitor","controller_setup_or_handoff")]):
        raise ValueError("VendorLink final accounting is incomplete or over budget")
    return dict(ready=True, status="VENDORLINK_COMPLETE_RELEASED", completion=record(root/"completion.json"),
                settled_budget=record(root/"budget.json"))


def resources(g, wait):
    limits = g["limits"]
    obs = resource_observation(dict(g, experiment_root=str(wait)), wait, True, limits=dict(
        limits, free_disk_reserve=limits["free_reserve_bytes"]))
    obs["available_cores"] = len(psutil.Process().cpu_affinity())
    obs["ready"] = bool(obs["ready"] and obs["available_cores"] >= 1)
    return obs


def training_inputs(workspace, project, install=False):
    import step28_history_benchmark as b
    reference = b.five.base.contract()
    data = Path(reference["data_root"])
    model = Path(reference["models"]["split_rank"]["path"])
    for name in b.NAMES:
        module = (b.history if name == b.H else b.forever if name == b.forever.NAME else
                  b.comparison if name in b.comparison.NEW_METHODS else b.five)
        c = module.config(b.native(name,"ABC",4))
        if c["data_root"] != str(data) or c["models"]["split_rank"] != reference["models"]["split_rank"]:
            raise ValueError("Unexpected model or training archive")
    files = ("manifest.json","validation.json","groups.csv","train/items.jsonl","train/supervision/pairs.csv")
    inputs = []
    for rel in [data/f for f in files]+[model]:
        original, dest = project/rel, workspace/rel
        if dest.parent.resolve()!=dest.parent or not (original.is_dir() if rel==model else original.is_file()):
            raise ValueError("Wrong source or linked input parent: "+str(rel))
        if dest.is_symlink() or dest.exists():
            if not dest.is_symlink() or dest.resolve()!=original.resolve():
                raise ValueError("Input destination already occupied: "+str(rel))
        elif install:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(original,target_is_directory=rel==model)
        inputs.append(dict(relative=str(rel),available=True,linked=dest.is_symlink()))
    for role in ("valid","heldout","owners","private_custody"):
        extra = workspace/data/role
        if extra.exists() or extra.is_symlink():
            raise ValueError("Additional role entry in isolated workspace")
    from step30_evolution_io import sha
    for name,key in (("manifest.json","data_manifest_sha256"),("validation.json","data_validation_sha256")):
        if sha(project/data/name)!=reference[key]:
            raise ValueError("Frozen training metadata differs")
    return inputs


def verify_handoff(g, gate):
    wait = gate.parent/"wait"
    ready, armed = read(wait/"ready.json"), read(wait/"controller.json")
    if (ready["gate"] != record(gate) or ready["identity"] != process_identity() or
            ready["identity"] != armed["identity"] or ready["gate"] != armed["gate"] or
            ready["identity"]["command"][-3:] != [g["listener"]["path"], "--gate", str(gate)] or
            not ready["dependency"]["ready"] or not ready["resources"]["ready"]):
        raise ValueError("Only the live resource waiter can start this authorized campaign")
    return math.ceil(ready["active_seconds_at_handoff"]+time.monotonic()-ready["handoff_monotonic"])


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
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n") or not line.startswith(b"{") or b'"updates"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if event.get("event") != "updates" or event.get("stage") != 1:
                    continue
                if event.get("run_id") != expected:
                    continue
                count = event.get("completed")
                if row["name"] == bench.forever.NAME and event.get("counts", {}).get("current") != count:
                    continue
                if (type(count) is int and count > 0 and matches(item["identity"])
                        and read(claim)["id"] == job["task_id"]):
                    return dict(task=row["id"], attempt=job["attempt"], stage=1, first_observed_updates=count,
                                native_run=expected, worker=item["identity"], console=str(path/"console.txt"),
                                offset=offset, line_sha256=hashlib.sha256(line).hexdigest())


def retire(g, root, active):
    if (root/"waiter_deleted.json").is_file():
        return
    gate, journal = root.parent/"formal_gate.json", root.parent/"wait/first_update.json"
    script = Path(g["workspace"])/"scripts/once_wait.py"
    if script != Path(g["listener"]["path"]) or script.parent.resolve() != script.parent or script.is_symlink():
        raise ValueError("Wrong disposable listener")
    if not journal.is_file():
        event = first_update(g, active)
        if event is None:
            return
        verify(g["listener"])
        st = script.stat()
        proof = dict(source=g["listener"], gate=record(gate), controller=process_identity(),
                     file_identity=[st.st_dev,st.st_ino], observed=time.time(), **event)
        write(journal, proof)
    # The original attempt, process, log line and file identity also bind recovery
    # after interruption. Reconciliation never launches a controller or worker.
    proof, armed = read(journal), read(root.parent/"wait/controller.json")
    key = (proof["task"], proof["attempt"])
    if (Path(g["campaign"]) != root or proof["source"] != g["listener"] or
            proof["gate"] != record(gate) or proof["gate"] != armed["gate"] or proof["controller"] != armed["identity"] or
            key[0] not in {row["id"] for row in g["runs"]} or key[1] not in (1,2)):
        raise ValueError("First-update proof belongs to a different deployment")
    path = root/"training"/key[0]/f"attempt-{key[1]}"
    process = read(path/"process.json")
    if ((process["task_id"],process["attempt"]) != key or read(path/"worker_claim.json")["id"] != key[0] or
            any(proof["worker"][k] != process[k] for k in ("pid","owner","cwd","command","boot_id","start_ticks")) or
            proof["console"] != str(path/"console.txt")):
        raise ValueError("First-update proof belongs to a different worker")
    with (path/"console.txt").open("rb") as stream:
        stream.seek(proof["offset"])
        if hashlib.sha256(stream.readline()).hexdigest() != proof["line_sha256"]:
            raise ValueError("The original update line changed")
    present = script.exists()
    if present:
        verify(g["listener"])
        st = script.stat()
        if [st.st_dev,st.st_ino] != proof["file_identity"]:
            raise ValueError("Refusing to remove the replacement listener")
        script.unlink()
    try:
        write(root/"waiter_deleted.json", dict(**proof, removed=True,
              disposition="unlinked" if present else "already_absent_on_reconciliation"))
    except OSError:
        if not active:
            raise
        # The next dispatch poll can finish this receipt from the durable proof.
        # The disk guard in the same poll still enforces the campaign reserve.


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
                project = gate.parent.parent.parent.parent.parent
                if not all(i["linked"] for i in training_inputs(Path(g["workspace"]),project)):
                    raise ValueError("A named training input link disappeared")
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
                handoff = time.monotonic()
                write(wait/"ready.json", dict(status, charged_control_seconds=math.ceil(charged),
                      active_seconds_at_handoff=charged,handoff_monotonic=handoff))
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
              charged_control_seconds=(math.ceil(charged) if (Path(g["campaign"])/"budget.json").exists() else
                                       math.ceil(charged+time.monotonic()-handoff)),
              included_in_campaign_ledger=(Path(g["campaign"])/"budget.json").exists(),ready_receipt=str(wait/"ready.json"),
              campaign_ledger=str(Path(g["campaign"])/"budget.json")))
        raise


def main(start):
    import step28_history_benchmark as bench
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True, type=Path)
    gate = p.parse_args().gate.resolve()
    try:
        g = bench.read_gate(gate)
        if (Path(sys.executable).resolve() != Path(PYTHON).resolve() or
                Path(sys.argv[0]).resolve() != verify(g["listener"]) or
                Path.cwd() != Path(g["workspace"]) or not g["predecessor"]):
            raise ValueError("Run the bound one-shot listener in the Chinese workspace")
    except BaseException as error:
        if (gate.parent/"start/listener_console.txt").exists() and not (gate.parent/"wait").exists():
            with (gate.parent/"start/listener_entry_failure.json").open("x",encoding="utf-8") as out:
                json.dump(dict(error=repr(error),active_seconds=time.monotonic()-start,automatic_restart=False),out)
        raise
    os.umask(0o077)
    def terminate(sig, frame):
        raise InterruptedError("Resource listener received SIGTERM")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        wait_then_start(g, gate, start)
    finally:
        signal.signal(signal.SIGTERM, previous)
