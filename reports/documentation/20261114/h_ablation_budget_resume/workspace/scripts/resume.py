"""One explicit three-task restart; auxiliary fees remain recorded, without a cap."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

TASKS = {"BCA_s4_ER", "BCA_s4_KD", "BCA_s4_SEQ"}
CAPS = dict(training=378*3600, confirmation=12*3600, auxiliary=None)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pinned(row):
    if record(row["path"]) != row:
        raise ValueError("Bound file changed: "+row["path"])
    return Path(row["path"])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    p = psutil.Process()
    return dict(pid=p.pid, owner=p.username(), command=p.cmdline(), cwd=p.cwd(),
                created=p.create_time(), start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def no_predecessors(request):
    for p in psutil.process_iter(["pid", "cmdline", "status"]):
        if p.pid == os.getpid() or p.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        command = p.info["cmdline"] or []
        if any(request["workspace"] in a or request["beta"]["root"] in a
               or request["prior_request"] == a for a in command):
            raise RuntimeError(f"Related predecessor still exists: {p.pid}")


def continuation(snapshot, ordered):
    value = copy.deepcopy(snapshot)
    failed = [r for r in value["completed"] if not r["success"]]
    indices = [r["index"] for r in value["completed"]]
    if (value["status"] != "STOP_NO_RETRY" or value["active"]
            or value["retired_slots"] != [0, 1, 2]
            or {r["id"] for r in failed} != TASKS or len(failed) != 3
            or len(value["completed"]) != 24 or len(set(indices)) != 24
            or any(ordered[r["index"]] != r["id"] or not r["released"] for r in value["completed"])
            or any(r["error"] != "CONTROLLER_STOP_NO_RETRY" for r in failed)
            or value["pending"] != [s for i, s in enumerate(ordered) if i not in indices]):
        raise ValueError("Only the three released budget-stop interruptions may be restarted")
    value["completed"] = [r for r in value["completed"] if r["success"]]
    done = {r["index"] for r in value["completed"]}
    value.update(pending=[s for i, s in enumerate(ordered) if i not in done],
                 retired_slots=[], status="AUTHORIZED_BUDGET_STOP_RESTART")
    return value, failed


def restore_beta(request, request_path, state):
    beta = request["beta"]
    root = Path(beta["root"])
    old = read(pinned(beta["gate"]))
    failure = read(pinned(beta["failure"]))
    if (root/"campaign").exists() or failure["error"] != "RuntimeError('Chinese dependency failed; no automatic beta launch')":
        raise ValueError("Only the failed, unstarted dependency waiter may be replaced")
    prior = failure["prior_seconds"]["auxiliary"]+failure["charged_seconds"]+90
    if prior+150 >= old["caps"]["auxiliary"]:
        raise TimeoutError("Beta retains its separate finite allowance")
    archive = root/"wait_before_budget_resume"
    gate_path = root/"gate_after_budget_resume.json"
    if archive.exists() or gate_path.exists():
        raise ValueError("Waiting restoration already attempted")
    new = copy.deepcopy(old)
    new["dependency"].update(request=record(request_path), controller=identity(), state=str(state))
    new["prior_seconds"]["auxiliary"] = prior
    (root/"wait").rename(archive)
    write(gate_path, new)
    command = [sys.executable, "-B", "-u", str(pinned(beta["entry"])), "--gate", str(gate_path)]
    with (state/"beta_console.txt").open("x") as log:
        child = subprocess.Popen(command, cwd=old["workspace"], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    end = time.monotonic()+60
    while not (root/"wait/status.json").exists():
        if child.poll() is not None or time.monotonic() >= end:
            raise RuntimeError("Restored waiter did not acknowledge; no automatic retry")
        time.sleep(.1)
    current = read(root/"wait/status.json")
    if current["status"] != "WAITING_CHINESE_COMPLETION" or current["identity"]["pid"] != child.pid:
        raise RuntimeError("Unexpected restored waiter identity/state")
    write(state/"beta_handover.json", dict(identity=current["identity"], gate=record(gate_path),
          archived_wait=str(archive), prior_auxiliary_seconds=prior,
          handover_conservative_seconds=90, unchanged_fields_except=["dependency", "prior_seconds"],
          scientific_sources_unchanged=True, training_started=False))


def execute(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    if (request.get("approved") is not True or request.get("review_closed") is not True
            or request.get("budget_caps") != CAPS or set(request.get("retry_tasks", [])) != TASKS
            or request.get("retry_count") != 1):
        raise ValueError("Explicit three-task restart and uncapped-auxiliary supplement required")
    for row in [request["authorization"], request["review"], *request["updated_sources"],
                *request["prior_bindings"]]:
        pinned(row)
    state = request_path.parent/"state"
    state.mkdir(exist_ok=True)
    with (state/"claim.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Path(request["campaign"])
        try:
            no_predecessors(request)
            if (campaign/"pause_dispatch").exists() or (campaign/"completion.json").exists():
                raise ValueError("New pause or complete campaign must not be overridden")
            sys.path.insert(0, str(Path(request["workspace"])/"scripts"))
            control = load(pinned(request["control"]), "budget_supplement_control")
            dispatcher = load(pinned(request["dispatch"]), "reviewed_fixed_dispatch")
            allowed = control.run.gate(Path(request["original_gate"]), "train")
            snapshot, budget = read(campaign/"training/dispatch.json"), read(campaign/"budget.json")
            effective, failed = continuation(snapshot, [s.run_id for s in control.m.specifications()])
            if (budget["caps"] != control.run.BUDGETS or set(budget["used_seconds"]) != set(CAPS)
                    or any(not math.isfinite(v) or v < 0 for v in budget["used_seconds"].values())
                    or budget["used_seconds"]["training"]+len(effective["pending"])*21600 > CAPS["training"]
                    or budget["used_seconds"]["confirmation"]+120 >= CAPS["confirmation"]
                    or "Cumulative allowance exhausted" not in read(campaign/"failure.json")["error"]):
                raise ValueError("Unexpected prior budget or insufficient remaining bounded buckets")
            if snapshot["sources"] != allowed["sources"]:
                raise ValueError("Original scientific source identity differs")
            previous = request_path.parent/"previous_segment"
            previous.mkdir()
            for name in ["budget.json", "execution_time.json", "cost_records.json", "failure.json"]:
                (previous/name).write_bytes((campaign/name).read_bytes())
            (previous/"dispatch.json").write_bytes((campaign/"training/dispatch.json").read_bytes())
            archives = []
            for row in failed:
                job = campaign/"training"/row["id"]
                if (job.is_symlink() or job.resolve() != job.absolute() or job.stat().st_uid != os.getuid()
                        or read(job/"dispatch_receipt.json") != row or (job/"run/completion.json").exists()):
                    raise ValueError("Interrupted job path/receipt/completeness differs")
                destination = campaign/"attempts"/row["id"]/"attempt0"
                if destination.exists():
                    raise ValueError("Interrupted attempt already archived")
                files = [record(p) for p in sorted(job.rglob("*")) if p.is_file()]
                destination.parent.mkdir(parents=True, exist_ok=True)
                job.rename(destination)
                for f in files:
                    new = record(destination/Path(f["path"]).relative_to(job))
                    if {**new, "path": f["path"]} != f:
                        raise RuntimeError("Interrupted attempt changed during move")
                archives.append(dict(task=row["id"], archive=str(destination), receipt=row, files=files))
                write(state/"attempt_resolution.json", dict(archives=archives, effective_queue=effective,
                      successes_retrained=False, prior_fees_retained=True))
            budget["caps"] = CAPS
            preparation = math.ceil(time.time()-request["preparation_started_epoch"])
            budget["used_seconds"]["auxiliary"] += preparation
            budget["events"].append(dict(id="budget_stop_recovery_preparation", bucket="auxiliary",
                  conservative_seconds=preparation, basis="Calendar envelope includes diagnosis/preparation; not CPU time"))
            update = dict(request=str(request_path), updated_sources=request["updated_sources"], review=request["review"])
            continued = dict(dispatch=effective, budget=budget, controller_update=update,
                prior_controller=dict(identity=request["prior_controller"], stop_reason="AUXILIARY_CAP_EXHAUSTED",
                                      evidence=str(previous)))
            write(state/"continuation.json", continued)
            # The prior failure remains byte-identical in previous_segment before retiring the current marker.
            (campaign/"failure.json").unlink()
            write(state/"controller.json", dict(identity=identity(), phase="RUNNING_CONTINUATION", request=str(request_path)))
            restore_beta(request, request_path, state)
            control.execute(Path(request["original_gate"]), continuation=continued,
                            dispatch_impl=dispatcher.dispatch)
            write(state/"completion.json", dict(status="CONTROLLED_CAMPAIGN_COMPLETE", epoch=time.time()))
        except BaseException as error:
            write(state/"failure.json", dict(error=repr(error), automatic_retry=False))
            if not (campaign/"failure.json").exists():
                write(campaign/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error)))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    execute(parser.parse_args().request)
