"""One claimed waiter for the approved finite H history search; no retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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
    if (path.stat().st_size != row["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
        raise ValueError("Waiter binding differs: "+str(path))
    return path


def predecessor(binding, complete):
    """Wait for an identified live predecessor, reject failures or missing receipts."""
    state = Path(binding["state"])
    process = read(verify(binding["process"]))
    for key in ("forever_gate", "vendor_gate"):
        verify(binding[key])
        if process[key] != binding[key]:
            raise ValueError("Pinned predecessor identity differs")
        campaign = Path(read(binding[key]["path"])["campaign"])
        if (campaign/"failure.json").exists():
            raise RuntimeError("Predecessor batch failed; no launch")
    if (state/"failure.json").exists():
        raise RuntimeError("Predecessor chain failed; no launch")
    gates = {binding[key]["path"] for key in ("forever_gate", "vendor_gate")}
    alive = []
    for p in psutil.process_iter(["pid", "username", "cmdline", "status"]):
        try:
            info = p.info
            if (p.pid != os.getpid() and info["username"] == process["owner"]
                    and info["status"] != psutil.STATUS_ZOMBIE
                    and gates.intersection(info["cmdline"] or [])):
                alive.append(info["pid"])
        except psutil.NoSuchProcess:
            pass
    if not (state/"completion.json").exists():
        if not alive:
            raise RuntimeError("Predecessor exited without complete chain receipt")
        return dict(ready=False, reason="PREDECESSOR_STILL_RUNNING", pids=alive)
    if alive:
        return dict(ready=False, reason="PREDECESSOR_EXIT_PENDING", pids=alive)
    complete(binding)  # Production checks both exact task sets, sources and full fees.
    return dict(ready=True, reason="WHOLE_CHAIN_COMPLETE_AND_EXITED", pids=[])


def resources(gate):
    # Use the real production admission rule, with the existing parent as the empty campaign.
    import step28_formal_queue as queue
    return queue.resources(gate, Path(gate["campaign"]).parent)


def poll(binding, gate, complete, available=resources):
    previous = predecessor(binding, complete)
    resource = available(gate) if previous["ready"] else None
    return dict(epoch=time.time(), predecessor=previous, resources=resource,
                status="READY" if resource and resource["ready"] else "WAITING")


def observe_update(campaign, listener, source, state):
    listener, campaign, state = Path(listener), Path(campaign), Path(state)
    if not listener.exists():
        return False
    batch_path = campaign/"trial_00/batch.json"
    if not batch_path.exists():
        return False
    batch = read(batch_path)
    if batch["sources"] is None:
        raise ValueError("Missing actual batch source binding")
    for log in sorted((campaign/"trial_00").glob("*/console.txt")):
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (event.get("event") == "updates" and event.get("stage") == 1
                    and event.get("completed", 0) > 0
                    and event.get("run_id") == log.parent.name):
                verify({**source, "path": str(listener)})
                write(state/"first_update.json", dict(epoch=time.time(), log=str(log), event=event,
                      listener_deleted_after_real_update=True))
                listener.unlink()
                return True
    return False


def execute(config_path):
    config = read(config_path)
    if config.get("approved") is not True or config.get("review_closed") is not True:
        raise ValueError("Waiter remains closed before delta review")
    state = Path(config["state"])
    state.mkdir()  # A second invocation cannot wait or launch again.
    process = psutil.Process()
    write(state/"process.json", dict(pid=process.pid, owner=process.username(), created=process.create_time(),
          command=process.cmdline(), cwd=process.cwd()))
    started = time.monotonic()
    used = float(config["prior_auxiliary_seconds"])
    pending_tick = True
    try:
        gate_path = verify(config["gate_template"])
        gate = read(gate_path)
        for row in config["listener_sources"]:
            verify(row)
        verify(config["review_evidence"])
        if (Path(__file__).resolve() != Path(config["listener"]).resolve()
                or gate["workspace"] != config["workspace"] or Path(gate["campaign"]).exists()
                or gate["prior_used_seconds"]["auxiliary"] != used):
            raise ValueError("Exact fresh waiter/workspace/fees required")
        os.chdir(config["workspace"])
        sys.path.insert(0, str(Path(config["workspace"])/"scripts"))
        import step28_history_optuna_control as control
        for row in gate["sources"]:
            verify({**row, "path": str(Path(gate["workspace"])/row["path"])})
        # Gate validation is closed-source/metadata-only and precedes any formal reads.
        control.run.gate(gate_path, "train")
        while True:
            try:
                observation = poll(gate["predecessor"], gate, control.chain_complete)
                write(state/"waiting.json", observation)
            finally:
                used += time.monotonic()-started
                pending_tick = False
                write(state/"budget.json", dict(auxiliary_used_seconds=used, limit=14400,
                      idle_sleep_charged=False))
                if not math.isfinite(used) or used >= 14400-150:
                    raise TimeoutError("Waiter auxiliary allowance exhausted")
            if observation["status"] == "READY":
                break
            time.sleep(60)  # Idle waiting is excluded; formal-controller accounting is unchanged.
            started = time.monotonic()
            pending_tick = True
        # All waiting/source/import work is carried into the one existing formal ledger.
        gate["prior_used_seconds"]["auxiliary"] = used
        activated = state/"formal_gate.json"
        write(activated, gate)
        write(state/"start.json", dict(epoch=time.time(), gate=str(activated), pid=os.getpid(),
              prior_auxiliary_seconds=used))
        source = next(r for r in config["listener_sources"] if r["path"] == config["listener"])
        control.execute(activated, observe=lambda: observe_update(gate["campaign"], config["listener"], source, state))
        write(state/"completion.json", dict(status="FORMAL_CONTROLLER_COMPLETED", epoch=time.time()))
    except BaseException as error:
        if pending_tick:
            used += time.monotonic()-started
            write(state/"budget.json", dict(auxiliary_used_seconds=used, limit=14400,
                  idle_sleep_charged=False))
        write(state/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error),
              auxiliary_used_before_formal=used, epoch=time.time()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    execute(parser.parse_args().config.resolve())
