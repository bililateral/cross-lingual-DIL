"""Three owned workers per candidate; no retry, resume, or live launch by default."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

import step28_history_optuna_run as run
import step28_history_optuna_evaluate as evaluate

m, data, queue = run.m, run.data, run.queue
FINAL_TAIL_SECONDS = 30


def chain_complete(binding):
    from step28_formal_wait import relevant_processes, valid_budget
    state = Path(binding["state"]).resolve()
    process = data.read_json(queue.verify(binding["process"]))
    if (Path(binding["process"]["path"]).resolve() != state/"process.json"
            or (state/"failure.json").exists() or not (state/"completion.json").exists()
            or data.read_json(state/"completion.json")["status"] != "BOTH_FORMAL_BATCHES_COMPLETE"):
        raise ValueError("Existing whole FOREVER/VendorLink chain must have completed")
    for key, experiment, extra in (("forever_gate", "FOREVER", {"generation", "confirmation", "analysis"}),
                                   ("vendor_gate", "VendorLink", {"scoring", "analysis"})):
        gate_path = queue.verify(binding[key])
        gate = data.read_json(gate_path)
        if process[key] != binding[key] or gate["experiment"] != experiment:
            raise ValueError("Existing chain identity differs")
        campaign = Path(gate["campaign"])
        if relevant_processes(gate_path, campaign, process["owner"]):
            raise ValueError("Existing chain process/worker has not exited")
        result = data.read_json(campaign/"completion.json")
        expected = ({f'{r["order"]}_s{r["seed"]}_{r["name"]}' for r in gate["runs"]}
                    if experiment == "FOREVER" else {r["id"] for r in gate["runs"]}) | extra
        if (result["status"] != "ALL_FORMAL_ACTIONS_COMPLETE" or result["sources"] != gate["sources"]
                or len(expected) != (12 if experiment == "FOREVER" else 38) or (campaign/"failure.json").exists()):
            raise ValueError("Prior full batch completion differs")
        valid_budget(result, expected)


def group_members(pgid):
    result = []
    if pgid == os.getpgrp():
        raise ValueError("Refuse to manage the controller process group")
    for process in psutil.process_iter(["pid", "status", "username"]):
        try:
            if os.getpgid(process.pid) == pgid and process.status() != psutil.STATUS_ZOMBIE:
                if process.username() != psutil.Process().username():
                    raise RuntimeError("Foreign process in an owned worker group")
                result.append(process.pid)
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
    return result


def release(process):
    """A parent's exit does not prove that its descendants have stopped."""
    process.poll()
    unexpected = process.returncode is not None and bool(group_members(process.pid))
    for sig, seconds in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        if not group_members(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic()+seconds
        while group_members(process.pid) and time.monotonic() < end:
            time.sleep(.02)
    process.wait(timeout=1)
    if group_members(process.pid):
        raise RuntimeError("Owned worker process group has not been released")
    return unexpected


class Ledger:
    def __init__(self, root, prior_used, caps):
        self.root, self.caps = root, caps
        self.used = dict(prior_used)
        if set(self.used) != set(caps) or any(v < 0 or v >= caps[k] for k, v in self.used.items()):
            raise ValueError("Finite cumulative prior fees required")
        self.tick = time.monotonic()
        # One finite conservative charge covers the final save/check/receipt/exit.
        # It is inside the existing auxiliary cap, never an additional allowance.
        self.used["auxiliary"] += FINAL_TAIL_SECONDS
        self.events = [dict(id="final_tail", bucket="auxiliary",
            conservative_seconds=FINAL_TAIL_SECONDS)]

    def control(self):
        now = time.monotonic()
        self.used["auxiliary"] += now-self.tick
        self.tick = now
        self.save()
        self.check()

    def save(self):
        data.write_json(self.root/"budget.json", dict(used_seconds=self.used, caps=self.caps, events=self.events))

    def check(self):
        if any(self.used[k] >= self.caps[k] for k in self.used):
            raise TimeoutError("Cumulative allowance exhausted; failure fees remain")

    def pause(self, seconds, *, active):
        self.control()
        started = time.monotonic()
        time.sleep(seconds)
        if active:
            # Child wall time is charged by its bucket; exclude polling sleep only.
            self.tick += time.monotonic()-started
        self.control()

    def finish(self, bucket, identity, started, released):
        self.used[bucket] += released-started
        self.events.append(dict(id=identity, bucket=bucket, actual_seconds=released-started))
        self.save()


def close_and_settle(monitor, ledger):
    try:
        if monitor is not None:
            monitor.close()
    finally:
        ledger.control()


def slot_admission(allowed, snapshot, capacity, occupied):
    # Capacity is frozen only while all owned process groups are released.
    # Never refund slot credit using non-atomic per-PID usage observations.
    slot_bytes = run.LIMITS["process_gpu_bytes"]
    observation = queue.resources(allowed, Path(allowed["campaign"]), snapshot, ())
    reserved = occupied*slot_bytes
    return dict(**{k: v for k, v in observation.items() if k != "ready"},
        cohort_capacity_bytes=capacity, reserved_slot_bytes=reserved,
        ready=observation["ready"] and capacity-reserved >= slot_bytes)


def dispatch(root, rows, command, validate, allowed, ledger, *, monitor=None, poll_seconds=2, observe=None,
             admit=slot_admission):
    """Failed slots retire; healthy slots continue to claim pending trajectories."""
    active, pending, retired, receipts = {}, list(enumerate(rows)), set(), []
    capacity = None
    width = allowed["workers"] if rows[0]["bucket"] == "training" else 1
    cores = psutil.Process().cpu_affinity()
    if len(cores) < width:
        raise RuntimeError("Three distinct permitted CPU cores required")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    def saved(status):
        value = dict(status=status, sources=allowed["sources"], completed=receipts,
            retired_slots=sorted(retired), pending=[r["id"] for _, r in pending],
            active={str(k): dict(id=v["row"]["id"], pid=v["process"].pid) for k, v in active.items()})
        data.write_json(Path(allowed.get("dispatch_index", root))/"dispatch.json", value)
        return value
    def finish(slot, error=None):
        item = active[slot]
        process, row = item["process"], item["row"]
        unexpected = release(process)
        ended = time.monotonic()  # Before validation: charge validation only to auxiliary.
        item["log"].close()
        ledger.finish(row["bucket"], row["id"], item["started"], ended)
        active.pop(slot)
        if process.returncode or unexpected:
            error = error or f"Worker exit={process.returncode}; unexpected descendants={unexpected}"
        if ended-item["started"] > row["seconds"]:
            error = error or "Actual task wall time exceeds its cap"
        if error is None:
            try:
                validate(row, item["job"])
            except Exception as exc:
                error = repr(exc)
        receipt = dict(id=row["id"], index=item["index"], slot=slot, success=error is None, error=error,
            released=True, actual_seconds=ended-item["started"], bucket=row["bucket"])
        receipts.append(receipt)
        data.write_json(item["job"]/"dispatch_receipt.json", receipt)
        if error:
            retired.add(slot)
        ledger.control()
    try:
        while pending or active:
            if observe is not None:
                observe()
            ledger.control()
            queue.check_running(None, allowed, Path(allowed["campaign"]))
            snapshot = monitor.snapshot() if monitor else None
            for slot, item in list(active.items()):
                process, row = item["process"], item["row"]
                error = None
                if snapshot and snapshot["processes"].get(process.pid, 0) > run.LIMITS["process_gpu_bytes"]:
                    error = "PROCESS_GPU_LIMIT_EXCEEDED"
                elif time.monotonic() >= item["started"]+row["seconds"]-row.get("shutdown_reserve", 60):
                    error = "SHUTDOWN_RESERVE_REACHED"
                if error or process.poll() is not None:
                    finish(slot, error)
            for slot in range(width):
                if not pending or slot in active or slot in retired:
                    continue
                index, row = pending[0]
                bucket = row["bucket"]
                reserved = sum(item["row"]["seconds"] for item in active.values() if item["row"]["bucket"] == bucket)
                if ledger.used[bucket]+reserved+row["seconds"] > ledger.caps[bucket]:
                    raise TimeoutError("Remaining allowance cannot reserve the complete next task")
                if row.get("gpu", True):
                    if monitor is None:
                        raise ValueError("Real GPU admission observation required")
                    current = monitor.snapshot()
                    if not active:
                        capacity = current["free"]
                    occupied = sum(item["row"].get("gpu", True) for item in active.values())
                    observation = admit(allowed, current, capacity, occupied)
                    data.write_json(root/"admission.json", dict(id=row["id"], **observation))
                    if not observation["ready"]:
                        break
                ledger.control()
                job = root/row["id"]
                job.mkdir()
                begin = time.monotonic()
                log = (job/"console.txt").open("x", encoding="utf-8")
                try:
                    argv = ["taskset", "-c", str(cores[slot]), *command(row, job, begin+row["seconds"])]
                    child_env = {**env, "CUDA_VISIBLE_DEVICES": "0" if row.get("gpu", True) else ""}
                    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=child_env,
                        cwd=allowed["workspace"], start_new_session=True)
                except BaseException:
                    log.close()
                    ledger.finish(bucket, row["id"], begin, time.monotonic())
                    raise
                active[slot] = dict(process=process, row=row, job=job, started=begin, log=log, index=index)
                child = psutil.Process(process.pid)
                data.write_json(job/"process.json", dict(pid=process.pid, created=child.create_time(),
                    owner=child.username(), command=argv, cwd=allowed["workspace"], slot=slot))
                pending.pop(0)
            saved("RUNNING")
            if pending and not active and len(retired) == width:
                raise RuntimeError("All worker slots retired; no retry")
            if pending or active:
                ledger.pause(poll_seconds, active=bool(active))
        complete = len(receipts) == len(rows) and all(r["success"] for r in receipts)
        result = saved("ALL_COMPLETE_RELEASED" if complete else "INCOMPLETE_NO_DOWNSTREAM")
        if not complete:
            raise RuntimeError("Incomplete batch; no Optuna feedback, selection or confirmation")
        return result
    except BaseException:
        for slot in list(active):
            try:
                finish(slot, "CONTROLLER_STOP_NO_RETRY")
            except Exception:
                # Keep attempting cleanup of the other owned groups, then retain failure.
                continue
        saved("STOP_NO_RETRY")
        raise


def batch(campaign, specs, gate_path, allowed):
    root = campaign/(f"trial_{specs[0].number:02}" if specs[0].phase == "search" else "replication")
    root.mkdir()
    path = root/"batch.json"
    data.write_json(path, dict(status="APPROVED_IMPLEMENTATION_AND_RESOURCES", sources=allowed["sources"],
        runs=[s.identity() for s in specs], allocator_bytes=run.LIMITS["allocator_bytes"], main_gate_sha256=data.sha256(gate_path)))
    return path


def retire_search_weights(batch_path, specs, allowed):
    # Search scores/metrics/logs remain; search models are not replication starting points.
    targets = []
    for spec in specs:
        root = batch_path.parent/spec.run_id/"run"
        row = run.validate_completion(root, spec, allowed)
        for stage, point in row["points"].items():
            path = root/point["model"]["path"]
            if (path.resolve() != (root/"models"/f"stage{stage}.pt").resolve()
                    or path.is_symlink() or path.stat().st_nlink != 1 or path.stat().st_uid != os.getuid()):
                raise ValueError("Only owned, unshared search inference files may retire")
            targets.append(queue.record(path))
    data.write_json(batch_path.parent/"weight_retirement_manifest.json", dict(files=targets))
    deleted = []
    for row in targets:
        path = queue.verify(row)
        path.unlink()
        deleted.append(dict(**row, absent=not path.exists()))
        data.write_json(batch_path.parent/"weight_retirement_receipt.json", dict(files=deleted))


def execute(gate_path, *, observe=None):
    started = time.monotonic()
    allowed = run.gate(gate_path, "train")
    if set(allowed["actions"]) != set(run.ACTIONS) or allowed.get("retire_search_weights") is not True:
        raise ValueError("Whole finite chain and conditional search-model retirement must be approved")
    # No listener: predecessor must already be complete and all its pinned processes exited.
    chain_complete(allowed["predecessor"])
    allowed["_gate_sha256"] = data.sha256(gate_path)
    campaign = Path(allowed["campaign"])
    if campaign.is_symlink() or not campaign.resolve().is_relative_to(data.ROOT):
        raise ValueError("Fresh independent campaign must be inside its workspace")
    campaign.mkdir()
    ledger = Ledger(campaign, allowed["prior_used_seconds"], run.BUDGETS)
    ledger.tick = started
    monitor, search = None, None
    def train_batch(path, specs):
        def command(row, job, deadline):
            data.write_json(job/"spec.json", row["spec"])
            return [sys.executable, str(data.ROOT/"scripts/step28_history_optuna_run.py"), "--job", str(job),
                "--spec", str(job/"spec.json"), "--deadline", str(deadline), "--gate", str(gate_path), "--batch", str(path)]
        rows = [dict(id=s.run_id, spec=s.identity(), bucket="training", seconds=21600) for s in specs]
        dispatch(path.parent, rows, command, lambda row, job: run.validate_completion(job/"run", m.Spec.from_dict(row["spec"]), allowed), allowed, ledger, monitor=monitor, observe=observe)
    def action(name, path):
        bucket = "confirmation" if name == "confirmation" else "auxiliary"
        # Auxiliary actions share the remaining allowance, including controller overhead.
        seconds = ledger.caps[bucket]-ledger.used[bucket]-120
        if seconds <= 60:
            raise TimeoutError("Insufficient followup allowance")
        action_root = path.parent if name.endswith("evaluate") else campaign
        job_name = "evaluation" if name.endswith("evaluate") else {"generate": "generation", "confirmation": "confirmation", "analyze": "analysis"}[name]
        rows = [dict(id=job_name, bucket=bucket, seconds=seconds, gpu=name == "confirmation")]
        def command(row, job, deadline):
            return [sys.executable, str(data.ROOT/"scripts/step28_history_optuna_evaluate.py"), name, "--gate", str(gate_path),
                "--deadline", str(deadline), "--batch", str(path), "--job", str(job)]
        # A separate dispatch index must not overwrite the training batch's completion gate.
        action_container = action_root/(job_name+"_dispatch")
        action_container.mkdir()
        local_allowed = {**allowed, "dispatch_index": str(action_container)}
        dispatch(action_root, rows, command, lambda row, job: data.read_json(job/("feedback.json" if name == "search-evaluate" else "result.json")), local_allowed, ledger, monitor=monitor)
    try:
        ledger.control()  # Enforce the tail reservation before reading old inputs.
        run.references(allowed)  # Full old weight/restore/source check before any new training.
        ledger.control()
        monitor = queue.GpuMemory()
        search = m.Search(storage="sqlite:///"+str(campaign/"study.sqlite3"))
        for _ in range(m.TRIALS):
            specs = search.ask()
            path = batch(campaign, specs, gate_path, allowed)
            train_batch(path, specs)
            action("search-evaluate", path)
            feedback = evaluate.verified_feedback(path.parent/"evaluation", path, allowed)
            if feedback != data.read_json(path.parent/"evaluation/feedback.json"):
                raise ValueError("Saved feedback differs from actual RAW final MAP")
            search.tell(feedback)
            data.write_json(campaign/"trials.json", dict(rows=search.rows))
            retire_search_weights(path, specs, allowed)
            ledger.control()
        # Recompute every feedback again before freezing the winner; weights are no longer needed.
        for row in search.rows:
            path = campaign/f'trial_{row["number"]:02}/batch.json'
            feedback = evaluate.verified_feedback(path.parent/"evaluation", path, allowed)
            if (row["feedback"] != feedback
                    or row["value"] != sum(feedback["final_map_by_order"][o] for o in m.ORDERS)/3):
                raise ValueError("Selection is not bound to actual completed development matrices")
        selected = search.selected()
        data.write_json(campaign/"selected.json", selected)
        if selected["parameters"] == m.ANCHOR:
            status = "COMPLETE_ANCHOR_SELECTED_NO_NEW_CONFIRMATION"
        else:
            specs = m.specifications("replication", selected["number"], selected["parameters"])
            path = batch(campaign, specs, gate_path, allowed)
            train_batch(path, specs)
            for name in ("replication-evaluate", "generate", "confirmation", "analyze"):
                action(name, path)
            status = "COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED"
        ledger.control()
    except BaseException as exc:
        if search is not None:
            search.fail(repr(exc))
        data.write_json(campaign/"failure.json", dict(status="STOP_NO_RETRY", error=repr(exc), used_seconds=ledger.used))
        raise
    finally:
        try:
            close_and_settle(monitor, ledger)
        except BaseException as exc:
            path = campaign/"failure.json"
            previous = data.read_json(path) if path.exists() else {}
            data.write_json(path, dict(**{k: v for k, v in previous.items() if k != "used_seconds"},
                cleanup_error=repr(exc), used_seconds=ledger.used))
            raise
    # A success receipt is never visible before final cleanup and accounting pass.
    data.write_json(campaign/"completion.json", dict(status=status, selected=selected,
        sources=allowed["sources"], used_seconds=ledger.used))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    execute(parser.parse_args().gate)
