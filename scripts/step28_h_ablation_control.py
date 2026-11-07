"""One fixed 63-run dispatch; failed slots retire, no automatic launch or retry."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import step28_h_ablation_run as run
import step28_history_optuna_control as inherited
from step28_formal_wait import relevant_processes

m, data, queue = run.m, run.data, run.queue


class Ledger(inherited.Ledger):
    def pause(self, seconds, *, active):
        self.control()
        start = time.monotonic()
        time.sleep(seconds)
        # Both polling and resource-wait sleep are excluded; active checks remain charged.
        self.tick += time.monotonic()-start
        self.control()


def admission(allowed, snapshot, capacity, occupied):
    result = inherited.slot_admission(allowed, snapshot, capacity, occupied)
    result["required_available_ram_bytes"] = run.LIMITS["available_ram_bytes"]
    result["ready"] = result["ready"] and result["free_ram_bytes"] >= run.LIMITS["available_ram_bytes"]
    return result


class ObservedGpuMemory:
    """Retain sampled NVML peaks for owned workers, using dispatch's observations."""
    def __init__(self, inner, root, *, resume=False):
        self.inner, self.path, self.peaks = inner, root/"gpu_observed_peaks.json", {}
        if resume:
            self.peaks = data.read_json(self.path)["by_process"]
        self.save()

    def save(self):
        data.write_json(self.path, dict(kind="OWNED_PROCESS_PERIODIC_NVML_OBSERVED_PEAKS",
            continuous_lifetime_maximum=False, poll_sleep_seconds=2,
            sampling="Existing dispatch loop and additional admission snapshots; not continuous",
            by_process=self.peaks))

    def snapshot(self):
        snapshot = self.inner.snapshot()
        changed = False
        for process in inherited.psutil.Process().children():
            try:
                if process.pid not in snapshot["processes"]:
                    continue
                created = process.create_time()
                key = f"{process.pid}@{created}"
                value = snapshot["processes"][process.pid]
                previous = self.peaks.get(key)
                if previous is None or value > previous["observed_peak_gpu_bytes"]:
                    self.peaks[key] = dict(pid=process.pid, created=created, observed_peak_gpu_bytes=value)
                    changed = True
            except inherited.psutil.NoSuchProcess:
                pass
        if changed:
            self.save()  # Persist new peaks also when a later task/monitor fails.
        return snapshot

    def close(self):
        try:
            self.inner.close()
        finally:
            root, rows = self.path.parent, []
            paths = list(root.glob("training/*/process.json"))
            paths += [root/name/"process.json" for name in ("generation", "confirmation", "analysis")]
            for path in sorted(p for p in paths if p.exists()):
                process = data.read_json(path)
                key = f"{process['pid']}@{process['created']}"
                peak = self.peaks.get(key)
                receipt_path = path.parent/"dispatch_receipt.json"
                receipt = data.read_json(receipt_path) if receipt_path.exists() else None
                rows.append(dict(task=str(path.parent.relative_to(root)), process=process,
                    dispatch_receipt=receipt,
                    observed_peak_gpu_bytes=None if peak is None else peak["observed_peak_gpu_bytes"],
                    gpu_observation_status="NOT_OBSERVED" if peak is None else "OBSERVED"))
            data.write_json(root/"cost_records.json", dict(tasks=rows,
                gpu_measurement="Periodic NVML observed maximum process usage; missing is null, not zero",
                poll_sleep_seconds=2, source="gpu_observed_peaks.json",
                continuous_lifetime_maximum=False))


def predecessors(allowed):
    rows = allowed["predecessors"]
    if {r["name"] for r in rows} != {"H_pos_history", "english_abc"} or len(rows) != 2:
        raise ValueError("Both existing whole predecessors must be bound")
    for item in rows:
        path = queue.verify(item["completion"])
        row = data.read_json(path)
        statuses = ({"COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED"} if item["name"] == "H_pos_history"
                    else {"STOP_PRECHECK_NO_VENDOR_DREAM_FORGETTING", "COMPLETE_PENDING_RESULT_REVIEW"})
        if row["status"] not in statuses or (path.parent/"failure.json").exists():
            raise ValueError("Predecessor incomplete or invalid")
        if relevant_processes(Path(item["gate_path"]), path.parent, item["owner"]):
            raise ValueError("Predecessor process has not exited")


def execute(gate_path, *, continuation=None, dispatch_impl=None, dispatch_paused=None):
    started = time.monotonic()
    started_epoch = time.time()
    allowed = run.gate(gate_path, "train")
    if set(allowed["actions"]) != set(run.ACTIONS):
        raise ValueError("Whole future chain must be explicitly authorized")
    predecessors(allowed)
    allowed["_gate_sha256"] = data.sha256(gate_path)
    campaign = Path(allowed["campaign"])
    if campaign.is_symlink() or not campaign.resolve().is_relative_to(data.ROOT):
        raise ValueError("Fresh independent campaign must be within its workspace")
    campaign.mkdir(exist_ok=continuation is not None)
    if continuation is None:
        ledger = Ledger(campaign, allowed["prior_used_seconds"], run.BUDGETS)
    else:
        budget = continuation["budget"]
        if (budget["caps"] != run.BUDGETS or (campaign/"failure.json").exists()
                or (campaign/"completion.json").exists()):
            raise ValueError("Invalid budget or already closed campaign")
        ledger = Ledger.__new__(Ledger)
        ledger.root, ledger.caps = campaign, dict(run.BUDGETS)
        ledger.used, ledger.events = dict(budget["used_seconds"]), list(budget["events"])
        ledger.check()  # Preserve the existing final-tail charge, failures and all used fees.
    ledger.tick = started
    dispatch_impl = dispatch_impl or inherited.dispatch
    if dispatch_paused is None:
        dispatch_paused = lambda: (campaign/"pause_dispatch").exists()
    monitor = None
    try:
        ledger.control()
        run.references(allowed)
        ledger.control()
        training = campaign/"training"
        training.mkdir(exist_ok=continuation is not None)
        batch_path = training/"batch.json"
        specs = m.specifications()
        batch = dict(status="APPROVED_IMPLEMENTATION_AND_RESOURCES",
            sources=allowed["sources"], runs=[s.identity() for s in specs],
            allocator_bytes=run.LIMITS["allocator_bytes"], main_gate_sha256=data.sha256(gate_path))
        if continuation is None:
            data.write_json(batch_path, batch)
        elif data.read_json(batch_path) != batch:
            raise ValueError("The original fixed batch must remain byte-identical")
        monitor = ObservedGpuMemory(queue.GpuMemory(), campaign, resume=continuation is not None)
        rows = [dict(id=s.run_id, spec=s.identity(), bucket="training", seconds=21600) for s in specs]
        def command(row, job, deadline):
            data.write_json(job/"spec.json", row["spec"])
            return [sys.executable, str(data.ROOT/"scripts/step28_h_ablation_run.py"),
                "--job", str(job), "--spec", str(job/"spec.json"), "--deadline", str(deadline),
                "--gate", str(gate_path), "--batch", str(batch_path)]
        dispatch_impl(training, rows, command,
            lambda row, job: run.validate_completion(job/"run", m.Spec.from_dict(row["spec"]), allowed),
            allowed, ledger, monitor=monitor, admit=admission, dispatch_paused=dispatch_paused,
            continuation=None if continuation is None else continuation["dispatch"])
        # One dispatch above owns all 63 trajectories, so retirement spans the entire list.
        for action, name in (("generate", "generation"), ("confirmation", "confirmation"), ("analyze", "analysis")):
            bucket = "confirmation" if action == "confirmation" else "auxiliary"
            seconds = ledger.caps[bucket]-ledger.used[bucket]-120
            if seconds <= 60:
                raise TimeoutError("Insufficient followup allowance")
            index = campaign/(name+"_dispatch")
            index.mkdir()
            local = {**allowed, "dispatch_index": str(index)}
            row = dict(id=name, bucket=bucket, seconds=seconds, gpu=action == "confirmation")
            def action_command(row, job, deadline):
                return [sys.executable, str(data.ROOT/"scripts/step28_h_ablation_evaluate.py"), action,
                    "--gate", str(gate_path), "--deadline", str(deadline), "--batch", str(batch_path), "--job", str(job)]
            def validate_action(row, job):
                result = data.read_json(job/"result.json")
                expected = dict(generate="GENERATED_VALIDATED_OWNERS_SEALED",
                    confirmation="ALL_ABLATION_CONFIRMATION_METRICS_COMPLETE", analyze="ANALYSIS_COMPLETE")[action]
                if result.get("status") != expected:
                    raise ValueError("Incomplete followup result")
            dispatch_impl(campaign, [row], action_command, validate_action,
                local, ledger, monitor=monitor, admit=admission, dispatch_paused=dispatch_paused)
        ledger.control()
    except BaseException as error:
        data.write_json(campaign/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error), used_seconds=ledger.used))
        raise
    finally:
        try:
            inherited.close_and_settle(monitor, ledger)
        except BaseException as error:
            path = campaign/"failure.json"
            previous = data.read_json(path) if path.exists() else {}
            data.write_json(path, {**previous, "cleanup_error": repr(error), "used_seconds": ledger.used})
            raise
        finally:
            data.write_json(campaign/"execution_time.json", dict(started_epoch=started_epoch,
                finished_epoch=time.time(), elapsed_calendar_seconds=time.monotonic()-started,
                prior_controller=None if continuation is None else continuation["prior_controller"],
                meaning="This controller segment including waits and cleanup; prior segment is separate after handover; distinct from cumulative task fees"))
    data.write_json(campaign/"completion.json", dict(status="COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED",
        sources=allowed["sources"], used_seconds=ledger.used, new_trajectories=63, full_reused=9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    execute(parser.parse_args().gate)
