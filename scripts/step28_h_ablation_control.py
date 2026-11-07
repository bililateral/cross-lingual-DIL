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


def execute(gate_path):
    started = time.monotonic()
    allowed = run.gate(gate_path, "train")
    if set(allowed["actions"]) != set(run.ACTIONS):
        raise ValueError("Whole future chain must be explicitly authorized")
    predecessors(allowed)
    allowed["_gate_sha256"] = data.sha256(gate_path)
    campaign = Path(allowed["campaign"])
    if campaign.is_symlink() or not campaign.resolve().is_relative_to(data.ROOT):
        raise ValueError("Fresh independent campaign must be within its workspace")
    campaign.mkdir()
    ledger = Ledger(campaign, allowed["prior_used_seconds"], run.BUDGETS)
    ledger.tick = started
    monitor = None
    try:
        ledger.control()
        run.references(allowed)
        ledger.control()
        training = campaign/"training"
        training.mkdir()
        batch_path = training/"batch.json"
        specs = m.specifications()
        data.write_json(batch_path, dict(status="APPROVED_IMPLEMENTATION_AND_RESOURCES",
            sources=allowed["sources"], runs=[s.identity() for s in specs],
            allocator_bytes=run.LIMITS["allocator_bytes"], main_gate_sha256=data.sha256(gate_path)))
        monitor = queue.GpuMemory()
        rows = [dict(id=s.run_id, spec=s.identity(), bucket="training", seconds=21600) for s in specs]
        def command(row, job, deadline):
            data.write_json(job/"spec.json", row["spec"])
            return [sys.executable, str(data.ROOT/"scripts/step28_h_ablation_run.py"),
                "--job", str(job), "--spec", str(job/"spec.json"), "--deadline", str(deadline),
                "--gate", str(gate_path), "--batch", str(batch_path)]
        inherited.dispatch(training, rows, command,
            lambda row, job: run.validate_completion(job/"run", m.Spec.from_dict(row["spec"]), allowed),
            allowed, ledger, monitor=monitor, admit=admission)
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
            inherited.dispatch(campaign, [row], action_command, validate_action,
                local, ledger, monitor=monitor, admit=admission)
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
    data.write_json(campaign/"completion.json", dict(status="COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED",
        sources=allowed["sources"], used_seconds=ledger.used, new_trajectories=63, full_reused=9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    execute(parser.parse_args().gate)
