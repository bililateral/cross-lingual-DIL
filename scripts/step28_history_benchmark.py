"""Chinese trial-3 H versus five fixed baselines on a single prospective cohort."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import step28_continual_five_run as five_run
import step28_continual_comparison_run as comparison_run
import step28_forever_formal as forever_run
import step28_formal_queue as queue
import step28_history_optuna as history

five, comparison, forever = five_run.m, comparison_run.m, forever_run.m
data = five.data
H = "H_trial3"
NAMES = (H, "ER0.1_ref", "LOGIT0.1_ref", *comparison.NEW_METHODS, forever.NAME)
ORDERS, SEEDS = ("ABC", "BCA", "CAB"), (4, 5, 6)
ALPHA, BETA = .07049361507739538, .767438161798796
NAMESPACE, CONFIRM_SEED = "trial3_six_confirmation_20261118", 2026111804
EXPERIMENT = "chinese_trial3_six"
TRAIN_CAL_COMPLETE = "COMPLETE_864_UPDATES_3_ENDPOINTS_TRAIN_CAL_ONLY"
POST = ("generation", "confirmation", "analysis")
SCRIPTS = Path(__file__).resolve().parent
SCHEMAS = ("step28_chinese_base_policy", "step28_alias_ranking_policy", "step28_record_replay_policy",
           "step28_bge_continual_policy", "step28_continual_expression_policy", "step28_continual_expression_catalog")


def native(name, order, seed):
    if name == H:
        return history.Spec("replication", 3, order, seed, ALPHA, BETA)
    if name in comparison.NEW_METHODS:
        return comparison.Spec(name, order, seed)
    if name == forever.NAME:
        return forever.Spec(order, seed)
    if name in NAMES:
        return five.Spec(name, order, seed)
    raise ValueError("Unknown benchmark arm: " + name)


def plan():
    return [dict(id=f"{order}_s{seed}_{name}", name=name, order=order, seed=seed,
                 native=native(name, order, seed).identity())
            for order in ORDERS for seed in SEEDS for name in NAMES]


def cohort():
    return dict(namespace=NAMESPACE, root_seed=CONFIRM_SEED, domains=list("ABC"),
                groups_per_domain=40, trajectories=54, endpoints=162, cells=324, group_forwards=12960)


def gradient_stopping(row):
    # Only this historical success used the repaired L-BFGS-B termination setting.
    return row["id"] == "CAB_s6_L2R_text_fixed"


def source_paths():
    """Resolve project imports from the actual entry and evaluator, without importing data."""
    paths = set()
    pending = [Path(__file__).resolve(), SCRIPTS/"step28_history_benchmark_evaluate.py"]
    while pending:
        path = pending.pop()
        if path in paths:
            continue
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            pending.extend(SCRIPTS/(name+".py") for name in names if name.startswith("step28_"))
    paths.update(SCRIPTS.parent/"schema"/(name+".json") for name in SCHEMAS)
    return sorted(paths)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_gate(path):
    allowed = data.read_json(path)
    if (sys.platform != "linux" or allowed.get("experiment") != EXPERIMENT or
            allowed.get("approved") is not True or allowed.get("review_closed") is not True or
            allowed.get("runs") != plan() or allowed.get("cohort") != cohort() or
            allowed.get("actions") != ["training", *POST] or allowed.get("listener") or
            allowed.get("predecessor") or allowed.get("automatic_retry")):
        raise ValueError("A separate reviewed and explicitly authorized six-method gate is required")
    workspace = SCRIPTS.parent
    campaign = Path(allowed["campaign"]).resolve()
    if (Path(allowed["workspace"]).resolve() != workspace or campaign != path.resolve().parent/"campaign" or
            campaign.parent/"workspace" != workspace or allowed["sources"] !=
            [queue.record(p) for p in source_paths()]):
        raise ValueError("Fresh campaign path or exact deployed sources differ")
    approved = data.read_json(queue.verify(allowed["authorization"]))
    fields = ("experiment", "runs", "cohort", "actions", "sources", "limits", "workers", "seconds_per_method",
              "budget_seconds", "prior_used_seconds", "post_seconds", "worker_failure_policy", "analysis_plan")
    if approved.get("authorize_execution") is not True or any(approved[k] != allowed[k] for k in fields):
        raise ValueError("Execution scope or resource allocations differ from the decision")
    queue.verify(allowed["analysis_plan"])
    if (type(allowed["workers"]) is not int or allowed["workers"] < 1 or
            allowed["worker_failure_policy"] not in ("stop_batch", "retire_failed_slot_continue_healthy") or
            set(allowed["seconds_per_method"]) != set(NAMES) or set(allowed["post_seconds"]) != set(POST) or
            set(allowed["budget_seconds"]) != {"training", "control", *POST} or
            set(allowed["prior_used_seconds"]) != set(allowed["budget_seconds"]) or
            not all(positive(v) and v > 60 for k in ("seconds_per_method", "post_seconds")
                    for v in allowed[k].values()) or not all(positive(v) for v in allowed["budget_seconds"].values()) or
            not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= allowed["budget_seconds"][k]
                    for k, v in allowed["prior_used_seconds"].items())):
        raise ValueError("Incomplete independent task, controller or cumulative budget")
    for key in ("rss_bytes", "disk_bytes", "small_bytes", "free_reserve_bytes",
                "free_gpu_bytes", "allocator_bytes", "process_gpu_bytes"):
        if type(allowed["limits"].get(key)) is not int or allowed["limits"][key] <= 0:
            raise ValueError("Missing positive resource allocation: " + key)
    return allowed


class Budget(forever_run.Budget):
    snapshot = forever_run.Budget.summary

    @contextmanager
    def saving(self, reserve):
        with (self.root/"checkpoint.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.check(reserve)
            yield


def train(job, row, deadline, allowed):
    name = row["name"]
    spec = native(name, row["order"], row["seed"])
    if name == forever.NAME:
        return forever_run.execute(job, spec, deadline, allowed)
    budget = Budget(Path(allowed["campaign"]), deadline, allowed["limits"])
    if name in comparison.NEW_METHODS:
        return comparison_run.train_trajectory(job, spec, allowed, budget,
                                               calibration_stop_on_gradient=gradient_stopping(row))
    runtime = history if name == H else five
    return five_run.train_trajectory(job, spec, {**allowed, "allocator_bytes": allowed["limits"]["allocator_bytes"]},
                                     runtime, budget, train_inputs=comparison_run.training_inputs,
                                     completion_status=TRAIN_CAL_COMPLETE)


def validate_training(job, row, allowed, *, verify_models=True):
    root = job/"run"
    saved = data.read_json(root/"completion.json")
    name = row["name"]
    if name == forever.NAME and verify_models:
        forever_run.validate_completion(native(name, row["order"], row["seed"]), root, allowed)
    status = (forever_run.COMPLETE if name == forever.NAME else
              comparison_run.COMPLETE if name in comparison.NEW_METHODS else TRAIN_CAL_COMPLETE)
    count = saved["counts"]["current"] if name == forever.NAME else saved["updates"]
    if (saved["status"] != status or count != 864 or saved["spec"] != row["native"] or
            saved["source_files"] != allowed["sources"] or set(saved["points"]) != {"1", "2", "3"} or
            saved["access"] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
        raise ValueError("Training identity, access or 864 real current updates differ: " + row["id"])
    for stage in (1, 2, 3):
        point = saved["points"][str(stage)]
        domain = row["order"][stage-1]
        expected = [x["group_uid"] for x in saved["partition"]["calibration"] if x["domain"] == domain]
        if (point["stage"] != stage or not point["inference_restore_verified"] or len(expected) != 12 or
                point["calibration_ids"] != expected or point.get("development_ids") or
                (name != forever.NAME and (not point["full_restore_verified"] or point["adam_step"] != 288*stage))):
            raise ValueError("Stage restoration/current calibration differs: " + row["id"])
        mapping = point["calibration" if name == forever.NAME else "mapping"]
        actual = data.read_json(data.verify(root/mapping["path"], mapping))
        expected_ftol = 0.0 if gradient_stopping(row) else five.parent.calibration.OPTIONS["ftol"]
        if (actual["status"] != "PASS_CALIBRATION_FIT" or
                actual["options"]["ftol"] != expected_ftol or
                dict(zip(("a", "b"), five.parent.calibration.parameters(actual))) != point["maps"]["stage"] or
                point["maps"]["first"] != saved["points"]["1"]["maps"]["stage"]):
            raise ValueError("Saved stage/first calibration differs")
        if name != forever.NAME and verify_models:
            data.verify(root/point["model"]["path"], point["model"])
    return saved


def completed(campaign, allowed, *, verify_models=True, check=lambda: None):
    saved = data.read_json(campaign/"training_complete.json")
    rows = plan()
    receipts = saved["completed"]
    if (saved["status"] != "ALL_TRAINING_COMPLETE_EVALUATION_CLOSED" or saved["sources"] != allowed["sources"] or
            len(receipts) != 54 or {r["id"] for r in receipts} != {r["id"] for r in rows} or
            not all(r["success"] is True and r["bucket"] == "training" for r in receipts)):
        raise ValueError("All 54 successful training dispatch receipts are required")
    records = []
    for row in rows:
        check()
        job = campaign/row["id"]
        if data.read_json(job/"dispatch_receipt.json") != next(r for r in receipts if r["id"] == row["id"]):
            raise ValueError("Dispatch receipt differs from batch")
        rec = validate_training(job, row, allowed, verify_models=verify_models)
        if records and rec["partition"] != records[0]["record"]["partition"]:
            raise ValueError("Methods do not share the same training/calibration groups")
        records.append({**row, "root": job/"run", "record": rec})
    return records


def check_dispatch(job, gate_path, action, deadline, seconds):
    receipt_path = job/"process.json"
    until = min(deadline, time.monotonic()+5)
    while not receipt_path.is_file() and time.monotonic() < until:
        time.sleep(.05)
    receipt = data.read_json(receipt_path)
    expected = [sys.executable, "-B", str(Path(__file__).resolve()), action, "--gate", str(gate_path),
                "--id", job.name, "--deadline", str(deadline)]
    if (receipt["pid"] != os.getpid() or receipt["command"][-len(expected):] != expected or
            Path(receipt["cwd"]).resolve() != SCRIPTS.parent or not 0 < deadline-time.monotonic() <= seconds):
        raise ValueError("Worker does not own the controller's live reservation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "campaign", "worker", *POST))
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(dict(cohort=cohort(), runs=plan()), indent=2))
        return
    if args.gate is None:
        parser.error("a separately authorized gate is required")
    gate_path = args.gate.resolve()
    allowed = read_gate(gate_path)
    campaign = Path(allowed["campaign"])
    rows = [{**r, "seconds": allowed["seconds_per_method"][r["name"]]} for r in plan()]
    posts = [dict(id=k, action=k, bucket=k, seconds=allowed["post_seconds"][k], gpu=k=="confirmation") for k in POST]
    if args.action == "campaign":
        def command(row, job, deadline):
            return [sys.executable, "-B", __file__, row.get("action", "worker"), "--gate", str(gate_path),
                    "--id", row["id"], "--deadline", str(deadline)]
        def validate(row, job):
            if row["id"] not in POST:
                validate_training(job, row, allowed)
            elif data.read_json(job/"result.json")["status"] != {
                    "generation": "GENERATED_VALIDATED_OWNERS_SEALED", "confirmation": "ALL_54_BLIND_AND_SCORED",
                    "analysis": "SIX_METHOD_PAIRED_TABLE"}[row["id"]]:
                raise ValueError("Incomplete post-training step")
        queue.dispatch(allowed, rows, command, validate, posts)
        return
    tasks = {r["id"]: r for r in [*rows, *posts]}
    row = tasks.get(args.id)
    if (row is None or row.get("action", "worker") != args.action or args.deadline is None or
            not positive(args.deadline) or not 0 < args.deadline-time.monotonic() <= row["seconds"]):
        parser.error("action must match a reserved finite task")
    job = campaign/row["id"]
    # Atomic entry claim also prevents concurrent/restored invocations reopening pairs.
    if not job.is_dir():
        parser.error("controller has not claimed the job")
    check_dispatch(job, gate_path, args.action, args.deadline, row["seconds"])
    with (job/"worker_claim.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(id=row["id"], deadline=args.deadline), stream)
    try:
        if args.action == "worker":
            train(job, row, args.deadline, allowed)
        else:
            import step28_history_benchmark_evaluate as evaluation
            budget = Budget(campaign, args.deadline, allowed["limits"])
            evaluation.run(args.action, campaign, job, allowed, budget)
    except BaseException as error:
        queue.write(job/"failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
