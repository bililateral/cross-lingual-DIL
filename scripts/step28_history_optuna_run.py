"""Source-bound, closed-by-default trajectory entry and shared custody checks."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

import step28_history_optuna as m
import step28_continual_five_run as inherited
import step28_formal_queue as queue

data = m.data
CONTEXT = "reports/documentation/20261102/history_optuna"
LIMITS = dict(allocator_bytes=int(9.25*2**30), process_gpu_bytes=int(10.25*2**30),
              free_gpu_bytes=int(10.25*2**30), rss_bytes=64*2**30,
              disk_bytes=64*2**30, small_bytes=2*2**30, free_reserve_bytes=64*2**30)
BUDGETS = dict(training=270*3600, confirmation=6*3600, auxiliary=4*3600)
ACTIONS = ("train", "search-evaluate", "replication-evaluate", "generate", "confirmation", "analyze")


def sources():
    paths = {data.ROOT/p for p in (
        "scripts/step28_history_optuna.py", "scripts/step28_history_optuna_run.py",
        "scripts/step28_history_optuna_control.py", "scripts/step28_history_optuna_evaluate.py",
        "scripts/run_step28_history_optuna_cpu.py",
        "tests/test_step28_history_optuna.py")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and (name.startswith("step28_") or name.startswith("test_step28_")):
                    path = data.ROOT/("tests" if name.startswith("test_") else "scripts")/(name+".py")
                    if path not in paths:
                        paths.add(path)
                        todo.append(path)
    for name in ("chinese_base", "alias_ranking", "record_replay", "bge_continual", "continual_expression"):
        paths.add(data.ROOT/"schema"/("step28_"+name+"_policy.json"))
    paths.add(data.ROOT/"schema/step28_continual_expression_catalog.json")
    return [data.record(path, data.ROOT) for path in sorted(paths)]


def gate(path, action):
    allowed = data.read_json(path)
    # Reject before source resolution, reading any formal corpus, CUDA or creating outputs.
    if (sys.platform != "linux" or allowed.get("experiment") != "HISTORY_OPTUNA"
            or allowed.get("approved") is not True or allowed.get("review_closed") is not True
            or action not in allowed.get("actions", ()) or allowed.get("workers") != 3
            or allowed.get("train_data") is not True or allowed.get("automatic_retry") is not False
            or allowed.get("new_confirmation_data") is not True):
        raise ValueError("History Optuna formal execution is closed; explicit reviewed scope required")
    if (Path(allowed["workspace"]).resolve() != data.ROOT.resolve()
            or allowed["sources"] != sources() or allowed["limits"] != LIMITS
            or allowed["budget_seconds"] != BUDGETS or allowed["per_run_seconds"] != 21600
            or allowed.get("trials") != m.TRIALS or allowed.get("space") != {k: list(v) for k, v in m.SPACE.items()}
            or allowed.get("anchor") != m.ANCHOR):
        raise ValueError("Actual sources, finite search or resources differ")
    for key in ("authorization", "review_evidence", "cpu_receipt"):
        queue.verify(allowed[key])
    cpu = data.read_json(Path(allowed["cpu_receipt"]["path"]))
    if cpu.get("status") != "PASS" or cpu.get("sources") != allowed["sources"]:
        raise ValueError("Actual source-bound CPU acceptance required")
    for record in allowed["input_metadata"]:
        queue.verify(record)
    c = m.config(m.specifications("search", 0, m.ANCHOR)[0])
    required = {str((data.ROOT/c["data_root"]/name).resolve()) for name in ("manifest.json", "validation.json")}
    if required != {str(Path(r["path"]).resolve()) for r in allowed["input_metadata"]}:
        raise ValueError("Pinned actual dataset manifest and validation required")
    return allowed


class Budget(inherited.Budget):
    def __init__(self, campaign, deadline, limits=LIMITS):
        super().__init__(campaign, deadline)
        self.limits = limits

    def check(self, reserve=0):
        now = time.monotonic()
        if now >= self.deadline-self.tail:
            raise TimeoutError("Owned action reached its shutdown reserve")
        if not reserve and now-self.last_checked < 5:
            return
        self.last_checked = now
        rss = inherited.resident_bytes(inherited.owned_processes(os.getpid()))
        files = queue.campaign_files(self.root)
        used = sum(size for _, size in files)
        small = sum(size for suffix, size in files if suffix != ".pt")
        self.peak_rss, self.peak_disk, self.peak_small = (
            max(self.peak_rss, rss), max(self.peak_disk, used), max(self.peak_small, small))
        if (rss > self.limits["rss_bytes"] or used+reserve > self.limits["disk_bytes"]
                or small > self.limits["small_bytes"]
                or shutil.disk_usage(self.root).free < reserve+self.limits["free_reserve_bytes"]):
            raise RuntimeError("History action RSS/disk/evidence allowance exceeded")


def read_batch(path, allowed):
    path = path.resolve()
    campaign = Path(allowed["campaign"]).resolve()
    if not path.is_relative_to(campaign):
        raise ValueError("Batch must belong to the fresh campaign")
    batch = data.read_json(path)
    specs = [m.Spec.from_dict(row) for row in batch["runs"]]
    if not specs:
        raise ValueError("Empty candidate")
    first = specs[0]
    expected = m.specifications(first.phase, first.number, first.parameters)
    if (batch["status"] != "APPROVED_IMPLEMENTATION_AND_RESOURCES" or batch["sources"] != allowed["sources"]
            or batch["runs"] != [s.identity() for s in expected]
            or batch["allocator_bytes"] != LIMITS["allocator_bytes"]
            or batch["main_gate_sha256"] != allowed["_gate_sha256"]):
        raise ValueError("Candidate batch differs from the authorized study")
    if first.phase == "replication":
        selected = data.read_json(campaign/"selected.json")
        if (selected["number"] != first.number or selected["parameters"] != first.parameters
                or first.parameters == m.ANCHOR):
            raise ValueError("Only the single frozen non-anchor winner may replicate")
    return batch, specs


def validate_completion(root, spec, allowed, *, verify_models=True):
    row = data.read_json(root/"completion.json")
    if (row["status"] != inherited.COMPLETE or row["spec"] != spec.identity()
            or row["source_files"] != allowed["sources"] or row["updates"] != 864
            or row["gradient_group_presentations"] != 1440 or set(row["points"]) != {"1", "2", "3"}
            or row["access"] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
        raise ValueError("Complete source-bound three-stage trajectory required")
    ids = [r["group_uid"] for r in row["partition"]["development"]]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("Sixty unique development groups required")
    for stage in ("1", "2", "3"):
        point = row["points"][stage]
        expected = [r["group_uid"] for r in row["partition"]["calibration"] if r["domain"] == spec.order[int(stage)-1]]
        if (point["stage"] != int(stage) or point["adam_step"] != 288*int(stage)
                or point["development_ids"] != ids or point["calibration_ids"] != expected
                or not point["full_restore_verified"] or not point["inference_restore_verified"]
                or point["continual_spec"] != spec.identity() or len(point["memory_summary"]["members"]) != 6):
            raise ValueError("Endpoint calibration, history or restored identity differs")
        if verify_models:
            data.verify(root/point["model"]["path"], point["model"])
        fit = data.read_json(data.verify(root/point["mapping"]["path"], point["mapping"]))
        current_map = dict(zip(("a", "b"), m.parent.calibration.parameters(fit)))
        if fit["status"] != "PASS_CALIBRATION_FIT" or current_map != point["maps"]["stage"]:
            raise ValueError("Training-side calibration differs")
        update = data.read_json(data.verify(root/row["training"][stage]["path"], row["training"][stage]))
        if update["updates"] != 288 or len(update["rows"]) != 288:
            raise ValueError("Actual update log incomplete")
        for step, log in enumerate(update["rows"], 1):
            if (log["step"] != step or (log.get("history_uid") is not None) != (int(stage) > 1)
                    or (log["encoder_lr"], log["head_lr"]) != spec.rates(step)):
                raise ValueError("Actual stage/history/update rates differ")
            for role in (("current", "history") if int(stage) > 1 else ("current",)):
                part = log[role]
                weight = 1. if role == "current" else spec.history_weight
                expected_supervised = weight*part["supervised"]
                expected_teacher = spec.teacher_weight*part["distillation"]
                if (not np.allclose([part["supervised_contribution"], part["teacher_contribution"], part["weighted"]],
                                   [expected_supervised, expected_teacher, expected_supervised+expected_teacher], atol=1e-5, rtol=1e-5)
                        or role == "current" and part["distillation"] != 0.):
                    raise ValueError("Actual logged history coefficients differ")
        for role in ("raw", "calibration"):
            values = []
            for prefix in ("", "full-", "inference-"):
                rec = point["scores"][prefix+role]
                value = np.load(data.verify(root/rec["path"], rec), allow_pickle=False)
                if value.shape != (60 if role == "raw" else 12, 378) or not np.isfinite(value).all():
                    raise ValueError("Incomplete/nonfinite blind score matrix")
                values.append(value)
            if any(not np.array_equal(values[0], v) for v in values[1:]):
                raise ValueError("Actual saved/restored scores differ")
        for role in ("stage-cal", "first-cal"):
            rec = point["scores"][role]
            value = np.load(data.verify(root/rec["path"], rec), allow_pickle=False)
            raw_rec = point["scores"]["raw"]
            raw = np.load(data.verify(root/raw_rec["path"], raw_rec), allow_pickle=False)
            m.parent.calibration.preserve_order(raw, value)
    return row


def references(allowed, *, verify_models=True):
    old_gate = data.read_json(queue.verify(allowed["h_reference_gate"]))
    records, seen = [], set()
    for item in allowed["h_reference_runs"]:
        path = queue.verify(item["completion"])
        row = data.read_json(path)
        spec = m.five.Spec.from_dict(row["spec"])
        if (spec.name != "H_pos_tuned" or (spec.order, spec.seed) in seen
                or row["status"] != inherited.COMPLETE or row["source_files"] != old_gate["sources"]
                or row["updates"] != 864 or set(row["points"]) != {"1", "2", "3"}):
            raise ValueError("Pinned original H reference differs")
        seen.add((spec.order, spec.seed))
        for point in row["points"].values():
            if not point["full_restore_verified"] or not point["inference_restore_verified"]:
                raise ValueError("Original H restoration evidence incomplete")
            if verify_models:
                data.verify(path.parent/point["model"]["path"], point["model"])
            data.verify(path.parent/point["mapping"]["path"], point["mapping"])
        records.append(dict(name="H_pos_tuned", spec=spec, root=path.parent, record=row))
    if seen != {(order, seed) for order in m.ORDERS for seed in m.REPLICATION_SEEDS}:
        raise ValueError("All nine existing H references required; never retrain missing references")
    return records


def execute(job, spec, deadline, main_gate, batch_path):
    allowed = gate(main_gate, "train")
    allowed["_gate_sha256"] = data.sha256(main_gate)
    batch, specs = read_batch(batch_path, allowed)
    if (spec.identity() not in batch["runs"] or job.resolve() != batch_path.parent/spec.run_id
            or not job.is_dir() or set(p.name for p in job.iterdir()) - {"console.txt", "process.json", "spec.json"}):
        raise ValueError("Only the freshly claimed worker directory is accepted")
    implementation = SimpleNamespace(config=m.config, Method=m.Method, specifications=lambda: specs)
    try:
        return inherited.execute(job, spec, deadline, batch_path, implementation=implementation,
            source_files=allowed["sources"], budget_factory=lambda _, end: Budget(Path(allowed["campaign"]), end))
    except BaseException as error:
        data.write_json(job/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error),
                                                traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    execute(args.job, m.Spec.from_dict(data.read_json(args.spec)), args.deadline, args.gate, args.batch)


if __name__ == "__main__":
    main()
