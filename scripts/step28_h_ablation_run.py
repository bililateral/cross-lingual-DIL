"""Closed formal entry and exact eight-arm/reference completion checks."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

import step28_h_ablation as m
import step28_history_optuna_run as old

data, queue, inherited = m.data, old.queue, old.inherited
LIMITS = {**old.LIMITS, "disk_bytes": 288*2**30, "available_ram_bytes": 72*2**30}
BUDGETS = dict(training=378*3600, confirmation=12*3600, auxiliary=4*3600)
ACTIONS = ("train", "generate", "confirmation", "analyze")


def sources():
    paths = {data.ROOT/"scripts"/("step28_h_ablation"+suffix+".py")
             for suffix in ("", "_run", "_control", "_evaluate")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name and name.startswith("step28_"):
                    path = data.ROOT/"scripts"/(name+".py")
                    if path not in paths:
                        paths.add(path)
                        todo.append(path)
    for name in ("chinese_base", "alias_ranking", "record_replay", "bge_continual", "continual_expression"):
        paths.add(data.ROOT/"schema"/("step28_"+name+"_policy.json"))
    paths.add(data.ROOT/"schema/step28_continual_expression_catalog.json")
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def gate(path, action):
    value = data.read_json(path)
    # No formal data, checkpoint, CUDA or output access before this closed gate.
    if (sys.platform != "linux" or value.get("experiment") != "H_HISTORY_ABLATION"
            or value.get("approved") is not True or value.get("review_closed") is not True
            or action not in value.get("actions", ()) or value.get("workers") != 3
            or value.get("automatic_retry") is not False or value.get("train_data") is not True
            or value.get("new_confirmation_data") is not True):
        raise ValueError("Ablation formal execution is closed")
    if (Path(value["workspace"]).resolve() != data.ROOT.resolve() or value["sources"] != sources()
            or value["runs"] != [s.identity() for s in m.specifications()]
            or value["limits"] != LIMITS or value["budget_seconds"] != BUDGETS
            or value["per_run_seconds"] != 21600 or value["mother_parameters"] != m.SELECTED):
        raise ValueError("Fixed sources, 63 runs, mother or limits differ")
    for key in ("authorization", "review_evidence", "cpu_receipt", "gpu_reuse_decision"):
        queue.verify(value[key])
    cpu = data.read_json(Path(value["cpu_receipt"]["path"]))
    if cpu.get("status") != "PASS" or cpu.get("sources") != value["sources"]:
        raise ValueError("Source-bound CPU evidence required")
    for item in value["input_metadata"]:
        queue.verify(item)
    c = m.config(m.specifications()[0])
    expected = {str((data.ROOT/c["data_root"]/n).resolve()) for n in ("manifest.json", "validation.json")}
    if expected != {str(Path(r["path"]).resolve()) for r in value["input_metadata"]}:
        raise ValueError("Actual dataset metadata binding required")
    return value


class Budget(old.Budget):
    def __init__(self, campaign, deadline):
        super().__init__(campaign, deadline, LIMITS)


def read_batch(path, allowed):
    path = path.resolve()
    if path != Path(allowed["campaign"]).resolve()/"training/batch.json":
        raise ValueError("One fixed training list required")
    batch = data.read_json(path)
    specs = m.specifications()
    if (batch["status"] != "APPROVED_IMPLEMENTATION_AND_RESOURCES" or batch["sources"] != allowed["sources"]
            or batch["runs"] != [s.identity() for s in specs]
            or batch["main_gate_sha256"] != allowed["_gate_sha256"]
            or batch["allocator_bytes"] != LIMITS["allocator_bytes"]):
        raise ValueError("Fixed ablation batch differs")
    return batch, specs


def validate_completion(root, spec, allowed, *, verify_models=True):
    row = data.read_json(root/"completion.json")
    expected_presentations = 1440 if spec.uses_history else 864
    if (row["status"] != inherited.COMPLETE or row["spec"] != spec.identity()
            or row["source_files"] != allowed["sources"] or row["updates"] != 864
            or row["gradient_group_presentations"] != expected_presentations
            or set(row["points"]) != {"1", "2", "3"}
            or row["access"] != dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)):
        raise ValueError("Complete source-bound ablation trajectory required")
    ids = [r["group_uid"] for r in row["partition"]["development"]]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("Blind development identity differs")
    for stage in (1, 2, 3):
        point = row["points"][str(stage)]
        cal_ids = [r["group_uid"] for r in row["partition"]["calibration"] if r["domain"] == spec.order[stage-1]]
        if (point["stage"] != stage or point["adam_step"] != 288*stage or point["development_ids"] != ids
                or point["calibration_ids"] != cal_ids or point["continual_spec"] != spec.identity()
                or not point["full_restore_verified"] or not point["inference_restore_verified"]):
            raise ValueError("Restored endpoint/configuration differs")
        memory = point["memory_summary"]
        if spec.uses_history:
            if (len(memory["members"]) != 6 or memory["target_kind"] != spec.target_kind
                    or memory["bytes"] > 1048576
                    or len(memory["tables"]) != (6 if spec.beta else 0)):
                raise ValueError("Actual typed history differs")
        elif memory is not None or point["memory"] is not None:
            raise ValueError("SEQ must not create history")
        if verify_models:
            data.verify(root/point["model"]["path"], point["model"])
        fit = data.read_json(data.verify(root/point["mapping"]["path"], point["mapping"]))
        if (fit["status"] != "PASS_CALIBRATION_FIT"
                or dict(zip(("a", "b"), m.parent.calibration.parameters(fit))) != point["maps"]["stage"]):
            raise ValueError("Training-side calibration differs")
        updates = data.read_json(data.verify(root/row["training"][str(stage)]["path"], row["training"][str(stage)]))
        if updates["updates"] != 288 or len(updates["rows"]) != 288:
            raise ValueError("Incomplete actual update log")
        for step, log in enumerate(updates["rows"], 1):
            history = stage > 1 and spec.uses_history
            if (log["step"] != step or (log["history_uid"] is not None) != history
                    or (log["encoder_lr"], log["head_lr"]) != spec.rates(step)
                    or ("history" in log) != history):
                raise ValueError("Actual history/update schedule differs")
            for role in (("current", "history") if history else ("current",)):
                part = log[role]
                supervision = role == "current" or spec.alpha > 0
                teacher = role == "history" and spec.beta > 0
                views = (1 if spec.name == "NoRegroup" else 2) if supervision else 0
                if (part["supervised_enabled"] != supervision or part["teacher_enabled"] != teacher
                        or len(part["views"]) != views
                        or (part["assignment_sha256"] is not None) != (views == 2)
                        or not supervision and part["supervised"] != 0
                        or not teacher and part["distillation"] != 0):
                    raise ValueError("Deleted operation still appears in actual log")
                sup = (1. if role == "current" else spec.alpha)*part["supervised"]
                kd = spec.beta*part["distillation"]
                if not np.allclose([part["supervised_contribution"], part["teacher_contribution"], part["weighted"]],
                                   [sup, kd, sup+kd], rtol=1e-5, atol=1e-5):
                    raise ValueError("Actual explicit role weights differ")
        for role in ("raw", "calibration"):
            arrays = []
            for prefix in ("", "full-", "inference-"):
                rec = point["scores"][prefix+role]
                value = np.load(data.verify(root/rec["path"], rec), allow_pickle=False)
                if value.shape != (60 if role == "raw" else 12, 378) or not np.isfinite(value).all():
                    raise ValueError("Incomplete blind endpoint matrix")
                arrays.append(value)
            if any(not np.array_equal(arrays[0], a) for a in arrays[1:]):
                raise ValueError("Saved/restored predictions differ")
        raw_rec = point["scores"]["raw"]
        raw = np.load(data.verify(root/raw_rec["path"], raw_rec), allow_pickle=False)
        for role in ("stage-cal", "first-cal"):
            rec = point["scores"][role]
            m.parent.calibration.preserve_order(raw, np.load(data.verify(root/rec["path"], rec), allow_pickle=False))
    return row


def references(allowed, *, verify_models=True):
    original = data.read_json(queue.verify(allowed["full_reference_gate"]))
    completed_path = queue.verify(allowed["mother_completion"])
    completed = data.read_json(completed_path)
    selected = data.read_json(queue.verify(allowed["mother_selected"]))
    queue.verify(allowed["mother_result_disposition"])
    if (completed["status"] != "COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED"
            or (completed_path.parent/"failure.json").exists() or completed["selected"] != selected
            or selected["number"] != 3 or selected["parameters"] != m.SELECTED
            or completed["sources"] != original["sources"]):
        raise ValueError("Complete reviewed selected mother required")
    rows, seen = [], set()
    for item in allowed["full_reference_runs"]:
        path = queue.verify(item["completion"])
        rec = data.read_json(path)
        spec = m.history.Spec.from_dict(rec["spec"])
        if (spec.phase != "replication" or spec.number != 3 or spec.parameters != m.SELECTED
                or (spec.order, spec.seed) in seen):
            raise ValueError("Only the nine trial3 replication trajectories may be Full")
        rec = old.validate_completion(path.parent, spec, original, verify_models=verify_models)
        seen.add((spec.order, spec.seed))
        rows.append(dict(name="Full", spec=spec, root=path.parent, record=rec))
    if seen != {(o, s) for o in m.ORDERS for s in m.SEEDS}:
        raise ValueError("Missing Full references; no fallback or automatic retraining")
    return rows


def execute(job, spec, deadline, main_gate, batch_path):
    allowed = gate(main_gate, "train")
    allowed["_gate_sha256"] = data.sha256(main_gate)
    batch, specs = read_batch(batch_path, allowed)
    if (spec.identity() not in batch["runs"] or job.resolve() != batch_path.parent/spec.run_id
            or not job.is_dir() or set(p.name for p in job.iterdir()) - {"console.txt", "process.json", "spec.json"}):
        raise ValueError("Fresh owned ablation job required")
    implementation = SimpleNamespace(config=m.config, Method=m.Method, specifications=lambda: specs)
    try:
        return inherited.execute(job, spec, deadline, batch_path, implementation=implementation,
            source_files=allowed["sources"], budget_factory=lambda _, end: Budget(Path(allowed["campaign"]), end))
    except BaseException as error:
        data.write_json(job/"failure.json", dict(status="STOP_NO_RETRY", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    execute(args.job, m.Spec.from_dict(data.read_json(args.spec)), args.deadline, args.gate, args.batch)
