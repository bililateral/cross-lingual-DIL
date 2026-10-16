"""H_cons entry; reuse the single-arm lifecycle with explicit dependencies."""
from __future__ import annotations

import argparse
import ast
from functools import partial
import os
from pathlib import Path
import sys

import numpy as np

import step28_g_half_run as single
import step28_h_consistency as method

record, data, parent, core = single.record, single.data, single.parent, single.core
base, previous, diag = single.base, single.previous, single.diag
POLICY = data.ROOT/"schema/step28_h_consistency_policy.json"
BASELINE = data.ROOT/"reports/documentation/20261016/h_consistency/baseline_sources.json"
AUTHORIZATION = data.ROOT/"reports/documentation/20261016/h_consistency_implementation/authorization.json"
RECOVERY_AUTHORIZATION = data.ROOT/"reports/documentation/20261016/h_consistency_implementation/qualification/preload_recovery_authorization.json"
POINTS = ("A_H_cons", "B_H_cons")
COMPLETE = single.COMPLETE
COMPLETION_STATUS = "COMPLETE_H_CONS_FIXED_AB"
DIRECT_COMPARISON_KEY = "direct_H_new_B_positive"
CPU_TESTS = ("test_js_value_gradient_precision_and_faults", "test_shared_table_pullback_and_finite_difference",
             "test_complete_objective_vjp_and_actual_updates", "test_actual_capture_restore_and_training_identity",
             "test_loss_gate_and_saved_comparison_interfaces", "test_unchanged_single_arm_defaults",
             "test_authorized_preload_recovery")
projection, continuity = single.projection, single.continuity
fresh, pretrained_archive = single.fresh, single.pretrained_archive
public_inputs, attach_labels, partition = single.public_inputs, single.attach_labels, single.partition


def policy() -> dict:
    p = data.read_json(POLICY)
    old = single.policy()
    expected = {**old, "study": "fixed_H_natural_regrouped_query_consistency", "methods": ["H_cons"],
        "pooling": method.identity(), "training": method.training_identity(),
        "evaluation": dict(bootstrap_replicates=5000, bootstrap_seed=20260930,
            comparisons=[["H_cons", "S"], ["H_cons", "G_half"]], primary_metric="map", primary_role="raw",
            success="B_end_H_cons_minus_S_each_AB_MAP_CI_lower_strictly_positive")}
    if p != expected:
        raise ValueError("Approved fixed H_cons policy differs")
    return p


def config() -> dict:
    c = record.config()
    c["runtime"] = policy()["runtime"]
    return c


def sources() -> list[dict]:
    paths = {data.ROOT/n for n in ("scripts/step28_h_consistency_run.py", "scripts/step28_h_consistency_verify.py",
        "tests/test_step28_h_consistency.py", "tests/test_step28_g_half.py", "tests/test_step28_record_replay.py")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT/"scripts"/(name+".py")
                    if child not in paths:
                        paths.add(child)
                        todo.append(child)
    paths.update(data.ROOT/"schema"/name for name in ("step28_h_consistency_policy.json", "step28_g_half_policy.json",
        "step28_record_replay_policy.json", "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json",
        "step28_chinese_base_policy.json"))
    paths.update((BASELINE, AUTHORIZATION, RECOVERY_AUTHORIZATION, data.ROOT/"docs/SELLER_ALIAS_H_CONSISTENCY.zh.md",
                  data.ROOT/"scripts/run_step28_h_consistency_linux_20261016.sh",
                  single.BASELINE, single.AUTHORIZATION, data.ROOT/"docs/SELLER_ALIAS_G_HALF.zh.md",
                  data.ROOT/"scripts/run_step28_g_half_linux_20261015.sh"))
    return [data.record(path, data.ROOT) for path in sorted(paths)]


def baseline_runs() -> dict:
    manifest = data.read_json(BASELINE)
    if (data.sha256(BASELINE) != "b71119f5f278776cfa64f2a175acbadecc1614e8103a869319d9270522ececf3"
            or len(manifest["files"]) != 52 or set(manifest["baselines"]) != {"S", "H"}):
        raise ValueError("Approved S/H saved scope differs")
    for row in manifest["files"]:
        data.verify(data.ROOT/row["path"], row)
    for row in manifest["unchanged_direct_sources"]:
        data.verify(data.ROOT/row["path"], row)
    runs = {}
    for info in manifest["baselines"].values():
        root = data.ROOT/info["root"]
        old = data.read_json(root/"run/manifest.json")
        col = data.read_json(root/"evaluation/collected.json")
        split = data.read_json(root/"run/partition.json")
        if (col["group_ids"] != [r["group_uid"] for r in split["development"]]
                or col["domains"] != [r["domain"] for r in split["development"]]
                or col["columns"] != list(parent.metrics.COLUMNS)
                or old["initials"][info["arm"]] != info["initial_identity"]):
            raise ValueError("Saved alignment/initial identity differs")
        for name in info["points"]:
            if set(col["points"][name]) != set(parent.ROLES):
                raise ValueError("Saved role coverage differs")
        runs[info["arm"]] = (root, col, old)
    a, b = runs["S"], runs["G_half"]
    if (data.read_json(a[0]/"run/partition.json") != data.read_json(b[0]/"run/partition.json")
            or a[2]["public_inputs"] != b[2]["public_inputs"]):
        raise ValueError("S/H inputs/partition differ")
    return runs


def baseline_inputs():
    runs = baseline_runs()
    root, col, old = runs["S"]
    manifest = data.read_json(BASELINE)
    return {"initial_identity": manifest["baselines"]["S"]["initial_identity"]}, root, col, old


def loss_total(row: dict, stage: int, arm: str) -> float:
    if any(row.get(k) != v for k, v in method.metadata(arm).items()):
        raise ValueError("Update scoring/training identity differs")
    total = 0.
    for role, weight in (("current", 1.), ("history", .1)):
        if role == "history" and stage == 1:
            if "history" in row:
                raise ValueError("Unexpected A history")
            continue
        r = row[role]
        if r.get("qualification_without_c") is not False or r["mse1"] is not None:
            raise ValueError("Formal update used qualification mode or wrong teacher")
        if role == "current" and r["mse0"] != 0:
            raise ValueError("Current branch used birth teacher")
        expected = dict(supervised_contribution=weight*r["supervised"],
            consistency_contribution=weight*r["consistency"], teacher_contribution=.5*r["mse0"],
            distillation=r["mse0"])
        for key, value in expected.items():
            if not np.isfinite(value) or not np.isclose(r[key], value, atol=2e-6, rtol=2e-6):
                raise ValueError("Wrong explicit H_cons contribution: "+role+"/"+key)
        contribution = sum(expected[k] for k in ("supervised_contribution", "consistency_contribution", "teacher_contribution"))
        if not np.isclose(r["weighted"], contribution, atol=2e-6, rtol=2e-6):
            raise ValueError("Weighted branch omitted a contribution")
        total += contribution
    if not np.isclose(row["total"], total, atol=2e-6, rtol=2e-6):
        raise ValueError("Actual total omitted H_cons contribution")
    return total


def validate_diagnostic(value: dict, arm: str) -> None:
    if any(value.get(k) != v for k, v in method.metadata(arm).items()):
        raise ValueError("Diagnostic H_cons identity missing")


def validate_native(value: dict) -> None:
    if (value.get("training_identity") != method.training_identity()
            or value.get("consistency_precision_verified") is not True
            or len(value.get("table_precision", [])) != 5):
        raise ValueError("Actual native H_cons table precision/training identity missing")
    for phase in ("current", "paired"):
        for component in ("encoder", "head"):
            effect = value.get("effects", {}).get(phase, {}).get(component, {})
            adam = effect.get("adam", {})
            rows = adam.get("parameters", [])
            active = [r for r in rows if r.get("state_present")]
            if (adam.get("finite") is not True or not active
                    or {r["name"] for r in active} != set(effect.get("active_parameter_names", []))):
                raise ValueError("Native on/off Adam mapping/evidence missing")
            for row in active:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    delta = row.get(key, {}).get("max_abs_delta", float("nan"))
                    if not np.isfinite(delta) or delta < 0:
                        raise ValueError("Native on/off Adam difference missing/nonfinite")
                expected_step = 2 if phase == "current" else 337
                if row["step"].get("off") != expected_step or row["step"].get("on") != expected_step:
                    raise ValueError("Native on/off Adam clock differs")


def validate_preload_recovery(gate: dict, prior_seconds: float, output: Path) -> None:
    if gate.get("preload_recovery_authorization") != data.record(RECOVERY_AUTHORIZATION, data.ROOT):
        raise ValueError("Specific preload recovery approval missing")
    auth = data.read_json(RECOVERY_AUTHORIZATION)
    if (auth["status"] != "APPROVED_ONE_PRELOAD_RECOVERY" or prior_seconds != auth["prior_charged_seconds"]
            or prior_seconds != 3 or output != data.ROOT/auth["attempt_output"]
            or auth["maximum_cumulative_gpu_seconds"] != 600 or auth["maximum_total_updates"] != 6):
        raise ValueError("Preload recovery scope or cumulative charge differs")
    for key in ("supplement", "failed_result", "failed_wallclock"):
        data.verify(data.ROOT/auth[key]["path"], auth[key])
    failed = data.read_json(data.ROOT/auth["failed_result"]["path"])
    wall = data.read_json(data.ROOT/auth["failed_wallclock"]["path"])
    if (failed["status"] != "FAIL" or failed.get("actual_updates", 0) != 0 or failed.get("updates")
            or failed.get("stage") is not None or failed.get("native") is not None
            or failed.get("max_reserved_bytes") != 0 or wall["cumulative_charged_seconds"] != prior_seconds
            or not failed.get("traceback", "").rstrip().endswith("ValueError: Pinned BGE archive differs or is missing")):
        raise ValueError("Only the approved zero-update preload failure can recover")


class Budget(single.Budget):
    def __init__(self, root: Path):
        super().__init__(root)
        self.limits = policy()["runtime"]


# The shared lifecycle receives this module explicitly; no process-global monkey patch.
train_stage = partial(single.train_stage, entry=sys.modules[__name__])
memory_from = partial(single.memory_from, entry=sys.modules[__name__])
restore_point = partial(single.restore_point, entry=sys.modules[__name__])
checkpoint = partial(single.checkpoint, entry=sys.modules[__name__])
train = partial(single.train, entry=sys.modules[__name__])
blind_gate = partial(single.blind_gate, entry=sys.modules[__name__])
open_valid = partial(single.open_valid, entry=sys.modules[__name__])
collect = partial(single.collect, entry=sys.modules[__name__])
finalize = partial(single.finalize, entry=sys.modules[__name__])
reviewed_gate = partial(single.reviewed_gate, entry=sys.modules[__name__])
validate_gate = partial(single.validate_gate, entry=sys.modules[__name__])
execute = partial(single.execute, entry=sys.modules[__name__])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Scientific execution is Linux-only")
    execute(args.job.resolve(), args.gate.resolve())
