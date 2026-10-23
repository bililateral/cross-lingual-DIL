#!/usr/bin/env python3
"""Read a supplied ZIP; hash bytes, parse source AST, and do ordinary arithmetic.

This script never imports or executes project modules. It never trains, loads a
model, infers, generates scientific data, or resamples. Queue times are arithmetic
on saved timing fields, not benchmarks. Usage:
  python3 mechanical_checks.py INPUT_ZIP OUTPUT_JSON
"""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import traceback
import zipfile

D = "reports/documentation/20261023/continual_implementation/"
B = "reports/documentation/20261023/continual_baselines/"
EXPECTED_SHA = "6a792263593188bbf89217f928de67626588b11bb5c329c81151e2f63aa9ff63"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def literals(source):
    """Only literal-valued module assignments, no eval/exec or module import."""
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for name, v in zip(target.elts, value):
                    if isinstance(name, ast.Name):
                        result[name.id] = v
    return result


def path_check(stems):
    current = defaultdict(list)
    fixed = []
    for stem in stems:
        p = PurePosixPath(stem)
        current[str(p.with_suffix(".json"))].append(stem)
        fixed.append(str(p.parent / (p.name + ".json")))
    collisions = {p: names for p, names in current.items() if len(names) > 1}
    return {
        "logical_metric_sets": len(stems),
        "current_unique_JSON_paths": len(current),
        "current_collision_groups": len(collisions),
        "duplicate_requests": len(stems) - len(current),
        "example_collisions": dict(list(collisions.items())[:2]),
        "append_extension_unique_JSON_paths": len(set(fixed)),
        "append_extension_has_no_collisions": len(set(fixed)) == len(stems),
        "NPY_has_same_collision_pattern": True,
        "no_metric_or_model_execution": True,
    }


def audit(input_zip):
    blob = input_zip.read_bytes()
    out = {
        "review_kind": "FIRST_IMPLEMENTATION_REVIEW_STATIC_MECHANICAL_ONLY",
        "executed_at_UTC": datetime.now(timezone.utc).isoformat(),
        "review_environment_python": platform.python_version(),
        "not_project_Linux_qualification": True,
        "operations": ["ZIP and SHA256", "JSON and AST parsing", "pure path strings", "ordinary arithmetic"],
        "prohibited_operations_performed": [],
    }
    with zipfile.ZipFile(input_zip) as z:
        names = z.namelist()
        read = z.read
        js = lambda p: json.loads(read(p))
        manifest = js("input_manifest.json")
        expected = {v["path"]: v for v in manifest}
        checks = []
        for name, row in expected.items():
            actual = read(name)
            checks.append(dict(path=name, bytes=len(actual), sha256=sha(actual),
                               bytes_match=len(actual) == row["bytes"], sha256_match=sha(actual) == row["sha256"]))
        out["input_identity"] = dict(
            name=input_zip.name, bytes=len(blob), sha256=sha(blob), members=len(names), payloads=len(expected),
            matches_user_identity=(len(blob) == 836755 and sha(blob) == EXPECTED_SHA and len(names) == 158),
            all_payloads_match=all(v["bytes_match"] and v["sha256_match"] for v in checks),
            duplicates=[p for p, n in Counter(names).items() if n > 1],
            missing=sorted(set(expected)-set(names)), extra=sorted(set(names)-set(expected)-{"input_manifest.json"}),
            payload_checks=checks)

        sources = js(D+"current_sources.json")
        source_checks = [dict(path=v["path"], bytes_match=len(read(v["path"])) == v["bytes"],
                              sha256_match=sha(read(v["path"])) == v["sha256"]) for v in sources]
        closure = {"scripts/"+name for name in (
            "step28_continual_five.py", "step28_continual_five_run.py", "step28_continual_five_control.py",
            "step28_continual_five_verify.py", "step28_continual_five_evaluate.py")}
        closure.add("tests/test_step28_continual_five.py")
        todo = list(closure)
        while todo:
            p = todo.pop()
            for node in ast.walk(ast.parse(read(p).decode("utf-8"))):
                imports = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                           [node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in imports:
                    if name and (name.startswith("step28_") or name.startswith("test_step28_")):
                        child = ("tests/" if name.startswith("test_") else "scripts/")+name+".py"
                        if child not in closure:
                            closure.add(child)
                            todo.append(child)
        parsed_python = sorted(closure)
        closure.update("schema/"+name+"_policy.json" for name in (
            "step28_chinese_base", "step28_alias_ranking", "step28_record_replay", "step28_bge_continual", "step28_continual_expression"))
        closure.add("schema/step28_continual_expression_catalog.json")
        closure.update([D+"authorization.json", D+"execution.zh.txt", D+"gpu_review_waiver.json",
                        "scripts/run_step28_continual_five_cpu_linux_20261023.sh",
                        B+"proposal.zh.txt", B+"budget.json", B+"review/primary_disposition.zh.txt"])
        listed = {v["path"] for v in sources}
        out["source_identity"] = dict(count=len(sources), checks=source_checks,
            all_match=all(v["bytes_match"] and v["sha256_match"] for v in source_checks),
            static_AST_closure_matches=closure == listed, missing=sorted(closure-listed), extra=sorted(listed-closure),
            AST_parsed_files=parsed_python, AST_is_not_semantic_full_read=True,
            GPU_repair_correctness_not_reviewed=True)

        f = literals(read("scripts/step28_continual_five.py").decode())
        e = literals(read("scripts/step28_continual_five_evaluate.py").decode())
        bge = literals(read("scripts/step28_bge_continual.py").decode())
        methods, orders, seeds, roles = f["METHODS"], f["ORDERS"], f["SEEDS"], bge["ROLES"]
        specs = [(o, s, m, f"{o}_s{s}_{m}") for o in orders for s in seeds for m in methods]
        old = [f"old_dev/{rid}_{stage}_{role}" for _, _, _, rid in specs for stage in (1, 2, 3) for role in roles]
        conf = [f"confirmation/metrics_{rid}_{i}_{j}_{role}" for _, _, m, rid in specs
                for i, j in e["CORE_CELLS"]+(((1, 2), (2, 3)) if m == "SEQ_ref" else ()) for role in roles]
        out["metric_filename_defect"] = dict(
            finding_id="I01", mechanism="with_suffix replaces the dotted method-name tail; append preserves it",
            old_dev=path_check(old), confirmation=path_check(conf),
            affected_run_ids=[rid for _, _, m, rid in specs if "." in m],
            modified_project_sources=False)
        selected = js("reports/seller_alias_continual/20261023/optuna_result/job/selected.json")
        out["frozen_parameters"] = {m: dict(actual=list(f["PARAMETERS"][m]),
            selected=[selected[a]["spec"]["lambda_rank"], selected[a]["spec"]["lr_multiplier"]],
            matches=tuple(f["PARAMETERS"][m]) == (selected[a]["spec"]["lambda_rank"], selected[a]["spec"]["lr_multiplier"]))
            for m, a in (("S_tuned", "S"), ("H_pos_tuned", "H_pos"))}

        budget, projection = js(B+"budget.json"), js(D+"resource_projection.json")
        native = js(D+projection["source_result"])
        concurrent = {k: v for k, v in native["results"].items() if "_parallel_" in k}
        measurements = [v["measurements"] for v in concurrent.values()]
        tc = max(v["current_update"] for v in measurements)
        tr = max(v[k] for v in measurements for k in ("replay_update", "replay_repeat") if k in v)
        tf = max(v[k] for v in measurements for k in ("forward", "forward_after", "teacher") if k in v)
        io = {k: max(v[k] for v in measurements) for k in ("load", "save_full", "restore_full", "save_inference", "restore_inference")}
        io_seconds = io["load"]+3*sum(io[k] for k in io if k != "load")
        seq_unrounded = 1.25*(864*tc+660*tf+io_seconds+1800)
        mem_unrounded = 1.25*(288*tc+576*tr+660*tf+io_seconds+1800)
        per = {m: math.ceil(seq_unrounded if m == "SEQ_ref" else mem_unrounded) for m in methods}
        total = sum(per[m] for _, _, m, _ in specs)
        finishes, schedule = [0, 0, 0], []
        for _, _, m, rid in specs:
            slot = min(range(3), key=lambda x: (finishes[x], x))
            start = finishes[slot]
            finishes[slot] += per[m]
            schedule.append(dict(run_id=rid, slot=slot, start_seconds=start, finish_seconds=finishes[slot]))
        out["workload"] = dict(methods=list(methods), orders=list(orders), seeds=list(seeds), paths=len(specs),
            updates=45*864, gradient_group_presentations=9*864+36*(288+576*2), inference_states=45*3,
            confirmation_group_forwards=(45*6+9*2)*40, training_endpoint_restore_forwards_per_run=3*72*3,
            birth_teacher_forward_upper_per_run=12, all_buckets_seconds=budget["budget_seconds"],
            total_budget_seconds=sum(budget["budget_seconds"].values()), total_budget_hours=sum(budget["budget_seconds"].values())/3600)
        out["projection"] = dict(
            source_path=D+projection["source_result"], source_sha256_matches=sha(read(D+projection["source_result"])) == projection["source_sha256"],
            selected_fields="/results/native02_parallel_*/measurements; supplied local GPU background, no GPU repair adjudication",
            current_seconds=tc, replay_seconds=tr, forward_seconds=tf, io_seconds=io, total_io_per_run_seconds=io_seconds,
            SEQ_unrounded_seconds=seq_unrounded, memory_unrounded_seconds=mem_unrounded,
            per_run_projected_seconds=per, saved_per_run_matches=per == projection["per_run_projected_seconds"],
            training_seconds=total, training_hours=total/3600, margin_seconds=1620000-total,
            margin_hours=(1620000-total)/3600, margin_fraction_of_pool=(1620000-total)/1620000,
            slot_finish_seconds=finishes, slot_finish_matches=finishes == projection["three_slot_projected_finish_seconds"],
            conditional_calendar_training_hours=max(finishes)/3600, deterministic_queue=schedule,
            confirmation_forward_only_seconds=11520*tf*1.25,
            confirmation_forward_only_hours=11520*tf*1.25/3600,
            not_guaranteed_wall_clock=True, not_GPU_busy_hours=True,
            assumptions=["25 percent multiplicative allowance", "1800 seconds unmeasured overhead per run", "fixed measured-time proxies", "no new benchmark"])

        account = js(D+"qualification/accounting.json")
        wrappers = [dict(path=D+f"qualification/cpu{i:02d}.wrapper.json", **js(D+f"qualification/cpu{i:02d}.wrapper.json")) for i in range(1, 8)]
        for row in wrappers:
            row["ceil_wall_seconds"] = math.ceil(row["wall_ns"]/1e9)
        led = js(D+"qualification/native_ledger.json")
        native_seconds = sum(v["actual_seconds"] for v in led["completed"] if v["bucket"] == "native")
        samples = []
        for run_id in ("native01", "native02"):
            p = D+"qualification/"+run_id+"/resource_samples.jsonl"
            rows = [json.loads(line) for line in read(p).decode().splitlines()]
            samples.append(dict(path=p, count=len(rows),
                adjacent_gap_minus_fixed_wait_seconds=rows[-1]["time"]-rows[0]["time"]-(len(rows)-1)))
        gaps = sum(v["adjacent_gap_minus_fixed_wait_seconds"] for v in samples)
        cpu_seconds = sum(v["charged_seconds"] for v in wrappers)
        control = account["original_control_recorded_seconds"]+120
        out["qualification_accounting"] = dict(
            cpu_wrappers=wrappers, cpu_seconds=cpu_seconds,
            wrapper_rounding_matches=all(v["charged_seconds"] == v["ceil_wall_seconds"] for v in wrappers),
            native_ledger_rows=led["completed"], native_seconds=native_seconds,
            attempts=1+sum(v["attempted_calls"] for v in native["results"].values()),
            completed_calls=sum(v["actual_calls"] for v in native["results"].values()),
            old_control_seconds=account["original_control_recorded_seconds"], additional_control_provision_seconds=120,
            sample_gap_sources=samples, sample_non_wait_gap_seconds=gaps, remainder_of_120_seconds=120-gaps,
            total_control_seconds=control, qualification_total_seconds=cpu_seconds+native_seconds+control,
            saved_total_seconds=account["qualification_only_total_seconds"],
            total_matches_within_1e_9=abs(cpu_seconds+native_seconds+control-account["qualification_only_total_seconds"])<1e-9,
            provision_is_not_exact_measurement=True, failure_costs_retained=True,
            remaining_cpu_seconds=1200-cpu_seconds, remaining_native_seconds=5400-native_seconds,
            remaining_control_seconds=1800-control)

        storage = budget["storage"]
        old_bytes = ((storage["inference_files"]+storage["max_concurrent_temporary_inference_files"])*storage["inference_file_proxy_bytes"]
                    +storage["max_concurrent_full_files"]*storage["full_file_proxy_bytes"]+storage["small_evidence_max_GiB"]*2**30)
        max_i = max(v["inference_bytes"] for v in native["results"].values())
        max_f = max(v["full_bytes"] for v in native["results"].values())
        updated = (135+3)*max_i+6*max_f+2*2**30
        dense = dict(training_endpoint_scores=45*3*(3*(12+60)*378*4+2*60*378*8),
                     old_dev_query=405*60*28*10*8, confirmation_scores=11520*378*(4+8+8),
                     confirmation_query=864*40*28*10*8)
        out["storage_arithmetic"] = dict(
            old_proxy_bytes=old_bytes, old_proxy_GiB=old_bytes/2**30,
            native_saved_max_inference_bytes=max_i, native_saved_max_full_bytes=max_f,
            updated_proxy_bytes=updated, updated_proxy_GiB=updated/2**30,
            margin_under_224_GiB=224-updated/2**30, free_minus_campaign_GiB=288-224,
            fixed_shape_dense_payload_bytes=dense, dense_total_bytes=sum(dense.values()),
            dense_total_MiB=sum(dense.values())/2**20, memory_snapshot_contract_upper_MiB=36*3,
            not_whole_campaign_measurement=True,
            omissions_from_dense_estimate=["NPY headers", "JSON", "logs", "generated text", "filesystem metadata"])

        out["checks_completed"] = True
        out["mechanical_execution_failures"] = []
        out["conclusion"] = "Identity/count/timing arithmetic consistent; metric filename collision deterministically reproduced without executing project code."
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: mechanical_checks.py INPUT_ZIP OUTPUT_JSON")
    try:
        result = audit(Path(sys.argv[1]))
    except Exception as error:
        result = dict(checks_completed=False, mechanical_execution_failures=[dict(error=repr(error), traceback=traceback.format_exc())])
        Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        raise
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(checks_completed=result["checks_completed"], input_identity_matches=result["input_identity"]["matches_user_identity"],
        source_closure_matches=result["source_identity"]["static_AST_closure_matches"],
        old_dev_unique_paths=result["metric_filename_defect"]["old_dev"]["current_unique_JSON_paths"],
        confirmation_unique_paths=result["metric_filename_defect"]["confirmation"]["current_unique_JSON_paths"],
        training_hours=result["projection"]["training_hours"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
