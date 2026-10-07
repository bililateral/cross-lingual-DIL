#!/usr/bin/env python3
"""Read-only, stdlib audit of an already-completed result bundle.

No project modules are imported. No model, Memory body, raw text, labels,
execute/native command, network, or server is accessed. Numeric operations
only recheck saved update logs and recorded identities/budgets.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import difflib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    read_scope = {}
    checks = []
    result = {"scope": "Read-only attachment provenance and saved update records; no project imports or new training",
              "python": platform.python_version()}

    def raw(path, purpose="identity"):
        path = Path(path)
        data = path.read_bytes()
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        item = read_scope.setdefault(name, {"path": name, "bytes": len(data),
                                           "sha256": hashlib.sha256(data).hexdigest(), "purposes": []})
        if purpose not in item["purposes"]:
            item["purposes"].append(purpose)
        return data

    def load(path, purpose="saved record semantics"):
        return json.loads(raw(path, purpose))

    def require(value, label):
        if not value:
            raise AssertionError(label)
        checks.append(label)

    def identity(path, record):
        payload = raw(path)
        require(len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == record["sha256"],
                "bytes/SHA: " + str(path.relative_to(root)))

    outer = load(root / "manifest.json", "package manifest")
    require(isinstance(outer, list) and len(outer) == 252, "outer manifest has 252 payload records")
    require(len({r["path"] for r in outer}) == 252, "outer payload paths unique")
    for record in outer:
        identity(root / record["path"], record)
    result["outer_payloads_verified"] = len(outer)

    r = root / "reports/seller_alias_continual/20261007/function_memory_result"
    src = r / "source"
    source = load(r / "source_inventory.json")
    require(len(source) == 23, "23 frozen source records")
    require({x["path"] for x in source} == {p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()},
            "frozen source directory equals exactly 23 manifest paths")
    for record in source:
        identity(src / record["path"], record)
    policy = load(src / "schema/step28_function_memory_policy.json", "frozen policy")
    baseline_policy = load(src / "schema/step28_bge_continual_policy.json", "inherited schedule seed")
    gate = load(r / "gate.json")
    execution = load(r / "job/execution.json")
    manifest = load(r / "job/run/manifest.json")
    for label, records in (("gate", gate["source_files"]), ("execution", execution["sources"]),
                            ("manifest", manifest["source_files"])):
        require(records == source, label + " source path/bytes/SHA list exactly equals frozen inventory")
    identity(r / "gate.json", execution["gate"])
    require(gate["runtime"] == policy["runtime"] and gate["supervision"] == policy["supervision"],
            "formal gate runtime/supervision equals frozen policy")
    require(policy["loss"] == {"coordinate": "fp16_forward_fp32_straight_through", "current": 1,
                               "epsilon": 0.001, "function": 0.5, "trace": 3, "dimension": 129, "history": 0.1},
            "frozen fixed candidate loss specification")
    require(policy["seed"] == "s0" and policy["orders"] == ["ABC", "BCA", "CAB"], "seed and three orders")
    for record in policy["reference"]["collections"]:
        identity(root / policy["reference"]["local_root"] / record["path"], record)

    documentation = root / "reports/documentation/20261006/function_memory"
    cpu_path = documentation / "repair/cpu_20261006_231000/result/result.json"
    gpu_path = documentation / "native_20261006_231500/result/result.json"
    cpu, gpu = load(cpu_path), load(gpu_path)
    # Original gate paths are server deployment paths. The supplied copies are
    # mapped explicitly here; neither copy is altered or treated as a new run.
    identity(cpu_path, gate["integration_cpu"])
    identity(gpu_path, gate["native"])
    for label, receipt, mode in (("CPU", cpu, "cpu"), ("native", gpu, "gpu")):
        require(receipt["source_files"] == source, label + " receipt exact 23-source binding")
        require(receipt["mode"] == mode and receipt["status"] == "PASS_HANDWRITTEN_ONLY", label + " receipt mode/status")
    require([cpu[k] for k in ("tests_run", "failures", "errors", "skipped")] == [9, 0, 0, 0],
            "repaired CPU 9/9 with no errors/failures/skips")
    native = gpu["native"]
    require(native["kind"] == "native_handwritten_first_optimizer_step" and native["official_data"] is False,
            "actual native kind and handwritten-only scope")
    require(native["input_shapes"] == [{"texts": 448, "minimum": 256, "maximum": 256}] * 2,
            "native current/history maximum shape 448 x 256")
    require(native["reference_max_difference"] == 0, "native canonical reference/live exactness record")
    require(len(native["history_only_gradient_norms"]) == 3 and
            all(math.isfinite(v) and v > 0 for v in native["history_only_gradient_norms"]),
            "native three finite nonzero D-only probe gradients")
    require(native["optimizer"]["adam_step"] == 1 and native["adam_parameters_with_state"] == 393,
            "native real first Adam step and state record")
    require(native["dtypes"] == {"parameters": 395, "gradients": 393, "adam_moments": 786, "dtype": "float32"},
            "native parameter/gradient/moment precision record")
    require(len(native["dropout_before"]) == 97 and native["dropout_before"] == native["dropout_after"] and
            set(native["dropout_in_history"]) == set(native["dropout_before"]) and
            all(v == 0 for v in native["dropout_in_history"].values()), "native 97 effective mode values restored")
    require(native["checkpointing"] is True and native["first_layer_calls"] == 560 and
            all(v > 0 for v in native["probe_max_parameter_changes"]), "native recomputation and parameter changes")
    native_gate = load(documentation / "native_gate.json")
    require(native_gate["source_files"] == source, "native gate 23-source binding")
    identity(cpu_path, native_gate["integration_cpu"])
    require(native["max_reserved_bytes"] < 28 * 2**30 and gpu["peak_rss_bytes"] < 64 * 2**30 and
            gpu["elapsed_seconds"] < 600, "native internal recorded budgets")
    require(gate["user_incremental_review_waiver"] == "不用送审了，可以直接开始训练的话就开始吧" and
            "not a result-review waiver" in gate["waiver_scope"], "waiver explicitly limited to incremental admission")

    inv = load(r / "inventory.json")
    available, omitted = [], []
    for rec in inv["files"]:
        if (r / rec["path"]).is_file():
            identity(r / rec["path"], rec)
            available.append(rec["path"])
        else:
            omitted.append(rec)
    require(len(inv["files"]) == 171 and len(available) == 162 and len(omitted) == 9 and
            all(x["path"].startswith("job/run/memory/") and x["path"].endswith(".bin") for x in omitted),
            "171 remote non-weight records: 162 supplied payloads plus 9 explicitly omitted Memory bodies")
    models = {x["path"]: x for x in inv["retained_model_files"]}
    memories = {x["path"]: x for x in omitted}
    require(len(models) == 9 and sum(x["bytes"] for x in models.values()) == 11758264524,
            "9 retained model records total 11758264524 bytes")
    for path, rec in models.items():
        require(rec["absolute_path"] == inv["remote_run"] + "/" + path and rec["uid"] == 1014,
                "retained model path/owner record: " + path)
    result["received_nonweight_inventory"] = {"registered": 171, "payloads_hashed": 162,
                                               "omitted_memory_metadata_only": omitted,
                                               "retained_models_metadata_only": list(models.values())}

    before = load(r / "job/before_valid.json")
    completion = load(r / "job/completion.json")
    access = load(r / "job/access.json")
    resource = load(r / "job/resource.json")
    require(before["access"] == {"train": 1, "valid": 0, "heldout": 0, "owners": 0}, "blind gate access before valid")
    require(before["status"] == "PASS_COMPLETE_BLIND_GATE" and before["points"] == 9, "complete blind gate record")
    identity(r / "job/run/manifest.json", before["manifest"])
    require(access == completion["access"] == policy["supervision"], "final access exactly train/valid=1, heldout/owners=0")
    require(completion["status"] == "COMPLETE_FUNCTION_MEMORY_FIXED_POINT" and completion["physical_updates"] == 1728,
            "completed fixed point status and physical updates")
    require(completion["development_criteria_pass"] is False, "completion preserves failed development criteria")
    identity(r / "job/evaluation/evaluation.json", completion["evaluation"])
    for k in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "peak_observed_bytes", "peak_rss_bytes"):
        require(resource[k] == completion["budget"][k], "resource/completion consistency: " + k)
    require(abs(resource["elapsed_seconds"] - completion["budget"]["elapsed_seconds"]) < .01,
            "resource final refresh only 0.0019 seconds after completion budget")
    require(resource["elapsed_seconds"] < 86400 and resource["peak_observed_bytes"] < 24 * 2**30 and
            resource["peak_cuda_reserved_bytes"] < 28 * 2**30 and resource["peak_rss_bytes"] < 64 * 2**30,
            "formal recorded wall/output/reserved/RSS all within frozen budgets")
    wrap = raw(r / "job.wrapper.txt", "wrapper chronology").decode()
    parsed = dict(line.split("=", 1) for line in wrap.splitlines() if "=" in line)
    duration = (dt.datetime.fromisoformat(parsed["ended_at"]) - dt.datetime.fromisoformat(parsed["started_at"])).total_seconds()
    require(parsed["exit_code"] == "0" and duration == float(parsed["total_wall_seconds"]) == 12360,
            "actual wrapper start/end/exit0/12360 seconds")
    require(not list((r / "job").rglob("*failure*")), "no supplied formal failure ledger")
    result["completion"] = {"wrapper": parsed, "resources": resource, "access": access}

    part = load(r / "job/run/partition.json")
    uid_role = {}
    for role, each in (("fit", 48), ("calibration", 12), ("development", 20)):
        require(collections.Counter(x["domain"] for x in part[role]) == {d: each for d in "ABC"}, role + " per-domain cardinality")
        for entry in part[role]:
            require(entry["group_uid"] not in uid_role, "unique partition UID: " + entry["group_uid"])
            uid_role[entry["group_uid"]] = (role, entry["domain"])
    def seed_for(seed, *parts):
        payload = (json.dumps([seed, *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    def schedule(uids, order, stage):
        rng = random.Random(seed_for(baseline_policy["schedule_seed"], order, stage, "current"))
        values = []
        for epoch in range(6):
            perm = sorted(uids)
            rng.shuffle(perm)
            values.extend(perm)
        return values

    require(manifest["physical_updates"] == 1728 and manifest["gradient_group_presentations"] == 3456 and
            manifest["logical_updates"] == 2592, "manifest distinct physical/group/logical counts")
    require(len(manifest["points"]) == len(manifest["training"]) == 9, "9 logical endpoint/training records")
    points = []
    all_updates = []
    max_total_error = max_base_error = 0.0
    for order in policy["orders"]:
        reservoir = []
        seen = 0
        rr = random.Random(seed_for(20260930, order, "retention"))
        for stage in (1, 2, 3):
            name = f"{order}_function_memory_stage{stage}"
            identity(r / "job/run" / manifest["points"][name]["path"], manifest["points"][name])
            identity(r / "job/run" / manifest["training"][name]["path"], manifest["training"][name])
            point = load(r / "job/run" / manifest["points"][name]["path"])
            tr = load(r / "job/run" / manifest["training"][name]["path"])
            fit = [x["group_uid"] for x in part["fit"] if x["domain"] == order[stage - 1]]
            expected = schedule(fit, order, stage) if stage > 1 else []
            hr = random.Random(seed_for(20260930, order, stage, "history_draws"))
            history = [reservoir[hr.randrange(6)] for _ in range(288)] if stage > 1 else []
            require(tr["current_ids"] == expected and tr["history_ids"] == history,
                    name + " exact independent current and history sequences")
            require(len(tr["updates"]) == (288 if stage > 1 else 0), name + " physical update count")
            require(point["order"] == tr["order"] == order and point["stage"] == tr["stage"] == stage,
                    name + " endpoint path identity")
            require(point["adam_step"] == tr["adam_step"] == stage * 288 and point["full_restore_verified"] is True,
                    name + " completed Adam count and restoration record")
            for i, row in enumerate(tr["updates"], 1):
                require(row["stage"] == stage and row["step"] == i and row["adam_step"] == (stage - 1) * 288 + i,
                        name + f" step {i} numbering")
                require(all(math.isfinite(v) for v in row.values()), name + f" step {i} finite")
                total_error = abs(row["total"] - (row["current_total"] + .1 * row["history_total"] + .5 * row["function"]))
                max_total_error = max(max_total_error, total_error)
                require(total_error <= 1e-12, name + f" step {i} actual scalar objective")
                base_error = max(abs(row[f"{side}_total"] - (row[f"{side}_bce"] + row[f"{side}_rank"] + .5 * row[f"{side}_hard"]))
                                 for side in ("current", "history"))
                max_base_error = max(max_base_error, base_error)
                # B is a float32 tensor sum; decoded components are Python doubles.
                require(base_error <= 1e-6, name + f" step {i} full base scalar within float32 roundoff")
                lr = 1e-5 * (i / 29 if i <= 29 else (288 - i) / 259)
                require(row["encoder_lr"] == lr and row["head_lr"] == .001, name + f" step {i} frozen learning rates")
            all_updates.extend(tr["updates"])
            old_reservoir = list(reservoir)
            old_seen = seen
            if stage < 3:
                for uid in sorted(fit):
                    seen += 1
                    j = len(reservoir) if len(reservoir) < 6 else rr.randrange(seen)
                    if j < 6:
                        if j == len(reservoir): reservoir.append(uid)
                        else: reservoir[j] = uid
            require(point["memory_summary"] == {"count": seen, "seen": seen, "stage": min(stage, 2), "members": reservoir},
                    name + " independent Algorithm R and cumulative N")
            require(point["memory"]["bytes"] == point["memory_bytes"] <= 1048576,
                    name + " full Memory byte bound record")
            for kind, ledger in (("model", models), ("memory", memories)):
                item = point[kind]
                recorded = ledger["job/run/" + item["path"]]
                require(item["bytes"] == recorded["bytes"] and item["sha256"] == recorded["sha256"],
                        name + " " + kind + " metadata matches independent sync ledger")
            for record in [point["mapping"], *point["scores"].values()]:
                identity(r / "job/run" / record["path"], record)
            require(point["calibration_ids"] == [x["group_uid"] for x in part["calibration"] if x["domain"] == order[stage - 1]],
                    name + " calibration groups belong to current domain only")
            retention = point["retention"]
            if stage < 3:
                require(retention["old_count"] == old_seen and retention["old_members"] == old_reservoir and
                        retention["new_count"] == seen and retention["new_members"] == reservoir and
                        retention["reservoir_seen"] == seen and retention["serialized_bytes"] == point["memory_bytes"],
                        name + " retention event consistent with stage and memory")
            else:
                require(retention is None, name + " no consolidation without successor stage")
            if stage == 1:
                shared = manifest["restored_starts"][order]
                require(shared["adam_step"] == 288 and shared["model_state_sha256"] == point["model_state_sha256"] and
                        shared["first_scores_replayed_exactly"] is True, name + " shared recorded model identity and no new updates")
            rows = tr["updates"]
            rec = {"name": name, "updates": len(rows), "memory_bytes": point["memory_bytes"],
                   "memory_count": seen, "cache_domains": dict(collections.Counter(uid_role[u][1] for u in reservoir)),
                   "first_function": rows[0]["function"] if rows else None,
                   "mean_function": statistics.mean(x["function"] for x in rows) if rows else None,
                   "max_function": max(x["function"] for x in rows) if rows else None,
                   "mean_current_B": statistics.mean(x["current_total"] for x in rows) if rows else None,
                   "mean_history_B": statistics.mean(x["history_total"] for x in rows) if rows else None,
                   "gradient_norm_min": min(x["gradient_norm"] for x in rows) if rows else None,
                   "gradient_norm_max": max(x["gradient_norm"] for x in rows) if rows else None,
                   "T_singular_extrema": [min(retention["transport_singular_values"]), max(retention["transport_singular_values"])]
                       if retention and "transport_singular_values" in retention else None}
            points.append(rec)
    require(len(all_updates) == 1728, "independently summed all physical updates = 1728")
    require(all(x["gradient_norm"] > 1 for x in all_updates), "all 1728 saved preclip norms exceed 1")
    result["update_evidence"] = {"points": points, "all_new_updates": len(all_updates),
                                  "max_objective_reconstruction_error": max_total_error,
                                  "max_B_component_roundoff": max_base_error,
                                  "all_preclip_norms_exceed_1": True,
                                  "gradient_norm_range": [min(x["gradient_norm"] for x in all_updates), max(x["gradient_norm"] for x in all_updates)]}

    # Read and hash historical evidence; do not rerun historical CPU/GPU tests.
    hist_zip = root / "history/function_memory_implementation_external_evidence.zip"
    hist_blob = raw(hist_zip, "original implementation audit ZIP")
    require(len(hist_blob) == 13204125 and hashlib.sha256(hist_blob).hexdigest() ==
            "545ec223fcf2b7408568310a19589c8fc203a721ab8a015e48c9d8435f3a589a", "original implementation evidence archive identity")
    with zipfile.ZipFile(io.BytesIO(hist_blob)) as z:
        hmanifest = json.loads(z.read("manifest.json"))
        require(len(z.infolist()) == 87 and len(hmanifest["files"]) == 86, "historical 87 members / 86 manifest payloads")
        for item in hmanifest["files"]:
            b = z.read(item["path"])
            require(len(b) == item["bytes"] and hashlib.sha256(b).hexdigest() == item["sha256"],
                    "historical payload identity: " + item["path"])
        report = z.read("FUNCTION_MEMORY_IMPLEMENTATION.external_review.zh.md")
        require(hashlib.sha256(report).hexdigest() == native_gate["external_review_sha256"],
                "native admission binds actual historical implementation review original")
        differences = {}
        for record in source:
            rel = record["path"]
            old_name = "input/" + rel
            if old_name not in z.namelist():
                continue
            old, new = z.read(old_name), raw(src / rel, "historical-to-formal source delta")
            if old != new:
                differences[rel] = "\n".join(difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(),
                                                                 fromfile="implementation-review", tofile="formal-source", n=4))
        require(set(differences) == {"scripts/step28_function_memory_run.py", "scripts/step28_function_memory_verify.py",
                                    "tests/test_step28_function_memory_run.py"},
                "only R1/F1 entry changes, native observation and corresponding test file changed since implementation review")
        require(z.read("input/scripts/step28_function_memory.py") == raw(src / "scripts/step28_function_memory.py"),
                "mathematical production core unchanged since actual implementation review")
        (out / "historical_to_formal.diff.txt").write_text("\n\n".join(k + "\n" + v for k, v in differences.items()), encoding="utf-8")
    result["history"] = {"members": 87, "payloads_verified": 86, "changed_files": list(differences),
                         "core_unchanged": True, "historical_tests_rerun": False}

    # Static function examination only: neither function is called/imported here.
    runner_text = raw(src / "scripts/step28_function_memory_run.py", "static access and repair inspection").decode()
    verify_text = raw(src / "scripts/step28_function_memory_verify.py", "static R1 repair inspection").decode()
    runner_tree, verify_tree = ast.parse(runner_text), ast.parse(verify_text)
    vg = next(n for n in runner_tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_gate")
    vgtext = ast.get_source_segment(runner_text, vg)
    require('evidence.get("mode")' in vgtext and 'native_handwritten_first_optimizer_step' in vgtext,
            "current gate has actual mode/kind correction")
    mainfn = next(n for n in verify_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    watchdog = next(n for n in mainfn.body if isinstance(n, ast.FunctionDef) and n.name == "watchdog")
    require(any(isinstance(n, ast.Try) and n.finalbody and any(isinstance(k, ast.Call) and isinstance(k.func, ast.Attribute)
                and k.func.attr == "_exit" for final in n.finalbody for k in ast.walk(final)) for n in ast.walk(watchdog)),
            "current watchdog finally executes exit even when receipt write raises")
    result["repaired_admission"] = {"cpu_tests": 9, "cpu_failures": 0, "cpu_errors": 0, "cpu_skipped": 0,
                                     "native_input_shapes": native["input_shapes"],
                                     "native_D_only_gradient_norms": native["history_only_gradient_norms"],
                                     "native_modes": 97, "R1_current_static_and_bound_original_tests": "closed",
                                     "F1_current_static_and_bound_original_tests": "closed",
                                     "incremental_external_review": "waived, not claimed performed",
                                     "result_external_review": "still required by frozen contract section 7"}
    result["limits"] = ["162 supplied nonweight payloads directly hashed; 9 Memory and 9 model bodies intentionally absent and not requested/read",
                         "No direct validation of source text/labels, actual model states or Memory tensor contents",
                         "Successful restore and shared model/Adam/RNG have recorded runtime evidence; no model reload performed by this auditor",
                         "Server timestamps/owner/path/status are preserved records; no live server inspection performed",
                         "Original pre-retry CPU failure originals were deleted by explicit user direction; brief failure/deletion/authorization preserved",
                         "New analysis-script syntax/path corrections are audit/analysis operations, not formal training failures"]
    result["status"] = "PASS_WITH_DECLARED_EVIDENCE_LIMITS"
    result["scientific_blockers_identified"] = 0
    result["current_reproducibility_defects_identified"] = 0
    result["check_count"] = len(checks)
    result["elapsed_seconds"] = time.monotonic() - tick
    (out / "provenance_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "provenance_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "read_scope.json").write_text(json.dumps(list(read_scope.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": len(checks), "outer_payloads": 252,
                      "sources": 23, "physical_updates": len(all_updates), "max_total_error": max_total_error,
                      "gradient_norm_range": result["update_evidence"]["gradient_norm_range"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
