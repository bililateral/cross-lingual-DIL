#!/usr/bin/env python3
"""Read-only independent reconstruction of LOGIT0.1 saved run evidence.

No project modules, Torch, native model, cache payload, or formal label are loaded.
ID-only schedules and reservoir membership are independently reconstructed.
The native handmade arrays are reused evidence, not new native execution.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import traceback

import numpy as np


def main(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    checks = []
    summary = {"status": "RUNNING", "scope": "LOGIT0.1 saved execution path only",
               "argv": sys.argv, "python": platform.python_version(), "numpy": np.__version__,
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_imports": False, "models_loaded": False,
               "cache_payloads_read": False, "formal_label_parses": 0,
               "formal_updates": 0, "checks": checks}

    def load_bytes(path):
        path = Path(path)
        data = path.read_bytes()
        inputs[str(path.relative_to(root))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def read(path):
        return json.loads(load_bytes(path))

    def load_array(path):
        from io import BytesIO
        return np.load(BytesIO(load_bytes(path)), allow_pickle=False)

    def require(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    def verified(path, record):
        blob = load_bytes(path)
        require(len(blob) == record["bytes"] and hashlib.sha256(blob).hexdigest() == record["sha256"],
                "record identity: " + str(Path(path).relative_to(root)))

    def seed(parts):
        blob = (json.dumps(parts, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n").encode()
        return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**63 - 1)

    def json_size(value):
        return len((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())

    try:
        package = root / "reports/seller_alias_continual/20261004/logit_low_result"
        src, job = package / "source", package / "job"
        run = job / "run"
        baseline = root / "dependencies/shared"
        policy = read(src / "schema/step28_logit_low_policy.json")
        parent = read(src / "schema/step28_bge_continual_policy.json")
        execution = read(job / "execution.json")
        manifest = read(run / "manifest.json")
        oldmanifest = read(baseline / "run/manifest.json")
        part = read(run / "partition.json")
        oldpart = read(baseline / "run/partition.json")
        collected = read(job / "evaluation/collected.json")
        auth = read(package / "authorization.json")
        cpuaudit = read(root / "direct_reviews/logit_cpu/audit.json")
        source_list = execution["source_files"]
        require(len(source_list) == 37 and len({r["path"] for r in source_list}) == 37,
                "37 unique actually executed scientific sources")
        for record in source_list:
            verified(src / record["path"], record)
        require(source_list == manifest["source_files"] == collected["source_files"] == cpuaudit["source_files"] == auth["source_files"],
                "execution/manifest/collected/authorization/native CPU source chains match")
        verified(root / "direct_reviews/logit_cpu/audit.json", execution["audit"])
        verified(package / "authorization.json", execution["authorization"])
        require(manifest["policy_sha256"] == inputs[str((src / "schema/step28_logit_low_policy.json").relative_to(root))]["sha256"] == auth["policy_sha256"],
                "actual frozen policy identity matches authorization and manifest")
        require(policy["arms"] == {"logit_tenth": .1} and policy["loss"]["logit_mse"] == .5,
                "frozen history weight .1 and separate MSE weight .5")
        require(part == oldpart, "current partition equals shared frozen partition")
        require(len(part["fit"]) == 144 and len(part["calibration"]) == 36 and len(part["development"]) == 60,
                "144 fit / 36 calibration / 60 development groups")
        for role, per_domain in [("fit", 48), ("calibration", 12), ("development", 20)]:
            require(Counter(r["domain"] for r in part[role]) == Counter({d: per_domain for d in "ABC"}),
                    role + " actual-domain sizes")
        role_ids = [{r["group_uid"] for r in part[x]} for x in ("fit", "calibration", "development")]
        require(all(not role_ids[i] & role_ids[j] for i in range(3) for j in range(i)), "saved fit/cal/development IDs are disjoint")
        require(collected["group_ids"] == [r["group_uid"] for r in part["development"]]
                and collected["domains"] == [r["domain"] for r in part["development"]]
                and len(collected["metric_columns"]) == 22, "60-group/22-column collected row identity")
        for name, spec in policy["continuation_references"].items():
            for record in spec["records"].values():
                verified(root / "dependencies" / name / record["path"], record)
        for record in policy["baseline"]["records"].values():
            verified(baseline / record["path"], record)

        summary["stages"] = []
        summary["restored_starts"] = []
        residuals = []
        memory_sizes = []
        all_grads = []
        generated_target_ids = {}
        expected_names = {f"{o}_logit_tenth_stage{s}" for o in ("ABC", "BCA", "CAB") for s in (2, 3)}
        require(set(manifest["points"]) == expected_names == set(manifest["training"]), "all six expected points and training summaries exist")
        require(set(collected["points"]) == expected_names, "all six points were collected together")
        raw_arrays = 0
        metric_sets = 0
        for order in ("ABC", "BCA", "CAB"):
            shared = read(baseline / "run/points" / (order + "_shared.json"))
            start = manifest["restored_starts"][order + "_logit_tenth"]
            oldmemory = oldmanifest["memories"][order + "_logit_after1"]
            require(start["full_checkpoint"] == shared["full_checkpoint"] and start["adam_step"] == 288
                    and start["model_state_sha256"] == shared["model_state_sha256"]
                    and start["first_map"] == shared["first_map_parameters"]
                    and start["memory_source"] == oldmemory["file"] and start["first_scores_replayed_exactly"],
                    order + " shared model/Adam288/first-map/source and full first-score replay records agree")
            require(start["memory_summary"]["references"] == oldmemory["references"]
                    and start["memory_summary"]["reference_origins"] == oldmemory["reference_origins"]
                    and set(oldmemory["reference_origins"].values()) == {1}, order + " first-domain references are unchanged")
            first_members = start["memory_summary"]["members"]
            reservoir, seen = [], 0
            retain_rng = random.Random(seed([parent["memory_seed"], order, "retention"]))
            expected_members = {}
            for origin in (1, 2):
                # ID-only Algorithm R, independent of model scores or retained payload.
                for uid in sorted(r["group_uid"] for r in part["fit"] if r["domain"] == order[origin - 1]):
                    seen += 1
                    if len(reservoir) < 6:
                        reservoir.append(uid)
                    else:
                        replace = retain_rng.randrange(seen)
                        if replace < 6:
                            reservoir[replace] = uid
                expected_members[origin] = reservoir.copy()
            retained = manifest["memories"][order + "_logit_tenth_stage2"]
            require(first_members == expected_members[1] == oldmemory["members"], order + " independent six-group first reservoir")
            require(retained["members"] == expected_members[2] == oldmanifest["memories"][order + "_er_stage2"]["members"],
                    order + " independent six-group second reservoir")
            refs = retained["references"]
            require(set(refs) == set(retained["members"]), order + " six retained references cover exactly surviving groups")
            survivors = set(refs) & set(oldmemory["references"])
            require(all(refs[k] == oldmemory["references"][k] for k in survivors), order + " old surviving target digests not refreshed")
            new_ids = set(refs) - set(oldmemory["references"])
            generation = retained["reference_update"]
            require(set(generation["new_target_ids"]) == new_ids and generation["mode"] == "eval" and generation["old_survivors_unchanged"],
                    order + " target generation restricted to branch-new retained groups")
            fit_domain = {r["group_uid"]: r["domain"] for r in part["fit"]}
            require(all(fit_domain[k] == order[v - 1] and v in (1, 2) for k, v in retained["reference_origins"].items()),
                    order + " target origins agree with actual fitting domains")
            memory_sizes.extend([start["memory_summary"]["serialized_bytes"], retained["serialized_bytes"]])
            generated_target_ids[order] = sorted(new_ids)
            summary["restored_starts"].append({"order": order, "adam_step": start["adam_step"],
                    "model_state_sha256": start["model_state_sha256"], "first_map": start["first_map"],
                    "first_scores_replayed_exactly_record": True, "new_target_count_after_stage2": len(new_ids),
                    "first_score_replay_rerun_here": False})

            for stage in (2, 3):
                name = f"{order}_logit_tenth_stage{stage}"
                log = read(run / "updates" / (name + ".json"))
                oldlog = read(baseline / "run/updates" / f"{order}_er_stage{stage}.json")
                point = read(run / "points" / (name + ".json"))
                stage_budget = read(run / "memory" / (name + "_budget.json"))
                verified(run / manifest["training"][name]["path"], manifest["training"][name])
                verified(run / manifest["points"][name]["path"], manifest["points"][name])
                require(all(log[k] == oldlog[k] for k in ["current_ids", "history_ids", "current_dropout_stream", "adam_step", "updates"]),
                        name + " matched original ER supply and logical update count")
                current_uids = sorted(r["group_uid"] for r in part["fit"] if r["domain"] == order[stage - 1])
                stream = seed([parent["schedule_seed"], order, stage, "current"])
                shuffle = random.Random(stream)
                expected_current = []
                for epoch in range(6):
                    slots = list(current_uids)
                    shuffle.shuffle(slots)
                    expected_current.extend(slots)
                    require(set(log["current_ids"][48*epoch:48*(epoch+1)]) == set(current_uids), name + f" epoch {epoch+1} exactly current 48 groups")
                require(log["current_dropout_stream"] == stream and log["current_ids"] == expected_current,
                        name + " independent current schedule and RNG seed")
                members = expected_members[stage - 1]
                draw = random.Random(seed([parent["memory_seed"], order, stage, "history_draws"]))
                expected_history = [members[draw.randrange(6)] for _ in range(288)]
                require(log["history_ids"] == expected_history, name + " independent history draw sequence")
                require(not set(log["history_ids"]) & set(current_uids), name + " historical and current groups disjoint")
                current_seeds = [seed([stream, i, "dropout"]) for i in range(288)]
                history_seeds = [seed([parent["memory_seed"], order, stage, i, "history_dropout"]) for i in range(288)]
                require(not set(current_seeds) & set(history_seeds), name + " explicit current/history dropout seed streams are separate")
                verified(run / log["update_file"]["path"], log["update_file"])
                arr = load_array(run / log["update_file"]["path"])
                require(arr.shape == (288, 16) and arr.dtype == np.float64 and np.isfinite(arr).all(), name + " all 288x16 finite update rows")
                col = {k: arr[:, i] for i, k in enumerate(log["update_columns"])}
                require(np.all(col["history_weight"] == .1) and np.all(col["logit_weight"] == .5), name + " 288 fixed loss coefficients")
                base_errors = []
                for role in ("current", "history"):
                    ref_loss = col[role + "_bce"] + col[role + "_rank"] + .5 * col[role + "_hard"]
                    err = np.abs(ref_loss - col[role + "_total"])
                    base_errors.append(float(err.max()))
                    residuals.extend(err.tolist())
                    require(np.all(err <= 3e-6 + 3e-6 * np.abs(ref_loss)), name + " " + role + " independent base loss decomposition")
                expected_total = col["current_total"] + .1*col["history_total"] + .5*col["logit_mse"]
                require(np.array_equal(expected_total, col["total"]), name + " independent unnormalized total, MSE not scaled by .1")
                require(np.array_equal(.1*col["history_total"], col["weighted_history_total"]), name + " complete historical loss times .1")
                require(np.all(col["logit_mse"] > 0), name + " saved actual MSE term nonzero at every update")
                expected_lr = np.array([1e-5*(i/29 if i <= 29 else (288-i)/259) for i in range(1,289)])
                require(np.array_equal(expected_lr, col["encoder_lr"]) and np.all(col["head_lr"] == .001), name + " independent 29-step warmup and decay schedule")
                all_grads.extend(col["gradient_norm"].tolist())
                require(set(log["observations"]) == {"1", "29", "30", "288"}, name + " four required parameter observations present")
                for step, obs in log["observations"].items():
                    for module in ("encoder", "head"):
                        require(obs[module]["finite_nonzero_combined_gradient"] and obs[module]["parameters_changed"] == (module == "head" or int(step) != 288),
                                name + f" {module} observed at step {step}, last encoder LR zero respected")
                expected_cache = oldmemory if stage == 2 else retained
                consumed = log["memory_after_training"]
                require(all(consumed[k] == expected_cache[k] for k in ("members", "with_logits", "references", "reference_origins"))
                        and consumed["draw_stage"] == stage and consumed["draw_count"] == 288,
                        name + " consumed cache descriptor has correct frozen targets and 288 draws")
                require(stage_budget["auxiliary_serialized_bytes"] == point["learner_auxiliary_serialized_bytes"] == json_size(point["learner_auxiliary"]),
                        name + " serialized learner auxiliary byte charge reconstructed from saved metadata")
                memory_sizes.extend([consumed["serialized_bytes"], stage_budget["serialized_bytes"]])
                metadata = point["learner_auxiliary"]["checkpoint_metadata"]
                require(metadata["completed_updates"] == point["completed_updates"] == log["adam_step"] == 288*stage
                        and metadata["history_weight"] == .1 and metadata["logit_weight"] == .5
                        and point["history_weight"] == .1 and point["full_model_adam_and_rng_restore_verified"],
                        name + " full endpoint metadata and logical Adam step agree")
                require(len(metadata["rng"]["cpu"]) > 0 and len(metadata["rng"]["cuda"]) == 1,
                        name + " retained CPU and single-GPU RNG descriptors present")
                require(point["first_map_parameters"] == start["first_map"], name + " fixed first map unchanged")
                if stage == 2:
                    require(generation["model_state_sha256"] == point["model_state_sha256"], name + " new targets bound to this branch endpoint")
                mapping = read(run / point["mapping"]["path"])
                verified(run / point["mapping"]["path"], point["mapping"])
                expected_cal_ids = [r["group_uid"] for r in part["calibration"] if r["domain"] == order[stage-1]]
                require(mapping["calibration_group_ids"] == expected_cal_ids and mapping["actual_domain"] == order[stage-1]
                        and mapping["group_count"] == 12 and mapping["pair_count"] == 12*378
                        and mapping["positive_count"] == 12*20 and mapping["status"] == "PASS_CALIBRATION_FIT",
                        name + " calibration descriptors identify exactly current 12 groups, 4536 pairs, 240 positives")
                require(mapping["a"] > 0 and mapping["score_source"] == point["scores"]["calibration"]
                        and mapping["model_state_sha256"] == point["model_state_sha256"], name + " positive affine map bound to endpoint scores")
                score = {}
                for role, rec in point["scores"].items():
                    verified(run / rec["path"], rec)
                    score[role] = load_array(run / rec["path"])
                    shape = (12,378) if role == "calibration" else (60,378)
                    dtype = np.float32 if role in ("calibration", "development") else np.float64
                    require(score[role].shape == shape and score[role].dtype == dtype and np.isfinite(score[role]).all(),
                            name + " complete saved score role: " + role)
                    raw_arrays += 1
                raw = score["development"].astype(np.float64)
                raw_order = np.argsort(raw, axis=1, kind="stable")
                for role, usemap in [("stage-cal", mapping), ("first-cal", start["first_map"])]:
                    transformed = raw*usemap["a"] + usemap["b"]
                    require(np.array_equal(transformed, score[role]), name + " independent affine transform: " + role)
                    require(np.array_equal(np.argsort(score[role],axis=1,kind="stable"), raw_order), name + " score ordering and stable tie order preserved: " + role)
                    sorted_raw = np.take_along_axis(raw,raw_order,axis=1)
                    sorted_out = np.take_along_axis(score[role],raw_order,axis=1)
                    require(np.array_equal(np.diff(sorted_raw,axis=1)==0,np.diff(sorted_out,axis=1)==0), name + " exact tie blocks preserved: " + role)
                for role, recs in collected["points"][name].items():
                    require(role in ("raw","stage-cal","first-cal"), name + " legitimate metric role")
                    for rec in recs.values():
                        verified(job / "evaluation" / rec["path"], rec)
                    matrix = load_array(job / "evaluation" / recs["matrix"]["path"])
                    require(matrix.shape == (60,22) and matrix.dtype == np.float64 and np.isfinite(matrix).all(), name + " complete metric matrix " + role)
                    metric_sets += 1
                summary["stages"].append({"name": name, "updates": 288, "adam_step": log["adam_step"],
                    "current_unique": len(set(log["current_ids"])), "history_unique": len(set(log["history_ids"])),
                    "current_dropout_stream": stream,
                    "current_dropout_seeds_sha256": hashlib.sha256(json.dumps(current_seeds).encode()).hexdigest(),
                    "history_dropout_seeds_sha256": hashlib.sha256(json.dumps(history_seeds).encode()).hexdigest(),
                    "current_base_decomposition_max_abs_error": base_errors[0], "history_base_decomposition_max_abs_error": base_errors[1],
                    "total_max_abs_error": float(np.abs(col["total"]-expected_total).max()),
                    "mean_current_base": float(col["current_total"].mean()),
                    "mean_weighted_history_base": float((.1*col["history_total"]).mean()),
                    "mean_half_mse": float((.5*col["logit_mse"]).mean()),
                    "half_mse_min": float((.5*col["logit_mse"]).min()),
                    "half_mse_max": float((.5*col["logit_mse"]).max()),
                    "preclip_gradient_norm_min": float(col["gradient_norm"].min()),
                    "preclip_gradient_norm_max": float(col["gradient_norm"].max()),
                    "calibration_a": mapping["a"], "calibration_b": mapping["b"],
                    "recorded_calibration_positive_count": mapping["positive_count"],
                    "max_recorded_memory_bytes": max(consumed["serialized_bytes"], stage_budget["serialized_bytes"]),
                    "cuda_allocator": log["cuda_allocator"], "training_seconds": log["training_seconds"]})

        require(raw_arrays == 24 and metric_sets == 18, "24 complete saved score arrays and 18 new metric/count sets")
        require(sum(x["updates"] for x in summary["stages"]) == manifest["physical_updates"] == 1728
                and manifest["gradient_group_presentations"] == 3456, "saved stage rows total 1728 updates / 3456 group-gradient presentations")
        require(max(memory_sizes) <= 1048576, "all observed complete memory/auxiliary charges below 1MiB")
        summary["maximum_recorded_memory_bytes"] = max(memory_sizes)
        summary["maximum_base_loss_decomposition_error"] = max(residuals)
        summary["update_rows"] = 1728
        summary["gradient_norm_min"] = min(all_grads)
        summary["gradient_norm_max"] = max(all_grads)
        summary["updates_with_gradient_norm_above_one"] = sum(x > 1 for x in all_grads)
        summary["new_reference_counts"] = {k:len(v) for k,v in generated_target_ids.items()}

        before = read(job / "before_valid.json")
        access = read(job / "access.json")
        completion = read(job / "completion.json")
        verified(job / before["manifest"]["path"], before["manifest"])
        require(before["label_parses"] == {"train":1,"valid":0,"heldout":0,"owners":0}, "complete blind gate records valid0/train1")
        require(access == completion["label_parses"] == {"train":1,"valid":1,"heldout":0,"owners":0}, "final access records train1/valid1/heldout0/owners0")
        require(load_bytes(job/"exit_status.txt").strip() == b"0" and not (job/"failure.json").exists(), "formal wrapper exit0; no saved failure")
        start_time = datetime.fromisoformat(load_bytes(job/"started.txt").decode().strip())
        finish_time = datetime.fromisoformat(load_bytes(job/"finished.txt").decode().strip())
        wall = (finish_time-start_time).total_seconds()
        require(0 < completion["budget"]["elapsed_seconds"] <= policy["runtime"]["maximum_gpu_stage_seconds"]
                and completion["budget"]["peak_observed_bytes"] <= policy["runtime"]["maximum_output_bytes"], "time/output observation within fixed 12h / 16GiB limits")
        train_text = load_bytes(job/"train.log").decode()
        events = [json.loads(line) for line in train_text.splitlines() if line.startswith("{")]
        updates = [x for x in events if x.get("event") == "updates"]
        require(len(updates) == 72, "72 real progress records (6 stages x 12 intervals)")
        for name in expected_names:
            order, stage = name[:3], int(name[-1])
            batch = [x for x in updates if x["order"] == order and x["stage"] == stage]
            require([x["stage_updates"] for x in batch] == list(range(24,289,24))
                    and all(x["logical_updates"] == 288*(stage-1)+x["stage_updates"] for x in batch), name + " progress has no extra/duplicate/restarted interval")
        require(all(updates[i]["elapsed_seconds"] < updates[i+1]["elapsed_seconds"] for i in range(len(updates)-1)), "saved progress elapsed times monotonic")
        summary["runtime"] = {"started":start_time.isoformat(), "finished":finish_time.isoformat(),
                              "timestamp_difference_seconds":wall, **completion["budget"],
                              "access_before_valid":before["label_parses"],"final_access":access,
                              "progress_events":len(updates)}

        custody = read(package/"verification/payload_custody.json")
        weights = custody["weights_retained_linux"]
        require(len(weights) == 6 and not custody["models_loaded"] and not custody["formal_labels_parsed"], "Linux custody lists six hashes without model load or formal parsing")
        by_name = {Path(x["path"]).stem:x for x in weights}
        for name in expected_names:
            point = read(run/"points"/(name+".json"))
            require(all(point["model"][k] == by_name[name][k] for k in ("bytes","sha256")), name + " inference descriptor matches fresh Linux custody record")
            require(not (run / point["model"]["path"]).exists(), name + " native model not in current attachment")
        summary["weight_custody_total_bytes"] = sum(r["bytes"] for r in weights)

        # Recompute scalar MSE from saved handmade probe logits/targets, not formal data.
        cpu = root / "direct_reviews/logit_cpu"
        er, logit = read(cpu/"tenth.json"), read(cpu/"logit_tenth.json")
        for record in list(cpuaudit["native"].values()) + list(cpuaudit["native_reference"].values()):
            verified(cpu / record["path"], record)
        require(er["native_updates_actually_executed"] == logit["native_updates_actually_executed"] == 2,
                "reused native evidence is exactly two updates per arm, four total")
        require(er["initial_model_state_sha256"] == logit["initial_model_state_sha256"]
                and er["after_warm_model_state_sha256"] == logit["after_warm_model_state_sha256"]
                and er["before_stage2_optimizer_sha256"] == logit["before_stage2_optimizer_sha256"]
                and er["captured_current_logits"] == logit["captured_current_logits"]
                and er["captured_history_logits"] == logit["captured_history_logits"], "native reused paired forward inputs and pre-update states agree")
        mse = math.fsum((a-b)**2 for a,b in zip(logit["captured_history_logits"],logit["origin_eval_reference"],strict=True))/378
        require(abs(mse-logit["independent_logit_mse"]) < 1e-15,
                "handmade native probe MSE independently recomputed from saved 378 raw logits/target")
        require(er["after_model_state_sha256"] != logit["after_model_state_sha256"]
                and all(x["norm"] > 0 for x in logit["mse_gradient_reference"].values()),
                "reused native evidence has nonzero MSE gradient probes and differing post-update models")
        summary["reused_native_scalar_check"] = {"raw_mse":mse,"half_mse":.5*mse,
                "mse_log_abs_error":abs(mse-logit["weighted_update"]["logit_mse"]),
                "new_native_updates_here":0,
                "gradient_tensors_available_here":False,
                "recorded_gradient_decomposition":cpuaudit["native_component_comparison"]}
        summary["limitations"] = [
            "Native model/Adam/RNG restoration and all-60-group replay were checked through frozen code and recorded descriptors; not reexecuted here.",
            "No cache payloads are supplied; membership/provenance/byte summaries are independently cross-checked without deserializing texts/labels.",
            "Only saved metadata supports calibration truth counts. No calibration fit or AP/MAP label-level recomputation here.",
            "Update loss reconstruction validates recorded arithmetic and scheduling; the actual computational graph is supported separately by frozen source and directly reusable prior native gradient evidence.",
            "Fresh native weight hashes are supplied Linux evidence; webpage did not hash/read the missing original model bytes.",
        ]
        summary["status"] = "PASS_WITH_DECLARED_EVIDENCE_BOUNDARIES"
        summary["confirmed_scientific_blockers"] = []
        summary["delivery_reproducibility_defects"] = []
    except Exception:
        summary["status"] = "REFERENCE_FAILED"
        summary["error"] = traceback.format_exc()
        print(summary["error"],file=sys.stderr)
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["checks_passed"] = len(checks)
        summary["torch_imported"] = "torch" in sys.modules
        (output/"result.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
        (output/"input_inventory.json").write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({k:summary[k] for k in ["status","checks_passed","torch_imported"]},ensure_ascii=False))
    return 0 if summary["status"] == "PASS_WITH_DECLARED_EVIDENCE_BOUNDARIES" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.root,args.output))
