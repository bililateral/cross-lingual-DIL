"""Independent, read-only audit of saved Optuna execution evidence.

No project module imports, model loads, formal labels, network, or training.
Input: extracted submitted result bundle. Output: local review evidence only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root, out = args.input_root.resolve(), args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scope, problems, checks = {}, [], collections.Counter()

    def record(p, method, detail):
        p = p.resolve()
        rel = p.relative_to(root).as_posix()
        row = scope.setdefault(rel, {"path": rel, "methods": [], "details": []})
        if method not in row["methods"]:
            row["methods"].append(method)
        if detail not in row["details"]:
            row["details"].append(detail)

    def read(p):
        record(p, "MACHINE_JSON_CHECK", "JSON fields parsed by audit_execution.py; not a claim of human full-text reading")
        return json.loads(p.read_text())

    def assert_(condition, tag, detail):
        checks[tag] += 1
        if not condition:
            problems.append({"check": tag, "detail": detail})

    def verify_file(p, expected):
        record(p, "HASH_AND_SIZE", "Cross-check against the enclosing native manifest, not merely the outer ZIP manifest")
        data = p.read_bytes()
        assert_(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], "embedded_identity", str(p.relative_to(root)))

    def array(p):
        record(p, "MACHINE_ARRAY_CHECK", "Saved score array loaded with allow_pickle=False; no labels")
        return np.load(p, allow_pickle=False)

    job = root / "result/job"
    runs = sorted(p for p in job.iterdir() if p.is_dir() and p.name.startswith(("search_", "replication_")))
    assert_(len(runs) == 45, "run_count", len(runs))
    expected_ids = {f"search_{n:02}_{a}_s0" for n in range(12) for a in ("S", "S_pos", "H_pos")}
    expected_ids |= {f"replication_{n:02}_{a}_s{n+1}" for n in range(3) for a in ("S", "S_pos", "H_pos")}
    assert_({p.name for p in runs} == expected_ids, "exact_run_id_set", [p.name for p in runs])
    top = read(job / "completion.json")
    source_current = {x["path"]: x for x in top["source_files"]}
    for s in source_current.values():
        verify_file(root / "result/sources" / s["path"], s)
    source_counts, source_versions = collections.Counter(), collections.defaultdict(set)
    per_run, source_changes = [], []
    total_updates = total_presentations = total_endpoints = total_captures = 0
    access_total = collections.Counter()
    partition_hashes = set()
    initial_by_seed = collections.defaultdict(set)
    positive_gradient_values = 0
    model_states = set()

    for p in runs:
        c = read(p / "completion.json")
        b = read(p / "before_valid.json")
        e = read(p / "execution.json")
        m = read(p / "run/manifest.json")
        acc = read(p / "access.json")
        spec = c["trial"]
        phase, arm, seed = spec["phase"], spec["arm"], spec["seed"]
        sources = m["source_files"]
        source_counts[len(sources)] += 1
        assert_(sources == c["source_files"] == b["source_files"] == e["source_files"], "run_source_manifest_consistency", p.name)
        assert_(b["trial"] == e["trial"] == spec, "run_trial_consistency", p.name)
        assert_(b["status"] == "ALL_BLIND_GATES_PASS", "blind_gate_status", p.name)
        assert_(c["status"] == ("SEARCH_COMPLETE" if phase == "search" else "REPLICATION_BLIND_COMPLETE"), "completion_status", p.name)
        assert_(acc == c["access"] == {"train": 1, "valid": int(phase == "search"), "heldout": 0, "owners": 0}, "run_access", p.name)
        assert_(not (p / "failure.json").exists(), "no_run_failure", p.name)
        access_total.update(acc)
        for s in sources:
            verify_file(p / "run/sources" / s["path"], s)
            source_versions[s["path"]].add(s["sha256"])
            if s["path"] not in source_current or s["sha256"] != source_current[s["path"]]["sha256"]:
                source_changes.append({"run_id": p.name, "path": s["path"], "frozen_sha256": s["sha256"], "current_sha256": source_current.get(s["path"], {}).get("sha256")})
        assert_(m["physical_updates"] == 576 and m["gradient_group_presentations"] == 864, "manifest_update_counts", p.name)
        assert_(m["status"] == "COMPLETE_576_UPDATES_2_RESTORES_12_DIAGNOSTICS_VALID_BLIND", "manifest_complete", p.name)
        verify_file(p / "run" / m["partition"]["path"], m["partition"])
        partition_hashes.add(m["partition"]["sha256"])
        # Initial Adam identity incorporates trial-specific learning rates.
        # The fixed shared start is encoder/head weights, seed and empty moments.
        initial_by_seed[seed].add(json.dumps({k: v for k, v in m["initials"][arm].items() if k != "adam"}, sort_keys=True))
        assert_(m["starts"][arm] == m["points"]["A_" + arm]["continuity"], "A_to_B_continuity", p.name)
        assert_(m["initials"][arm]["empty_moments"] is True, "empty_initial_adam", p.name)
        run_updates = run_presentations = run_captures = 0
        for stage, endpoint in enumerate(("A_" + arm, "B_" + arm), start=1):
            point = read(p / "run/points" / (endpoint + ".json"))
            assert_(point == m["points"][endpoint], "point_vs_manifest", p.name + "/" + endpoint)
            assert_(point["trial"] == spec and point["adam_step"] == 288 * stage, "point_identity_step", p.name + "/" + endpoint)
            assert_(point["full_restore_verified"] is True and point["inference_restore_verified"] is True, "restore_flags", p.name + "/" + endpoint)
            model_states.add(point["model"]["state_sha256"])
            verify_file(p / "run" / point["mapping"]["path"], point["mapping"])
            assert_(point["memory"]["bytes"] <= 1024 * 1024 and len(point["memory_summary"]["members"]) == 6, "memory_summary_budget", p.name + "/" + endpoint)
            for score in point["scores"].values():
                verify_file(p / "run" / score["path"], score)
            raw = array(p / "run" / point["scores"]["raw"]["path"])
            full = array(p / "run" / point["scores"]["full-raw"]["path"])
            inf = array(p / "run" / point["scores"]["inference-raw"]["path"])
            assert_(np.array_equal(raw, full) and np.array_equal(raw, inf), "restore_scores_equal", p.name + "/" + endpoint)
            assert_(np.isfinite(raw).all(), "finite_raw_scores", p.name + "/" + endpoint)
            verify_file(p / "run" / m["training"][endpoint]["path"], m["training"][endpoint])
            updates = read(p / "run" / m["training"][endpoint]["path"])
            rows = updates["rows"]
            assert_(len(rows) == updates["updates"] == 288, "actual_update_rows", p.name + "/" + endpoint)
            assert_([x["step"] for x in rows] == list(range(1, 289)), "sequential_stage_steps", p.name + "/" + endpoint)
            assert_(updates["adam_step"] == 288 * stage, "actual_adam_step", p.name + "/" + endpoint)
            assert_(len(set(updates["current_ids"])) == 48 and len(updates["current_ids"]) == 288, "six_epochs_48_groups", p.name + "/" + endpoint)
            assert_(set(collections.Counter(updates["current_ids"]).values()) == {6}, "six_presentations_per_current_group", p.name + "/" + endpoint)
            for index, row in enumerate(rows):
                detail = p.name + "/" + endpoint + "/" + str(index + 1)
                assert_(row["trial"] == spec and row["training"]["lambda_rank"] == spec["lambda_rank"], "update_identity_lambda", detail)
                history = "history" in row
                assert_(history == (stage == 2), "history_branch_count", detail)
                assert_(row["history_uid"] is None if stage == 1 else row["history_uid"] in updates["memory_before"]["members"], "history_member", detail)
                assert_(all(math.isfinite(row[k]) for k in ("encoder_lr", "head_lr", "gradient_norm", "total")), "finite_training_summary", detail)
                assert_(row["gradient_norm"] > 0, "positive_total_gradient", detail)
                for branch in ("current", "history") if history else ("current",):
                    g = row[branch]["record_gradient_norm"]
                    assert_(math.isfinite(g) and g > 0, "positive_record_gradient", detail + "/" + branch)
                    positive_gradient_values += 1
                run_presentations += 1 + int(history)
            run_updates += len(rows)
            for epoch, ds in enumerate(m["diagnostics"][endpoint], start=1):
                dp = p / "run/diagnostics" / ds["path"]
                verify_file(dp, ds)
                d = read(dp)
                assert_(d["trial"] == spec and d["epoch"] == epoch and d["stage_step"] == epoch * 48 and d["adam_step"] == (stage - 1) * 288 + epoch * 48, "diagnostic_identity_step", str(dp.relative_to(root)))
                assert_(d["valid_labels_used"] is False and d["mode"] == "A0_eval_no_autograd", "diagnostic_blind", str(dp.relative_to(root)))
                assert_(d["seed"] == f"s{seed}", "diagnostic_effective_seed", str(dp.relative_to(root)))
                for key in ("fit", "cache", "valid_blind"):
                    verify_file(p / "run/diagnostics" / d[key]["path"], d[key])
                assert_(len(d["cache_ids"]) == (0 if stage == 1 else 6), "diagnostic_cache_size", str(dp.relative_to(root)))
                run_captures += 1
        total_updates += run_updates
        total_presentations += run_presentations
        total_endpoints += len(m["points"])
        total_captures += run_captures
        per_run.append({"run_id": p.name, "phase": phase, "arm": arm, "seed": seed, "sources": len(sources), "updates": run_updates, "gradient_presentations": run_presentations, "endpoints": len(m["points"]), "captures": run_captures, "valid": acc["valid"], "reported_elapsed_seconds": c["budget"]["elapsed_seconds"], "peak_cuda_reserved_bytes": c["budget"]["peak_cuda_reserved_bytes"]})

    assert_(len(partition_hashes) == 1, "shared_partition", sorted(partition_hashes))
    for seed, initials in initial_by_seed.items():
        assert_(len(initials) == 1, "shared_initial_per_seed", {"seed": seed, "variants": len(initials)})
    joint_access = read(job / "joint/access.json")
    interruption_access = read(job / "interrupted_attempt/access.json")
    access_total.update(joint_access)
    access_total.update(interruption_access)
    assert_(dict(access_total) == top["access"] == {"train": 46, "valid": 37, "heldout": 0, "owners": 0}, "total_access", dict(access_total))
    reps = read(job / "replications.json")
    for rep in reps:
        spec = rep["trial"]
        name = f"replication_{spec['number']:02}_{spec['arm']}_s{spec['seed']}"
        verify_file(job / name / "completion.json", rep["completion_identity"])
        assert_(rep["child_exit_code"] == 0 and rep["resources_released"] is True and rep["access"]["valid"] == 0, "replication_exit_before_open", name)

    console_path = root / "result/execution/formal.console.txt"
    record(console_path, "MACHINE_JSON_CHECK", "Parse FINAL_ACCOUNTING terminal stdout record and all interval fields")
    finals = [json.loads(line[len("FINAL_ACCOUNTING "):]) for line in console_path.read_text().splitlines() if line.startswith("FINAL_ACCOUNTING ")]
    assert_(len(finals) == 1, "single_final_accounting", len(finals))
    ledger = finals[0]
    snapshot = read(job / "ledger.json")
    wallclock = read(root / "result/analysis_wallclock.json")
    exit_record = read(root / "result/execution/formal_exit.json")
    assert_(exit_record["exit_code"] == 0 and ledger["failed"] is False, "campaign_exit", exit_record)
    kind_sums = {k: math.fsum(x["seconds"] for x in ledger["intervals"] if x["kind"] == k) for k in ledger["used"]}
    for k, s in kind_sums.items():
        assert_(abs(s - ledger["used"][k]) < 1e-8, "interval_sum", {"kind": k, "sum": s, "used": ledger["used"][k]})
    candidate_sums = collections.defaultdict(float)
    for row in ledger["intervals"]:
        if row["run_id"]:
            candidate_sums[row["run_id"]] += row["seconds"]
    assert_(set(candidate_sums) == expected_ids, "all_candidates_charged", sorted(candidate_sums))
    assert_(max(candidate_sums.values()) < 21600, "single_candidate_ceiling", max(candidate_sums.values()))
    rep_time_checks = []
    for rep in reps:
        spec = rep["trial"]
        name = f"replication_{spec['number']:02}_{spec['arm']}_s{spec['seed']}"
        elapsed, child, charged = rep["budget"]["elapsed_seconds"], rep["child_seconds"], candidate_sums[name]
        assert_(elapsed <= child <= charged, "replication_local_elapsed", {"run_id": name, "elapsed": elapsed, "child": child, "charged": charged})
        rep_time_checks.append({"run_id": name, "local_budget_seconds": elapsed, "child_launch_through_exit_seconds": child, "charged_seconds": charged})
    for phase in ("search", "replication"):
        for arm, s in ledger["by_arm"][phase].items():
            re = math.fsum(x["seconds"] for x in ledger["intervals"] if x["kind"] == phase and x["arm"] == arm)
            assert_(abs(re - s) < 1e-8, "arm_time_sum", {"phase": phase, "arm": arm, "sum": re, "saved": s})
    rec = read(job / "recovery.json")
    bound = read(root / "context/reports/documentation/20261021/optuna_reboot/recovery_bound.json")
    inter = bound["interruption"]
    interrupted_search = math.ceil(inter["shutdown_epoch"] - inter["dispatch_epoch"])
    interrupted_control = math.ceil(inter["dispatch_epoch"] - inter["previous_completion_epoch"])
    assert_(interrupted_search == inter["charged_search_seconds"] == 4051 and interrupted_control == inter["charged_control_seconds"] == 4, "reboot_upper_bounds", inter)
    assert_(rec["completed_runs_retrained"] == 0 and rec["interrupted_run_restarts"] == 1, "recovery_counts", rec["status"])
    old_completed = set()
    old_payload_matches = 0
    old_payload_absent = []
    for entry in bound["files"]:
        parts = Path(entry["path"]).parts
        if len(parts) < 2 or parts[0] not in expected_ids or not parts[0].startswith("search_") or parts[0] == "search_08_S_s0":
            continue
        pp = job / entry["path"]
        if pp.exists():
            verify_file(pp, entry)
            old_payload_matches += 1
            old_completed.add(parts[0])
        else:
            old_payload_absent.append(entry["path"])
    assert_(len(old_completed) == 25, "retained_old_completed_runs", sorted(old_completed))

    total = math.fsum(ledger["used"].values()) + ledger["qualification_seconds"] + wallclock["charged_seconds"]
    report_total = 325695.192355
    result = {
        "status": "PASS" if not problems else "ISSUES_REQUIRE_REVIEW",
        "environment": "Isolated review workspace; saved-evidence arithmetic only, not native project qualification",
        "formal_label_reads": 0, "model_loads": 0, "project_module_imports": 0,
        "run_count": len(runs), "successful_run_updates": total_updates,
        "successful_run_gradient_presentations": total_presentations,
        "endpoints": total_endpoints, "diagnostic_captures": total_captures,
        "positive_record_gradient_values": positive_gradient_values,
        "unique_endpoint_model_states": len(model_states),
        "access_total": dict(access_total),
        "source_count_histogram": dict(source_counts),
        "frozen_run_source_count": sum(k * v for k, v in source_counts.items()),
        "current_source_count": len(source_current),
        "source_files_with_multiple_versions": {k: len(v) for k, v in source_versions.items() if len(v) > 1},
        "old_completed_runs_retained": len(old_completed), "old_retained_payload_matches": old_payload_matches,
        "old_bound_payloads_not_in_current_result_scope": old_payload_absent,
        "shared_partition_hash": sorted(partition_hashes),
        "initial_variants_per_seed": {str(k): len(v) for k, v in initial_by_seed.items()},
        "budget": {
            "final_stdout_used": ledger["used"], "interval_sums": kind_sums,
            "qualification_seconds": ledger["qualification_seconds"],
            "additional_analysis_charged_seconds": wallclock["charged_seconds"],
            "total_seconds": total, "total_hours": total / 3600,
            "report_total_seconds": report_total,
            "report_minus_recomputed_seconds": report_total - total,
            "final_stdout_minus_snapshot_control_seconds": ledger["used"]["control"] - snapshot["used"]["control"],
            "longest_candidate": max(candidate_sums, key=candidate_sums.get),
            "longest_candidate_seconds": max(candidate_sums.values()),
            "failed_or_conservative_intervals": [x for x in ledger["intervals"] if x["error"] is not None],
            "by_arm": ledger["by_arm"],
            "interval_kind_counts": dict(collections.Counter(x["kind"] for x in ledger["intervals"])),
            "reboot_search_upper_bound": interrupted_search,
            "reboot_control_upper_bound": interrupted_control,
            "note": "The historical reboot C=4-second charge is a disclosed conservative upper bound and may slightly overlap the prior child tail; do not describe all historical intervals as exact mutually exclusive measurements."
        },
        "checks": dict(checks), "problems": problems,
    }
    (out / "execution_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "machine_read_scope.json").write_text(json.dumps(sorted(scope.values(), key=lambda x: x["path"]), ensure_ascii=False, indent=2) + "\n")
    for filename, rows in (("run_audit.csv", per_run), ("source_changes.csv", source_changes), ("replication_timing.csv", rep_time_checks)):
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "old_bound_payloads_not_in_current_result_scope")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
