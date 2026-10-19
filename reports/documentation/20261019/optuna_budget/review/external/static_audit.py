"""Attachment-only identity, AST, and saved-field arithmetic audit.

Standard library only. Never imports or executes the submitted project.
No models, formal data, labels, metric recomputation, network, or GPU access.
Usage: python static_audit.py EXTRACTED_INPUT OUTPUT_DIRECTORY
"""
from __future__ import annotations

import ast
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def receipt(root, row):
    path = root / row["path"]
    if not path.is_file():
        return {**row, "present": False, "match": False}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    return {**row, "present": True, "actual_bytes": len(body), "actual_sha256": digest,
            "match": len(body) == row["bytes"] and digest == row["sha256"]}


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def main():
    root, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "input_manifest.json").read_text())
    payloads = [receipt(root, row) for row in manifest]
    listed = {row["path"] for row in manifest}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    integrity = {"payload_count": len(payloads), "all_payloads_match": all(r["match"] for r in payloads),
                 "missing": sorted(listed - actual),
                 "unlisted": sorted(actual - listed - {"input_manifest.json"}), "payloads": payloads}
    dump(out / "input_integrity.json", integrity)

    ast_rows = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=path.relative_to(root).as_posix())
            ast_rows.append({"path": path.relative_to(root).as_posix(), "lines": len(text.splitlines()),
                             "ast_parse": "pass", "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                                                          for n in ast.walk(tree))})
        except (SyntaxError, UnicodeError) as exc:
            ast_rows.append({"path": path.relative_to(root).as_posix(), "ast_parse": "fail", "error": str(exc)})
    dump(out / "ast_static_checks.json", ast_rows)

    prefix = "reports/documentation/20261018/optuna_implementation/"
    native_prefix = prefix + "repair/native_completed/qualification/"
    native_path = root / (native_prefix + "native/result.json")
    native = json.loads(native_path.read_text())
    decimal_native = json.loads(native_path.read_text(), parse_float=Decimal)
    calc = json.loads((root / "reports/documentation/20261019/optuna_budget/calculations.json").read_text())
    bindings = [receipt(root, r) for r in native["source_files"]]
    calc_bindings = [receipt(root, r) for r in calc["sources"]]
    dump(out / "source_binding_checks.json", {
        "native_source_count": len(bindings), "native_all_match_current_attachment": all(r["match"] for r in bindings),
        "native_sources": bindings, "calculation_source_count": len(calc_bindings),
        "calculation_all_match": all(r["match"] for r in calc_bindings), "calculation_sources": calc_bindings,
        "future_amended_sources_available": False,
        "limit": "Current 48-source binding only; no future implementation or evidence-reuse gate was supplied."})

    arms = {}
    for arm in ("S", "S_pos", "H_pos"):
        row = native["arms"][arm]
        m = row["measurements"]
        train = 288 * (m["t_A"] + m["t_B"])
        forward = 1446 * m["v"] + 36 * m["h"]
        io = m["L"] + 2 * (m["s_F"] + m["r_F"] + m["s_I"] + m["r_I"])
        p = 1.25 * (train + forward + io + 1800)
        d = decimal_native["arms"][arm]["measurements"]
        dp = Decimal("1.25") * (288 * (d["t_A"] + d["t_B"]) + 1446*d["v"] + 36*d["h"] + d["L"]
                                   + 2*(d["s_F"] + d["r_F"] + d["s_I"] + d["r_I"]) + 1800)
        arms[arm] = {"measurements": m, "training_seconds": train,
                     "additional_forward_residual_seconds": forward, "load_io_seconds": io,
                     "allowance_seconds": 1800, "factor": 1.25, "projection_seconds": p,
                     "decimal_projection_from_serialized_inputs": str(dp),
                     "difference_from_native_saved_projection": p-row["old_cost_projection"]["projected_total_seconds"],
                     "difference_from_proposal_calculation": p-calc["arms"][arm]["search_projection_seconds"],
                     "cap_seconds": 21600, "margin_seconds": 21600-p,
                     "search_12_seconds": 12*p, "replication_3_seconds": 3*p,
                     "old_9000_gate_pass": p <= 9000, "new_21600_arithmetic_gate_pass": p <= 21600,
                     "actual_updates_saved": row["actual_updates"], "arm_charged_seconds_saved": row["charged_seconds"],
                     "five_saved_check_flags": {k: row[k] for k in ("current_history_lambda_effect", "actual_lr_effect",
                         "next_update_bitwise_equal", "inference_restore_equal", "maximum_shape_verified")}}

    native_wrapper = fields(root / (native_prefix + "native.wrapper.txt"))
    cpu_01 = fields(root / (prefix + "qualification/attempt01.wrapper.txt"))
    cpu_07 = fields(root / (prefix + "repair/qualification/attempt07.wrapper.txt"))
    historical_root = root / "reports/seller_alias_continual/20261018/positive_ranking_result"
    wrapper_lines = (historical_root / "job.wrapper.txt").read_text().splitlines()
    historical_wall = (datetime.fromisoformat(wrapper_lines[2])-datetime.fromisoformat(wrapper_lines[0])).total_seconds()
    console = (historical_root / "job.console.txt").read_text().splitlines()
    events = [(n, json.loads(s)) for n, s in enumerate(console, 1) if s.startswith("{")]
    last_line, last = [(n, r) for n, r in events if r.get("phase") == "B_H_pos" and r.get("completed") == 288][-1]
    tail = historical_wall-last["elapsed_seconds"]
    historical_completion = json.loads((historical_root / "job/completion.json").read_text())
    inner_end = historical_completion["budget"]["elapsed_seconds"]
    rmax = max(row["measurements"]["r_I"] for row in native["arms"].values())
    tc, tf = int(cpu_01["charged_seconds"]), int(cpu_07["charged_seconds"])
    shared_values = {"control": 1.25*(72*rmax+tc+120),
                     "evaluation": 1.25*(9*tail+18*rmax+tf+300),
                     "analysis": 1.25*(tf+120)}
    shared_caps = {"control": 900, "evaluation": 3600, "analysis": 600}
    shared = {key: {"projection_seconds": val, "cap_seconds": shared_caps[key],
                    "margin_seconds": shared_caps[key]-val,
                    "difference_from_proposal": val-calc["shared"][key]["projected_seconds"],
                    "nature": "engineering_proxy_plus_explicit_reserve_not_new_whole_stage_measurement"}
              for key, val in shared_values.items()}
    sum_p = sum(row["projection_seconds"] for row in arms.values())
    counts = {"updates_per_run": 2*6*48, "gradient_groups_per_run": 6*48+2*6*48,
              "A_diagnostic_forwards": 6*(48+40), "B_diagnostic_forwards": 6*(48+6+40+6),
              "endpoint_forwards": 2*3*(12+40), "memory_birth_forwards": 6,
              "extra_forwards": 6*(48+40)+6*(48+6+40+6)+2*3*(12+40)+6,
              "residual_cpu_groups": 6*6, "runs": 36+9, "total_updates": 45*576,
              "total_gradient_groups": 45*864, "total_endpoints": 45*2, "total_captures": 45*12}
    budgets = {"per_run": 21600, "search_per_arm": 12*21600, "replication_per_arm": 3*21600,
               "search_total": 36*21600, "replication_total": 9*21600,
               "shared_control": 900, "shared_evaluation": 3600, "shared_analysis": 600,
               "shared_total": 900+3600+600, "cpu": 300, "native": 1800}
    budgets["hard_total_seconds"] = budgets["search_total"]+budgets["replication_total"]+budgets["shared_total"]+300+1800
    budgets["hard_total_hours"] = budgets["hard_total_seconds"]/3600
    budgets["increase_seconds"] = budgets["hard_total_seconds"]-410400
    budgets["increase_hours"] = budgets["increase_seconds"]/3600
    cpu_used = int(cpu_07["cumulative_charged_seconds"])
    native_used = int(native_wrapper["charged_seconds"])
    qualification = {"cpu_outer_cumulative_seconds": cpu_used, "cpu_remaining_seconds": 300-cpu_used,
                     "cpu_attempt07_outer_seconds": tf, "cpu_prior_to_attempt07_seconds": cpu_used-tf,
                     "native_wall_ns": int(native_wrapper["wall_ns"]),
                     "native_outer_seconds": native_used,
                     "native_wall_ceiling_matches_outer": math.ceil(int(native_wrapper["wall_ns"])/1e9)==native_used,
                     "native_inner_seconds": native["cumulative_charged_seconds"],
                     "native_arm_charged_sum_seconds": sum(r["charged_seconds"] for r in native["arms"].values()),
                     "native_accounting_remaining_seconds_not_retry_permission": 1800-native_used,
                     "qualification_used_total_seconds": cpu_used+native_used,
                     "nominal_total_remaining_not_fungible": budgets["hard_total_seconds"]-cpu_used-native_used,
                     "native_total_updates": sum(r["actual_updates"] for r in native["arms"].values()),
                     "native_top_level_updates_active_arm_only": native["actual_updates"],
                     "formal_started_from_saved_wait_status": json.loads((root/(native_prefix+"native_wait_status.json")).read_text())["formal_started"]}
    projections = {"one_each_sum": sum_p, "search_36": 12*sum_p, "replication_9": 3*sum_p,
                   "all_45": 15*sum_p, "shared_proxies_sum": sum(shared_values.values()),
                   "with_full_shared_and_qualification_reserves": 15*sum_p+5100+2100,
                   "with_shared_proxies_and_full_qualification_reserves": 15*sum_p+sum(shared_values.values())+2100,
                   "full_reserve_hours": (15*sum_p+5100+2100)/3600,
                   "hard_limit_minus_full_reserve_projection": budgets["hard_total_seconds"]-(15*sum_p+5100+2100)}
    arithmetic = {"scope": "Static saved-field arithmetic, not scientific recomputation or native qualification",
                  "counts": counts, "arms": arms, "shared": shared,
                  "shared_basis": {"R_max_r_I": rmax, "T_C_outer_ceiling": tc, "T_F_outer_ceiling": tf,
                      "historical_wrapper_seconds": historical_wall, "last_update_log_line": last_line,
                      "last_update_elapsed": last["elapsed_seconds"], "T_tail_proxy": tail,
                      "historical_completion_elapsed": inner_end,
                      "same_internal_clock_last_update_to_completion": inner_end-last["elapsed_seconds"],
                      "wrapper_minus_internal_completion": historical_wall-inner_end},
                  "budgets": budgets, "qualification_accounting": qualification, "projections": projections}
    dump(out / "independent_arithmetic.json", arithmetic)
    checks = {"payload_identity": integrity["all_payloads_match"] and not integrity["missing"] and not integrity["unlisted"],
              "all_36_python_files_parse": len(ast_rows)==36 and all(r["ast_parse"]=="pass" for r in ast_rows),
              "native_48_binding": len(bindings)==48 and all(r["match"] for r in bindings),
              "calc_source_binding": all(r["match"] for r in calc_bindings),
              "saved_measurements_equal_calculations": all(native["arms"][a]["measurements"]==calc["arms"][a]["measurements"] for a in arms),
              "projections_equal_saved": all(r["difference_from_native_saved_projection"]==0 and r["difference_from_proposal_calculation"]==0 for r in arms.values()),
              "shared_equal_saved": all(r["difference_from_proposal"]==0 for r in shared.values()),
              "old_gate_fails_all_three": all(not r["old_9000_gate_pass"] for r in arms.values()),
              "new_caps_arithmetic_sufficient": all(r["new_21600_arithmetic_gate_pass"] for r in arms.values()) and all(r["margin_seconds"]>=0 for r in shared.values()),
              "total_272h": budgets["hard_total_seconds"]==979200,
              "native_18_calls_outer_876": qualification["native_total_updates"]==18 and native_used==876}
    dump(out / "static_check_summary.json", {"checks": checks, "all_static_checks_pass": all(checks.values()),
         "not_certified": ["amended implementation", "Linux CPU timing", "project execution gate", "whole-stage C/E/A measurement", "scientific performance"]})
    print(json.dumps({"checks": checks, "arms": {a:r["projection_seconds"] for a,r in arms.items()},
                      "shared": shared_values, "total_seconds": budgets["hard_total_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
