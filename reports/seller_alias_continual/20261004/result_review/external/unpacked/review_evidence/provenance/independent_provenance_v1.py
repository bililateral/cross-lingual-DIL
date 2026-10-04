#!/usr/bin/env python3
"""Independent saved-artifact identity and binding audit; stdlib only.

No training modules are imported and no model, cache payload, or truth is read.
The attachment root is read-only. This program only writes its own output dir.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    failures, checks, hash_checks = [], [], []

    def check(name, passed, detail=None):
        row = {"name": name, "passed": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)
        if not passed:
            failures.append(row)

    def bound(path, expected, label):
        path = Path(path)
        actual = record(path) if path.is_file() else None
        ok = actual is not None and all(actual[k] == expected[k] for k in ("bytes", "sha256"))
        row = {"label": label, "path": str(path.relative_to(root)),
               "expected": {k: expected[k] for k in ("bytes", "sha256")},
               "actual": actual, "passed": ok}
        hash_checks.append(row)
        check(label, ok)
        return ok

    manifest = read(root / "MANIFEST.json")
    listed = [item["path"] for item in manifest["files"]]
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    check("MANIFEST_unique_paths", len(listed) == len(set(listed)))
    check("MANIFEST_file_set", actual_files == set(listed) | {"MANIFEST.json"},
          {"listed": len(listed), "actual": len(actual_files),
           "unlisted": sorted(actual_files - set(listed) - {"MANIFEST.json"}),
           "absent": sorted(set(listed) - actual_files)})
    for item in manifest["files"]:
        bound(root / item["path"], item, "MANIFEST:" + item["path"])

    mapping = read(root / "dependencies/mapping.json")
    mapped = {name: root / item["attachment"] for name, item in mapping.items()}
    old_collection = read(mapped["shared"] / "evaluation/collected.json")
    direct_files = {str(f.relative_to(root)): record(f)
                    for f in (root / "direct_reviews").rglob("*") if f.is_file()}
    summaries, sources_all, source_diff = {}, {}, []
    policy_names = {"logit_low": "step28_logit_low_policy.json", "risk": "step28_risk_policy.json"}

    for study, expected_sources in (("logit_low", 37), ("risk", 32)):
        prefix = f"reports/seller_alias_continual/20261004/{study}_result"
        result_root = root / prefix
        source_root, job = result_root / "source", result_root / "job"
        inventory = read(result_root / "inventory.json")
        for item in inventory["files"]:
            bound(result_root / item["path"], item, study + ":return_inventory:" + item["path"])
        inventory_paths = {item["path"] for item in inventory["files"]}
        check(study + ":returned_inventory_unique", len(inventory_paths) == len(inventory["files"]))
        # Verification was a second transfer and is independently inventoried.
        main_payload = {str(p.relative_to(result_root)) for p in result_root.rglob("*")
                        if p.is_file() and "verification" not in p.relative_to(result_root).parts
                        and str(p.relative_to(result_root)) not in {"inventory.json", "current_status.json"}}
        check(study + ":returned_inventory_complete", main_payload == inventory_paths,
              {"unlisted": sorted(main_payload - inventory_paths), "absent": sorted(inventory_paths - main_payload)})

        workspace = inventory["workspace"]
        project = workspace.split("/reports/", 1)[0]
        project_relative_job = workspace[len(project) + 1:] + "/reports/job"
        for item in read(result_root / "verification/inventory.json")["files"]:
            check(study + ":verification_path_prefix:" + item["path"],
                  item["path"].startswith(project_relative_job + "/"))
            local = job / item["path"][len(project_relative_job) + 1:]
            bound(local, item, study + ":verification_inventory:" + item["path"])
        transfer = read(result_root / "verification/transfer_inventory.json")
        for item in transfer["files"]:
            bound(result_root / "verification" / item["path"], item,
                  study + ":verification_transfer:" + item["path"])

        execution = read(job / "execution.json")
        authorization = read(result_root / "authorization.json")
        run = read(job / "run/manifest.json")
        startup = read(job / "run/startup.json")
        collected = read(job / "evaluation/collected.json")
        evaluation = read(job / "evaluation/evaluation.json")
        completion = read(job / "completion.json")
        policy_path = source_root / "schema" / policy_names[study]
        policy = read(policy_path)
        policy_hash = record(policy_path)["sha256"]
        sources = execution["source_files"]
        sources_all[study] = {item["path"]: item for item in sources}
        check(study + ":source_count", len(sources) == expected_sources == inventory["scientific_sources_verified"])
        for label, obj in (("authorization", authorization), ("manifest", run), ("startup", startup),
                           ("collected", collected), ("evaluation", evaluation)):
            check(study + ":same_source_list:" + label, obj["source_files"] == sources)
            check(study + ":same_policy:" + label, obj["policy_sha256"] == policy_hash)
        for item in sources:
            bound(source_root / item["path"], item, study + ":frozen_source:" + item["path"])
        source_file_paths = {str(f.relative_to(source_root)) for f in source_root.rglob("*") if f.is_file()}
        check(study + ":source_tree_exact", source_file_paths == set(sources_all[study]),
              {"unlisted": sorted(source_file_paths - set(sources_all[study]))})
        bound(result_root / "authorization.json", execution["authorization"], study + ":execution_authorization")
        cpu_name = "logit_cpu" if study == "logit_low" else "risk_cpu"
        cpu_path = root / "direct_reviews" / cpu_name / "audit.json"
        bound(cpu_path, execution["audit"], study + ":execution_prior_cpu")
        check(study + ":prior_cpu_same_sources", read(cpu_path)["source_files"] == sources)
        primary_matches = [name for name, rec in direct_files.items()
                           if rec["sha256"] == authorization["primary_disposition_sha256"]]
        check(study + ":primary_disposition_present", len(primary_matches) == 1, primary_matches)

        closure, stack, import_edges, missing = set(), ["step28_er_weight_run", "step28_er_weight_evaluate"], [], []
        while stack:
            name = stack.pop()
            if name in closure:
                continue
            closure.add(name)
            path = source_root / "scripts" / (name + ".py")
            if not path.is_file():
                missing.append(name)
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                imports = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                           else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
                for dependency in imports:
                    if dependency.startswith("step28_"):
                        import_edges.append([name, dependency, node.lineno])
                        stack.append(dependency)
        check(study + ":static_local_import_closure", not missing and all("scripts/" + name + ".py" in sources_all[study]
                                                                         for name in closure), missing)

        baseline_spec = policy["baseline"]
        check(study + ":baseline_mapping", baseline_spec["linux_job"] == mapping["shared"]["linux_job"])
        for name, item in baseline_spec["records"].items():
            bound(mapped["shared"] / name, item, study + ":baseline_record:" + name)
        check(study + ":baseline_manifest_binding", run["baseline_records"] == baseline_spec["records"])
        references = dict(policy["continuation_references"])
        binding_details = None
        if study == "risk":
            binding = read(job / "evaluation/logit_reference_binding.json")
            spec = policy["pending_logit_reference"]
            check(study + ":preselected_logit_exact", binding["preselected_job"] == spec)
            check(study + ":not_selected_by_result", binding["selection_by_results"] is False)
            check(study + ":preselected_logit_mapping", spec["linux_job"] == mapping["logit_tenth"]["linux_job"])
            bound(mapped["logit_tenth"] / "execution.json", spec["execution"], study + ":preselected_logit_execution")
            references["logit_tenth"] = {**spec, "records": binding["completed_records"]}
            binding_details = binding
        reference_collections = {"shared": old_collection}
        for arm, spec in references.items():
            check(study + ":reference_mapping:" + arm, spec["linux_job"] == mapping[arm]["linux_job"])
            for name, item in spec["records"].items():
                bound(mapped[arm] / name, item, study + ":reference_record:" + arm + ":" + name)
            documents = {name: read(mapped[arm] / name) for name in spec["records"]}
            for name in ("run/manifest.json", "evaluation/collected.json", "evaluation/evaluation.json"):
                check(study + ":reference_policy:" + arm + ":" + name,
                      documents[name]["policy_sha256"] == spec["policy_sha256"])
            ref_sources = documents["execution.json"]["source_files"]
            for name in ("run/manifest.json", "evaluation/collected.json", "evaluation/evaluation.json"):
                check(study + ":reference_source_lists:" + arm + ":" + name,
                      documents[name]["source_files"] == ref_sources)
            ref_collection = documents["evaluation/collected.json"]
            for name in ("domains", "group_ids", "metric_columns"):
                check(study + ":reference_alignment:" + arm + ":" + name, ref_collection[name] == old_collection[name])
            reference_collections[arm] = ref_collection

        reused = read(job / "evaluation/reference/collected.json")
        expected_points = {order + "_shared" for order in ("ABC", "BCA", "CAB")}
        expected_points.update(f"{order}_seq_stage{stage}" for order in ("ABC", "BCA", "CAB") for stage in (2, 3))
        for arm in references:
            expected_points.update(f"{order}_{arm}_stage{stage}" for order in ("ABC", "BCA", "CAB") for stage in (2, 3))
        check(study + ":reference_point_set", set(reused["points"]) == expected_points)
        check(study + ":reference_set_count", len(reused["points"]) * 3 == evaluation["reused_metric_count_sets"] == (63 if study == "logit_low" else 81))
        check(study + ":reference_original_collected", reused["original_collected"] == baseline_spec["records"]["evaluation/collected.json"])
        for arm, spec in references.items():
            check(study + ":reference_continuation_collected:" + arm,
                  reused["continuation_collected"][arm] == spec["records"]["evaluation/collected.json"])
        for name in ("domains", "group_ids", "metric_columns"):
            check(study + ":new_alignment:" + name, collected[name] == old_collection[name])
            check(study + ":reference_alignment:" + name, reused[name] == old_collection[name])
        reference_identity_rows = []
        for name, roles in reused["points"].items():
            candidates = [(arm, c["points"][name]) for arm, c in reference_collections.items() if name in c["points"]]
            check(study + ":reference_unique_origin:" + name, len(candidates) == 1)
            arm, original_roles = candidates[0]
            check(study + ":reference_original_descriptors:" + name, roles == original_roles)
            check(study + ":reference_roles:" + name, set(roles) == {"raw", "stage-cal", "first-cal"})
            for role, info in roles.items():
                for kind in ("matrix", "counts"):
                    item = info[kind]
                    bound(job / "evaluation/reference" / item["path"], item,
                          study + ":reference_artifact:" + name + ":" + role + ":" + kind)
            reference_identity_rows.append({"point": name, "origin": arm, "roles": sorted(roles), "identity_match": roles == original_roles})
        new_expected = {f"{order}_{'logit_tenth' if study == 'logit_low' else 'risk'}_stage{stage}"
                        for order in ("ABC", "BCA", "CAB") for stage in (2, 3)}
        check(study + ":new_point_set", set(collected["points"]) == new_expected)
        check(study + ":new_set_count", len(collected["points"]) * 3 == evaluation["new_metric_count_sets"] == 18)
        for name, roles in collected["points"].items():
            check(study + ":new_roles:" + name, set(roles) == {"raw", "stage-cal", "first-cal"})
            for role, info in roles.items():
                for kind in ("matrix", "counts"):
                    item = info[kind]
                    bound(job / "evaluation" / item["path"], item,
                          study + ":new_artifact:" + name + ":" + role + ":" + kind)
        for key in ("collected", "reused_collection", "draws", "stage_metrics"):
            item = evaluation[key]
            bound(job / "evaluation" / item["path"], item, study + ":evaluation_output:" + key)
        bound(job / completion["evaluation"]["path"], completion["evaluation"], study + ":completion_evaluation")

        custody = read(result_root / "verification/payload_custody.json")
        weights = custody["weights_retained_linux"]
        check(study + ":weight_descriptor_count", len(weights) == 6)
        model_descriptors = []
        for item in weights:
            stem = Path(item["path"]).stem
            point = read(job / "run/points" / (stem + ".json"))
            d = point["model"]
            check(study + ":weight_report_matches_point:" + stem,
                  item["bytes"] == d["bytes"] and item["sha256"] == d["sha256"]
                  and item["path"].endswith("/run/" + d["path"]))
            check(study + ":weight_not_supplied:" + stem, not (job / "run" / d["path"]).exists())
            model_descriptors.append({"point": stem, "bytes": item["bytes"], "sha256": item["sha256"]})
        check(study + ":custody_no_reloading", custody["models_loaded"] is False and custody["formal_labels_parsed"] is False)
        prior_audit = read(result_root / "verification/audit/audit.json")
        for key, filename in (("script", "step28_er_weight_audit.py"), ("helper", "step28_bge_continual_audit.py")):
            bound(root / "scripts" / filename, prior_audit[key], study + ":linux_saved_audit_source:" + key)
        verify_execution = read(result_root / "verification/execution.json")
        argv = verify_execution["argv"]
        def option(flag):
            return argv[argv.index(flag) + 1]
        check(study + ":linux_argv_source_root", option("--source-root") == workspace)
        check(study + ":linux_argv_job", option("--job") == workspace + "/reports/job")
        check(study + ":linux_argv_baseline", option("--baseline") == project + "/" + mapping["shared"]["linux_job"])
        check(study + ":linux_argv_study", option("--study") == study)
        check(study + ":linux_audit_exit", verify_execution["exit_status"] == 0
              and (result_root / "verification/exit_status.txt").read_text().strip() == "0")
        check(study + ":linux_audit_stdout", (result_root / "verification/stdout.log").read_bytes()
              == (result_root / "verification/audit/audit.json").read_bytes())

        summaries[study] = {
            "workspace_recorded_linux": workspace, "mapped_attachment_job": str(job.relative_to(root)),
            "return_inventory_files": len(inventory["files"]),
            "verification_inventory_files": len(read(result_root / "verification/inventory.json")["files"]),
            "source_count": len(sources), "policy_sha256": policy_hash, "primary_disposition": primary_matches,
            "source_import_closure": sorted(closure), "import_edges": sorted(import_edges),
            "new_metric_count_sets": len(collected["points"]) * 3,
            "reused_metric_count_sets": len(reused["points"]) * 3,
            "reference_identities": reference_identity_rows,
            "preselected_logit_binding": binding_details,
            "weight_descriptor_count": len(weights), "weight_recorded_bytes": sum(x["bytes"] for x in weights),
            "weight_records": model_descriptors,
            "weight_verification_scope": "Compared saved Linux fresh-hash records to endpoint descriptors; no native weight bytes present or loaded by this review",
            "verification_actual_argv_record": verify_execution,
        }

    common = set(sources_all["logit_low"]) & set(sources_all["risk"])
    changed = []
    for name in sorted(common):
        a, b = sources_all["logit_low"][name], sources_all["risk"][name]
        if a != b:
            changed.append({"path": name, "logit_low": a, "risk": b})
            a_path = root / "reports/seller_alias_continual/20261004/logit_low_result/source" / name
            b_path = root / "reports/seller_alias_continual/20261004/risk_result/source" / name
            source_diff.extend(difflib.unified_diff(a_path.read_text().splitlines(True), b_path.read_text().splitlines(True),
                                                  fromfile="logit_low/source/" + name, tofile="risk/source/" + name))
    # The supplied top-level scripts are post-run result-audit tools, not a training source tree.
    top_scripts = sorted(str(f.relative_to(root)) for f in (root / "scripts").iterdir() if f.is_file())
    check("training_versions_separate", {x["path"] for x in changed} == {
        "scripts/step28_er_weight.py", "scripts/step28_er_weight_evaluate.py", "scripts/step28_er_weight_run.py"})
    (output / "source_differences.diff").write_text("".join(source_diff), encoding="utf-8")

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "status": "PASS_IDENTITY_AND_BINDING" if not failures else "FAIL_IDENTITY_AND_BINDING",
        "scope": "Saved attachment identity, frozen-source coverage, comparator provenance; scientific statistics and native computation reviewed elsewhere",
        "completed_at_utc": now, "elapsed_seconds": time.monotonic() - start,
        "actual_argv": sys.argv, "python": sys.version, "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "script": record(__file__), "input_manifest": record(root / "MANIFEST.json"),
        "manifest_payloads": len(listed), "actual_files": len(actual_files),
        "hash_check_count": len(hash_checks), "hash_checks_passed": sum(x["passed"] for x in hash_checks),
        "all_checks_count": len(checks), "all_checks_passed": sum(x["passed"] for x in checks),
        "unique_hashed_paths": len({x["path"] for x in hash_checks}),
        "studies": summaries, "common_source_paths": len(common), "changed_common_sources": changed,
        "top_level_scripts_are_audit_only": top_scripts,
        "failures": failures,
        "limits": [
            "SHA equality establishes byte identity, not the correctness of training or scientific conclusions.",
            "No server access, no model loading, no serialized training-cache parsing, and no formal truth parsing were performed.",
            "Native model fresh hashes are Linux-produced records compared against saved descriptors; this review did not hash original native models.",
            "Post-execution command receipts do not independently authenticate backend execution chronology.",
            "AST import closure covers explicit local imports in runner/evaluator source, not arbitrary undisclosed dynamic code injection.",
        ],
    }
    for name, value in (("result.json", report), ("checks.json", checks), ("hash_checks.json", hash_checks)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "manifest_payloads", "actual_files", "hash_check_count",
                                            "hash_checks_passed", "all_checks_count", "all_checks_passed", "failures")},
                     ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
