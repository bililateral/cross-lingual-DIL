"""Only hashes, saved JSON fields, source-text locations, and contract arithmetic.

No project module imports, labels, models, Optuna calls, inference or resampling.
"""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUT = ROOT / "mechanical"

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for row in manifest["files"]:
        raw = (INPUT / row["path"]).read_bytes()
        actual = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        rows.append({"path": row["path"], "source_path": row["source_path"], **actual,
                     "bytes_match": actual["bytes"] == row["bytes"],
                     "sha256_match": actual["sha256"] == row["sha256"]})
    expected_paths = {r["path"] for r in rows} | {"manifest.json"}
    actual_paths = {p.relative_to(INPUT).as_posix() for p in INPUT.rglob("*") if p.is_file()}
    identity = {"declared_payloads": manifest["payload_count"], "actual_payloads": len(rows),
                "all_payloads_match": all(r["bytes_match"] and r["sha256_match"] for r in rows),
                "missing": sorted(expected_paths-actual_paths), "extra": sorted(actual_paths-expected_paths), "files": rows}
    write("manifest_check.json", identity)
    if not identity["all_payloads_match"] or identity["missing"] or identity["extra"]:
        raise ValueError("Payload manifest mismatch")

    components = {"cpu": 300, "native": 1800, "search": 36*9000,
                  "replication": 9*9000, "controller_and_independent_analysis": 3300}
    arithmetic = {"component_seconds": components, "development_seconds": sum(components.values()),
                  "development_hours": sum(components.values())/3600,
                  "final_inference_reserve_seconds": 21600,
                  "with_final_inference_hours": (sum(components.values())+21600)/3600,
                  "new_data_generation_budget_included": False,
                  "search_updates": 36*576, "search_gradient_presentations": 36*864,
                  "replication_updates": 9*576, "replication_gradient_presentations": 9*864,
                  "formal_runs": 36+9, "formal_updates": 45*576,
                  "formal_gradient_presentations": 45*864, "endpoints": 45*2,
                  "diagnostic_captures": 45*12, "diagnostic_judgements": 45*9,
                  "max_train_parses": 36+9, "max_valid_label_parses": 36+1,
                  "old_outer_seconds_per_run_mean": 12661/2,
                  "rough_45_run_hours_excluding_other_phases": 45*(12661/2)/3600,
                  "bonferroni_one_sided_alpha": .05/4}
    receipt = json.loads((INPUT / "reports/maintenance/20261018/positive_ranking_weights/receipt.json").read_text(encoding="utf-8"))
    arithmetic["receipt_removed_bytes_sum"] = sum(r["bytes"] for r in receipt["removed"])
    arithmetic["receipt_total_matches"] = arithmetic["receipt_removed_bytes_sum"] == receipt["removed_bytes"]
    size = Fraction(arithmetic["receipt_removed_bytes_sum"], len(receipt["removed"])) * 18
    arithmetic["estimated_18_weights_bytes_using_old_mean"] = float(size)
    arithmetic["estimated_18_weights_GB_decimal"] = float(size/10**9)
    arithmetic["estimated_18_weights_GiB"] = float(size/2**30)
    arithmetic["lr_arithmetic"] = [{"step": t, "factor_fraction": str(Fraction(t,29) if t<=29 else Fraction(288-t,259)),
        "encoder_default": float(Fraction(1,100000)*(Fraction(t,29) if t<=29 else Fraction(288-t,259))),
        "head_default": .001} for t in (1,29,30,288)]
    write("budget_arithmetic.json", arithmetic)

    pattern = re.compile(r"20260909|20260918|20260930|20261008|initialization_seed|schedule_seed|memory_seed|stage_lr|param_groups|initial_identity|qualification_old_rank|best_trial|partition.*seed")
    located = []
    spans = {}
    for path in sorted((INPUT / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.name)  # Syntax parsing, never import/execute.
        name = path.relative_to(INPUT).as_posix()
        spans[name] = [{"name": node.name, "line_start": node.lineno, "line_end": node.end_lineno,
                        "kind": type(node).__name__} for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        located.extend({"path": name, "line": i, "text": line} for i,line in enumerate(source.splitlines(),1) if pattern.search(line))
    write("source_locations.json", {"meaning": "Static locations only; not dynamic coverage", "matches": located, "top_level_spans": spans})

    summary_path = INPUT / "reports/seller_alias_continual/20261018/positive_ranking_result/analysis/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = []
    def visit(value, keys=()):
        if isinstance(value, dict):
            for k,v in value.items():
                if k == "map":
                    selected.append({"json_path": list(keys+(k,)), "saved_value": v})
                visit(v, keys+(k,))
        elif isinstance(value, list):
            for i,v in enumerate(value):
                if isinstance(v,(dict,list)):
                    visit(v,keys+(i,))
    visit(summary)
    baseline = json.loads((INPUT / "reports/documentation/20261016/h_consistency/baseline_sources.json").read_text(encoding="utf-8"))
    fields = {"summary_top_level_keys": list(summary), "saved_map_fields": selected,
              "baseline_top_level_keys": list(baseline), "baseline_metadata": {k:v for k,v in baseline.items() if k != "files"},
              "baseline_file_records": baseline.get("files", []),
              "limits": "Existing saved fields only. Not a recalculation from labels, scores or bootstrap samples. Baseline payloads named by the index are not assumed included."}
    write("saved_field_extract.json", fields)
    print(json.dumps({"manifest": {k:v for k,v in identity.items() if k != "files"}, "arithmetic": arithmetic,
                      "summary_top_level_keys": list(summary), "map_field_count": len(selected),
                      "first_saved_map_fields": selected[:8], "baseline_top_level_keys": list(baseline)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
