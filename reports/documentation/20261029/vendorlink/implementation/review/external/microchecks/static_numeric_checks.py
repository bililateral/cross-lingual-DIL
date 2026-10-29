"""Read-only review checks: archive/source hashes, AST, and handwritten algebra.

No Torch/Transformers import, models, training, test-suite execution, or network.
Usage: python -B static_numeric_checks.py INPUT_ZIP EXTRACTED_INPUT OUTPUT_DIR
"""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import json
import math
import platform
import sys
import zipfile


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    archive, root, destination = map(Path, sys.argv[1:4])
    if destination.resolve() == root.resolve():
        raise ValueError("Use a separate review-output directory; inputs stay read-only")
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"microchecks").mkdir(exist_ok=True)
    content = archive.read_bytes()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = json.loads(z.read("MANIFEST.json"))
        assert len(names) == len(set(names)) == 28
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        assert len(manifest["files"]) == manifest["payload_count"] == 27
        assert set(names) == {x["path"] for x in manifest["files"]} | {"MANIFEST.json"}
        entries = []
        for row in manifest["files"]:
            b = z.read(row["path"])
            item = {"path": row["path"], "expected_bytes": row["bytes"],
                    "actual_bytes": len(b), "expected_sha256": row["sha256"],
                    "actual_sha256": digest(b), "extracted_source_matches_archive": (root/row["path"]).read_bytes() == b}
            item["pass"] = (len(b) == row["bytes"] and digest(b) == row["sha256"]
                            and item["extracted_source_matches_archive"])
            assert item["pass"]
            entries.append(item)
        identity = {"zip_name": archive.name, "zip_bytes": len(content), "zip_sha256": digest(content),
                    "expected_zip_bytes": 48396,
                    "expected_zip_sha256": "5274c7fe508ade3fa0197f3b189ade7d97c0027d897a8e7da35a670b62a7a7e2",
                    "zip_members": len(names), "payload_count": len(entries),
                    "member_set_exact": True, "duplicate_members": False, "unsafe_paths": False,
                    "manifest_sha256": digest(z.read("MANIFEST.json")), "files": entries}
        assert len(content) == identity["expected_zip_bytes"] and digest(content) == identity["expected_zip_sha256"]
        (destination/"input_verification.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2)+"\n")
        lines = ["path\tbytes\tsha256\tlines\treading_scope"]
        for name in names:
            b = z.read(name)
            lines.append("\t".join([name, str(len(b)), digest(b), str(len(b.decode().splitlines())),
                                     "complete_text_read_and_byte_identity_verified"]))
        (destination/"reading_scope.tsv").write_text("\n".join(lines)+"\n")

    source_paths = [x["path"] for x in manifest["files"] if x["path"].endswith(".py")]
    ast_checks = []
    for relative in source_paths:
        tree = ast.parse((root/relative).read_text(encoding="utf-8"), filename=relative)
        ast_checks.append({"path": relative, "syntax_parse": "PASS", "executed": False,
                           "function_count": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))})
    rounds = []
    for attempt in ("cpu01", "cpu02", "cpu03", "cpu04"):
        receipt = json.loads((root/"implementation"/attempt/"receipt.json").read_text())
        sources = []
        for row in receipt["sources"]:
            b = (root/row["path"]).read_bytes()
            sources.append({**row, "current_bytes": len(b), "current_sha256": digest(b),
                            "matches_current": len(b) == row["bytes"] and digest(b) == row["sha256"]})
        rounds.append({"attempt": attempt, **{k: v for k, v in receipt.items() if k != "sources"}, "sources": sources})
    crosswalk = {"rounds": rounds, "raw_wall_seconds_sum": sum(r["wall_seconds"] for r in rounds),
                 "charged_seconds_sum": sum(r["charged_seconds"] for r in rounds),
                 "maximum_rss_bytes": max(r["peak_rss_bytes"] for r in rounds),
                 "cpu03_sources_equal_cpu02": rounds[2]["sources"] == rounds[1]["sources"],
                 "cpu04_all_sources_match_current": all(x["matches_current"] for x in rounds[3]["sources"])}
    assert [r["cumulative_seconds"] for r in rounds] == [5, 10, 15, 20]
    assert crosswalk["cpu03_sources_equal_cpu02"] and crosswalk["cpu04_all_sources_match_current"]
    (destination/"cpu_evidence_crosswalk.json").write_text(json.dumps(crosswalk, ensure_ascii=False, indent=2)+"\n")

    # Independent arithmetic for the exact handwritten vectors in vendor_math.
    z = [(1., 0., 0.), (0., 1., 0.), (1., 1., 0.), (0., 0., 1.), (1., 0., 1.)]
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    def unit(a):
        length = math.sqrt(dot(a, a))
        return tuple(x/length for x in a)
    z = list(map(unit, z))
    a = sum(dot(u, v) for u in z[:2] for v in z[:2])/4
    b = sum(dot(u, v) for u in z[2:] for v in z[2:])/9
    cross = sum(dot(u, v) for u in z[:2] for v in z[2:])/6
    direct = 2*cross/(a+b)
    averages = [tuple(sum(v[j] for v in rows)/len(rows) for j in range(3)) for rows in (z[:2], z[2:])]
    factored = 2*dot(*averages)/(dot(averages[0], averages[0])+dot(averages[1], averages[1]))
    stored = json.loads((root/"implementation/cpu01/vendor_math.json").read_text())["normalized_score"]
    assert abs(direct-factored) < 1e-14 and abs(direct-stored) < 1e-6

    # Tiny finite-difference verification of the H pooling derivative; no model.
    values = [-.7, .1, .5, 1.2]
    def pool(v):
        peak = max(v)
        return .5*sum(v)/len(v)+.5*(peak+math.log(sum(math.exp(x-peak) for x in v)/len(v)))
    denom = sum(math.exp(x) for x in values)
    analytic = [.5/len(values)+.5*math.exp(x)/denom for x in values]
    numerical = []
    for i in range(len(values)):
        plus, minus = values.copy(), values.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numerical.append((pool(plus)-pool(minus))/2e-6)
    error = max(abs(a-b) for a, b in zip(analytic, numerical))
    assert error < 1e-8
    result = {"environment": {"python": platform.python_version(), "platform": platform.platform()},
              "scope": "isolated review workspace; archive and AST checks plus handwritten scalar algebra",
              "ast": ast_checks,
              "vendor_handwritten": {"independent_float64_score": direct, "mean_vector_score": factored,
                    "provided_cpu01_float32_score": stored, "absolute_difference": abs(direct-stored)},
              "h_pool_derivative_handwritten": {"values": values, "analytic": analytic,
                    "finite_difference": numerical, "maximum_error": error,
                    "production_autograd_executed": False},
              "vjp_coverage_counts": {"saved_actual_fixture_records": 20, "saved_actual_record_pairs": 190,
                    "saved_actual_channel_texts": 40, "head_block_size": 512, "test_encoder_microbatch": 64,
                    "proposed_minimal_fixture_records": 35, "proposed_record_pairs": 595,
                    "proposed_channel_texts": 70, "proposed_fixture_executed": False},
              "resume_endpoint_static_proof": {"kind": "control_flow_inference_not_execution",
                    "saved_stage1_cursor_stage": 1, "subsequent_zero_based_stages": [1, 2],
                    "endpoint_files_written_in_fresh_resume_job": ["stage2.pt", "stage3.pt"],
                    "completion_claims": ["stage1.pt", "stage2.pt", "stage3.pt"],
                    "uncreated_relative_endpoint": "stage1.pt",
                    "source": "scripts/step29_vendorlink_run.py:87-99,149-159"},
              "models_loaded": 0, "training_runs": 0, "test_suite_runs": 0,
              "torch_or_transformers_imported": "torch" in sys.modules or "transformers" in sys.modules,
              "project_server_accesses": 0, "formal_data_or_labels_read": 0}
    assert not result["torch_or_transformers_imported"]
    (destination/"microchecks/static_numeric_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"input_verified": True, "ast_files": len(ast_checks),
                      "cpu04_sources_current": True, "charged_seconds": crosswalk["charged_seconds_sum"],
                      "vendor_reference_score": direct, "h_derivative_error": error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
