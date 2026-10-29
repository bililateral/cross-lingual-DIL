"""Targeted review-only source and file-copy checks. No model/training imports.

Usage: python -B source_and_copy_checks.py BASE_DIR REPAIR_DIR OLD_DIR INPUT_ZIP OUTPUT_DIR
"""
from pathlib import Path, PurePosixPath
import ast
import copy
import difflib
import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
import zipfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def find_node(tree, name):
    node = tree
    for part in name.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return node


def dump(node):
    return ast.dump(node, include_attributes=False)


base, current, old, archive, out = map(Path, sys.argv[1:])
assert out.resolve() not in (base.resolve(), current.resolve(), old.resolve())
out.mkdir(parents=True, exist_ok=True)
(out/"checks").mkdir(exist_ok=True)
raw = archive.read_bytes()
assert len(raw) == 51586 and sha(raw) == "e45f69e07f956d881baf1f4b8ac528e279ca853214135257b51e143186f82bde"
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    manifest = json.loads(z.read("MANIFEST.json"))
    assert len(names) == len(set(names)) == 17 and manifest["payload_count"] == 16
    assert set(names) == {r["path"] for r in manifest["files"]} | {"MANIFEST.json"}
    assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
    payloads = []
    for r in manifest["files"]:
        b = z.read(r["path"])
        item = {**r, "actual_bytes": len(b), "actual_sha256": sha(b),
                "matches_manifest": len(b) == r["bytes"] and sha(b) == r["sha256"],
                "extracted_copy_equal": b == (current/r["path"]).read_bytes()}
        assert item["matches_manifest"] and item["extracted_copy_equal"]
        payloads.append(item)
    inventory = [{"path": n, "bytes": len(z.read(n)), "sha256": sha(z.read(n))} for n in names]

nested = current/"implementation/cpu_transfer.zip"
nested_raw = nested.read_bytes()
assert len(nested_raw) == 19270 and sha(nested_raw) == "ef4b86268496f603ec690c8385db6fe907786854a0b022deb1e377f79f513e6b"
receipt01 = json.loads((base/"implementation/cpu01/receipt.json").read_text())
receipt05 = json.loads((current/"implementation/cpu05/receipt.json").read_text())
with zipfile.ZipFile(nested) as z:
    assert len(z.namelist()) == len(set(z.namelist())) == 6
    assert set(z.namelist()) == {r["path"] for r in receipt01["sources"]} | {"cpu_sources.json"}
    inner_manifest = json.loads(z.read("cpu_sources.json"))
    assert inner_manifest == receipt01["sources"]
    nested_sources = []
    for r in receipt01["sources"]:
        b = z.read(r["path"])
        ok = len(b) == r["bytes"] and sha(b) == r["sha256"] and b == (old/r["path"]).read_bytes()
        assert ok
        nested_sources.append({**r, "actual_bytes": len(b), "actual_sha256": sha(b), "cpu01_match": ok})
    nested_inventory = [{"path": n, "bytes": len(z.read(n)), "sha256": sha(z.read(n))} for n in z.namelist()]
identity = {"input_name": archive.name, "bytes": len(raw), "sha256": sha(raw), "members": 17,
            "payload_count": 16, "member_set_exact": True, "duplicate_or_unsafe_names": False,
            "base_input_sha256": manifest["base_input_sha256"], "payloads": payloads,
            "all_members": inventory,
            "nested_archive": {"path": "implementation/cpu_transfer.zip", "bytes": len(nested_raw),
                "sha256": sha(nested_raw), "members": nested_inventory,
                "cpu_sources_json_equals_cpu01_receipt": True, "source_comparisons": nested_sources}}
(out/"input_and_nested_verification.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2)+"\n")

current_sources = []
trees = {}
diffs = []
for r in receipt05["sources"]:
    p = r["path"]
    texts = [(d/p).read_text(encoding="utf-8") for d in (old, base, current)]
    trees[p] = [ast.parse(s, filename=p) for s in texts]
    b = (current/p).read_bytes()
    ok = len(b) == r["bytes"] and sha(b) == r["sha256"]
    assert ok
    current_sources.append({**r, "cpu05_match": ok, "byte_equal_to_first_review": b == (base/p).read_bytes()})
    for i, a, c in ((0, "cpu01", "first_review"), (1, "first_review", "repair")):
        diff = "".join(difflib.unified_diff(texts[i].splitlines(True), texts[i+1].splitlines(True),
                                         fromfile=a+"/"+p, tofile=c+"/"+p))
        if diff:
            diffs.append(diff)
(out/"source_diffs.txt").write_text("\n".join(diffs))

selected = {
 "scripts/step29_vendorlink_methods.py": ["vendor_embeddings", "vendor_scores", "h_pool", "h_loss", "h_objective",
    "Model.__init__", "Model.encode", "Model.records", "Model.table", "Model.backward_h", "Model.teacher", "Model.score",
    "update_h", "update_vendor", "Memory"],
 "scripts/step29_vendorlink_run.py": ["rng_state", "restore_rng", "save_state", "load_state", "set_rates"],
 "scripts/step29_vendorlink_data.py": ["seed_for", "Ad", "View", "Group.validate", "Group.labels", "Group.from_dict", "assignment"],
 "tests/test_step29_vendorlink.py": ["fixture", "model_for", "independent_pool", "independent_loss",
    "Qualification.test_vendor_representation_and_diagonal_reference", "Qualification.test_h_math_and_missing_operations"]
}
functions = []
for p, members in selected.items():
    for name in members:
        nodes = [find_node(t, name) for t in trees[p]]
        codes = [dump(n) for n in nodes]
        assert codes[0] == codes[1] == codes[2], (p, name)
        functions.append({"path": p, "symbol": name, "ast_equal_across_cpu01_base_repair": True,
                          "lines": {k: [n.lineno, n.end_lineno] for k, n in zip(("cpu01", "base", "repair"), nodes)}})
method_trees = trees["scripts/step29_vendorlink_methods.py"]
h_returns = [dump(find_node(t, "make_optimizer").body[-1]) for t in method_trees]
assert h_returns[0] == h_returns[1] == h_returns[2]
old_settings = find_node(method_trees[0], "Settings")
base_settings = copy.deepcopy(find_node(method_trees[1], "Settings"))
base_settings.body = [n for n in base_settings.body if not (isinstance(n, ast.AnnAssign)
    and isinstance(n.target, ast.Name) and n.target.id == "vendor_weight_decay")]
assert dump(old_settings) == dump(base_settings)

run_tree = trees["scripts/step29_vendorlink_run.py"][2]
file_sha = find_node(run_tree, "file_sha256")
execute = find_node(run_tree, "execute")
try_body = next(n for n in execute.body if isinstance(n, ast.Try)).body
copy_block = try_body[0]
assert isinstance(copy_block, ast.If) and isinstance(try_body[1], ast.For)
namespace = {"Path": Path, "hashlib": hashlib, "shutil": shutil}
exec(compile(ast.Module(body=[file_sha], type_ignores=[]), "extracted_file_sha", "exec"), namespace)
copy_code = compile(ast.Module(body=[copy_block], type_ignores=[]), "extracted_endpoint_copy", "exec")
copy_results = []
with tempfile.TemporaryDirectory(prefix="vendorlink_review_bytes_") as temp:
    temp = Path(temp)
    for stage in (1, 2):
        source, target = temp/f"source_{stage}", temp/f"target_{stage}"
        source.mkdir(); target.mkdir()
        for i in range(1, stage+1):
            (source/f"stage{i}.pt").write_bytes((b"handwritten-byte-fixture" if i == 1 else b"x"*(2**20+17)))
        env = dict(namespace, resume=source/f"stage{stage}.pt", cursor={"stage": stage},
                   job=target, inherited=[], check=lambda: None)
        exec(copy_code, env)
        assert len(env["inherited"]) == stage
        for item in env["inherited"]:
            original = source/item["path"]
            copied = target/item["path"]
            assert original.read_bytes() == copied.read_bytes()
            assert item["source"] == str(original.resolve()) and item["sha256"] == sha(original.read_bytes())
        copy_results.append({"completed_stages": stage, "copied_endpoint_count": stage,
                             "bytes_hash_and_source_paths_match": True})
    source, target = temp/"missing_source", temp/"missing_target"
    source.mkdir(); target.mkdir()
    env = dict(namespace, resume=source/"stage1.pt", cursor={"stage": 1}, job=target, inherited=[], check=lambda: None)
    try:
        exec(copy_code, env)
    except FileNotFoundError:
        missing_failed = True
    else:
        raise AssertionError("Missing endpoint did not fail")
    assert list(target.iterdir()) == []
assert "torch" not in sys.modules and "transformers" not in sys.modules
result = {"environment": {"python": platform.python_version(), "platform": platform.platform()},
          "scope": "file/AST/diff checks and extracted plain-byte copying only; not project native qualification",
          "cpu05_sources": current_sources, "unchanged_function_checks": functions,
          "H_optimizer_return_ast_unchanged": True, "Settings_only_added_VendorLink_weight_decay": True,
          "cpu05_accounting": {"wall_seconds": receipt05["wall_seconds"],
               "charged_seconds": math.ceil(receipt05["wall_seconds"]), "prior_seconds": 20,
               "cumulative_seconds": 20+math.ceil(receipt05["wall_seconds"]),
               "peak_rss_bytes": receipt05["peak_rss_bytes"]},
          "endpoint_copy_microchecks": {"normal_cases": copy_results, "maximum_handwritten_bytes": 2**20+17,
               "missing_endpoint_raises": missing_failed, "missing_case_files_copied": 0,
               "block_precedes_load_fit_and_training": True,
               "whole_execute_or_load_state_executed": False},
          "models_loaded": 0, "training_runs": 0, "test_suite_runs": 0,
          "formal_data_reads": 0, "project_server_accesses": 0, "torch_or_transformers_imported": False}
assert receipt05["charged_seconds"] == result["cpu05_accounting"]["charged_seconds"] == 6
assert receipt05["cumulative_seconds"] == result["cpu05_accounting"]["cumulative_seconds"] == 26
(out/"source_and_cpu05_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"payloads_verified": 16, "old_sources_matching_cpu01": 5,
                  "current_sources_matching_cpu05": 5, "unchanged_symbols": len(functions),
                  "copy_cases_pass": True, "models_or_training": 0}, ensure_ascii=False))
