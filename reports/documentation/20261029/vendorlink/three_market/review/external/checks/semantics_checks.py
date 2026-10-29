"""Package-only static and handwritten data checks. No models, Torch, or training.

Usage: python3 -B semantics_checks.py --input EXTRACTED_PACKAGE --output result.json
The historical builder is read from the supplied prior_data_sources.zip only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import types
import zipfile

sys.dont_write_bytecode = True


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_data(data, name):
    # The two reviewed modules import only stdlib and perform no I/O on import.
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(data, name, "exec"), module.__dict__)
    return module


def symbol_dumps(data):
    tree = ast.parse(data)
    answer = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            answer[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        answer[node.name + "." + child.name] = ast.dump(child, include_attributes=False)
    return answer


def hand_rows(count, market="A", prefix="person", ads=20):
    return [{"market": market, "vendor": f"{prefix}{i}",
             "title": f"{market} product {prefix} {i} {j}",
             "description": f"{prefix}{i} unique description {market} {i} {j}"}
            for i in range(count) for j in range(ads)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    report_root = root / "reports/documentation/20261029/vendorlink"
    three = report_root / "three_market"
    old_zip = three / "prior_data_sources.zip"
    with zipfile.ZipFile(old_zip) as archive:
        archive_comment = archive.comment.decode("ascii")
        nested = []
        for info in archive.infolist():
            content = archive.read(info.filename)
            nested.append({"path": info.filename, "bytes": len(content), "sha256": digest(content),
                           "directory": info.is_dir()})
        old_data = archive.read("scripts/step29_vendorlink_data.py")
        old_tests = archive.read("tests/test_step29_vendorlink.py")
    cpu05 = json.loads((report_root / "implementation/cpu05/receipt.json").read_text())
    prior = {row["path"]: row for row in cpu05["sources"]}
    for name, content in [("scripts/step29_vendorlink_data.py", old_data),
                          ("tests/test_step29_vendorlink.py", old_tests)]:
        assert len(content) == prior[name]["bytes"]
        assert digest(content) == prior[name]["sha256"]
    reused = []
    for name in ["scripts/step29_vendorlink_methods.py", "scripts/step29_vendorlink_run.py",
                 "scripts/step29_vendorlink_cpu.py"]:
        content = (root / name).read_bytes()
        assert len(content) == prior[name]["bytes"] and digest(content) == prior[name]["sha256"]
        reused.append({"path": name, "bytes": len(content), "sha256": digest(content), "cpu05_match": True})
    cpu06 = json.loads((three / "cpu06/receipt.json").read_text())
    for source in cpu06["sources"]:
        content = (root / source["path"]).read_bytes()
        assert len(content) == source["bytes"] and digest(content) == source["sha256"]
    new_data = (root / "scripts/step29_vendorlink_data.py").read_bytes()
    new_tests = (root / "tests/test_step29_vendorlink.py").read_bytes()
    old_symbols, new_symbols = symbol_dumps(old_data), symbol_dumps(new_data)
    stable_data = ["seed_for", "Ad", "View", "Group", "canonical", "assignment"]
    assert all(old_symbols[key] == new_symbols[key] for key in stable_data)
    old_test_symbols, new_test_symbols = symbol_dumps(old_tests), symbol_dumps(new_tests)
    stable_tests = [key for key in old_test_symbols if key != "Qualification"]
    assert all(old_test_symbols[key] == new_test_symbols[key] for key in stable_tests)
    old = load_data(old_data, "review_previous_data")
    new = load_data(new_data, "review_current_data")
    cases = [
        ("empty", [], 2, 8),
        ("three_market", hand_rows(30, "A") + hand_rows(30, "B") + hand_rows(30, "C"), 2, 8),
        ("19_vs_20", hand_rows(8, ads=19) + hand_rows(8, prefix="kept", ads=20), 2, 8),
        ("larger_view_capacity", hand_rows(16, ads=23) + hand_rows(16, prefix="kept", ads=24), 12, 8),
    ]
    copied = hand_rows(20, "A") + hand_rows(20, "B")
    copied[400] = {**copied[0], "market": "B"}
    copied.append(dict(copied[0]))
    cases.append(("joint_raw_exact_and_within_account_duplicate", copied, 2, 8))
    masked = hand_rows(20)
    masked[0] = {"market": "A", "vendor": "person0", "title": "person0 shared",
                 "description": "person0 shared details"}
    masked[20] = {"market": "A", "vendor": "person1", "title": "person1 shared",
                  "description": "person1 shared details"}
    cases.append(("post_mask_exact_shared", masked, 2, 8))
    case_results = []
    for label, rows, records, group_size in cases:
        options = dict(split_seed=20261029, train_fraction=.7, dev_fraction=.15,
                       records_per_view=records, identities_per_group=group_size)
        old_result = old.build_views(rows, **options)
        new_result = new.build_views(rows, **options)
        assert old_result == new_result, label
        preview = new.preview_counts(rows, **options)
        payload, labels, actual = new_result
        assert preview["rejected"] == actual["rejected"]
        assert {key for key, value in preview["partitions"].items() if value["groups"]} == set(payload)
        for key, part in preview["partitions"].items():
            groups = payload.get(key, [])
            views = [view for group in groups for view in group["views"]]
            assert part["groups"] == len(groups)
            assert part["group_account_sizes"] == dict(Counter(len(group["views"])//2 for group in groups))
            assert part["views"] == len(views)
            assert part["selected_ads"] == sum(len(view["records"]) for view in views)
            assert part["used_accounts"] == len(views)//2
        assert preview["views_or_qrels_created"] is False
        case_results.append({"case": label, "handwritten_rows": len(rows), "old_new_outputs_equal": True,
                             "preview_full_aggregate_agreement": True,
                             "rejected": actual["rejected"], "group_counts": actual["group_counts"]})
    eligible, _, rejected, _ = new.eligible_ads(copied, 2)
    assert ("A", "person0") not in eligible and ("B", "person0") not in eligible
    assert rejected["too_few_ads"] == 2 and rejected["shared_exact_text"] == 2
    eligible, _, rejected_masked, _ = new.eligible_ads(masked, 2)
    assert ("A", "person0") not in eligible and ("A", "person1") not in eligible
    assert rejected_masked["too_few_ads"] == 2 and rejected_masked["shared_exact_text"] == 2
    chunks = {}
    for number in range(25):
        groups = new.group_chunks(list(range(number)), 8)
        # Reference count and tail sizes, computed from Euclidean division.
        q, remainder = divmod(number, 8)
        expected_sizes = ([8]*q + ([remainder] if remainder else []))
        if remainder in (1, 2, 3):
            expected_sizes = [8]*(q-1) + [8+remainder] if q else []
        assert list(map(len, groups)) == expected_sizes
        assert sorted(item for group in groups for item in group) == (list(range(number)) if number >= 4 else [])
        chunks[str(number)] = expected_sizes
    for handle in ["same", "SAME", "00123", "ß", "SS"]:
        value = int.from_bytes(hashlib.sha256(json.dumps((20261029, "identity_split", handle.casefold()),
                            ensure_ascii=False).encode()).digest()[:8], "big") % (2**31) / (2**31)
        expected = "fit" if value < .7 else "dev" if value < .85 else "heldout"
        assert new.split_for(handle, 20261029, .7, .15) == expected
    assert new.split_for("same", 20261029, .7, .15) == new.split_for("SAME", 20261029, .7, .15)
    inventory = json.loads((three / "inventory.json").read_text())
    for key, part in inventory["partitions"].items():
        counts = {int(size): count for size, count in part["group_account_sizes"].items()}
        assert sum(counts.values()) == part["groups"]
        assert sum(size*count for size, count in counts.items()) == part["used_accounts"] == part["accounts"]
        assert part["views"] == 2*part["accounts"] and part["selected_ads"] == 4*part["accounts"]
        assert part["h_updates_per_epoch"] == part["groups"]
        assert part["vendor_updates_per_epoch_batch32"] == (part["selected_ads"]+31)//32
    result = {
        "scope": "web_review_container_stdlib_static_and_handwritten_only_not_project_py310_qualification",
        "python": sys.version, "platform": platform.platform(),
        "formal_text_qrels_or_models_read": False, "torch_or_transformers_imported": False,
        "training_or_native_test_suite_executed": False,
        "nested_archive": {"bytes": old_zip.stat().st_size, "sha256": digest(old_zip.read_bytes()),
                           "comment_claimed_git_commit": archive_comment,
                           "members": nested, "old_source_matches_cpu05_receipt": True},
        "byte_identical_cpu05_sources": reused,
        "cpu06_all_six_source_files_match": True,
        "unchanged_data_ast": stable_data, "unchanged_previous_test_ast": stable_tests,
        "differential_handwritten_cases": case_results,
        "independent_joint_raw_and_masked_duplicate_threshold_oracles": True,
        "group_chunk_size_oracle_0_through_24": chunks,
        "split_formula_and_casefold_oracle": True,
        "all_nine_inventory_partition_arithmetic_checks": True,
        "limitations": ["No original source/cache bytes available; aggregate source counts not independently recomputed.",
                        "Shared helpers alone cannot exclude common-mode errors; old-builder equality and handwritten independent expectations are used here.",
                        "No new mathematical, gradient, optimizer, model, or GPU evidence is produced."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "status": "all scoped assertions satisfied", "cases": len(cases)}))


if __name__ == "__main__":
    main()
