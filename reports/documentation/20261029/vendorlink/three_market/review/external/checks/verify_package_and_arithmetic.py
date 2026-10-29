"""Review the supplied ZIP only: identities, source bridge and aggregate arithmetic.

Usage: python3 -B verify_package_and_arithmetic.py INPUT_ZIP OUTPUT_JSON
No source modules, models, data archives or original test suite are executed.
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import zipfile

sys.dont_write_bytecode = True
EXPECTED_SHA = "c7f59348077e7163f6ace756eaebfbbe388edf2375a4ec8c5a6c467c02247949"
BASE = "reports/documentation/20261029/vendorlink/"
THREE = BASE + "three_market/"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    source, output = map(Path, sys.argv[1:])
    raw = source.read_bytes()
    assert len(raw) == 106444 and sha(raw) == EXPECTED_SHA
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        assert len(names) == len(set(names)) == 29
        for item in members:
            path = PurePosixPath(item.filename)
            assert not path.is_absolute() and ".." not in path.parts
            assert not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
        contents = {name: archive.read(name) for name in names}
    manifest = json.loads(contents["MANIFEST.json"])
    assert len(manifest) == len({row["path"] for row in manifest}) == 28
    assert set(contents) == {row["path"] for row in manifest} | {"MANIFEST.json"}
    payloads = []
    for row in manifest:
        data = contents[row["path"]]
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
        payloads.append({**row, "size_and_sha_match": True})
    parsed_python = []
    for name, data in contents.items():
        if name.endswith(".py"):
            ast.parse(data)
            parsed_python.append(name)
    get_json = lambda name: json.loads(contents[name])
    cpu05 = get_json(BASE + "implementation/cpu05/receipt.json")
    cpu06 = get_json(THREE + "cpu06/receipt.json")
    cpu05_sources = {row["path"]: row for row in cpu05["sources"]}
    assert cpu06["sources"] == get_json(THREE + "cpu_sources.json")
    for row in cpu06["sources"]:
        assert len(contents[row["path"]]) == row["bytes"]
        assert sha(contents[row["path"]]) == row["sha256"]
    assert len(cpu06["sources"]) == 6
    old_bytes = contents[THREE + "prior_data_sources.zip"]
    with zipfile.ZipFile(io.BytesIO(old_bytes)) as old:
        old_members = [{"path": item.filename, "directory": item.is_dir(),
                        "bytes": item.file_size, "sha256": sha(old.read(item.filename))}
                       for item in old.infolist()]
        assert set(old.namelist()) == {"scripts/", "tests/", "scripts/step29_vendorlink_data.py",
                                      "tests/test_step29_vendorlink.py"}
        for name in ("scripts/step29_vendorlink_data.py", "tests/test_step29_vendorlink.py"):
            assert len(old.read(name)) == cpu05_sources[name]["bytes"]
            assert sha(old.read(name)) == cpu05_sources[name]["sha256"]
        git_comment = old.comment.decode("ascii")
    reused = []
    for name in ("scripts/step29_vendorlink_methods.py", "scripts/step29_vendorlink_run.py",
                 "scripts/step29_vendorlink_cpu.py"):
        assert len(contents[name]) == cpu05_sources[name]["bytes"]
        assert sha(contents[name]) == cpu05_sources[name]["sha256"]
        reused.append({"path": name, "sha256": sha(contents[name])})
    # These hashes identify the already issued reports, not a substitute review verdict.
    prior_reports = {
        BASE + "implementation/review/external/report.zh.txt":
            "83d731f083f42414f2d166605e4e7a2a23b9303805ad7c6ae2722086a7fe0a98",
        BASE + "implementation/review/repair/external/report.zh.txt":
            "5b155e3481fc39c407f3c6dfed7a1a96955e00cc385eee0cced0096a4b14ac91"}
    assert all(sha(contents[name]) == value for name, value in prior_reports.items())
    config = get_json(THREE + "source_config.json")
    inv = get_json(THREE + "inventory.json")
    assert inv["config_sha256"] == sha(contents[THREE + "source_config.json"])
    assert inv["sources"] == config["sources"]
    assert inv["script_sha256"] == sha(contents["scripts/step29_vendorlink_prepare.py"])
    assert inv["data_module_sha256"] == sha(contents["scripts/step29_vendorlink_data.py"])
    source_equations = []
    for market, values in inv["source_report"]["markets"].items():
        rejected = sum(values.get(k, 0) for k in ("malformed_rows", "wrong_market_rows",
                                                 "missing_required_rows", "replacement_character_rows"))
        assert values["parsed_rows"] == values["valid_rows"] + rejected
        source_equations.append({"market": market, "parsed": values["parsed_rows"],
                                 "valid": values["valid_rows"], "rejected_rows": rejected})
        if market != "Agora":
            assert values["valid_rows"] == values["repeated_listing_rows"] + values["distinct_listing_keys"]
            assert values["canonical_rows"] == values["distinct_listing_keys"] - values.get("ambiguous_account_listings_removed", 0)
        else:
            assert values["canonical_rows"] == values["valid_rows"]
    assert sum(v["canonical_rows"] for v in inv["source_report"]["markets"].values()) == inv["source_rows"] == 236457
    eligible = sum(v["eligible_accounts"] for v in inv["markets"].values())
    assert eligible == 3012 and eligible + inv["rejected"]["too_few_ads"] == inv["observed_accounts"] == 8613
    totals = {}
    histograms = {split: Counter() for split in ("fit", "dev", "heldout")}
    for split in histograms:
        values = [v for k, v in inv["partitions"].items() if k.endswith(":" + split)]
        totals[split] = {field: sum(v[field] for v in values)
                         for field in ("accounts", "groups", "views", "selected_ads")}
    for key, part in inv["partitions"].items():
        market, split = key.split(":")
        histogram = {int(k): v for k, v in part["group_account_sizes"].items()}
        assert sum(histogram.values()) == part["groups"]
        assert sum(k * v for k, v in histogram.items()) == part["accounts"] == part["used_accounts"]
        assert part["views"] == 2 * part["accounts"] and part["selected_ads"] == 4 * part["accounts"]
        assert part["h_updates_per_epoch"] == part["groups"]
        assert part["vendor_updates_per_epoch_batch32"] == math.ceil(part["selected_ads"] / 32)
        histograms[split].update(histogram)
    for market, detail in inv["markets"].items():
        assert sum(inv["partitions"][market + ":" + split]["accounts"] for split in histograms) == detail["eligible_accounts"]
    assert totals == {"fit": {"accounts": 2052, "groups": 257, "views": 4104, "selected_ads": 8208},
                      "dev": {"accounts": 504, "groups": 63, "views": 1008, "selected_ads": 2016},
                      "heldout": {"accounts": 456, "groups": 57, "views": 912, "selected_ads": 1824}}
    assert max(histograms["fit"]) == 8
    assert max(max(histograms["dev"]), max(histograms["heldout"])) == 11
    shapes = {}
    for name, accounts in (("training_max", 8), ("evaluation_max", 11)):
        records, views = 4 * accounts, 2 * accounts
        shapes[name] = {"accounts": accounts, "records": records, "h_channel_texts": 2 * records,
                        "record_pairs": math.comb(records, 2), "views": views,
                        "view_pairs": math.comb(views, 2), "candidates_per_query": views - 1}
    heldout_groups = {m: inv["partitions"][m + ":heldout"]["groups"] for m in inv["markets"]}
    order_groups = {}
    orders = (("AlphaBay", "Dream", "Agora"), ("Dream", "Agora", "AlphaBay"), ("Agora", "AlphaBay", "Dream"))
    for order in orders:
        order_groups["/".join(order)] = sum((3 - position) * heldout_groups[m] for position, m in enumerate(order))
    planned = {"trajectories_not_authorized": 36, "physical_stages": 108,
               "updates_per_epoch_factor_all_trajectories": 36 * totals["fit"]["groups"],
               "current_ad_exposures_per_epoch_factor_all_trajectories": 36 * totals["fit"]["selected_ads"],
               "evaluation_cells": 36 * 6,
               "heldout_group_forwards_all_trajectories": 4 * 3 * sum(order_groups.values()),
               "group_forwards_per_single_seed_arm_by_order": order_groups}
    assert planned["updates_per_epoch_factor_all_trajectories"] == 9252
    assert planned["current_ad_exposures_per_epoch_factor_all_trajectories"] == 295488
    assert planned["heldout_group_forwards_all_trajectories"] == 4104
    random_rank_reference = {}
    for accounts in (4, 8, 9, 11):
        candidates = 2 * accounts - 1
        value = sum((Fraction(1, rank) for rank in range(1, candidates + 1)), Fraction()) / candidates
        random_rank_reference[str(accounts)] = {"candidates": candidates, "one_positive_uniform_rank_mean_ap": float(value)}
    assert math.ceil(cpu06["wall_seconds"]) == cpu06["charged_seconds"] == 4
    assert cpu05["cumulative_seconds"] + cpu06["charged_seconds"] == cpu06["cumulative_seconds"] == 30
    assert 900 + 3 * 300 + 300 == 2100 and 4 * 3 + 1 + 2 + 3 * 3 == 24
    result = {
        "scope": "Input ZIP, source bridge, AST syntax and provided aggregate arithmetic only; no project Linux execution",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "zip": {"name": source.name, "bytes": len(raw), "sha256": sha(raw), "members": len(contents), "payloads": len(manifest)},
        "manifest": {"bytes": len(contents["MANIFEST.json"]), "sha256": sha(contents["MANIFEST.json"]), "all_match": True},
        "payload_verification": payloads,
        "nested": {"bytes": len(old_bytes), "sha256": sha(old_bytes), "members": old_members,
                   "declared_git_commit": git_comment, "git_server_contacted": False,
                   "payloads_match_cpu05_receipt": True},
        "byte_identical_reused_sources": reused, "prior_report_identity_matches": prior_reports,
        "six_python_files_parse": parsed_python, "cpu06_all_six_sources_and_manifest_match": True,
        "source_config_inventory_code_identity_consistent": True,
        "source_equations": source_equations, "eligible_accounts": eligible,
        "clean_ads_in_eligible_accounts": sum(v["clean_ads_in_eligible_accounts"] for v in inv["markets"].values()),
        "partition_totals": totals, "group_account_size_histograms": histograms, "maximum_shapes": shapes,
        "planning_arithmetic": planned,
        "illustrative_random_rank_reference_not_results_or_tie_targets": random_rank_reference,
        "cpu06": {"wall_seconds": cpu06["wall_seconds"], "charged_seconds": 4,
                  "prior_charged_seconds": 26, "cumulative_charged_seconds": 30,
                  "peak_rss_bytes": cpu06["peak_rss_bytes"]},
        "gpu_proposal": {"cumulative_task_seconds_ceiling": 2100, "optimizer_steps_ceiling": 24,
                         "implemented": False, "authorized": False, "executed": False},
        "formal_archives_cache_models_or_qrels_read": False,
        "original_counts_independently_recomputed": False,
        "training_tests_or_gpu_executed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "all scoped assertions satisfied", "payloads": len(payloads), "output": str(output)}))


if __name__ == "__main__":
    main()
