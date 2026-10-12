#!/usr/bin/env python3
"""Mechanical byte/manifest/frozen-source audit. Does not import project code."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import difflib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import tarfile
import zipfile


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_closure(z):
    # Independent mechanical reconstruction of run.sources()' declared roots.
    paths = {"scripts/step28_crossfit_retention_run.py", "scripts/step28_crossfit_retention_verify.py",
             "tests/test_step28_crossfit_retention.py", "tests/test_step28_record_replay.py"}
    todo = list(paths)
    while todo:
        text = z.read(todo.pop()).decode()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            for name in names:
                if name and name.startswith("step28_"):
                    path = f"scripts/{name}.py"
                    if path not in paths:
                        paths.add(path)
                        todo.append(path)
    paths.update("schema/" + n for n in ("step28_crossfit_retention_policy.json", "step28_record_replay_policy.json",
                 "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update(("docs/SELLER_ALIAS_CROSSFIT_RETENTION.zh.md", "scripts/run_step28_crossfit_retention_linux_20261012.sh",
                  "scripts/step28_crossfit_retention_wait.py"))
    return sorted(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.input_zip.read_bytes()
    claimed = {"local_original_name": "review_input.zip", "web_display_name": "review_input(20261012-061828).zip",
               "bytes": 570664, "member_count": 64,
               "sha256": "f5e48cf6d4bd06dd2eb546a171b41ebc34fe2d5a40651c86fdbd72466242a477"}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos, names = z.infolist(), z.namelist()
        manifest_raw = z.read("input_manifest.json")
        manifest = json.loads(manifest_raw)
        paths = [row["path"] for row in manifest["payloads"]]
        rows = []
        for row in manifest["payloads"]:
            b = z.read(row["path"])
            rows.append({**row, "actual_bytes": len(b), "actual_sha256": sha(b),
                         "size_matches": len(b) == row["bytes"], "sha256_matches": sha(b) == row["sha256"]})
        closure = source_closure(z)
        attempts, delta_text = [], []
        wall_total = Decimal(0)
        root = "reports/documentation/20261012/crossfit_retention/qualification/cpu/"
        for attempt in ("attempt01", "attempt02"):
            prefix = root + attempt + "/"
            result = json.loads(z.read(prefix + "result/result.json"))
            wrapper = z.read(prefix + "wrapper.txt").decode()
            stamps = [Decimal(t) for t in wrapper.splitlines() if re.fullmatch(r"\d{10}\.\d+", t)]
            assert len(stamps) == 2
            elapsed = stamps[1] - stamps[0]
            wall_total += elapsed
            with tarfile.open(fileobj=io.BytesIO(z.read(prefix + "source.tar.gz")), mode="r:gz") as tar:
                members = tar.getmembers()
                assert all(m.isfile() or m.isdir() for m in members)
                frozen = {m.name.removeprefix("./"): tar.extractfile(m).read() for m in members if m.isfile()}
            receipt = {r["path"]: r for r in result["source_files"]}
            source_rows = []
            for name, b in sorted(frozen.items()):
                current = z.read(name)
                r = receipt.get(name)
                source_rows.append({"path": name, "frozen_bytes": len(b), "frozen_sha256": sha(b),
                                    "current_bytes": len(current), "current_sha256": sha(current),
                                    "matches_current": b == current,
                                    "matches_own_receipt": r is not None and len(b) == r["bytes"] and sha(b) == r["sha256"]})
                if b != current:
                    delta_text.extend(difflib.unified_diff(b.decode().splitlines(True), current.decode().splitlines(True),
                                        fromfile=f"{attempt}/frozen/{name}", tofile=f"current/{name}"))
            unittest = z.read(prefix + "result/unittest.txt").decode()
            attempts.append({"attempt": attempt, "wrapper_start_epoch": str(stamps[0]),
                             "wrapper_end_epoch": str(stamps[1]), "wrapper_elapsed_seconds": str(elapsed),
                             "wrapper_exit_zero": "exit_code=0" in wrapper,
                             "receipt_status": result["status"], "tests_run": result["tests_run"],
                             "failures": result["failures"], "errors": result["errors"], "skipped": result["skipped"],
                             "internal_elapsed_seconds": result["elapsed_seconds"],
                             "internal_cumulative_seconds": result["cumulative_seconds"],
                             "prior_compute_seconds": result["prior_compute_seconds"],
                             "peak_rss_bytes": result["peak_rss_bytes"], "peak_work_bytes": result["peak_work_bytes"],
                             "frozen_regular_file_count": len(frozen),
                             "frozen_directory_members": [m.name for m in members if m.isdir()],
                             "frozen_path_duplicates": [n for n,c in Counter(m.name for m in members).items() if c>1],
                             "paths_equal_current_closure": sorted(frozen) == closure,
                             "all_frozen_files_match_own_receipt": all(r["matches_own_receipt"] for r in source_rows),
                             "current_equal_count": sum(r["matches_current"] for r in source_rows),
                             "different_current_paths": [r["path"] for r in source_rows if not r["matches_current"]],
                             "unittest_reported_seconds": re.search(r"Ran \d+ tests in ([\d.]+)s", unittest).group(1),
                             "test_names": re.findall(r"^(test_\S+)", unittest, flags=re.M),
                             "source_rows": source_rows,
                             "receipt_file_sha256": sha(z.read(prefix + "result/result.json")),
                             "frozen_tar_sha256": sha(z.read(prefix + "source.tar.gz"))})
        identity = {"audit_environment": "ChatGPT webpage workspace; mechanical check only, not project Linux py310",
                    "audited_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
                    "claimed_identity": claimed,
                    "actual_identity": {"received_local_path": str(args.input_zip.resolve()), "bytes": len(raw),
                                        "sha256": sha(raw), "member_count": len(infos),
                                        "sum_uncompressed_member_bytes": sum(i.file_size for i in infos)},
                    "claimed_zip_identity_matches": len(raw)==claimed["bytes"] and sha(raw)==claimed["sha256"] and len(infos)==claimed["member_count"],
                    "crc_test_first_bad_member": z.testzip(),
                    "zip_duplicate_names": [n for n,c in Counter(names).items() if c>1],
                    "manifest_duplicate_payload_paths": [n for n,c in Counter(paths).items() if c>1],
                    "undeclared_members_except_manifest": sorted(set(names) - set(paths) - {"input_manifest.json"}),
                    "missing_declared_members": sorted(set(paths) - set(names)),
                    "manifest_file_bytes": len(manifest_raw), "manifest_file_sha256": sha(manifest_raw),
                    "manifest_payload_count": len(rows),
                    "all_declared_sizes_and_hashes_match": all(r["size_matches"] and r["sha256_matches"] for r in rows),
                    "independent_ast_source_closure_count": len(closure), "source_closure": closure,
                    "manifest_source_count": manifest["source_count"],
                    "manifest_native_gpu_executed": manifest["native_gpu_executed"],
                    "manifest_formal_inputs_accessed": manifest["formal_inputs_accessed"],
                    "declarations_are_not_model_or_native_observation": True,
                    "payloads": rows,
                    "zip_members": [{"path": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size,
                                     "crc32": f"{i.CRC:08x}"} for i in infos]}
        cpu = {"scope": "Mechanical verification of submitted Linux receipts; not independent observation of that host",
               "attempts": attempts, "wrapper_elapsed_total_seconds": str(wall_total),
               "conservative_cpu_seconds_used": int(wall_total.to_integral_value(rounding=ROUND_CEILING)),
               "approved_cpu_seconds": 300,
               "conservative_remaining_seconds": 300-int(wall_total.to_integral_value(rounding=ROUND_CEILING)),
               "same_eleven_test_names": attempts[0]["test_names"] == attempts[1]["test_names"],
               "not_twenty_two_unique_checks": True,
               "current_source_receipt_attempt": "attempt02",
               "native_gpu_or_formal_effects_proved": False}
        assert identity["claimed_zip_identity_matches"] and identity["all_declared_sizes_and_hashes_match"]
        assert not identity["undeclared_members_except_manifest"] and not identity["missing_declared_members"]
        assert len(closure) == manifest["source_count"] == 29
        assert attempts[0]["current_equal_count"] == 25 and attempts[1]["current_equal_count"] == 29
        assert all(a["all_frozen_files_match_own_receipt"] for a in attempts)
    write(args.output / "04_INPUT_IDENTITY.json", identity)
    write(args.output / "CPU_EVIDENCE_AUDIT.json", cpu)
    (args.output / "CPU_ATTEMPT01_TO_CURRENT.diff").write_text("".join(delta_text), encoding="utf-8")
    print("Input ZIP identity, CRC, exact member coverage and all 63 declared payload hashes: MATCH.")
    print("Independent AST source closure: 29 files; attempt01 matches 25/29 current, attempt02 29/29 current.")
    print("Both frozen archives match their own CPU receipts.")
    print(f"Outer cumulative CPU seconds: {wall_total}; conservative used 34/300, remaining 266.")
    print("No project modules, formal inputs, model or qualification were executed by this audit.")


if __name__ == "__main__":
    main()
