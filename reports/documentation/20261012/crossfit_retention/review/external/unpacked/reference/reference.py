#!/usr/bin/env python3
"""Two narrow webpage references. No project module imports or model/data use.

R01 is a Python alias-semantics illustration, NOT a PyTorch reproduction.
R02 executes the exact submitted Budget class with controlled resource probes
and actual small filesystem objects. It is NOT a native resource benchmark.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import threading
import time
import types
import zipfile


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def alias_reference(c: dict) -> dict:
    def simulate(clone_on_restore: bool) -> dict:
        saved = {"adam": {"state": {0: {"step": [c["baseline_step"]]}}}}
        original = copy.deepcopy(saved)
        trials = []
        for i in range(c["attempts"]):
            incoming = copy.deepcopy(saved["adam"]) if clone_on_restore else saved["adam"]
            # This *assumes* the documented tensor-reference behavior; list is
            # only a mutable scalar stand-in. No tensor/autograd/Adam is used.
            live = {"state": {0: {"step": incoming["state"][0]["step"]}}}
            before = live["state"][0]["step"][0]
            row = {"attempt": i + 1, "live_step_before": before,
                   "shares_saved_step": live["state"][0]["step"] is saved["adam"]["state"][0]["step"]}
            if before != c["baseline_step"]:
                row.update(status="REJECTED_EXPECTED_288_BEFORE_UPDATE", actual_update=False)
                trials.append(row)
                break
            live["state"][0]["step"][0] += 1
            row.update(status="SIMULATED_UPDATE", actual_update=False,
                       live_step_after=live["state"][0]["step"][0],
                       saved_step_after=saved["adam"]["state"][0]["step"][0])
            trials.append(row)
        return {"clone_on_restore": clone_on_restore, "trials": trials,
                "saved_state_unchanged": saved == original,
                "saved_step_final": saved["adam"]["state"][0]["step"][0]}

    submitted_semantics, cloned_semantics = simulate(False), simulate(True)
    assert submitted_semantics["trials"][1]["status"].startswith("REJECTED_")
    assert not submitted_semantics["saved_state_unchanged"]
    assert cloned_semantics["saved_state_unchanged"]
    assert all(t["live_step_before"] == 288 for t in cloned_semantics["trials"])
    return {"status": "PASS_SEMANTIC_ILLUSTRATION_ONLY", "assumption": c["assumption"],
            "pytorch_executed": False, "submitted_semantics": submitted_semantics,
            "clone_on_restore_semantics": cloned_semantics}


def budget_reference(source: str, c: dict) -> dict:
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Budget")
    class_text = "\n".join(source.splitlines()[cls.lineno - 1:cls.end_lineno]) + "\n"
    limits = {"maximum_gpu_stage_seconds": 1000, "maximum_output_bytes": c["maximum_output_bytes"],
              "maximum_rss_bytes": 1000000, "maximum_cuda_reserved_bytes": 1000000}
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_initialized=lambda: False,
                                           max_memory_reserved=lambda: 0)
    missing = object()
    original_torch = sys.modules.get("torch", missing)
    original_epoch = os.environ.pop("RECORD_REPLAY_STARTED_EPOCH", None)
    env = {"Path": Path, "time": time, "os": os, "threading": threading, "shutil": shutil,
           "method": types.SimpleNamespace(policy=lambda: {"runtime": limits}),
           "psutil": types.SimpleNamespace(Process=lambda: types.SimpleNamespace(
               memory_info=lambda: types.SimpleNamespace(rss=0))),
           "data": types.SimpleNamespace(write_json=dump)}
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    sys.modules["torch"] = fake_torch
    try:
        exec(compile(module, "submitted/step28_record_replay_run.py", "exec"), env)
        with tempfile.TemporaryDirectory(prefix="crossfit_web_budget_") as tmp:
            base = Path(tmp)
            job = base / "reports" / "job"
            qualification = base / "reports" / "qualification"
            job.mkdir(parents=True)
            qualification.mkdir()
            (job / "payload.bin").write_bytes(b"J" * c["formal_job_bytes"])
            (qualification / "evidence.bin").write_bytes(b"Q" * c["retained_qualification_bytes"])
            budget = env["Budget"](job)
            # First scan executes immediately; write_json is the only output
            # serializer replaced. The scan path and refusal check are exact.
            accepted, error = True, None
            try:
                budget.check()
            except RuntimeError as exc:
                accepted, error = False, str(exc)
            snapshot = budget.snapshot()
            job_after = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
            qualifier_bytes = sum(p.stat().st_size for p in qualification.rglob("*") if p.is_file())
            job_before = c["formal_job_bytes"]
            combined_before = job_before + qualifier_bytes
            # A minimal fixed-base accounting amendment would reject the
            # fixture. This expression is arithmetic, NOT a project patch.
            fixed_would_reject = combined_before > limits["maximum_output_bytes"]
            result = {"status": "PASS_CONTROLLED_OMISSION_WITNESS",
                      "executed_exact_class": True,
                      "source_lines": [cls.lineno, cls.end_lineno],
                      "class_text_sha256": digest(class_text.encode()),
                      "controlled_dependencies": c["controlled_dependencies"],
                      "submitted_check_accepted": accepted, "submitted_error": error,
                      "submitted_peak_output_bytes": snapshot["peak_output_bytes"],
                      "formal_job_bytes_before_check": job_before,
                      "retained_qualification_bytes": qualifier_bytes,
                      "combined_bytes_before_check": combined_before,
                      "formal_job_bytes_after_resource_json": job_after,
                      "combined_bytes_after_resource_json": job_after + qualifier_bytes,
                      "limit_bytes": limits["maximum_output_bytes"],
                      "fixed_base_would_reject": fixed_would_reject,
                      "native_resource_measurement": False,
                      "temporary_fixture_removed_on_exit": True}
            assert accepted and snapshot["peak_output_bytes"] == job_before
            assert fixed_would_reject
            return result
    finally:
        if original_torch is missing:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = original_torch
        if original_epoch is not None:
            os.environ["RECORD_REPLAY_STARTED_EPOCH"] = original_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, wall = time.monotonic(), datetime.now(timezone.utc).isoformat()
    c = json.loads(args.inputs.read_text())
    raw = args.input_zip.read_bytes()
    assert digest(raw) == c["input_zip_sha256"]
    actual_torch_available = importlib.util.find_spec("torch") is not None
    actual_psutil_available = importlib.util.find_spec("psutil") is not None
    with zipfile.ZipFile(args.input_zip) as z:
        name = "scripts/step28_record_replay_run.py"
        source_bytes = z.read(name)
    output = {"scope": c["scope"], "input_zip_sha256": digest(raw),
              "R01": alias_reference(c["alias_semantic_model"]),
              "R02": budget_reference(source_bytes.decode(), c["budget_controlled_fixture"])}
    output["R02"].update(source_path=name, source_sha256=digest(source_bytes))
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "results.json", output)
    execution = {"environment": "ChatGPT webpage workspace; NOT project Linux py310",
                 "started_utc": wall, "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": time.monotonic() - started,
                 "python": sys.version, "platform": platform.platform(),
                 "torch_installed_before_stub": actual_torch_available,
                 "psutil_installed_before_stub": actual_psutil_available,
                 "command": sys.argv, "exit_code": 0,
                 "project_modules_imported": False, "formal_inputs_accessed": False,
                 "pretrained_model_loaded": False, "gpu_used": False,
                 "real_autograd_or_adam_updates": 0,
                 "source_sha256": digest(Path(__file__).read_bytes()),
                 "inputs_sha256": digest(args.inputs.read_bytes()),
                 "results_sha256": digest((args.output / "results.json").read_bytes()),
                 "torch_restore_probe_status": "NOT_EXECUTED_TORCH_UNAVAILABLE"}
    dump(args.output / "execution.json", execution)
    print("WEB REVIEW ONLY; not project Linux py310.")
    print("R01: semantic illustration; no torch/autograd/AdamW execution.")
    print("R01: shared mutable step -> saved 288 becomes 289 -> next expected-288 guard rejects.")
    print("R01: clone-on-restore illustration preserves the saved 288 baseline.")
    print("R02: unchanged submitted Budget class executed on controlled files.")
    print("R02: job 7000 + retained qualification 4000 > limit 10000; submitted check accepted job-only peak 7000.")
    print("No formal data, pretrained model, GPU, or native qualification was used.")
    print("STATUS: PASS_NARROW_WEB_REFERENCES_WITH_EXPLICIT_LIMITS")


if __name__ == "__main__":
    main()
