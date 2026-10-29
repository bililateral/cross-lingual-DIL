"""Standard-library simulation of extracted control functions; no real child/GPU.

Usage: python3 -B control_budget_microcheck.py EXTRACTED_INPUT OUTPUT_JSON
The fake clock/process demonstrate accounting and admission logic only.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import tempfile
import types

sys.dont_write_bytecode = True


def extract(path, names, env):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), env)


class Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


def simulate_supervise(path, root):
    clock = Clock()
    processes = {}
    class Child:
        def __init__(self):
            self.pid = 12345
            self.returncode = None
            self.finish = clock.now + 2
        def poll(self):
            if clock.now >= self.finish: self.returncode = 0
            return self.returncode
        def wait(self):
            if self.poll() is None: clock.now = self.finish; self.returncode = 0
            return self.returncode
    def popen(*args, **kwargs):
        child = Child(); processes[child.pid] = child; return child
    def killpg(pid, sig): processes[pid].returncode = -sig
    snapshots = 0
    def snapshot():
        nonlocal snapshots
        snapshots += 1
        if snapshots == 1: clock.now += 5.0  # before the first Popen
        return {"at": clock.now, "free": 100 * 2**30, "total": 100 * 2**30, "processes": {}}
    process_stat = types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0), children=lambda **k: [])
    env = {"time": clock, "math": math, "json": json, "GIB": 2**30,
           "subprocess": types.SimpleNamespace(Popen=popen, STDOUT=-2),
           "os": types.SimpleNamespace(killpg=killpg), "signal": signal,
           "psutil": types.SimpleNamespace(Process=lambda pid: process_stat, NoSuchProcess=LookupError),
           "gpu_snapshot": snapshot}
    extract(path, ("write", "fits", "supervise"), env)
    job = root / "supervise"; job.mkdir()
    ledger = {"aux": 0, "g1": 0, "g2": 0, "attempts": []}
    _, elapsed = env["supervise"]([["simulated"]], [job / "g1"], 100, "g1", job, {}, ledger, {}, reserve=1)
    # Exact existing main:220-221 formula. No implementation is repaired here.
    child_union = sum(row["supervision_wall_seconds"] for row in ledger["attempts"])
    ledger["aux"] += math.ceil(max(0, clock.now - child_union))
    assert ledger["aux"] == 0
    assert clock.now >= 7 and ledger["g1"] < clock.now
    return {"pre_popen_resource_seconds": 5, "simulated_elapsed_seconds": clock.now,
            "charged_g1_seconds": ledger["g1"], "charged_aux_seconds": ledger["aux"],
            "whole_supervise_subtracted_seconds": child_union,
            "all_pre_popen_seconds_excluded": True,
            "real_subprocess_or_gpu_started": False}


def simulate_main_aux_admission(path, root):
    clock = Clock(); job = root / "main"; job.mkdir()
    export_manifest = job / "manifest.json"; export_manifest.write_text("{}")
    config_path = root / "config.json"
    config_path.write_text(json.dumps({"export_manifest": str(export_manifest)}))
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    permit_path = root / "permit.json"
    permit_path.write_text(json.dumps({"operation": "vendorlink_native_qualification", "approved": True,
        "implementation_review_closed": True, "config_sha256": digest(config_path), "job": str(job.resolve())}))
    ledger = {"status": "WAIT_RESOURCE", "config_sha256": digest(config_path), "aux": 285, "g1": 0,
        "g2": 0, "attempts": [], "export_manifest_sha256": digest(export_manifest)}
    (job / "ledger.json").write_text(json.dumps(ledger))
    args = types.SimpleNamespace(config=config_path, permit=permit_path, job=job)
    class Parser:
        def add_argument(self, *a, **k): pass
        def parse_args(self): return args
    calls = []
    def source_check(config): clock.now += 16.0
    def supervise(commands, folders, seconds, phase, *a, **k):
        calls.append({"phase": phase, "simulated_parent_seconds": clock.now,
                      "aux_already_used_at_call": 285 + clock.now})
        raise RuntimeError("SIMULATION_STOP_NO_SUBPROCESS")
    env = {"__file__": str(path), "argparse": types.SimpleNamespace(ArgumentParser=Parser), "Path": Path,
           "sys": types.SimpleNamespace(platform="linux", executable="/home/yongpeng/miniconda3/envs/py310/bin/python"),
           "json": json, "math": math, "time": clock, "GIB": 2**30,
           "os": types.SimpleNamespace(environ={}), "sha": digest, "source_check": source_check,
           "supervise": supervise,
           "gpu_snapshot": lambda: {"free": 100 * 2**30, "total": 100 * 2**30, "processes": {}},
           "psutil": types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=100*2**30),
                                            disk_usage=lambda p: types.SimpleNamespace(free=100*2**30)),
           "print": lambda *a, **k: None}
    extract(path, ("write", "main"), env)
    try: env["main"]()
    except RuntimeError as exc: assert str(exc) == "SIMULATION_STOP_NO_SUBPROCESS"
    else: raise AssertionError("Expected simulated halt")
    result = json.loads((job / "ledger.json").read_text())
    assert calls == [{"phase": "g1", "simulated_parent_seconds": 16.0, "aux_already_used_at_call": 301.0}]
    assert result["aux"] == 301 and result["status"] == "BUDGET_EXCEEDED_STOP"
    return {"initial_aux": 285, "this_invocation_source_check_seconds": 16,
            "attempted_phase_calls": calls, "final_status": result["status"],
            "budget_exceeded_detected_only_after_attempted_gpu_admission": True,
            "real_subprocess_or_gpu_started": False}


def simulate_active_publish_window(path, root):
    """A legal write_text('w') intermediate state; no actual concurrent process."""
    clock = Clock()
    child = types.SimpleNamespace(pid=23456, returncode=None)
    child.poll = lambda: child.returncode
    child.wait = lambda: child.returncode
    def killpg(pid, sig):
        assert pid == child.pid
        child.returncode = -sig
    def snapshot():
        clock.now += .05
        return {"at": clock.now, "free": 100 * 2**30, "total": 100 * 2**30, "processes": {}}
    process_stat = types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=0), children=lambda **k: [])
    env = {"time": clock, "math": math, "json": json, "GIB": 2**30,
           "subprocess": types.SimpleNamespace(Popen=lambda *a, **k: child, STDOUT=-2),
           "os": types.SimpleNamespace(killpg=killpg), "signal": signal,
           "psutil": types.SimpleNamespace(Process=lambda pid: process_stat, NoSuchProcess=LookupError),
           "gpu_snapshot": snapshot}
    extract(path, ("write", "fits", "supervise"), env)
    job = root / "active_window"; job.mkdir()
    folder = job / "g1"; folder.mkdir()
    # This is the state after an existing publisher has opened in 'w' mode,
    # before the new JSON bytes become visible; it is not production data.
    (folder / "active.json").write_text("")
    ledger = {"aux": 0, "g1": 0, "g2": 0, "attempts": []}
    try:
        env["supervise"]([["simulated"]], [folder], 100, "g1", job, {}, ledger, {})
    except RuntimeError as exc:
        assert "JSONDecodeError" in str(exc)
    else:
        raise AssertionError("Truncated active publication should expose existing STOP path")
    assert child.returncode == -9 and ledger["status"] == "STOP_NO_AUTOMATIC_RETRY"
    assert ledger["g1"] == 1
    return {"fixture_state": "active.json exists with zero bytes, a non-atomic write_text publication intermediate state",
            "resulting_failure": ledger["attempts"][0]["failure"], "status": ledger["status"],
            "fake_child_returncode": child.returncode, "charged_g1_seconds": ledger["g1"],
            "real_concurrency_or_project_incident_claimed": False,
            "real_subprocess_or_gpu_started": False}


def worker_entry_gate(native_path, control_path):
    """Evaluate the actual permission if-expressions only, never either main()."""
    def gate_test(path):
        tree = ast.parse(path.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        node = next(node for node in fn.body if isinstance(node, ast.If))
        return compile(ast.Expression(node.test), str(path), "eval"), node.lineno, ast.unparse(node.test)
    native_test, native_line, native_source = gate_test(native_path)
    control_test, control_line, _ = gate_test(control_path)
    pinned_hash = hashlib.sha256(b"handwritten fixed config").hexdigest()
    permit = {"operation": "vendorlink_native_qualification", "approved": True,
              "implementation_review_closed": True, "config_sha256": pinned_hash,
              "job": "/tmp/handwritten_permitted_root"}
    base = {"sys": types.SimpleNamespace(platform="linux", executable="/home/yongpeng/miniconda3/envs/py310/bin/python"),
            "permit": permit, "file_sha256": lambda p: pinned_hash, "sha": lambda p: pinned_hash}
    args = types.SimpleNamespace(config=Path("/tmp/handwritten_config.json"),
                                 job=Path("/tmp/handwritten_outside_root"), mode="g1")
    native_rejects = eval(native_test, {**base, "args": args})
    control_rejects = eval(control_test, {**base, "args": args})
    assert native_rejects is False and control_rejects is True
    args2 = types.SimpleNamespace(config=args.config, job=Path(permit["job"]) / "g2_0", mode="g2",
                                  checkpoint=Path("/tmp/foreign_stage3.pt"), gate=Path("/tmp/foreign_open_gate.json"))
    assert eval(native_test, {**base, "args": args2}) is False
    assert all("args." + field not in native_source for field in ("job", "checkpoint", "gate"))
    return {"execution": "Only permission Boolean expressions from current AST; no file, model, subprocess or main() call",
            "native_gate_line": native_line, "control_gate_line": control_line,
            "g1_outside_permit_root_accepted_by_worker_gate": True,
            "same_outside_job_rejected_by_control_gate": True,
            "g2_foreign_checkpoint_and_gate_accepted_by_worker_gate": True,
            "later_main_was_not_executed": True}


def main():
    input_root, output = map(Path, sys.argv[1:])
    path = input_root / "scripts/step29_vendorlink_native_control.py"
    with tempfile.TemporaryDirectory(prefix="vendorlink_control_review_") as tmp:
        root = Path(tmp)
        result = {"scope": "AST extracted source with fake clock/process functions; no real process, GPU, model or training",
                  "environment": {"python": sys.version, "platform": platform.platform()},
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "pre_popen_accounting": simulate_supervise(path, root),
                  "auxiliary_admission": simulate_main_aux_admission(path, root),
                  "active_publish_window": simulate_active_publish_window(path, root),
                  "worker_entry_gate": worker_entry_gate(input_root / "scripts/step29_vendorlink_native.py", path)}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "observed_accounting_defects": 2}))


if __name__ == "__main__": main()
