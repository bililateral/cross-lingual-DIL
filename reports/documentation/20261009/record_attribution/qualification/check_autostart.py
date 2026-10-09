"""One-off Linux orchestration checks; all GPU reports and input files are fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

started = time.monotonic()
workspace, out = map(Path, sys.argv[1:])
out.mkdir()
script_name = "run_step28_record_attribution_auto_linux_20261009.sh"
script = (workspace / "scripts" / script_name).read_text()
subprocess.run(["bash", "-n"], input=script, text=True, check=True)
gate = json.loads((workspace / "reports/qualification/native_gate.json").read_text())
data_root = "reports/seller_alias_continual/20260910/expression_generation/20260910_150500/data"
input_names = ("manifest.json", "validation.json", "groups.csv", "train/items.jsonl",
               "development/items.jsonl", "train/supervision/pairs.csv", "development/supervision/pairs.csv")
gpu = dict(mode="gpu", status="PASS_HANDWRITTEN_ONLY", source_files=gate["source_files"],
           cumulative_seconds=200, peak_rss_bytes=2**30, max_reserved_bytes=2**30,
           formal_inputs=False, formal_labels=False,
           native=dict(projected_total_seconds=160000, projected_peak_output_bytes=45000000000,
                       neutrality_all_arms=True, architectures={a:dict(native_state_unchanged=True,
                       next_model_adam_grad_bitwise_equal=True, maximum_records=224, channel_tokens=256)
                       for a in ("LOGIT0.1", "R0", "S")}))
checks = []
for case in ("valid", "native_failed", "missing_finished", "projection_over_budget", "update_mismatch", "source_drift"):
    root = out / "fixtures" / case
    fake_project, fake_workspace = root / "project", root / "workspace"
    fake_workspace.mkdir(parents=True)
    for rec in [*gate["source_files"], gate["cpu"], gate["primary_review"]]:
        target = fake_workspace / rec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / rec["path"], target)
    (fake_workspace / "scripts" / script_name).write_text(script)
    q = fake_workspace / "reports/qualification"
    (q / "native_gate.json").write_text(json.dumps(gate))
    nq, attempt = q / "native_listener", q / "gpu/attempt01"
    nq.mkdir()
    (attempt / "result").mkdir(parents=True)
    (nq / "pid.txt").write_text(str(os.getpid()))
    (nq / "finished.txt").write_text("2026-10-09T12:03:20+08:00")
    (nq / "exit_status.txt").write_text("0")
    (attempt / "started.txt").write_text("2026-10-09T12:00:00+08:00")
    (attempt / "finished.txt").write_text("2026-10-09T12:03:20+08:00")
    (attempt / "exit_status.txt").write_text("0")
    fixture = copy.deepcopy(gpu)
    if case == "native_failed":
        (nq / "exit_status.txt").write_text("2")
    elif case == "missing_finished":
        (attempt / "finished.txt").unlink()
    elif case == "projection_over_budget":
        fixture["native"]["projected_total_seconds"] = 172801
    elif case == "update_mismatch":
        fixture["native"]["architectures"]["S"]["next_model_adam_grad_bitwise_equal"] = False
    elif case == "source_drift":
        with (fake_workspace / "scripts/step28_record_attribution_run.py").open("a") as stream:
            stream.write("\n# fixture source drift\n")
    (attempt / "result/result.json").write_text(json.dumps(fixture))
    for name in input_names:
        target = fake_project / data_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("HANDWRITTEN PLACEHOLDER: no formal text or label\n")
    # Execute the real shell prerequisite/gate section in an isolated fixture;
    # stop before resource probing or a training command can be reached.
    prefix = script.split("# GATE_END")[0] + "\nexit 0\n"
    prefix = prefix.replace("project=/home/yongpeng/cross-lingual", f"project='{fake_project}'", 1)
    prefix = prefix.replace('workspace="$project/reports/seller_alias_continual/20261009/record_attribution_execution/20261009_182606/workspace"',
                            f"workspace='{fake_workspace}'", 1)
    prefix = prefix.replace("export CUDA_VISIBLE_DEVICES=0 ", "export CUDA_VISIBLE_DEVICES='' ", 1)
    result = subprocess.run(["bash"], input=prefix, text=True, capture_output=True, timeout=15)
    gate_exists = (q / "formal_gate.json").exists()
    links = sorted(p.relative_to(fake_workspace / data_root).as_posix()
                   for p in (fake_workspace / data_root).rglob("*") if p.is_symlink())
    assert (result.returncode == 0) == (case == "valid"), (case, result.returncode, result.stderr)
    assert gate_exists == (case == "valid")
    assert links == (sorted(input_names) if case == "valid" else [])
    assert not (fake_workspace / "reports/job").exists()
    shutil.copyfile(q / "formal_listener/events.log", out / (case + ".log"))
    checks.append(dict(case=case, returncode=result.returncode, formal_gate_created=gate_exists,
                       placeholder_links=len(links), formal_training_started=False))

# Exercise the actual first-update parser and deletion/wait tail with a tiny
# fixture child. A mere PID or completed=0 must not delete the active copy.
root = out / "fixtures/self_delete"
root.mkdir()
queue, job, active = root / "queue", root / "job", root / "detector.sh"
queue.mkdir()
active.write_text(script)
tail = script[script.index("actual_update_recorded() {"):]
setup = f'''set -euo pipefail
self='{active}'
queue='{queue}'
job='{job}'
'''
function = tail[:tail.index('while kill -0')]
negative = setup + function + '''
printf '%s\n' '{"completed":0,"elapsed_seconds":1,"event":"updates"}' > "$job.console.txt"
if actual_update_recorded; then exit 11; fi
test -f "$self"
'''
subprocess.run(["bash"], input=negative, text=True, check=True)
positive = setup + '''
(sleep 0.2; printf '%s\n' '{"completed":1,"elapsed_seconds":2.3,"event":"updates","point":"ABC_LOGIT0.1_stage1"}' > "$job.console.txt"; sleep 3) &
child=$!
''' + tail
subprocess.run(["bash"], input=positive, text=True, check=True, timeout=8)
assert not active.exists()
assert (queue / "script_deleted_after_update.txt").is_file()
assert json.loads((queue / "first_update.json").read_text())["completed"] == 1
checks.append(dict(case="self_delete_after_update_only", passed=True, child_waited=True))
assert resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss < 2 * 2**20
report = dict(status="PASS_ORCHESTRATION_ONLY", at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              script_sha256=hashlib.sha256(script.encode()).hexdigest(), checks=checks,
              elapsed_seconds=time.monotonic()-started, child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              formal_texts=False, formal_labels=False, gpu_used=False,
              scope="Synthetic receipt fixtures test shell gating and self-deletion; not native GPU qualification or real training")
(out / "result.json").write_text(json.dumps(report, indent=2)+"\n")
shutil.rmtree(out / "fixtures")
print(json.dumps(report))
