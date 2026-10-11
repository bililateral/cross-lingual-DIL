"""Linux CPU checks for the narrow 48h-to-72h qualification reuse path."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

if len(sys.argv) == 4:
    root, case, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    sys.path.insert(0, str(root / "scripts"))
    import step28_record_attribution_run as run
    data = run.data
    q = root / "reports/qualification"
    native = data.read_json(q / "native_gate.json")
    gate = dict(status="APPROVED_RECORD_ATTRIBUTION_FORMAL", job="reports/job",
        source_files=run.sources(), review_disposition="NO_OPEN_BLOCKERS",
        verification_source_files=native["source_files"],
        verification_archive=data.record(q / "source_snapshot.zip",root),
        budget_authorization=data.record(q / "authorization_72h.json",root),
        cpu=native["cpu"], gpu=data.record(q / "gpu/attempt01/result/result.json",root),
        runtime=run.policy()["runtime"], supervision=run.policy()["supervision"])
    data.write_json(output, gate)
    rejected = None
    try:
        run.validate_gate(root / "reports/job", output)
    except ValueError as exc:
        rejected = str(exc)
    assert (rejected is None) == (case == "valid"), (case,rejected)
    if case == "valid":
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        budget_root = root / "reports/handwritten_budget"
        budget_root.mkdir()
        budget = run.Budget(budget_root)
        assert budget.limits["maximum_gpu_stage_seconds"] == 259200
        budget.started = time.monotonic()-49*3600
        budget.check()
        budget.started = time.monotonic()-72*3600
        try:
            budget.check()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Actual budget watchdog failed to stop at 72h")
        wrapper = (root / "scripts/run_step28_record_attribution_linux_20261009.sh").read_text()
        assert "--kill-after=5 259195" in wrapper
        assert not (root / "reports/job").exists()
    print(json.dumps(dict(case=case, passed=True, rejection=rejected)))
    sys.exit(0)

started = time.monotonic()
workspace, out = map(Path, sys.argv[1:])
out.mkdir()
native = json.loads((workspace / "reports/qualification/native_gate.json").read_text())
sources = [r["path"] for r in native["source_files"]]
evidence = ["reports/qualification/native_gate.json", "reports/qualification/source_snapshot.zip",
            "reports/qualification/authorization_72h.json", "reports/qualification/gpu/attempt01/result/result.json",
            native["cpu"]["path"]]
checks = []
for case in ("valid", "scientific_code_changed", "unrelated_source_changed", "extra_budget_change", "projection_over_72h", "missing_approval"):
    root = out / "fixtures" / case
    for name in sources + evidence:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / name, target)
    if case in ("scientific_code_changed", "unrelated_source_changed"):
        path, text = (("scripts/step28_record_attribution_run.py", "\nEXTRA_EXECUTED_ASSIGNMENT = True\n")
                      if case == "scientific_code_changed" else ("scripts/step28_record_replay.py", "\n# unrelated change\n"))
        with (root / path).open("a") as stream:
            stream.write(text)
    elif case == "extra_budget_change":
        path = root / "schema/step28_record_attribution_policy.json"
        path.write_text(path.read_text().replace("51539607552", "51539607553"))
    elif case == "projection_over_72h":
        path = root / "reports/qualification/gpu/attempt01/result/result.json"
        report = json.loads(path.read_text())
        report["native"]["projected_total_seconds"] = 259201
        path.write_text(json.dumps(report))
    elif case == "missing_approval":
        path = root / "reports/qualification/authorization_72h.json"
        approval = json.loads(path.read_text())
        approval["status"] = "NOT_APPROVED"
        path.write_text(json.dumps(approval))
    result = subprocess.run([sys.executable,"-B",__file__,str(root),case,str(root / "candidate.json")],
                            text=True,capture_output=True,timeout=20)
    (out / (case+".log")).write_text(result.stdout+result.stderr)
    assert result.returncode == 0, (case,result.stdout,result.stderr)
    checks.append(json.loads(result.stdout.splitlines()[-1]))
report = dict(status="PASS_BUDGET_EXTENSION_ONLY", checks=checks, elapsed_seconds=time.monotonic()-started,
              child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              formal_texts=False, formal_labels=False, gpu_used=False, native_rerun=False,
              coverage="Real gate validates frozen 29-source native evidence, permits exactly budget/gate changes, rejects scientific/source/budget/projection/approval drift; actual Budget.check accepts 49h and rejects 72h.")
assert report["child_peak_rss_kib"] < 2*2**20
(out / "result.json").write_text(json.dumps(report,indent=2)+"\n")
shutil.rmtree(out / "fixtures")
print(json.dumps(report))
