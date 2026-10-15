"""One bounded CPU-only runtime increment; no model, optimizer or formal input."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

root = Path.cwd()
sys.path[:0] = [str(root/"scripts"), str(root/"tests")]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ["BALANCED_POOLING_TEST_ROOT"] = str(out)
import step28_balanced_pooling_run as run
from test_step28_balanced_pooling import TerminalPoolingTests

expected = json.loads((root/"reports/documentation/20261015/balanced_pooling_repair/runtime/source_manifest.json").read_text())
assert run.sources() == expected
names = ["test_terminal_gate_budget_and_inherited_evidence", "test_runtime_inheritance"]
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(TerminalPoolingTests(n) for n in names))
shell = root/"scripts/run_step28_balanced_pooling_linux_20261015.sh"
subprocess.run(["bash", "-n", str(shell)], check=True)
prefix = shell.read_text().split('date -Is > "$job.wrapper.txt"')[0]
with tempfile.TemporaryDirectory(dir=out) as temporary:
    probe = subprocess.run(["bash", "-c", prefix + '\nprintf "%s %s %s\\n" "$RECORD_REPLAY_STARTED_EPOCH" "$limit" "$remaining"',
        str(shell), "0", "unused_gate", str(Path(temporary)/"not_created_job")], check=True, capture_output=True, text=True)
started, limit, remaining = map(int, probe.stdout.split())
assert limit == 78771 and 0 < remaining <= 78766
assert run.sources() == expected
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert peak_rss <= 2*2**30
summary = dict(selected_tests=names, tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
    skipped=len(result.skipped), checks_passed=result.wasSuccessful(), source_files=run.sources(),
    peak_rss_bytes=peak_rss, formal_inputs=False, formal_labels=False, optimizer_calls=0, model_loads=0,
    shell=dict(source=run.data.record(shell, root), prefix_executed=True, started_epoch=started,
               limit=limit, timeout_seconds=remaining, kill_grace_seconds=5, formal_command_executed=False),
    numerics="No solve, tensor update, inference or metric recomputation; saved cost projection and runtime gates only")
run.data.write_json(out/"checks.json", summary)
raise SystemExit(0 if result.wasSuccessful() else 1)
