"""One small metadata-only regression for the observed strict-JSON failure."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time


started = time.monotonic()
recovery, workspace, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(workspace/'scripts'))
spec = importlib.util.spec_from_file_location('fixed_entry', recovery/'workspace/scripts/evaluate.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
from step28_h_ablation_run import Budget, data

# Use the actual inherited snapshot and strict writer; resource inspection itself
# is mocked because this regression concerns serialization, not resource policy.
budget = Budget.__new__(Budget)
budget.started, budget.deadline, budget.tail = time.monotonic(), float('inf'), 60
budget.peak_rss, budget.peak_disk, budget.peak_small = 1, 2, 3
budget.check = lambda reserve=0: None
before = budget.snapshot()
try:
    data.json_bytes(before)
except ValueError as error:
    assert 'JSON compliant' in str(error)
else:
    raise AssertionError('The original strict-JSON failure was not reproduced')
entry.install_snapshot(Budget)
after = json.loads(data.json_bytes(budget.snapshot()))
assert after['deadline_monotonic'] is None and after['auxiliary_limit_disabled'] is True
assert budget.deadline == float('inf')
budget.deadline = 123456.75
finite = json.loads(data.json_bytes(budget.snapshot()))
assert finite['deadline_monotonic'] == 123456.75 and 'auxiliary_limit_disabled' not in finite
for invalid in (float('-inf'), float('nan')):
    try:
        entry.serializable_snapshot(dict(before, deadline_monotonic=invalid))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid deadline silently accepted')
try:
    data.json_bytes(entry.serializable_snapshot(dict(before, elapsed_seconds=float('nan'))))
except ValueError:
    pass
else:
    raise AssertionError('Unrelated NaN silently accepted')
with tempfile.TemporaryDirectory() as tmp:
    request = Path(tmp)/'request.json'
    request.write_text(json.dumps(dict(approved=True, review_closed=True, actions=['confirmation', 'analyze'])))
    for action in ('train', 'generate'):
        sys.argv = ['evaluate.py', '--recovery-request', str(request), action, '--gate', 'unused']
        try:
            entry.main()
        except ValueError as error:
            assert 'cannot train or generate' in str(error)
        else:
            raise AssertionError('Forbidden recovery action accepted')
result = dict(status='PASS', tests=1, errors=0, failures=0,
    evidence=dict(original_failure_reproduced=True, actual_snapshot_and_writer=True,
        runtime_infinity_preserved=True, json_null_for_unbounded_deadline=True,
        finite_deadline_unchanged=True, unrelated_nonfinite_rejected=True,
        training_and_generation_entry_rejected=True, resource_inspection_mocked=True),
    formal_data_read=False, model_loaded=False, gpu_compute=False, seconds=time.monotonic()-started)
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result))
