"""Only the changed preparation-bound check; previous five behaviors are reused."""
import importlib.util
import json
from pathlib import Path
import time

r = Path(__file__).parent
spec = importlib.util.spec_from_file_location('retry_budget_check', r/'workspace/scripts/retry_once.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
started = time.monotonic()
caps = dict(training=1360800, confirmation=43200, auxiliary=14400)
budget = dict(caps=caps, used_seconds=dict(training=152265., confirmation=0., auxiliary=12210.))
module.check_allowance(budget, caps, 1800, 2)
rejected = 0
for prep, auxiliary, training in ((1801,12210,152265), (1800,12500,152265), (1,12210,1350000)):
    bad = dict(caps=caps, used_seconds=dict(auxiliary=auxiliary, confirmation=0, training=training))
    try:
        module.check_allowance(bad, caps, prep, 2)
    except TimeoutError:
        rejected += 1
assert rejected == 3
assert budget['used_seconds']['auxiliary'] == 12210.
module.write(r/'cpu03.json', dict(status='PASS', tests=4, errors=0, failures=0,
    seconds=time.monotonic()-started, source=module.record(r/'workspace/scripts/retry_once.py'),
    check=module.record(Path(__file__)), scope='1800s internal preparation bound, unchanged caps, no process access'))
print(json.dumps(module.read(r/'cpu03.json')))
