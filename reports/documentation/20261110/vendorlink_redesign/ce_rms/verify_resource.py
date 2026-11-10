"""One-shot CPU check of the observed foreign-process admission compatibility."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
started = time.monotonic()
import psutil
import step30_evolution_control as control
from step35_vendorlink_account_policy import LIMITS
from step35_vendorlink_account_io import write, record
root = Path(sys.argv[1])
root.mkdir(exist_ok=False)


class Process:
    same_owner = False
    def __init__(self, pid=None): self.pid = pid
    def username(self): return 'us' if self.pid is None or self.same_owner else 'other'
    def create_time(self): return 123.
    def cmdline(self): return ['python', 'foreign_job.py']
    def cwd(self): raise psutil.AccessDenied(self.pid)


try:
    allowed = dict(experiment_root=str(root), other_gpu_commitments=[], allow_observed_foreign_gpu_processes=True)
    nv = SimpleNamespace(snapshot=lambda: dict(free=20*2**30, processes={99:4*2**30}))
    with patch.object(control.psutil, 'Process', Process), \
         patch.object(control.psutil, 'virtual_memory', return_value=SimpleNamespace(available=100*2**30)), \
         patch('shutil.disk_usage', return_value=SimpleNamespace(free=100*2**30)):
        good = control.resource_observation(allowed, root, True, nv=nv, limits=LIMITS)
        assert good['ready'] and good['observed_foreign_gpu_processes'][0]['cwd'] == 'ACCESS_DENIED'
        Process.same_owner = True
        same = control.resource_observation(allowed, root, True, nv=nv, limits=LIMITS)
        assert not same['ready']
        Process.same_owner = False
        old = control.resource_observation({**allowed, 'allow_observed_foreign_gpu_processes':False}, root, True, nv=nv, limits=LIMITS)
        assert not old['ready']
        nv.snapshot=lambda: dict(free=4*2**30, processes={99:4*2**30})
        low = control.resource_observation(allowed, root, True, nv=nv, limits=LIMITS)
        assert not low['ready']
    result = dict(status='PASS', tests=4, errors=0, failures=0,
        foreign_observed_admitted=True, same_owner_unbound_rejected=True,
        legacy_policy_unchanged=True, insufficient_free_rejected=True)
except BaseException as e:
    result = dict(status='FAIL', tests=0, errors=1, failures=0, error=repr(e))
    raise
finally:
    result.update(actual_seconds=time.monotonic()-started,
        source=record(Path(__file__)), runtime_source=record(Path(control.__file__)),
        scientific_compute=False, pretrained_model=False, actual_gpu=False)
    write(root/'receipt.json', result)
    print(json.dumps(result))
