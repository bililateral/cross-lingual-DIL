"""Bounded Linux CPU-only orchestration increment, charged to the original lines."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

p=argparse.ArgumentParser()
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--job',type=Path,required=True)
p.add_argument('--kind',choices=('forever','vendorlink'),required=True)
p.add_argument('--prior',type=int,required=True)
p.add_argument('--seconds',type=int,required=True)
p.add_argument('--only',action='append')
a=p.parse_args()
assert sys.platform=='linux' and 0<a.seconds<=60 and a.prior+a.seconds<=600
sources=json.loads((a.workspace/'cpu_sources.json').read_text())
for row in sources:
    raw=(a.workspace/row['path']).read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
a.job.mkdir()
tests=(['test_actual_forever_budget_tolerates_peer_retirement_preserves_caps',
    'test_forever_actual_parent_passes_three_slots_and_all_followups',
    'test_failure_retires_slot_and_healthy_slots_take_remaining_queue',
    'test_process_limit_stops_only_offender_then_refills',
    'test_reservations_and_resource_wait_then_three_admissions',
    'test_hs_full_gate_checks_all_endpoints',
    'test_wait_integrated_and_listener_deletes_only_after_update'] if a.kind=='forever' else
    ['test_single_vendor_slot_and_serial_followups_preserved',
     'test_chain_gate_missing_failed_and_valid_negative_science'])
if a.only:
    assert set(a.only).issubset(tests)
    tests=a.only
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(a.job),
    PYTHONPATH=str(a.workspace/'scripts'),FORMAL_TEST_EVIDENCE=str(a.job/'evidence.json'),
    TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1')
command=[sys.executable,'-B',str(a.workspace/'tests/test_step28_formal_wait.py'),
         *['FormalWaitTests.'+t for t in tests]]
begin=time.monotonic()
peak=peak_disk=0
reason=None
with (a.job/'console.txt').open('x') as log:
    child=subprocess.Popen(command,cwd=a.workspace,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        try:
            process=psutil.Process(child.pid)
            rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)])
            size=sum(p.stat().st_size for p in a.job.rglob('*') if p.is_file())
            peak,peak_disk=max(peak,rss),max(peak_disk,size)
            if time.monotonic()-begin>=a.seconds-1 or rss>4*2**30 or size>64*2**20:
                reason='TIME_OR_RESOURCE_LIMIT'
                os.killpg(child.pid,signal.SIGKILL)
                break
        except (psutil.NoSuchProcess,FileNotFoundError):
            pass
        time.sleep(.05)
    code=child.wait()
elapsed=time.monotonic()-begin
result=dict(kind=a.kind,exit_code=code,limit_reason=reason,wall_seconds=elapsed,
    charged_seconds=math.ceil(elapsed),cumulative_seconds=a.prior+math.ceil(elapsed),
    peak_rss_bytes=peak,peak_disk_bytes=peak_disk,sources=sources,tests=tests,
    gpu_training=False,formal_data_reads=0,pretrained_model_loads=0)
(a.job/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
raise SystemExit(code or bool(reason))
