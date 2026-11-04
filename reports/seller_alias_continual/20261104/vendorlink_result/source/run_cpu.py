"""Bounded incremental handwritten checks; no model/data/network downloads."""
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

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--job', type=Path, required=True)
parser.add_argument('--prior', type=int, required=True)
parser.add_argument('--kind', choices=('vendorlink', 'forever'), required=True)
parser.add_argument('--only', action='append')
args = parser.parse_args()
if sys.platform != 'linux' or args.job.exists() or not 0 <= args.prior < 600:
    raise ValueError('Fresh Linux attempt and actual prior fees required')
sources = json.loads((args.workspace/'cpu_sources.json').read_text())
for row in sources:
    content = (args.workspace/row['path']).read_bytes()
    if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('Actual deployment differs from CPU input')
tests = (['test_compact_vendor_endpoints_preserve_actual_trajectory_and_scores',
          'test_vendor_workload_counts_actual_calls_and_survives_compact_save',
          'test_single_queue_failure_charged_and_next_child_not_launched',
          'test_serial_resource_failure_stops_only_its_child',
          'test_followup_waits_for_training_completion_and_uses_its_own_bucket',
          'test_vendor_query_metrics_and_cross_market_dependencies'] if args.kind == 'vendorlink' else
         ['test_calibration_failure_preserves_actual_fit_before_raising',
          'test_forever_real_formal_three_stage_flow_on_handwritten_tiny_model',
          'test_forever_metric_formulas_and_complete_blind_gate'])
if args.only:
    if not set(args.only).issubset(tests):
        raise ValueError('Requested test is outside this increment')
    tests = args.only
args.job.mkdir(parents=True)
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
    TMPDIR=str(args.job), FOREVER_TEST_TMP=str(args.job), VENDORLINK_TEST_ROOT=str(args.job),
    FORMAL_TEST_EVIDENCE=str(args.job/'evidence.json'), CUBLAS_WORKSPACE_CONFIG=':4096:8',
    PYTHONPATH=str(args.workspace/'scripts') + ':' + str(args.workspace/'tests'),
    TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
command = [sys.executable, '-B', str(args.workspace/'tests/test_step28_formal_training.py'),
           *['FormalTrainingTests.' + test for test in tests]]
started, peak, peak_disk, reason = time.monotonic(), 0, 0, None
with (args.job/'console.txt').open('w') as output:
    child = subprocess.Popen(command, cwd=args.workspace, env=environment, stdout=output,
                             stderr=subprocess.STDOUT, start_new_session=True)
    while child.poll() is None:
        try:
            process = psutil.Process(child.pid)
            rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
            size = sum(p.stat().st_size for p in args.job.rglob('*') if p.is_file())
            peak, peak_disk = max(peak, rss), max(peak_disk, size)
            if time.monotonic() - started >= 600 - args.prior - 2:
                reason = 'CPU_TIME_LIMIT'
            elif rss > 4 * 2**30 or size > 64 * 2**20:
                reason = 'CPU_MEMORY_OR_DISK_LIMIT'
            if reason:
                os.killpg(child.pid, signal.SIGKILL)
                break
        except (psutil.NoSuchProcess, FileNotFoundError):
            pass
        time.sleep(.1)
    code = child.wait()
elapsed = time.monotonic() - started
row = dict(kind=args.kind, exit_code=code, limit_reason=reason, wall_seconds=elapsed,
    charged_seconds=math.ceil(elapsed), cumulative_seconds=args.prior + math.ceil(elapsed),
    peak_rss_bytes=peak, peak_disk_bytes=peak_disk, sources=sources,
    formal_data_reads=0, pretrained_model_loads=0, gpu=False, tests=tests)
(args.job/'receipt.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps({k: v for k, v in row.items() if k != 'sources'}), flush=True)
raise SystemExit(code if code else (1 if reason else 0))
