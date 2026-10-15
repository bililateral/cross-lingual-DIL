#!/usr/bin/env bash
set -euo pipefail
cd '/home/yongpeng/cross-lingual/reports/documentation/20261015/g_half_implementation/20261015_234000/workspace'
attempt=reports/qualification/gpu/attempt01
test ! -e "$attempt"
mkdir -p "$attempt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import json,hashlib,pathlib
w=pathlib.Path('.')
for r in json.load(open('deployment.json'))['files']:
 b=(w/r['path']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
PY
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
date -Is > "$attempt/wrapper.txt"
remaining=$(/home/yongpeng/miniconda3/envs/py310/bin/python -c 'import os,time; print(max(0.,598-(time.time()-float(os.environ["QUALIFICATION_STARTED_EPOCH"]))))')
set +e
timeout --signal=TERM --kill-after=2 "$remaining" taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_g_half_verify.py gpu --output "$attempt/result" --gate reports/documentation/20261015/g_half_implementation/qualification/native_gate.json --prior-seconds 0 > "$attempt/stdout.txt" 2> "$attempt/stderr.txt"
code=$?
printf 'exit_code=%s\n' "$code" >> "$attempt/wrapper.txt"
date -Is >> "$attempt/wrapper.txt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import os,time,json,math,pathlib
p=pathlib.Path('reports/qualification/gpu/attempt01')
elapsed=time.time()-float(os.environ['QUALIFICATION_STARTED_EPOCH'])
json.dump(dict(elapsed_seconds=elapsed,charged_seconds=math.ceil(elapsed),prior_charged_seconds=0,cumulative_charged_seconds=math.ceil(elapsed)),open(p/'wallclock.json','w'),indent=2)
PY
exit "$code"
