#!/usr/bin/env bash
set -euo pipefail
cd '/home/yongpeng/cross-lingual/reports/documentation/20261016/h_consistency_execution/20261016_124900/workspace'
attempt=reports/qualification/gpu/attempt03
test ! -e "$attempt"
mkdir -p "$attempt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import json,hashlib,pathlib
w=pathlib.Path('.')
for r in json.load(open('deployment_sources.json')):
 b=(w/r['path']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
PY
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
date -Is > "$attempt/wrapper.txt"
remaining=$(/home/yongpeng/miniconda3/envs/py310/bin/python -c 'import os,time; print(max(0.,508-(time.time()-float(os.environ["QUALIFICATION_STARTED_EPOCH"]))))')
set +e
timeout --signal=TERM --kill-after=2 "$remaining" taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_h_consistency_verify.py gpu --output "$attempt/result" --gate reports/documentation/20261016/h_consistency_implementation/qualification/native_objective_gate.json --prior-seconds 90 > "$attempt/stdout.txt" 2> "$attempt/stderr.txt"
code=$?
printf 'exit_code=%s\n' "$code" >> "$attempt/wrapper.txt"
date -Is >> "$attempt/wrapper.txt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import os,time,json,math,pathlib
p=pathlib.Path('reports/qualification/gpu/attempt03')
elapsed=time.time()-float(os.environ['QUALIFICATION_STARTED_EPOCH'])
result=json.load(open(p/'result/result.json')) if (p/'result/result.json').exists() else {}
charge=max(90+math.ceil(elapsed),result.get('cumulative_charged_seconds',90))
json.dump(dict(elapsed_seconds=elapsed,charged_seconds=charge-90,prior_charged_seconds=90,cumulative_charged_seconds=charge),open(p/'wallclock.json','w'),indent=2)
PY
exit "$code"
