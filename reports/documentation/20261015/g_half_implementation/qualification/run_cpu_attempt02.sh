#!/usr/bin/env bash
set -euo pipefail
cd '/home/yongpeng/cross-lingual/reports/documentation/20261015/g_half_implementation/20261015_224000/workspace'
attempt=reports/qualification/cpu/attempt02
test ! -e "$attempt"
mkdir -p "$attempt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import json,hashlib,pathlib,shutil
root=pathlib.Path('.')
files=json.load(open('deployment.json'))['files']
for row in files:
 p=root/row['path'];b=p.read_bytes()
 assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
 target=root/'reports/qualification/cpu/attempt02/sources'/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
json.dump(files,open(root/'reports/qualification/cpu/attempt02/source_files.json','w'),indent=2)
PY
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
date -Is > "$attempt/wrapper.txt"
set +e
timeout --signal=TERM --kill-after=2 173 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_g_half_verify.py cpu --output "$attempt/result" --prior-seconds 5 --tests test_saved_comparisons_directions_permissions_and_budgets test_single_candidate_orchestration_and_blind_gate > "$attempt/stdout.txt" 2> "$attempt/stderr.txt"
code=$?
printf 'exit_code=%s\n' "$code" >> "$attempt/wrapper.txt"
date -Is >> "$attempt/wrapper.txt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import os,time,json,math,pathlib
p=pathlib.Path('reports/qualification/cpu/attempt02')
elapsed=time.time()-float(os.environ['QUALIFICATION_STARTED_EPOCH'])
json.dump(dict(elapsed_seconds=elapsed,charged_seconds=math.ceil(elapsed),prior_charged_seconds=5,cumulative_charged_seconds=5+math.ceil(elapsed)),open(p/'wallclock.json','w'),indent=2)
PY
exit "$code"
