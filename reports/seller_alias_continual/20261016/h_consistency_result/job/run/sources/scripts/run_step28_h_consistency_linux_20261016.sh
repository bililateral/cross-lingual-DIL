#!/usr/bin/env bash
set -euo pipefail
export RECORD_REPLAY_STARTED_EPOCH="$(date +%s)"
cd "$(dirname "$0")/.."
cpu="$1"
gate="$2"
job="$3"
test ! -e "$job"
test -d "$(dirname "$job")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# Wrapper and interpreter startup, validation, and TERM grace are inside one deadline.
limit=$(/home/yongpeng/miniconda3/envs/py310/bin/python -c 'import json; p=json.load(open("schema/step28_h_consistency_policy.json")); r=p["runtime"]; assert r["maximum_gpu_stage_seconds"]==28800; print(r["maximum_gpu_stage_seconds"])')
remaining=$((RECORD_REPLAY_STARTED_EPOCH + limit - $(date +%s) - 5))
test "$remaining" -gt 0
date -Is > "$job.wrapper.txt"
set +e
timeout --signal=TERM --kill-after=5 "$remaining" taskset -c "$cpu" /home/yongpeng/miniconda3/envs/py310/bin/python \
  scripts/step28_h_consistency_run.py --job "$job" --gate "$gate" > "$job.console.txt" 2>&1
status=$?
printf 'exit_code=%s\n' "$status" >> "$job.wrapper.txt"
date -Is >> "$job.wrapper.txt"
exit "$status"
