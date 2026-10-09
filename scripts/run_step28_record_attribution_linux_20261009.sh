#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
core="$1"
gate="$2"
job="$3"
test ! -e "$job"
test -d "$(dirname "$job")"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export RECORD_REPLAY_STARTED_EPOCH="$(date +%s)"
date -Is > "$job.wrapper.txt"
set +e
timeout --signal=TERM --kill-after=5 172795 taskset -c "$core" /home/yongpeng/miniconda3/envs/py310/bin/python \
  scripts/step28_record_attribution_run.py --out "$job" --gate "$gate" > "$job.console.txt" 2>&1
status=$?
printf 'exit_code=%s\n' "$status" >> "$job.wrapper.txt"
date -Is >> "$job.wrapper.txt"
exit "$status"
