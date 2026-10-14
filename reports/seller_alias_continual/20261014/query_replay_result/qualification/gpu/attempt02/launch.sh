#!/usr/bin/env bash
set -u
cd '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261013/query_replay_execution/20261013_151500/workspace' || exit 98
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
out=reports/qualification/gpu/attempt02
date -Is > "$out/wrapper.txt"
date +%s.%N >> "$out/wrapper.txt"
timeout --signal=TERM --kill-after=5 394 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_query_replay_verify.py gpu --output "$out/result" --gate reports/qualification/gpu/attempt02/gate.json --prior-seconds 201 > "$out/console.txt" 2>&1
status=$?
printf 'exit_code=%s\n' "$status" >> "$out/wrapper.txt"
date +%s.%N >> "$out/wrapper.txt"
date -Is >> "$out/wrapper.txt"
exit "$status"
