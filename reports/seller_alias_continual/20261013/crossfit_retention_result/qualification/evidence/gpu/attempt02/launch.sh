#!/usr/bin/env bash
set -u
cd '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261012/crossfit_retention_execution/20261012_140200/workspace'
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
out=reports/qualification/gpu/attempt02
date -Is > "$out/wrapper.txt"
date +%s.%N >> "$out/wrapper.txt"
timeout --signal=TERM --kill-after=5 473 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_crossfit_retention_verify.py gpu --output "$out/result" --gate reports/documentation/20261012/crossfit_retention/qualification/native_retry_gate.json --prior-seconds 122 > "$out/console.txt" 2>&1
status=$?
printf 'exit_code=%s\n' "$status" >> "$out/wrapper.txt"
date +%s.%N >> "$out/wrapper.txt"
date -Is >> "$out/wrapper.txt"
exit "$status"
