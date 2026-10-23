#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 2
output="$1"
prior="$2"
remaining=$((1200-prior))
test "$remaining" -gt 0 && test ! -e "$output" || exit 2
mkdir -p "$output" || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
started_ns="$(date +%s%N)"
timeout --signal=KILL "${remaining}s" taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python -B \
    scripts/step28_continual_five_verify.py cpu --job "$output" > "${output}.console.txt" 2>&1
code=$?
ended_ns="$(date +%s%N)"
elapsed_ns=$((ended_ns-started_ns))
charged=$(((elapsed_ns+999999999)/1000000000))
printf '{"exit_code":%s,"wall_ns":%s,"charged_seconds":%s,"cumulative_charged_seconds":%s}\n' \
    "$code" "$elapsed_ns" "$charged" "$((prior+charged))" > "${output}.wrapper.json"
exit "$code"
