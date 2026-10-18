#!/usr/bin/env bash
set -u
workspace="$1"
gate="$2"
output="$3"
cd "$workspace" || exit 2
test ! -e "$output" && test -d "$(dirname "$output")" || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
started_ns="$(date +%s%N)"
timeout --signal=KILL 1800s taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python -B \
    scripts/step28_optuna_verify.py --output "$output" --prior-seconds 0 --native-gate "$gate" > "${output}.console.txt" 2>&1
code=$?
ended_ns="$(date +%s%N)"
elapsed_ns=$((ended_ns-started_ns))
charged=$(((elapsed_ns+999999999)/1000000000))
printf 'exit_code=%s\nwall_ns=%s\ncharged_seconds=%s\n' "$code" "$elapsed_ns" "$charged" > "${output}.wrapper.txt"
exit "$code"
