#!/usr/bin/env bash
# Start only with an approved source-bound gate. No restart/listener.
set -u
cd "$(dirname "$0")/.." || exit 2
core=${1:?idle CPU index}
gate=${2:?reviewed gate JSON}
job=${3:?new reports job}
[[ "$core" =~ ^[0-9]+$ && ! -e "$job" && ! -e "$job.console.txt" ]] || exit 2
[[ -d "$(dirname "$job")" ]] || exit 2
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export FUNCTION_MEMORY_STARTED_EPOCH=$(date +%s)
printf 'started_at=%s\n' "$(date -Is)" > "$job.wrapper.txt"
timeout --signal=TERM --kill-after=5 86395 taskset -c "$core" /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_function_memory_run.py execute --out "$job" --gate "$gate" > "$job.console.txt" 2>&1
status=$?
printf 'exit_code=%s\nended_at=%s\ntotal_wall_seconds=%s\n' "$status" "$(date -Is)" "$(( $(date +%s) - FUNCTION_MEMORY_STARTED_EPOCH ))" >> "$job.wrapper.txt"
exit "$status"
