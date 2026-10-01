#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
stage="${1:?Pass audit or execute}"
out="$(realpath -m -- "${2:?Pass a new output directory}")"
case "$out" in
  "$PWD/reports/"*) ;;
  *) echo 'Output must stay in project reports' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "$stage" = audit ]; then
  export CUDA_VISIBLE_DEVICES=''
  command=(python -u -B scripts/step28_alias_test_audit.py
    --out "$out/evidence")
elif [ "$stage" = execute ]; then
  export CUDA_VISIBLE_DEVICES="${TEST_GPU:?Select a confirmed free GPU}"
  command=(python -u -B scripts/step28_alias_test_run.py execute
    --out "$out/job"
    --authorization "$(realpath -- "${3:?Pass reviewed readiness receipt}")")
else
  echo 'Only audit or execute supported' >&2
  exit 2
fi
date --iso-8601=seconds > "$out/started.txt"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  timeout --signal=TERM --kill-after=10s 1h \
  taskset -c "${TEST_CPU:-0}" \
  "${command[@]}" > "$out/stdout.log" 2> "$out/stderr.log"
status=$?
set -e
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
exit "$status"
