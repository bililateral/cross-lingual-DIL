#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
stage="${1:?Pass audit or execute}"
out="$(realpath -m -- "${2:?Pass a new stage directory}")"
case "$out" in
  "$PWD/reports/"*) ;;
  *) echo 'Output must be within project reports' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date --iso-8601=seconds > "$out/started.txt"
set +e
if [ "$stage" = audit ]; then
  /usr/bin/time -v -o "$out/resource_usage.log" \
    timeout --signal=TERM --kill-after=10s 30m \
    taskset -c "${CALIBRATION_CPU:-0}" \
    python -u -B scripts/step28_alias_calibration_audit.py \
    --out "$out/evidence" > "$out/stdout.log" 2> "$out/stderr.log"
  status=$?
elif [ "$stage" = execute ]; then
  /usr/bin/time -v -o "$out/resource_usage.log" \
    timeout --signal=TERM --kill-after=10s 1h \
    taskset -c "${CALIBRATION_CPU:-0}" \
    python -u -B scripts/step28_alias_calibration_run.py execute \
    --out "$out/job" \
    --authorization "${3:?Pass the matching readiness receipt}" \
    > "$out/stdout.log" 2> "$out/stderr.log"
  status=$?
else
  echo 'Only audit or execute is supported' >&2
  status=2
fi
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
exit "$status"
