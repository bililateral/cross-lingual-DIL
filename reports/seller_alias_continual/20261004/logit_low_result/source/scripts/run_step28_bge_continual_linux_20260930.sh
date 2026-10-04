#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
: "${CUDA_VISIBLE_DEVICES:?Select an inspected idle GPU}"
out="$(realpath -m -- "${1:?Pass a new reports job directory}")"
audit="$(realpath -- "${2:?Pass current CPU audit receipt}")"
authorization="$(realpath -- "${3:?Pass separate formal authorization receipt}")"
case "$out" in
  "$PWD/reports/"*) ;;
  *) echo 'Output must be within project reports' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
unset LD_LIBRARY_PATH LD_PRELOAD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date --iso-8601=seconds > "$out/started.txt"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  timeout --signal=TERM --kill-after=30s 36h \
  python -u -B scripts/step28_bge_continual_run.py execute \
  --out "$out" \
  --audit "$audit" \
  --authorization "$authorization" > "$out/train.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
exit "$status"
