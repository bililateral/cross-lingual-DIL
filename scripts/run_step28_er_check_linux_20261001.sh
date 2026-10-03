#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
out="$(realpath -m -- "${1:?Pass a new reports CPU directory}")"
study="${2:-weight}"
case "$study" in
  weight|low|logit|logit_low) ;;
  *) echo 'Unknown confirmed ER study' >&2; exit 2 ;;
esac
case "$out" in
  "$PWD/reports/"*) ;;
  *) echo 'Output must be within project reports' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=''
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date --iso-8601=seconds > "$out/started.txt"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  timeout --signal=TERM --kill-after=30s 1h \
  python -u -B scripts/step28_er_weight_check.py \
  --study "$study" \
  --out "$out/audit.json" > "$out/stdout.log" 2> "$out/stderr.log"
status=$?
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
exit "$status"
