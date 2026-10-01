#!/usr/bin/env bash
# Approved population study; invoke once through nohup after file/resource checks.
set -euo pipefail
cd /home/yongpeng/cross-lingual
out="reports/seller_alias_continual/20260909/population_execution/20260909_162607"
test -d "$out"
test ! -e "$out/run"
test ! -e "$out/started.txt"
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B scripts/step28_continual_population_run.py --help \
  > "$out/command_help.log"
date -Is > "$out/started.txt"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  nice -n 10 python -u -B scripts/step28_continual_population_run.py \
  --out "$out/run" > "$out/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$out/exit_status.txt"
date -Is > "$out/finished.txt"
exit "$status"
