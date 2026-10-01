#!/usr/bin/env bash
set -euo pipefail
cd /home/yongpeng/cross-lingual
source /home/yongpeng/miniconda3/etc/profile.d/conda.sh
conda activate py310
out="$(realpath -m -- "${1:?Pass a new CPU evidence directory}")"
case "$out" in
  "$PWD/reports/"*) ;;
  *) echo 'Output must be inside project reports' >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
mkdir "$out/work"
unset LD_LIBRARY_PATH LD_PRELOAD
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$out/work"
export CHINESE_CHECK_OUT="$out"
date --iso-8601=seconds > "$out/started.txt"
set +e
/usr/bin/time -v -o "$out/resource_usage.log" \
  timeout --signal=TERM --kill-after=30s 1h bash -euo pipefail -c '
    python -B scripts/step28_chinese_base.py --help
    python -B scripts/step28_chinese_base_audit.py --help
    bash -n scripts/run_step28_chinese_base_linux_20260924.sh
    python -B -m unittest discover -s tests \
      -p test_step28_chinese_base_contracts.py -v
    for arm in mean_bce mean_rank split_bce split_rank; do
      python -u -B scripts/step28_chinese_base_audit.py \
        --arm "$arm" \
        --out "$CHINESE_CHECK_OUT/$arm.json"
    done
  ' > "$out/runtime.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/exit_status.txt"
date --iso-8601=seconds > "$out/finished.txt"
bytes="$(du -sb "$out" | cut -f1)"
if (( bytes > 268435456 )); then
  echo 'CPU evidence exceeded the declared 256 MiB budget' >&2
  exit 3
fi
exit "$status"
