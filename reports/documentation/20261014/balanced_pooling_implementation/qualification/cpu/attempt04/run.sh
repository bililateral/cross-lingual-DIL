#!/bin/bash
cd '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261014/balanced_pooling_execution/20261014_143200/workspace' || exit 1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
start=$(date +%s.%N)
timeout --signal=TERM --kill-after=2s 236s taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_balanced_pooling_verify.py cpu --output '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261014/balanced_pooling_execution/20261014_143200/qualification/cpu/attempt04/result' --prior-seconds 62 > '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261014/balanced_pooling_execution/20261014_143200/qualification/cpu/attempt04/console.txt' 2>&1
code=$?
end=$(date +%s.%N)
printf '{"exit_code":%s,"started_epoch":%s,"ended_epoch":%s}\n' "$code" "$start" "$end" > '/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261014/balanced_pooling_execution/20261014_143200/qualification/cpu/attempt04/wrapper.json'
exit "$code"
