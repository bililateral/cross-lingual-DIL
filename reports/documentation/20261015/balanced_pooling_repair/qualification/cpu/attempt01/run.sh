#!/usr/bin/env bash
set -uo pipefail
cd /home/yongpeng/cross-lingual/reports/seller_alias_continual/20261015/balanced_pooling_repair/20261015_105800/workspace
out=reports/qualification/cpu/attempt01
mkdir -p "$out"
test ! -e "$out/result" || exit 3
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
remaining=200
timeout --signal=TERM --kill-after=2 "$remaining" taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_balanced_pooling_verify.py cpu --scope terminal-q --prior-seconds 97 --output "$out/result" > "$out/console.txt" 2>&1
status=$?
/home/yongpeng/miniconda3/envs/py310/bin/python - "$out" "$status" <<'PY'
import json,math,os,sys,time
from pathlib import Path
end=time.time();start=float(os.environ['QUALIFICATION_STARTED_EPOCH']);elapsed=end-start
p=Path(sys.argv[1]);p.joinpath('wrapper.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),started_epoch=start,ended_epoch=end,actual_wall_seconds=elapsed,charged_seconds=math.ceil(elapsed),prior_charged_seconds=97,cumulative_charged_seconds=97+math.ceil(elapsed)),indent=2)+'\n')
PY
exit "$status"
