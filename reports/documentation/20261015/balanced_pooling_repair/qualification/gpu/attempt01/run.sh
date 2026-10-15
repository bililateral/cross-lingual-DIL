#!/usr/bin/env bash
set -uo pipefail
cd /home/yongpeng/cross-lingual/reports/seller_alias_continual/20261015/balanced_pooling_repair/20261015_120600/workspace || exit 3
out=reports/qualification/gpu/attempt01
mkdir -p "$out"
test ! -e "$out/result" || exit 3
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
# 532 seconds + at most 2 seconds termination grace leaves wrapper accounting
# inside the remaining 536 seconds; all actual elapsed time is charged below.
timeout --signal=TERM --kill-after=2 532 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python -B scripts/step28_balanced_pooling_verify.py gpu --scope terminal-q --prior-seconds 364 --gate reports/documentation/20261015/balanced_pooling_repair/qualification/native_gate.json --output "$out/result" > "$out/console.txt" 2>&1
status=$?
/home/yongpeng/miniconda3/envs/py310/bin/python - "$out" "$status" <<'PY'
import json, math, os, sys, time
from pathlib import Path
end=time.time(); start=float(os.environ['QUALIFICATION_STARTED_EPOCH']); elapsed=end-start
p=Path(sys.argv[1]); p.joinpath('wrapper.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),started_epoch=start,ended_epoch=end,actual_wall_seconds=elapsed,charged_seconds=math.ceil(elapsed),prior_charged_seconds=364,cumulative_charged_seconds=364+math.ceil(elapsed)),indent=2)+'\n')
PY
exit "$status"
