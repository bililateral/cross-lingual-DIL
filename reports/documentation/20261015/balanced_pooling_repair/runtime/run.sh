#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
out=reports/qualification/cpu/runtime
mkdir -p "$out"
test ! -e "$out/result" || exit 3
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
timeout --signal=TERM --kill-after=2 116 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python -B \
  reports/documentation/20261015/balanced_pooling_repair/runtime/verify.py "$out/result" > "$out/console.txt" 2>&1
status=$?
/home/yongpeng/miniconda3/envs/py310/bin/python - "$out" "$status" <<'PY'
import json,math,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]);end=time.time();start=float(os.environ['QUALIFICATION_STARTED_EPOCH']);elapsed=end-start
wrapper=dict(exit_code=int(sys.argv[2]),started_epoch=start,ended_epoch=end,actual_wall_seconds=elapsed,
             charged_seconds=math.ceil(elapsed),prior_charged_seconds=172,cumulative_charged_seconds=172+math.ceil(elapsed))
p.joinpath('wrapper.json').write_text(json.dumps(wrapper,indent=2)+'\n')
checks=p/'result/checks.json'
result=json.loads(checks.read_text()) if checks.exists() else {}
result.update(wrapper)
result['status']='PASS_RUNTIME_ONLY' if wrapper['exit_code']==0 and result.get('checks_passed') is True and wrapper['cumulative_charged_seconds']<=300 else 'FAIL_RUNTIME_NO_FORMAL'
p.joinpath('result.json').write_text(json.dumps(result,indent=2)+'\n')
PY
exit "$status"
