#!/usr/bin/env bash
set -euo pipefail
cd '/home/yongpeng/cross-lingual/reports/documentation/20261016/h_consistency_execution/20261016_124900/workspace'
attempt=reports/qualification/cpu/attempt01
test ! -e "$attempt"
mkdir -p "$attempt"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import pathlib,json,shutil
root=pathlib.Path.cwd();out=root/'reports/qualification/cpu/attempt01'
rows=json.loads((root/'deployment_sources.json').read_text())
for r in rows:
 p=out/'sources'/r['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/r['path'],p)
(out/'sources.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
export QUALIFICATION_STARTED_EPOCH="$(date +%s.%N)"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$QUALIFICATION_STARTED_EPOCH" > "$attempt/started_epoch.txt"
set +e
timeout --signal=TERM --kill-after=2 178 taskset -c 0 /home/yongpeng/miniconda3/envs/py310/bin/python scripts/step28_h_consistency_verify.py cpu --output "$attempt/result" --prior-seconds 0 > "$attempt/console.txt" 2>&1
status=$?
export HC_EXIT="$status" HC_FINISHED="$(date +%s.%N)"
/home/yongpeng/miniconda3/envs/py310/bin/python - <<'PY'
import pathlib,json,os,math
p=pathlib.Path('reports/qualification/cpu/attempt01')
elapsed=float(os.environ['HC_FINISHED'])-float(os.environ['QUALIFICATION_STARTED_EPOCH'])
r=json.loads((p/'result/result.json').read_text()) if (p/'result/result.json').exists() else {}
record={'exit_code':int(os.environ['HC_EXIT']),'outer_elapsed_seconds':elapsed,'charged_seconds':max(math.ceil(elapsed),r.get('cumulative_charged_seconds',0)),'prior_seconds':0,'cumulative_charged_seconds':max(math.ceil(elapsed),r.get('cumulative_charged_seconds',0)),'started_epoch':float(os.environ['QUALIFICATION_STARTED_EPOCH']),'finished_epoch':float(os.environ['HC_FINISHED'])}
(p/'wrapper.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
exit "$status"
