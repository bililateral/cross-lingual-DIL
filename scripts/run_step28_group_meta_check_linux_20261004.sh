#!/usr/bin/env bash
# Explicit handwritten check, not a formal training launcher or retry loop.
set -euo pipefail
test "$(uname -s)" = Linux
mode=${1:?tests or native}
cpu=${2:?idle CPU chosen from resource inspection}
seconds=${3:?remaining approved aggregate time budget}
run=${4:?fresh evidence run name}
[[ "$cpu" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ && "$run" =~ ^[a-z0-9_]+$ ]]
(( seconds > 0 && seconds <= 7200 ))
[[ "$mode" = tests || "$mode" = native ]]
test ! -e "evidence/$run"
mkdir -p "evidence/$run/source"
cp scripts/step28_group_meta.py scripts/step28_group_meta_check.py \
   tests/test_step28_group_meta.py "evidence/$run/source/"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:tests
ulimit -v 67108864
ulimit -f 131072
args=()
if [[ "$mode" = native ]]; then args+=(--native-only); fi
date -Is > "evidence/$run/started.txt"
set +e
/usr/bin/time -v timeout --kill-after=20 "$seconds" taskset -c "$cpu" \
    /home/yongpeng/miniconda3/envs/py310/bin/python -u \
    scripts/step28_group_meta_check.py --output "evidence/$run/result.json" "${args[@]}" \
    > "evidence/$run/stdout.txt" 2> "evidence/$run/stderr.txt"
code=$?
set -e
printf '%s\n' "$code" > "evidence/$run/exit_status.txt"
date -Is > "evidence/$run/finished.txt"
bytes=$(du -sb evidence | cut -f1)
printf '%s\n' "$bytes" > "evidence/$run/evidence_bytes.txt"
(( bytes <= 134217728 )) || exit 90
exit "$code"
