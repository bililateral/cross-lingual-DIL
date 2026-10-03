#!/usr/bin/env bash
# Handwritten CPU only. Run from the isolated risk_replay workspace.
set -euo pipefail
test "$(uname -s)" = Linux
test ! -e evidence/native.json
mkdir -p evidence
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:tests
ulimit -v 33554432
ulimit -f 65536
date -Is > evidence/native.started
set +e
timeout 2700 taskset -c 46 /home/yongpeng/miniconda3/envs/py310/bin/python -u \
    scripts/step28_risk_replay_check.py --native --output evidence/native.json \
    > evidence/native.stdout 2> evidence/native.stderr
code=$?
set -e
printf '%s\n' "$code" > evidence/native.exit
date -Is > evidence/native.finished
exit "$code"
