#!/bin/bash
set -u
cd /mnt/data/hcons_review/input
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/mnt/data/hcons_review/mplconfig
python -V > /mnt/data/hcons_review/return/logs/environment.txt 2>&1
python -c 'import numpy,matplotlib,platform; print(platform.platform()); print("numpy",numpy.__version__); print("matplotlib",matplotlib.__version__)' >> /mnt/data/hcons_review/return/logs/environment.txt 2>&1
/usr/bin/time -v -o /mnt/data/hcons_review/return/logs/submitted_audit.time.txt python audit.py --workspace evidence --output /mnt/data/hcons_review/return/reviewer_analysis > /mnt/data/hcons_review/return/logs/submitted_audit.stdout.txt 2> /mnt/data/hcons_review/return/logs/submitted_audit.stderr.txt
rc=$?
printf '%s\n' "$rc" > /mnt/data/hcons_review/return/logs/submitted_audit.exit_code.txt
exit "$rc"
