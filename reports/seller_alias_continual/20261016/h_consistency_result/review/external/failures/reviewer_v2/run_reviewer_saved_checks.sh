#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v -o /mnt/data/hcons_review/return/logs/reviewer_saved_checks.time.txt python /mnt/data/hcons_review/return/scripts/reviewer_saved_checks.py --input /mnt/data/hcons_review/input --output /mnt/data/hcons_review/return/reviewer_independent_v2 > /mnt/data/hcons_review/return/logs/reviewer_saved_checks.stdout.txt 2> /mnt/data/hcons_review/return/logs/reviewer_saved_checks.stderr.txt
rc=$?
printf '%s\n' "$rc" > /mnt/data/hcons_review/return/logs/reviewer_saved_checks.exit_code.txt
exit "$rc"
