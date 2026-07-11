#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NON_DOMAIN_EXPERIMENT="step15_v5r_identity_only_curriculum_public_noise_weighted_strong_weighted_mixup"
DOMAIN_EXPERIMENT="step15_v5r_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_weighted_mixup"
SEEDS=(20260320 20260321 20260322)

echo "[1/6] Validate synchronized code and policy"
python3 -m py_compile \
  scripts/step15_train_incremental_hard_negative.py \
  scripts/step15_slice_level_audit.py \
  scripts/step12_statistical_robustness_audit.py
python3 -m unittest tests/test_step15_weighted_mixup.py -v

echo "[2/6] Rebuild Step15 evidence-type labels from the active frozen boundary"
python3 scripts/step15_build_evidence_type_labels.py \
  --policy schema/step15_evidence_type_policy.json \
  --pool en_content_train_pool \
  --pool zh_target_strict

echo "[3/6] Train isolated Step15 v5r Phase3/Phase4 experiments for all seeds"
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment "$NON_DOMAIN_EXPERIMENT" \
  --experiment "$DOMAIN_EXPERIMENT" \
  --phase phase3_add_contact_url_noise \
  --phase phase4_add_positive_pair_mixup \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322

echo "[4/6] Check that every Phase4 parent-provenance manifest exists"
for experiment in "$NON_DOMAIN_EXPERIMENT" "$DOMAIN_EXPERIMENT"; do
  for seed in "${SEEDS[@]}"; do
    manifest="reports/${experiment}_phase4_add_positive_pair_mixup_seed_${seed}_positive_mixup_manifest.csv"
    test -s "$manifest"
  done
done

echo "[5/6] Generate the fixed-test Step15 slice audit"
python3 scripts/step15_slice_level_audit.py \
  --policy schema/step15_evidence_type_policy.json

echo "[6/6] Generate isolated Step12 grouped-bootstrap comparisons"
python3 scripts/step12_statistical_robustness_audit.py \
  --output-json reports/step12_v5r_statistical_robustness_zh_test_weighted_mixup_20260711.json \
  --output-metrics reports/step12_v5r_statistical_robustness_model_metrics_weighted_mixup_20260711.csv \
  --output-comparisons reports/step12_v5r_statistical_robustness_paired_comparisons_weighted_mixup_20260711.csv

echo "Step15 v5r rerun completed. Do not run Step11 until the paired Step12 result is reviewed."
