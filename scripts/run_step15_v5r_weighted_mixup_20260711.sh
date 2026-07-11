#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NON_DOMAIN_EXPERIMENT="step15_v5r_identity_only_curriculum_public_noise_weighted_strong_weighted_mixup"
DOMAIN_EXPERIMENT="step15_v5r_identity_only_curriculum_domain_balanced_public_noise_weighted_strong_weighted_mixup"
SEEDS=(20260320 20260321 20260322)

echo "[1/7] Validate synchronized code and policy"
python3 -m py_compile \
  scripts/step15_train_incremental_hard_negative.py \
  scripts/step15_slice_level_audit.py \
  scripts/step12_statistical_robustness_audit.py
python3 -m unittest tests/test_step15_weighted_mixup.py -v
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment "$NON_DOMAIN_EXPERIMENT" \
  --experiment "$DOMAIN_EXPERIMENT" \
  --phase phase3_add_contact_url_noise \
  --phase phase4_add_positive_pair_mixup \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322 \
  --validate-config-only

echo "[2/7] Rebuild Step15 evidence-type labels from the active frozen boundary"
python3 scripts/step15_build_evidence_type_labels.py \
  --policy schema/step15_evidence_type_policy.json \
  --pool en_content_train_pool \
  --pool zh_target_strict

echo "[3/7] Train isolated Step15 v5r Phase3/Phase4 experiments for all seeds"
python3 scripts/step15_train_incremental_hard_negative.py \
  --policy schema/step15_evidence_type_policy.json \
  --experiment "$NON_DOMAIN_EXPERIMENT" \
  --experiment "$DOMAIN_EXPERIMENT" \
  --phase phase3_add_contact_url_noise \
  --phase phase4_add_positive_pair_mixup \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322

echo "[4/7] Validate every v5r artifact, parent manifest, inherited weight, and domain mass"
python3 scripts/step15_validate_v5r_outputs.py \
  --output reports/step15_v5r_output_contract_validation.json

echo "[5/7] Generate the fixed-test Step15 slice audit"
python3 scripts/step15_slice_level_audit.py \
  --policy schema/step15_evidence_type_policy.json

echo "[6/7] Generate isolated Step12 grouped-bootstrap comparisons"
python3 scripts/step12_statistical_robustness_audit.py \
  --output-json reports/step12_v5r_statistical_robustness_zh_test_weighted_mixup_20260711.json \
  --output-metrics reports/step12_v5r_statistical_robustness_model_metrics_weighted_mixup_20260711.csv \
  --output-comparisons reports/step12_v5r_statistical_robustness_paired_comparisons_weighted_mixup_20260711.csv

echo "[7/7] Verify final required reports"
test -s reports/step15_v5r_weighted_mixup_summary.json
test -s reports/step15_v5r_output_contract_validation.json
test -s reports/step15_v5r_weighted_mixup_slice_level_audit.json
test -s reports/step15_v5r_weighted_mixup_slice_level_audit.csv
test -s reports/step12_v5r_statistical_robustness_zh_test_weighted_mixup_20260711.json
test -s reports/step12_v5r_statistical_robustness_model_metrics_weighted_mixup_20260711.csv
test -s reports/step12_v5r_statistical_robustness_paired_comparisons_weighted_mixup_20260711.csv

echo "Step15 v5r rerun completed. Do not run Step11 until the paired Step12 result is reviewed."
