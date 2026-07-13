#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
V6_POLICY="schema/step15_v6_paper_hardening_policy.json"
STEP12_POLICY="schema/step12_v6_statistical_robustness_policy.json"
UNIVERSE_BEFORE="reports/step15_v6/manifests/step4_candidate_universe_before_v6.json"

echo "[1/11] Static syntax and configuration contracts"
"$PYTHON_BIN" -m py_compile \
  scripts/immutable_artifact_io.py \
  scripts/step4_build_silver_candidates.py \
  scripts/step4_verify_candidate_universe.py \
  scripts/step7_build_pair_feature_preview.py \
  scripts/step7_refresh_nonsemantic_pair_features.py \
  scripts/step7_train_baseline_models.py \
  scripts/step15_v6_refresh_step7_control.py \
  scripts/step15_build_v6_inductive_pair_features.py \
  scripts/step9_run_few_shot_adaptation.py \
  scripts/step15_train_incremental_hard_negative.py \
  scripts/step15_build_active_run_manifest.py \
  scripts/step15_validate_v6_outputs.py \
  scripts/step15_v6_source_only_lr_baseline.py \
  scripts/step12_v6_statistical_robustness_audit.py \
  scripts/step13_concept_drift_audit.py
"$PYTHON_BIN" -m unittest discover -s tests -p "test_*.py"
"$PYTHON_BIN" scripts/step15_train_incremental_hard_negative.py \
  --policy "$V6_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_build_v6_inductive_pair_features.py \
  --policy "$V6_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_v6_source_only_lr_baseline.py \
  --policy "$V6_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step12_v6_statistical_robustness_audit.py \
  --policy "$STEP12_POLICY" \
  --validate-config-only

echo "[2/11] Verify frozen canonical inputs without rewriting Step4, Step7, or evidence labels"
"$PYTHON_BIN" scripts/step4_verify_candidate_universe.py \
  --mode verify \
  --manifest "$UNIVERSE_BEFORE" \
  --verification-output reports/step15_v6/manifests/step4_candidate_universe_verification.linux.json
echo "[3/11] Build isolated v6 features from frozen train-seller corpus references"
"$PYTHON_BIN" scripts/step15_build_v6_inductive_pair_features.py \
  --policy "$V6_POLICY" \
  --allow-identical-replay

echo "[4/11] Refresh the Step7 metric-v2 control without retraining or overwriting canonical Step7 artifacts"
"$PYTHON_BIN" scripts/step15_v6_refresh_step7_control.py

echo "[5/11] Refresh only the isolated Step9 clean/operational controls required by Step12-v6"
"$PYTHON_BIN" scripts/step9_run_few_shot_adaptation.py \
  --output-root reports/step15_v6/baselines/step9 \
  --en-pair-features reports/step15_v6/features/step7_pair_features.en_content_train_pool.inductive_train_reference.csv \
  --zh-pair-features reports/step15_v6/features/step7_pair_features.zh_target_strict.inductive_train_reference.csv \
  --experiment core_few_shot_labse_lr_l2 \
  --experiment core_few_shot_multilingual_e5_large_lr_l2 \
  --experiment core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup \
  --experiment core_few_shot_bge_m3_residual_lr \
  --experiment identifier_augmented_few_shot_default_lr_l2 \
  --ratio 1.0 \
  --seed 20260320 \
  --seed 20260321 \
  --seed 20260322 \
  --seed 20260323 \
  --seed 20260324 \
  --seed 20260325 \
  --seed 20260326 \
  --seed 20260327 \
  --seed 20260328 \
  --seed 20260329

echo "[6/11] Train the strict-clean source-only LR/L2 control"
"$PYTHON_BIN" scripts/step15_v6_source_only_lr_baseline.py \
  --policy "$V6_POLICY"

echo "[7/11] Train preregistered Step15-v6 M0-M5 plus matched-budget controls for ten seeds"
"$PYTHON_BIN" scripts/step15_train_incremental_hard_negative.py \
  --policy "$V6_POLICY"

echo "[8/11] Train feature-lineage and label-provenance ablations; merge into the same manifest-safe summary"
"$PYTHON_BIN" scripts/step15_train_incremental_hard_negative.py \
  --policy "$V6_POLICY" \
  --experiment step15_v6_m3_normalized_retrieval_ablation \
  --experiment step15_v6_m3_gold_only_ablation \
  --experiment step15_v6_m3_gold_plus_high_confidence_silver_ablation

echo "[9/11] Validate complete run coverage, fixed budgets, endpoint-only test, and matched pre-treatment predictions"
"$PYTHON_BIN" scripts/step15_validate_v6_outputs.py \
  --policy "$V6_POLICY" \
  --output reports/step15_v6/manifests/step15_v6_output_validation_v4_20260712.json

echo "[10/11] Freeze all Step15-v6 and selected Step9 outputs in an explicit active manifest"
"$PYTHON_BIN" scripts/step15_build_active_run_manifest.py \
  --run-id step15-v6-method-audit-v4-inductive-internal-dev-20260712 \
  --summary reports/step15_v6/step15_v6_training_summary.json \
  --extra-summary reports/step15_v6/baselines/source_only_lr_l2_summary.json \
  --step9-summary reports/step15_v6/baselines/step9/step9_few_shot_summary.json \
  --step9-experiment core_few_shot_labse_lr_l2 \
  --step9-experiment core_few_shot_multilingual_e5_large_lr_l2 \
  --step9-experiment core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup \
  --step9-experiment core_few_shot_bge_m3_residual_lr \
  --step9-experiment identifier_augmented_few_shot_default_lr_l2 \
  --step9-ratio-token 100pct \
  --step9-seed 20260320 \
  --step9-seed 20260321 \
  --step9-seed 20260322 \
  --step9-seed 20260323 \
  --step9-seed 20260324 \
  --step9-seed 20260325 \
  --step9-seed 20260326 \
  --step9-seed 20260327 \
  --step9-seed 20260328 \
  --step9-seed 20260329 \
  --extra-file reports/step15_v6/manifests/step15_v6_output_validation_v4_20260712.json \
  --extra-file reports/step15_v6/manifests/step15_v6_inductive_feature_manifest.json \
  --extra-file reports/step15_v6/features/train_only_corpus_reference.json \
  --extra-file reports/step15_v6/baselines/step7_source_only_default_fusion_summary.json \
  --extra-file reports/step7_training_summary.json \
  --extra-file reports/step7_core_zero_shot_default_predictions.zh_target_strict_test.csv \
  --extra-file reports/step5_en_frozen_silver_labels.csv \
  --extra-file reports/step5_zh_target_strict_frozen_silver_labels.csv \
  --extra-file reports/step3_seller_profiles.en_content_train_pool.jsonl \
  --extra-file reports/step3_seller_profiles.zh_target_strict.jsonl \
  --extra-file reports/step4_en_silver_candidate_pairs.csv \
  --extra-file reports/step4_zh_target_strict_silver_candidate_pairs.csv \
  --extra-file reports/step7_pair_features.en_content_train_pool.csv \
  --extra-file reports/step7_pair_features.zh_target_strict.csv \
  --extra-file reports/step15_v6/features/step7_pair_features.en_content_train_pool.inductive_train_reference.csv \
  --extra-file reports/step15_v6/features/step7_pair_features.zh_target_strict.inductive_train_reference.csv \
  --extra-file reports/step15_evidence_type_labels.en_content_train_pool.csv \
  --extra-file reports/step15_evidence_type_labels.zh_target_strict.csv \
  --extra-file schema/step7_training_policy.json \
  --extra-file schema/step9_training_policy.json \
  --extra-file scripts/step15_build_v6_inductive_pair_features.py \
  --extra-file scripts/step15_train_incremental_hard_negative.py \
  --extra-file scripts/step15_v6_source_only_lr_baseline.py \
  --extra-file scripts/step9_run_few_shot_adaptation.py \
  --extra-file scripts/step7_train_baseline_models.py \
  --policy "$V6_POLICY" \
  --output-json reports/step15_v6/manifests/step15_v6_internal_dev_v4_20260712.json \
  --output-csv reports/step15_v6/manifests/step15_v6_internal_dev_v4_20260712.csv \
  --evaluation-role fixed_internal_development_test_not_prospective_final_holdout

echo "[11/11] Validate and run Step12-v6 grouped/permutation/two-level audit"
"$PYTHON_BIN" scripts/step12_v6_statistical_robustness_audit.py \
  --policy "$STEP12_POLICY" \
  --validate-inputs-only
"$PYTHON_BIN" scripts/step12_v6_statistical_robustness_audit.py \
  --policy "$STEP12_POLICY" \
  --workers 24

echo "Step15-v6 core rerun complete. Inspect Step12 promotion before Step11-v6; final Step13 is generated only after the explicit Step11 audit."
