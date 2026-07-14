#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
V7_POLICY="schema/step15_v7_two_stage_policy.json"
STEP12_POLICY="schema/step12_v7_statistical_robustness_policy.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[1/8] Validate Python syntax and v7 configuration contracts"
"$PYTHON_BIN" -m py_compile \
  scripts/step15_freeze_v6_negative_result.py \
  scripts/step20_build_representative_validation.py \
  scripts/step15_build_v7_clean_embedding_cache.py \
  scripts/step15_build_v7_inductive_pair_features.py \
  scripts/step15_v7_common.py \
  scripts/step9_run_v7_latent_pair_mixup.py \
  scripts/step15_v7_apply_two_stage_veto.py \
  scripts/step12_v7_statistical_robustness_audit.py \
  scripts/step20_prepare_prospective_holdout.py \
  scripts/step20_freeze_prospective_holdout.py \
  scripts/step20_build_prospective_features.py \
  scripts/step20_score_prospective_holdout.py \
  scripts/step20_evaluate_prospective_holdout.py
"$PYTHON_BIN" -m unittest tests.test_step15_v7_two_stage_contracts
"$PYTHON_BIN" scripts/step15_build_v7_clean_embedding_cache.py \
  --policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_build_v7_inductive_pair_features.py \
  --policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step9_run_v7_latent_pair_mixup.py \
  --policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_v7_apply_two_stage_veto.py \
  --policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step12_v7_statistical_robustness_audit.py \
  --policy "$STEP12_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step20_prepare_prospective_holdout.py \
  --policy schema/step20_prospective_holdout_policy.json \
  --v7-policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step20_build_prospective_features.py \
  --policy schema/step20_prospective_holdout_policy.json \
  --v7-policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step20_score_prospective_holdout.py \
  --policy schema/step20_prospective_holdout_policy.json \
  --v7-policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step20_evaluate_prospective_holdout.py \
  --policy schema/step20_prospective_holdout_policy.json \
  --validate-config-only

echo "[2/8] Materialize or verify the immutable Step15-v6 strict-negative freeze"
"$PYTHON_BIN" scripts/step15_freeze_v6_negative_result.py \
  --policy schema/step15_v6_negative_freeze.json

echo "[3/8] Build the score-blind, seller-component-disjoint representative validation overlay"
"$PYTHON_BIN" scripts/step20_build_representative_validation.py \
  --policy "$V7_POLICY" \
  --allow-identical-replay

echo "[4/8] Encode identifier-redacted seller content with the frozen Multilingual-E5 model"
"$PYTHON_BIN" scripts/step15_build_v7_clean_embedding_cache.py \
  --policy "$V7_POLICY"

echo "[5/8] Build OOV-safe 20d strict-clean features from v7-train-only references"
"$PYTHON_BIN" scripts/step15_build_v7_inductive_pair_features.py \
  --policy "$V7_POLICY" \
  --allow-identical-replay

echo "[6/8] Run all support ratios, ten seeds, and the three matched Step9-v7 controls"
"$PYTHON_BIN" scripts/step9_run_v7_latent_pair_mixup.py \
  --policy "$V7_POLICY"

echo "[7/8] Select the clean ranker on representative validation and apply the fixed reliability veto"
"$PYTHON_BIN" scripts/step15_v7_apply_two_stage_veto.py \
  --policy "$V7_POLICY"

echo "[8/8] Run grouped, paired-permutation, and two-level Step12-v7 diagnostics; freeze models/thresholds"
"$PYTHON_BIN" scripts/step12_v7_statistical_robustness_audit.py \
  --policy "$STEP12_POLICY" \
  --workers "${STEP12_WORKERS:-24}"

echo "Step15-v7 core chain complete. Publication promotion remains blocked until Step20 succeeds once."
