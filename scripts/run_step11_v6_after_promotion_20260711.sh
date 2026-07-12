#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
CLEAN_RUNTIME_POLICY="reports/step15_v6/manifests/step11_v6_clean_topology_runtime_policy.json"
OPERATIONAL_RUNTIME_POLICY="reports/step15_v6/manifests/step11_v6_identifier_operational_runtime_policy.json"

SEEDS=(20260320 20260321 20260322 20260323 20260324 20260325 20260326 20260327 20260328 20260329)

echo "[1/9] Build a promotion-gated, auto-selector-disabled Step11 runtime policy"
"$PYTHON_BIN" -m py_compile \
  scripts/immutable_artifact_io.py \
  scripts/step11_build_v6_runtime_policy.py \
  scripts/step11_cluster_chinese_graph.py \
  scripts/step11_build_explicit_summary_manifest.py \
  scripts/step11_cluster_level_audit.py \
  scripts/step13_concept_drift_audit.py
"$PYTHON_BIN" scripts/step11_build_v6_runtime_policy.py \
  --step12-summary reports/step12_v6/method_audit_v4_inductive_20260712/step12_v6_statistical_robustness.json \
  --validation-mode clean_topology \
  --output "$CLEAN_RUNTIME_POLICY"
"$PYTHON_BIN" scripts/step11_build_v6_runtime_policy.py \
  --step12-summary reports/step12_v6/method_audit_v4_inductive_20260712/step12_v6_statistical_robustness.json \
  --validation-mode identifier_assisted_operational \
  --output "$OPERATIONAL_RUNTIME_POLICY"

echo "[2/9] Run only the validation-selected clean Step15-v6 ensemble"
"$PYTHON_BIN" scripts/step11_cluster_chinese_graph.py \
  --policy "$CLEAN_RUNTIME_POLICY" \
  --scorer-family step15

echo "[3/9] Run the matched-binary M0 ten-seed ensemble"
STEP15_SEED_ARGS=()
for seed in "${SEEDS[@]}"; do
  STEP15_SEED_ARGS+=(--step15-seed "$seed")
done
"$PYTHON_BIN" scripts/step11_cluster_chinese_graph.py \
  --policy "$CLEAN_RUNTIME_POLICY" \
  --scorer-family step15 \
  --step15-experiment step15_v6_m0_all_at_once_binary \
  --step15-phase phase3_add_contact_url_noise \
  "${STEP15_SEED_ARGS[@]}"

echo "[4/9] Run raw BGE semantic control with the Step12-frozen zh_valid threshold"
"$PYTHON_BIN" scripts/step11_cluster_chinese_graph.py \
  --policy "$CLEAN_RUNTIME_POLICY" \
  --scorer-family raw_feature \
  --raw-feature-control raw_bge_m3_cosine

STEP9_SEED_ARGS=()
for seed in "${SEEDS[@]}"; do
  STEP9_SEED_ARGS+=(--step9-seed "$seed")
done

echo "[5/9] Run one validation-selected strongest-clean Step9 ten-seed mean graph"
"$PYTHON_BIN" scripts/step11_cluster_chinese_graph.py \
  --policy "$CLEAN_RUNTIME_POLICY" \
  --scorer-family step9 \
  --step9-ratio 1.0 \
  "${STEP9_SEED_ARGS[@]}"

echo "[6/9] Run one separate identifier-operational ten-seed mean graph"
"$PYTHON_BIN" scripts/step11_cluster_chinese_graph.py \
  --policy "$OPERATIONAL_RUNTIME_POLICY" \
  --scorer-family step9 \
  --step9-experiment identifier_augmented_few_shot_default_lr_l2 \
  --step9-ratio 1.0 \
  "${STEP9_SEED_ARGS[@]}"

FINAL_SUMMARY="reports/step11_v6/clean_topology/step11_step15_v6_final_selected_seed_mean_clustering_summary.json"
M0_SUMMARY="reports/step11_v6/clean_topology/step11_step15_v6_m0_seed_mean_clustering_summary.json"
RAW_BGE_SUMMARY="reports/step11_v6/clean_topology/step11_raw_bge_m3_cosine_clustering_summary.json"
STEP9_MEAN_SUMMARY="reports/step11_v6/clean_topology/step11_step9_v6_strongest_clean_selected_seed_mean_clustering_summary.json"
IDENTIFIER_MEAN_SUMMARY="reports/step11_v6/identifier_operational/step11_step9_v6_identifier_operational_seed_mean_clustering_summary.json"
CLEAN_SUMMARY_ARGS=(--summary "$FINAL_SUMMARY" --summary "$M0_SUMMARY" --summary "$RAW_BGE_SUMMARY" --summary "$STEP9_MEAN_SUMMARY")
OPERATIONAL_SUMMARY_ARGS=(--summary "$IDENTIFIER_MEAN_SUMMARY")
CLEAN_EXPECTED_SCORERS=(
  --expected-scorer-token step15_v6_final_selected_seed_mean
  --expected-scorer-token step15_v6_m0_seed_mean
  --expected-scorer-token raw_bge_m3_cosine
  --expected-scorer-token step9_v6_strongest_clean_selected_seed_mean
)
OPERATIONAL_EXPECTED_SCORERS=(
  --expected-scorer-token step9_v6_identifier_operational_seed_mean
)

echo "[7/9] Freeze separate clean and identifier-operational allow-lists; no reports glob is permitted"
"$PYTHON_BIN" scripts/step11_build_explicit_summary_manifest.py \
  --run-id step15-v6-v4-step11-clean-validation-20260712 \
  --publication-v6 \
  --validation-mode clean_topology \
  "${CLEAN_SUMMARY_ARGS[@]}" \
  "${CLEAN_EXPECTED_SCORERS[@]}" \
  --output-json reports/step11_v6/step11_v6_clean_explicit_manifest.json \
  --output-csv reports/step11_v6/step11_v6_clean_explicit_manifest.csv
"$PYTHON_BIN" scripts/step11_build_explicit_summary_manifest.py \
  --run-id step15-v6-v4-step11-identifier-operational-20260712 \
  --publication-v6 \
  --validation-mode identifier_assisted_operational \
  "${OPERATIONAL_SUMMARY_ARGS[@]}" \
  "${OPERATIONAL_EXPECTED_SCORERS[@]}" \
  --output-json reports/step11_v6/step11_v6_identifier_operational_manifest.json \
  --output-csv reports/step11_v6/step11_v6_identifier_operational_manifest.csv

echo "[8/9] Run separate clean and operational cluster audits over their explicit allow-lists"
"$PYTHON_BIN" scripts/step11_cluster_level_audit.py \
  --publication-v6 \
  --manifest reports/step11_v6/step11_v6_clean_explicit_manifest.json \
  --output-csv reports/step11_v6/step11_cluster_level_audit.step15_v6_clean_20260712.csv \
  --output-summary reports/step11_v6/step11_cluster_level_audit.step15_v6_clean_20260712.json
"$PYTHON_BIN" scripts/step11_cluster_level_audit.py \
  --publication-v6 \
  --manifest reports/step11_v6/step11_v6_identifier_operational_manifest.json \
  --output-csv reports/step11_v6/step11_cluster_level_audit.step15_v6_identifier_operational_20260712.csv \
  --output-summary reports/step11_v6/step11_cluster_level_audit.step15_v6_identifier_operational_20260712.json

echo "[9/9] Regenerate Step13 from the clean graph audit; operational results remain separate"
"$PYTHON_BIN" scripts/step13_concept_drift_audit.py \
  --step12-v6-summary reports/step12_v6/method_audit_v4_inductive_20260712/step12_v6_statistical_robustness.json \
  --step7-summary reports/step15_v6/baselines/step7_source_only_default_fusion_summary.json \
  --step9-summary reports/step15_v6/baselines/step9/step9_few_shot_summary.json \
  --step11-manifest reports/step11_v6/step11_v6_clean_explicit_manifest.json \
  --step11-audit reports/step11_v6/step11_cluster_level_audit.step15_v6_clean_20260712.json \
  --output-json reports/step13_concept_drift_audit.step15_v6_step11_validation_20260712.json \
  --output-csv reports/step13_concept_drift_audit.step15_v6_step11_validation_20260712.csv \
  --output-md docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP15_V6_STEP11_VALIDATION_20260712.md

echo "Completed explicit Step11-v6 clean and operational validation. Do not interpret clusters as ground truth without blind evidence review."
