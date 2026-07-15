#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
READINESS_ROOT="reports/step16_v8_validation_refreeze/readiness_expansion_v2_20260715"
V7_POLICY="$READINESS_ROOT/step15_v7_readiness_policy.json"
V8_POLICY="$READINESS_ROOT/step15_v8_readiness_policy.json"
V8_RUN_ID="${V8_RUN_ID:-bridge_v8_readiness_20260715}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[1/10] Verify the reviewed readiness freeze, contracts, and generated policies"
"$PYTHON_BIN" -m py_compile \
  scripts/step15_build_v7_clean_embedding_cache.py \
  scripts/step15_build_v7_inductive_pair_features.py \
  scripts/step15_v8_common.py \
  scripts/step15_v8_preflight.py \
  scripts/step15_build_v8_clean_semantics.py \
  scripts/step15_run_v8_bridge_audit.py \
  scripts/step15_train_v8_contextual_evidence.py \
  scripts/step12_v8_statistical_robustness_audit.py \
  scripts/step15_v8_downstream_gate.py \
  scripts/step15_v8_build_sync_manifest.py \
  scripts/step15_v8_verify_readiness_runtime.py \
  scripts/step16_build_v8_context_review_queues.py \
  scripts/step16_materialize_v8_reviewed_readiness_freeze.py
"$PYTHON_BIN" -m unittest tests.test_step15_v8_contextual_evidence_contracts
"$PYTHON_BIN" scripts/step16_materialize_v8_reviewed_readiness_freeze.py \
  --run-id readiness_expansion_20260715 \
  --output-root "$READINESS_ROOT" \
  --check-only
"$PYTHON_BIN" scripts/step15_build_v7_clean_embedding_cache.py \
  --policy "$V7_POLICY" \
  --pool zh_target_strict \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_build_v7_inductive_pair_features.py \
  --policy "$V7_POLICY" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_build_v8_clean_semantics.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_run_v8_bridge_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --validate-config-only
"$PYTHON_BIN" scripts/step16_build_v8_context_review_queues.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --validate-config-only
"$PYTHON_BIN" scripts/step15_train_v8_contextual_evidence.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --validate-config-only
"$PYTHON_BIN" scripts/step12_v8_statistical_robustness_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --validate-config-only

RUNTIME_STATUS="$($PYTHON_BIN scripts/step15_v8_verify_readiness_runtime.py \
  --policy "$V8_POLICY" \
  --status | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["status"])')"

echo "[2/10] Build or reuse the content-addressed Chinese identifier-redacted E5 cache"
if [[ "$RUNTIME_STATUS" == "absent" ]]; then
  "$PYTHON_BIN" scripts/step15_build_v7_clean_embedding_cache.py \
    --policy "$V7_POLICY" \
    --pool zh_target_strict \
    --device cuda
else
  echo "Verified runtime bundle exists; reusing the immutable E5 cache."
fi

echo "[3/10] Build or reuse the atomic English/Chinese v7 feature publication"
if [[ "$RUNTIME_STATUS" == "absent" ]]; then
  "$PYTHON_BIN" scripts/step15_build_v7_inductive_pair_features.py \
    --policy "$V7_POLICY"
else
  echo "Verified runtime bundle exists; reusing the immutable v7 feature publication."
fi
"$PYTHON_BIN" scripts/step15_v8_verify_readiness_runtime.py \
  --policy "$V8_POLICY"

echo "[4/10] Run the read-only v8 preflight and enforce the 20/20/15 readiness gate"
"$PYTHON_BIN" scripts/step15_v8_preflight.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[5/10] Snapshot the score-blind occurrence review queues for run provenance"
"$PYTHON_BIN" scripts/step16_build_v8_context_review_queues.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[6/10] Build identifier-redacted BGE/LaBSE/reranker bridge features on GPU"
"$PYTHON_BIN" scripts/step15_build_v8_clean_semantics.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --device cuda

echo "[7/10] Run B0-B3 component-grouped train-OOF bridge audit"
"$PYTHON_BIN" scripts/step15_run_v8_bridge_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[8/10] Train the occurrence-level direction-constrained evidence expert"
"$PYTHON_BIN" scripts/step15_train_v8_contextual_evidence.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[9/10] Apply the preregistered Step12-v8 grouped-bootstrap gates"
"$PYTHON_BIN" scripts/step12_v8_statistical_robustness_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --resamples "${STEP12_RESAMPLES:-5000}"

echo "[10/10] Build the Windows return-sync manifest and evaluate the Step20 gate"
"$PYTHON_BIN" scripts/step15_v8_build_sync_manifest.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

set +e
"$PYTHON_BIN" scripts/step15_v8_downstream_gate.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --stage step20
GATE_STATUS=$?
set -e

if [ "$GATE_STATUS" -eq 0 ]; then
  echo "Step15-v8 passed the preregistered method gate. Freeze Step20 before Step11/17."
elif [ "$GATE_STATUS" -eq 3 ]; then
  echo "Step15-v8 completed as a strict negative result. Do not run Step20 or Step11/17."
else
  exit "$GATE_STATUS"
fi

echo "Step15-v8 readiness run complete: reports/step15_v8/$V8_RUN_ID"
