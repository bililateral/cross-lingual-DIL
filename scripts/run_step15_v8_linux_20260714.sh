#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
V8_POLICY="schema/step15_v8_contextual_evidence_policy.json"
V8_RUN_ID="${V8_RUN_ID:-bridge_v1_20260714}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[1/7] Validate syntax, contracts, and isolated v8 output identity"
"$PYTHON_BIN" -m py_compile \
  scripts/step15_v8_common.py \
  scripts/step15_v8_preflight.py \
  scripts/step15_build_v8_clean_semantics.py \
  scripts/step15_run_v8_bridge_audit.py \
  scripts/step16_build_v8_context_review_queues.py \
  scripts/step15_train_v8_contextual_evidence.py \
  scripts/step12_v8_statistical_robustness_audit.py \
  scripts/step15_v8_downstream_gate.py \
  scripts/step15_v8_build_sync_manifest.py
"$PYTHON_BIN" -m unittest tests.test_step15_v8_contextual_evidence_contracts
"$PYTHON_BIN" scripts/step15_v8_preflight.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"
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

echo "[2/7] Build score-blind risky-only, mixed-context, and direct-evidence review queues"
"$PYTHON_BIN" scripts/step16_build_v8_context_review_queues.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[3/7] Encode identifier-redacted BGE/LaBSE semantics and reranker scores on GPU"
"$PYTHON_BIN" scripts/step15_build_v8_clean_semantics.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --device cuda

echo "[4/7] Run B0-B3 repeated component-grouped train-OOF bridge audit and linear ranker control"
"$PYTHON_BIN" scripts/step15_run_v8_bridge_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[5/7] Train the occurrence-level contextual evidence expert from clean OOF probabilities"
"$PYTHON_BIN" scripts/step15_train_v8_contextual_evidence.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID"

echo "[6/7] Apply grouped-bootstrap validation gates; internal test remains diagnostic only"
"$PYTHON_BIN" scripts/step12_v8_statistical_robustness_audit.py \
  --policy "$V8_POLICY" \
  --run-id "$V8_RUN_ID" \
  --resamples "${STEP12_RESAMPLES:-5000}"

echo "[7/7] Build the exact Windows return-sync manifest and evaluate the Step20 gate"
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
  echo "Step15-v8 passed its preregistered method gate. Prepare and freeze Step20 before any Step11/17 run."
elif [ "$GATE_STATUS" -eq 3 ]; then
  echo "Step15-v8 completed as a strict negative/blocked result. Do not run Step20, Step11, or Step17."
else
  exit "$GATE_STATUS"
fi

echo "Step15-v8 run complete: reports/step15_v8/$V8_RUN_ID"
