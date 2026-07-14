#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
REFREEZE_POLICY="${REFREEZE_POLICY:-schema/step16_v8_validation_refreeze_policy.json}"
REVIEWER_A_FILE="${REVIEWER_A_FILE:-reports/step15_v8/validation_expansion_queue_v1_20260714/context_review/reviewer_a_blind_packet.completed.csv}"
REVIEWER_B_FILE="${REVIEWER_B_FILE:-reports/step15_v8/validation_expansion_queue_v1_20260714/context_review/reviewer_b_blind_packet.completed.csv}"
ADJUDICATION_FILE="${ADJUDICATION_FILE:-reports/step15_v8/validation_expansion_queue_v1_20260714/context_review/reviewer_adjudicator_blind_packet.completed.csv}"

echo "[1/4] Validate review application code"
"$PYTHON_BIN" -m py_compile \
  scripts/step15_v8_common.py \
  scripts/step16_apply_v8_context_reviews.py \
  scripts/step15_v8_preflight.py \
  scripts/step12_v8_statistical_robustness_audit.py
"$PYTHON_BIN" -m unittest tests.test_step15_v8_contextual_evidence_contracts
"$PYTHON_BIN" scripts/step16_apply_v8_context_reviews.py \
  --policy "$REFREEZE_POLICY" \
  --validate-config-only

echo "[2/4] Resolve dual reviews and report readiness without writing a freeze"
"$PYTHON_BIN" scripts/step16_apply_v8_context_reviews.py \
  --policy "$REFREEZE_POLICY" \
  --reviewer-a-file "$REVIEWER_A_FILE" \
  --reviewer-b-file "$REVIEWER_B_FILE" \
  --adjudication-file "$ADJUDICATION_FILE" \
  --check-only

echo "[3/4] Materialize the isolated component-safe overlay only when 20/20/15 is met"
"$PYTHON_BIN" scripts/step16_apply_v8_context_reviews.py \
  --policy "$REFREEZE_POLICY" \
  --reviewer-a-file "$REVIEWER_A_FILE" \
  --reviewer-b-file "$REVIEWER_B_FILE" \
  --adjudication-file "$ADJUDICATION_FILE"

echo "[4/4] Print the generated policy for the subsequent full v8 run"
echo "Generated policy: reports/step16_v8_validation_refreeze/context_reviewed_v1_20260714/step15_v8_context_reviewed_policy.json"
echo "No original Step5, Step15 evidence-label, or representative-validation file was modified."
