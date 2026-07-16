#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="reports/step16i_retrospective_dev2/preparation_v1_20260716"
REVIEW_ROOT="reports/step16i_retrospective_dev2/agent_review_sensitivity_20260716"
OUTPUT_ROOT="reports/step16i_retrospective_dev2/ai_review_reconciliation_v1_20260716"

echo "[1/2] Validate the isolated AI-review reconciliation contracts"
"$PYTHON_BIN" -m unittest tests.test_step16i_ai_review_reconciliation_contracts

echo "[2/2] Reconcile by blind_id without reading blind_mapping or creating labels"
"$PYTHON_BIN" scripts/step16i_reconcile_ai_sensitivity_reviews.py \
  --reviewer-a-queue "$PREP_ROOT/reviewer_a_queue.csv" \
  --reviewer-b-queue "$PREP_ROOT/reviewer_b_queue.csv" \
  --reviewer-a-completed "$REVIEW_ROOT/reviewer_a_completed.csv" \
  --reviewer-b-completed "$REVIEW_ROOT/reviewer_b_completed.csv" \
  --preparation-manifest "$PREP_ROOT/preparation_manifest.json" \
  --policy schema/step16i_retrospective_dev2_policy.json \
  --output-directory "$OUTPUT_ROOT"

echo "Step16I AI-sensitivity reconciliation completed."
echo "Output: $OUTPUT_ROOT"
echo "No Step5 labels were created or modified."
