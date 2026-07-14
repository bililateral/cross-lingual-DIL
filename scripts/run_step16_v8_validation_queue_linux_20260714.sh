#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
V8_POLICY="${V8_POLICY:-schema/step15_v8_contextual_evidence_policy.json}"
REVIEW_RUN_ID="${REVIEW_RUN_ID:-validation_expansion_queue_v1_20260714}"

echo "[1/3] Validate Step16-v8 review/refreeze code and contracts"
"$PYTHON_BIN" -m py_compile \
  scripts/step15_v8_common.py \
  scripts/step16_build_v8_context_review_queues.py \
  scripts/step16_apply_v8_context_reviews.py
"$PYTHON_BIN" -m unittest tests.test_step15_v8_contextual_evidence_contracts
"$PYTHON_BIN" scripts/step16_build_v8_context_review_queues.py \
  --policy "$V8_POLICY" \
  --run-id "$REVIEW_RUN_ID" \
  --validate-config-only
"$PYTHON_BIN" scripts/step16_apply_v8_context_reviews.py \
  --validate-config-only

echo "[2/3] Build immutable occurrence-context queues and separately shuffled blind packets"
"$PYTHON_BIN" scripts/step16_build_v8_context_review_queues.py \
  --policy "$V8_POLICY" \
  --run-id "$REVIEW_RUN_ID"

echo "[3/3] Stop before training; review data must return before refreeze"
echo "Sync this directory back to Windows: reports/step15_v8/$REVIEW_RUN_ID/context_review/"
echo "Give reviewer_a_blind_packet.template.csv and reviewer_b_blind_packet.template.csv to different reviewers."
echo "Save completed copies as reviewer_a_blind_packet.completed.csv and reviewer_b_blind_packet.completed.csv."
echo "Use reviewer_adjudicator_blind_packet.template.csv only for disagreements reported by --check-only."
echo "Do not edit the immutable *_blind_review_queue.csv files."
