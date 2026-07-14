#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step20_prospective_holdout_policy.json"
V7_POLICY="schema/step15_v7_two_stage_policy.json"
ACTION="${1:-}"

case "$ACTION" in
  prepare)
    echo "[Step20 prepare] Build blinded, score-free, post-v7 candidate queues"
    "$PYTHON_BIN" scripts/step20_prepare_prospective_holdout.py \
      --policy "$POLICY" \
      --v7-policy "$V7_POLICY"
    echo "Complete reviewer_a_queue.csv and reviewer_b_queue.csv independently before freezing."
    ;;
  freeze-and-score)
    echo "[Step20 freeze 1/4] Reconcile dual reviews/adjudication and freeze labels fail-closed"
    "$PYTHON_BIN" scripts/step20_freeze_prospective_holdout.py \
      --policy "$POLICY" \
      --v7-policy "$V7_POLICY"
    echo "[Step20 freeze 2/4] Build identifier-redacted prospective E5 cache without reading labels"
    "$PYTHON_BIN" scripts/step15_build_v7_clean_embedding_cache.py \
      --policy "$V7_POLICY" \
      --prospective-policy "$POLICY" \
      --pool zh_prospective
    echo "[Step20 freeze 3/4] Transform prospective pairs with the frozen v7 corpus references"
    "$PYTHON_BIN" scripts/step20_build_prospective_features.py \
      --policy "$POLICY" \
      --v7-policy "$V7_POLICY"
    echo "[Step20 freeze 4/4] Score every preregistered model without reading labels"
    "$PYTHON_BIN" scripts/step20_score_prospective_holdout.py \
      --policy "$POLICY" \
      --v7-policy "$V7_POLICY"
    echo "Scores are frozen. Do not evaluate until the one-time unsealing is authorized."
    ;;
  evaluate-once)
    if [[ "${CONFIRM_ONE_TIME_PROSPECTIVE_EVALUATION:-}" != "YES" ]]; then
      echo "ERROR: export CONFIRM_ONE_TIME_PROSPECTIVE_EVALUATION=YES before the irreversible one-time evaluation." >&2
      exit 2
    fi
    echo "[Step20 evaluate] Irreversibly unseal labels and evaluate exactly once"
    "$PYTHON_BIN" scripts/step20_evaluate_prospective_holdout.py \
      --policy "$POLICY"
    ;;
  *)
    echo "Usage: $0 {prepare|freeze-and-score|evaluate-once}" >&2
    exit 2
    ;;
esac
