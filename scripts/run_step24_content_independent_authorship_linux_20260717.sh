#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step24_content_independent_authorship_policy.json"

echo "[1/6] Validate Step24 policy and pure contracts"
"$PYTHON_BIN" scripts/step24_build_style_embedding_cache.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step24_build_pair_features.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step24_evaluate_content_independent_authorship.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step24_build_sync_manifest.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" -m unittest tests.test_step24_content_independent_authorship_contracts

echo "[2/6] Verify the two frozen local model directories"
test -f models/step24/authorship/multilingual_style_representation/config.json
test -f models/step24/authorship/multilingual_style_representation/step24_model_provenance.json
test -f models/step24/authorship/mstyledistance/config.json
test -f models/step24/authorship/mstyledistance/step24_model_provenance.json
"$PYTHON_BIN" -c "import sentence_transformers, torch; print({'sentence_transformers': sentence_transformers.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()})"

echo "[3/6] Replay v7 identifier redaction and encode canonical-train sellers"
"$PYTHON_BIN" scripts/step24_build_style_embedding_cache.py \
  --policy "$POLICY" \
  --device auto

echo "[4/6] Build the three fixed train-only pair cosine features"
"$PYTHON_BIN" scripts/step24_build_pair_features.py \
  --policy "$POLICY"

echo "[5/6] Run source-only and seller-component grouped-OOF comparisons"
"$PYTHON_BIN" scripts/step24_evaluate_content_independent_authorship.py \
  --policy "$POLICY"

echo "[6/6] Build the complete hash-bound return manifest"
"$PYTHON_BIN" scripts/step24_build_sync_manifest.py \
  --policy "$POLICY"

"$PYTHON_BIN" -c 'import json; p="reports/step24_content_independent_authorship/v1_20260717/step24_grouped_oof_evaluation.json"; d=json.load(open(p, encoding="utf-8")); print({"promotion_eligible": d["promotion_eligible"], "key_deltas": d["key_deltas"], "failed_gates": [k for k,v in d["gate_results"].items() if not v]})'

echo "Step24 completed. Return the entire reports/step24_content_independent_authorship/v1_20260717 directory."
