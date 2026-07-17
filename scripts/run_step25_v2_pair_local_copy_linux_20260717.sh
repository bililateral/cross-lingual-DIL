#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step25_v2_pair_local_copy_diagnostic_policy.json"
STEP24_ROOT="reports/step24_content_independent_authorship/v1_20260717"
STEP25_V1_ROOT="reports/step25_template_decontaminated_authorship/v1_20260717"

echo "[1/7] Validate Step25-v2 policy, entry points and pure contracts"
"$PYTHON_BIN" scripts/step25_v2_build_pair_local_texts.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v2_encode_pair_local_style.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v2_build_pair_features.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v2_evaluate_pair_local_copy.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v2_build_sync_manifest.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" -m unittest tests.test_step25_v2_pair_local_copy_contracts

echo "[2/7] Verify frozen Step24/Step25-v1 inputs and Linux encoder directories"
test -f "$STEP24_ROOT/pair_features.en_content_train_pool.csv"
test -f "$STEP24_ROOT/pair_features.zh_target_strict.csv"
test -f "$STEP25_V1_ROOT/pair_features.en_content_train_pool.csv"
test -f "$STEP25_V1_ROOT/pair_features.zh_target_strict.csv"
test -f "$STEP25_V1_ROOT/step25_evaluation_summary.json"
test -f models/step24/authorship/multilingual_style_representation/config.json
test -f models/step24/authorship/multilingual_style_representation/step24_model_provenance.json
test -f models/step24/authorship/mstyledistance/config.json
test -f models/step24/authorship/mstyledistance/step24_model_provenance.json
"$PYTHON_BIN" -c "import sentence_transformers, torch; print({'sentence_transformers': sentence_transformers.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()})"

echo "[3/7] Detect pair-local copied spans in identifier-redacted canonical-train text"
"$PYTHON_BIN" scripts/step25_v2_build_pair_local_texts.py --policy "$POLICY"

echo "[4/7] Encode pair-side cleaned text with the frozen Step24 authorship encoders"
"$PYTHON_BIN" scripts/step25_v2_encode_pair_local_style.py \
  --policy "$POLICY" \
  --device auto

echo "[5/7] Build P0-P3 style representations without zero-valued missingness"
"$PYTHON_BIN" scripts/step25_v2_build_pair_features.py --policy "$POLICY"

echo "[6/7] Run source-only, English OOF, target grouped-OOF and grouped bootstrap diagnostics"
"$PYTHON_BIN" scripts/step25_v2_evaluate_pair_local_copy.py --policy "$POLICY"

echo "[7/7] Build the closed hash-bound Step25-v2 return manifest"
"$PYTHON_BIN" scripts/step25_v2_build_sync_manifest.py --policy "$POLICY"

"$PYTHON_BIN" -c 'import json; p="reports/step25_template_decontaminated_authorship/v2_pair_local_diagnostic_20260717/step25_v2_evaluation_summary.json"; d=json.load(open(p, encoding="utf-8")); print({"mechanism_hypothesis_supported": d["mechanism_hypothesis_supported"], "d1_candidate_eligible": d["d1_candidate_eligible"], "publication_promotion_eligible": d["publication_promotion_eligible"], "key_deltas": d["key_deltas"], "failed_mechanism_gates": [k for k,v in d["mechanism_gate_results"].items() if not v]})'

echo "Step25-v2 completed. Return the entire reports/step25_template_decontaminated_authorship/v2_pair_local_diagnostic_20260717 directory."
