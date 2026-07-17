#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step25_template_decontaminated_authorship_policy.json"
STEP24_ROOT="reports/step24_content_independent_authorship/v1_20260717"

echo "[1/8] Validate Step25 policy, config-only entry points and pure contracts"
"$PYTHON_BIN" scripts/step25_build_template_decontamination.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_build_decontaminated_style_embeddings.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_build_pair_features.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_evaluate_template_decontaminated_authorship.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_train_occurrence_reliability.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_build_sync_manifest.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" -m unittest tests.test_step25_template_decontaminated_authorship_contracts

echo "[2/8] Verify frozen Step24 raw caches and the two existing Linux model directories"
test -f "$STEP24_ROOT/embeddings/pcm_multilingual_authorship.en_content_train_pool.json"
test -f "$STEP24_ROOT/embeddings/pcm_multilingual_authorship.zh_target_strict.json"
test -f "$STEP24_ROOT/pair_features.en_content_train_pool.csv"
test -f "$STEP24_ROOT/pair_features.zh_target_strict.csv"
test -f models/step24/authorship/multilingual_style_representation/config.json
test -f models/step24/authorship/multilingual_style_representation/step24_model_provenance.json
test -f models/step24/authorship/mstyledistance/config.json
test -f models/step24/authorship/mstyledistance/step24_model_provenance.json
"$PYTHON_BIN" -c "import sentence_transformers, torch; print({'sentence_transformers': sentence_transformers.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()})"

echo "[3/8] Fit label-free component-cross-fitted boilerplate catalogs and remove repeated spans"
"$PYTHON_BIN" scripts/step25_build_template_decontamination.py --policy "$POLICY"

echo "[4/8] Encode the decontaminated train-only seller texts with the frozen Step24 models"
"$PYTHON_BIN" scripts/step25_build_decontaminated_style_embeddings.py \
  --policy "$POLICY" \
  --device auto

echo "[5/8] Build the fixed raw/decontaminated/delta pair feature table"
"$PYTHON_BIN" scripts/step25_build_pair_features.py --policy "$POLICY"

echo "[6/8] Run source-only, English OOF and target grouped-OOF matched comparisons"
"$PYTHON_BIN" scripts/step25_evaluate_template_decontaminated_authorship.py --policy "$POLICY"

echo "[7/8] Train and apply the independent occurrence-level reliability expert"
"$PYTHON_BIN" scripts/step25_train_occurrence_reliability.py --policy "$POLICY"

echo "[8/8] Build the complete hash-bound Step25 return manifest"
"$PYTHON_BIN" scripts/step25_build_sync_manifest.py --policy "$POLICY"

"$PYTHON_BIN" -c 'import json; p="reports/step25_template_decontaminated_authorship/v1_20260717/step25_evaluation_summary.json"; d=json.load(open(p, encoding="utf-8")); print({"d1_candidate_eligible": d["d1_candidate_eligible"], "publication_promotion_eligible": d["publication_promotion_eligible"], "key_deltas": d["key_deltas"], "failed_d0_gates": [k for k,v in d["d0_continuation_gate_results"].items() if not v]})'

echo "Step25 D0 completed. Return the entire reports/step25_template_decontaminated_authorship/v1_20260717 directory."
