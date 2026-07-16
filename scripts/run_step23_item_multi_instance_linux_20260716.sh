#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step23_item_multi_instance_policy.json"
OUTPUT_ROOT="reports/step23_item_multi_instance/v1_20260716"

echo "[1/8] Validate Step22 freeze and Step23 contracts"
"$PYTHON_BIN" -m unittest \
  tests.test_step22_same_seller_split_contracts \
  tests.test_step23_item_multi_instance_contracts
"$PYTHON_BIN" scripts/step23_build_item_text_cache.py \
  --policy "$POLICY" \
  --validate-config-only

echo "[2/8] Complete the frozen Step22 post-hoc grouped-bootstrap audit"
"$PYTHON_BIN" scripts/step22_grouped_bootstrap_audit.py \
  --root reports/step22_same_seller_split/v1_20260716 \
  --resamples 5000 \
  --seed 20260716

echo "[3/8] Build the label-blind identifier-redacted real train-item cache"
"$PYTHON_BIN" scripts/step23_build_item_text_cache.py \
  --policy "$POLICY"

echo "[4/8] Encode selected real train items with frozen Multilingual-E5"
"$PYTHON_BIN" scripts/step23_encode_selected_items.py \
  --policy "$POLICY" \
  --device "${STEP23_DEVICE:-cuda}"

echo "[5/8] Build symmetric item-to-item distribution features"
"$PYTHON_BIN" scripts/step23_build_multi_instance_features.py \
  --policy "$POLICY"

echo "[6/8] Run source-only and EN-plus-ZH seller-component grouped OOF controls"
"$PYTHON_BIN" scripts/step23_evaluate_item_multi_instance.py \
  --policy "$POLICY"

echo "[7/8] Verify the no-valid/test and no-synthetic scope"
"$PYTHON_BIN" -c 'import json,pathlib; root=pathlib.Path("reports/step23_item_multi_instance/v1_20260716"); item=json.loads((root/"item_selection_summary.json").read_text(encoding="utf-8")); result=json.loads((root/"step23_grouped_oof_evaluation.json").read_text(encoding="utf-8")); assert item["valid_test_items_encoded"] is False and item["synthetic_item_count"] == 0; assert result["valid_or_test_scores_used"] is False and result["publication_holdout_untouched"] is True; print({"status":"scope_verified","selected_items":item["selected_item_count"],"promotion_eligible":result["promotion"]["eligible"]})'

echo "[8/8] Build the complete content-addressed synchronization manifest"
"$PYTHON_BIN" scripts/step23_build_sync_manifest.py \
  --policy "$POLICY"

echo "Step23 completed. Outputs: $OUTPUT_ROOT/"
