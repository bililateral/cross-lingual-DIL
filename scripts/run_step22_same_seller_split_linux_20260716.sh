#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step22_same_seller_split_policy.json"
OUTPUT_ROOT="reports/step22_same_seller_split/v1_20260716"

echo "[1/6] Validate Step22 contracts and frozen input availability"
"$PYTHON_BIN" -m unittest tests.test_step22_same_seller_split_contracts
"$PYTHON_BIN" -c 'import json,pathlib; p=json.loads(pathlib.Path("schema/step22_same_seller_split_policy.json").read_text(encoding="utf-8")); missing=[v for v in p["inputs"].values() if not pathlib.Path(v).is_file()]; assert not missing, f"Missing Step22 inputs: {missing}"; print({"status":"inputs_present","count":len(p["inputs"])})'

echo "[2/6] Build item-disjoint same-seller positives and reviewed-negative item views"
"$PYTHON_BIN" scripts/step22_build_same_seller_split_augmentation.py \
  --policy "$POLICY"

echo "[3/6] Encode pseudo profiles with frozen identifier-redacted Multilingual-E5"
"$PYTHON_BIN" scripts/step22_encode_pseudo_profiles.py \
  --policy "$POLICY" \
  --device "${STEP22_DEVICE:-cuda}"

echo "[4/6] Run canonical Chinese-train seller-component grouped OOF controls"
"$PYTHON_BIN" scripts/step22_evaluate_same_seller_split.py \
  --policy "$POLICY"

echo "[5/6] Enforce train-only and holdout-untouched scope"
"$PYTHON_BIN" -c 'import csv,json,pathlib; root=pathlib.Path("reports/step22_same_seller_split/v1_20260716"); rows=list(csv.DictReader((root/"pseudo_pair_labels.csv").open(encoding="utf-8-sig",newline=""))); summary=json.loads((root/"step22_grouped_oof_evaluation.json").read_text(encoding="utf-8")); assert rows and all(r["split_name"]=="train" and r["benchmark_eligible"]=="0" for r in rows); assert summary["publication_holdout_untouched"] is True and summary["valid_or_test_scores_used"] is False; print({"status":"train_only_verified","pairs":len(rows),"promotion_eligible":summary["comparisons"]["promotion_eligible"]})'

echo "[6/6] Build complete content-addressed synchronization manifest"
"$PYTHON_BIN" scripts/step22_build_sync_manifest.py \
  --policy "$POLICY"

echo "Step22 completed. Outputs: $OUTPUT_ROOT/"
