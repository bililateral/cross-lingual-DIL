#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step21_synthetic_train_only_policy.json"

echo "[1/6] Validate Step21 train-only synthetic-data contracts"
"$PYTHON_BIN" -m unittest tests.test_step21_synthetic_train_only_contracts

echo "[2/6] Generate isolated primary and silver-sensitivity synthetic tracks"
"$PYTHON_BIN" scripts/step21_build_synthetic_zh_train.py \
  --policy "$POLICY"

echo "[3/6] Encode identifier-redacted synthetic profiles with frozen Multilingual-E5"
"$PYTHON_BIN" scripts/step21_encode_synthetic_profiles.py \
  --policy "$POLICY" \
  --device "${STEP21_DEVICE:-cuda}"

echo "[4/6] Run five-fold seller-component grouped train-OOF controls"
"$PYTHON_BIN" scripts/step21_evaluate_synthetic_augmentation.py \
  --policy "$POLICY" \
  --folds 5 \
  --seed 20260716

echo "[5/6] Verify that no synthetic row was promoted to benchmark data"
"$PYTHON_BIN" -c 'import json, pathlib; p=pathlib.Path("reports/step21_synthetic_train_only/v1_20260716/step21_synthetic_augmentation_evaluation_summary.json"); d=json.loads(p.read_text(encoding="utf-8")); assert d["publication_holdout_untouched"] is True; print(json.dumps({"status": d["status"], "tracks": list(d["tracks"])}, ensure_ascii=False, indent=2))'

echo "[6/6] Build the complete content-addressed sync manifest"
"$PYTHON_BIN" scripts/step21_build_sync_manifest.py \
  --policy "$POLICY"

echo "Step21 completed. Outputs: reports/step21_synthetic_train_only/v1_20260716/"
