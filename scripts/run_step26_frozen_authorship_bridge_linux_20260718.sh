#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step26_frozen_authorship_bridge_policy.json"

echo "[1/6] Validate Step26 contracts"
"$PYTHON_BIN" -m unittest tests.test_step26_frozen_authorship_bridge_contracts

echo "[2/6] Validate the frozen encoder and source-artifact configuration"
"$PYTHON_BIN" scripts/step26_build_frozen_style_cache.py \
  --policy "$POLICY" \
  --validate-config-only

echo "[3/6] Validate the exact corrected 120/200 evaluation boundary"
"$PYTHON_BIN" scripts/step26_evaluate_frozen_authorship_bridge.py \
  --policy "$POLICY" \
  --validate-config-only

echo "[4/6] Blindly encode evaluation sellers with frozen Step24 authorship models"
"$PYTHON_BIN" scripts/step26_build_frozen_style_cache.py \
  --policy "$POLICY" \
  --device "${STEP26_DEVICE:-auto}"

echo "[5/6] Apply frozen English source scorers and run same-pair statistical gates"
"$PYTHON_BIN" scripts/step26_evaluate_frozen_authorship_bridge.py \
  --policy "$POLICY"

echo "[6/6] Freeze the explicit Step26 input/output manifest"
"$PYTHON_BIN" scripts/step26_build_sync_manifest.py \
  --policy "$POLICY"

echo "Step26 completed. Sync this directory back to Windows:"
echo "reports/step26_frozen_authorship_bridge/v1_20260718/"
