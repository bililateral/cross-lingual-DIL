#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/2] Validate frozen Step22 audit contracts"
"$PYTHON_BIN" -m unittest tests.test_step22_same_seller_split_contracts

echo "[2/2] Run the post-hoc component-grouped uncertainty audit"
"$PYTHON_BIN" scripts/step22_grouped_bootstrap_audit.py \
  --root reports/step22_same_seller_split/v1_20260716 \
  --resamples 5000 \
  --seed 20260716

echo "Step22 negative audit completed. Promotion remains blocked."
