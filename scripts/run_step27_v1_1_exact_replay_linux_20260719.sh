#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step27_v1_1_exact_replay_policy.json"
OUTPUT_ROOT="reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719"

echo "[1/13] Validate Step27 contracts"
"$PYTHON_BIN" -m unittest tests.test_step27_english_pretrained_synthetic_contracts

echo "[2/13] Validate every Step27 runtime configuration without numerical execution"
for script in \
  step27_build_parent_manifest.py \
  step27_generate_train_only_views.py \
  step27_encode_profiles.py \
  step27_build_pair_features.py \
  step27_audit_synthetic_data.py \
  step27_train_residual_models.py \
  step12_step27_statistical_audit.py \
  step27_build_sync_manifest.py; do
  "$PYTHON_BIN" "scripts/$script" --policy "$POLICY" --validate-config-only
done

echo "[2b/13] Verify the local E5 directory against the frozen v7 model fingerprint"
"$PYTHON_BIN" scripts/step27_encode_profiles.py \
  --policy "$POLICY" \
  --validate-model-contract-only

echo "[3/13] Freeze the canonical 573/120/200 boundary, component folds, and matched parents"
"$PYTHON_BIN" scripts/step27_build_parent_manifest.py --policy "$POLICY"

echo "[4/13] Generate primary and physically isolated silver-sensitivity views for ten seeds"
"$PYTHON_BIN" scripts/step27_generate_train_only_views.py --policy "$POLICY"

echo "[5/13] Replay real v7 embeddings exactly and encode synthetic views with frozen Multilingual-E5"
"$PYTHON_BIN" scripts/step27_encode_profiles.py \
  --policy "$POLICY" \
  --device "${STEP27_DEVICE:-auto}"

echo "[6/13] Recompute every real, synthetic, and duplication pair feature"
"$PYTHON_BIN" scripts/step27_build_pair_features.py --policy "$POLICY"

echo "[7/13] Enforce lineage, leakage, shortcut, and effective-weight audits"
"$PYTHON_BIN" scripts/step27_audit_synthetic_data.py --policy "$POLICY"

echo "[8/13] Score S0 and train M0/M1/M2 plus diagnostics; score train OOF only"
"$PYTHON_BIN" scripts/step27_train_residual_models.py --policy "$POLICY"

echo "[9/13] Apply the preregistered train-OOF gate before opening valid"
"$PYTHON_BIN" scripts/step12_step27_statistical_audit.py \
  --policy "$POLICY" \
  --mode oof_gate

OOF_ELIGIBLE=$(
  "$PYTHON_BIN" -c '
import json
from pathlib import Path
path = Path("reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/statistical_audit/oof_gate/step12_step27_statistical_audit.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(str(bool(payload["promotion"]["eligible_for_valid"])).lower())
'
)

OOF_TECHNICAL_PASS=$(
  "$PYTHON_BIN" -c '
import json
from pathlib import Path
path = Path("reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/statistical_audit/oof_gate/step12_step27_statistical_audit.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(str(bool(payload["promotion"]["technical_oof_gate_pass"])).lower())
'
)

echo "Step27 v1.1 technical OOF gate pass: $OOF_TECHNICAL_PASS"
echo "Step27 OOF-gate eligible for one valid opening: $OOF_ELIGIBLE"
if [[ "$OOF_ELIGIBLE" != "false" ]]; then
  echo "ERROR: v1.1 is a post-hoc source-contract repair and must never authorize existing valid."
  exit 1
fi

echo "[10/13] Existing valid is intentionally unopened by the v1.1 repair runner"
echo "[11/13] Existing valid gate is intentionally unavailable"
echo "[12/13] Retrospective internal test is intentionally unopened"

echo "[13/13] Freeze the explicit run-scoped sync manifest"
"$PYTHON_BIN" scripts/step27_build_sync_manifest.py --policy "$POLICY"

echo "Step27 completed. Sync this exact directory back to Windows:"
echo "$OUTPUT_ROOT/"
