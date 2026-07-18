#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step25_v3_copy_aware_dual_channel_policy.json"

echo "[1/7] Validate Step25-v3 contracts"
"$PYTHON_BIN" -m unittest tests.test_step25_v3_copy_aware_dual_channel_contracts

echo "[2/7] Validate all Step25-v3 entry points without numerical execution"
"$PYTHON_BIN" scripts/step25_v3_build_dual_channel_features.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_evaluate_copy_aware_fusion.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_train_operational_identifier_control.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_build_sync_manifest.py --policy "$POLICY" --validate-config-only

echo "[3/7] Join immutable Step24/Step25-v1/v2 train-only feature channels"
"$PYTHON_BIN" scripts/step25_v3_build_dual_channel_features.py --policy "$POLICY"

echo "[4/7] Fit C0-C3 source-only and component-grouped OOF models"
"$PYTHON_BIN" scripts/step25_v3_evaluate_copy_aware_fusion.py --policy "$POLICY"

echo "[5/7] Train the separate English-only operational identifier control"
"$PYTHON_BIN" scripts/step25_v3_train_operational_identifier_control.py --policy "$POLICY"

echo "[6/7] Build the closed synchronization manifest"
"$PYTHON_BIN" scripts/step25_v3_build_sync_manifest.py --policy "$POLICY"

echo "[7/7] Print the bounded D0 conclusion"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("reports/step25_template_decontaminated_authorship/v3_copy_aware_dual_channel_20260718/step25_v3_evaluation_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps({
    "status": summary["status"],
    "d1_replication_candidate_eligible": summary["d1_replication_candidate_eligible"],
    "publication_promotion_eligible": summary["publication_promotion_eligible"],
    "step11_or_step17_entry_allowed": summary["step11_or_step17_entry_allowed"],
    "target_grouped_oof_C2_minus_C0_ap": summary["key_deltas"]["target_grouped_oof_C2_minus_C0_average_precision"],
    "gate_results": summary["d0_to_d1_gate_results"],
    "summary": str(path),
}, indent=2))
PY

echo "Step25-v3 completed. Return the entire isolated v3 directory."
