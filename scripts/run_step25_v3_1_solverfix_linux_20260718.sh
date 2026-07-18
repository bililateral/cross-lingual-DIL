#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY="schema/step25_v3_1_solver_convergence_policy.json"

echo "[1/7] Validate Step25-v3.1 solver-repair contracts"
"$PYTHON_BIN" -m unittest tests.test_step25_v3_1_solver_convergence_contracts

echo "[2/7] Validate every v3.1 entry point without numerical execution"
"$PYTHON_BIN" scripts/step25_v3_1_build_dual_channel_features.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_1_evaluate_copy_aware_fusion.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_1_train_operational_identifier_control.py --policy "$POLICY" --validate-config-only
"$PYTHON_BIN" scripts/step25_v3_1_build_sync_manifest.py --policy "$POLICY" --validate-config-only

echo "[3/7] Replay the frozen v3 feature join into an isolated v3.1 root"
"$PYTHON_BIN" scripts/step25_v3_1_build_dual_channel_features.py --policy "$POLICY"

echo "[4/7] Refit the unchanged C0-C3 matrix with KKT-only convergence"
"$PYTHON_BIN" scripts/step25_v3_1_evaluate_copy_aware_fusion.py --policy "$POLICY"

echo "[5/7] Replay the unchanged English-only operational identifier control"
"$PYTHON_BIN" scripts/step25_v3_1_train_operational_identifier_control.py --policy "$POLICY"

echo "[6/7] Build the closed manifest and reject every non-KKT artifact"
"$PYTHON_BIN" scripts/step25_v3_1_build_sync_manifest.py --policy "$POLICY"

echo "[7/7] Print the bounded repaired conclusion"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path("reports/step25_template_decontaminated_authorship/v3_1_solverfix_20260718")
summary = json.loads((root / "step25_v3_1_evaluation_summary.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "step25_v3_1_sync_manifest.json").read_text(encoding="utf-8"))
print(json.dumps({
    "status": summary["status"],
    "repair_scope": manifest["repair_scope"],
    "solver_audit": manifest["solver_audit"],
    "feature_parity_audit": manifest["feature_parity_audit"],
    "d1_replication_candidate_eligible": summary["d1_replication_candidate_eligible"],
    "publication_promotion_eligible": summary["publication_promotion_eligible"],
    "step11_or_step17_entry_allowed": summary["step11_or_step17_entry_allowed"],
    "target_grouped_oof_C2_minus_C0_ap": summary["key_deltas"]["target_grouped_oof_C2_minus_C0_average_precision"],
    "gate_results": summary["d0_to_d1_gate_results"],
    "output_root": str(root),
}, indent=2))
PY

echo "Step25-v3.1 completed. Return the entire isolated v3_1_solverfix_20260718 directory."
