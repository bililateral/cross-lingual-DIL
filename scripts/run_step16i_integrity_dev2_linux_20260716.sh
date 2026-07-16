#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
INTEGRITY_POLICY="schema/step16i_data_integrity_policy.json"
DEV2_POLICY="schema/step16i_retrospective_dev2_policy.json"
INTEGRITY_RUN_ID="${STEP16I_RUN_ID:-step16i_integrity_20260716_v1}"
INTEGRITY_ROOT="reports/step16i_data_integrity/${INTEGRITY_RUN_ID}"
READINESS_ROOT="reports/step16_v8_validation_refreeze/readiness_expansion_v3_reprofix_20260716_112833_31791"
READINESS_ASSIGNMENT="${READINESS_ROOT}/representative_validation_assignments.v8_readiness.csv"
EXCLUSION_MANIFEST="${INTEGRITY_ROOT}/permanent_exclusion_manifest.csv"

if [[ ! -f "$READINESS_ASSIGNMENT" ]]; then
  echo "ERROR: Missing the latest V8 readiness assignment: $READINESS_ASSIGNMENT" >&2
  echo "Sync the complete readiness root from the Linux experiment archive before Step16I." >&2
  exit 2
fi

echo "[1/4] Validate Step16I contracts"
"$PYTHON_BIN" -m unittest tests.test_step16i_data_integrity_contracts

echo "[2/4] Recompute seller components and freeze permanent exclusions"
"$PYTHON_BIN" scripts/step16i_audit_data_integrity.py \
  --policy "$INTEGRITY_POLICY" \
  --run-id "$INTEGRITY_RUN_ID" \
  --v8-readiness-assignment "$READINESS_ASSIGNMENT"

echo "[3/4] Enforce integrity and V8-readiness gates"
"$PYTHON_BIN" - "$INTEGRITY_ROOT/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)

if summary.get("status") == "fail":
    raise SystemExit("Step16I integrity audit failed; retrospective dev2 preparation is blocked")
for dataset, record in summary.get("datasets", {}).items():
    if record.get("leakage", {}).get("detected") is not False:
        raise SystemExit(f"Step16I detected split leakage in {dataset}")
readiness = summary.get("v8_readiness_assignment_check", {})
if readiness.get("passed") is not True or readiness.get("status") not in {"pass", "warning"}:
    raise SystemExit(
        "The latest V8 readiness component partition is unsafe; dev2 preparation is blocked"
    )
if readiness.get("status") == "warning":
    print(
        "WARNING: V8 persisted components conservatively merge disconnected seller subgraphs; "
        "this reduces effective component count but does not create cross-split leakage."
    )
print("Step16I integrity gates: PASS")
PY

echo "[4/4] Prepare score-blind retrospective zh_dev2 review queues"
"$PYTHON_BIN" scripts/step16i_prepare_retrospective_dev2.py \
  --policy "$DEV2_POLICY" \
  --permanent-exclusion-manifest "$EXCLUSION_MANIFEST"

echo "Step16I preparation completed."
echo "Integrity summary: ${INTEGRITY_ROOT}/summary.json"
echo "Reviewer queues: reports/step16i_retrospective_dev2/preparation_v1_20260716/"
echo "Do not give blind_mapping.csv to either reviewer. No labels have been created."
