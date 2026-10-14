#!/usr/bin/env bash
# Reproduces only this review's fixed handwritten checks in a NEW local directory.
# Not a project qualification or formal-data execution command.
set -euo pipefail
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:?Usage: bash reproduce_handwritten.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$DEST" ]]; then echo 'Output must not already exist; preserve earlier evidence.' >&2; exit 2; fi
mkdir -p "$DEST/input" "$DEST/evidence"
python - "$PACKAGE/authorized_input/review_input.zip" "$DEST/input" <<'PY'
import sys, zipfile, hashlib
from pathlib import Path, PurePosixPath
p=Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()=='821415de49e848672d6ae5a025bbe88e63dfb4c4990c86aaf1cc0061b2fb6b6a'
with zipfile.ZipFile(p) as z:
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in z.namelist())
    z.extractall(sys.argv[2])
PY
python "$PACKAGE/evidence/audit_input.py" "$PACKAGE/authorized_input/review_input.zip" "$DEST/input" "$DEST/evidence"
python "$PACKAGE/evidence/verify_source_identity.py" "$DEST/input" "$DEST/evidence"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python "$PACKAGE/evidence/independent_checks.py" "$DEST/input" "$DEST/evidence" > "$DEST/evidence/independent_checks.stdout.txt" 2> "$DEST/evidence/independent_checks.stderr.txt"
