"""Mechanical ZIP identity/inventory extraction only. Does not import project code."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "upload" / "optuna_plan_review.zip"
DEST = ROOT / "input"
OUT = ROOT / "mechanical"
EXPECTED = {
    "size_bytes": 186342,
    "member_count": 26,
    "sha256": "60a248b5599f96e3a944a530fa6818e31d8bfc90133e74a0d7634fd0b9fd81cd",
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    rows = []
    with zipfile.ZipFile(SOURCE) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        for info in z.infolist():
            rel = Path(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unexpected ZIP path")
            data = z.read(info.filename)
            rows.append({"path": info.filename, "size_bytes": len(data), "sha256": digest(data), "directory": info.is_dir()})
        actual = {"size_bytes": len(raw), "member_count": len(names), "sha256": digest(raw)}
        matches = {key: actual[key] == EXPECTED[key] for key in EXPECTED}
        result = {"source_name": SOURCE.name, "expected": EXPECTED, "actual": actual, "matches": matches, "members": rows}
        (OUT / "input_identity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not all(matches.values()):
            raise ValueError("Input identity does not match submission")
        DEST.mkdir(parents=True, exist_ok=True)
        for info in z.infolist():
            target = DEST / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info.filename))
    print(json.dumps({"actual": actual, "matches": matches, "members": [{"path": r["path"], "size_bytes": r["size_bytes"]} for r in rows]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
