from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "schema" / "step5_v2_milestone_snapshot_policy.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    policy = load_json(POLICY_PATH)
    destination_root = ROOT / policy["destination_dir"]
    destination_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for relative_path in policy["files"]:
        source_path = ROOT / relative_path
        if not source_path.exists():
            raise SystemExit(f"Milestone snapshot source file not found: {relative_path}")
        dest_path = destination_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied.append(
            {
                "relative_path": relative_path,
                "source_size_bytes": source_path.stat().st_size,
                "source_sha256": sha256_file(source_path),
                "destination_path": str(dest_path.relative_to(ROOT)),
            }
        )

    manifest = {
        "snapshot_version": policy["snapshot_version"],
        "description": policy.get("description", ""),
        "destination_dir": policy["destination_dir"],
        "file_count": len(copied),
        "files": copied,
    }

    write_json(destination_root / "manifest.json", manifest)
    write_json(ROOT / policy["summary_output"], manifest)

    print(f"Created milestone snapshot: {destination_root}")
    print(f"file_count={len(copied)}")


if __name__ == "__main__":
    main()
