#!/usr/bin/env python3
"""Build a complete content-addressed manifest for Step23 Linux outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step23_item_multi_instance_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    output_root = resolve(policy["outputs_root"])
    sync_name = policy["outputs"]["sync_manifest"]
    sync_path = output_root / sync_name
    expected = {
        value for key, value in policy["outputs"].items() if key != "sync_manifest"
    }
    missing = sorted(name for name in expected if not (output_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Step23 synchronization manifest is missing outputs: {missing}")
    files = []
    for path in sorted(output_root.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path == sync_path:
            continue
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    actual_names = {Path(row["path"]).name for row in files}
    unexpected = sorted(actual_names - expected)
    if unexpected:
        raise ValueError(f"Step23 output root contains undeclared files: {unexpected}")
    payload = {
        "step": "step23_full_output_sync_manifest",
        "policy_version": policy["version"],
        "policy_path": policy_path.relative_to(ROOT).as_posix(),
        "policy_sha256": sha256_file(policy_path),
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if sync_path.exists() and sync_path.read_text(encoding="utf-8") != rendered:
        raise ValueError("Refusing to overwrite a different Step23 synchronization manifest")
    sync_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "file_count": len(files),
        "total_size_bytes": payload["total_size_bytes"],
        "manifest": sync_path.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
