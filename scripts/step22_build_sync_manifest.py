#!/usr/bin/env python3
"""Build the complete content-addressed Step22 synchronization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = ROOT / "schema" / "step22_same_seller_split_policy.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
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
    required_names = (
        "summary",
        "manifest",
        "profiles",
        "pair_labels",
        "pair_lineage",
        "item_lineage",
        "embedding_matrix",
        "embedding_metadata",
        "evaluation_summary",
        "evaluation_predictions",
    )
    required = [output_root / policy["outputs"][name] for name in required_names]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step22 sync manifest is missing required outputs: {missing}")
    manifest_path = output_root / policy["outputs"]["sync_manifest"]
    files = sorted(path for path in output_root.rglob("*") if path.is_file() and path != manifest_path)
    records = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    payload = {
        "step": "step22_full_output_sync_manifest",
        "policy_version": policy["version"],
        "policy_path": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        "policy_sha256": sha256(policy_path),
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("step", "file_count", "total_size_bytes")}, indent=2))


if __name__ == "__main__":
    main()
