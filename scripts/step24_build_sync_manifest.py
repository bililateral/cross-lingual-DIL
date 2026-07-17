#!/usr/bin/env python3
"""Build a hash manifest for the complete Step24 Linux return bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path = common.resolve(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    common.validate_policy(policy)
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "numerical_execution_performed": False}, indent=2))
        return
    output_root = common.resolve(policy["outputs_root"])
    if not output_root.is_dir():
        raise FileNotFoundError(f"Step24 output root is missing: {output_root}")
    manifest_path = output_root / policy["outputs"]["sync_manifest"]
    required = [
        policy["outputs"]["clean_text_manifest"],
        policy["outputs"]["embedding_manifest"],
        policy["outputs"]["pair_features_en"],
        policy["outputs"]["pair_features_zh"],
        policy["outputs"]["pair_feature_summary"],
        policy["outputs"]["oof_predictions"],
        policy["outputs"]["evaluation_summary"],
        policy["outputs"]["model_artifacts"],
    ]
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Step24 return bundle is incomplete; first missing={missing[0]}")
    files = []
    for path in sorted(candidate for candidate in output_root.rglob("*") if candidate.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
        )
    payload = {
        "step": "step24_sync_manifest",
        "version": policy["version"],
        "output_root": str(output_root.relative_to(common.ROOT)).replace("\\", "/"),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "bundle_files_sha256": common.canonical_hash(files),
        "policy_sha256": common.sha256_file(policy_path),
        "producer_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    payload["manifest_sha256"] = common.canonical_hash(payload)
    common.write_json_immutable(manifest_path, payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "manifest": str(manifest_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
