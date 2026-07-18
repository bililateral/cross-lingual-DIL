#!/usr/bin/env python3
"""Freeze the complete run-scoped Step27 Linux-to-Windows sync manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step27_train_residual_models as step27


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step27_english_pretrained_synthetic_adaptation_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path)


def referenced_paths(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "output_paths":
                found.extend(referenced_paths(child))
            elif isinstance(child, (dict, list)):
                found.extend(referenced_paths(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(referenced_paths(child))
    elif isinstance(value, str):
        found.append(value)
    return found


def main() -> None:
    args = parse_args()
    policy_path = step27.resolve(args.policy)
    policy = step27.load_json(policy_path)
    cfg = step27.validate_policy(policy, policy_path)
    root = step27.outputs_root(policy)
    manifest_path = root / "manifests" / "step27_sync_manifest.json"
    if args.validate_config_only:
        print(json.dumps({"status": "pass", "run_id": cfg["run_id"], "output_root": relative(root)}, indent=2))
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Step27 output root is missing: {root}")

    final_audit = (
        root
        / "statistical_audit"
        / "final_diagnostic"
        / "step12_step27_statistical_audit.json"
    )
    valid_audit = (
        root
        / "statistical_audit"
        / "valid_gate"
        / "step12_step27_statistical_audit.json"
    )
    oof_audit = (
        root
        / "statistical_audit"
        / "oof_gate"
        / "step12_step27_statistical_audit.json"
    )
    selected_audit = (
        final_audit
        if final_audit.is_file()
        else valid_audit
        if valid_audit.is_file()
        else oof_audit
    )
    required = {
        root / "synthetic_audit" / "step27_synthetic_data_audit.json": "pass",
        root / "training" / "step27_training_summary.json": "complete",
        selected_audit: "complete",
    }
    loaded_summaries = {}
    for path, expected_status in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Step27 required stage summary is missing: {path}")
        payload = step27.load_json(path)
        if payload.get("status") != expected_status or payload.get("run_id") != cfg["run_id"]:
            raise ValueError(f"Step27 stage summary status/run_id mismatch: {path}")
        loaded_summaries[path] = payload

    training_manifest = step27.load_json(root / "training" / "step27_training_input_manifest.json")
    training_sha = training_manifest.get("manifest_sha256")
    if not training_sha:
        raise ValueError("Step27 training input manifest has no self hash")
    if loaded_summaries[root / "training" / "step27_training_summary.json"].get(
        "input_manifest_sha256"
    ) != training_sha:
        raise ValueError("Step27 training summary is not bound to the training input manifest")
    synthetic_summary = loaded_summaries[
        root / "synthetic_audit" / "step27_synthetic_data_audit.json"
    ]
    training_summary = loaded_summaries[root / "training" / "step27_training_summary.json"]
    if not synthetic_summary.get("input_manifest_sha256"):
        raise ValueError("Step27 synthetic audit has no stage-specific input manifest binding")
    if synthetic_summary.get("pair_feature_bundle_sha256") != training_summary.get(
        "pair_feature_bundle_sha256"
    ):
        raise ValueError("Step27 synthetic audit and training do not bind the same pair-feature bundle")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path and not path.name.startswith(".") and not path.name.endswith(".tmp")
    )
    if not files:
        raise ValueError("Step27 sync manifest found no run-scoped files")
    file_set = {path.resolve() for path in files}
    manifest_closure_records = []
    for candidate in files:
        if candidate.suffix.casefold() != ".json" or "manifest" not in candidate.name.casefold():
            continue
        payload = step27.load_json(candidate)
        if payload.get("identity") is not None and payload.get("identity_sha256") != step27.canonical_hash(
            payload["identity"]
        ):
            raise ValueError(f"Step27 manifest identity hash mismatch: {candidate}")
        if payload.get("manifest_content_sha256") is not None:
            content = dict(payload)
            expected_content_sha = content.pop("manifest_content_sha256")
            if expected_content_sha != step27.canonical_hash(content):
                raise ValueError(f"Step27 manifest content hash mismatch: {candidate}")
        if payload.get("manifest_sha256") is not None:
            content = dict(payload)
            expected_manifest_sha = content.pop("manifest_sha256")
            if expected_manifest_sha != step27.canonical_hash(content):
                raise ValueError(f"Step27 manifest self hash mismatch: {candidate}")
        checked = 0
        for field in ("inputs", "outputs"):
            records = payload.get(field)
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                    raise ValueError(f"Step27 manifest has a malformed {field} record: {candidate}")
                referenced = step27.resolve(record["path"])
                if not referenced.is_file():
                    raise FileNotFoundError(
                        f"Step27 manifest closure is missing {field}: {candidate}->{referenced}"
                    )
                if step27.sha256_file(referenced) != record["sha256"]:
                    raise ValueError(
                        f"Step27 manifest closure hash mismatch: {candidate}->{referenced}"
                    )
                if field == "outputs" and referenced.resolve() not in file_set:
                    raise ValueError(
                        f"Step27 stage output is absent from the sync file universe: {referenced}"
                    )
                checked += 1
        manifest_closure_records.append(
            {"manifest": relative(candidate), "checked_input_output_records": checked}
        )
    for summary_path, summary in loaded_summaries.items():
        for path_value in referenced_paths(summary.get("output_paths") or {}):
            referenced = step27.resolve(path_value)
            if referenced.resolve() not in file_set:
                raise ValueError(f"Step27 summary references an absent/unmanifested output: {summary_path}->{referenced}")

    records = [
        {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": step27.sha256_file(path),
        }
        for path in files
    ]
    policy_record = {
        "path": relative(policy_path),
        "size_bytes": policy_path.stat().st_size,
        "sha256": step27.sha256_file(policy_path),
    }
    code_paths = [
        ROOT / "scripts" / "step27_common.py",
        ROOT / "scripts" / "step27_build_parent_manifest.py",
        ROOT / "scripts" / "step27_generate_train_only_views.py",
        ROOT / "scripts" / "step27_encode_profiles.py",
        ROOT / "scripts" / "step27_build_pair_features.py",
        ROOT / "scripts" / "step27_train_residual_models.py",
        ROOT / "scripts" / "step27_audit_synthetic_data.py",
        ROOT / "scripts" / "step12_step27_statistical_audit.py",
        ROOT / "scripts" / "step27_build_sync_manifest.py",
        ROOT / "scripts" / "run_step27_english_pretrained_synthetic_linux_20260718.sh",
        ROOT / "tests" / "test_step27_english_pretrained_synthetic_contracts.py",
        ROOT / "scripts" / "step15_build_v7_clean_embedding_cache.py",
        ROOT / "scripts" / "step24_common.py",
        ROOT / "scripts" / "step7_build_semantic_pair_features.py",
    ]
    code_records = []
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Step27 required implementation file is missing: {path}")
        code_records.append(
            {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": step27.sha256_file(path)}
        )
    core = {
        "status": "complete",
        "run_id": cfg["run_id"],
        "policy_version": policy.get("version"),
        "git_commit": step27.git_commit(),
        "output_root": relative(root),
        "training_input_manifest_sha256": training_sha,
        "policy": policy_record,
        "implementation_files": code_records,
        "files": records,
        "manifest_closure": manifest_closure_records,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "sync_instruction": "Sync every path in files; do not glob older Step27/report directories.",
    }
    manifest = {**core, "manifest_sha256": step27.canonical_hash(core)}
    if args.validate_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Step27 sync manifest is missing: {manifest_path}")
        existing = step27.load_json(manifest_path)
        if existing != manifest:
            raise ValueError("Step27 sync manifest no longer matches the run-scoped artifacts")
        print(json.dumps({"status": "pass", "manifest": relative(manifest_path)}, indent=2))
        return
    step27.write_json_immutable(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": relative(manifest_path),
                "file_count": len(records),
                "total_size_bytes": core["total_size_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
