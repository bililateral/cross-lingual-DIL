#!/usr/bin/env python3
"""Build the closed hash-bound Step25-v3 return manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as step24
import step25_v3_common as common


PRODUCERS = [
    "scripts/step25_v3_common.py",
    "scripts/step25_v3_build_dual_channel_features.py",
    "scripts/step25_v3_evaluate_copy_aware_fusion.py",
    "scripts/step25_v3_train_operational_identifier_control.py",
    "scripts/step25_v3_build_sync_manifest.py",
    "scripts/run_step25_v3_copy_aware_dual_channel_linux_20260718.sh",
    "scripts/step25_v2_common.py",
    "scripts/step25_v2_evaluate_pair_local_copy.py",
    "scripts/step25_common.py",
    "scripts/step25_evaluate_template_decontaminated_authorship.py",
    "scripts/step24_common.py",
    "scripts/step24_evaluate_content_independent_authorship.py",
    "scripts/step15_v7_common.py",
    "scripts/step9_run_few_shot_adaptation.py",
    "schema/step25_v3_copy_aware_dual_channel_policy.json",
    "schema/step25_v2_pair_local_copy_diagnostic_policy.json",
    "schema/step25_template_decontaminated_authorship_policy.json",
    "schema/step24_content_independent_authorship_policy.json",
    "schema/step15_v7_two_stage_policy.json",
    "tests/test_step25_v3_copy_aware_dual_channel_contracts.py",
    "docs/STEP25_V3_COPY_AWARE_DUAL_CHANNEL_PLAN_20260718.zh.md",
]


def expected_paths(policy: dict) -> list[Path]:
    root = common.resolve(policy["outputs_root"])
    paths = [
        root / value
        for key, value in policy["outputs"].items()
        if key != "sync_manifest"
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("Step25-v3 output policy maps multiple artifacts to one path")
    return sorted(paths, key=lambda path: str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, *_parents = common.load_policy(args.policy)
    paths = expected_paths(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "expected_payload_count": len(paths),
                    "output_root": policy["outputs_root"],
                    "publication_promotion_allowed": False,
                    "step11_or_step17_entry_allowed": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    parent_manifests = common.require_parent_manifests(policy)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step25-v3 return payload is incomplete: {missing[0]}")
    records = [
        {
            "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": step24.sha256_file(path),
        }
        for path in paths
    ]
    producers = []
    for relative in PRODUCERS:
        path = common.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Step25-v3 producer is missing: {path}")
        producers.append({"path": relative, "sha256": step24.sha256_file(path)})
    root = common.resolve(policy["outputs_root"])
    evaluation = json.loads(
        (root / policy["outputs"]["evaluation_summary"]).read_text(encoding="utf-8")
    )
    operational = json.loads(
        (root / policy["outputs"]["operational_summary"]).read_text(encoding="utf-8")
    )
    for payload in (evaluation, operational):
        if payload.get("publication_promotion_eligible") is not False:
            raise ValueError("Step25-v3 return payload carries an invalid publication claim")
        if payload.get("step11_or_step17_entry_allowed") is not False:
            raise ValueError("Step25-v3 return payload improperly unlocks Step11/17")
    if evaluation.get("valid_or_test_rows_read_or_scored") != 0:
        raise ValueError("Step25-v3 evaluation read valid/test")
    manifest = {
        "step": "step25_v3_sync_manifest",
        "version": policy["version"],
        "boundary": policy["boundary"]["name"],
        "hypothesis_informed_retrospective": True,
        "d1_replication_candidate_eligible": bool(
            evaluation["d1_replication_candidate_eligible"]
        ),
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
        "parent_manifests": parent_manifests,
        "payload_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "payload_files_sha256": step24.canonical_hash(records),
        "producers": producers,
        "producer_files_sha256": step24.canonical_hash(producers),
        "policy_path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = step24.canonical_hash(manifest)
    path = root / policy["outputs"]["sync_manifest"]
    step24.write_json_immutable(path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_count": len(records),
                "d1_replication_candidate_eligible": manifest[
                    "d1_replication_candidate_eligible"
                ],
                "publication_promotion_eligible": False,
                "manifest": str(path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
