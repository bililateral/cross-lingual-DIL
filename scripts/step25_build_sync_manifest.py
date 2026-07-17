#!/usr/bin/env python3
"""Build a complete hash-bound synchronization manifest for Step25 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as step24
import step25_common as common


PRODUCERS = [
    "scripts/step25_common.py",
    "scripts/step25_build_template_decontamination.py",
    "scripts/step25_build_decontaminated_style_embeddings.py",
    "scripts/step25_build_pair_features.py",
    "scripts/step25_evaluate_template_decontaminated_authorship.py",
    "scripts/step25_train_occurrence_reliability.py",
    "scripts/step25_build_sync_manifest.py",
    "scripts/run_step25_template_decontaminated_authorship_linux_20260717.sh",
    "scripts/step9_run_few_shot_adaptation.py",
    "scripts/step15_v7_common.py",
    "scripts/step24_common.py",
    "scripts/step24_build_style_embedding_cache.py",
    "scripts/step24_evaluate_content_independent_authorship.py",
    "schema/step15_v7_two_stage_policy.json",
    "schema/step24_content_independent_authorship_policy.json",
]


def expected_paths(policy: dict, step24_policy: dict) -> list[Path]:
    root = common.resolve(policy["outputs_root"])
    paths = []
    for pool_name in step24_policy["pools"]:
        paths.extend(common.template_output_paths(root, pool_name))
        for encoder_key in policy["frozen_style_encoders"]:
            paths.extend(common.embedding_output_paths(root, encoder_key, pool_name))
    for output_name, relative in policy["outputs"].items():
        if output_name == "sync_manifest":
            continue
        paths.append(root / relative)
    unique = sorted(set(paths), key=lambda path: str(path))
    if len(unique) != len(paths):
        raise ValueError("Step25 output policy maps multiple artifacts to the same path")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, step24_policy = common.load_policy(args.policy)
    paths = expected_paths(policy, step24_policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "expected_payload_count": len(paths),
                    "output_root": policy["outputs_root"],
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step25 synchronization payload is incomplete: {missing[0]}")
    records = [
        {
            "path": str(path.relative_to(common.ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": step24.sha256_file(path),
        }
        for path in paths
    ]
    producer_records = []
    for relative in PRODUCERS:
        path = common.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Step25 producer is missing: {path}")
        producer_records.append(
            {
                "path": relative,
                "sha256": step24.sha256_file(path),
            }
        )
    evaluation_path = common.resolve(policy["outputs_root"]) / policy["outputs"][
        "evaluation_summary"
    ]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("publication_promotion_eligible") is not False:
        raise ValueError("Step25 D0 synchronization manifest cannot carry a promotion claim")
    manifest = {
        "step": "step25_sync_manifest",
        "version": policy["version"],
        "boundary": "d0_current_canonical_train",
        "publication_promotion_eligible": False,
        "payload_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "payload_files_sha256": step24.canonical_hash(records),
        "producers": producer_records,
        "producer_files_sha256": step24.canonical_hash(producer_records),
        "policy_path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
        "policy_sha256": step24.sha256_file(policy_path),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = step24.canonical_hash(manifest)
    manifest_path = common.resolve(policy["outputs_root"]) / policy["outputs"][
        "sync_manifest"
    ]
    step24.write_json_immutable(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_count": len(records),
                "total_size_bytes": manifest["total_size_bytes"],
                "manifest": str(manifest_path.relative_to(common.ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
