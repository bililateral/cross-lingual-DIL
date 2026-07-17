#!/usr/bin/env python3
"""Build the closed hash-bound return manifest for Step25-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as step24
import step25_v2_common as common


PRODUCERS = [
    "scripts/step25_v2_common.py",
    "scripts/step25_v2_build_pair_local_texts.py",
    "scripts/step25_v2_encode_pair_local_style.py",
    "scripts/step25_v2_build_pair_features.py",
    "scripts/step25_v2_evaluate_pair_local_copy.py",
    "scripts/step25_v2_build_sync_manifest.py",
    "scripts/run_step25_v2_pair_local_copy_linux_20260717.sh",
    "scripts/step25_evaluate_template_decontaminated_authorship.py",
    "scripts/step25_common.py",
    "scripts/step24_evaluate_content_independent_authorship.py",
    "scripts/step24_common.py",
    "scripts/step15_v7_common.py",
    "scripts/step9_run_few_shot_adaptation.py",
    "schema/step25_v2_pair_local_copy_diagnostic_policy.json",
    "schema/step25_template_decontaminated_authorship_policy.json",
    "schema/step24_content_independent_authorship_policy.json",
    "schema/step15_v7_two_stage_policy.json",
]


def expected_paths(policy: dict) -> list[Path]:
    root = common.resolve(policy["outputs_root"])
    output_keys = (
        "pair_local_texts_en",
        "pair_local_texts_zh",
        "detector_summary",
        "embedding_manifest",
        "pair_features_en",
        "pair_features_zh",
        "pair_feature_summary",
        "predictions_en",
        "predictions_zh",
        "model_artifacts",
        "evaluation_summary",
    )
    paths = [root / policy["outputs"][key] for key in output_keys]
    for pool_name in ("en_content_train_pool", "zh_target_strict"):
        for encoder_key in policy["frozen_style_encoders"]:
            matrix_path, metadata_path = common.embedding_paths(
                policy, encoder_key, pool_name
            )
            paths.extend((matrix_path, metadata_path))
    unique = sorted(set(paths), key=lambda path: str(path))
    if len(unique) != len(paths):
        raise ValueError("Step25-v2 output policy maps multiple artifacts to one path")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, _step24_policy, _step25_v1_policy = common.load_policy(args.policy)
    paths = expected_paths(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "expected_payload_count": len(paths),
                    "output_root": policy["outputs_root"],
                    "publication_promotion_allowed": False,
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Step25-v2 synchronization payload is incomplete: {missing[0]}"
        )
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
            raise FileNotFoundError(f"Step25-v2 producer is missing: {path}")
        producer_records.append(
            {"path": relative, "sha256": step24.sha256_file(path)}
        )
    evaluation_path = common.resolve(policy["outputs_root"]) / policy["outputs"][
        "evaluation_summary"
    ]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    hard_false = (
        "d1_candidate_eligible",
        "publication_promotion_eligible",
        "step11_or_step17_entry_allowed",
    )
    if any(evaluation.get(key) is not False for key in hard_false):
        raise ValueError("Step25-v2 return manifest cannot carry a promotion claim")
    manifest = {
        "step": "step25_v2_sync_manifest",
        "version": policy["version"],
        "boundary": policy["boundary"]["name"],
        "mechanism_diagnostic_only": True,
        "d1_candidate_eligible": False,
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
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
