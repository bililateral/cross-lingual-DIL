#!/usr/bin/env python3
"""Create a content-addressed sync manifest for one completed Step15-v8 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    _, policy, v7_policy = common.load_policy(args.policy)
    common.validate_policy_contract(policy, v7_policy)
    runtime_chain = common.verify_readiness_runtime_chain(policy, v7_policy)
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    required = [
        root / "clean_semantics" / "clean_semantics_manifest.json",
        root / "bridge_audit" / "step15_v8_bridge_audit_summary.json",
        root / "context_review" / "step16_v8_context_review_summary.json",
        root / "contextual_evidence" / "step15_v8_contextual_evidence_summary.json",
        root / "step12" / "step12_v8_statistical_robustness.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete Step15-v8 run, missing: {path}")
    output = root / "step15_v8_sync_manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Step15-v8 sync manifest: {output}")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != output):
        files.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": common.sha256(path),
            }
        )
    producer_paths = [
        common.resolve(args.policy),
        ROOT / "scripts" / "run_step15_v8_readiness_linux_20260715.sh",
        ROOT / "scripts" / "step15_v8_common.py",
        ROOT / "scripts" / "step15_v8_preflight.py",
        ROOT / "scripts" / "step15_build_v8_clean_semantics.py",
        ROOT / "scripts" / "step15_run_v8_bridge_audit.py",
        ROOT / "scripts" / "step16_build_v8_context_review_queues.py",
        ROOT / "scripts" / "step15_train_v8_contextual_evidence.py",
        ROOT / "scripts" / "step12_v8_statistical_robustness_audit.py",
        ROOT / "scripts" / "step15_v8_downstream_gate.py",
        ROOT / "scripts" / "step15_v8_build_sync_manifest.py",
        ROOT / "scripts" / "step15_v8_verify_readiness_runtime.py",
        ROOT / "scripts" / "step15_v7_common.py",
        ROOT / "scripts" / "step15_build_v7_inductive_pair_features.py",
        ROOT / "scripts" / "step15_build_v7_clean_embedding_cache.py",
        ROOT / "scripts" / "step16_apply_v8_context_reviews.py",
        ROOT / "scripts" / "step16_build_v8_context_review_queues.py",
        ROOT / "scripts" / "step16_build_v8_identity_control_queues.py",
        ROOT / "scripts" / "step16_reconcile_v8_dual_reviews.py",
        ROOT / "scripts" / "step16_reconcile_v8_identity_control_reviews.py",
        ROOT / "scripts" / "step16_reconcile_v8_profile_url_reviews.py",
        ROOT / "scripts" / "step16_materialize_v8_reviewed_readiness_freeze.py",
        ROOT / "scripts" / "step7_build_semantic_pair_features.py",
        ROOT / "scripts" / "step7_train_baseline_models.py",
        ROOT / "scripts" / "step9_run_few_shot_adaptation.py",
    ]
    for path in producer_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Step15-v8 producer source missing: {path}")
    producers = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256(path),
        }
        for path in producer_paths
    ]
    manifest = {
        "step": "step15_v8_sync_manifest",
        "run_id": run_id,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "required_outputs": [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in required
        ],
        "producer_sources": producers,
        "readiness_runtime_chain": runtime_chain,
        "files": files,
    }
    manifest["manifest_sha256"] = common.canonical_hash(manifest)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": run_id,
                "file_count": len(files),
                "manifest": str(output.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
