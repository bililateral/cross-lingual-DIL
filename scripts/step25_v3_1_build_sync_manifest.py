#!/usr/bin/env python3
"""Build the closed hash-bound Step25-v3.1 solver-repair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step24_common as step24
import step25_v3_1_common as common


PRODUCERS = [
    "scripts/step25_v3_1_common.py",
    "scripts/step25_v3_1_build_dual_channel_features.py",
    "scripts/step25_v3_1_evaluate_copy_aware_fusion.py",
    "scripts/step25_v3_1_train_operational_identifier_control.py",
    "scripts/step25_v3_1_build_sync_manifest.py",
    "scripts/run_step25_v3_1_solverfix_linux_20260718.sh",
    "scripts/step25_v3_common.py",
    "scripts/step25_v3_build_dual_channel_features.py",
    "scripts/step25_v3_evaluate_copy_aware_fusion.py",
    "scripts/step25_v3_train_operational_identifier_control.py",
    "scripts/step25_v2_common.py",
    "scripts/step25_v2_evaluate_pair_local_copy.py",
    "scripts/step25_common.py",
    "scripts/step25_evaluate_template_decontaminated_authorship.py",
    "scripts/step24_common.py",
    "scripts/step24_evaluate_content_independent_authorship.py",
    "scripts/step15_v7_common.py",
    "scripts/step9_run_few_shot_adaptation.py",
    "schema/step25_v3_1_solver_convergence_policy.json",
    "schema/step25_v3_copy_aware_dual_channel_policy.json",
    "schema/step25_v2_pair_local_copy_diagnostic_policy.json",
    "schema/step25_template_decontaminated_authorship_policy.json",
    "schema/step24_content_independent_authorship_policy.json",
    "schema/step15_v7_two_stage_policy.json",
    "tests/test_step25_v3_1_solver_convergence_contracts.py",
    "docs/STEP25_V3_1_SOLVER_CONVERGENCE_REPAIR_20260718.zh.md",
]


def expected_paths(policy: dict) -> list[Path]:
    root = common.resolve(policy["outputs_root"])
    paths = [
        root / value
        for key, value in policy["outputs"].items()
        if key != "sync_manifest"
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("Step25-v3.1 output policy contains a collision")
    return sorted(paths, key=lambda item: str(item))


def iter_solver_artifacts(value):
    if isinstance(value, dict):
        if value.get("model_family") == "step25_v3_1_direction_constrained_logistic_l2":
            yield value
        for child in value.values():
            yield from iter_solver_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_solver_artifacts(child)


def validate_solver_artifacts(policy: dict, payload: dict) -> dict:
    tolerance = float(policy["evaluation"]["logistic"]["projected_gradient_tolerance"])
    records = list(iter_solver_artifacts(payload))
    if not records:
        raise ValueError("Step25-v3.1 manifest found no repaired solver artifacts")
    failures = [
        record
        for record in records
        if record.get("solver_converged") is not True
        or record.get("solver_termination_reason")
        != "projected_gradient_kkt_tolerance"
        or float(record.get("solver_final_projected_gradient", float("inf")))
        > tolerance
        or record.get("relative_loss_used_for_convergence") is not False
    ]
    if failures:
        raise ValueError("Step25-v3.1 return bundle contains a non-KKT solver artifact")
    return {
        "artifact_count": len(records),
        "projected_gradient_tolerance": tolerance,
        "maximum_final_projected_gradient": max(
            float(record["solver_final_projected_gradient"]) for record in records
        ),
        "termination_reasons": sorted(
            {record["solver_termination_reason"] for record in records}
        ),
        "relative_loss_used_for_convergence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy_path, policy, *_ = common.load_policy(args.policy)
    paths = expected_paths(policy)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "expected_payload_count": len(paths),
                    "output_root": policy["outputs_root"],
                    "solver_convergence_contract": "projected_gradient_kkt_only_fail_closed",
                    "numerical_execution_performed": False,
                },
                indent=2,
            )
        )
        return

    parent_manifests = common.require_parent_manifests(policy)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Step25-v3.1 return payload is incomplete: {missing[0]}")
    root = common.resolve(policy["outputs_root"])
    evaluation_path = root / policy["outputs"]["evaluation_summary"]
    operational_path = root / policy["outputs"]["operational_summary"]
    artifacts_path = root / policy["outputs"]["model_artifacts"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    for payload in (evaluation, operational):
        if payload.get("publication_promotion_eligible") is not False:
            raise ValueError("Step25-v3.1 payload carries an invalid publication claim")
        if payload.get("step11_or_step17_entry_allowed") is not False:
            raise ValueError("Step25-v3.1 payload improperly unlocks Step11/17")
    if evaluation.get("valid_or_test_rows_read_or_scored") != 0:
        raise ValueError("Step25-v3.1 evaluation read valid/test")
    solver_audit = validate_solver_artifacts(policy, artifacts)

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
            raise FileNotFoundError(f"Step25-v3.1 producer is missing: {path}")
        producers.append({"path": relative, "sha256": step24.sha256_file(path)})
    manifest = {
        "step": "step25_v3_1_sync_manifest",
        "version": policy["version"],
        "boundary": policy["boundary"]["name"],
        "repair_scope": "solver_termination_only",
        "scientific_matrix_changed_from_v3": False,
        "d1_replication_candidate_eligible": bool(
            evaluation["d1_replication_candidate_eligible"]
        ),
        "publication_promotion_eligible": False,
        "step11_or_step17_entry_allowed": False,
        "solver_audit": solver_audit,
        "parent_manifests": parent_manifests,
        "payload_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "payload_files_sha256": step24.canonical_hash(records),
        "producers": producers,
        "producer_files_sha256": step24.canonical_hash(producers),
        "policy_path": str(policy_path.relative_to(common.ROOT)).replace("\\", "/"),
        "policy_sha256": step24.sha256_file(policy_path),
        "base_v3_policy_sha256": step24.sha256_file(
            common.resolve(common.load_json(policy_path)["base_policy"])
        ),
        "producer_sha256": step24.sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_sha256"] = step24.canonical_hash(manifest)
    output_path = root / policy["outputs"]["sync_manifest"]
    step24.write_json_immutable(output_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_count": len(records),
                "solver_audit": solver_audit,
                "d1_replication_candidate_eligible": manifest[
                    "d1_replication_candidate_eligible"
                ],
                "manifest": str(output_path.relative_to(common.ROOT)).replace(
                    "\\", "/"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
