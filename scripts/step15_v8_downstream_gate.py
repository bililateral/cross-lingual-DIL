#!/usr/bin/env python3
"""Fail closed before any Step20, Step11, or Step17 publication action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import step15_v8_common as common


ROOT = Path(__file__).resolve().parent.parent


def validate_step20_lock(lock: dict, run_id: str, freeze_sha256: str) -> list[str]:
    checks = {
        "step15_v8_run_id": lock.get("step15_v8_run_id") == run_id,
        "step15_v8_model_freeze_manifest_sha256": lock.get(
            "step15_v8_model_freeze_manifest_sha256"
        )
        == freeze_sha256,
        "evaluation_completed_once": lock.get("evaluation_completed_once") is True,
        "evaluation_count": type(lock.get("evaluation_count")) is int
        and lock.get("evaluation_count") == 1,
        "model_configuration_frozen_before_holdout_unseal": lock.get(
            "model_configuration_frozen_before_holdout_unseal"
        )
        is True,
        "threshold_frozen_before_holdout_unseal": lock.get(
            "threshold_frozen_before_holdout_unseal"
        )
        is True,
        "prospective_holdout_used_for_model_selection": lock.get(
            "prospective_holdout_used_for_model_selection"
        )
        is False,
    }
    return [name for name, passed in checks.items() if not passed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--stage",
        choices=["step20", "step11_step17"],
        required=True,
    )
    args = parser.parse_args()

    _, policy, v7_policy = common.load_policy(args.policy)
    common.validate_policy_contract(policy, v7_policy)
    run_id = args.run_id or policy["default_run_id"]
    root = common.run_root(policy, run_id)
    step12_path = root / "step12" / "step12_v8_statistical_robustness.json"
    if not step12_path.is_file():
        raise FileNotFoundError(f"Step15-v8 downstream gate lacks Step12 summary: {step12_path}")
    step12 = common.load_json(step12_path)
    promotion = bool(step12.get("promotion", {}).get("eligible"))
    freeze_path = root / "step12" / "step15_v8_model_freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Step15-v8 downstream gate lacks model freeze: {freeze_path}")
    step20_lock = common.resolve(
        policy["downstream_gate"]["step20_evaluation_lock_template"].format(run_id=run_id)
    )
    step20_lock_failures = ["lock_file_missing"]
    if step20_lock.is_file():
        step20_lock_failures = validate_step20_lock(
            common.load_json(step20_lock), run_id, common.sha256(freeze_path)
        )
    step20_complete = not step20_lock_failures
    if args.stage == "step20":
        eligible = promotion
        reason = (
            "step12_method_promotion_passed"
            if eligible
            else "blocked_step12_method_or_validation_slice_gates_failed"
        )
    else:
        eligible = promotion and step20_complete
        reason = (
            "eligible_explicit_allowlist_only"
            if eligible
            else (
                "blocked_step12_promotion_failed"
                if not promotion
                else (
                    "blocked_step20_prospective_evaluation_not_complete"
                    if step20_lock_failures == ["lock_file_missing"]
                    else "blocked_step20_evaluation_lock_invalid"
                )
            )
        )
    result = {
        "status": "eligible" if eligible else "blocked",
        "requested_stage": args.stage,
        "run_id": run_id,
        "step12_promotion_eligible": promotion,
        "step20_evaluation_complete": step20_complete,
        "step20_evaluation_lock": str(step20_lock.relative_to(ROOT)).replace("\\", "/"),
        "step20_lock_validation_failures": step20_lock_failures,
        "reason": reason,
        "auto_selector_allowed": False,
        "step11_step17_require_explicit_allowlist": True,
    }
    print(json.dumps(result, indent=2))
    if not eligible:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
