#!/usr/bin/env python3
"""Freeze collision-free identity-value salts for the training-ready release."""

from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_common as common
import step28_v13_identity_values as identity_values


RELEASE_POLICY = (
    ROOT / "schema" / "step28_v13_training_ready_dataset_policy.json"
)


def main() -> None:
    release = common.load_json(RELEASE_POLICY)
    if (
        release.get("status") != "IMPLEMENTATION_LOCK_IN_PROGRESS"
        or release.get("generation_enabled") is not False
    ):
        raise common.ContractError(
            "Identity salts may only be built during the pre-generation lock"
        )
    base = common.load_policy(mode="development_smoke")
    policy = copy.deepcopy(base)
    policy["modes"]["formal"]["world_counts"] = dict(
        release["world_counts"]
    )
    policy["modes"]["formal"]["power_design_path"] = (
        release["design_diagnostics"]["mechanism_stratified_gate"][
            "path"
        ]
    )
    artifact = identity_values.build_salt_artifact(
        policy,
        mode="formal",
    )
    if (
        artifact["candidate_count_per_type"]
        != sum(release["world_counts"].values()) * 96
        or artifact["deny_intersection_count"] != 0
        or artifact["same_mode_cross_type_intersection_count"] != 0
        or artifact["config_projection"]["handle_encoding"]
        != release["handle_encoding"]
        or (
            release["identity_value_salts"][
                "per_type_salt_counters"
            ]
            is not None
            and artifact["salt_counters"]
            != release["identity_value_salts"][
                "per_type_salt_counters"
            ]
        )
    ):
        raise common.ContractError(
            "Training-ready identity salt artifact failed its boundary"
        )
    output = common.repo_path(
        release["identity_value_salts"]["artifact"]["path"]
    )
    common.write_json(output, artifact)
    print(f"Identity salt artifact: {output}")
    print(f"SHA-256: {common.sha256_file(output)}")
    print(f"Salt counters: {artifact['salt_counters']}")


if __name__ == "__main__":
    main()
