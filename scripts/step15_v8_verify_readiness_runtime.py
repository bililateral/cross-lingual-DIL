#!/usr/bin/env python3
"""Verify or classify the immutable Step15-v8 readiness runtime bundle."""

from __future__ import annotations

import argparse
import json

import step15_v8_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(common.DEFAULT_POLICY))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    _, policy, v7_policy = common.load_policy(args.policy)
    common.validate_policy_contract(policy, v7_policy)
    expected = [
        common.resolve(v7_policy["clean_semantic_encoder"]["manifest_output"]),
        common.resolve(v7_policy["inductive_features"]["manifest_output"]),
        common.resolve(v7_policy["inductive_features"]["reference_bundle_output"]),
        *[
            common.resolve(pool["v7_pair_features"])
            for pool in v7_policy["pools"].values()
        ],
    ]
    present = [path.is_file() for path in expected]
    if not any(present):
        if not args.status:
            raise FileNotFoundError("Step15-v8 readiness runtime outputs are absent")
        print(json.dumps({"status": "absent", "expected": [str(p) for p in expected]}))
        return
    if not all(present):
        missing = [str(path) for path, exists in zip(expected, present, strict=True) if not exists]
        raise FileNotFoundError(
            "Partial Step15-v8 readiness runtime bundle; refusing mixed rebuild/reuse: "
            + ", ".join(missing)
        )
    verified = common.verify_readiness_runtime_chain(policy, v7_policy)
    print(json.dumps({"status": "complete", **verified}, indent=2))


if __name__ == "__main__":
    main()
