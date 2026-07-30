#!/usr/bin/env python3
"""Fail closed unless a Step28-v13 model receives the exact registered files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import step28_v13_build_training_ready_dataset as builder
import step28_v13_common as common


ROLES = ("m0", "m1_train", "m2_train", "adapter_eval")


def validate_mount(
    *,
    split_directory: Path,
    role: str,
    mounted_relative_paths: list[str],
    replicate: str | None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise common.ContractError("Unknown Step28-v13 model role")
    manifest = common.load_json(split_directory / "split_manifest.json")
    builder._validate_split_tree(
        split_directory,
        expected_manifest=manifest,
    )
    split = str(manifest["split"])
    contract = common.load_json(split_directory / "model_mounts.json")
    if contract != builder.model_mount_contract(split):
        raise common.ContractError("Model-input contract file drift")
    if role == "m0":
        expected = contract["m0_exact_allowlist"]
        if replicate is not None:
            raise common.ContractError("M0 does not accept a replicate")
    elif role == "m2_train":
        expected = contract["m2_training_exact_allowlist"]
        if split != "train" or replicate is not None:
            raise common.ContractError("M2 training requires train only")
    elif role == "m1_train":
        if (
            split != "train"
            or replicate
            not in contract["m1_training_exact_allowlist_by_replicate"]
        ):
            raise common.ContractError(
                "M1 training requires one registered train replicate"
            )
        expected = contract[
            "m1_training_exact_allowlist_by_replicate"
        ][replicate]
    else:
        expected = contract["adapter_evaluation_exact_allowlist"]
        if split == "train" or replicate is not None:
            raise common.ContractError(
                "Adapter evaluation requires a non-train split"
            )
    observed = list(mounted_relative_paths)
    if (
        observed != expected
        or len(observed) != len(set(observed))
        or any(
            Path(value).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(value).parts)
            or not (split_directory / value).is_file()
            for value in observed
        )
    ):
        raise common.ContractError(
            "Mounted dataset files differ from the exact role allow-list"
        )
    forbidden_prefixes = (
        "private_oracle/",
        "private_audit/",
        "sealed_supervision/",
    )
    if any(
        value.startswith(forbidden_prefixes) for value in observed
    ):
        raise common.ContractError(
            "Private or sealed dataset material was model-mounted"
        )
    return {
        "status": "PASS_EXACT_MODEL_INPUT_ALLOWLIST",
        "run_id": manifest["run_id"],
        "split": split,
        "role": role,
        "replicate": replicate,
        "mounted_relative_paths": observed,
        "mounted_file_sha256": {
            value: common.sha256_file(split_directory / value)
            for value in observed
        },
        "join_keys_must_not_enter_feature_matrix": contract[
            "join_keys_must_not_enter_feature_matrix"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-directory", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--replicate")
    parser.add_argument(
        "--mounted-relative-path",
        action="append",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = validate_mount(
        split_directory=args.split_directory.resolve(),
        role=args.role,
        mounted_relative_paths=args.mounted_relative_path,
        replicate=args.replicate,
    )
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
