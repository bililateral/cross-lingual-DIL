#!/usr/bin/env python3
"""Create one fresh ignored V9.3-R2 authority without printing key values."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
import secrets
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_method_dataset_builder_v9_3 as builder
import step28_v13_v1_13_method_policy_v9_3 as method_policy
import step28_v13_v1_13_scientific_common_v9 as scientific


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _collect_hex64(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if HEX64.fullmatch(value) else set()
    if isinstance(value, Mapping):
        output: set[str] = set()
        for child in value.values():
            output.update(_collect_hex64(child))
        return output
    if isinstance(value, list):
        output = set()
        for child in value:
            output.update(_collect_hex64(child))
        return output
    return set()


def _forbidden_values(policy: Mapping[str, Any], target: Path) -> set[str]:
    forbidden = scientific._collect_random_authorities(policy.get("randomness", {}))
    custody = common.repo_path("private_custody")
    if custody.is_dir():
        for path in custody.rglob("*.json"):
            if path.resolve() == target.resolve():
                continue
            try:
                forbidden.update(_collect_hex64(common.load_json(path)))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise builder.MethodDatasetBuilderV93Error(
                    f"Cannot validate prior private authority: {path.name}"
                ) from exc
    return forbidden


def create(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path != builder.DEFAULT_AUTHORITY.resolve():
        raise builder.MethodDatasetBuilderV93Error(
            "Method random authority must use the frozen private path"
        )
    policy = method_policy.load_policy()
    if policy.get("status") != "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED":
        raise builder.MethodDatasetBuilderV93Error(
            "Method random authority is forbidden before all file pins are frozen"
        )
    base = common.load_json(
        common.repo_path(str(policy["frozen_inputs"]["parent_policy"]))
    )
    forbidden = _forbidden_values(base, path)
    values: list[str] = []
    while len(values) < 14:
        value = secrets.token_hex(32)
        if value not in forbidden and value not in values:
            values.append(value)
    keys = {
        "id_namespace_key_hex": values[0],
        "structure_key_hex": values[1],
        "id_key_hex": values[2],
        "identity_value_key_hex": values[3],
        "text_key_hex": values[4],
        "candidate_key_hex": values[5],
        "query_key_hex": values[6],
        "document_variation_key_hex": values[7],
        "anonymous_handle_key_hex": values[8],
        "rewire_key_hexes": values[9:14],
    }
    payload: dict[str, Any] = {
        "version": builder.AUTHORITY_VERSION,
        "status": "FROZEN_FRESH_SINGLE_USE",
        "canonical_self_sha256": None,
        "single_use": True,
        "world_counts": dict(builder.WORLD_COUNTS),
        "method_policy_canonical_self_sha256": policy["canonical_self_sha256"],
        "keys": keys,
    }
    canonical = deepcopy(payload)
    canonical["canonical_self_sha256"] = None
    payload["canonical_self_sha256"] = common.canonical_sha256(canonical)
    builder._validate_authority(payload)
    common.write_json(path, payload)
    return {
        "status": "CREATED_FRESH_SINGLE_USE_V9_3_R2_AUTHORITY",
        "path": path.relative_to(common.ROOT).as_posix(),
        "file_sha256": common.sha256_file(path),
        "canonical_self_sha256": payload["canonical_self_sha256"],
        "key_count": len(values),
        "key_values_printed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=builder.DEFAULT_AUTHORITY)
    args = parser.parse_args()
    print(json.dumps(create(args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
