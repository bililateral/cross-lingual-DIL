#!/usr/bin/env python3
"""Materialize the label-invariant real-Chinese train seller allow-list."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import step28_v13_common as common


ALLOWED_SOURCE_COLUMNS = frozenset(
    {
        "data_bucket",
        "split_name",
        "seller_uid_left",
        "seller_uid_right",
    }
)
OUTPUT_COLUMNS = ["seller_uid"]


class BoundaryOnlyRow(Mapping[str, str]):
    """Expose only the four fields allowed to define train membership."""

    def __init__(self, row: Mapping[str, str]) -> None:
        self._values = {name: str(row[name]) for name in ALLOWED_SOURCE_COLUMNS}

    def __getitem__(self, key: str) -> str:
        if key not in ALLOWED_SOURCE_COLUMNS:
            raise common.ContractError(f"Protected label column access attempted: {key}")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._values))

    def __len__(self) -> int:
        return len(self._values)


def extract_train_sellers(rows: list[Mapping[str, str]]) -> list[str]:
    sellers: set[str] = set()
    for row in rows:
        if row["data_bucket"] != "zh_target_strict" or row["split_name"] != "train":
            continue
        for column in ("seller_uid_left", "seller_uid_right"):
            seller_uid = row[column].strip()
            if not seller_uid:
                raise common.ContractError("Train pair contains an empty seller UID")
            sellers.add(seller_uid)
    if not sellers:
        raise common.ContractError("No zh_target_strict train sellers were found")
    return common.utf8_sort(sellers)


def read_source_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = ALLOWED_SOURCE_COLUMNS - set(fieldnames)
        if missing:
            raise common.ContractError(f"Protected membership source lacks columns: {sorted(missing)}")
        return list(reader), fieldnames


def prove_label_invariance(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    expected: list[str],
) -> dict[str, Any]:
    guarded = [BoundaryOnlyRow(row) for row in rows]
    if extract_train_sellers(guarded) != expected:
        raise common.ContractError("Boundary-only mapping changed train seller membership")

    protected_columns = sorted(set(fieldnames) - ALLOWED_SOURCE_COLUMNS)
    mutated: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        replacement = dict(row)
        for column_index, column in enumerate(protected_columns):
            replacement[column] = f"opaque_{row_index:06d}_{column_index:03d}"
        mutated.append(replacement)
    if extract_train_sellers(mutated) != expected:
        raise common.ContractError("Protected value mutation changed train seller membership")
    return {
        "boundary_only_mapping_equal": True,
        "all_nonboundary_columns_mutated_equal": True,
        "protected_column_count": len(protected_columns),
    }


def run(policy: dict[str, Any], mode: str, policy_path: Path) -> dict[str, Any]:
    source_spec = policy["frozen_inputs"]["protected_train_membership"]
    source = common.verify_file_pin(source_spec, label="protected_train_membership")
    rows, fieldnames = read_source_rows(source)
    sellers = extract_train_sellers(rows)
    invariance = prove_label_invariance(rows, fieldnames, sellers)

    output_root = common.mode_output_root(policy, mode)
    relative_output = Path(policy["style_reference_boundary"]["membership_output"])
    output = output_root / relative_output
    output_rows = [{"seller_uid": seller_uid} for seller_uid in sellers]
    common.write_csv(output, output_rows, OUTPUT_COLUMNS)

    producer = Path(__file__).resolve()
    manifest = {
        "step": "step28_v13",
        "stage": "materialize_style_source_boundary",
        "mode": mode,
        "run_id": policy["modes"][mode]["run_id"],
        "policy_sha256": common.sha256_file(policy_path.resolve()),
        "producer_sha256": common.sha256_file(producer),
        "source_file_sha256": source_spec["sha256"],
        "source_schema_sha256": common.canonical_sha256(fieldnames),
        "membership_rule": {
            "data_bucket": "zh_target_strict",
            "split_name": "train",
            "endpoint_columns": ["seller_uid_left", "seller_uid_right"],
        },
        "label_invariance": invariance,
        "output": common.artifact_record(
            output,
            role="style_source_train_seller_allowlist",
            root=output_root,
        ),
    }
    manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
    manifest_path = output_root / "manifests" / "style_source_boundary_manifest.json"
    common.write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_policy_argument(parser)
    common.add_mode_argument(parser)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()
    policy = common.load_policy(args.policy, mode=args.mode)
    if args.validate_config_only:
        print("Step28-v13 style source boundary configuration is valid")
        return
    manifest = run(policy, args.mode, args.policy)
    print(
        "Step28-v13 style source boundary materialized:",
        manifest["output"]["path"],
        manifest["canonical_self_hash"],
    )


if __name__ == "__main__":
    main()
