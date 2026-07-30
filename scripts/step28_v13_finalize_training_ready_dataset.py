#!/usr/bin/env python3
"""Finalize the four-split Step28-v13 training-ready dataset release.

This privileged validator rereads every split from disk, verifies all manifest
members, independently replays classification/retrieval supervision, checks
the five persisted M1 matrices, and rejects any cross-split identifier or
identity-value overlap.  It is the only command allowed to publish the parent
``release_manifest.json`` status.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_build_training_ready_dataset as builder
import step28_v13_common as common


FINAL_MANIFEST_VERSION = (
    "2026-07-30-step28-v13-training-ready-release-manifest-v3"
)


def _has_reparse_attribute(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _require_plain_directory(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or _has_reparse_attribute(path)
    ):
        raise common.ContractError(
            f"{label} must be a plain non-reparse directory"
        )


def _require_plain_file(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or _has_reparse_attribute(path)
    ):
        raise common.ContractError(
            f"{label} must be a plain non-reparse regular file"
        )


def _reject_reparse_tree(root: Path) -> None:
    _require_plain_directory(root, label="split release root")
    for directory, directory_names, file_names in os.walk(
        common.filesystem_path(root),
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        _require_plain_directory(
            directory_path,
            label="split release directory member",
        )
        for name in directory_names:
            _require_plain_directory(
                directory_path / name,
                label="split release directory member",
            )
        for name in file_names:
            _require_plain_file(
                directory_path / name,
                label="split release file member",
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_plain_file(path, label="JSONL release member")
    rows: list[dict[str, Any]] = []
    with open(
        common.filesystem_path(path),
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise common.ContractError(
                    f"Malformed JSONL line {line_number}: {path.name}"
                )
            value = json.loads(
                line,
                object_pairs_hook=common._reject_duplicate_pairs,
            )
            if not isinstance(value, dict):
                raise common.ContractError(
                    f"Non-object JSONL line {line_number}: {path.name}"
                )
            rows.append(value)
    return rows


def _set_receipt(values: set[str]) -> dict[str, Any]:
    if any(not isinstance(value, str) or not value for value in values):
        raise common.ContractError("Release identifier set has invalid values")
    return {
        "count": len(values),
        "sorted_values_sha256": common.canonical_sha256(
            common.utf8_sort(values)
        ),
    }


def _pairwise_disjoint(
    values_by_split: Mapping[str, set[str]],
    *,
    label: str,
) -> dict[str, Any]:
    intersections: dict[str, int] = {}
    for left_index, left in enumerate(builder.SPLITS):
        for right in builder.SPLITS[left_index + 1 :]:
            count = len(values_by_split[left] & values_by_split[right])
            intersections[f"{left}__{right}"] = count
            if count:
                raise common.ContractError(
                    f"Cross-split {label} overlap: {left}/{right}={count}"
                )
    return {
        "split_sets": {
            split: _set_receipt(values_by_split[split])
            for split in builder.SPLITS
        },
        "pairwise_intersection_counts": intersections,
        "all_pairwise_disjoint": True,
    }


def _validate_one_split(
    *,
    policy: Mapping[str, Any],
    overlay: Mapping[str, Any],
    split: str,
    directory: Path,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    _reject_reparse_tree(directory)
    manifest = common.load_json(directory / "split_manifest.json")
    builder._validate_split_tree(
        directory,
        expected_manifest=manifest,
    )
    expected_worlds = int(overlay["world_counts"][split])
    expected_positive = (
        expected_worlds
        * int(overlay["classification_positive_count_per_world"][split])
    )
    if (
        manifest.get("version") != builder.MANIFEST_VERSION
        or manifest.get("status") != "PASS_SPLIT_DATASET_READY"
        or manifest.get("run_id") != overlay["run_id"]
        or manifest.get("split") != split
        or manifest.get("claim_level")
        != overlay["target_release_claim_level"]
        or manifest.get("overlay_canonical_sha256")
        != common.canonical_sha256(overlay)
        or manifest.get("implementation_contract")
        != overlay["implementation_contract"]
        or manifest.get("implementation_contract_sha256")
        != builder.implementation_contract_sha256(overlay)
        or manifest.get("scientific_contract")
        != overlay["scientific_contract"]
        or manifest.get("base_policy") != overlay["base_policy"]
        or manifest.get("dataset_builder") != overlay["dataset_builder"]
        or manifest.get("structure_key_sha256_commitment")
        != overlay["private_structure_key_custody"]["commitments"][
            split
        ]
        or int(manifest.get("world_count", -1)) != expected_worlds
        or int(manifest.get("seller_count", -1)) != expected_worlds * 28
        or int(manifest.get("complete_pair_count", -1))
        != expected_worlds * 378
        or int(manifest.get("candidate_pair_count", -1))
        != expected_worlds * 40
        or int(manifest.get("positive_count", -1)) != expected_positive
        or manifest.get("metadata_shortcut_status")
        != "PASS_METADATA_SHORTCUT_ONLY"
        or manifest.get("label_formula_exact") is not True
        or manifest.get("aggregate_integrity", {}).get(
            "all_keysets_and_foreign_keys_exact"
        )
        is not True
        or manifest.get("metadata_shortcut_values_withheld")
        is not (split in {"audit_a", "audit_b"})
        or (
            manifest.get("metadata_shortcut_max_symmetric_auc") is None
        )
        is not (split in {"audit_a", "audit_b"})
        or (
            manifest.get("metadata_shortcut_bootstrap_95_upper") is None
        )
        is not (split in {"audit_a", "audit_b"})
        or manifest.get("identity33_no_all_zero_columns_required")
        is not (split == "train")
        or (
            split == "train"
            and manifest.get("identity33_no_all_zero_columns") is not True
        )
    ):
        raise common.ContractError(f"Split manifest semantic drift: {split}")
    schema = policy["relational_integrity"]["observed_core_schemas"]
    pair_fields = policy["relational_integrity"][
        "pair_projection_contract"
    ]["complete_model_pair_endpoints_schema"]
    worlds = builder._read_csv_exact(
        directory / "observed/worlds.csv",
        fields=schema["worlds.csv"],
    )
    sellers = builder._read_csv_exact(
        directory / "observed/sellers.csv",
        fields=schema["sellers.csv"],
    )
    endpoints = builder._read_csv_exact(
        directory / "observed/complete_model_pair_endpoints.csv",
        fields=pair_fields,
    )
    candidates = builder._read_csv_exact(
        directory / "observed/candidate_pairs.csv",
        fields=(
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        ),
    )
    memberships = builder._read_csv_exact(
        directory / "private_oracle/controller_membership.csv",
        fields=("world_uid", "controller_uid", "seller_uid"),
    )
    supervision_root = (
        "supervision"
        if split in {"train", "development"}
        else "sealed_supervision"
    )
    labels = builder._read_csv_exact(
        directory
        / supervision_root
        / "classification_labels.csv",
        fields=("canonical_pair_uid", "label"),
    )
    formula = builder._validate_formula_independently(
        candidate_rows=candidates,
        membership_rows=memberships,
        labels=labels,
    )
    if (
        formula["exact_rowwise_equal"] is not True
        or int(formula["positive_count"]) != expected_positive
    ):
        raise common.ContractError(
            f"Persisted label formula replay failed: {split}"
        )
    mount_contract = common.load_json(directory / "model_mounts.json")
    if mount_contract != builder.model_mount_contract(split):
        raise common.ContractError(f"Model-input allow-list drift: {split}")

    identity_fields = [
        "canonical_pair_uid",
        "world_uid",
        *policy["history_features"]["feature_names"],
    ]
    identity33 = builder._read_csv_exact(
        directory / "observed/identity33_all_pairs.csv",
        fields=identity_fields,
    )
    if (
        len(worlds) != expected_worlds
        or len(sellers) != expected_worlds * 28
        or len(endpoints) != expected_worlds * 378
        or len(candidates) != expected_worlds * 40
        or len(labels) != len(candidates)
        or len(identity33) != len(endpoints)
        or int(manifest.get("identity33_row_count", -1))
        != len(identity33)
    ):
        raise common.ContractError(
            f"Persisted public/supervision row counts drift: {split}"
        )
    m1_audit: dict[str, Any] = {
        "replicate_count": 0,
        "persisted_replay_exact": True,
    }
    receipts = common.load_json(
        directory / "audit/m1_derangement_receipts.json"
    )
    if split == "train":
        if (
            not isinstance(receipts, list)
            or len(receipts) != 5
            or len({row["rewire_seed_id"] for row in receipts}) != 5
            or len({row["matrix_sha256"] for row in receipts}) != 5
            or len({row["mapping_sha256"] for row in receipts}) != 5
        ):
            raise common.ContractError("Persisted M1 receipt set drift")
        for receipt in receipts:
            matrix_path = directory / str(receipt["matrix_path"])
            mapping_path = directory / str(receipt["mapping_path"])
            if (
                common.sha256_file(matrix_path)
                != receipt["matrix_sha256"]
                or common.sha256_file(mapping_path)
                != receipt["mapping_sha256"]
            ):
                raise common.ContractError(
                    "Persisted M1 receipt file hash drift"
                )
            replay = builder._validate_persisted_m1(
                policy,
                seed_id=str(receipt["rewire_seed_id"]),
                matrix_path=matrix_path,
                mapping_path=mapping_path,
                m2_rows=identity33,
                candidate_rows=candidates,
                endpoint_rows=endpoints,
            )
            if not all(value is True for value in replay.values()):
                raise common.ContractError(
                    "Persisted M1 final replay failed"
                )
        m1_audit["replicate_count"] = 5
    elif receipts != []:
        raise common.ContractError(
            f"Non-train split contains M1 training matrices: {split}"
        )

    retrieval_audit: dict[str, Any] = {
        "enabled": split in {"audit_a", "audit_b"},
        "exact_replay": True,
    }
    query_uids: set[str] = set()
    relation_uids: set[str] = set()
    if split in {"audit_a", "audit_b"}:
        queries = builder._read_csv_exact(
            directory / "retrieval/queries.csv",
            fields=("query_uid", "world_uid", "query_seller_uid"),
        )
        relations = builder._read_csv_exact(
            directory / "retrieval/relations.csv",
            fields=(
                "relation_uid",
                "query_uid",
                "world_uid",
                "query_seller_uid",
                "gallery_seller_uid",
            ),
        )
        qrels = builder._read_csv_exact(
            directory / "sealed_supervision/retrieval_qrels.csv",
            fields=("relation_uid", "query_uid", "relevance"),
        )
        expected_queries, expected_relations, expected_qrels = (
            builder._build_retrieval(
                policy,
                sellers=sellers,
                memberships=memberships,
                queries_per_world=4,
            )
        )
        if (
            queries != expected_queries
            or relations != expected_relations
            or qrels != expected_qrels
        ):
            raise common.ContractError(
                f"Persisted retrieval formula replay failed: {split}"
            )
        query_uids = {row["query_uid"] for row in queries}
        relation_uids = {row["relation_uid"] for row in relations}
        retrieval_audit.update(
            {
                "query_count": len(queries),
                "relation_count": len(relations),
                "qrel_count": len(qrels),
            }
        )
    raw_items = _read_jsonl(
        directory / "private_oracle/raw_identity_bearing_items.jsonl"
    )
    identity_assets = _read_jsonl(
        directory / "private_oracle/identity_assets.jsonl"
    )
    if int(manifest.get("item_count", -1)) != len(raw_items):
        raise common.ContractError(f"Persisted item count drift: {split}")
    if split in {"audit_a", "audit_b"}:
        if (
            int(manifest.get("retrieval_query_count", -1))
            != expected_worlds * 4
            or int(manifest.get("retrieval_relation_count", -1))
            != expected_worlds * 4 * 27
        ):
            raise common.ContractError(
                f"Persisted retrieval manifest counts drift: {split}"
            )
    elif (
        int(manifest.get("retrieval_query_count", -1)) != 0
        or int(manifest.get("retrieval_relation_count", -1)) != 0
    ):
        raise common.ContractError(
            f"Unexpected retrieval counts outside audits: {split}"
        )
    identifiers = {
        "world_uid": {str(row["world_uid"]) for row in worlds},
        "seller_uid": {str(row["seller_uid"]) for row in sellers},
        "item_uid": {str(row["item_uid"]) for row in raw_items},
        "pair_uid": {
            str(row["canonical_pair_uid"]) for row in endpoints
        },
        "controller_uid": {
            str(row["controller_uid"]) for row in memberships
        },
        "identity_asset_uid": {
            str(row["identity_asset_uid"]) for row in identity_assets
        },
        "identity_uid": {
            str(row["identity_uid"]) for row in identity_assets
        },
        "identity_value": {
            str(row["identity_value"]) for row in identity_assets
        },
        "query_uid": query_uids,
        "relation_uid": relation_uids,
    }
    expected_unique_counts = {
        "world_uid": expected_worlds,
        "seller_uid": expected_worlds * 28,
        "item_uid": len(raw_items),
        "pair_uid": expected_worlds * 378,
    }
    if any(
        len(identifiers[name]) != count
        for name, count in expected_unique_counts.items()
    ):
        raise common.ContractError(
            f"Persisted identifier uniqueness failed: {split}"
        )
    receipt = {
        "manifest_sha256": common.sha256_file(
            directory / "split_manifest.json"
        ),
        "manifest_self_sha256": manifest["canonical_self_hash"],
        "label_formula_replay": formula,
        "retrieval_replay": retrieval_audit,
        "m1_replay": m1_audit,
        "model_mount_contract_exact": True,
    }
    return receipt, identifiers


def finalize(overlay_path: Path) -> Path:
    overlay = builder.load_overlay(
        overlay_path,
        require_generation_frozen=True,
    )
    release_root = common.repo_path(str(overlay["output_root"]))
    _require_plain_directory(
        release_root,
        label="training-ready release root",
    )
    target = release_root / "release_manifest.json"
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite final release manifest: {target}"
        )
    split_receipts: dict[str, dict[str, Any]] = {}
    identifiers_by_kind: dict[str, dict[str, set[str]]] = {}
    for split in builder.SPLITS:
        structure_key = builder._load_split_key(overlay, split=split)
        base = builder._load_pinned_base(overlay)
        policy = builder._execution_policy(
            base,
            overlay,
            structure_key_hex=structure_key,
        )
        receipt, identifiers = _validate_one_split(
            policy=policy,
            overlay=overlay,
            split=split,
            directory=release_root / split,
        )
        split_receipts[split] = receipt
        for kind, values in identifiers.items():
            identifiers_by_kind.setdefault(kind, {})[split] = values
    cross_split = {
        kind: _pairwise_disjoint(
            values_by_split,
            label=kind,
        )
        for kind, values_by_split in identifiers_by_kind.items()
    }
    if any(
        set(values_by_split) != set(builder.SPLITS)
        for values_by_split in identifiers_by_kind.values()
    ):
        raise common.ContractError("Cross-split identifier audit is incomplete")
    finalizer_path = (
        common.ROOT
        / "scripts"
        / "step28_v13_finalize_training_ready_dataset.py"
    )
    ceremony = overlay["private_structure_key_custody"][
        "ceremony_receipt"
    ]
    manifest: dict[str, Any] = {
        "version": FINAL_MANIFEST_VERSION,
        "status": overlay["release_status_required"],
        "run_id": overlay["run_id"],
        "claim_level": overlay["target_release_claim_level"],
        "blind_custody_attested": False,
        "fixed_holdout_bytes_ready": True,
        "overlay_path": overlay_path.relative_to(common.ROOT).as_posix(),
        "overlay_sha256": common.sha256_file(overlay_path),
        "overlay_canonical_sha256": common.canonical_sha256(overlay),
        "implementation_contract": dict(
            overlay["implementation_contract"]
        ),
        "implementation_contract_sha256": (
            builder.implementation_contract_sha256(overlay)
        ),
        "scientific_contract": dict(overlay["scientific_contract"]),
        "dataset_builder": dict(overlay["dataset_builder"]),
        "exact_implementation_preflights": dict(
            overlay["exact_implementation_preflights"]
        ),
        "base_policy": dict(overlay["base_policy"]),
        "release_contract": dict(overlay["release_contract"]),
        "key_ceremony_receipt": dict(ceremony),
        "split_receipts": split_receipts,
        "cross_split_disjointness": cross_split,
        "all_split_manifests_exact": True,
        "all_label_formula_replays_exact": True,
        "all_retrieval_formula_replays_exact": True,
        "all_model_mount_contracts_exact": True,
        "train_m1_persisted_replays_exact": True,
        "finalizer_path": (
            "scripts/step28_v13_finalize_training_ready_dataset.py"
        ),
        "finalizer_sha256": common.sha256_file(finalizer_path),
    }
    manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
    temporary = release_root / ".release-manifest.tmp"
    if temporary.exists():
        raise common.ContractError(
            "Stale final release-manifest staging file exists"
        )
    common.write_json(temporary, manifest)
    observed = common.load_json(temporary)
    if common.canonical_json_bytes(observed) != common.canonical_json_bytes(
        manifest
    ):
        raise common.ContractError("Final release manifest write drift")
    common.atomic_rename_no_replace(temporary, target)
    if common.load_json(target) != manifest:
        raise common.ContractError(
            "Final release manifest changed after atomic publish"
        )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=builder.DEFAULT_OVERLAY,
    )
    return parser.parse_args()


def main() -> None:
    target = finalize(parse_args().overlay.resolve())
    print(f"Published final training-ready manifest: {target}")


if __name__ == "__main__":
    main()
