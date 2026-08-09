#!/usr/bin/env python3
"""Finalize the four published v1.12 splits into one dataset-only release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_formal_executor as executor
import step28_v13_v1_12_formal_quality_audit as quality
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]


class ReleaseFinalizerError(ValueError):
    """Raised when v1.12 is not a complete sealed four-split release."""


def _load_receipt(path: Path, *, status: str, split: str | None = None) -> dict[str, Any]:
    document = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        document, label=f"v1.12 finalizer receipt {path.name}"
    )
    if document.get("status") != status or (
        split is not None and document.get("split") != split
    ):
        raise ReleaseFinalizerError(f"Finalizer receipt drift: {path}")
    return document


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _validate_seed_custody_hash_only(
    *, validated: dict[str, Any], private_root: Path
) -> dict[str, Any]:
    """Validate complete seed custody without opening any raw secret value."""

    bundle = private_root / "seed_custody"
    manifest_path = bundle / "private_manifest.json"
    manifest = preceremony.load_json_strict(manifest_path)
    preceremony.validate_canonical_self_hash(
        manifest, label="v1.12 final seed custody manifest"
    )
    ceremony_receipt = validated["ceremony_receipt"]
    records = manifest.get("files", [])
    if (
        manifest.get("status") != "PASS_PRIVATE_SEED_CUSTODY"
        or manifest.get("run_id") != validated["draft"]["run_id"]
        or int(manifest.get("master_document_count", -1)) != 4
        or int(manifest.get("generator_document_count", -1)) != 4
        or int(manifest.get("m1_document_count", -1)) != 5
        or not isinstance(records, list)
        or len(records) != 13
        or preceremony.sha256_file(manifest_path)
        != ceremony_receipt.get("private_manifest_sha256")
        or manifest.get("canonical_self_hash")
        != ceremony_receipt.get("private_manifest_canonical_self_hash")
    ):
        raise ReleaseFinalizerError("Seed custody manifest drift")
    expected_files = {
        "private_manifest.json",
        "public_receipt_copy.json",
        "execution_lock_copy.json",
    }
    seen_record_paths: set[str] = set()
    for index, record in enumerate(records):
        path = preceremony.verify_file_pin(
            record, label=f"seed custody raw file {index}"
        )
        try:
            relative = path.relative_to(bundle).as_posix()
        except ValueError as exc:
            raise ReleaseFinalizerError(
                "Seed custody manifest path escapes its bundle"
            ) from exc
        if relative in seen_record_paths or path.is_symlink():
            raise ReleaseFinalizerError("Seed custody file/path type drift")
        seen_record_paths.add(relative)
    expected_files.update(seen_record_paths)
    observed_files = {
        path.relative_to(bundle).as_posix()
        for path in preceremony.walk_files_long_path(bundle)
    }
    if observed_files != expected_files:
        raise ReleaseFinalizerError("Seed custody tree member set drift")
    ceremony_path = preceremony._repo_path(
        str(validated["execution_lock"]["ceremony_receipt"]["path"])
    )
    if (
        preceremony.read_bytes_long_path(bundle / "public_receipt_copy.json")
        != preceremony.read_bytes_long_path(ceremony_path)
        or preceremony.read_bytes_long_path(bundle / "execution_lock_copy.json")
        != preceremony.read_bytes_long_path(formal.DEFAULT_EXECUTION_LOCK_PATH)
    ):
        raise ReleaseFinalizerError("Seed custody public-copy replay drift")
    return executor._repo_pin(manifest_path, include_self_hash=True)


def finalize(
    *,
    audit_a_lock_path: Path = formal.DEFAULT_AUDIT_A_LOCK_PATH,
    audit_b_lock_path: Path = formal.DEFAULT_AUDIT_B_LOCK_PATH,
    output: Path | None = None,
) -> dict[str, Any]:
    validated_a = formal.load_and_validate_audit_lock(audit_a_lock_path)
    validated = formal.load_and_validate_audit_lock(audit_b_lock_path)
    if validated["audit_a_lock"] != validated_a["audit_a_lock"]:
        raise ReleaseFinalizerError("Audit A/B authorization ladder drift")
    draft = validated["draft"]
    public_root = preceremony._repo_path(str(draft["release"]["public_root"]))
    private_root = preceremony._repo_path(str(draft["release"]["private_root"]))
    output = output or (public_root / "release_manifest.json")
    if output != public_root / "release_manifest.json":
        raise ReleaseFinalizerError("Root release manifest path is not canonical")

    with os.scandir(preceremony._filesystem_path(public_root)) as entries:
        observed_public_members = {
            entry.name for entry in entries if entry.name != output.name
        }
    if observed_public_members != set(formal.SPLITS):
        raise ReleaseFinalizerError("Public release top-level split set drift")
    with os.scandir(preceremony._filesystem_path(private_root)) as entries:
        observed_private_members = {entry.name for entry in entries}
    if observed_private_members != {"seed_custody", "splits"}:
        raise ReleaseFinalizerError("Private release top-level member set drift")
    with os.scandir(
        preceremony._filesystem_path(private_root / "splits")
    ) as entries:
        observed_private_splits = {entry.name for entry in entries}
    if observed_private_splits != set(formal.SPLITS):
        raise ReleaseFinalizerError("Private release split set drift")
    seed_custody_manifest = _validate_seed_custody_hash_only(
        validated=validated, private_root=private_root
    )

    publication_receipts: dict[str, dict[str, Any]] = {}
    quality_receipts: dict[str, dict[str, Any]] = {}
    inventories: dict[str, dict[str, Any]] = {}
    split_manifests: dict[str, dict[str, Any]] = {}
    core_start_receipts: dict[str, dict[str, Any]] = {}
    train_m1_start_receipts: dict[str, dict[str, Any]] = {}
    for split in formal.SPLITS:
        paths = executor._paths(draft, split)
        published = executor._validate_published_split(
            public_root=paths["public_final"],
            private_root=paths["private_final"],
            split=split,
            draft=draft,
        )
        split_manifests[split] = published["public_manifest"]
        receipt_path = audit_b_lock_path.parent / f"{split}_publication_receipt.json"
        publication = _load_receipt(
            receipt_path,
            status="PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE",
            split=split,
        )
        if (
            publication["public_manifest_sha256"]
            != preceremony.sha256_file(paths["public_final"] / "split_manifest.json")
            or publication["private_manifest_sha256"]
            != preceremony.sha256_file(paths["private_final"] / "private_manifest.json")
        ):
            raise ReleaseFinalizerError("Publication/manifest pin drift")
        core_start_spec = publication.get("core_start_receipt", {})
        core_start_path = preceremony.verify_file_pin(
            core_start_spec,
            label=f"formal {split} core start receipt",
        )
        core_start = preceremony.load_json_strict(core_start_path)
        preceremony.validate_canonical_self_hash(
            core_start, label=f"formal {split} core start receipt"
        )
        if (
            core_start.get("status")
            != "FORMAL_CORE_GENERATION_STARTED_NO_RESTART"
            or core_start.get("split") != split
            or core_start.get("canonical_self_hash")
            != core_start_spec.get("canonical_self_hash")
        ):
            raise ReleaseFinalizerError("Core start receipt drift")
        core_start_receipts[split] = dict(core_start_spec)
        m1_specs = publication.get("m1_start_receipts", {})
        if split == "train":
            if set(m1_specs) != set(formal.M1_ROLES):
                raise ReleaseFinalizerError("Train M1 start receipt set drift")
            for role, spec in m1_specs.items():
                start_path = preceremony.verify_file_pin(
                    spec, label=f"formal {role} start receipt"
                )
                start = preceremony.load_json_strict(start_path)
                preceremony.validate_canonical_self_hash(
                    start, label=f"formal {role} start receipt"
                )
                if (
                    start.get("status")
                    != "FORMAL_M1_MATERIALIZATION_STARTED_NO_RESTART"
                    or start.get("role") != role
                    or start.get("canonical_self_hash")
                    != spec.get("canonical_self_hash")
                ):
                    raise ReleaseFinalizerError("Train M1 start receipt drift")
                train_m1_start_receipts[role] = dict(spec)
        elif m1_specs != {}:
            raise ReleaseFinalizerError("Non-train split has M1 start receipts")
        publication_receipts[split] = executor._repo_pin(
            receipt_path, include_self_hash=True
        )
        quality_path = preceremony._repo_path(
            str(publication["quality_receipt"]["path"])
        )
        expected_status = (
            "PASS_FORMAL_TRAIN_DEVELOPMENT_QUALITY_GATE"
            if split in {"train", "development"}
            else "PASS_FORMAL_SEALED_AUDIT_SPLIT_QUALITY"
        )
        quality_document = _load_receipt(
            quality_path,
            status=expected_status,
            split=split if split in {"audit_a", "audit_b"} else None,
        )
        if preceremony.sha256_file(quality_path) != publication["quality_receipt"]["sha256"]:
            raise ReleaseFinalizerError("Publication/quality pin drift")
        quality_receipts[split] = executor._repo_pin(
            quality_path, include_self_hash=True
        )
        inventories[split] = quality._sealed_public_inventory(
            public=paths["public_final"],
            private=paths["private_final"],
            split=split,
        )
        supervision = paths["public_final"] / "supervision/classification_labels.csv"
        if preceremony.exists_long_path(supervision) is not (
            split in {"train", "development"}
        ):
            raise ReleaseFinalizerError("Public supervision split boundary drift")
        if any(
            "c40" in path.relative_to(paths["public_final"]).as_posix().casefold()
            for path in preceremony.walk_files_long_path(paths["public_final"])
        ):
            raise ReleaseFinalizerError("C40 entered a public v1.12 split")

    intersection_fields = (
        "world_uid",
        "seller_uid",
        "item_uid",
        "canonical_pair_uid",
        "query_uid",
        "identity_value_hash",
        "item_document_hash",
        "seller_document_hash",
    )
    pairwise: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(formal.SPLITS):
        for right in formal.SPLITS[left_index + 1 :]:
            key = f"{left}__{right}"
            counts = {
                name: len(inventories[left][name] & inventories[right][name])
                for name in intersection_fields
            }
            if any(counts.values()):
                raise ReleaseFinalizerError(
                    f"Root cross-split isolation failed: {key}"
                )
            pairwise[key] = counts
    with os.scandir(preceremony._filesystem_path(private_root)) as entries:
        temporary_private_members = {
            entry.name
            for entry in entries
            if entry.name in {"_staging", "builds", "executions"}
        }
    if temporary_private_members:
        raise ReleaseFinalizerError("Private temporary stage remains at finalization")

    manifest = preceremony.with_canonical_self_hash(
        {
            "version": draft["release"]["release_manifest_version"],
            "status": "PASS_V1_12_DATASET_ONLY_READY",
            "run_id": draft["run_id"],
            "claim_level": (
                "SYNTHETIC_DATASET_AND_SPLITS_READY_NO_MODEL_RESULTS_NO_REAL_WORLD_CLAIM"
            ),
            "split_order": list(formal.SPLITS),
            "worlds_per_split": 500,
            "pairs_per_split": 189000,
            "positive_pairs_per_split_by_frozen_formula": 10000,
            "negative_pairs_per_split_by_frozen_formula": 179000,
            "identity_assets_per_split": dict(
                draft["dataset_shape"]["identity_assets_per_split"]
            ),
            "total_world_count": 2000,
            "total_pair_count": 756000,
            "total_identity_asset_count": sum(
                int(value)
                for value in draft["dataset_shape"][
                    "identity_assets_per_split"
                ].values()
            ),
            "publication_receipts": publication_receipts,
            "core_generation_start_receipts": core_start_receipts,
            "train_m1_materialization_start_receipts": train_m1_start_receipts,
            "seed_custody_manifest": seed_custody_manifest,
            "quality_receipts": quality_receipts,
            "split_manifests": {
                split: {
                    "path": f"{split}/split_manifest.json",
                    "sha256": preceremony.sha256_file(
                        public_root / split / "split_manifest.json"
                    ),
                    "canonical_self_hash": split_manifests[split][
                        "canonical_self_hash"
                    ],
                }
                for split in formal.SPLITS
            },
            "cross_split_intersection_counts": pairwise,
            "c40_member_count": 0,
            "audit_label_or_qrel_rows_unsealed_for_modeling": False,
            "formal_seed_or_capability_persisted_publicly": False,
            "m0_m1_m2_m3_results_produced": False,
            "model_training_started": False,
            "dataset_ready_for_frozen_m0_scoring_and_registered_training": True,
            "real_chinese_external_validity_claimed": False,
            "producer_path": "scripts/step28_v13_v1_12_finalize_release.py",
            "producer_sha256": preceremony.sha256_file(Path(__file__)),
            "audit_generation_locks": {
                "audit_a": executor._repo_pin(
                    audit_a_lock_path, include_self_hash=True
                ),
                "audit_b": executor._repo_pin(
                    audit_b_lock_path, include_self_hash=True
                ),
            },
        }
    )
    payload = _json_bytes(manifest)
    if preceremony.exists_long_path(output):
        existing = _load_receipt(
            output, status="PASS_V1_12_DATASET_ONLY_READY"
        )
        if preceremony.read_bytes_long_path(output) != payload or existing != manifest:
            raise ReleaseFinalizerError(
                "Existing root release manifest differs from complete replay"
            )
        return existing
    preceremony.write_bytes_no_replace_long_path(output, payload)
    replay = preceremony.load_json_strict(output)
    preceremony.validate_canonical_self_hash(
        replay, label="v1.12 root release manifest"
    )
    if replay != manifest:
        raise ReleaseFinalizerError("Root release manifest byte replay failed")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-a-lock", type=Path, default=formal.DEFAULT_AUDIT_A_LOCK_PATH
    )
    parser.add_argument(
        "--audit-b-lock", type=Path, default=formal.DEFAULT_AUDIT_B_LOCK_PATH
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = finalize(
        audit_a_lock_path=args.audit_a_lock.resolve(),
        audit_b_lock_path=args.audit_b_lock.resolve(),
        output=args.output.resolve() if args.output else None,
    )
    output = preceremony._repo_path(
        str(formal.load_and_validate_audit_lock(args.audit_b_lock.resolve())["draft"]["release"]["public_root"])
    ) / "release_manifest.json"
    print(
        manifest["status"],
        manifest["total_world_count"],
        manifest["total_pair_count"],
        preceremony.sha256_file(output),
    )


if __name__ == "__main__":
    main()
