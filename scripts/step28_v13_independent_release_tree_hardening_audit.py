#!/usr/bin/env python3
"""Independently bind and replay the formal Step28-v13 v1.2 release tree.

This post-release audit deliberately does not import the dataset producer,
preflight, finalizer, or the earlier row-audit implementation.  It hashes every
manifest member and independently checks the complete world/C40 file layout and
the selected_global_rank HMAC order.  It does not parse labels, qrels,
controller membership, identity assets, or M1 supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION = (
    "2026-07-31-step28-v13-independent-release-tree-"
    "hardening-audit-v1"
)
EXPECTED_RUN_ID = "v13_training_ready_v1_2_order_repair_20260731"
EXPECTED_RELEASE_STATUS = "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
EXPECTED_RELEASE_SHA256 = (
    "81ca7d9d2040d500b3bcb2ffc9af6aeb72c581754dbd075b94dd6cf8904b8275"
)
EXPECTED_RELEASE_SELF_SHA256 = (
    "59001459bc9b3a908ab0efa1f9f46a6c821bf6078ba3dc5f3308f910d0c5e00b"
)
EXPECTED_SPLITS = ("train", "development", "audit_a", "audit_b")
EXPECTED_SPLIT_MANIFEST_SHA256 = {
    "train": (
        "b178f7869014be6ba7130cd5294e31aa232d6a52ad589547217810b0caaa9f9c"
    ),
    "development": (
        "902307ab9a8bc17cbf4e8564d14c4c8b4bc2bb17c70e48ba8de13696160da088"
    ),
    "audit_a": (
        "ebcd400fa07aa3cfe4a6d3270ff791e57fc69825c6e1d857f56035db8e503d02"
    ),
    "audit_b": (
        "a166567ea2b41ddd69d83df17be6799e3a73236653af333614e64f7383cffe3f"
    ),
}
EXPECTED_MEMBER_COUNTS = {
    "train": 44,
    "development": 34,
    "audit_a": 38,
    "audit_b": 38,
}
EXPECTED_WORLD_COUNT = 500
EXPECTED_CANDIDATES_PER_WORLD = 40
EXPECTED_CANDIDATE_COUNT = (
    EXPECTED_WORLD_COUNT * EXPECTED_CANDIDATES_PER_WORLD
)
EXPECTED_TOP_LEVEL = {
    "audit_a",
    "audit_b",
    "development",
    "train",
    "release_manifest.json",
    "repair_equivalence_report.json",
}
EXPECTED_WORLD_FIELDS = ("world_uid",)
EXPECTED_CANDIDATE_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
FIELD_SEPARATOR = b"\x1f"
REPARSE_POINT_ATTRIBUTE = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x400,
)


class AuditError(ValueError):
    """Raised when the frozen release tree fails closed."""


def reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AuditError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return bool(attributes & REPARSE_POINT_ATTRIBUTE)


def require_plain_file(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or is_reparse_point(path)
    ):
        raise AuditError(f"{label} is absent or not a plain file: {path}")


def require_plain_directory(path: Path, *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or is_reparse_point(path)
    ):
        raise AuditError(
            f"{label} is absent or not a plain directory: {path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    require_plain_file(path, label="JSON input")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicate_pairs,
        )
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_canonical_self_hash(
    document: Mapping[str, Any],
    *,
    label: str,
) -> str:
    expected = document.get("canonical_self_hash")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AuditError(f"{label} has no canonical self hash")
    payload = dict(document)
    del payload["canonical_self_hash"]
    observed = canonical_sha256(payload)
    if observed != expected:
        raise AuditError(f"{label} canonical self hash drift")
    return observed


def sha256_file(path: Path) -> str:
    require_plain_file(path, label="Hashed member")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_relative_member_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AuditError("Manifest member path is empty")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
        or "\\" in value
    ):
        raise AuditError(f"Unsafe manifest member path: {value!r}")
    return value


def read_world_uids(path: Path) -> list[str]:
    require_plain_file(path, label="World table")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_WORLD_FIELDS:
            raise AuditError("World table header drift")
        rows = [dict(row) for row in reader]
    if any(tuple(row) != EXPECTED_WORLD_FIELDS for row in rows):
        raise AuditError("Malformed world table row")
    world_uids = [row["world_uid"] for row in rows]
    if (
        len(world_uids) != EXPECTED_WORLD_COUNT
        or len(set(world_uids)) != EXPECTED_WORLD_COUNT
        or any(not value for value in world_uids)
    ):
        raise AuditError("World table cardinality or uniqueness drift")
    return world_uids


def output_order_digest(
    *,
    key_hex: str,
    world_uid: str,
    pair_uid: str,
) -> bytes:
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise AuditError("Candidate HMAC key is not hexadecimal") from exc
    if len(key) != 32:
        raise AuditError("Candidate HMAC key is not 32 bytes")
    parts = (
        world_uid.encode("utf-8"),
        b"selected_global_rank",
        pair_uid.encode("utf-8"),
    )
    if any(FIELD_SEPARATOR in part for part in parts):
        raise AuditError("Candidate HMAC part contains field separator")
    return hmac.new(
        key,
        FIELD_SEPARATOR.join(parts),
        hashlib.sha256,
    ).digest()


def validate_candidate_rows(
    path: Path,
    *,
    world_uids: Sequence[str],
    candidate_key_hex: str,
) -> dict[str, Any]:
    require_plain_file(path, label="Candidate table")
    completed_worlds: list[str] = []
    seen_worlds: set[str] = set()
    seen_pairs: set[str] = set()
    current_world: str | None = None
    current_pairs: list[str] = []
    row_count = 0

    def finish_world() -> None:
        nonlocal current_world, current_pairs
        if current_world is None:
            return
        position = len(completed_worlds)
        if (
            position >= len(world_uids)
            or current_world != world_uids[position]
            or current_world in seen_worlds
        ):
            raise AuditError("Candidate world block order drift")
        if (
            len(current_pairs) != EXPECTED_CANDIDATES_PER_WORLD
            or len(set(current_pairs)) != EXPECTED_CANDIDATES_PER_WORLD
        ):
            raise AuditError("Candidate world block cardinality drift")
        expected = sorted(
            current_pairs,
            key=lambda pair_uid: (
                output_order_digest(
                    key_hex=candidate_key_hex,
                    world_uid=current_world,
                    pair_uid=pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        if current_pairs != expected:
            raise AuditError(
                "Candidate selected_global_rank HMAC order drift"
            )
        seen_worlds.add(current_world)
        completed_worlds.append(current_world)
        current_pairs = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_CANDIDATE_FIELDS:
            raise AuditError("Candidate table header drift")
        for source in reader:
            row = dict(source)
            if tuple(row) != EXPECTED_CANDIDATE_FIELDS:
                raise AuditError("Malformed candidate table row")
            world_uid = row["world_uid"]
            pair_uid = row["canonical_pair_uid"]
            left = row["seller_uid_left"]
            right = row["seller_uid_right"]
            canonical_endpoints = sorted(
                (left, right),
                key=lambda value: value.encode("utf-8"),
            )
            if (
                not world_uid
                or not pair_uid
                or left == right
                or pair_uid
                != f"{canonical_endpoints[0]}||{canonical_endpoints[1]}"
                or pair_uid in seen_pairs
            ):
                raise AuditError(
                    "Candidate pair key, endpoints, or uniqueness drift"
                )
            if world_uid != current_world:
                finish_world()
                current_world = world_uid
            current_pairs.append(pair_uid)
            seen_pairs.add(pair_uid)
            row_count += 1
    finish_world()
    if (
        completed_worlds != list(world_uids)
        or len(completed_worlds) != EXPECTED_WORLD_COUNT
        or row_count != EXPECTED_CANDIDATE_COUNT
        or len(seen_pairs) != EXPECTED_CANDIDATE_COUNT
    ):
        raise AuditError("Candidate table is incomplete or reordered")
    return {
        "world_count": len(completed_worlds),
        "candidate_pair_count": row_count,
        "world_uid_set_and_order_exact": True,
        "world_blocks_contiguous_and_exact": True,
        "pairs_per_world_exact": EXPECTED_CANDIDATES_PER_WORLD,
        "pair_uids_globally_unique": True,
        "pair_endpoint_formula_exact": True,
        "independent_selected_global_rank_exact": True,
        "labels_qrels_controller_membership_or_identity_assets_parsed": False,
    }


def manifest_file_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise AuditError("Split manifest has no file list")
    by_path: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "model_mount_allowed",
        "path",
        "sha256",
        "size_bytes",
    }
    for source in records:
        if not isinstance(source, dict) or set(source) != expected_fields:
            raise AuditError("Split manifest file-record schema drift")
        relative = validate_relative_member_path(source["path"])
        if relative in by_path:
            raise AuditError("Duplicate split manifest member")
        if (
            type(source["model_mount_allowed"]) is not bool
            or type(source["size_bytes"]) is not int
            or source["size_bytes"] < 0
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
        ):
            raise AuditError("Malformed split manifest file record")
        by_path[relative] = dict(source)
    return by_path


def validate_manifest_members(
    split_root: Path,
    *,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    actual: set[str] = set()
    for path in split_root.rglob("*"):
        if path.is_dir():
            if path.is_symlink() or is_reparse_point(path):
                raise AuditError(
                    f"Release-tree directory is a link: {path}"
                )
            continue
        require_plain_file(path, label="Release-tree member")
        relative = path.relative_to(split_root).as_posix()
        if relative != "split_manifest.json":
            actual.add(relative)
    if actual != set(records):
        missing = sorted(set(records) - actual)
        extra = sorted(actual - set(records))
        raise AuditError(
            f"Split member set drift: missing={missing}, extra={extra}"
        )
    total_bytes = 0
    model_mount_allowed_count = 0
    for relative in sorted(records):
        record = records[relative]
        path = split_root / PurePosixPath(relative)
        observed_size = path.stat().st_size
        observed_hash = sha256_file(path)
        if (
            observed_size != record["size_bytes"]
            or observed_hash != record["sha256"]
        ):
            raise AuditError(f"Split member byte drift: {relative}")
        total_bytes += observed_size
        model_mount_allowed_count += int(
            record["model_mount_allowed"]
        )
    return {
        "member_count": len(records),
        "total_member_bytes": total_bytes,
        "member_set_exact": True,
        "all_member_sizes_and_sha256_exact": True,
        "model_mount_allowed_member_count": model_mount_allowed_count,
    }


def validate_split(
    *,
    dataset: Path,
    split: str,
    release_receipt: Mapping[str, Any],
    candidate_key_hex: str,
) -> dict[str, Any]:
    split_root = dataset / split
    require_plain_directory(split_root, label=f"{split} directory")
    manifest_path = split_root / "split_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if (
        manifest_hash != EXPECTED_SPLIT_MANIFEST_SHA256[split]
        or release_receipt.get("manifest_sha256") != manifest_hash
    ):
        raise AuditError(f"{split} manifest file hash drift")
    manifest = load_json(manifest_path)
    manifest_self = validate_canonical_self_hash(
        manifest,
        label=f"{split} manifest",
    )
    if (
        manifest.get("status") != "PASS_SPLIT_DATASET_READY"
        or manifest.get("run_id") != EXPECTED_RUN_ID
        or manifest.get("split") != split
        or manifest.get("world_count") != EXPECTED_WORLD_COUNT
        or manifest.get("candidate_pair_count")
        != EXPECTED_CANDIDATE_COUNT
        or release_receipt.get("manifest_self_sha256")
        != manifest_self
        or manifest.get("candidate_output_order_audit")
        != release_receipt.get("candidate_output_order_replay")
    ):
        raise AuditError(f"{split} manifest semantics drift")
    records = manifest_file_records(manifest)
    if len(records) != EXPECTED_MEMBER_COUNTS[split]:
        raise AuditError(f"{split} manifest member count drift")
    member_audit = validate_manifest_members(
        split_root,
        records=records,
    )
    world_uids = read_world_uids(
        split_root / "observed" / "worlds.csv"
    )
    candidate_audit = validate_candidate_rows(
        split_root / "observed" / "candidate_pairs.csv",
        world_uids=world_uids,
        candidate_key_hex=candidate_key_hex,
    )
    expected_order_receipt = {
        "world_count": EXPECTED_WORLD_COUNT,
        "candidate_pair_count": EXPECTED_CANDIDATE_COUNT,
        "world_blocks_contiguous_and_exact": True,
        "independent_selected_global_rank_exact": True,
        "labels_or_controller_membership_read": False,
    }
    if (
        manifest.get("candidate_output_order_audit")
        != expected_order_receipt
        or release_receipt.get("candidate_output_order_replay")
        != expected_order_receipt
    ):
        raise AuditError(f"{split} producer/finalizer order receipt drift")
    return {
        "split_manifest_sha256": manifest_hash,
        "split_manifest_self_sha256": manifest_self,
        "manifest_receipt_exact": True,
        "manifest_members": member_audit,
        "candidate_output_order_independent_replay": candidate_audit,
    }


def require_output_outside_dataset(
    *,
    dataset: Path,
    output: Path,
) -> None:
    try:
        output.resolve().relative_to(dataset.resolve())
    except ValueError:
        return
    raise AuditError("Audit output must remain outside the formal dataset")


def validate_top_level_member_set(dataset: Path) -> None:
    require_plain_directory(dataset, label="Formal dataset")
    observed = {path.name for path in dataset.iterdir()}
    if observed != EXPECTED_TOP_LEVEL:
        raise AuditError("Formal release top-level member set drift")


def build_report(dataset: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    validate_top_level_member_set(dataset)
    release_path = dataset / "release_manifest.json"
    release_hash = sha256_file(release_path)
    if release_hash != EXPECTED_RELEASE_SHA256:
        raise AuditError("Formal release-manifest file hash drift")
    release = load_json(release_path)
    release_self = validate_canonical_self_hash(
        release,
        label="Release manifest",
    )
    if (
        release_self != EXPECTED_RELEASE_SELF_SHA256
        or release.get("status") != EXPECTED_RELEASE_STATUS
        or release.get("run_id") != EXPECTED_RUN_ID
        or release.get("all_candidate_output_order_replays_exact")
        is not True
        or release.get("parent_order_only_repair_equivalence_exact")
        is not True
        or release.get("fixed_holdout_bytes_ready") is not True
        or release.get("blind_custody_attested") is not False
        or set(release.get("split_receipts", {}))
        != set(EXPECTED_SPLITS)
    ):
        raise AuditError("Formal release-manifest semantics drift")

    base_policy_receipt = release.get("base_policy")
    if not isinstance(base_policy_receipt, dict):
        raise AuditError("Release has no base-policy receipt")
    if (
        base_policy_receipt.get("path")
        != "schema/step28_v13_synthetic_chinese_dataset_policy.json"
    ):
        raise AuditError("Base-policy path drift")
    base_policy_path = ROOT / base_policy_receipt["path"]
    base_policy_hash = sha256_file(base_policy_path)
    if base_policy_hash != base_policy_receipt.get("sha256"):
        raise AuditError("Base-policy hash drift")
    base_policy = load_json(base_policy_path)
    try:
        candidate_key_hex = str(
            base_policy["randomness"]["formal"]["candidate_key_hex"]
        )
    except (KeyError, TypeError) as exc:
        raise AuditError("Base-policy candidate key is unavailable") from exc

    equivalence_receipt = release.get("repair_equivalence_report")
    if not isinstance(equivalence_receipt, dict):
        raise AuditError("Release has no equivalence receipt")
    equivalence_path = dataset / "repair_equivalence_report.json"
    equivalence_hash = sha256_file(equivalence_path)
    equivalence = load_json(equivalence_path)
    equivalence_self = validate_canonical_self_hash(
        equivalence,
        label="Repair-equivalence report",
    )
    if (
        equivalence_hash != equivalence_receipt.get("sha256")
        or equivalence_self
        != equivalence_receipt.get("canonical_self_hash")
        or equivalence.get("status")
        != "PASS_C40_OUTPUT_ORDER_ONLY_REPAIR_EQUIVALENCE"
    ):
        raise AuditError("Repair-equivalence receipt drift")

    per_split = {
        split: validate_split(
            dataset=dataset,
            split=split,
            release_receipt=release["split_receipts"][split],
            candidate_key_hex=candidate_key_hex,
        )
        for split in EXPECTED_SPLITS
    }
    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "version": VERSION,
        "status": "PASS_INDEPENDENT_FORMAL_RELEASE_TREE_HARDENING_AUDIT",
        "dataset_root": dataset.relative_to(ROOT).as_posix(),
        "producer": {
            "path": script_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(script_path),
            "imports_dataset_builder_preflight_finalizer_or_row_audit": False,
        },
        "formal_release": {
            "path": release_path.relative_to(ROOT).as_posix(),
            "sha256": release_hash,
            "canonical_self_hash": release_self,
            "status": release["status"],
            "blind_custody_attested": release[
                "blind_custody_attested"
            ],
        },
        "base_policy": {
            "path": base_policy_path.relative_to(ROOT).as_posix(),
            "sha256": base_policy_hash,
        },
        "repair_equivalence_receipt": {
            "path": equivalence_path.relative_to(ROOT).as_posix(),
            "sha256": equivalence_hash,
            "canonical_self_hash": equivalence_self,
            "historical_parent_bytes_reparsed": False,
        },
        "top_level_member_set_exact": True,
        "per_split": per_split,
        "all_split_manifest_receipts_exact": True,
        "all_manifest_member_sets_sizes_and_hashes_exact": True,
        "all_2000_worlds_present_in_frozen_order": True,
        "all_80000_candidate_rows_in_contiguous_40_row_world_blocks": True,
        "all_candidate_output_orders_independently_replayed": True,
        "explicit_boundary": {
            "all_release_member_bytes_hashed": True,
            "labels_parsed": False,
            "qrels_parsed": False,
            "controller_membership_parsed": False,
            "identity_assets_parsed": False,
            "m1_mappings_parsed": False,
            "parent_v1_dataset_bytes_available_or_reparsed": False,
            "claim": (
                "This hardening audit binds the current v1.2 release tree "
                "and independently replays its complete C40 serialization "
                "order. It does not repeat label, qrel, M1, or historical "
                "parent/child semantic-equivalence proofs."
            ),
        },
    }
    report["canonical_self_hash"] = canonical_sha256(report)
    return report


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise AuditError("Existing hardening report has different bytes")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise AuditError("Stale hardening-report temporary file exists")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "step28_synthetic_chinese_dataset"
            / EXPECTED_RUN_ID
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "step28_synthetic_chinese_dataset"
            / "post_release_audits"
            / (
                "formal_order_repair_v1_2_release_tree_"
                "hardening_audit_v1_20260731.json"
            )
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_output_outside_dataset(
        dataset=args.dataset,
        output=args.output,
    )
    report = build_report(args.dataset)
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output.relative_to(ROOT).as_posix(),
                "canonical_self_hash": report["canonical_self_hash"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
