#!/usr/bin/env python3
"""Independently validate sealed labels against controller membership."""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as dataset_common
import step28_v13_metadata_shortcut_common as shortcut_common


RECEIPT_VERSION = shortcut_common.LABEL_FORMULA_RECEIPT_VERSION


def validate_formula(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    expected_world_count: int,
    expected_pairs_per_world: int = 40,
    expected_sellers_per_world: int = 28,
) -> dict[str, Any]:
    """Validate via an independently constructed positive-pair set."""

    if (
        expected_world_count < 1
        or expected_pairs_per_world != 40
        or expected_sellers_per_world != 28
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula-validator cardinality contract drift"
        )

    sellers_by_controller: dict[
        tuple[str, str], list[str]
    ] = defaultdict(list)
    membership_keys: set[tuple[str, str]] = set()
    membership_world_counts: Counter[str] = Counter()
    for row in membership_rows:
        if tuple(row) != shortcut_common.MEMBERSHIP_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Formula membership schema/order drift"
            )
        world_uid = str(row["world_uid"])
        controller_uid = str(row["controller_uid"])
        seller_uid = str(row["seller_uid"])
        membership_key = (world_uid, seller_uid)
        if (
            not world_uid
            or not controller_uid
            or not seller_uid
            or membership_key in membership_keys
        ):
            raise shortcut_common.ShortcutAuditError(
                "Formula membership key contract failed"
            )
        membership_keys.add(membership_key)
        membership_world_counts[world_uid] += 1
        sellers_by_controller[(world_uid, controller_uid)].append(
            seller_uid
        )
    if (
        len(membership_world_counts) != expected_world_count
        or set(membership_world_counts.values())
        != {expected_sellers_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula membership cardinality drift"
        )
    controller_sizes_by_world: dict[str, list[int]] = defaultdict(list)
    for (world_uid, _controller_uid), seller_uids in (
        sellers_by_controller.items()
    ):
        controller_sizes_by_world[world_uid].append(
            len(seller_uids)
        )
    expected_controller_sizes = [2] * 8 + [3] * 4
    if any(
        sorted(controller_sizes_by_world[world_uid])
        != expected_controller_sizes
        for world_uid in membership_world_counts
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula controller partition is not 8 dyads plus 4 triads"
        )

    positive_pairs: set[tuple[str, str]] = set()
    for (world_uid, _controller_uid), seller_uids in sorted(
        sellers_by_controller.items(),
        key=lambda item: (
            item[0][0].encode("utf-8"),
            item[0][1].encode("utf-8"),
        ),
    ):
        for left, right in itertools.combinations(
            sorted(seller_uids, key=lambda value: value.encode("utf-8")),
            2,
        ):
            positive_pairs.add(
                (
                    world_uid,
                    dataset_common.canonical_pair_uid(left, right),
                )
            )

    candidate_world_by_pair: dict[str, str] = {}
    candidate_world_counts: Counter[str] = Counter()
    for row in candidate_rows:
        if tuple(row) != shortcut_common.CANDIDATE_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Formula candidate schema/order drift"
            )
        pair_uid = str(row["canonical_pair_uid"])
        world_uid = str(row["world_uid"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        if (
            not world_uid
            or not left
            or not right
            or left == right
            or pair_uid
            != dataset_common.canonical_pair_uid(left, right)
            or pair_uid in candidate_world_by_pair
            or (world_uid, left) not in membership_keys
            or (world_uid, right) not in membership_keys
        ):
            raise shortcut_common.ShortcutAuditError(
                "Formula candidate key contract failed"
            )
        candidate_world_by_pair[pair_uid] = world_uid
        candidate_world_counts[world_uid] += 1
    if (
        set(candidate_world_counts) != set(membership_world_counts)
        or set(candidate_world_counts.values())
        != {expected_pairs_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Formula candidate C40 cardinality drift"
        )

    observed_labels: dict[str, int] = {}
    for row in label_rows:
        if tuple(row) != shortcut_common.LABEL_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Formula sealed-label schema/order drift"
            )
        pair_uid = str(row["canonical_pair_uid"])
        value = str(row["label"])
        if (
            pair_uid in observed_labels
            or pair_uid not in candidate_world_by_pair
            or value not in {"0", "1"}
        ):
            raise shortcut_common.ShortcutAuditError(
                "Formula sealed-label key/value contract failed"
            )
        observed_labels[pair_uid] = int(value)
    if set(observed_labels) != set(candidate_world_by_pair):
        raise shortcut_common.ShortcutAuditError(
            "Formula sealed-label and C40 keysets differ"
        )

    for pair_uid, world_uid in candidate_world_by_pair.items():
        expected = int((world_uid, pair_uid) in positive_pairs)
        if observed_labels[pair_uid] != expected:
            raise shortcut_common.ShortcutAuditError(
                "Sealed label disagrees with independent formula"
            )
    return {
        "validated": True,
        "row_count": len(observed_labels),
        "world_count": len(candidate_world_counts),
        "rows_per_world": expected_pairs_per_world,
        "class_counts_withheld": True,
        "alternative_derivation": (
            shortcut_common.LABEL_FORMULA_ALTERNATIVE_DERIVATION
        ),
    }


def write_validation_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    candidate_path: Path,
    membership_path: Path,
    labels_path: Path,
    label_manifest_path: Path,
    output_dir: Path,
) -> Path:
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=split,
        operation="label_formula_validation",
    )
    expected_basenames = lock["label_formula_validator"][
        "input_basenames"
    ]
    paths = {
        "candidate_pairs": candidate_path,
        "controller_membership": membership_path,
        "labels": labels_path,
        "label_manifest": label_manifest_path,
    }
    if {
        key: path.name for key, path in paths.items()
    } != expected_basenames:
        raise shortcut_common.ShortcutAuditError(
            "Formula-validator input basename allow-list drift"
        )
    candidate_rows, candidate_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            candidate_path,
            fieldnames=shortcut_common.CANDIDATE_FIELDS,
        )
    )
    membership_rows, membership_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            membership_path,
            fieldnames=shortcut_common.MEMBERSHIP_FIELDS,
        )
    )
    label_rows, label_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            labels_path,
            fieldnames=shortcut_common.LABEL_FIELDS,
        )
    )
    label_manifest, label_manifest_snapshot = (
        shortcut_common.load_json_snapshot(label_manifest_path)
    )
    if not isinstance(label_manifest, dict):
        raise shortcut_common.ShortcutAuditError(
            "Label manifest is not an object"
        )
    shortcut_common.validate_label_manifest_release(
        lock=lock,
        lock_path=lock_path,
        split=split,
        labels_path=labels_path,
        label_rows=label_rows,
        label_snapshot=label_snapshot,
        manifest_path=label_manifest_path,
        manifest=label_manifest,
        manifest_snapshot=label_manifest_snapshot,
    )
    result = validate_formula(
        candidate_rows=candidate_rows,
        membership_rows=membership_rows,
        label_rows=label_rows,
        expected_world_count=int(
            lock["formal_world_counts"][split]
        ),
    )
    input_snapshots = {
        "candidate_pairs": candidate_snapshot,
        "controller_membership": membership_snapshot,
        "sealed_labels": label_snapshot,
        "label_manifest": label_manifest_snapshot,
    }
    parent_manifests = [
        {
            "role": "classification_label_manifest",
            "file_sha256": label_manifest_snapshot.sha256,
            "content_sha256": label_manifest[
                "canonical_self_hash"
            ],
        }
    ]

    def writer(stage: Path) -> None:
        input_rows = []
        for role in (
            "candidate_pairs",
            "controller_membership",
            "sealed_labels",
            "label_manifest",
        ):
            input_rows.append(
                input_snapshots[role].record(role=role)
            )
        receipt = shortcut_common.add_self_hash(
            {
                "version": RECEIPT_VERSION,
                "status": "PASS_LABEL_FORMULA_ONLY",
                "mode": "formal",
                "split": split,
                **result,
                **shortcut_common.manifest_identity(
                    lock,
                    lock_path=lock_path,
                    stage="validate_label_formula",
                    producer_relative_path=(
                        "scripts/"
                        "step28_v13_validate_label_formula.py"
                    ),
                    additional_parent_manifests=parent_manifests,
                ),
                "input_allowlist": input_rows,
                "access_isolation_status": (
                    shortcut_common.BLOCKED_ACCESS_STATUS
                ),
                "forbidden_open_count_not_self_asserted": True,
            }
        )
        dataset_common.write_json(
            stage / shortcut_common.LABEL_VALIDATION_FILENAME,
            receipt,
        )

    return shortcut_common.publish_directory(
        output_dir,
        writer=writer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=shortcut_common.DEFAULT_LOCK_PATH,
    )
    parser.add_argument(
        "--split",
        choices=shortcut_common.SPLITS,
        required=True,
    )
    parser.add_argument("--candidate-pairs", type=Path)
    parser.add_argument("--controller-membership", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = shortcut_common.load_lock(args.lock)
    if args.validate_config_only:
        print(
            "Step28-v13 label-formula validator is locked; "
            "formal execution remains blocked"
        )
        return
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=args.split,
        operation="label_formula_validation",
    )
    required = (
        args.candidate_pairs,
        args.controller_membership,
        args.labels,
        args.label_manifest,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise shortcut_common.ShortcutAuditError(
            "Formula validation requires every exact input and output"
        )
    try:
        write_validation_release(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            candidate_path=args.candidate_pairs,
            membership_path=args.controller_membership,
            labels_path=args.labels,
            label_manifest_path=args.label_manifest,
            output_dir=args.output_dir,
        )
    except Exception as error:
        shortcut_common.publish_stage_failure(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            stage="validate_label_formula_failure",
            producer_relative_path=(
                "scripts/step28_v13_validate_label_formula.py"
            ),
            output_dir=args.output_dir,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
