#!/usr/bin/env python3
"""Seal Step 28-v13 pair labels from controller membership only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as dataset_common
import step28_v13_metadata_shortcut_common as shortcut_common


MANIFEST_VERSION = shortcut_common.LABEL_MANIFEST_VERSION


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def build_labels(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    expected_world_count: int,
    expected_pairs_per_world: int = 40,
    expected_sellers_per_world: int = 28,
) -> list[dict[str, str]]:
    """Apply only controller equality to the exact C40 keyset."""

    if (
        expected_world_count < 1
        or expected_pairs_per_world != 40
        or expected_sellers_per_world != 28
    ):
        raise shortcut_common.ShortcutAuditError(
            "Label-sealer cardinality contract drift"
        )

    controller_by_seller: dict[tuple[str, str], str] = {}
    membership_world_counts: Counter[str] = Counter()
    controller_sizes: Counter[tuple[str, str]] = Counter()
    for source in membership_rows:
        if tuple(source) != shortcut_common.MEMBERSHIP_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Controller-membership schema/order drift"
            )
        world_uid = str(source["world_uid"])
        controller_uid = str(source["controller_uid"])
        seller_uid = str(source["seller_uid"])
        key = (world_uid, seller_uid)
        if (
            not world_uid
            or not controller_uid
            or not seller_uid
            or key in controller_by_seller
        ):
            raise shortcut_common.ShortcutAuditError(
                "Controller-membership key contract failed"
            )
        controller_by_seller[key] = controller_uid
        membership_world_counts[world_uid] += 1
        controller_sizes[(world_uid, controller_uid)] += 1
    if (
        len(membership_world_counts) != expected_world_count
        or set(membership_world_counts.values())
        != {expected_sellers_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Controller-membership world cardinality drift"
        )
    sizes_by_world: dict[str, list[int]] = defaultdict(list)
    for (world_uid, _controller_uid), size in controller_sizes.items():
        sizes_by_world[world_uid].append(size)
    expected_sizes = [2] * 8 + [3] * 4
    if any(
        sorted(sizes_by_world[world_uid]) != expected_sizes
        for world_uid in membership_world_counts
    ):
        raise shortcut_common.ShortcutAuditError(
            "Controller partition is not exact 8 dyads plus 4 triads"
        )

    candidates: list[dict[str, str]] = []
    pair_keys: set[str] = set()
    candidate_world_counts: Counter[str] = Counter()
    for source in candidate_rows:
        if tuple(source) != shortcut_common.CANDIDATE_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Candidate row schema/order drift"
            )
        row = {key: str(source[key]) for key in source}
        pair_uid = row["canonical_pair_uid"]
        world_uid = row["world_uid"]
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        if (
            not world_uid
            or not left
            or not right
            or left == right
            or pair_uid
            != dataset_common.canonical_pair_uid(left, right)
            or pair_uid in pair_keys
            or (world_uid, left) not in controller_by_seller
            or (world_uid, right) not in controller_by_seller
        ):
            raise shortcut_common.ShortcutAuditError(
                "Candidate endpoint or pair key contract failed"
            )
        pair_keys.add(pair_uid)
        candidate_world_counts[world_uid] += 1
        candidates.append(row)
    if (
        set(candidate_world_counts) != set(membership_world_counts)
        or set(candidate_world_counts.values())
        != {expected_pairs_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Candidate labels are not exact C40 on the membership worlds"
        )

    output: list[dict[str, str]] = []
    for row in sorted(
        candidates,
        key=lambda value: (
            _utf8_key(value["world_uid"]),
            _utf8_key(value["canonical_pair_uid"]),
        ),
    ):
        world_uid = row["world_uid"]
        left_controller = controller_by_seller[
            (world_uid, row["seller_uid_left"])
        ]
        right_controller = controller_by_seller[
            (world_uid, row["seller_uid_right"])
        ]
        output.append(
            {
                "canonical_pair_uid": row["canonical_pair_uid"],
                "label": str(int(left_controller == right_controller)),
            }
        )
    if len(output) != expected_world_count * expected_pairs_per_world:
        raise shortcut_common.ShortcutAuditError(
            "Sealed label row count drift"
        )
    return output


def seal_from_files(
    *,
    lock: Mapping[str, Any],
    split: str,
    candidate_path: Path,
    membership_path: Path,
) -> list[dict[str, str]]:
    rows, _snapshots = _seal_from_files_with_snapshots(
        lock=lock,
        split=split,
        candidate_path=candidate_path,
        membership_path=membership_path,
    )
    return rows


def _seal_from_files_with_snapshots(
    *,
    lock: Mapping[str, Any],
    split: str,
    candidate_path: Path,
    membership_path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[str, shortcut_common.FileSnapshot],
]:
    paths = {
        "candidate_pairs": candidate_path,
        "controller_membership": membership_path,
    }
    if {
        key: path.name for key, path in paths.items()
    } != lock["label_sealer"]["input_basenames"]:
        raise shortcut_common.ShortcutAuditError(
            "Label-sealer input basename allow-list drift"
        )
    candidates, candidate_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            candidate_path,
            fieldnames=shortcut_common.CANDIDATE_FIELDS,
        )
    )
    memberships, membership_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            membership_path,
            fieldnames=shortcut_common.MEMBERSHIP_FIELDS,
        )
    )
    rows = build_labels(
        candidate_rows=candidates,
        membership_rows=memberships,
        expected_world_count=int(
            lock["formal_world_counts"][split]
        ),
    )
    return rows, {
        "candidate_pairs": candidate_snapshot,
        "controller_membership": membership_snapshot,
    }


def write_label_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    candidate_path: Path,
    membership_path: Path,
    output_dir: Path,
) -> Path:
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=split,
        operation="label_sealing",
    )
    rows, input_snapshots = _seal_from_files_with_snapshots(
        lock=lock,
        split=split,
        candidate_path=candidate_path,
        membership_path=membership_path,
    )

    def writer(stage: Path) -> None:
        label_path = stage / shortcut_common.LABEL_FILENAME
        dataset_common.write_csv(
            label_path,
            rows,
            shortcut_common.LABEL_FIELDS,
        )
        manifest = shortcut_common.add_self_hash(
            {
                "version": MANIFEST_VERSION,
                "status": "SEALED_PRIVATE_CLASSIFICATION_LABELS",
                "mode": "formal",
                "split": split,
                "row_count": len(rows),
                "world_count": int(
                    lock["formal_world_counts"][split]
                ),
                "rows_per_world": 40,
                "label_schema": list(shortcut_common.LABEL_FIELDS),
                "formula": (
                    "int(controller(left)==controller(right))"
                ),
                "formula_equality_required": True,
                "class_counts_withheld": True,
                "label_content_sha256": (
                    dataset_common.canonical_sha256(rows)
                ),
                **shortcut_common.manifest_identity(
                    lock,
                    lock_path=lock_path,
                    stage="seal_classification_labels",
                    producer_relative_path=(
                        "scripts/"
                        "step28_v13_seal_classification_labels.py"
                    ),
                ),
                "input_allowlist": [
                    {
                        **input_snapshots[role].record(role=role),
                    }
                    for role in (
                        "candidate_pairs",
                        "controller_membership",
                    )
                ],
                "access_isolation_status": (
                    shortcut_common.BLOCKED_ACCESS_STATUS
                ),
                "forbidden_open_count_not_self_asserted": True,
                "files": [
                    shortcut_common.file_record(
                        label_path,
                        role="private_classification_labels",
                        root=stage,
                    )
                ],
            }
        )
        dataset_common.write_json(
            stage / shortcut_common.LABEL_MANIFEST_FILENAME,
            manifest,
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = shortcut_common.load_lock(args.lock)
    if args.validate_config_only:
        print(
            "Step28-v13 label-sealer configuration is locked; "
            "formal execution remains blocked"
        )
        return
    shortcut_common.require_formal_execution_envelope(lock)
    shortcut_common.require_split_supervision_authorization(
        lock,
        split=args.split,
        operation="label_sealing",
    )
    required = (
        args.candidate_pairs,
        args.controller_membership,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise shortcut_common.ShortcutAuditError(
            "Label sealing requires every exact input and output"
        )
    try:
        write_label_release(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            candidate_path=args.candidate_pairs,
            membership_path=args.controller_membership,
            output_dir=args.output_dir,
        )
    except Exception as error:
        shortcut_common.publish_stage_failure(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            stage="seal_classification_labels_failure",
            producer_relative_path=(
                "scripts/"
                "step28_v13_seal_classification_labels.py"
            ),
            output_dir=args.output_dir,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
