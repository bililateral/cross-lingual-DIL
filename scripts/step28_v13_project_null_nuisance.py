#!/usr/bin/env python3
"""Project the frozen 14 null-nuisance pair features without labels."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as dataset_common
import step28_v13_metadata_shortcut_common as shortcut_common


MANIFEST_VERSION = shortcut_common.PROJECTION_MANIFEST_VERSION


def _utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def build_projection(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
    history_item_rows: Sequence[Mapping[str, Any]],
    expected_world_count: int,
    expected_pairs_per_world: int = 40,
) -> list[dict[str, str]]:
    """Build the exact label-free projection in deterministic order."""

    if expected_world_count < 1 or expected_pairs_per_world != 40:
        raise shortcut_common.ShortcutAuditError(
            "Projection world or C40 count contract drift"
        )
    candidates: list[dict[str, str]] = []
    pair_keys: set[str] = set()
    pair_world_counts: Counter[str] = Counter()
    candidate_sellers: set[tuple[str, str]] = set()
    for source in candidate_rows:
        if tuple(source) != shortcut_common.CANDIDATE_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Candidate row schema/order drift"
            )
        row = {key: str(source[key]) for key in source}
        world_uid = row["world_uid"]
        left = row["seller_uid_left"]
        right = row["seller_uid_right"]
        pair_uid = row["canonical_pair_uid"]
        if (
            not world_uid
            or not left
            or not right
            or left == right
            or pair_uid
            != dataset_common.canonical_pair_uid(left, right)
            or pair_uid in pair_keys
        ):
            raise shortcut_common.ShortcutAuditError(
                "Candidate endpoint or pair key contract failed"
            )
        pair_keys.add(pair_uid)
        pair_world_counts[world_uid] += 1
        candidate_sellers.add((world_uid, left))
        candidate_sellers.add((world_uid, right))
        candidates.append(row)
    if (
        len(pair_world_counts) != expected_world_count
        or set(pair_world_counts.values()) != {expected_pairs_per_world}
    ):
        raise shortcut_common.ShortcutAuditError(
            "Candidate projection is not exact C40 per world"
        )
    candidate_worlds = set(pair_world_counts)

    item_index: dict[str, tuple[str, str, int]] = {}
    history_sellers_by_world: dict[str, set[str]] = defaultdict(set)
    history_item_counts: Counter[tuple[str, str]] = Counter()
    for source in history_item_rows:
        if tuple(source) != shortcut_common.HISTORY_ITEM_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "History item-index schema/order drift"
            )
        item_uid = str(source["item_uid"])
        world_uid = str(source["world_uid"])
        seller_uid = str(source["seller_uid"])
        try:
            time_bucket = int(source["time_bucket"])
        except (TypeError, ValueError) as error:
            raise shortcut_common.ShortcutAuditError(
                "History item time bucket is not an integer"
            ) from error
        if (
            not item_uid
            or not world_uid
            or not seller_uid
            or item_uid in item_index
            or not 0 <= time_bucket <= 3
            or str(source["time_bucket"]) != str(time_bucket)
        ):
            raise shortcut_common.ShortcutAuditError(
                "History item-index key or time bucket failed"
            )
        item_index[item_uid] = (
            world_uid,
            seller_uid,
            time_bucket,
        )
        history_sellers_by_world[world_uid].add(seller_uid)
        history_item_counts[(world_uid, seller_uid)] += 1
    if set(history_sellers_by_world) != candidate_worlds:
        raise shortcut_common.ShortcutAuditError(
            "History item world set differs from C40 world set"
        )
    if any(
        len(sellers) != 28
        for sellers in history_sellers_by_world.values()
    ):
        raise shortcut_common.ShortcutAuditError(
            "History item index is not exact 28 sellers per world"
        )
    if any(
        count < 2 or count > 8
        for count in history_item_counts.values()
    ):
        raise shortcut_common.ShortcutAuditError(
            "Seller item count is outside the frozen [2,8] range"
        )

    seller_accumulator: dict[
        tuple[str, str], dict[str, Any]
    ] = defaultdict(
        lambda: {
            "item_count": 0,
            "title_missing": 0,
            "description_missing": 0,
            "time_buckets": [0, 0, 0, 0],
        }
    )
    observed_items: set[str] = set()
    for source in redacted_items:
        if set(source) != shortcut_common.REDACTED_ITEM_FIELDS:
            raise shortcut_common.ShortcutAuditError(
                "Redacted item schema drift"
            )
        if any(
            not isinstance(source[key], str)
            for key in shortcut_common.REDACTED_ITEM_FIELDS
        ):
            raise shortcut_common.ShortcutAuditError(
                "Redacted item contains a non-string field"
            )
        item_uid = source["item_uid"]
        world_uid = source["world_uid"]
        seller_uid = source["seller_uid"]
        if (
            not item_uid
            or not world_uid
            or not seller_uid
            or item_uid in observed_items
            or item_uid not in item_index
            or item_index[item_uid][:2]
            != (world_uid, seller_uid)
        ):
            raise shortcut_common.ShortcutAuditError(
                "Redacted/history item keyset or foreign key drift"
            )
        observed_items.add(item_uid)
        time_bucket = item_index[item_uid][2]
        accumulator = seller_accumulator[(world_uid, seller_uid)]
        accumulator["item_count"] += 1
        accumulator["title_missing"] += int(source["title"] == "")
        accumulator["description_missing"] += int(
            source["description"] == ""
        )
        accumulator["time_buckets"][time_bucket] += 1
    if observed_items != set(item_index):
        raise shortcut_common.ShortcutAuditError(
            "Redacted/history item keysets differ"
        )
    if set(seller_accumulator) != set(history_item_counts):
        raise shortcut_common.ShortcutAuditError(
            "Redacted/history seller keysets differ"
        )
    if not candidate_sellers <= set(seller_accumulator):
        raise shortcut_common.ShortcutAuditError(
            "A candidate endpoint lacks observed items"
        )

    seller_vectors: dict[tuple[str, str], tuple[float, ...]] = {}
    for key, accumulator in seller_accumulator.items():
        count = int(accumulator["item_count"])
        if count <= 0:
            raise shortcut_common.ShortcutAuditError(
                "Seller item count must be positive"
            )
        vector = (
            float(count),
            float(accumulator["title_missing"]) / count,
            float(accumulator["description_missing"]) / count,
            *(
                float(bucket_count) / count
                for bucket_count in accumulator["time_buckets"]
            ),
        )
        if len(vector) != 7 or not all(
            math.isfinite(value) for value in vector
        ):
            raise shortcut_common.ShortcutAuditError(
                "Seller nuisance vector is invalid"
            )
        seller_vectors[key] = vector

    output: list[dict[str, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            _utf8_key(row["world_uid"]),
            _utf8_key(row["canonical_pair_uid"]),
        ),
    ):
        world_uid = candidate["world_uid"]
        left = seller_vectors[
            (world_uid, candidate["seller_uid_left"])
        ]
        right = seller_vectors[
            (world_uid, candidate["seller_uid_right"])
        ]
        values = tuple(
            abs(left[index] - right[index])
            for index in range(7)
        ) + tuple(
            left[index] + right[index] for index in range(7)
        )
        if not all(math.isfinite(value) for value in values):
            raise shortcut_common.ShortcutAuditError(
                "Pair nuisance vector is nonfinite"
            )
        output.append(
            {
                "canonical_pair_uid": candidate[
                    "canonical_pair_uid"
                ],
                "world_uid": world_uid,
                **{
                    name: format(value, ".12f")
                    for name, value in zip(
                        shortcut_common.PAIR_FEATURES,
                        values,
                        strict=True,
                    )
                },
            }
        )
    if len(output) != expected_world_count * expected_pairs_per_world:
        raise shortcut_common.ShortcutAuditError(
            "Projected row count drift"
        )
    return output


def project_from_files(
    *,
    lock: Mapping[str, Any],
    split: str,
    candidate_path: Path,
    redacted_path: Path,
    history_item_path: Path,
) -> list[dict[str, str]]:
    rows, _snapshots = _project_from_files_with_snapshots(
        lock=lock,
        split=split,
        candidate_path=candidate_path,
        redacted_path=redacted_path,
        history_item_path=history_item_path,
    )
    return rows


def _project_from_files_with_snapshots(
    *,
    lock: Mapping[str, Any],
    split: str,
    candidate_path: Path,
    redacted_path: Path,
    history_item_path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[str, shortcut_common.FileSnapshot],
]:
    expected_basenames = lock["projector"]["input_basenames"]
    paths = {
        "candidate_pairs": candidate_path,
        "redacted_items": redacted_path,
        "history_item_index": history_item_path,
    }
    if {
        key: path.name for key, path in paths.items()
    } != expected_basenames:
        raise shortcut_common.ShortcutAuditError(
            "Projector input basename allow-list drift"
        )
    candidates, candidate_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            candidate_path,
            fieldnames=shortcut_common.CANDIDATE_FIELDS,
        )
    )
    redacted, redacted_snapshot = (
        shortcut_common.read_jsonl_exact_snapshot(
            redacted_path,
            keys=shortcut_common.REDACTED_ITEM_FIELDS,
        )
    )
    history_items, history_snapshot = (
        shortcut_common.read_csv_exact_snapshot(
            history_item_path,
            fieldnames=shortcut_common.HISTORY_ITEM_FIELDS,
        )
    )
    rows = build_projection(
        candidate_rows=candidates,
        redacted_items=redacted,
        history_item_rows=history_items,
        expected_world_count=int(
            lock["formal_world_counts"][split]
        ),
    )
    return rows, {
        "candidate_pairs": candidate_snapshot,
        "history_item_index": history_snapshot,
        "redacted_items": redacted_snapshot,
    }


def write_projection_release(
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    split: str,
    candidate_path: Path,
    redacted_path: Path,
    history_item_path: Path,
    output_dir: Path,
) -> Path:
    shortcut_common.require_formal_execution_envelope(lock)
    rows, input_snapshots = _project_from_files_with_snapshots(
        lock=lock,
        split=split,
        candidate_path=candidate_path,
        redacted_path=redacted_path,
        history_item_path=history_item_path,
    )

    def writer(stage: Path) -> None:
        projection_path = (
            stage / shortcut_common.PROJECTION_FILENAME
        )
        dataset_common.write_csv(
            projection_path,
            rows,
            shortcut_common.PROJECTION_FIELDS,
        )
        world_count = len({row["world_uid"] for row in rows})
        manifest = shortcut_common.add_self_hash(
            {
                "version": MANIFEST_VERSION,
                "status": "SEALED_LABEL_FREE_PROJECTION",
                "mode": "formal",
                "split": split,
                "row_count": len(rows),
                "world_count": world_count,
                "rows_per_world": 40,
                "projection_schema": list(
                    shortcut_common.PROJECTION_FIELDS
                ),
                "projection_content_sha256": (
                    dataset_common.canonical_sha256(rows)
                ),
                **shortcut_common.manifest_identity(
                    lock,
                    lock_path=lock_path,
                    stage="project_null_nuisance",
                    producer_relative_path=(
                        "scripts/"
                        "step28_v13_project_null_nuisance.py"
                    ),
                ),
                "input_allowlist": [
                    {
                        **input_snapshots[role].record(role=role),
                    }
                    for role in (
                        "candidate_pairs",
                        "history_item_index",
                        "redacted_items",
                    )
                ],
                "access_isolation_status": (
                    shortcut_common.BLOCKED_ACCESS_STATUS
                ),
                "forbidden_open_count_not_self_asserted": True,
                "files": [
                    shortcut_common.file_record(
                        projection_path,
                        role="null_nuisance_projection",
                        root=stage,
                    )
                ],
            }
        )
        dataset_common.write_json(
            stage / shortcut_common.PROJECTION_MANIFEST_FILENAME,
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
    parser.add_argument("--redacted-items", type=Path)
    parser.add_argument("--history-item-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = shortcut_common.load_lock(args.lock)
    if args.validate_config_only:
        print(
            "Step28-v13 null-nuisance projector configuration "
            "is locked; formal execution remains blocked"
        )
        return
    shortcut_common.require_formal_execution_envelope(lock)
    required = (
        args.candidate_pairs,
        args.redacted_items,
        args.history_item_index,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise shortcut_common.ShortcutAuditError(
            "Projector execution requires every exact input and output"
        )
    try:
        write_projection_release(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            candidate_path=args.candidate_pairs,
            redacted_path=args.redacted_items,
            history_item_path=args.history_item_index,
            output_dir=args.output_dir,
        )
    except Exception as error:
        shortcut_common.publish_stage_failure(
            lock=lock,
            lock_path=args.lock,
            split=args.split,
            stage="project_null_nuisance_failure",
            producer_relative_path=(
                "scripts/step28_v13_project_null_nuisance.py"
            ),
            output_dir=args.output_dir,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
