#!/usr/bin/env python3
"""Join V9.4 training/development truth only after proxy matrices are frozen."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_quality_probe_core_v9_4 as core_v94
import step28_v13_v1_13_quality_probe_preparer_v9_4 as preparer_v94


VERSION = "2026-08-27-step28-v13-v1-13-quality-probe-labels-v9-4"
LABEL_FIELDS = ("world_uid", "canonical_pair_uid", "y_true")
FORMAL_WORLD_COUNT = 500
FORMAL_PAIRS_PER_WORLD = 378
FORMAL_POSITIVES_PER_WORLD = 20
TRUTH_FORMULA = "same_private_controller_membership"
_LABEL_ISSUER = object()
_OPENED_TRUTH_CONNECTORS: set[tuple[str, str]] = set()


class QualityProbeLabelsV94Error(core_v94.QualityProbeCoreV94Error):
    """Raised when post-freeze label joining or its commitment drifts."""


@dataclass(frozen=True)
class FrozenLabelSplit:
    split: str
    row_keys: tuple[tuple[str, str], ...]
    values: np.ndarray
    commitment: Mapping[str, Any]
    _issuer: object


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _row_keys_sha256(row_keys: tuple[tuple[str, str], ...]) -> str:
    return _canonical_sha256([list(key) for key in row_keys])


def _freeze_labels_after_preparation(
    *,
    prepared: preparer_v94.PreparedSplit,
    label_rows: Sequence[Mapping[str, Any]],
    split_schedule_commitment_sha256: str,
    private_controller_truth_sha256: str,
    truth_source_version: str,
    truth_formula: str,
    truth_read_count: int,
    audit_truth_read_count: int,
    expected_world_count: int = FORMAL_WORLD_COUNT,
) -> FrozenLabelSplit:
    """Create an immutable label capability bound to one prepared matrix."""

    preparer_v94.verify_prepared_split(
        prepared,
        expected_world_count=expected_world_count,
    )
    if (
        split_schedule_commitment_sha256
        != prepared.commitment["split_schedule_commitment_sha256"]
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                split_schedule_commitment_sha256,
                private_controller_truth_sha256,
            )
        )
        or type(truth_source_version) is not str
        or not truth_source_version
        or truth_formula != TRUTH_FORMULA
        or type(truth_read_count) is not int
        or truth_read_count != 1
        or type(audit_truth_read_count) is not int
        or audit_truth_read_count != 0
    ):
        raise QualityProbeLabelsV94Error("Label truth provenance drift")
    expected_rows = expected_world_count * FORMAL_PAIRS_PER_WORLD
    if type(label_rows) is not tuple or len(label_rows) != expected_rows:
        raise QualityProbeLabelsV94Error("Label connector row container/count drift")
    row_keys: list[tuple[str, str]] = []
    raw_labels = bytearray()
    for row in label_rows:
        if type(row) is not MappingProxyType or tuple(row) != LABEL_FIELDS:
            raise QualityProbeLabelsV94Error("Label connector row schema/type drift")
        world_uid = row["world_uid"]
        pair_uid = row["canonical_pair_uid"]
        y_true = row["y_true"]
        if (
            type(world_uid) is not str
            or not world_uid
            or type(pair_uid) is not str
            or not pair_uid
            or type(y_true) is not int
            or y_true not in {0, 1}
        ):
            raise QualityProbeLabelsV94Error("Label connector row value drift")
        row_keys.append((world_uid, pair_uid))
        raw_labels.append(y_true)
    frozen_keys = tuple(row_keys)
    if frozen_keys != prepared.matrix.row_keys:
        raise QualityProbeLabelsV94Error("Label connector matrix-key join drift")
    values = np.frombuffer(bytes(raw_labels), dtype=np.dtype("int8"))
    positive_counts = Counter()
    pair_counts = Counter()
    for (world_uid, _), y_true in zip(frozen_keys, values, strict=True):
        pair_counts[world_uid] += 1
        positive_counts[world_uid] += int(y_true)
    if (
        len(pair_counts) != expected_world_count
        or set(pair_counts.values()) != {FORMAL_PAIRS_PER_WORLD}
        or set(positive_counts) != set(pair_counts)
        or set(positive_counts.values()) != {FORMAL_POSITIVES_PER_WORLD}
    ):
        raise QualityProbeLabelsV94Error("Label connector per-world closure drift")
    commitment_payload = {
        "version": VERSION,
        "split": prepared.split,
        "prepared_commitment_sha256": prepared.commitment[
            "prepared_commitment_sha256"
        ],
        "split_schedule_commitment_sha256": (
            split_schedule_commitment_sha256
        ),
        "private_controller_truth_sha256": private_controller_truth_sha256,
        "truth_source_version": truth_source_version,
        "truth_formula": truth_formula,
        "truth_read_count": truth_read_count,
        "audit_truth_read_count": audit_truth_read_count,
        "row_count": expected_rows,
        "positive_count": expected_world_count * FORMAL_POSITIVES_PER_WORLD,
        "negative_count": expected_world_count
        * (FORMAL_PAIRS_PER_WORLD - FORMAL_POSITIVES_PER_WORLD),
        "row_keys_sha256": _row_keys_sha256(frozen_keys),
        "labels_raw_i1_c_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    commitment_payload["label_commitment_sha256"] = _canonical_sha256(
        commitment_payload
    )
    frozen = FrozenLabelSplit(
        split=prepared.split,
        row_keys=frozen_keys,
        values=values,
        commitment=MappingProxyType(commitment_payload),
        _issuer=_LABEL_ISSUER,
    )
    verify_frozen_labels(
        frozen,
        prepared=prepared,
        expected_world_count=expected_world_count,
    )
    return frozen


def _open_controller_truth_after_preparation(
    *,
    prepared: preparer_v94.PreparedSplit,
    schedule: schedule_v94.SplitSchedule,
) -> FrozenLabelSplit:
    """Materialize train/development labels after the public matrix is committed."""

    preparer_v94.verify_prepared_split(prepared)
    schedule_v94.verify_split_schedule(schedule)
    if (
        prepared.split != schedule.split
        or prepared.commitment["world_source_sha256"]
        != schedule.commitment["public_worlds_sha256"]
        or prepared.commitment["split_schedule_commitment_sha256"]
        != schedule.commitment["split_schedule_commitment_sha256"]
    ):
        raise QualityProbeLabelsV94Error(
            "Controller truth/public prepared schedule binding drift"
        )
    connector_key = (
        prepared.commitment["prepared_commitment_sha256"],
        schedule.commitment["split_schedule_commitment_sha256"],
    )
    if connector_key in _OPENED_TRUTH_CONNECTORS:
        raise QualityProbeLabelsV94Error(
            "Controller truth connector has already been consumed"
        )
    _OPENED_TRUTH_CONNECTORS.add(connector_key)
    membership: dict[str, int] = {}
    for world_index, groups in enumerate(schedule.controller_groups_by_world):
        for controller_index, group in enumerate(groups):
            for seller_uid in group:
                if seller_uid in membership:
                    raise QualityProbeLabelsV94Error(
                        "Controller truth seller membership collision"
                    )
                membership[seller_uid] = world_index * len(groups) + controller_index
    label_rows: list[Mapping[str, Any]] = []
    for world_uid, pair_uid in prepared.matrix.row_keys:
        endpoints = pair_uid.split("||")
        if (
            len(endpoints) != 2
            or any(endpoint not in membership for endpoint in endpoints)
            or not all(endpoint.startswith(world_uid + "_seller_") for endpoint in endpoints)
        ):
            raise QualityProbeLabelsV94Error("Controller truth pair-key drift")
        label_rows.append(MappingProxyType({
            "world_uid": world_uid,
            "canonical_pair_uid": pair_uid,
            "y_true": int(membership[endpoints[0]] == membership[endpoints[1]]),
        }))
    return _freeze_labels_after_preparation(
        prepared=prepared,
        label_rows=tuple(label_rows),
        split_schedule_commitment_sha256=schedule.commitment[
            "split_schedule_commitment_sha256"
        ],
        private_controller_truth_sha256=schedule.commitment[
            "private_controller_truth_sha256"
        ],
        truth_source_version=schedule_v94.VERSION,
        truth_formula=TRUTH_FORMULA,
        truth_read_count=1,
        audit_truth_read_count=0,
    )


def verify_frozen_labels(
    frozen: FrozenLabelSplit,
    *,
    prepared: preparer_v94.PreparedSplit,
    expected_world_count: int = FORMAL_WORLD_COUNT,
) -> None:
    preparer_v94.verify_prepared_split(
        prepared,
        expected_world_count=expected_world_count,
    )
    expected_fields = (
        "version",
        "split",
        "prepared_commitment_sha256",
        "split_schedule_commitment_sha256",
        "private_controller_truth_sha256",
        "truth_source_version",
        "truth_formula",
        "truth_read_count",
        "audit_truth_read_count",
        "row_count",
        "positive_count",
        "negative_count",
        "row_keys_sha256",
        "labels_raw_i1_c_sha256",
        "label_commitment_sha256",
    )
    expected_rows = expected_world_count * FORMAL_PAIRS_PER_WORLD
    if (
        type(frozen) is not FrozenLabelSplit
        or frozen._issuer is not _LABEL_ISSUER
        or type(frozen.commitment) is not MappingProxyType
        or tuple(frozen.commitment) != expected_fields
        or frozen.split != prepared.split
        or frozen.row_keys != prepared.matrix.row_keys
        or frozen.values.dtype != np.dtype("int8")
        or frozen.values.shape != (expected_rows,)
        or frozen.values.flags.writeable
        or set(frozen.values.tolist()) != {0, 1}
        or frozen.commitment["version"] != VERSION
        or frozen.commitment["split"] != frozen.split
        or frozen.commitment["prepared_commitment_sha256"]
        != prepared.commitment["prepared_commitment_sha256"]
        or frozen.commitment["split_schedule_commitment_sha256"]
        != prepared.commitment["split_schedule_commitment_sha256"]
        or any(
            type(frozen.commitment[field]) is not str
            or len(frozen.commitment[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in frozen.commitment[field]
            )
            for field in (
                "split_schedule_commitment_sha256",
                "private_controller_truth_sha256",
            )
        )
        or frozen.commitment["truth_source_version"] != schedule_v94.VERSION
        or frozen.commitment["truth_formula"] != TRUTH_FORMULA
        or frozen.commitment["truth_read_count"] != 1
        or frozen.commitment["audit_truth_read_count"] != 0
        or frozen.commitment["row_count"] != expected_rows
        or frozen.commitment["positive_count"]
        != expected_world_count * FORMAL_POSITIVES_PER_WORLD
        or frozen.commitment["negative_count"]
        != expected_world_count
        * (FORMAL_PAIRS_PER_WORLD - FORMAL_POSITIVES_PER_WORLD)
        or frozen.commitment["row_keys_sha256"]
        != _row_keys_sha256(frozen.row_keys)
        or frozen.commitment["labels_raw_i1_c_sha256"]
        != hashlib.sha256(frozen.values.tobytes()).hexdigest()
        or frozen.commitment["label_commitment_sha256"]
        != _canonical_sha256({
            key: frozen.commitment[key]
            for key in expected_fields[:-1]
        })
    ):
        raise QualityProbeLabelsV94Error("Frozen label capability drift")


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "label_fields": list(LABEL_FIELDS),
        "allowed_splits": ["train", "development"],
        "audit_truth_access_count": 0,
        "join_order": "after_prepared_matrix_commitment",
        "formal_truth_source": schedule_v94.VERSION,
        "formal_truth_formula": TRUTH_FORMULA,
        "formal_truth_read_count": 1,
        "public_truth_connector": False,
    }
