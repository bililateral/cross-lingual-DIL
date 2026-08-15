#!/usr/bin/env python3
"""Label-free matrix preparation for the Step28-v13 v1.13 v9 quality audit.

This module accepts only already-materialized model views, pair endpoints, and
the private code-decoder capability.  It has no truth-file reader and never
fits a model.  Every matrix is committed before a supervised validator may
open train/development labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any

import numpy as np

import step28_v13_v1_13_quality_channel_views_v9 as channel
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views


VERSION = "2026-08-14-step28-v13-v1-13-quality-probe-preparer-v9"
MATRIX_CHUNK_BYTES = 8 * 1024 * 1024
ENDPOINT_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
PUBLIC_ROW_FIELDS = (
    "world_uid",
    "seller_uid",
    "owned_codes",
    "item_occurrences",
    "profile_occurrences",
    "numeric_profile_deltas",
)
OCCURRENCE_FIELDS = ("field", "code", "is_own")
EXPECTED_SELLERS_PER_WORLD = 28
EXPECTED_PAIRS_PER_WORLD = 378
WORLD_STRIDE = 256
SELLER_STRIDE = 8
ELIGIBILITY_FIELDS = (
    "world_uid",
    "canonical_pair_uid",
    "text_probe_eligible",
)
TEXT_SURFACES = (
    "surface_full",
    "surface_code_masked",
    "surface_code_neutralized",
)


class QualityProbePreparationError(ValueError):
    """Raised when label-free feature preparation drifts or is contaminated."""


@dataclass(frozen=True)
class SourceCommitment:
    """Hash binding for one persisted, label-free input file."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class FrozenFeatureMatrix:
    """An in-memory matrix plus an immutable pre-label commitment."""

    family: str
    view: str
    values: np.ndarray
    row_keys: tuple[tuple[str, str], ...]
    column_names: tuple[str, ...]
    sources: tuple[SourceCommitment, ...]
    commitment_json: bytes
    commitment_sha256: str


@dataclass(frozen=True)
class FrozenTextEligibility:
    """The exact complete and 372-row text keyspaces frozen before truth."""

    values: np.ndarray
    row_keys: tuple[tuple[str, str], ...]
    sources: tuple[SourceCommitment, ...]
    commitment_json: bytes
    commitment_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise QualityProbePreparationError(f"{name} must be a nonempty string")
    return value


def _validate_sources(
    sources: Sequence[SourceCommitment],
) -> tuple[SourceCommitment, ...]:
    normalized = tuple(sources)
    if not normalized:
        raise QualityProbePreparationError("Source commitment list is empty")
    paths: list[str] = []
    for source in normalized:
        if not isinstance(source, SourceCommitment):
            raise QualityProbePreparationError("Source commitment type drift")
        if type(source.path) is not str:
            raise QualityProbePreparationError("Source commitment path type drift")
        path = PurePosixPath(source.path)
        if (
            not source.path
            or path.is_absolute()
            or ".." in path.parts
            or source.path != path.as_posix()
            or isinstance(source.size_bytes, bool)
            or not isinstance(source.size_bytes, int)
            or source.size_bytes < 0
            or not isinstance(source.sha256, str)
            or len(source.sha256) != 64
            or any(character not in "0123456789abcdef" for character in source.sha256)
        ):
            raise QualityProbePreparationError("Source commitment schema drift")
        paths.append(source.path)
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")) or len(
        paths
    ) != len(set(paths)):
        raise QualityProbePreparationError(
            "Source commitments must be unique and UTF-8 byte sorted"
        )
    return normalized


def _matrix_sha256(values: np.ndarray) -> str:
    if (
        values.dtype != np.dtype("<f8")
        or values.ndim != 2
        or not values.flags.c_contiguous
    ):
        raise QualityProbePreparationError("Matrix hash input representation drift")
    digest = hashlib.sha256()
    raw = memoryview(values).cast("B")
    chunk_size = MATRIX_CHUNK_BYTES
    for start in range(0, len(raw), chunk_size):
        digest.update(raw[start : start + chunk_size])
    return digest.hexdigest()


def _stream_nonfinite_and_missing_bitmap(values: np.ndarray) -> tuple[int, str]:
    """Count non-finite cells and hash the little-endian missing bitmap in chunks."""

    if (
        values.dtype != np.dtype("<f8")
        or values.ndim != 2
        or not values.flags.c_contiguous
    ):
        raise QualityProbePreparationError("Matrix bitmap input representation drift")
    flattened = values.reshape(-1)
    # One MiB of float64 cells per pass.  The size is divisible by eight, so
    # only the final chunk may require packbits padding.
    if MATRIX_CHUNK_BYTES <= 0 or MATRIX_CHUNK_BYTES % values.dtype.itemsize:
        raise QualityProbePreparationError("Matrix chunk-size contract drift")
    cells_per_chunk = MATRIX_CHUNK_BYTES // values.dtype.itemsize
    nonfinite_count = 0
    missing_digest = hashlib.sha256()
    for start in range(0, flattened.size, cells_per_chunk):
        chunk = flattened[start : start + cells_per_chunk]
        finite = np.isfinite(chunk)
        nonfinite_count += int(chunk.size - np.count_nonzero(finite))
        del finite
        missing = np.isnan(chunk)
        missing_digest.update(
            np.packbits(missing, bitorder="little").tobytes()
        )
        del missing
    return nonfinite_count, missing_digest.hexdigest()


def _commitment_payload(
    *,
    family: str,
    view: str,
    values: np.ndarray,
    row_keys: tuple[tuple[str, str], ...],
    column_names: tuple[str, ...],
    sources: tuple[SourceCommitment, ...],
) -> dict[str, Any]:
    nonfinite_count, missing_bitmap_sha256 = (
        _stream_nonfinite_and_missing_bitmap(values)
    )
    return {
        "version": VERSION,
        "stage": "LABEL_FREE_FEATURE_FREEZE_BEFORE_TRUTH_OPEN",
        "family": family,
        "view": view,
        "dtype": "<f8",
        "order": "C",
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "matrix_sha256": _matrix_sha256(values),
        "nonfinite_count": nonfinite_count,
        "missing_bitmap_sha256": missing_bitmap_sha256,
        "row_count": len(row_keys),
        "row_keys_canonical_json_sha256": _sha256(
            _canonical_json_bytes([list(value) for value in row_keys])
        ),
        "column_count": len(column_names),
        "column_names_canonical_json_sha256": _sha256(
            _canonical_json_bytes(list(column_names))
        ),
        "sources": [
            {
                "path": source.path,
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            }
            for source in sources
        ],
        "label_count_read": 0,
        "audit_truth_open_count": 0,
        "uid_or_split_value_used_as_feature": False,
        "decoded_values_persisted": False,
    }


def _freeze_feature_matrix(
    *,
    family: str,
    view: str,
    values: np.ndarray,
    row_keys: Sequence[tuple[str, str]],
    column_names: Sequence[str],
    sources: Sequence[SourceCommitment],
    copy_values: bool,
) -> FrozenFeatureMatrix:
    if not family or not view:
        raise QualityProbePreparationError("Matrix family/view is empty")
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("<f8")
        or values.ndim != 2
    ):
        raise QualityProbePreparationError(
            "Feature matrix must already be a two-dimensional float64 array"
        )
    if not values.flags.c_contiguous:
        raise QualityProbePreparationError("Feature matrix must already be C-contiguous")
    array = (
        np.array(values, dtype=np.dtype("<f8"), order="C", copy=True)
        if copy_values
        else values
    )
    rows = tuple(row_keys)
    columns = tuple(column_names)
    normalized_sources = _validate_sources(sources)
    if (
        array.ndim != 2
        or array.shape != (len(rows), len(columns))
        or not rows
        or not columns
        or any(
            not isinstance(world_uid, str)
            or not world_uid
            or not isinstance(pair_uid, str)
            or not pair_uid
            for world_uid, pair_uid in rows
        )
        or len(rows) != len(set(rows))
        or any(not isinstance(name, str) or not name for name in columns)
        or len(columns) != len(set(columns))
    ):
        raise QualityProbePreparationError("Feature matrix failed closure")
    array.setflags(write=False)
    payload = _commitment_payload(
        family=family,
        view=view,
        values=array,
        row_keys=rows,
        column_names=columns,
        sources=normalized_sources,
    )
    if payload["nonfinite_count"] != 0:
        raise QualityProbePreparationError("Feature matrix contains nonfinite values")
    commitment_json = _canonical_json_bytes(payload)
    return FrozenFeatureMatrix(
        family=family,
        view=view,
        values=array,
        row_keys=rows,
        column_names=columns,
        sources=normalized_sources,
        commitment_json=commitment_json,
        commitment_sha256=_sha256(commitment_json),
    )


def freeze_feature_matrix(
    *,
    family: str,
    view: str,
    values: np.ndarray,
    row_keys: Sequence[tuple[str, str]],
    column_names: Sequence[str],
    sources: Sequence[SourceCommitment],
) -> FrozenFeatureMatrix:
    """Copy, validate, and commit a caller-owned dense float64 matrix."""

    return _freeze_feature_matrix(
        family=family,
        view=view,
        values=values,
        row_keys=row_keys,
        column_names=column_names,
        sources=sources,
        copy_values=True,
    )


def _freeze_owned_feature_matrix(
    *,
    family: str,
    view: str,
    values: np.ndarray,
    row_keys: Sequence[tuple[str, str]],
    column_names: Sequence[str],
    sources: Sequence[SourceCommitment],
) -> FrozenFeatureMatrix:
    """Freeze a preparer-owned matrix without a second full-size copy."""

    return _freeze_feature_matrix(
        family=family,
        view=view,
        values=values,
        row_keys=row_keys,
        column_names=column_names,
        sources=sources,
        copy_values=False,
    )


def verify_frozen_feature_matrix(frozen: FrozenFeatureMatrix) -> dict[str, Any]:
    """Rehash a frozen matrix immediately before supervised use."""

    if not isinstance(frozen, FrozenFeatureMatrix):
        raise QualityProbePreparationError("Frozen matrix type drift")
    sources = _validate_sources(frozen.sources)
    payload = _commitment_payload(
        family=frozen.family,
        view=frozen.view,
        values=frozen.values,
        row_keys=frozen.row_keys,
        column_names=frozen.column_names,
        sources=sources,
    )
    observed_json = _canonical_json_bytes(payload)
    if (
        observed_json != frozen.commitment_json
        or _sha256(observed_json) != frozen.commitment_sha256
        or frozen.values.dtype != np.dtype("<f8")
        or frozen.values.ndim != 2
        or not frozen.values.flags.c_contiguous
        or frozen.values.flags.writeable
    ):
        raise QualityProbePreparationError(
            "Pre-label feature commitment changed before supervised validation"
        )
    return json.loads(observed_json.decode("utf-8"))


def current_feature_matrix_commitment_json(
    frozen: FrozenFeatureMatrix,
) -> bytes:
    """Recompute current matrix bytes without trusting its stored commitment fields."""

    if not isinstance(frozen, FrozenFeatureMatrix):
        raise QualityProbePreparationError("Frozen matrix type drift")
    sources = _validate_sources(frozen.sources)
    return _canonical_json_bytes(
        _commitment_payload(
            family=frozen.family,
            view=frozen.view,
            values=frozen.values,
            row_keys=frozen.row_keys,
            column_names=frozen.column_names,
            sources=sources,
        )
    )


def _eligibility_payload(
    *,
    values: np.ndarray,
    row_keys: tuple[tuple[str, str], ...],
    sources: tuple[SourceCommitment, ...],
) -> dict[str, Any]:
    eligible_keys = [
        list(key) for key, keep in zip(row_keys, values.tolist()) if keep
    ]
    return {
        "version": VERSION,
        "stage": "LABEL_FREE_TEXT_ELIGIBILITY_FREEZE_BEFORE_TRUTH_OPEN",
        "complete_row_count": len(row_keys),
        "eligible_row_count": int(np.count_nonzero(values)),
        "complete_row_keys_canonical_json_sha256": _sha256(
            _canonical_json_bytes([list(value) for value in row_keys])
        ),
        "eligible_row_keys_canonical_json_sha256": _sha256(
            _canonical_json_bytes(eligible_keys)
        ),
        "eligibility_bitmap_sha256": _sha256(
            np.packbits(values, bitorder="little").tobytes()
        ),
        "sources": [
            {
                "path": source.path,
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            }
            for source in sources
        ],
        "label_count_read": 0,
        "audit_truth_open_count": 0,
    }


def verify_frozen_text_eligibility(
    frozen: FrozenTextEligibility,
) -> dict[str, Any]:
    if not isinstance(frozen, FrozenTextEligibility):
        raise QualityProbePreparationError("Frozen eligibility type drift")
    sources = _validate_sources(frozen.sources)
    payload = _eligibility_payload(
        values=frozen.values, row_keys=frozen.row_keys, sources=sources
    )
    observed_json = _canonical_json_bytes(payload)
    if (
        frozen.values.dtype != np.dtype(bool)
        or frozen.values.ndim != 1
        or len(frozen.values) != len(frozen.row_keys)
        or frozen.values.flags.writeable
        or observed_json != frozen.commitment_json
        or _sha256(observed_json) != frozen.commitment_sha256
    ):
        raise QualityProbePreparationError(
            "Text eligibility changed before supervised validation"
        )
    return json.loads(observed_json.decode("utf-8"))


def current_text_eligibility_commitment_json(
    frozen: FrozenTextEligibility,
) -> bytes:
    """Recompute current eligibility bytes without trusting stored fields."""

    if not isinstance(frozen, FrozenTextEligibility):
        raise QualityProbePreparationError("Frozen eligibility type drift")
    sources = _validate_sources(frozen.sources)
    return _canonical_json_bytes(
        _eligibility_payload(
            values=frozen.values,
            row_keys=frozen.row_keys,
            sources=sources,
        )
    )


def _validate_endpoints(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    ordered_world_uids: Sequence[str],
    expected_pairs_per_world: int,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], dict[str, set[str]]]:
    if (
        isinstance(expected_pairs_per_world, bool)
        or not isinstance(expected_pairs_per_world, int)
        or expected_pairs_per_world <= 0
    ):
        raise QualityProbePreparationError("Expected pair count is invalid")
    required_worlds = tuple(ordered_world_uids)
    if (
        not required_worlds
        or len(required_worlds) != len(set(required_worlds))
        or any(not isinstance(value, str) or not value for value in required_worlds)
    ):
        raise QualityProbePreparationError("Frozen world-ordinal order is invalid")
    row_keys: list[tuple[str, str]] = []
    ordered_worlds: list[str] = []
    sellers_by_world: defaultdict[str, set[str]] = defaultdict(set)
    seller_pairs_by_world: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    pair_uids_by_world: defaultdict[str, list[str]] = defaultdict(list)
    global_pair_uids: set[str] = set()
    seller_world: dict[str, str] = {}
    closed_worlds: set[str] = set()
    active_world: str | None = None
    for row in endpoints:
        if not isinstance(row, Mapping) or tuple(row) != ENDPOINT_FIELDS:
            raise QualityProbePreparationError("Pair endpoint schema/order drift")
        pair_uid = _required_text(row["canonical_pair_uid"], name="pair UID")
        world_uid = _required_text(row["world_uid"], name="world UID")
        left = _required_text(row["seller_uid_left"], name="left seller UID")
        right = _required_text(row["seller_uid_right"], name="right seller UID")
        if left == right:
            raise QualityProbePreparationError("Pair endpoint value drift")
        if pair_uid in global_pair_uids:
            raise QualityProbePreparationError("Pair UID is reused across the split")
        global_pair_uids.add(pair_uid)
        for seller_uid in (left, right):
            previous_world = seller_world.setdefault(seller_uid, world_uid)
            if previous_world != world_uid:
                raise QualityProbePreparationError(
                    "Seller UID crosses a world boundary"
                )
        if world_uid != active_world:
            if active_world is not None:
                closed_worlds.add(active_world)
            if world_uid in closed_worlds:
                raise QualityProbePreparationError("World endpoint rows are not contiguous")
            ordered_worlds.append(world_uid)
            active_world = world_uid
        row_keys.append((world_uid, pair_uid))
        pair_uids_by_world[world_uid].append(pair_uid)
        sellers_by_world[world_uid].update((left, right))
        seller_pairs_by_world[world_uid].append(
            tuple(sorted((left, right), key=lambda value: value.encode("utf-8")))
        )
    if not row_keys or len(row_keys) != len(set(row_keys)):
        raise QualityProbePreparationError("Pair endpoint key collision or empty input")
    if tuple(ordered_worlds) != required_worlds:
        raise QualityProbePreparationError(
            "Endpoint blocks disagree with frozen world-ordinal order"
        )
    for world_uid in ordered_worlds:
        pair_uids = pair_uids_by_world[world_uid]
        sellers = sorted(
            sellers_by_world[world_uid], key=lambda value: value.encode("utf-8")
        )
        expected_seller_pairs = {
            (sellers[left], sellers[right])
            for left in range(len(sellers))
            for right in range(left + 1, len(sellers))
        }
        observed_seller_pairs = seller_pairs_by_world[world_uid]
        if (
            len(pair_uids) != expected_pairs_per_world
            or pair_uids != sorted(pair_uids, key=lambda value: value.encode("utf-8"))
            or len(observed_seller_pairs) != len(set(observed_seller_pairs))
            or set(observed_seller_pairs) != expected_seller_pairs
        ):
            raise QualityProbePreparationError(
                "Per-world endpoint count/order/unordered-pair universe drift"
            )
    return tuple(row_keys), tuple(ordered_worlds), dict(sellers_by_world)


def _parse_occurrence(row: Mapping[str, Any]) -> channel.CodeOccurrence:
    if (
        not isinstance(row, Mapping)
        or set(row) != set(OCCURRENCE_FIELDS)
        or type(row["is_own"]) is not bool
    ):
        raise QualityProbePreparationError("Code occurrence schema/type drift")
    return channel.CodeOccurrence(
        code=_required_text(row["code"], name="registered code"),
        field=_required_text(row["field"], name="visible code field"),
        is_own=row["is_own"],
    )


def freeze_text_eligibility(
    *,
    eligibility_rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    sources: Sequence[SourceCommitment],
    expected_pairs_per_world: int = EXPECTED_PAIRS_PER_WORLD,
    expected_excluded_pairs_per_world: int = 6,
) -> FrozenTextEligibility:
    """Freeze the one authoritative text mask without reading labels."""

    row_keys, worlds, _sellers = _validate_endpoints(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=expected_pairs_per_world,
    )
    rows = tuple(eligibility_rows)
    if len(rows) != len(row_keys):
        raise QualityProbePreparationError("Text eligibility row count drift")
    values = np.empty(len(rows), dtype=bool)
    excluded: Counter[str] = Counter()
    for index, (row, (world_uid, pair_uid)) in enumerate(zip(rows, row_keys)):
        if (
            not isinstance(row, Mapping)
            or set(row) != set(ELIGIBILITY_FIELDS)
            or row["world_uid"] != world_uid
            or row["canonical_pair_uid"] != pair_uid
            or type(row["text_probe_eligible"]) is not bool
        ):
            raise QualityProbePreparationError(
                "Text eligibility schema drift"
            )
        values[index] = row["text_probe_eligible"]
        if not values[index]:
            excluded[world_uid] += 1
    if excluded != Counter(
        {world_uid: expected_excluded_pairs_per_world for world_uid in worlds}
    ):
        raise QualityProbePreparationError(
            "Text eligibility excluded-pair count drift"
        )
    normalized_sources = _validate_sources(sources)
    values.setflags(write=False)
    payload = _eligibility_payload(
        values=values, row_keys=row_keys, sources=normalized_sources
    )
    commitment_json = _canonical_json_bytes(payload)
    return FrozenTextEligibility(
        values=values,
        row_keys=row_keys,
        sources=normalized_sources,
        commitment_json=commitment_json,
        commitment_sha256=_sha256(commitment_json),
    )


def _parse_public_rows(
    public_rows: Sequence[Mapping[str, Any]],
    *,
    ordered_worlds: Sequence[str],
    sellers_by_world: Mapping[str, set[str]],
    expected_sellers_per_world: int,
) -> dict[tuple[str, str], tuple[channel.SellerCodeView, tuple[str, ...]]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_world_order: list[str] = []
    active_world: str | None = None
    closed_worlds: set[str] = set()
    for row in public_rows:
        if not isinstance(row, Mapping) or set(row) != set(PUBLIC_ROW_FIELDS):
            raise QualityProbePreparationError("Public code row schema drift")
        world_uid = _required_text(row["world_uid"], name="public row world UID")
        if world_uid != active_world:
            if active_world is not None:
                closed_worlds.add(active_world)
            if world_uid in closed_worlds:
                raise QualityProbePreparationError("Public code worlds are not contiguous")
            observed_world_order.append(world_uid)
            active_world = world_uid
        grouped[world_uid].append(row)
    if tuple(observed_world_order) != tuple(ordered_worlds):
        raise QualityProbePreparationError("Public code/endpoints world order drift")

    result: dict[
        tuple[str, str], tuple[channel.SellerCodeView, tuple[str, ...]]
    ] = {}
    for world_uid in ordered_worlds:
        rows = grouped[world_uid]
        seller_uids = [
            _required_text(row["seller_uid"], name="public row seller UID")
            for row in rows
        ]
        if (
            len(rows) != expected_sellers_per_world
            or seller_uids
            != sorted(seller_uids, key=lambda value: value.encode("utf-8"))
            or len(seller_uids) != len(set(seller_uids))
            or set(seller_uids) != sellers_by_world[world_uid]
        ):
            raise QualityProbePreparationError("Public code seller universe/order drift")
        for row in rows:
            seller_uid = _required_text(
                row["seller_uid"], name="public row seller UID"
            )
            if not isinstance(row["owned_codes"], (list, tuple)):
                raise QualityProbePreparationError("Owned-code container drift")
            owned = tuple(
                _required_text(value, name="owned registered code")
                for value in row["owned_codes"]
            )
            if not owned or owned != tuple(
                sorted(owned, key=lambda value: value.encode("ascii"))
            ) or len(owned) != len(set(owned)):
                raise QualityProbePreparationError("Owned-code universe/order drift")
            if not isinstance(row["item_occurrences"], (list, tuple)) or not isinstance(
                row["profile_occurrences"], (list, tuple)
            ):
                raise QualityProbePreparationError("Code occurrence container drift")
            item_occurrences = tuple(
                _parse_occurrence(value) for value in row["item_occurrences"]
            )
            profile_occurrences_flat = tuple(
                _parse_occurrence(value) for value in row["profile_occurrences"]
            )
            if any(
                value.field not in channel.ITEM_FIELDS for value in item_occurrences
            ) or any(
                value.field not in channel.PROFILE_FIELDS
                for value in profile_occurrences_flat
            ):
                raise QualityProbePreparationError("Code occurrence surface drift")
            profile_occurrences = {
                field: tuple(
                    value for value in profile_occurrences_flat if value.field == field
                )
                for field in channel.PROFILE_FIELDS
            }
            deltas = row["numeric_profile_deltas"]
            if (
                not isinstance(deltas, Mapping)
                or set(deltas) != set(channel.NUMERIC_DELTA_FIELDS)
                or any(
                    isinstance(deltas[name], bool)
                    or not isinstance(deltas[name], (int, float))
                    or not math.isfinite(float(deltas[name]))
                    for name in channel.NUMERIC_DELTA_FIELDS
                )
            ):
                raise QualityProbePreparationError("Numeric profile delta drift")
            view = channel.SellerCodeView(
                owned_codes=owned,
                visible_occurrences=item_occurrences,
                profile_occurrences=profile_occurrences,
                numeric_profile_deltas={
                    name: float(deltas[name]) for name in channel.NUMERIC_DELTA_FIELDS
                },
            )
            # Force the complete channel-level validator before accepting the row.
            channel.build_public_code_pair_features(view, view)
            result[(world_uid, seller_uid)] = (view, owned)
    for world_uid in ordered_worlds:
        owner_by_code: dict[str, str] = {}
        for seller_uid in sellers_by_world[world_uid]:
            _view, owned = result[(world_uid, seller_uid)]
            for code in owned:
                previous = owner_by_code.setdefault(code, seller_uid)
                if previous != seller_uid:
                    raise QualityProbePreparationError(
                        "Registered code has multiple owners in one world"
                    )
        for seller_uid in sellers_by_world[world_uid]:
            view, _owned = result[(world_uid, seller_uid)]
            occurrences = list(view.visible_occurrences)
            for field in channel.PROFILE_FIELDS:
                occurrences.extend(view.profile_occurrences[field])
            for occurrence in occurrences:
                owner = owner_by_code.get(occurrence.code)
                if owner is None or occurrence.is_own != (owner == seller_uid):
                    raise QualityProbePreparationError(
                        "Code occurrence disagrees with the world owner universe"
                    )
    return result


def prepare_public_code_matrix(
    *,
    public_rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    sources: Sequence[SourceCommitment],
    expected_sellers_per_world: int = EXPECTED_SELLERS_PER_WORLD,
    expected_pairs_per_world: int = EXPECTED_PAIRS_PER_WORLD,
) -> FrozenFeatureMatrix:
    """Build and freeze the exact 2,992-wide public code-channel matrix."""

    row_keys, worlds, sellers_by_world = _validate_endpoints(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=expected_pairs_per_world,
    )
    parsed = _parse_public_rows(
        public_rows,
        ordered_worlds=worlds,
        sellers_by_world=sellers_by_world,
        expected_sellers_per_world=expected_sellers_per_world,
    )
    matrix = np.empty((len(endpoints), channel.PUBLIC_FEATURE_WIDTH), dtype="<f8")
    for index, row in enumerate(endpoints):
        world_uid = row["world_uid"]
        left = parsed[(world_uid, row["seller_uid_left"])][0]
        right = parsed[(world_uid, row["seller_uid_right"])][0]
        matrix[index] = channel.build_public_code_pair_features(left, right)
    return _freeze_owned_feature_matrix(
        family="code_and_slot",
        view="public_code_2992",
        values=matrix,
        row_keys=row_keys,
        column_names=channel.public_feature_names(),
        sources=sources,
    )


def prepare_text_surface_matrices(
    *,
    surface: str,
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    sources: Sequence[SourceCommitment],
    expected_sellers_per_world: int = EXPECTED_SELLERS_PER_WORLD,
    expected_pairs_per_world: int = EXPECTED_PAIRS_PER_WORLD,
) -> tuple[FrozenFeatureMatrix, ...]:
    """Build and freeze seven label-free text views for one model surface."""

    if surface not in TEXT_SURFACES:
        raise QualityProbePreparationError("Unknown text model surface")
    row_keys, worlds, sellers_by_world = _validate_endpoints(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=expected_pairs_per_world,
    )
    if any(
        len(sellers_by_world[world_uid]) != expected_sellers_per_world
        for world_uid in worlds
    ):
        raise QualityProbePreparationError("Text seller count per world drift")
    matrices, names = text_views.build_text_probe_views(
        items=items, profiles=profiles, endpoints=endpoints
    )
    return tuple(
        _freeze_owned_feature_matrix(
            family="text",
            view=f"{surface}::{view}",
            values=matrices[view],
            row_keys=row_keys,
            column_names=names[view],
            sources=sources,
        )
        for view in text_views.VIEW_ORDER
    )


def prepare_decoded_slot_matrix(
    *,
    public_rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    expected_mode_global_ordinal_by_world: Mapping[str, int],
    expected_seller_slot_by_world_and_seller: Mapping[tuple[str, str], int],
    decode_coordinate: Callable[[str, str], int],
    sources: Sequence[SourceCommitment],
    expected_sellers_per_world: int = EXPECTED_SELLERS_PER_WORLD,
    expected_pairs_per_world: int = EXPECTED_PAIRS_PER_WORLD,
) -> FrozenFeatureMatrix:
    """Decode creation slots in memory and freeze the exact 388-wide matrix."""

    if not callable(decode_coordinate):
        raise QualityProbePreparationError("Private decoder capability is missing")
    row_keys, worlds, sellers_by_world = _validate_endpoints(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=expected_pairs_per_world,
    )
    parsed = _parse_public_rows(
        public_rows,
        ordered_worlds=worlds,
        sellers_by_world=sellers_by_world,
        expected_sellers_per_world=expected_sellers_per_world,
    )
    expected_ordinals = dict(expected_mode_global_ordinal_by_world)
    if (
        set(expected_ordinals) != set(worlds)
        or len(set(expected_ordinals.values())) != len(worlds)
        or any(
            type(value) is not int or not 0 <= value < (1 << 32)
            for value in expected_ordinals.values()
        )
    ):
        raise QualityProbePreparationError(
            "Expected mode-global world ordinal authority drift"
        )
    expected_seller_slots = dict(expected_seller_slot_by_world_and_seller)
    expected_seller_keys = set(parsed)
    if (
        set(expected_seller_slots) != expected_seller_keys
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or type(key[0]) is not str
            or type(key[1]) is not str
            or type(value) is not int
            or not 0 <= value < expected_sellers_per_world
            for key, value in expected_seller_slots.items()
        )
        or any(
            {
                expected_seller_slots[(world_uid, seller_uid)]
                for seller_uid in sellers_by_world[world_uid]
            }
            != set(range(expected_sellers_per_world))
            for world_uid in worlds
        )
    ):
        raise QualityProbePreparationError(
            "Expected seller creation-slot authority drift"
        )
    decoded: dict[tuple[str, str], tuple[int, tuple[int, ...]]] = {}
    mode_ordinal_by_world: dict[str, int] = {}
    seller_slots_by_world: defaultdict[str, set[int]] = defaultdict(set)
    for (world_uid, seller_uid), (_view, owned) in parsed.items():
        coordinates: list[int] = []
        for code in owned:
            value = decode_coordinate(world_uid, code)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < (1 << 40)
            ):
                raise QualityProbePreparationError("Decoded coordinate type/range drift")
            coordinates.append(value)
        world_ordinals = {value // WORLD_STRIDE for value in coordinates}
        seller_slots = {
            (value % WORLD_STRIDE) // SELLER_STRIDE for value in coordinates
        }
        item_slots = tuple(
            sorted(value % SELLER_STRIDE for value in coordinates)
        )
        if (
            len(world_ordinals) != 1
            or len(seller_slots) != 1
            or len(item_slots) != len(set(item_slots))
            or item_slots != tuple(range(len(item_slots)))
        ):
            raise QualityProbePreparationError("Decoded seller/item slot closure drift")
        mode_ordinal = next(iter(world_ordinals))
        seller_slot = next(iter(seller_slots))
        if mode_ordinal != expected_ordinals[world_uid]:
            raise QualityProbePreparationError(
                "Decoded world ordinal disagrees with frozen world authority"
            )
        if seller_slot != expected_seller_slots[(world_uid, seller_uid)]:
            raise QualityProbePreparationError(
                "Decoded seller slot disagrees with frozen seller authority"
            )
        if world_uid in mode_ordinal_by_world and mode_ordinal_by_world[world_uid] != mode_ordinal:
            raise QualityProbePreparationError("Decoded world ordinal disagreement")
        mode_ordinal_by_world[world_uid] = mode_ordinal
        if seller_slot in seller_slots_by_world[world_uid]:
            raise QualityProbePreparationError("Decoded seller slot collision")
        seller_slots_by_world[world_uid].add(seller_slot)
        decoded[(world_uid, seller_uid)] = (seller_slot, item_slots)
    if any(
        slots != set(range(expected_sellers_per_world))
        for slots in seller_slots_by_world.values()
    ) or len(set(mode_ordinal_by_world.values())) != len(mode_ordinal_by_world):
        raise QualityProbePreparationError("Decoded world/seller slot universe drift")

    names = channel.decoded_feature_names()
    matrix = np.empty((len(endpoints), len(names)), dtype="<f8")
    for index, row in enumerate(endpoints):
        world_uid = row["world_uid"]
        left_slot, left_items = decoded[(world_uid, row["seller_uid_left"])]
        right_slot, right_items = decoded[(world_uid, row["seller_uid_right"])]
        matrix[index] = channel.build_decoded_slot_pair_features(
            left_seller_slot=left_slot,
            right_seller_slot=right_slot,
            left_item_slots=left_items,
            right_item_slots=right_items,
        )
    return _freeze_owned_feature_matrix(
        family="code_and_slot",
        view="decoded_slot_388",
        values=matrix,
        row_keys=row_keys,
        column_names=names,
        sources=sources,
    )


def combine_frozen_matrices(
    *,
    view: str,
    matrices: Sequence[FrozenFeatureMatrix],
) -> FrozenFeatureMatrix:
    """Combine already-frozen views without changing their row semantics."""

    values = tuple(matrices)
    if not values:
        raise QualityProbePreparationError("No frozen matrices to combine")
    for value in values:
        verify_frozen_feature_matrix(value)
    first = values[0]
    if any(
        value.family != first.family or value.row_keys != first.row_keys
        for value in values[1:]
    ):
        raise QualityProbePreparationError("Frozen matrix family/row alignment drift")
    columns = tuple(
        f"{value.view}::{name}"
        for value in values
        for name in value.column_names
    )
    source_by_path = {
        source.path: source for value in values for source in value.sources
    }
    if sum(len(value.sources) for value in values) != len(source_by_path):
        for value in values:
            for source in value.sources:
                if source_by_path[source.path] != source:
                    raise QualityProbePreparationError("Conflicting source commitments")
    sources = tuple(
        source_by_path[path]
        for path in sorted(source_by_path, key=lambda value: value.encode("utf-8"))
    )
    return _freeze_owned_feature_matrix(
        family=first.family,
        view=view,
        values=np.column_stack([value.values for value in values]),
        row_keys=first.row_keys,
        column_names=columns,
        sources=sources,
    )


def reject_truth_input(*_args: object, **_kwargs: object) -> None:
    """Tripwire used by callers/tests: truth is not an input to this module."""

    raise QualityProbePreparationError(
        "Truth cannot be opened by the label-free feature preparer"
    )
