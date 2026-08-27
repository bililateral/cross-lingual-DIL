#!/usr/bin/env python3
"""Build the fixed 14-column V9.4 model-visible nuisance matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from typing import Any

import numpy as np


VERSION = "2026-08-27-step28-v13-v1-13-model-visible-matrix-v9-4"
KEY_FIELDS = ("world_uid", "canonical_pair_uid")
SELLER_FEATURES = (
    "item_count",
    "title_missing_rate",
    "description_missing_rate",
    "time_bucket_probability_00",
    "time_bucket_probability_01",
    "time_bucket_probability_02",
    "time_bucket_probability_03",
)
PAIR_FEATURES = tuple(f"absdiff__{name}" for name in SELLER_FEATURES) + tuple(
    f"sum__{name}" for name in SELLER_FEATURES
)
ROW_FIELDS = (*KEY_FIELDS, *PAIR_FEATURES)
FORBIDDEN_FIELD_FRAGMENTS = (
    "controller",
    "label",
    "treatment",
    "role",
    "mechanism",
    "candidate",
    "fallback",
    "seller_slot",
    "noise_slot",
    "pair_ordinal",
    "item_ordinal",
    "identity",
    "author_style",
    "model_score",
)


class ModelVisibleMatrixV94Error(ValueError):
    """Raised when the V9.4 public nuisance matrix contract drifts."""


@dataclass(frozen=True)
class FrozenMatrix:
    row_keys: tuple[tuple[str, str], ...]
    column_names: tuple[str, ...]
    values: np.ndarray
    raw_f8_c_sha256: str
    row_keys_sha256: str
    column_names_sha256: str
    joint_commitment_sha256: str


def _parse_fixed_float(value: Any, *, field: str) -> float:
    if not isinstance(value, str):
        raise ModelVisibleMatrixV94Error(f"{field} must be fixed text")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ModelVisibleMatrixV94Error(f"{field} is not float64") from error
    if not math.isfinite(parsed) or format(parsed, ".12f") != value:
        raise ModelVisibleMatrixV94Error(f"{field} serialization drift")
    if parsed == 0.0 and value != "0.000000000000":
        raise ModelVisibleMatrixV94Error(f"{field} noncanonical zero encoding")
    return parsed


def _validate_pair_semantics(values: Mapping[str, float]) -> None:
    count_difference = values["absdiff__item_count"]
    count_sum = values["sum__item_count"]
    if (
        not count_difference.is_integer()
        or not count_sum.is_integer()
        or not 0.0 <= count_difference <= 6.0
        or not 4.0 <= count_sum <= 16.0
        or int(count_sum - count_difference) % 2 != 0
        or not 2.0 <= (count_sum - count_difference) / 2.0 <= 8.0
        or not 2.0 <= (count_sum + count_difference) / 2.0 <= 8.0
    ):
        raise ModelVisibleMatrixV94Error("Item-count pair semantics drift")
    for name in SELLER_FEATURES[1:]:
        difference = values[f"absdiff__{name}"]
        total = values[f"sum__{name}"]
        tolerance = 5e-12
        if (
            difference < 0.0
            or difference > 1.0 + tolerance
            or total < -tolerance
            or total > 2.0 + tolerance
            or difference > total + tolerance
            or difference > 2.0 - total + tolerance
        ):
            raise ModelVisibleMatrixV94Error(
                f"Bounded pair semantics drift: {name}"
            )
    time_total = sum(
        values[f"sum__time_bucket_probability_{index:02d}"]
        for index in range(4)
    )
    if not math.isclose(time_total, 2.0, rel_tol=0.0, abs_tol=2e-11):
        raise ModelVisibleMatrixV94Error("Time-bucket probabilities do not sum to two")
    low_count = int((count_sum - count_difference) / 2.0)
    high_count = int((count_sum + count_difference) / 2.0)
    count_orientations = {(low_count, high_count), (high_count, low_count)}
    if not any(
        _has_rate_witness(values, left_count=left_count, right_count=right_count)
        for left_count, right_count in count_orientations
    ):
        raise ModelVisibleMatrixV94Error(
            "Pair rates have no integer-item endpoint witness"
        )


def _bounded(value: float) -> float:
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=5e-12):
        return 0.0
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=5e-12):
        return 1.0
    return value


def _rate_orientations(total: float, difference: float) -> tuple[tuple[float, float], ...]:
    low = _bounded((total - difference) / 2.0)
    high = _bounded((total + difference) / 2.0)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=5e-13):
        return ((low, high),)
    return ((low, high), (high, low))


def _is_item_fraction(value: float, count: int) -> bool:
    return 0.0 <= value <= 1.0 and math.isclose(
        value * count,
        round(value * count),
        rel_tol=0.0,
        abs_tol=5e-12,
    )


def _has_rate_witness(
    values: Mapping[str, float], *, left_count: int, right_count: int
) -> bool:
    for name in ("title_missing_rate", "description_missing_rate"):
        orientations = _rate_orientations(
            values[f"sum__{name}"], values[f"absdiff__{name}"]
        )
        if not any(
            _is_item_fraction(left, left_count)
            and _is_item_fraction(right, right_count)
            for left, right in orientations
        ):
            return False
    time_orientations = [
        _rate_orientations(
            values[f"sum__time_bucket_probability_{index:02d}"],
            values[f"absdiff__time_bucket_probability_{index:02d}"],
        )
        for index in range(4)
    ]
    for oriented_buckets in product(*time_orientations):
        left = tuple(bucket[0] for bucket in oriented_buckets)
        right = tuple(bucket[1] for bucket in oriented_buckets)
        if (
            math.isclose(sum(left), 1.0, rel_tol=0.0, abs_tol=2e-11)
            and math.isclose(sum(right), 1.0, rel_tol=0.0, abs_tol=2e-11)
            and all(_is_item_fraction(value, left_count) for value in left)
            and all(_is_item_fraction(value, right_count) for value in right)
        ):
            return True
    return False


def _seller_values(source: Mapping[str, float], *, side: str) -> tuple[float, ...]:
    if not isinstance(source, Mapping):
        raise ModelVisibleMatrixV94Error("Seller nuisance row is not a mapping")
    if tuple(source) != SELLER_FEATURES:
        raise ModelVisibleMatrixV94Error("Seller nuisance schema/order drift")
    if any(
        isinstance(source[name], bool)
        or not isinstance(source[name], (int, float, np.integer, np.floating))
        for name in SELLER_FEATURES
    ):
        raise ModelVisibleMatrixV94Error(
            f"{side} seller nuisance vector has a nonnumeric value"
        )
    values = tuple(float(source[name]) for name in SELLER_FEATURES)
    if not all(math.isfinite(value) for value in values):
        raise ModelVisibleMatrixV94Error(f"{side} seller nuisance vector is nonfinite")
    item_count = values[0]
    if not item_count.is_integer() or not 2.0 <= item_count <= 8.0:
        raise ModelVisibleMatrixV94Error(f"{side} seller item count is outside [2,8]")
    rates = values[1:]
    if any(not 0.0 <= value <= 1.0 for value in rates):
        raise ModelVisibleMatrixV94Error(f"{side} seller rate is outside [0,1]")
    if not math.isclose(sum(values[3:]), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ModelVisibleMatrixV94Error(f"{side} seller time probabilities do not sum to one")
    if any(
        not math.isclose(value * item_count, round(value * item_count), rel_tol=0.0, abs_tol=1e-10)
        for value in rates
    ):
        raise ModelVisibleMatrixV94Error(f"{side} seller rate is not an item-count fraction")
    return values


def pair_feature_row(
    *,
    world_uid: str,
    canonical_pair_uid: str,
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, str]:
    """Create one fixed-format row from two seven-value public vectors."""

    if not isinstance(world_uid, str) or not world_uid:
        raise ModelVisibleMatrixV94Error("World key is empty")
    if not isinstance(canonical_pair_uid, str) or not canonical_pair_uid:
        raise ModelVisibleMatrixV94Error("Pair key is empty")
    left_values = _seller_values(left, side="left")
    right_values = _seller_values(right, side="right")
    values = tuple(abs(a - b) for a, b in zip(left_values, right_values, strict=True))
    values += tuple(a + b for a, b in zip(left_values, right_values, strict=True))
    output = {
        "world_uid": world_uid,
        "canonical_pair_uid": canonical_pair_uid,
        **{
            name: format(value, ".12f")
            for name, value in zip(PAIR_FEATURES, values, strict=True)
        },
    }
    parsed = {name: float(output[name]) for name in PAIR_FEATURES}
    _validate_pair_semantics(parsed)
    return output


def freeze_matrix(
    rows: Sequence[Mapping[str, Any]], *, expected_row_count: int
) -> FrozenMatrix:
    """Validate and freeze exact ordered rows without reading labels."""

    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise ModelVisibleMatrixV94Error("Expected row count is invalid")
    if len(rows) != expected_row_count:
        raise ModelVisibleMatrixV94Error("Matrix row count drift")
    row_keys: list[tuple[str, str]] = []
    matrix = np.empty((len(rows), len(PAIR_FEATURES)), dtype=np.dtype("<f8"))
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ModelVisibleMatrixV94Error("Matrix row is not a mapping")
        if tuple(source) != ROW_FIELDS:
            raise ModelVisibleMatrixV94Error("Matrix row schema/order drift")
        world_uid = source["world_uid"]
        pair_uid = source["canonical_pair_uid"]
        if not isinstance(world_uid, str) or not isinstance(pair_uid, str):
            raise ModelVisibleMatrixV94Error("Matrix key type drift")
        if not world_uid or not pair_uid:
            raise ModelVisibleMatrixV94Error("Matrix key is empty")
        row_keys.append((world_uid, pair_uid))
        parsed = {
            name: _parse_fixed_float(source[name], field=name)
            for name in PAIR_FEATURES
        }
        _validate_pair_semantics(parsed)
        matrix[index] = [parsed[name] for name in PAIR_FEATURES]
    expected_order = sorted(
        row_keys, key=lambda key: (key[0].encode("utf-8"), key[1].encode("utf-8"))
    )
    if row_keys != expected_order or len(row_keys) != len(set(row_keys)):
        raise ModelVisibleMatrixV94Error("Matrix row keys are not unique canonical order")
    if not np.isfinite(matrix).all():
        raise ModelVisibleMatrixV94Error("Matrix contains nonfinite values")
    matrix = np.frombuffer(matrix.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        matrix.shape
    )
    raw_f8_c_sha256 = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    row_keys_sha256 = hashlib.sha256(
        json.dumps(
            row_keys, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    column_names_sha256 = hashlib.sha256(
        json.dumps(
            PAIR_FEATURES, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    joint_commitment_sha256 = hashlib.sha256(
        json.dumps(
            {
                "column_names_sha256": column_names_sha256,
                "dtype": "little-endian float64",
                "raw_f8_c_sha256": raw_f8_c_sha256,
                "row_keys_sha256": row_keys_sha256,
                "shape": [len(row_keys), len(PAIR_FEATURES)],
                "version": VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return FrozenMatrix(
        row_keys=tuple(row_keys),
        column_names=PAIR_FEATURES,
        values=matrix,
        raw_f8_c_sha256=raw_f8_c_sha256,
        row_keys_sha256=row_keys_sha256,
        column_names_sha256=column_names_sha256,
        joint_commitment_sha256=joint_commitment_sha256,
    )


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "key_fields": list(KEY_FIELDS),
        "seller_features": list(SELLER_FEATURES),
        "pair_features": list(PAIR_FEATURES),
        "pair_feature_count": len(PAIR_FEATURES),
        "forbidden_field_fragments": list(FORBIDDEN_FIELD_FRAGMENTS),
        "dtype": "little-endian float64",
        "row_order": "world_uid_utf8_then_canonical_pair_uid_utf8",
        "labels_read": False,
    }
