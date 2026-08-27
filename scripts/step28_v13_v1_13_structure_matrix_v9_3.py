#!/usr/bin/env python3
"""Frozen V9.3 shortcut-audit structure matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import step28_v13_common as common


VERSION = "2026-08-26-step28-v13-v1-13-structure-matrix-v9-3"
PAIR_KEY_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
SELLER_SLOT_RAW_FIELDS = (
    "seller_pair_ordinal",
    "seller_slot_min",
    "seller_slot_max",
    "seller_slot_diff",
    "seller_slot_sum",
)
ROLE_COUNT = 4
CATEGORY_COUNT = 8
TITLE_TEMPLATE_COUNT = 8
DESCRIPTION_TEMPLATE_COUNT = 8
SERVICE_COUNT = 6
DELIVERY_COUNT = 6
TIME_BUCKET_COUNT = 4
NOISE_SLOT_COUNT = 28
SELLER_SLOT_COUNT = 28
SELLER_PAIR_COUNT = 378
MARKET_PAIR_ORDINAL_WIDTH = 9
TREATMENT_WIDTH = 3
CANDIDATE_WIDTH = 32
FALLBACK_WIDTH = 1


class StructureMatrixV93Error(common.ContractError):
    """Raised when a structure row or matrix drifts from the frozen schema."""


def _symmetric_field_names(prefix: str, width: int) -> tuple[str, ...]:
    return tuple(
        name
        for index in range(width)
        for name in (
            f"{prefix}_{index:02d}_absdiff",
            f"{prefix}_{index:02d}_sum",
        )
    )


NOISE_VISIBLE_RAW_FIELDS = (
    "noise_pair_ordinal",
    "noise_slot_min",
    "noise_slot_max",
    "noise_slot_diff",
    "noise_slot_sum",
    "item_count_absdiff",
    "item_count_sum",
    "controller_size_min",
    "controller_size_max",
    "controller_size_diff",
    "controller_size_sum",
    "market_pair_ordinal",
    "candidate_index",
    "collision_fallback_type",
    "registered_treatment",
    "registered_endpoint_count",
    "registered_clone_endpoint_count",
    "registered_semantic_endpoint_count",
    "registered_item_ordinal_min",
    "registered_item_ordinal_max",
    "variation_min_absdiff",
    "variation_min_sum",
    "variation_max_absdiff",
    "variation_max_sum",
    *_symmetric_field_names("role", ROLE_COUNT),
    "title_present_mask_min",
    "title_present_mask_max",
    "title_present_mask_diff",
    "title_present_mask_sum",
    "description_present_mask_min",
    "description_present_mask_max",
    "description_present_mask_diff",
    "description_present_mask_sum",
    "joint_empty_mask_min",
    "joint_empty_mask_max",
    "joint_empty_mask_diff",
    "joint_empty_mask_sum",
    *_symmetric_field_names("category", CATEGORY_COUNT),
    *_symmetric_field_names("title_template", TITLE_TEMPLATE_COUNT),
    *_symmetric_field_names("description_template", DESCRIPTION_TEMPLATE_COUNT),
    *_symmetric_field_names("service", SERVICE_COUNT),
    *_symmetric_field_names("delivery", DELIVERY_COUNT),
    *_symmetric_field_names("time_bucket", TIME_BUCKET_COUNT),
)


def seller_matrix_feature_names() -> tuple[str, ...]:
    return (
        *(f"seller_pair_{index:03d}" for index in range(SELLER_PAIR_COUNT)),
        "seller_slot_min",
        "seller_slot_max",
        "seller_slot_diff",
        "seller_slot_sum",
    )


def noise_matrix_feature_names() -> tuple[str, ...]:
    passthrough = tuple(
        field
        for field in NOISE_VISIBLE_RAW_FIELDS
        if field
        not in {
            "noise_pair_ordinal",
            "market_pair_ordinal",
            "candidate_index",
            "collision_fallback_type",
            "registered_treatment",
        }
    )
    return (
        *(f"noise_pair_{index:03d}" for index in range(SELLER_PAIR_COUNT)),
        *(f"market_pair_ordinal_{index:02d}" for index in range(MARKET_PAIR_ORDINAL_WIDTH)),
        *(f"candidate_index_{index:02d}" for index in range(CANDIDATE_WIDTH)),
        *(f"collision_fallback_type_{index:02d}" for index in range(FALLBACK_WIDTH)),
        *(f"registered_treatment_{index:02d}" for index in range(TREATMENT_WIDTH)),
        *passthrough,
    )


def _validate_row_schema(
    row: Mapping[str, Any], *, raw_fields: Sequence[str], label: str
) -> None:
    expected = (*PAIR_KEY_FIELDS, *raw_fields)
    if tuple(row) != expected:
        observed = tuple(row)
        first = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(expected, observed, strict=False)
                )
                if left != right
            ),
            min(len(expected), len(observed)),
        )
        expected_name = expected[first] if first < len(expected) else "<end>"
        observed_name = observed[first] if first < len(observed) else "<end>"
        raise StructureMatrixV93Error(
            f"{label} raw column order drift at {first}: "
            f"expected={expected_name} observed={observed_name}"
        )
    for field in raw_fields:
        if type(row[field]) is not int:
            raise StructureMatrixV93Error(f"{label} field is not an integer: {field}")


def validate_world_rows(
    rows: Sequence[Mapping[str, Any]], *, raw_fields: Sequence[str], label: str
) -> None:
    if len(rows) != SELLER_PAIR_COUNT:
        raise StructureMatrixV93Error(f"{label} row count is not 378")
    pair_uids: set[str] = set()
    world_uids: set[str] = set()
    for row in rows:
        _validate_row_schema(row, raw_fields=raw_fields, label=label)
        pair_uid = str(row["canonical_pair_uid"])
        if pair_uid in pair_uids:
            raise StructureMatrixV93Error(f"{label} pair UID collision")
        pair_uids.add(pair_uid)
        world_uids.add(str(row["world_uid"]))
    if len(world_uids) != 1:
        raise StructureMatrixV93Error(f"{label} crosses world boundaries")


def _one_hot(value: int, width: int, *, label: str) -> list[float]:
    if type(value) is not int or not 0 <= value < width:
        raise StructureMatrixV93Error(f"{label} categorical value is outside its domain")
    output = [0.0] * width
    output[value] = 1.0
    return output


def seller_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    validate_world_rows(rows, raw_fields=SELLER_SLOT_RAW_FIELDS, label="seller-slot")
    matrix = np.empty((SELLER_PAIR_COUNT, len(seller_matrix_feature_names())), dtype=np.float64)
    for row_index, row in enumerate(rows):
        matrix[row_index] = [
            *_one_hot(int(row["seller_pair_ordinal"]), SELLER_PAIR_COUNT, label="seller-pair"),
            float(row["seller_slot_min"]),
            float(row["seller_slot_max"]),
            float(row["seller_slot_diff"]),
            float(row["seller_slot_sum"]),
        ]
    return matrix


def noise_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    validate_world_rows(rows, raw_fields=NOISE_VISIBLE_RAW_FIELDS, label="noise-visible")
    names = noise_matrix_feature_names()
    passthrough = tuple(
        field
        for field in NOISE_VISIBLE_RAW_FIELDS
        if field
        not in {
            "noise_pair_ordinal",
            "market_pair_ordinal",
            "candidate_index",
            "collision_fallback_type",
            "registered_treatment",
        }
    )
    matrix = np.empty((SELLER_PAIR_COUNT, len(names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        matrix[row_index] = [
            *_one_hot(int(row["noise_pair_ordinal"]), SELLER_PAIR_COUNT, label="noise-pair"),
            *_one_hot(int(row["market_pair_ordinal"]), MARKET_PAIR_ORDINAL_WIDTH, label="market-pair"),
            *_one_hot(int(row["candidate_index"]), CANDIDATE_WIDTH, label="candidate-index"),
            *_one_hot(int(row["collision_fallback_type"]), FALLBACK_WIDTH, label="fallback-type"),
            *_one_hot(int(row["registered_treatment"]), TREATMENT_WIDTH, label="treatment"),
            *(float(row[field]) for field in passthrough),
        ]
    if not np.isfinite(matrix).all():
        raise StructureMatrixV93Error("Noise-visible matrix contains non-finite values")
    return matrix


def contract_payload() -> dict[str, Any]:
    seller_names = list(seller_matrix_feature_names())
    noise_names = list(noise_matrix_feature_names())
    return {
        "version": VERSION,
        "pair_key_fields": list(PAIR_KEY_FIELDS),
        "row_count_per_world": SELLER_PAIR_COUNT,
        "seller_slot_raw_fields": list(SELLER_SLOT_RAW_FIELDS),
        "seller_matrix_feature_names": seller_names,
        "seller_matrix_feature_count": len(seller_names),
        "seller_matrix_feature_names_sha256": common.canonical_sha256(seller_names),
        "noise_visible_raw_fields": list(NOISE_VISIBLE_RAW_FIELDS),
        "noise_matrix_feature_names": noise_names,
        "noise_matrix_feature_count": len(noise_names),
        "noise_matrix_feature_names_sha256": common.canonical_sha256(noise_names),
        "dtype": "little-endian float64",
        "normalization": "StandardScaler fitted on all train worlds inside each fixed probe; tree receives raw float64",
        "empty_set_convention": "not applicable; every world has 28 sellers and 378 complete unordered pairs",
        "eligibility_mask": "all 378 pairs in every train and development world",
    }
