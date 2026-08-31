#!/usr/bin/env python3
"""Shared, label-free base24 feature construction for English replay and Chinese projection."""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as predecessor_common
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common


MODEL_PROFILE_FIELDS = (
    "seller_uid",
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
    "item_count",
    "title_length_stats",
    "description_length_stats",
    "style_stats",
)


def _load_step7_source_module():
    scripts = str(common.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import step7_v3_1_source_data as source

    return source


def split_concat_and_normalize(value: str) -> list[str]:
    source = _load_step7_source_module()
    return sorted(
        {
            normalized
            for segment in source.split_concat(str(value))
            if (normalized := source.normalize_signature(segment))
        }
    )


def project_model_profile(
    row: Mapping[str, Any], *, require_exact_schema: bool = True
) -> dict[str, Any]:
    """Project a model-visible seller profile into the legacy18 primitives."""

    if require_exact_schema:
        if tuple(row) != MODEL_PROFILE_FIELDS:
            raise common.ModelTrainingContractError(
                "Model seller-profile schema/order drift"
            )
    elif any(name not in row for name in MODEL_PROFILE_FIELDS):
        raise common.ModelTrainingContractError("Model seller-profile field is missing")
    title_stats = row["title_length_stats"]
    description_stats = row["description_length_stats"]
    style = row["style_stats"]
    if not isinstance(title_stats, Mapping) or not isinstance(
        description_stats, Mapping
    ) or not isinstance(style, Mapping):
        raise common.ModelTrainingContractError("Model seller-profile nested schema drift")
    numeric = {
        "item_count": float(row["item_count"]),
        "title_length_median": float(title_stats["median"]),
        "description_length_median": float(description_stats["median"]),
        "digit_ratio_mean": float(style["digit_ratio_mean"]),
        "punct_ratio_mean": float(style["punct_ratio_mean"]),
        "repeated_title_share": float(style["repeated_title_share"]),
        "repeated_description_share": float(style["repeated_description_share"]),
        "max_category_share": float(style["max_category_share"]),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise common.ModelTrainingContractError("Model seller profile is non-finite")
    return {
        "seller_uid": str(row["seller_uid"]),
        "clean_categories": split_concat_and_normalize(row["category_concat_top"]),
        "clean_titles": sorted(
            set(split_concat_and_normalize(row["signature_title_concat"]))
            | set(split_concat_and_normalize(row["title_concat_top"]))
        ),
        "clean_descriptions": sorted(
            set(split_concat_and_normalize(row["signature_description_concat"]))
            | set(split_concat_and_normalize(row["description_concat_top"]))
        ),
        "numeric_profile": numeric,
    }


def legacy18_row(
    pair: Mapping[str, str],
    seller_records: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Build one legacy18 row using the frozen Step7 feature semantics."""

    source = _load_step7_source_module()
    try:
        left = seller_records[pair["seller_uid_left"]]
        right = seller_records[pair["seller_uid_right"]]
    except KeyError as exc:
        raise common.ModelTrainingContractError(
            "Legacy18 pair endpoint is absent from seller records"
        ) from exc
    left_categories = set(left["clean_categories"])
    right_categories = set(right["clean_categories"])
    shared_categories = left_categories & right_categories
    shared_titles = set(left["clean_titles"]) & set(right["clean_titles"])
    shared_descriptions = set(left["clean_descriptions"]) & set(
        right["clean_descriptions"]
    )
    train_seller_count = int(reference["train_seller_count"])
    title_sum, title_mean = source.shared_idf(
        shared_titles, reference["title_df"], train_seller_count
    )
    description_sum, description_mean = source.shared_idf(
        shared_descriptions,
        reference["description_df"],
        train_seller_count,
    )
    values = {
        "clean_category_jaccard": source.jaccard(
            left_categories, right_categories
        ),
        "clean_shared_title_bool": int(bool(shared_titles)),
        "clean_shared_description_bool": int(bool(shared_descriptions)),
        "clean_shared_title_count_capped": min(len(shared_titles), 5),
        "clean_shared_description_count_capped": min(len(shared_descriptions), 5),
        "clean_shared_category_count_capped": min(len(shared_categories), 5),
        "clean_shared_title_idf_sum": title_sum,
        "clean_shared_description_idf_sum": description_sum,
        "clean_shared_title_idf_mean": title_mean,
        "clean_shared_description_idf_mean": description_mean,
    }
    for name in source.NUMERIC_PROFILE_FIELDS:
        left_percentile = source.empirical_percentile(
            reference["numeric_references"][name], left["numeric_profile"][name]
        )
        right_percentile = source.empirical_percentile(
            reference["numeric_references"][name], right["numeric_profile"][name]
        )
        values[f"{name}_train_percentile_gap_abs"] = abs(
            left_percentile - right_percentile
        )
    if list(values) != list(feature_names):
        raise common.ModelTrainingContractError("legacy18 feature order drift")
    result = np.asarray([values[name] for name in feature_names], dtype="<f8")
    if not np.isfinite(result).all():
        raise common.ModelTrainingContractError("legacy18 row is non-finite")
    return np.ascontiguousarray(result, dtype="<f8")


def legacy18_matrix(
    pairs: Sequence[Mapping[str, str]],
    seller_records: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    if not pairs:
        raise common.ModelTrainingContractError("Cannot build an empty legacy18 matrix")
    matrix = np.vstack(
        [legacy18_row(pair, seller_records, reference, feature_names) for pair in pairs]
    )
    matrix = np.ascontiguousarray(matrix, dtype="<f8")
    if matrix.shape != (len(pairs), len(feature_names)):
        raise common.ModelTrainingContractError("legacy18 matrix shape drift")
    return matrix


def combine_base24(
    legacy18: np.ndarray,
    labse6: np.ndarray,
) -> np.ndarray:
    left = np.asarray(legacy18, dtype=np.float64)
    right = np.asarray(labse6, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise common.ModelTrainingContractError("Base24 feature inputs must be matrices")
    if left.shape[0] != right.shape[0] or left.shape[1] != 18 or right.shape[1] != 6:
        raise common.ModelTrainingContractError("Base24 feature shape drift")
    if np.isinf(left).any() or np.isinf(right).any():
        raise common.ModelTrainingContractError("Base24 contains infinity")
    return np.ascontiguousarray(np.column_stack((left, right)), dtype="<f8")


@lru_cache(maxsize=1)
def _reconstruct_frozen_english_public_cached(
    successor_policy_canonical_self_hash: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    scripts = str(common.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import step7_v4_common as step7_common
    import step7_v4_prepare_source_data as step7_prepare

    predecessor = predecessor_common.load_policy()
    step7_policy = step7_common.load_policy()
    _parent, public, pairs, _safe = step7_prepare.replay_parent_public(step7_policy)
    reference = public.get("reference")
    seller_records = public.get("seller_records")
    if not isinstance(reference, dict) or not isinstance(seller_records, dict):
        raise common.ModelTrainingContractError("Step7 public reconstruction schema drift")
    expected = predecessor["frozen_english_reference"]
    if (
        int(reference.get("train_seller_count", -1)) != expected["fit_seller_count"]
        or reference.get("train_seller_uid_sha256") != expected["seller_uid_sha256"]
        or common.canonical_sha256(reference) != expected["feature_reference_sha256"]
    ):
        raise common.ModelTrainingContractError("Frozen English reference replay drift")
    if len(pairs) != 733 or len({row["pair_uid"] for row in pairs}) != 733:
        raise common.ModelTrainingContractError("Frozen English pair reconstruction drift")
    if successor_policy_canonical_self_hash != common.EXPECTED_POLICY_CANONICAL_SELF_HASH:
        raise common.ModelTrainingContractError("Shared builder successor policy drift")
    if 151 != sum(row["split_name"] == "valid" for row in pairs):
        raise common.ModelTrainingContractError("Frozen English valid split drift")
    return reference, seller_records, pairs


def reconstruct_frozen_english_public(
    successor_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    """Rebuild Step7 clean seller primitives and reference without supervision."""

    claimed = str(successor_policy.get("canonical_self_hash", ""))
    if claimed != common.EXPECTED_POLICY_CANONICAL_SELF_HASH:
        raise common.ModelTrainingContractError("Shared builder received another policy")
    return _reconstruct_frozen_english_public_cached(claimed)
