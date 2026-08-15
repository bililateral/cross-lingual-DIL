#!/usr/bin/env python3
"""Frozen label-free text shortcut views for Step28-v13 v1.13 v9."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import unicodedata
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

import step28_v13_v1_13_scientific_dataset_builder_v9 as dataset_builder


VERSION = "2026-08-14-step28-v13-v1-13-quality-text-probe-views-v9"
VISIBLE_PROFILE_FIELDS = dataset_builder.MODEL_PROFILE_TEXT_FIELDS
FIXED_SUPPORT_FIELDS = ("title", "description")
PRODUCTION_NUMERIC_FIELDS = (
    "item_count",
    "title_length_median",
    "description_length_median",
    "digit_ratio_mean",
    "punct_ratio_mean",
    "repeated_title_share",
    "repeated_description_share",
    "max_category_share",
)
SURFACE_METRICS = (
    "codepoint_length_absdiff",
    "codepoint_length_sum",
    "newline_count_absdiff",
    "newline_count_sum",
    "unicode_punctuation_count_absdiff",
    "unicode_punctuation_count_sum",
    "ascii_whitespace_count_absdiff",
    "ascii_whitespace_count_sum",
    "unicode_decimal_digit_count_absdiff",
    "unicode_decimal_digit_count_sum",
    "empty_both",
    "empty_xor",
)
FIXED_SUPPORT_SURFACE_METRICS = (
    "codepoint_length_absdiff",
    "codepoint_length_sum",
    "newline_count_absdiff",
    "newline_count_sum",
    "unicode_punctuation_count_absdiff",
    "unicode_punctuation_count_sum",
    "ascii_whitespace_count_absdiff",
    "ascii_whitespace_count_sum",
    "unicode_decimal_digit_count_absdiff",
    "unicode_decimal_digit_count_sum",
    "empty_rate_absdiff",
    "empty_rate_sum",
)
PUNCTUATION_CATEGORIES = frozenset({"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"})
ASCII_WHITESPACE = frozenset("\t\n\v\f\r ")
COMBINED_SEPARATOR = "\n␞\n"
VIEW_ORDER = (
    "fs_full",
    "fs_title",
    "fs_template_surface",
    "p_full",
    "p_topic",
    "p_template_surface",
    "u_joint_full",
)
EXPECTED_WIDTHS = (33, 14, 30, 75, 14, 56, 124)
EXPECTED_NAME_HASHES = {
    "fs_full": "e7e929d856423d03951612884bbffd57649190ecf5c414b76819e4129265957b",
    "fs_title": "71e72e4c3cf6ea36d78477acb0617f5436d2813c201998b13ae051b55fe9afe8",
    "fs_template_surface": "4ef95fb703e708e59f5334636c1bae539ed44dc35293832d23a908fea9252606",
    "p_full": "1c08e76c0f74ff126a0d3f722afa652c36393d3e30f200077c4c13c91820ec8b",
    "p_topic": "6a201d2afbc4b1579cef20e53ae81703924b69ceba53df2b2913002447d0e891",
    "p_template_surface": "57fd57edf108f3ad09f50e88c9b3ee23644dcc1f3abff53ddc86a2d051cd4156",
    "u_joint_full": "420333af4f991424cd7d65ebeeaeb0aafd43ea612eba8398852c14c25525a745",
}
ENDPOINT_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)


class QualityTextProbeViewError(ValueError):
    """Raised when a frozen label-free text-view contract drifts."""


def _required_text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise QualityTextProbeViewError(f"{name} string type/value drift")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def word12_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    ascii_run: list[str] = []

    def flush_ascii() -> None:
        if ascii_run:
            tokens.append("".join(ascii_run).lower())
            ascii_run.clear()

    for character in text:
        codepoint = ord(character)
        if (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
            or "0" <= character <= "9"
        ):
            ascii_run.append(character)
        elif (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            flush_ascii()
            tokens.append(character)
        else:
            flush_ascii()
    flush_ascii()
    return tokens


def template_mask(text: str) -> str:
    output: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L"):
            output.append("字")
        elif category == "Nd":
            output.append("数")
        else:
            output.append(character)
    return "".join(output)


def _char_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=None,
        ngram_range=(3, 3),
        analyzer="char",
        n_features=65536,
        binary=False,
        norm="l2",
        alternate_sign=False,
        dtype=np.float64,
    )


def _word_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=word12_tokens,
        stop_words=None,
        token_pattern=None,
        ngram_range=(1, 2),
        analyzer="word",
        n_features=65536,
        binary=False,
        norm="l2",
        alternate_sign=False,
        dtype=np.float64,
    )


def _pair_cosines(
    matrix: sparse.csr_matrix,
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    try:
        left = np.fromiter(
            (seller_row[row["seller_uid_left"]] for row in endpoints),
            dtype=np.int64,
            count=len(endpoints),
        )
        right = np.fromiter(
            (seller_row[row["seller_uid_right"]] for row in endpoints),
            dtype=np.int64,
            count=len(endpoints),
        )
    except KeyError as exc:
        raise QualityTextProbeViewError("Text seller/endpoint join drift") from exc
    values = np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
    if not np.isfinite(values).all():
        raise QualityTextProbeViewError("Text cosine is nonfinite")
    return values.astype(np.float64, copy=False)


def _surface_counts(text: str) -> tuple[int, int, int, int, int, int]:
    return (
        len(text),
        text.count("\n"),
        sum(
            unicodedata.category(character) in PUNCTUATION_CATEGORIES
            for character in text
        ),
        sum(character in ASCII_WHITESPACE for character in text),
        sum(unicodedata.category(character) == "Nd" for character in text),
        int(not text),
    )


def _surface_pair_features_from_counts(
    counts: Sequence[Sequence[float]],
    *,
    seller_row: Mapping[str, int],
    endpoints: Sequence[Mapping[str, Any]],
    empty_is_rate: bool,
) -> np.ndarray:
    if len(counts) != len(seller_row) or any(len(row) != 6 for row in counts):
        raise QualityTextProbeViewError("Surface-count matrix shape drift")
    width = 12
    output = np.empty((len(endpoints), width), dtype=np.float64)
    for index, row in enumerate(endpoints):
        left = counts[seller_row[row["seller_uid_left"]]]
        right = counts[seller_row[row["seller_uid_right"]]]
        if any(not math.isfinite(float(value)) for value in (*left, *right)):
            raise QualityTextProbeViewError("Surface count is nonfinite")
        values: list[float] = []
        if empty_is_rate:
            if not (0 <= float(left[5]) <= 1 and 0 <= float(right[5]) <= 1):
                raise QualityTextProbeViewError("Fixed-support empty rate drift")
            for offset in range(6):
                values.extend(
                    (
                        abs(float(left[offset]) - float(right[offset])),
                        float(left[offset]) + float(right[offset]),
                    )
                )
        else:
            for offset in range(5):
                values.extend(
                    (
                        abs(float(left[offset]) - float(right[offset])),
                        float(left[offset]) + float(right[offset]),
                    )
                )
            values.extend(
                (
                    float(bool(left[5]) and bool(right[5])),
                    float(bool(left[5]) ^ bool(right[5])),
                )
            )
        output[index] = values
    return output


def _seller_slot_matrix(
    *,
    texts_by_seller: Mapping[str, Sequence[str]],
    seller_uids: Sequence[str],
    vectorizer: HashingVectorizer,
    mask: bool = False,
) -> sparse.csr_matrix:
    rows: list[sparse.csr_matrix] = []
    for seller_uid in seller_uids:
        values = list(texts_by_seller[seller_uid])
        if not values:
            raise QualityTextProbeViewError("Fixed-support seller has no item slots")
        if mask:
            values = [template_mask(value) for value in values]
        values.sort(key=lambda value: value.encode("utf-8"))
        transformed = vectorizer.transform(values).tocsr()
        rows.append(sparse.csr_matrix(transformed.sum(axis=0), dtype=np.float64))
    return normalize(sparse.vstack(rows, format="csr"), norm="l2", axis=1)


def _build_fixed_support_views(
    *,
    items: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    item_uids: set[str] = set()
    by_seller: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in items:
        if (
            not isinstance(row, Mapping)
            or set(row) != set(dataset_builder.MODEL_REDACTED_ITEM_FIELDS)
        ):
            raise QualityTextProbeViewError("Fixed-support item schema drift")
        item_uid = _required_text(row["item_uid"], name="item UID")
        seller_uid = _required_text(row["seller_uid"], name="seller UID")
        _required_text(row["world_uid"], name="world UID")
        for field in FIXED_SUPPORT_FIELDS:
            _required_text(row[field], name=f"item {field}", allow_empty=True)
        if item_uid in item_uids:
            raise QualityTextProbeViewError("Fixed-support item key drift")
        item_uids.add(item_uid)
        by_seller[seller_uid].append(row)
    endpoint_sellers = {
        row[field]
        for row in endpoints
        for field in ("seller_uid_left", "seller_uid_right")
    }
    if set(by_seller) != endpoint_sellers:
        raise QualityTextProbeViewError("Fixed-support seller universe drift")
    seller_uids = sorted(endpoint_sellers, key=lambda value: value.encode("utf-8"))
    seller_row = {seller_uid: index for index, seller_uid in enumerate(seller_uids)}
    endpoint_worlds_by_seller: defaultdict[str, set[str]] = defaultdict(set)
    for row in endpoints:
        endpoint_worlds_by_seller[row["seller_uid_left"]].add(row["world_uid"])
        endpoint_worlds_by_seller[row["seller_uid_right"]].add(row["world_uid"])
    texts: dict[str, dict[str, list[str]]] = {
        field: {} for field in (*FIXED_SUPPORT_FIELDS, "item_joint")
    }
    surface_means: dict[str, list[tuple[float, ...]]] = {
        field: [] for field in FIXED_SUPPORT_FIELDS
    }
    for seller_uid in seller_uids:
        rows = by_seller[seller_uid]
        endpoint_worlds = endpoint_worlds_by_seller[seller_uid]
        item_worlds = {row["world_uid"] for row in rows}
        if len(endpoint_worlds) != 1 or item_worlds != endpoint_worlds:
            raise QualityTextProbeViewError("Fixed-support seller crosses world boundary")
        for field in FIXED_SUPPORT_FIELDS:
            values = [row[field] for row in rows]
            texts[field][seller_uid] = values
            counts = np.asarray(
                [_surface_counts(value) for value in values], dtype=np.float64
            )
            surface_means[field].append(tuple(counts.mean(axis=0).tolist()))
        texts["item_joint"][seller_uid] = [
            row["title"] + COMBINED_SEPARATOR + row["description"]
            for row in rows
        ]
    char = _char_vectorizer()
    word = _word_vectorizer()
    char_cos: dict[str, np.ndarray] = {}
    word_cos: dict[str, np.ndarray] = {}
    masked_cos: dict[str, np.ndarray] = {}
    for field in (*FIXED_SUPPORT_FIELDS, "item_joint"):
        char_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=char,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        word_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=word,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        masked_cos[field] = _pair_cosines(
            _seller_slot_matrix(
                texts_by_seller=texts[field],
                seller_uids=seller_uids,
                vectorizer=char,
                mask=True,
            ),
            seller_row=seller_row,
            endpoints=endpoints,
        )
    surfaces = {
        field: _surface_pair_features_from_counts(
            surface_means[field],
            seller_row=seller_row,
            endpoints=endpoints,
            empty_is_rate=True,
        )
        for field in FIXED_SUPPORT_FIELDS
    }
    full_similarity = np.column_stack(
        [
            char_cos["title"],
            word_cos["title"],
            char_cos["description"],
            word_cos["description"],
            char_cos["item_joint"],
            word_cos["item_joint"],
        ]
    )
    full = np.column_stack(
        [
            full_similarity,
            surfaces["title"],
            surfaces["description"],
            np.max(full_similarity, axis=1),
            np.mean(full_similarity, axis=1),
            np.mean(np.sort(full_similarity, axis=1)[:, -2:], axis=1),
        ]
    )
    full_names = tuple(
        [
            "char3_cosine__slot_title",
            "word12_cosine__slot_title",
            "char3_cosine__slot_description",
            "word12_cosine__slot_description",
            "char3_cosine__slot_item_joint",
            "word12_cosine__slot_item_joint",
        ]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
        + [
            f"{name}__slot_description_mean"
            for name in FIXED_SUPPORT_SURFACE_METRICS
        ]
        + [
            "similarity_max__fixed_slots",
            "similarity_mean__fixed_slots",
            "similarity_top2_mean__fixed_slots",
        ]
    )
    title = np.column_stack(
        [char_cos["title"], word_cos["title"], surfaces["title"]]
    )
    title_names = tuple(
        ["char3_cosine__slot_title", "word12_cosine__slot_title"]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
    )
    masked_similarity = np.column_stack(
        [
            masked_cos["title"],
            masked_cos["description"],
            masked_cos["item_joint"],
        ]
    )
    template = np.column_stack(
        [
            masked_similarity,
            surfaces["title"],
            surfaces["description"],
            np.max(masked_similarity, axis=1),
            np.mean(masked_similarity, axis=1),
            np.mean(np.sort(masked_similarity, axis=1)[:, -2:], axis=1),
        ]
    )
    template_names = tuple(
        [
            "masked_char3_cosine__slot_title",
            "masked_char3_cosine__slot_description",
            "masked_char3_cosine__slot_item_joint",
        ]
        + [f"{name}__slot_title_mean" for name in FIXED_SUPPORT_SURFACE_METRICS]
        + [
            f"{name}__slot_description_mean"
            for name in FIXED_SUPPORT_SURFACE_METRICS
        ]
        + [
            "masked_similarity_max__fixed_slots",
            "masked_similarity_mean__fixed_slots",
            "masked_similarity_top2_mean__fixed_slots",
        ]
    )
    return (
        {
            "fs_full": full,
            "fs_title": title,
            "fs_template_surface": template,
        },
        {
            "fs_full": full_names,
            "fs_title": title_names,
            "fs_template_surface": template_names,
        },
    )


def _build_production_views(
    *,
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    item_counts_by_seller: Mapping[str, int],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[str, ...]],
    np.ndarray,
    tuple[str, ...],
]:
    seller_uids: list[str] = []
    for row in profiles:
        if (
            not isinstance(row, Mapping)
            or set(row) != set(dataset_builder.MODEL_PROFILE_FIELDS)
        ):
            raise QualityTextProbeViewError("Text profile schema drift")
        seller_uids.append(
            _required_text(row["seller_uid"], name="profile seller UID")
        )
    if len(seller_uids) != len(set(seller_uids)):
        raise QualityTextProbeViewError("Text profile schema/seller collision drift")
    seller_row = {seller_uid: index for index, seller_uid in enumerate(seller_uids)}
    endpoint_sellers = {
        row[field]
        for row in endpoints
        for field in ("seller_uid_left", "seller_uid_right")
    }
    if endpoint_sellers != set(seller_uids):
        raise QualityTextProbeViewError("Text profile/endpoint seller join drift")
    if set(item_counts_by_seller) != set(seller_uids):
        raise QualityTextProbeViewError("Text profile/item seller join drift")
    for row in profiles:
        item_count = row["item_count"]
        if (
            type(item_count) is not int
            or item_count <= 0
            or item_count != item_counts_by_seller[row["seller_uid"]]
        ):
            raise QualityTextProbeViewError("Production item count drift")
    texts = {
        field: [
            _required_text(
                row[field], name=f"profile {field}", allow_empty=True
            )
            for row in profiles
        ]
        for field in VISIBLE_PROFILE_FIELDS
    }
    texts["all_fields"] = [
        COMBINED_SEPARATOR.join(
            texts[field][index] for field in VISIBLE_PROFILE_FIELDS
        )
        for index in range(len(profiles))
    ]
    text_only_fields = VISIBLE_PROFILE_FIELDS[1:]
    texts["all_text_fields"] = [
        COMBINED_SEPARATOR.join(texts[field][index] for field in text_only_fields)
        for index in range(len(profiles))
    ]
    char = _char_vectorizer()
    word = _word_vectorizer()
    char_cos: dict[str, np.ndarray] = {}
    word_cos: dict[str, np.ndarray] = {}
    for field in (*VISIBLE_PROFILE_FIELDS, "all_fields"):
        char_cos[field] = _pair_cosines(
            char.transform(texts[field]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        word_cos[field] = _pair_cosines(
            word.transform(texts[field]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
    masked_cos = {
        field: _pair_cosines(
            char.transform([template_mask(value) for value in texts[field]]).tocsr(),
            seller_row=seller_row,
            endpoints=endpoints,
        )
        for field in (*text_only_fields, "all_text_fields")
    }
    surfaces = {
        field: _surface_pair_features_from_counts(
            [_surface_counts(value) for value in texts[field]],
            seller_row=seller_row,
            endpoints=endpoints,
            empty_is_rate=False,
        )
        for field in VISIBLE_PROFILE_FIELDS
    }
    full_columns: list[np.ndarray] = []
    full_names: list[str] = []
    similarities: list[np.ndarray] = []
    for field in VISIBLE_PROFILE_FIELDS:
        full_columns.extend((char_cos[field], word_cos[field]))
        full_names.extend((f"char3_cosine__{field}", f"word12_cosine__{field}"))
        similarities.extend((char_cos[field], word_cos[field]))
    full_columns.extend((char_cos["all_fields"], word_cos["all_fields"]))
    full_names.extend(("char3_cosine__all_fields", "word12_cosine__all_fields"))
    for field in VISIBLE_PROFILE_FIELDS:
        full_columns.extend(
            surfaces[field][:, index] for index in range(len(SURFACE_METRICS))
        )
        full_names.extend(f"{name}__{field}" for name in SURFACE_METRICS)
    similarity_matrix = np.column_stack(similarities)
    full_columns.extend(
        (
            np.max(similarity_matrix, axis=1),
            np.mean(similarity_matrix, axis=1),
            np.mean(np.sort(similarity_matrix, axis=1)[:, -2:], axis=1),
        )
    )
    full_names.extend(
        (
            "similarity_max__field_char_word",
            "similarity_mean__field_char_word",
            "similarity_top2_mean__field_char_word",
        )
    )
    topic_columns: list[np.ndarray] = [
        char_cos["category_concat_top"],
        word_cos["category_concat_top"],
    ]
    topic_names = [
        "char3_cosine__category_concat_top",
        "word12_cosine__category_concat_top",
    ]
    topic_columns.extend(
        surfaces["category_concat_top"][:, index]
        for index in range(len(SURFACE_METRICS))
    )
    topic_names.extend(
        f"{name}__category_concat_top" for name in SURFACE_METRICS
    )
    template_columns: list[np.ndarray] = [
        masked_cos[field] for field in (*text_only_fields, "all_text_fields")
    ]
    template_names = [
        f"masked_char3_cosine__{field}"
        for field in (*text_only_fields, "all_text_fields")
    ]
    for field in text_only_fields:
        template_columns.extend(
            surfaces[field][:, index] for index in range(len(SURFACE_METRICS))
        )
        template_names.extend(f"{name}__{field}" for name in SURFACE_METRICS)
    template_similarity = np.column_stack(
        [masked_cos[field] for field in text_only_fields]
    )
    template_columns.extend(
        (
            np.max(template_similarity, axis=1),
            np.mean(template_similarity, axis=1),
            np.mean(np.sort(template_similarity, axis=1)[:, -2:], axis=1),
        )
    )
    template_names.extend(
        (
            "masked_similarity_max__text_fields",
            "masked_similarity_mean__text_fields",
            "masked_similarity_top2_mean__text_fields",
        )
    )
    numeric_by_seller: dict[str, np.ndarray] = {}
    for row in profiles:
        title_stats = row["title_length_stats"]
        description_stats = row["description_length_stats"]
        style_stats = row["style_stats"]
        if (
            not isinstance(title_stats, Mapping)
            or set(title_stats) != {"median"}
            or not isinstance(description_stats, Mapping)
            or set(description_stats) != {"median"}
            or not isinstance(style_stats, Mapping)
            or set(style_stats) != set(dataset_builder.MODEL_PROFILE_STYLE_FIELDS)
        ):
            raise QualityTextProbeViewError("Production numeric nested schema drift")
        vector = np.asarray(
            [
                row["item_count"],
                title_stats["median"],
                description_stats["median"],
                *(style_stats[name] for name in dataset_builder.MODEL_PROFILE_STYLE_FIELDS),
            ],
            dtype=np.float64,
        )
        raw_numeric = (
            row["item_count"],
            title_stats["median"],
            description_stats["median"],
            *(style_stats[name] for name in dataset_builder.MODEL_PROFILE_STYLE_FIELDS),
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_numeric
            )
            or vector.shape != (8,)
            or not np.isfinite(vector).all()
        ):
            raise QualityTextProbeViewError("Production numeric value drift")
        numeric_by_seller[row["seller_uid"]] = vector
    numeric = np.empty((len(endpoints), 16), dtype=np.float64)
    for index, row in enumerate(endpoints):
        left = numeric_by_seller[row["seller_uid_left"]]
        right = numeric_by_seller[row["seller_uid_right"]]
        numeric[index] = np.concatenate((np.abs(left - right), left + right))
    numeric_names = tuple(
        [f"absdiff__model_visible_{name}" for name in PRODUCTION_NUMERIC_FIELDS]
        + [f"sum__model_visible_{name}" for name in PRODUCTION_NUMERIC_FIELDS]
    )
    return (
        {
            "p_full": np.column_stack(full_columns),
            "p_topic": np.column_stack(topic_columns),
            "p_template_surface": np.column_stack(template_columns),
        },
        {
            "p_full": tuple(full_names),
            "p_topic": tuple(topic_names),
            "p_template_surface": tuple(template_names),
        },
        numeric,
        numeric_names,
    )


def build_text_probe_views(
    *,
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Build all seven frozen text views for one materialized surface."""

    pair_keys: set[tuple[str, str]] = set()
    for row in endpoints:
        if not isinstance(row, Mapping) or tuple(row) != ENDPOINT_FIELDS:
            raise QualityTextProbeViewError("Text endpoint schema/order drift")
        key = (
            _required_text(row["world_uid"], name="endpoint world UID"),
            _required_text(row["canonical_pair_uid"], name="endpoint pair UID"),
        )
        left = _required_text(row["seller_uid_left"], name="left seller UID")
        right = _required_text(row["seller_uid_right"], name="right seller UID")
        if (
            key in pair_keys
            or left == right
        ):
            raise QualityTextProbeViewError("Text endpoint key/value drift")
        pair_keys.add(key)
    if not pair_keys:
        raise QualityTextProbeViewError("Text endpoint input is empty")

    fixed, fixed_names = _build_fixed_support_views(
        items=items, endpoints=endpoints
    )
    item_counts_by_seller = Counter(row["seller_uid"] for row in items)
    production, production_names, numeric, numeric_names = _build_production_views(
        profiles=profiles,
        endpoints=endpoints,
        item_counts_by_seller=item_counts_by_seller,
    )
    views = {**fixed, **production}
    names = {**fixed_names, **production_names}
    joint_names = tuple(
        [f"p::{name}" for name in production_names["p_full"]]
        + [f"fs::{name}" for name in fixed_names["fs_full"]]
        + [f"numeric::{name}" for name in numeric_names]
    )
    views["u_joint_full"] = np.column_stack(
        (production["p_full"], fixed["fs_full"], numeric)
    )
    names["u_joint_full"] = joint_names
    if tuple(views) != VIEW_ORDER or tuple(names) != VIEW_ORDER:
        raise QualityTextProbeViewError("Text view order drift")
    if tuple(views[name].shape[1] for name in VIEW_ORDER) != EXPECTED_WIDTHS:
        raise QualityTextProbeViewError("Text view width drift")
    for name in VIEW_ORDER:
        matrix = views[name]
        feature_names = names[name]
        observed_hash = hashlib.sha256(
            _canonical_json_bytes(list(feature_names))
        ).hexdigest()
        if (
            matrix.shape != (len(endpoints), len(feature_names))
            or not np.isfinite(matrix).all()
            or observed_hash != EXPECTED_NAME_HASHES[name]
        ):
            raise QualityTextProbeViewError(f"Text feature closure drift: {name}")
    return views, names
