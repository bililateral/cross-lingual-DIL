#!/usr/bin/env python3
"""Counterfactual visible-text shortcut audit for Step28-v13 v1.12."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

import step28_v13_common as common
import step28_v13_v1_12_assignment_null as assignment_null
import step28_v13_v1_12_counterfactual_text as counterfactual_text
import step28_v13_v1_12_exact_shortcut_preflight as exact_preflight
import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony
import step28_v13_world_builder as world_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_12_text_shortcut_audit_policy.json"
)


class TextShortcutAuditError(ValueError):
    """Raised when the frozen visible-text audit cannot close exactly."""


@dataclass(frozen=True)
class WorldVisibleAttackMatrices:
    world_uid: str
    pair_uids: tuple[str, ...]
    labels: np.ndarray
    views: dict[str, np.ndarray]
    excluded_pair_uids: tuple[str, ...]
    feature_names_by_view: dict[str, tuple[str, ...]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class SplitVisibleAttackMatrices:
    split: str
    world_uids: tuple[str, ...]
    pair_uids: tuple[str, ...]
    labels: np.ndarray
    views: dict[str, np.ndarray]
    feature_names_by_view: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class DesignWorldAttackResult:
    visible: WorldVisibleAttackMatrices
    assignment: dict[str, Any]
    rerender_audit: dict[str, Any]


@dataclass(frozen=True)
class DesignSplitAttackData:
    visible: SplitVisibleAttackMatrices
    assignment_relation_names: tuple[str, ...]
    assignment_observed: np.ndarray
    assignment_expected: np.ndarray
    assignment_world_uids: tuple[str, ...]
    world_audits: tuple[dict[str, Any], ...]


def load_text_audit_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    preceremony.validate_canonical_self_hash(policy, label="text shortcut policy")
    contract = policy["contract"]
    contract_path = ROOT / str(contract["path"])
    payload = contract_path.read_bytes()
    runtime = policy["runtime_requirements"]
    if (
        len(payload) != int(contract["size_bytes"])
        or hashlib.sha256(payload).hexdigest() != str(contract["sha256"])
        or set(policy["authorizations"].values()) != {False}
        or platform.python_version() != str(runtime["python"])
        or np.__version__ != str(runtime["numpy"])
        or scipy.__version__ != str(runtime["scipy"])
        or sklearn.__version__ != str(runtime["scikit_learn"])
        or unicodedata.unidata_version
        != str(runtime["unicodedata_unidata_version"])
    ):
        raise TextShortcutAuditError("Text shortcut policy/contract closure failed")
    return policy


def word12_tokens(value: str) -> list[str]:
    """Apply the frozen ASCII-run plus individual-Han tokenizer."""

    if not isinstance(value, str):
        raise TextShortcutAuditError("word12 tokenizer input must be a string")
    output: list[str] = []
    ascii_run: list[str] = []

    def flush_ascii() -> None:
        if ascii_run:
            output.append("".join(ascii_run).lower())
            ascii_run.clear()

    for character in value:
        ordinal = ord(character)
        if (
            ord("A") <= ordinal <= ord("Z")
            or ord("a") <= ordinal <= ord("z")
            or ord("0") <= ordinal <= ord("9")
        ):
            ascii_run.append(character)
            continue
        flush_ascii()
        if (
            0x3400 <= ordinal <= 0x4DBF
            or 0x4E00 <= ordinal <= 0x9FFF
            or 0xF900 <= ordinal <= 0xFAFF
        ):
            output.append(character)
    flush_ascii()
    return output


def template_mask(value: str) -> str:
    """Mask Unicode letters/digits while preserving all other codepoints."""

    if not isinstance(value, str):
        raise TextShortcutAuditError("Template-mask input must be a string")
    output: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category in {"Lu", "Ll", "Lt", "Lm", "Lo"}:
            output.append("字")
        elif category == "Nd":
            output.append("数")
        else:
            output.append(character)
    return "".join(output)


def _vectorizer(
    policy: Mapping[str, Any], *, kind: str
) -> HashingVectorizer:
    try:
        frozen = dict(policy["visible_attack"]["vectorizers"][kind])
        kwargs = dict(frozen["constructor_kwargs"])
    except KeyError as exc:
        raise TextShortcutAuditError(f"Unknown frozen vectorizer: {kind}") from exc
    kwargs["ngram_range"] = tuple(int(value) for value in kwargs["ngram_range"])
    kwargs["dtype"] = np.float64
    if kind == "word12":
        if kwargs["tokenizer"] != "frozen_tokenizer_symbol":
            raise TextShortcutAuditError("word12 tokenizer symbol drift")
        kwargs["tokenizer"] = word12_tokens
    return HashingVectorizer(**kwargs)


def _rowwise_sorted_sparse_dot(left: Any, right: Any) -> np.ndarray:
    if left.shape != right.shape or left.shape[0] == 0:
        raise TextShortcutAuditError("Sparse cosine input shape drift")
    product = left.multiply(right).tocsr()
    product.sort_indices()
    output = np.empty(product.shape[0], dtype=np.float64)
    for row_index in range(product.shape[0]):
        start = int(product.indptr[row_index])
        end = int(product.indptr[row_index + 1])
        total = 0.0
        for value in product.data[start:end]:
            total += float(value)
        output[row_index] = total
    if not np.all(np.isfinite(output)):
        raise TextShortcutAuditError("Sparse cosine produced a nonfinite value")
    return output


def _cosine_by_pair(
    documents: Sequence[str],
    *,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    vectorizer: HashingVectorizer,
) -> np.ndarray:
    matrix = vectorizer.transform(documents).tocsr()
    matrix.sort_indices()
    if matrix.shape[0] != len(documents):
        raise TextShortcutAuditError("Seller text vector row-count drift")
    return _rowwise_sorted_sparse_dot(
        matrix[left_indices], matrix[right_indices]
    )


def _seller_surface_arrays(documents: Sequence[str]) -> dict[str, np.ndarray]:
    punctuation = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}
    ascii_whitespace = {"\t", "\n", "\v", "\f", "\r", " "}
    values: dict[str, list[int]] = {
        "codepoint_length": [],
        "newline_count": [],
        "unicode_punctuation_count": [],
        "ascii_whitespace_count": [],
        "unicode_decimal_digit_count": [],
        "empty": [],
    }
    for document in documents:
        if not isinstance(document, str):
            raise TextShortcutAuditError("Surface document must be a string")
        values["codepoint_length"].append(len(document))
        values["newline_count"].append(document.count("\n"))
        values["unicode_punctuation_count"].append(
            sum(unicodedata.category(character) in punctuation for character in document)
        )
        values["ascii_whitespace_count"].append(
            sum(character in ascii_whitespace for character in document)
        )
        values["unicode_decimal_digit_count"].append(
            sum(unicodedata.category(character) == "Nd" for character in document)
        )
        values["empty"].append(int(len(document) == 0))
    return {
        name: np.asarray(raw, dtype=np.int64) for name, raw in values.items()
    }


def _surface_pair_features(
    documents: Sequence[str],
    *,
    field_name: str,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = _seller_surface_arrays(documents)
    output: dict[str, np.ndarray] = {}
    for base_name in (
        "codepoint_length",
        "newline_count",
        "unicode_punctuation_count",
        "ascii_whitespace_count",
        "unicode_decimal_digit_count",
    ):
        left = arrays[base_name][left_indices]
        right = arrays[base_name][right_indices]
        output[f"{base_name}_absdiff__{field_name}"] = np.abs(
            left - right
        ).astype(np.float64)
        output[f"{base_name}_sum__{field_name}"] = (left + right).astype(
            np.float64
        )
    left_empty = arrays["empty"][left_indices].astype(bool)
    right_empty = arrays["empty"][right_indices].astype(bool)
    output[f"empty_both__{field_name}"] = np.logical_and(
        left_empty, right_empty
    ).astype(np.float64)
    output[f"empty_xor__{field_name}"] = np.logical_xor(
        left_empty, right_empty
    ).astype(np.float64)
    return output


def _validated_neutral_pairs(
    *,
    policy: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    negative_flags: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, tuple[str, ...], dict[str, Any]]:
    contract = policy["mechanism_neutral_eligibility"]
    excluded_flags = tuple(str(value) for value in contract["excluded_flag_order"])
    expected_flag_counts = {
        name: int(contract["per_world_flag_counts"][name])
        for name in excluded_flags
    }
    selected = [
        dict(row) for row in negative_flags if str(row.get("flag", "")) in excluded_flags
    ]
    if Counter(str(row["flag"]) for row in selected) != Counter(
        expected_flag_counts
    ):
        raise TextShortcutAuditError("Mechanism-neutral flag counts drift")
    selected_triples = {
        (
            str(contract["override_kind_by_flag"][str(row["flag"])]),
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]),
        )
        for row in selected
    }
    override_triples = {
        (
            str(row["override_kind"]),
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]),
        )
        for row in override_audit
    }
    if (
        len(selected_triples) != 6
        or len(override_triples) != 6
        or selected_triples != override_triples
    ):
        raise TextShortcutAuditError("Mechanism flag/override replay mismatch")
    excluded = tuple(
        sorted(
            {pair_uid for _kind, _asset, pair_uid in selected_triples},
            key=lambda value: value.encode("utf-8"),
        )
    )
    pair_index = {str(row["canonical_pair_uid"]): dict(row) for row in pair_rows}
    label_index = {str(row["canonical_pair_uid"]): int(row["label"]) for row in label_rows}
    if (
        len(pair_index) != 378
        or len(label_index) != 378
        or set(pair_index) != set(label_index)
        or any(label_index[pair_uid] != 0 for pair_uid in excluded)
    ):
        raise TextShortcutAuditError("Mechanism-neutral label universe drift")
    eligible_uids = sorted(
        set(pair_index) - set(excluded), key=lambda value: value.encode("utf-8")
    )
    eligible = [pair_index[pair_uid] for pair_uid in eligible_uids]
    labels = np.asarray([label_index[pair_uid] for pair_uid in eligible_uids], dtype=np.int8)
    if (
        len(eligible) != int(contract["per_world_eligible_rows"])
        or int(np.sum(labels)) != int(contract["per_world_eligible_positive_rows"])
        or len(labels) - int(np.sum(labels))
        != int(contract["per_world_eligible_negative_rows"])
    ):
        raise TextShortcutAuditError("Mechanism-neutral 20/352 closure failed")
    excluded_sellers = {
        str(pair_index[pair_uid][name])
        for pair_uid in excluded
        for name in ("seller_uid_left", "seller_uid_right")
    }
    eligible_sellers = {
        str(row[name])
        for row in eligible
        for name in ("seller_uid_left", "seller_uid_right")
    }
    if len(excluded_sellers) != 12 or not excluded_sellers.issubset(eligible_sellers):
        raise TextShortcutAuditError("Mechanism mask removed an endpoint seller")
    return eligible, labels, excluded, {
        "excluded_pair_count": len(excluded),
        "eligible_pair_count": len(eligible),
        "eligible_positive_count": int(np.sum(labels)),
        "eligible_negative_count": len(labels) - int(np.sum(labels)),
        "excluded_endpoint_seller_count": len(excluded_sellers),
        "excluded_endpoint_sellers_retained_elsewhere": True,
    }


def _summary_features(
    values: Mapping[str, np.ndarray],
    *,
    source_names: Sequence[str],
    prefix: str,
) -> dict[str, np.ndarray]:
    try:
        matrix = np.vstack([values[str(name)] for name in source_names])
    except KeyError as exc:
        raise TextShortcutAuditError("Similarity summary source is missing") from exc
    if matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise TextShortcutAuditError("Similarity summary matrix drift")
    ordered = np.sort(matrix, axis=0)
    return {
        f"{prefix}_max": ordered[-1],
        f"{prefix}_mean": np.mean(matrix, axis=0, dtype=np.float64),
        f"{prefix}_top2_mean": np.mean(ordered[-2:], axis=0, dtype=np.float64),
    }


def build_world_visible_attack_matrices(
    *,
    policy: Mapping[str, Any],
    seller_profiles: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    negative_flags: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> WorldVisibleAttackMatrices:
    """Build the frozen three-view matrix for one counterfactual world."""

    attack = policy["visible_attack"]
    fields = tuple(str(value) for value in attack["m0_fields_in_order"])
    if len(seller_profiles) != 28 or len(fields) != 5:
        raise TextShortcutAuditError("Visible attack seller/field boundary drift")
    profiles = {
        str(row["seller_uid"]): row for row in seller_profiles
    }
    if len(profiles) != 28:
        raise TextShortcutAuditError("Visible attack seller UID collision")
    world_uids = {
        str(row["world_uid"]) for row in pair_rows
    }
    if len(world_uids) != 1:
        raise TextShortcutAuditError("Visible attack requires one pair world")
    world_uid = next(iter(world_uids))
    eligible, labels, excluded, mask_audit = _validated_neutral_pairs(
        policy=policy,
        pair_rows=pair_rows,
        label_rows=label_rows,
        negative_flags=negative_flags,
        override_audit=override_audit,
    )
    seller_order = sorted(profiles, key=lambda value: value.encode("utf-8"))
    seller_ordinal = {seller_uid: index for index, seller_uid in enumerate(seller_order)}
    try:
        left_indices = np.asarray(
            [seller_ordinal[str(row["seller_uid_left"])] for row in eligible],
            dtype=np.intp,
        )
        right_indices = np.asarray(
            [seller_ordinal[str(row["seller_uid_right"])] for row in eligible],
            dtype=np.intp,
        )
    except KeyError as exc:
        raise TextShortcutAuditError("Pair endpoint is outside seller profiles") from exc
    documents = {
        field: [str(profiles[seller_uid][field]) for seller_uid in seller_order]
        for field in fields
    }
    uid_residue = re.compile(r"(?:w|sel|itm)_[0-9a-f]{64}")
    if any(
        uid_residue.search(document) is not None
        for field in fields
        for document in documents[field]
    ):
        raise TextShortcutAuditError("Join-only UID residue entered an M0 text field")
    separator = bytes.fromhex(
        str(attack["combined_field_separator_utf8_hex"])
    ).decode("utf-8")
    combined = attack["combined_documents"]
    documents["all_fields"] = [
        separator.join(str(profiles[seller_uid][field]) for field in fields)
        for seller_uid in seller_order
    ]
    text_fields = tuple(
        str(value) for value in combined["all_text_fields_source_fields_in_order"]
    )
    documents["all_text_fields"] = [
        separator.join(str(profiles[seller_uid][field]) for field in text_fields)
        for seller_uid in seller_order
    ]

    char3 = _vectorizer(policy, kind="char3")
    word12 = _vectorizer(policy, kind="word12")
    values: dict[str, np.ndarray] = {}
    for field in fields:
        values[f"char3_cosine__{field}"] = _cosine_by_pair(
            documents[field],
            left_indices=left_indices,
            right_indices=right_indices,
            vectorizer=char3,
        )
        values[f"word12_cosine__{field}"] = _cosine_by_pair(
            documents[field],
            left_indices=left_indices,
            right_indices=right_indices,
            vectorizer=word12,
        )
        values.update(
            _surface_pair_features(
                documents[field],
                field_name=field,
                left_indices=left_indices,
                right_indices=right_indices,
            )
        )
    for kind, vectorizer in (("char3", char3), ("word12", word12)):
        values[f"{kind}_cosine__all_fields"] = _cosine_by_pair(
            documents["all_fields"],
            left_indices=left_indices,
            right_indices=right_indices,
            vectorizer=vectorizer,
        )
    for field in text_fields:
        masked = [template_mask(value) for value in documents[field]]
        values[f"masked_char3_cosine__{field}"] = _cosine_by_pair(
            masked,
            left_indices=left_indices,
            right_indices=right_indices,
            vectorizer=char3,
        )
    values["masked_char3_cosine__all_text_fields"] = _cosine_by_pair(
        [template_mask(value) for value in documents["all_text_fields"]],
        left_indices=left_indices,
        right_indices=right_indices,
        vectorizer=char3,
    )
    full_summary = _summary_features(
        values,
        source_names=attack["similarity_summaries"]["cf_full_sources_in_order"],
        prefix="similarity",
    )
    values["similarity_max__field_char_word"] = full_summary["similarity_max"]
    values["similarity_mean__field_char_word"] = full_summary["similarity_mean"]
    values["similarity_top2_mean__field_char_word"] = full_summary[
        "similarity_top2_mean"
    ]
    masked_summary = _summary_features(
        values,
        source_names=attack["similarity_summaries"]["cf_template_sources_in_order"],
        prefix="masked_similarity",
    )
    values["masked_similarity_max__text_fields"] = masked_summary[
        "masked_similarity_max"
    ]
    values["masked_similarity_mean__text_fields"] = masked_summary[
        "masked_similarity_mean"
    ]
    values["masked_similarity_top2_mean__text_fields"] = masked_summary[
        "masked_similarity_top2_mean"
    ]

    views: dict[str, np.ndarray] = {}
    names_by_view: dict[str, tuple[str, ...]] = {}
    for view_name, view in attack["views"].items():
        names = tuple(str(value) for value in view["feature_names_in_order"])
        try:
            matrix = np.column_stack([values[name] for name in names]).astype(
                np.float64, copy=False
            )
        except KeyError as exc:
            raise TextShortcutAuditError(
                f"Frozen view references unknown feature: {view_name}"
            ) from exc
        if matrix.shape != (len(eligible), len(names)) or not np.all(
            np.isfinite(matrix)
        ):
            raise TextShortcutAuditError(f"Visible view matrix drift: {view_name}")
        views[str(view_name)] = matrix
        names_by_view[str(view_name)] = names
    expected_widths = {"cf_full": 75, "cf_topic": 14, "cf_template_surface": 56}
    if {name: matrix.shape[1] for name, matrix in views.items()} != expected_widths:
        raise TextShortcutAuditError("Visible view width contract drift")
    return WorldVisibleAttackMatrices(
        world_uid=world_uid,
        pair_uids=tuple(str(row["canonical_pair_uid"]) for row in eligible),
        labels=labels,
        views=views,
        excluded_pair_uids=excluded,
        feature_names_by_view=names_by_view,
        audit=mask_audit,
    )


def aggregate_world_visible_attack_matrices(
    worlds: Sequence[WorldVisibleAttackMatrices],
    *,
    split: str,
    expected_world_count: int,
) -> SplitVisibleAttackMatrices:
    if (
        split not in {"train", "development"}
        or len(worlds) != expected_world_count
        or expected_world_count <= 0
    ):
        raise TextShortcutAuditError("Visible split aggregation boundary drift")
    ordered = sorted(worlds, key=lambda value: value.world_uid.encode("utf-8"))
    world_uid_set = {value.world_uid for value in ordered}
    if len(world_uid_set) != expected_world_count:
        raise TextShortcutAuditError("Visible split world UID collision")
    names = ordered[0].feature_names_by_view
    if any(value.feature_names_by_view != names for value in ordered):
        raise TextShortcutAuditError("Visible split feature-name drift")
    world_uids = tuple(
        value.world_uid for value in ordered for _pair_uid in value.pair_uids
    )
    pair_uids = tuple(
        pair_uid for value in ordered for pair_uid in value.pair_uids
    )
    labels = np.concatenate([value.labels for value in ordered]).astype(
        np.int8, copy=False
    )
    views = {
        view_name: np.vstack([value.views[view_name] for value in ordered]).astype(
            np.float64, copy=False
        )
        for view_name in names
    }
    expected_rows = expected_world_count * 372
    if (
        len(world_uids) != expected_rows
        or len(pair_uids) != expected_rows
        or len(set(pair_uids)) != expected_rows
        or labels.shape != (expected_rows,)
        or int(np.sum(labels)) != expected_world_count * 20
        or any(matrix.shape[0] != expected_rows for matrix in views.values())
        or any(not np.all(np.isfinite(matrix)) for matrix in views.values())
    ):
        raise TextShortcutAuditError("Visible split matrix closure failed")
    return SplitVisibleAttackMatrices(
        split=split,
        world_uids=world_uids,
        pair_uids=pair_uids,
        labels=labels,
        views=views,
        feature_names_by_view=dict(names),
    )


def fold_by_world(
    world_uids: Sequence[str], *, seed: int, fold_count: int
) -> dict[str, int]:
    worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    if (
        fold_count <= 1
        or len(worlds) < fold_count
        or len(worlds) % fold_count != 0
    ):
        raise TextShortcutAuditError("Visible world-fold dimensions are invalid")
    ranked = sorted(
        worlds,
        key=lambda world_uid: (
            hashlib.sha256(
                str(seed).encode("ascii") + b"\x1f" + world_uid.encode("utf-8")
            ).digest(),
            world_uid.encode("utf-8"),
        ),
    )
    output = {world_uid: index % fold_count for index, world_uid in enumerate(ranked)}
    if Counter(output.values()) != Counter(
        {fold: len(worlds) // fold_count for fold in range(fold_count)}
    ):
        raise TextShortcutAuditError("Visible world-fold balance drift")
    return output


def _gradient_tree(
    policy: Mapping[str, Any], *, random_state: int | None = None
) -> HistGradientBoostingClassifier:
    config = dict(policy["visible_attack"]["gradient_tree"])
    implementation = config.pop("implementation")
    implicit = config.pop("implicit_defaults_allowed")
    if implementation != "sklearn.ensemble.HistGradientBoostingClassifier" or implicit:
        raise TextShortcutAuditError("Frozen gradient-tree implementation drift")
    if random_state is not None:
        config["random_state"] = int(random_state)
    return HistGradientBoostingClassifier(**config)


def _fit_one_model_family(
    *,
    policy: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_worlds: Sequence[str],
    x_development: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    attack = policy["visible_attack"]
    fold_count = int(attack["fold_count"])
    fold_seed = int(attack["fold_seed"])
    fold_map = fold_by_world(
        train_worlds, seed=fold_seed, fold_count=fold_count
    )
    fold_ids = np.asarray([fold_map[value] for value in train_worlds], dtype=np.int8)
    oof = {
        name: np.full(len(y_train), np.nan, dtype=np.float64)
        for name in ("logistic_l2", "gradient_tree")
    }
    fold_rows: list[dict[str, Any]] = []
    logistic_audits: list[dict[str, Any]] = []
    logistic_config = attack["logistic_l2"]
    if (
        logistic_config["implementation"]
        != "step28_v13_v1_12_exact_shortcut_preflight.fit_exact_logistic"
        or logistic_config["implicit_defaults_allowed"] is not False
    ):
        raise TextShortcutAuditError("Frozen logistic implementation drift")
    for fold in range(fold_count):
        held_out = fold_ids == fold
        fitted = ~held_out
        fit_worlds = {
            train_worlds[index] for index in np.flatnonzero(fitted)
        }
        held_worlds = {
            train_worlds[index] for index in np.flatnonzero(held_out)
        }
        if (
            fit_worlds & held_worlds
            or len(held_worlds) != len(set(train_worlds)) // fold_count
            or set(np.unique(y_train[fitted]).tolist()) != {0, 1}
            or set(np.unique(y_train[held_out]).tolist()) != {0, 1}
        ):
            raise TextShortcutAuditError("Visible OOF world isolation drift")
        logistic = exact_preflight.fit_exact_logistic(
            x_train[fitted],
            y_train[fitted],
            l2=float(logistic_config["l2"]),
            maximum_iterations=int(logistic_config["maximum_iterations"]),
            gradient_tolerance=float(
                logistic_config["normalized_gradient_tolerance"]
            ),
        )
        oof["logistic_l2"][held_out] = exact_preflight.score_exact_logistic(
            logistic, x_train[held_out]
        )
        tree = _gradient_tree(policy, random_state=int(attack["gradient_tree"]["random_state"]))
        tree.fit(x_train[fitted], y_train[fitted])
        oof["gradient_tree"][held_out] = tree.predict_proba(
            x_train[held_out]
        )[:, 1].astype(np.float64)
        logistic_audits.append(dict(logistic.audit))
        fold_rows.append(
            {
                "fold": fold,
                "fit_world_count": len(fit_worlds),
                "held_out_world_count": len(held_worlds),
                "fit_row_count": int(np.sum(fitted)),
                "held_out_row_count": int(np.sum(held_out)),
            }
        )
    if any(not np.all(np.isfinite(value)) for value in oof.values()):
        raise TextShortcutAuditError("Visible OOF score coverage failed")
    full_logistic = exact_preflight.fit_exact_logistic(
        x_train,
        y_train,
        l2=float(logistic_config["l2"]),
        maximum_iterations=int(logistic_config["maximum_iterations"]),
        gradient_tolerance=float(logistic_config["normalized_gradient_tolerance"]),
    )
    full_tree = _gradient_tree(policy)
    full_tree.fit(x_train, y_train)
    development = {
        "logistic_l2": exact_preflight.score_exact_logistic(
            full_logistic, x_development
        ),
        "gradient_tree": full_tree.predict_proba(x_development)[:, 1].astype(
            np.float64
        ),
    }
    if any(not np.all(np.isfinite(value)) for value in development.values()):
        raise TextShortcutAuditError("Visible development score validity failed")
    return oof, development, {
        "folds": fold_rows,
        "fold_logistic_optimizer_audits": logistic_audits,
        "full_logistic_optimizer_audit": dict(full_logistic.audit),
    }


def _single_feature_metrics(
    policy: Mapping[str, Any],
    matrix: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    if matrix.shape != (len(labels), len(names)):
        raise TextShortcutAuditError("Single-feature metric dimensions drift")
    rows: dict[str, Any] = {}
    maximum = 0.5
    for index, name in enumerate(names):
        values = matrix[:, index]
        auc = _point_roc_auc(policy, labels, values)
        symmetric = max(auc, 1.0 - auc)
        rows[str(name)] = {
            "roc_auc": auc,
            "symmetric_roc_auc": symmetric,
            "average_precision_forward": _point_average_precision(
                policy, labels, values
            ),
            "average_precision_reverse": _point_average_precision(
                policy, labels, -values
            ),
        }
        maximum = max(maximum, symmetric)
    return {"features": rows, "maximum_symmetric_roc_auc": maximum}


def _point_metric_kwargs(
    policy: Mapping[str, Any], *, metric: str
) -> dict[str, Any]:
    try:
        spec = policy["bootstrap"]["point_metrics"][metric]
        implementation = str(spec["implementation"])
        kwargs = dict(spec["kwargs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TextShortcutAuditError("Point-metric policy is malformed") from exc
    expected = {
        "roc_auc": (
            "sklearn.metrics.roc_auc_score",
            {
                "average": "macro",
                "sample_weight": None,
                "max_fpr": None,
                "multi_class": "raise",
                "labels": None,
            },
        ),
        "average_precision": (
            "sklearn.metrics.average_precision_score",
            {"average": "macro", "pos_label": 1, "sample_weight": None},
        ),
    }
    if metric not in expected or (implementation, kwargs) != expected[metric]:
        raise TextShortcutAuditError("Point-metric policy drift")
    return kwargs


def _point_roc_auc(
    policy: Mapping[str, Any], labels: np.ndarray, scores: np.ndarray
) -> float:
    return float(
        roc_auc_score(
            labels,
            scores,
            **_point_metric_kwargs(policy, metric="roc_auc"),
        )
    )


def _point_average_precision(
    policy: Mapping[str, Any], labels: np.ndarray, scores: np.ndarray
) -> float:
    return float(
        average_precision_score(
            labels,
            scores,
            **_point_metric_kwargs(policy, metric="average_precision"),
        )
    )


def _score_metrics(
    policy: Mapping[str, Any], labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    auc = _point_roc_auc(policy, labels, scores)
    return {
        "roc_auc": auc,
        "symmetric_roc_auc": max(auc, 1.0 - auc),
        "average_precision": _point_average_precision(policy, labels, scores),
    }


def _bootstrap_upper(policy: Mapping[str, Any], values: np.ndarray) -> float:
    config = policy["bootstrap"]
    quantile = float(config["quantile"])
    method = str(config["quantile_method"])
    if quantile != 0.95 or method != "higher":
        raise TextShortcutAuditError("Bootstrap quantile policy drift")
    return float(np.quantile(values, quantile, method=method))


def draw_world_multiplicities(
    *,
    policy: Mapping[str, Any],
    world_uids: Sequence[str],
    split: str,
    seed_field: str = "text_attack_seed",
) -> tuple[np.ndarray, np.ndarray, str]:
    config = policy["bootstrap"]
    ordered_worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    if split != "development" or len(ordered_worlds) != 500:
        raise TextShortcutAuditError("Visible bootstrap world universe drift")
    ordinal = {world_uid: index for index, world_uid in enumerate(ordered_worlds)}
    row_world = np.asarray([ordinal[value] for value in world_uids], dtype=np.int16)
    if seed_field not in {"text_attack_seed", "assignment_seed"}:
        raise TextShortcutAuditError("Visible bootstrap seed field drift")
    seed = int(config[seed_field])
    split_seed = int.from_bytes(
        hashlib.sha256(
            str(seed).encode("ascii") + b"\x1f" + split.encode("ascii")
        ).digest()[:16],
        "big",
        signed=False,
    )
    generator = np.random.Generator(np.random.PCG64DXSM(split_seed))
    draws = generator.integers(
        low=0,
        high=500,
        size=(int(config["replicates"]), 500),
        dtype=np.int16,
    )
    multiplicities = np.zeros_like(draws, dtype=np.int16)
    for replicate in range(len(draws)):
        multiplicities[replicate] = np.bincount(
            draws[replicate], minlength=500
        ).astype(np.int16)
    draw_hash = hashlib.sha256(
        np.ascontiguousarray(draws.astype(">i2", copy=False)).tobytes()
    ).hexdigest()
    return multiplicities, row_world, draw_hash


def bootstrap_rank_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    multiplicities: np.ndarray,
    row_world: np.ndarray,
    replicate_chunk: int = 256,
    row_chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact tie-aware weighted ROC-AUC/AP under world bootstrap."""

    if (
        labels.shape != scores.shape
        or row_world.shape != labels.shape
        or multiplicities.ndim != 2
        or multiplicities.shape[1] != 500
        or len(labels) != 500 * 372
        or not np.all(np.isfinite(scores))
        or np.any(multiplicities < 0)
        or not np.all(np.sum(multiplicities, axis=1) == 500)
    ):
        raise TextShortcutAuditError("Visible bootstrap metric input drift")
    positive_by_world = np.bincount(
        row_world, weights=labels.astype(np.float64), minlength=500
    )
    negative_by_world = np.bincount(
        row_world, weights=1.0 - labels.astype(np.float64), minlength=500
    )
    if not (
        np.all(positive_by_world == 20.0)
        and np.all(negative_by_world == 352.0)
    ):
        raise TextShortcutAuditError("Visible bootstrap per-world classes drift")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1], True]
    ).astype(np.int64, copy=False)
    replicate_count = multiplicities.shape[0]
    auc_output = np.empty(replicate_count, dtype=np.float64)
    ap_output = np.empty(replicate_count, dtype=np.float64)
    for replicate_start in range(0, replicate_count, replicate_chunk):
        replicate_end = min(replicate_count, replicate_start + replicate_chunk)
        counts = multiplicities[replicate_start:replicate_end].astype(
            np.float64, copy=False
        )
        total_positive = counts @ positive_by_world
        total_negative = counts @ negative_by_world
        prevalence = total_positive / (total_positive + total_negative)
        if not np.all(prevalence == (5.0 / 93.0)):
            raise TextShortcutAuditError("Visible bootstrap prevalence drift")
        batch = len(counts)
        auc_numerator = np.zeros(batch, dtype=np.float64)
        ap_numerator = np.zeros(batch, dtype=np.float64)
        cumulative_positive = np.zeros(batch, dtype=np.float64)
        cumulative_negative = np.zeros(batch, dtype=np.float64)
        cumulative_total = np.zeros(batch, dtype=np.float64)
        group_index = 0
        group_count = len(boundaries) - 1
        while group_index < group_count:
            row_start = int(boundaries[group_index])
            target = row_start + row_chunk
            group_stop = int(np.searchsorted(boundaries, target, side="right") - 1)
            group_stop = min(group_count, max(group_index + 1, group_stop))
            row_stop = int(boundaries[group_stop])
            indices = order[row_start:row_stop]
            weights = counts[:, row_world[indices]]
            local_starts = (boundaries[group_index:group_stop] - row_start).astype(
                np.intp, copy=False
            )
            local_labels = labels[indices].astype(np.float64, copy=False)
            positive = np.add.reduceat(
                weights * local_labels[None, :], local_starts, axis=1
            )
            negative = np.add.reduceat(
                weights * (1.0 - local_labels)[None, :], local_starts, axis=1
            )
            total = positive + negative
            negative_before = cumulative_negative[:, None] + np.cumsum(
                negative, axis=1
            ) - negative
            auc_numerator += np.sum(
                positive
                * (total_negative[:, None] - negative_before - 0.5 * negative),
                axis=1,
            )
            positive_at_end = cumulative_positive[:, None] + np.cumsum(
                positive, axis=1
            )
            total_at_end = cumulative_total[:, None] + np.cumsum(total, axis=1)
            precision_at_end = np.divide(
                positive_at_end,
                total_at_end,
                out=np.zeros_like(positive_at_end),
                where=total_at_end > 0.0,
            )
            ap_numerator += np.sum(positive * precision_at_end, axis=1)
            cumulative_positive = positive_at_end[:, -1]
            cumulative_negative = negative_before[:, -1] + negative[:, -1]
            cumulative_total = total_at_end[:, -1]
            group_index = group_stop
        if not (
            np.allclose(cumulative_positive, total_positive, rtol=0.0, atol=1e-9)
            and np.allclose(
                cumulative_negative, total_negative, rtol=0.0, atol=1e-9
            )
        ):
            raise TextShortcutAuditError("Visible bootstrap rank traversal did not close")
        raw_auc = auc_numerator / (total_positive * total_negative)
        auc_output[replicate_start:replicate_end] = np.maximum(
            raw_auc, 1.0 - raw_auc
        )
        ap_output[replicate_start:replicate_end] = ap_numerator / total_positive
    if not np.all(np.isfinite(auc_output)) or not np.all(np.isfinite(ap_output)):
        raise TextShortcutAuditError("Visible bootstrap metrics are nonfinite")
    return auc_output, ap_output


def evaluate_visible_attack_family(
    *,
    policy: Mapping[str, Any],
    train: SplitVisibleAttackMatrices,
    development: SplitVisibleAttackMatrices,
) -> dict[str, Any]:
    if (
        train.split != "train"
        or development.split != "development"
        or set(train.world_uids) & set(development.world_uids)
        or train.feature_names_by_view != development.feature_names_by_view
        or len(set(train.world_uids)) != 500
        or len(set(development.world_uids)) != 500
    ):
        raise TextShortcutAuditError("Visible train/development isolation drift")
    oof_scores: dict[str, np.ndarray] = {}
    development_scores: dict[str, np.ndarray] = {}
    fit_audits: dict[str, Any] = {}
    for view_name in policy["visible_attack"]["views"]:
        oof, dev, audit = _fit_one_model_family(
            policy=policy,
            x_train=train.views[view_name],
            y_train=train.labels,
            train_worlds=train.world_uids,
            x_development=development.views[view_name],
        )
        for model_name in ("logistic_l2", "gradient_tree"):
            family_name = f"{view_name}::{model_name}"
            oof_scores[family_name] = oof[model_name]
            development_scores[family_name] = dev[model_name]
        fit_audits[view_name] = audit
    single = {
        view_name: _single_feature_metrics(
            policy,
            development.views[view_name],
            development.labels,
            development.feature_names_by_view[view_name],
        )
        for view_name in policy["visible_attack"]["views"]
    }
    oof_metrics = {
        name: _score_metrics(policy, train.labels, scores)
        for name, scores in oof_scores.items()
    }
    development_metrics = {
        name: _score_metrics(policy, development.labels, scores)
        for name, scores in development_scores.items()
    }
    multiplicities, row_world, draw_hash = draw_world_multiplicities(
        policy=policy,
        world_uids=development.world_uids,
        split="development",
    )
    auc_bootstrap: list[np.ndarray] = []
    ap_uplift_bootstrap: list[np.ndarray] = []
    baseline = 5.0 / 93.0
    for name in sorted(development_scores):
        auc_values, ap_values = bootstrap_rank_metrics(
            labels=development.labels,
            scores=development_scores[name],
            multiplicities=multiplicities,
            row_world=row_world,
        )
        auc_bootstrap.append(auc_values)
        ap_uplift_bootstrap.append(ap_values - baseline)
    family_max_auc = np.maximum.reduce(auc_bootstrap)
    family_max_ap_uplift = np.maximum.reduce(ap_uplift_bootstrap)
    upper_auc = _bootstrap_upper(policy, family_max_auc)
    upper_ap = _bootstrap_upper(policy, family_max_ap_uplift)
    point_single = max(
        float(block["maximum_symmetric_roc_auc"]) for block in single.values()
    )
    point_family_auc = max(
        float(value["symmetric_roc_auc"]) for value in development_metrics.values()
    )
    point_family_ap_uplift = max(
        float(value["average_precision"]) - baseline
        for value in development_metrics.values()
    )
    gates = policy["hard_gates"]
    gate_results = {
        "development_single_feature_symmetric_auc": point_single
        <= float(gates["development_maximum_single_feature_symmetric_auc"]),
        "development_family_model_symmetric_auc": point_family_auc
        <= float(gates["development_maximum_family_model_symmetric_auc"]),
        "development_family_model_average_precision_uplift": point_family_ap_uplift
        <= float(
            gates["development_maximum_family_model_average_precision_uplift"]
        ),
        "development_bootstrap_family_max_symmetric_auc": upper_auc
        <= float(
            gates["development_bootstrap_95_upper_family_max_symmetric_auc"]
        ),
        "development_bootstrap_family_max_average_precision_uplift": upper_ap
        <= float(
            gates[
                "development_bootstrap_95_upper_family_max_average_precision_uplift"
            ]
        ),
    }
    return {
        "status": (
            "PASS_VISIBLE_TEXT_SHORTCUT_GATES"
            if all(gate_results.values())
            else "FAIL_VISIBLE_TEXT_SHORTCUT_GATES"
        ),
        "development_single_feature_metrics": single,
        "train_world_oof_diagnostic_metrics": oof_metrics,
        "development_family_metrics": development_metrics,
        "fit_audits": fit_audits,
        "random_average_precision_baseline": baseline,
        "bootstrap_replicates": int(policy["bootstrap"]["replicates"]),
        "bootstrap_draw_sha256": draw_hash,
        "bootstrap_95_upper_family_max_symmetric_auc": upper_auc,
        "bootstrap_95_upper_family_max_average_precision_uplift": upper_ap,
        "point_maximum_single_feature_symmetric_auc": point_single,
        "point_maximum_family_model_symmetric_auc": point_family_auc,
        "point_maximum_family_model_average_precision_uplift": point_family_ap_uplift,
        "hard_gates": gate_results,
        "development_only_used_for_hard_gates": True,
        "models_refit_inside_bootstrap": False,
    }


def _design_split_context(
    *, split: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if split not in {"train", "development"}:
        raise TextShortcutAuditError("Text design preflight split drift")
    validated = formal.load_and_validate_draft()
    draft = validated["draft"]
    if set(draft["authorizations"].values()) != {False}:
        raise TextShortcutAuditError("Design preflight encountered an authorization")
    master = bytes.fromhex(str(draft["randomness"]["design_only_master_hex"]))
    capabilities = {
        name: formal.derive_capabilities(master, split=name)
        for name in formal.SPLITS
    }
    commitments = {
        name: formal.capability_commitments(capabilities[name])["generator"][
            "structure"
        ]
        for name in formal.SPLITS
    }
    execution_policy = formal.build_execution_policy(
        draft=draft,
        split=split,
        generator_capabilities=capabilities[split]["generator"],
        structure_commitments=commitments,
    )
    template, fixture, style_profile = formal.load_release_inputs(execution_policy)
    records = formal.split_world_records(execution_policy, split=split)
    if len(records) != 500:
        raise TextShortcutAuditError("Text design world pool is not 500")
    return (
        execution_policy,
        template,
        fixture,
        style_profile,
        capabilities[split]["generator"],
        records,
        validated,
    )


def _attack_from_world(
    *,
    audit_policy: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    split: str,
    world: Mapping[str, Any],
) -> DesignWorldAttackResult:
    public = world["public"]
    private = world["private"]
    rerendered = counterfactual_text.rerender_counterfactual_world(
        execution_policy,
        mode="formal",
        split=split,
        template=template,
        sellers=public["sellers"],
        items=public["items"],
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
        override_audit=private["override_audit"],
    )
    labels = preceremony.validate_full_pair_labels(
        pair_rows=public["complete_model_pair_endpoints"],
        controller_membership=private["controller_membership"],
        expected_world_uid=str(rerendered["world_uid"]),
    )
    visible = build_world_visible_attack_matrices(
        policy=audit_policy,
        seller_profiles=rerendered["public"]["seller_profiles"],
        pair_rows=public["complete_model_pair_endpoints"],
        label_rows=labels,
        negative_flags=private["negative_flags"],
        override_audit=private["override_audit"],
    )
    pair_index = {
        str(row["canonical_pair_uid"]): row
        for row in public["complete_model_pair_endpoints"]
    }
    assignment = assignment_null.build_assignment_null_rows(
        policy=audit_policy,
        template=template,
        sellers=public["sellers"],
        render_asts=private["render_asts"],
        controller_membership=private["controller_membership"],
        controller_style_groups=private["controller_style_groups"],
        target_source_pairs=rerendered["audit"]["derangement"][
            "target_source_pairs"
        ],
        eligible_pair_rows=[pair_index[pair_uid] for pair_uid in visible.pair_uids],
        labels=visible.labels.tolist(),
    )
    return DesignWorldAttackResult(
        visible=visible,
        assignment=assignment,
        rerender_audit=dict(rerendered["audit"]),
    )


def build_design_world_attack(
    *,
    audit_policy: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    style_profile: Mapping[str, Any],
    split: str,
    world_record: Mapping[str, Any],
    generator_capabilities: Mapping[str, str],
) -> DesignWorldAttackResult:
    """Build one nonformal design world without identity-remap allocation."""

    with formal.mounted_structure_capability(
        split=split,
        structure_key_hex=str(generator_capabilities["structure"]),
    ):
        world = world_builder.build_world(
            policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode="formal",
            world_record=world_record,
            structure_key_hex=str(generator_capabilities["structure"]),
        )
    return _attack_from_world(
        audit_policy=audit_policy,
        execution_policy=execution_policy,
        template=template,
        split=split,
        world=world,
    )


def _attack_from_materialized_bundle(
    *,
    audit_policy: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    split: str,
    bundle: Mapping[str, Any],
) -> DesignWorldAttackResult:
    public = bundle["public"]
    private = bundle["private"]
    shadow_world = {
        "public": {
            "sellers": public["sellers"],
            "items": private["raw_identity_bearing_items"],
            "complete_model_pair_endpoints": public[
                "complete_model_pair_endpoints"
            ],
        },
        "private": {
            "identity_slots_audit": private["renderer_identity_slots"],
            "noise_slots_audit": private["renderer_noise_slots"],
            "render_asts": private["render_asts"],
            "override_audit": private["registered_override_audit"],
            "controller_membership": private["controller_membership"],
            "controller_style_groups": private["controller_style_groups"],
            "negative_flags": private["negative_flags"],
        },
    }
    return _attack_from_world(
        audit_policy=audit_policy,
        execution_policy=execution_policy,
        template=template,
        split=split,
        world=shadow_world,
    )


def validate_text_fast_full_parity() -> dict[str, Any]:
    """Prove the no-remap text fast path equals full materialization after redaction."""

    audit_policy = load_text_audit_policy()
    rows: list[dict[str, Any]] = []
    for split in ("train", "development"):
        (
            execution_policy,
            template,
            fixture,
            style_profile,
            generator_capabilities,
            records,
            validated,
        ) = _design_split_context(split=split)
        record = records[0]
        fast = build_design_world_attack(
            audit_policy=audit_policy,
            execution_policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=generator_capabilities,
        )
        full_bundle = formal.materialize_world_bundle(
            execution_policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=generator_capabilities,
            historical_forbidden_hashes=validated["baseline"][
                "failed_identity_hashes"
            ],
            allocated_identity_hashes=set(),
            maximum_identity_counter=int(
                validated["draft"]["identity_collision_resolution"][
                    "maximum_counter"
                ]
            ),
        )
        full = _attack_from_materialized_bundle(
            audit_policy=audit_policy,
            execution_policy=execution_policy,
            template=template,
            split=split,
            bundle=full_bundle,
        )
        if (
            fast.visible.world_uid != full.visible.world_uid
            or fast.visible.pair_uids != full.visible.pair_uids
            or not np.array_equal(fast.visible.labels, full.visible.labels)
            or fast.visible.feature_names_by_view
            != full.visible.feature_names_by_view
            or any(
                not np.array_equal(
                    fast.visible.views[view_name], full.visible.views[view_name]
                )
                for view_name in fast.visible.views
            )
            or preceremony.canonical_json_bytes(fast.assignment)
            != preceremony.canonical_json_bytes(full.assignment)
        ):
            raise TextShortcutAuditError(
                f"Text fast/full materialization parity failed: {split}"
            )
        rows.append(
            {
                "split": split,
                "world_uid": fast.visible.world_uid,
                "pair_count": len(fast.visible.pair_uids),
                "matrix_sha256_by_view": {
                    name: hashlib.sha256(
                        np.ascontiguousarray(matrix.astype(">f8", copy=False)).tobytes()
                    ).hexdigest()
                    for name, matrix in fast.visible.views.items()
                },
                "assignment_sha256": preceremony.canonical_sha256(
                    fast.assignment
                ),
            }
        )
    return {
        "status": "PASS_TEXT_FAST_FULL_REDACTED_PARITY",
        "design_only": True,
        "formal_seed_or_key_access": False,
        "world_count": 2,
        "rows": rows,
    }


def build_design_split_attack_data(
    *,
    split: str,
    world_count: int,
    progress_every: int = 10,
) -> DesignSplitAttackData:
    """Construct counterfactual design matrices without persisting raw worlds."""

    if not 1 <= world_count <= 500 or progress_every < 0:
        raise TextShortcutAuditError("Design text split scale drift")
    audit_policy = load_text_audit_policy()
    (
        execution_policy,
        template,
        fixture,
        style_profile,
        generator_capabilities,
        records,
        _validated,
    ) = _design_split_context(split=split)
    world_results: list[WorldVisibleAttackMatrices] = []
    relation_names = tuple(
        str(value)
        for value in audit_policy["assignment_null_audit"][
            "pair_gate_relations_in_order"
        ]
    )
    observed_by_world: dict[str, np.ndarray] = {}
    expected_by_world: dict[str, np.ndarray] = {}
    world_audit_by_uid: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records[:world_count]):
        result = build_design_world_attack(
            audit_policy=audit_policy,
            execution_policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=generator_capabilities,
        )
        assignment_rows = result.assignment["pair_rows"]
        if (
            len(assignment_rows) != 372
            or [str(row["canonical_pair_uid"]) for row in assignment_rows]
            != list(result.visible.pair_uids)
            or [int(row["label"]) for row in assignment_rows]
            != result.visible.labels.tolist()
        ):
            raise TextShortcutAuditError("Assignment/visible pair alignment drift")
        world_uid = result.visible.world_uid
        if (
            world_uid in observed_by_world
            or world_uid in expected_by_world
            or world_uid in world_audit_by_uid
        ):
            raise TextShortcutAuditError("Design text world UID collision")
        observed_by_world[world_uid] = np.asarray(
            [[float(row[name]) for name in relation_names] for row in assignment_rows],
            dtype=np.float64,
        )
        expected_by_world[world_uid] = np.asarray(
            [
                [float(row[f"expected__{name}"]) for name in relation_names]
                for row in assignment_rows
            ],
            dtype=np.float64,
        )
        world_results.append(result.visible)
        world_audit_by_uid[world_uid] = {
            "world_uid": world_uid,
            "derangement": dict(result.rerender_audit["derangement"]),
            "production_audit_sha256": preceremony.canonical_sha256(
                {
                    "parser": result.rerender_audit["parser"],
                    "redaction": result.rerender_audit["redaction"],
                    "profile": result.rerender_audit["profile"],
                }
            ),
            "parser_exact_rows_and_flags": bool(
                result.rerender_audit["parser"]["exact_rows_and_flags"]
            ),
            "planned_identity_surface_residue_count": int(
                result.rerender_audit["redaction"][
                    "planned_identity_surface_residue_count"
                ]
            ),
            "seller_profile_count": int(
                result.rerender_audit["profile"]["seller_count"]
            ),
            "source_style_changed_seller_count": int(
                result.rerender_audit["source_style_changed_seller_count"]
            ),
            "raw_title_changed_item_count": int(
                result.rerender_audit["raw_title_changed_item_count"]
            ),
            "raw_description_changed_item_count": int(
                result.rerender_audit["raw_description_changed_item_count"]
            ),
        }
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"TEXT_SHORTCUT_WORLD_PROGRESS {split} {index + 1}/{world_count}",
                flush=True,
            )
    visible = aggregate_world_visible_attack_matrices(
        world_results,
        split=split,
        expected_world_count=world_count,
    )
    ordered_world_uids = tuple(
        sorted(observed_by_world, key=lambda value: value.encode("utf-8"))
    )
    if (
        set(ordered_world_uids) != set(expected_by_world)
        or set(ordered_world_uids) != set(world_audit_by_uid)
    ):
        raise TextShortcutAuditError("Assignment split world keyset drift")
    observed = np.vstack(
        [observed_by_world[world_uid] for world_uid in ordered_world_uids]
    ).astype(np.float64, copy=False)
    expected = np.vstack(
        [expected_by_world[world_uid] for world_uid in ordered_world_uids]
    ).astype(np.float64, copy=False)
    assignment_worlds = tuple(
        world_uid for world_uid in ordered_world_uids for _row in range(372)
    )
    world_audits = tuple(
        world_audit_by_uid[world_uid] for world_uid in ordered_world_uids
    )
    if (
        observed.shape != (world_count * 372, len(relation_names))
        or expected.shape != observed.shape
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(expected))
        or np.any(expected < 0.0)
        or np.any(expected > 1.0)
        or assignment_worlds != visible.world_uids
    ):
        raise TextShortcutAuditError("Assignment split matrix closure failed")
    return DesignSplitAttackData(
        visible=visible,
        assignment_relation_names=relation_names,
        assignment_observed=observed,
        assignment_expected=expected,
        assignment_world_uids=assignment_worlds,
        world_audits=world_audits,
    )


def evaluate_assignment_null_gate(
    *, policy: Mapping[str, Any], development: DesignSplitAttackData
) -> dict[str, Any]:
    expected_names = tuple(
        str(value)
        for value in policy["assignment_null_audit"][
            "pair_gate_relations_in_order"
        ]
    )
    if (
        development.visible.split != "development"
        or len(set(development.assignment_world_uids)) != 500
        or development.assignment_relation_names != expected_names
        or development.assignment_observed.shape != (500 * 372, 10)
        or development.assignment_expected.shape
        != development.assignment_observed.shape
        or development.assignment_world_uids != development.visible.world_uids
        or development.visible.labels.shape != (500 * 372,)
        or not np.all(np.isfinite(development.assignment_observed))
        or not np.all(np.isfinite(development.assignment_expected))
    ):
        raise TextShortcutAuditError("Assignment development gate universe drift")
    labels = development.visible.labels
    relation_metrics: dict[str, Any] = {}
    point_maximum = 0.5
    for index, name in enumerate(development.assignment_relation_names):
        values = development.assignment_observed[:, index]
        auc = _point_roc_auc(policy, labels, values)
        symmetric = max(auc, 1.0 - auc)
        point_maximum = max(point_maximum, symmetric)
        expected = development.assignment_expected[:, index]
        residual = values - expected
        relation_metrics[name] = {
            "roc_auc": auc,
            "symmetric_roc_auc": symmetric,
            "observed_mean_positive": float(np.mean(values[labels == 1])),
            "observed_mean_negative": float(np.mean(values[labels == 0])),
            "exact_null_mean_positive": float(np.mean(expected[labels == 1])),
            "exact_null_mean_negative": float(np.mean(expected[labels == 0])),
            "residual_mean_positive": float(np.mean(residual[labels == 1])),
            "residual_mean_negative": float(np.mean(residual[labels == 0])),
        }
    multiplicities, row_world, draw_hash = draw_world_multiplicities(
        policy=policy,
        world_uids=development.assignment_world_uids,
        split="development",
        seed_field="assignment_seed",
    )
    bootstrap_auc = []
    for index in range(len(development.assignment_relation_names)):
        auc_values, _ap_values = bootstrap_rank_metrics(
            labels=labels,
            scores=development.assignment_observed[:, index],
            multiplicities=multiplicities,
            row_world=row_world,
        )
        bootstrap_auc.append(auc_values)
    family_maximum = np.maximum.reduce(bootstrap_auc)
    upper = _bootstrap_upper(policy, family_maximum)
    config = policy["assignment_null_audit"]
    gates = {
        "development_maximum_direct_symmetric_auc": point_maximum
        <= float(config["development_maximum_direct_symmetric_auc"]),
        "development_bootstrap_95_upper_family_max_symmetric_auc": upper
        <= float(
            config["development_bootstrap_95_upper_family_max_symmetric_auc"]
        ),
    }
    return {
        "status": (
            "PASS_ASSIGNMENT_NULL_GATES"
            if all(gates.values())
            else "FAIL_ASSIGNMENT_NULL_GATES"
        ),
        "classifier_fitted": False,
        "relation_metrics": relation_metrics,
        "point_maximum_direct_symmetric_auc": point_maximum,
        "bootstrap_95_upper_family_max_symmetric_auc": upper,
        "bootstrap_replicates": int(policy["bootstrap"]["replicates"]),
        "bootstrap_draw_sha256": draw_hash,
        "hard_gates": gates,
    }
