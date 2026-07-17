#!/usr/bin/env python3
"""Shared contracts for the Step25-v2 pair-local copy diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

import step24_common as step24
import step25_common as step25_v1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step25_v2_pair_local_copy_diagnostic_policy.json"


def resolve(value: str | Path) -> Path:
    return step24.resolve(value)


def load_policy(path: str | Path = DEFAULT_POLICY) -> tuple[Path, dict, dict, dict]:
    policy_path = resolve(path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    step24_policy = json.loads(
        resolve(policy["inputs"]["step24_policy"]).read_text(encoding="utf-8")
    )
    step25_v1_policy = json.loads(
        resolve(policy["inputs"]["step25_v1_policy"]).read_text(encoding="utf-8")
    )
    step24.validate_policy(step24_policy)
    step25_v1.validate_policy(step25_v1_policy, step24_policy)
    validate_policy(policy, step24_policy, step25_v1_policy)
    return policy_path, policy, step24_policy, step25_v1_policy


def validate_policy(policy: dict, step24_policy: dict, step25_v1_policy: dict) -> None:
    boundary = policy["boundary"]
    hard_false = (
        "d1_candidate_eligibility_hard_false",
        "publication_promotion_hard_false",
        "step11_or_step17_entry_forbidden",
    )
    if any(boundary.get(key) is not True for key in hard_false):
        raise ValueError("Step25-v2 retrospective boundary was relaxed")
    if not boundary["valid_or_test_read_forbidden"] or not boundary[
        "parameter_search_on_d0_forbidden"
    ]:
        raise ValueError("Step25-v2 cannot read valid/test or search parameters on D0")
    if policy["clean_text_contract"] != step24_policy["clean_text_contract"]:
        raise ValueError("Step25-v2 must exactly replay the Step24/v7 clean-text contract")
    roots = {
        policy["outputs_root"],
        policy["inputs"]["step24_outputs_root"],
        policy["inputs"]["step25_v1_outputs_root"],
    }
    if len(roots) != 3:
        raise ValueError("Step25-v2 output root must be isolated from Step24 and Step25-v1")

    detector = policy["pair_local_copy_detector"]
    required_true = (
        "label_free",
        "pair_local_only",
        "casefold",
        "collapse_whitespace",
        "mask_both_sides_symmetrically",
        "persist_shared_shingle_hashes_only",
        "labels_evidence_types_scores_and_split_metrics_forbidden",
        "detector_parameter_search_forbidden",
    )
    if any(detector.get(key) is not True for key in required_true):
        raise ValueError("Step25-v2 pair-local detector isolation was relaxed")
    if detector["global_document_frequency_required"] is not False:
        raise ValueError("Step25-v2 detector must recover pair-only copy without global support")
    if detector["persist_raw_shared_span_text"] is not False:
        raise ValueError("Step25-v2 cannot persist raw shared spans")
    fixed_detector = {
        "character_shingle_length": 12,
        "minimum_contiguous_mask_characters": 24,
        "minimum_reliable_remaining_content_characters": 32,
    }
    if any(int(detector.get(key, -1)) != value for key, value in fixed_detector.items()):
        raise ValueError("Step25-v2 fixed detector parameters differ from preregistration")
    if detector["unicode_normalization"] != "NFKC":
        raise ValueError("Step25-v2 Unicode normalization differs from preregistration")
    if float(detector["maximum_mask_fraction_per_side"]) != 0.95:
        raise ValueError("Step25-v2 maximum mask fraction differs from preregistration")
    if float(detector["minimum_alphanumeric_fraction_per_shingle"]) != 0.5:
        raise ValueError("Step25-v2 shingle content threshold differs from preregistration")

    for encoder_key, cfg in policy["frozen_style_encoders"].items():
        parent = step24_policy["frozen_style_encoders"].get(encoder_key)
        if parent is None:
            raise ValueError(f"Step25-v2 encoder is absent from Step24: {encoder_key}")
        for key in (
            "repo_id",
            "revision",
            "local_path",
            "loader",
            "expected_dimension",
            "maximum_sequence_length",
            "batch_size",
            "normalize_embeddings",
        ):
            if cfg[key] != parent[key]:
                raise ValueError(f"Step25-v2 changed frozen encoder contract: {encoder_key}:{key}")
        if cfg["local_finetuning_forbidden"] is not True:
            raise ValueError("Step25-v2 encoder fine-tuning is forbidden")

    expected_specs = {
        "P0_raw_style_matched_missingness": (
            ["raw_pcm_multilingual_authorship_cosine", "raw_mstyledistance_cosine"],
            "pair_local_style_reliable",
            "fold_train_reliable_median_plus_indicator",
        ),
        "P1_global_clean_style_matched_missingness": (
            [
                "global_clean_pcm_multilingual_authorship_cosine",
                "global_clean_mstyledistance_cosine",
            ],
            "global_and_pair_local_style_reliable",
            "fold_train_reliable_median_plus_indicator",
        ),
        "P2_pair_local_clean_style_matched_missingness": (
            [
                "pair_local_clean_pcm_multilingual_authorship_cosine",
                "pair_local_clean_mstyledistance_cosine",
            ],
            "pair_local_style_reliable",
            "fold_train_reliable_median_plus_indicator",
        ),
        "P3_pair_local_clean_raw_fallback": (
            [
                "pair_local_or_raw_pcm_multilingual_authorship_cosine",
                "pair_local_or_raw_mstyledistance_cosine",
            ],
            "pair_local_style_reliable",
            "raw_style_fallback_plus_indicator",
        ),
    }
    specs = policy["evaluation"]["model_specs"]
    if set(specs) != set(expected_specs):
        raise ValueError("Step25-v2 P0-P3 model matrix differs from preregistration")
    for name, (features, reliability, mode) in expected_specs.items():
        if (
            specs[name]["style_features"] != features
            or specs[name]["reliability_feature"] != reliability
            or specs[name]["missingness_mode"] != mode
        ):
            raise ValueError(f"Step25-v2 model spec differs from preregistration: {name}")
    evaluation = policy["evaluation"]
    if evaluation["primary_model"] != "P2_pair_local_clean_style_matched_missingness":
        raise ValueError("Step25-v2 primary diagnostic must be P2")
    if evaluation["matched_baseline_model"] != "P0_raw_style_matched_missingness":
        raise ValueError("Step25-v2 matched baseline must be P0")
    p4 = evaluation["P4_reliable_pair_only_sensitivity"]
    if (
        p4["compare"]
        != [
            "P0_raw_style_matched_missingness",
            "P2_pair_local_clean_style_matched_missingness",
        ]
        or p4["fit_scope"] != "same_models_as_full_boundary"
        or p4["metric_scope"] != "rows_with_pair_local_style_reliable_equal_1"
        or p4["selection_forbidden"] is not True
    ):
        raise ValueError("Step25-v2 P4 reliable-only sensitivity contract changed")
    if evaluation["zero_cosine_as_missing_value_forbidden"] is not True:
        raise ValueError("Step25-v2 cannot encode missing cleaned style as cosine zero")
    if evaluation["imputation_fit_scope"] != "inside_each_component_grouped_training_fold_only":
        raise ValueError("Step25-v2 imputation must be fitted inside each fold")
    if int(evaluation["fold_count"]) != 5 or evaluation["canonical_split"] != "train":
        raise ValueError("Step25-v2 uses the canonical train boundary and five grouped folds")
    logistic = evaluation["logistic"]
    if (
        float(logistic["l2_penalty"]) != 10.0
        or logistic["class_weight"] != "none"
        or logistic["standardize_features"] is not True
    ):
        raise ValueError("Step25-v2 fixed LR/L2 contract changed")
    if step25_v1_policy["outputs_root"] != policy["inputs"]["step25_v1_outputs_root"]:
        raise ValueError("Step25-v2 Step25-v1 input root disagrees with the frozen v1 policy")
    required_outputs = {
        "pair_local_texts_en",
        "pair_local_texts_zh",
        "detector_summary",
        "embedding_manifest",
        "pair_features_en",
        "pair_features_zh",
        "pair_feature_summary",
        "predictions_en",
        "predictions_zh",
        "model_artifacts",
        "evaluation_summary",
        "sync_manifest",
    }
    outputs = policy["outputs"]
    if set(outputs) != required_outputs or len(set(outputs.values())) != len(outputs):
        raise ValueError("Step25-v2 output map is incomplete or contains path collisions")


def load_rows(policy: dict, step24_policy: dict, step25_v1_policy: dict) -> dict[str, list[dict]]:
    rows = step25_v1.load_rows(step25_v1_policy, step24_policy)
    for pool_rows in rows.values():
        if any(row.get("split_name") != "train" for row in pool_rows):
            raise ValueError("Step25-v2 found a non-train canonical row")
    return rows


def replay_train_text_index(
    pool_name: str,
    rows: list[dict],
    step24_policy: dict,
) -> tuple[dict[str, str], dict]:
    pool_cfg = step24_policy["pools"][pool_name]
    metadata_path = resolve(pool_cfg["identifier_redacted_e5_metadata"])
    matrix_path = resolve(pool_cfg["identifier_redacted_e5_matrix"])
    _index, _matrix, metadata = step24.load_normalized_cache(metadata_path, matrix_path)
    if metadata.get("identifier_redacted") is not True:
        raise ValueError(f"Step25-v2 E5 corpus is not identifier-redacted: {pool_name}")
    all_sellers = list(metadata["seller_uids"])
    all_texts, diagnostics = step24.replay_v7_clean_texts(
        pool_cfg, step24_policy["clean_text_contract"], all_sellers
    )
    expected_hash = metadata.get("redaction_diagnostics", {}).get("clean_text_corpus_sha256")
    if expected_hash != diagnostics["clean_text_corpus_sha256"]:
        raise ValueError(f"Step25-v2 clean-text replay differs from frozen v7: {pool_name}")
    full = dict(zip(all_sellers, all_texts, strict=True))
    selected = step24.train_sellers(rows)
    missing = sorted(set(selected) - set(full))
    if missing:
        raise ValueError(f"Step25-v2 train seller lacks replayed text: {pool_name}:{missing[0]}")
    index = {seller: full[seller] for seller in selected}
    return index, {
        "pool": pool_name,
        "canonical_train_pair_count": len(rows),
        "canonical_train_seller_count": len(selected),
        "valid_test_seller_count": 0,
        "full_v7_clean_text_corpus_sha256_verified": expected_hash,
        "selected_train_text_sha256": step24.canonical_hash(sorted(index.items())),
        "e5_metadata_sha256": step24.sha256_file(metadata_path),
        "e5_matrix_sha256": step24.sha256_file(matrix_path),
    }


def normalize_text(text: str, cfg: dict) -> str:
    value = unicodedata.normalize(cfg["unicode_normalization"], str(text or ""))
    if cfg["casefold"]:
        value = value.casefold()
    if cfg["collapse_whitespace"]:
        value = re.sub(r"\s+", " ", value).strip()
    return value


def content_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def eligible_shingle_positions(text: str, cfg: dict):
    length = int(cfg["character_shingle_length"])
    minimum = int(math.ceil(length * float(cfg["minimum_alphanumeric_fraction_per_shingle"])))
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + int(character.isalnum()))
    for index in range(max(0, len(text) - length + 1)):
        if prefix[index + length] - prefix[index] >= minimum:
            yield index, text[index : index + length]


def bounded_mask_runs(mask: list[bool], minimum_run: int, maximum_chars: int) -> list[tuple[int, int]]:
    candidates = []
    start = None
    for index, marked in enumerate(mask + [False]):
        if marked and start is None:
            start = index
        elif not marked and start is not None:
            if index - start >= minimum_run:
                candidates.append((start, index))
            start = None
    selected = []
    remaining = max(0, maximum_chars)
    for start, end in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if remaining < minimum_run:
            break
        take = min(end - start, remaining)
        if take >= minimum_run:
            selected.append((start, start + take))
            remaining -= take
    return sorted(selected)


def apply_runs(text: str, runs: list[tuple[int, int]]) -> str:
    pieces = []
    cursor = 0
    for start, end in runs:
        pieces.extend((text[cursor:start], " "))
        cursor = end
    pieces.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def detect_pair_local_copy(left_text: str, right_text: str, cfg: dict) -> dict:
    left = normalize_text(left_text, cfg)
    right = normalize_text(right_text, cfg)
    left_positions: dict[str, list[int]] = defaultdict(list)
    right_positions: dict[str, list[int]] = defaultdict(list)
    for index, token in eligible_shingle_positions(left, cfg):
        left_positions[token].append(index)
    for index, token in eligible_shingle_positions(right, cfg):
        right_positions[token].append(index)
    shared = sorted(set(left_positions) & set(right_positions))
    length = int(cfg["character_shingle_length"])
    left_mask = [False] * len(left)
    right_mask = [False] * len(right)
    for token in shared:
        for start in left_positions[token]:
            for position in range(start, min(start + length, len(left_mask))):
                left_mask[position] = True
        for start in right_positions[token]:
            for position in range(start, min(start + length, len(right_mask))):
                right_mask[position] = True
    minimum_run = int(cfg["minimum_contiguous_mask_characters"])
    left_runs = bounded_mask_runs(
        left_mask,
        minimum_run,
        int(math.floor(len(left) * float(cfg["maximum_mask_fraction_per_side"]))),
    )
    right_runs = bounded_mask_runs(
        right_mask,
        minimum_run,
        int(math.floor(len(right) * float(cfg["maximum_mask_fraction_per_side"]))),
    )
    clean_left = apply_runs(left, left_runs)
    clean_right = apply_runs(right, right_runs)
    left_masked = sum(end - start for start, end in left_runs)
    right_masked = sum(end - start for start, end in right_runs)
    minimum_reliable = int(cfg["minimum_reliable_remaining_content_characters"])
    left_reliable = content_character_count(clean_left) >= minimum_reliable
    right_reliable = content_character_count(clean_right) >= minimum_reliable
    if not clean_left:
        clean_left = "content unavailable after pair-local copy removal"
    if not clean_right:
        clean_right = "content unavailable after pair-local copy removal"
    return {
        "left_clean_text": clean_left,
        "right_clean_text": clean_right,
        "left_source_sha256": hashlib.sha256(left.encode("utf-8")).hexdigest(),
        "right_source_sha256": hashlib.sha256(right.encode("utf-8")).hexdigest(),
        "left_clean_sha256": hashlib.sha256(clean_left.encode("utf-8")).hexdigest(),
        "right_clean_sha256": hashlib.sha256(clean_right.encode("utf-8")).hexdigest(),
        "left_original_character_count": len(left),
        "right_original_character_count": len(right),
        "left_remaining_character_count": len(clean_left),
        "right_remaining_character_count": len(clean_right),
        "left_masked_character_count": left_masked,
        "right_masked_character_count": right_masked,
        "left_mask_fraction": left_masked / max(len(left), 1),
        "right_mask_fraction": right_masked / max(len(right), 1),
        "left_masked_span_count": len(left_runs),
        "right_masked_span_count": len(right_runs),
        "shared_shingle_hashes": [
            hashlib.sha256(token.encode("utf-8")).hexdigest() for token in shared
        ],
        "shared_shingle_count": len(shared),
        "left_reliable": int(left_reliable),
        "right_reliable": int(right_reliable),
        "pair_reliable": int(left_reliable and right_reliable),
    }


def pair_side_key(pair_uid: str, side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown Step25-v2 pair side: {side}")
    return f"{pair_uid}::{side}"


def pair_text_path(policy: dict, pool_name: str) -> Path:
    key = "pair_local_texts_en" if pool_name == "en_content_train_pool" else "pair_local_texts_zh"
    return resolve(policy["outputs_root"]) / policy["outputs"][key]


def embedding_paths(policy: dict, encoder_key: str, pool_name: str) -> tuple[Path, Path]:
    root = resolve(policy["outputs_root"]) / "embeddings"
    stem = f"{encoder_key}.pair_local.{pool_name}"
    return root / f"{stem}.npy", root / f"{stem}.json"


def load_pair_embedding_cache(metadata_path: Path, matrix_path: Path) -> tuple[dict[str, int], np.ndarray, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r")
    keys = list(metadata.get("pair_side_keys", []))
    if list(matrix.shape) != list(metadata.get("shape", [])):
        raise ValueError(f"Step25-v2 cache shape mismatch: {matrix_path}")
    if len(keys) != matrix.shape[0] or len(set(keys)) != len(keys):
        raise ValueError(f"Step25-v2 pair-side index mismatch: {metadata_path}")
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError(f"Step25-v2 cache is not unit-normalized: {matrix_path}")
    return {key: index for index, key in enumerate(keys)}, matrix, metadata


def matched_missingness_design(
    train_style: np.ndarray,
    train_reliable: np.ndarray,
    score_style: np.ndarray,
    score_reliable: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_style = np.asarray(train_style, dtype=float)
    score_style = np.asarray(score_style, dtype=float)
    train_reliable = np.asarray(train_reliable, dtype=bool)
    score_reliable = np.asarray(score_reliable, dtype=bool)
    if train_style.ndim != 2 or score_style.ndim != 2:
        raise ValueError("Step25-v2 style matrices must be two-dimensional")
    if train_style.shape[1] != score_style.shape[1]:
        raise ValueError("Step25-v2 train/score style dimensions differ")
    if len(train_style) != len(train_reliable) or len(score_style) != len(score_reliable):
        raise ValueError("Step25-v2 style/reliability lengths differ")
    if mode == "fold_train_reliable_median_plus_indicator":
        if not np.any(train_reliable):
            raise ValueError("Step25-v2 fold contains no reliable rows for imputation")
        medians = []
        for column in range(train_style.shape[1]):
            values = train_style[train_reliable, column]
            if not np.all(np.isfinite(values)):
                raise ValueError("Step25-v2 reliable training style contains non-finite values")
            medians.append(float(np.median(values)))
        medians_array = np.asarray(medians, dtype=float)
        train_values = np.where(train_reliable[:, None], train_style, medians_array[None, :])
        score_values = np.where(score_reliable[:, None], score_style, medians_array[None, :])
    elif mode == "raw_style_fallback_plus_indicator":
        if not np.all(np.isfinite(train_style)) or not np.all(np.isfinite(score_style)):
            raise ValueError("Step25-v2 fallback style features must already be finite")
        medians = None
        train_values = train_style
        score_values = score_style
    else:
        raise ValueError(f"Unknown Step25-v2 missingness mode: {mode}")
    train_design = np.column_stack([train_values, train_reliable.astype(float)])
    score_design = np.column_stack([score_values, score_reliable.astype(float)])
    if not np.all(np.isfinite(train_design)) or not np.all(np.isfinite(score_design)):
        raise ValueError("Step25-v2 design matrix contains non-finite values")
    return train_design, score_design, {
        "mode": mode,
        "fold_train_style_medians": medians,
        "reliability_indicator_appended": True,
        "missing_encoded_as_fixed_zero": False,
        "train_reliable_count": int(np.sum(train_reliable)),
        "score_reliable_count": int(np.sum(score_reliable)),
    }
