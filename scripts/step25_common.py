#!/usr/bin/env python3
"""Shared contracts for Step25 template-decontaminated authorship experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step9_run_few_shot_adaptation as step9
import step24_common as step24


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step25_template_decontaminated_authorship_policy.json"


def resolve(value: str | Path) -> Path:
    return step24.resolve(value)


def load_policy(path: str | Path = DEFAULT_POLICY) -> tuple[Path, dict, dict]:
    policy_path = resolve(path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    step24_policy_path = resolve(policy["inputs"]["step24_policy"])
    step24_policy = json.loads(step24_policy_path.read_text(encoding="utf-8"))
    step24.validate_policy(step24_policy)
    validate_policy(policy, step24_policy)
    return policy_path, policy, step24_policy


def validate_policy(policy: dict, step24_policy: dict) -> None:
    boundary = policy["development_boundaries"]
    if boundary["d0_current_canonical_train"]["publication_promotion_allowed"]:
        raise ValueError("Step25 D0 cannot authorize a publication promotion")
    if not boundary["d1_future_independent_development"][
        "seller_component_overlap_with_d0_forbidden"
    ]:
        raise ValueError("Step25 D1 must remain component-disjoint from D0")
    if not boundary["f1_future_prospective_holdout"][
        "collection_after_model_freeze_required"
    ]:
        raise ValueError("Step25 F1 must be collected after model freeze")
    text_cfg = policy["clean_text_contract"]
    if text_cfg != step24_policy["clean_text_contract"]:
        raise ValueError("Step25 must replay the exact Step24/v7 identifier-redacted text contract")
    template_cfg = policy["template_decontamination"]
    required_true = (
        "label_free",
        "component_cross_fitted",
        "held_component_sellers_excluded_from_catalog_support",
        "raw_ngram_text_persistence_forbidden",
        "catalog_persists_hashes_only",
        "review_label_evidence_type_and_model_score_forbidden",
    )
    if any(not template_cfg.get(key) for key in required_true):
        raise ValueError("Step25 template-decontamination isolation was relaxed")
    if template_cfg["fit_scope"] != "within_domain_canonical_train_sellers_only":
        raise ValueError("Step25 template catalog must be fitted inside each train domain")
    numeric_contract = {
        "character_shingle_length": 12,
        "minimum_external_seller_document_frequency": 3,
        "minimum_external_component_document_frequency": 2,
        "minimum_contiguous_mask_characters": 24,
        "minimum_reliable_remaining_characters": 32,
    }
    if any(int(template_cfg.get(key, -1)) != value for key, value in numeric_contract.items()):
        raise ValueError("Step25 fixed boilerplate detector differs from preregistration")
    if float(template_cfg["maximum_mask_fraction"]) != 0.95:
        raise ValueError("Step25 maximum mask fraction differs from preregistration")
    if template_cfg["unicode_normalization"] != "NFKC" or not template_cfg["casefold"]:
        raise ValueError("Step25 text normalization differs from preregistration")
    for encoder_key, encoder_cfg in policy["frozen_style_encoders"].items():
        parent = step24_policy["frozen_style_encoders"].get(encoder_key)
        if parent is None:
            raise ValueError(f"Step25 encoder is absent from Step24: {encoder_key}")
        for key in (
            "repo_id",
            "revision",
            "local_path",
            "expected_dimension",
            "maximum_sequence_length",
        ):
            if encoder_cfg[key] != parent[key]:
                raise ValueError(f"Step25 changed the frozen encoder contract: {encoder_key}:{key}")
        if not encoder_cfg["local_finetuning_forbidden"]:
            raise ValueError("Step25 does not permit encoder fine-tuning")
    eval_cfg = policy["evaluation"]
    expected_sets = {
        "raw_style_lr_l2_control": [
            "raw_pcm_multilingual_authorship_cosine",
            "raw_mstyledistance_cosine",
        ],
        "decontaminated_style_lr_l2_primary": [
            "decontaminated_pcm_multilingual_authorship_cosine",
            "decontaminated_mstyledistance_cosine",
        ],
        "decontaminated_semantic_style_lr_l2_secondary": [
            "identifier_redacted_e5_cosine",
            "decontaminated_pcm_multilingual_authorship_cosine",
            "decontaminated_mstyledistance_cosine",
        ],
        "raw_clean_delta_exploratory": [
            "raw_pcm_multilingual_authorship_cosine",
            "raw_mstyledistance_cosine",
            "decontaminated_pcm_multilingual_authorship_cosine",
            "decontaminated_mstyledistance_cosine",
            "pcm_raw_minus_decontaminated",
            "mstyledistance_raw_minus_decontaminated",
            "pair_maximum_boilerplate_fraction",
            "pair_mean_boilerplate_fraction",
            "decontaminated_pair_reliable",
        ],
    }
    if eval_cfg["model_feature_sets"] != expected_sets:
        raise ValueError("Step25 model/control matrix differs from preregistration")
    if eval_cfg["primary_model"] != "decontaminated_style_lr_l2_primary":
        raise ValueError("Step25 primary must be the decontaminated style-only scorer")
    if eval_cfg["matched_baseline_model"] != "raw_style_lr_l2_control":
        raise ValueError("Step25 baseline must be the matched raw style-only scorer")
    if not eval_cfg["secondary_and_exploratory_models_forbidden_for_selection"]:
        raise ValueError("Step25 secondary models cannot participate in selection")
    if not eval_cfg["candidate_selection_forbidden"] or not eval_cfg[
        "valid_or_test_selection_forbidden"
    ]:
        raise ValueError("Step25 valid/test selection isolation was relaxed")
    logistic = eval_cfg["logistic"]
    if (
        float(logistic["l2_penalty"]) != 10.0
        or logistic["class_weight"] != "none"
        or not logistic["standardize_features"]
    ):
        raise ValueError("Step25 fixed LR/L2 contract was changed")
    if eval_cfg["canonical_split"] != "train" or int(eval_cfg["fold_count"]) != 5:
        raise ValueError("Step25 D0 must remain train-only with five grouped folds")
    if not policy["occurrence_reliability"][
        "review_label_evidence_type_model_error_and_split_membership_forbidden_as_features"
    ]:
        raise ValueError("Step25 occurrence reliability feature isolation was relaxed")


def load_rows(policy: dict, step24_policy: dict) -> dict[str, list[dict]]:
    rows_by_pool = step24.load_canonical_train_rows(step24_policy)
    output = {}
    for pool_name, rows in rows_by_pool.items():
        converted = []
        for row in rows:
            converted.append(
                {
                    **row,
                    "step25_pool": pool_name,
                    "step25_component_id": row["step24_component_id"],
                }
            )
        output[pool_name] = converted
    return output


def seller_component_map(rows: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        component = row["step25_component_id"]
        for field in ("seller_uid_left", "seller_uid_right"):
            seller = row[field]
            previous = result.setdefault(seller, component)
            if previous != component:
                raise ValueError(
                    f"Step25 seller spans multiple components: {seller}: {previous} / {component}"
                )
    return result


def normalize_template_text(text: str, cfg: dict) -> str:
    value = unicodedata.normalize(cfg["unicode_normalization"], str(text or ""))
    if cfg["casefold"]:
        value = value.casefold()
    if cfg["collapse_whitespace"]:
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _content_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _shingle_key(value: str) -> str:
    """Use the exact normalized shingle in memory; only its SHA-256 is persisted."""
    return value


def _eligible_shingle_positions(text: str, length: int):
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + int(character.isalnum()))
    for index in range(max(0, len(text) - length + 1)):
        if prefix[index + length] - prefix[index] >= length // 2:
            yield index, _shingle_key(text[index : index + length])


def _eligible_shingle_set(text: str, length: int) -> set[str]:
    return {token for _index, token in _eligible_shingle_positions(text, length)}


def build_template_support(
    seller_uids: list[str],
    texts: list[str],
    component_by_seller: dict[str, str],
    cfg: dict,
) -> tuple[Counter, Counter, Counter, list[str]]:
    if len(seller_uids) != len(texts):
        raise ValueError("Step25 seller/text lengths differ")
    if set(seller_uids) != set(component_by_seller):
        raise ValueError("Step25 template support seller/component map is incomplete")
    normalized = [normalize_template_text(text, cfg) for text in texts]
    length = int(cfg["character_shingle_length"])
    seller_df: Counter[str] = Counter()
    for text in normalized:
        hashes = _eligible_shingle_set(text, length)
        seller_df.update(hashes)
    minimum_df = int(cfg["minimum_external_seller_document_frequency"])
    candidates = {token_hash for token_hash, count in seller_df.items() if count >= minimum_df}
    component_seller_df: Counter[tuple[str, str]] = Counter()
    components_by_hash: dict[str, set[str]] = defaultdict(set)
    for seller, text in zip(seller_uids, normalized, strict=True):
        hashes = _eligible_shingle_set(text, length)
        component = component_by_seller[seller]
        for token_hash in hashes & candidates:
            component_seller_df[(token_hash, component)] += 1
            components_by_hash[token_hash].add(component)
    component_df: Counter[str] = Counter(
        {token_hash: len(components) for token_hash, components in components_by_hash.items()}
    )
    return seller_df, component_seller_df, component_df, normalized


def _mask_runs(mask: list[bool], minimum_run: int, maximum_chars: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, marked in enumerate(mask + [False]):
        if marked and start is None:
            start = index
        elif not marked and start is not None:
            if index - start >= minimum_run:
                runs.append((start, index))
            start = None
    selected = []
    used = 0
    for start, end in sorted(runs, key=lambda item: (-(item[1] - item[0]), item[0])):
        length = end - start
        if used + length <= maximum_chars:
            selected.append((start, end))
            used += length
    return sorted(selected)


def decontaminate_corpus(
    seller_uids: list[str],
    texts: list[str],
    component_by_seller: dict[str, str],
    cfg: dict,
) -> tuple[list[dict], list[dict], dict]:
    """Remove only spans supported by sellers outside the current seller component."""
    seller_df, component_seller_df, component_df, normalized = build_template_support(
        seller_uids, texts, component_by_seller, cfg
    )
    shingle_length = int(cfg["character_shingle_length"])
    min_seller_df = int(cfg["minimum_external_seller_document_frequency"])
    min_component_df = int(cfg["minimum_external_component_document_frequency"])
    minimum_run = int(cfg["minimum_contiguous_mask_characters"])
    records = []
    total_masked = 0
    reliable_count = 0
    for seller, original in zip(seller_uids, normalized, strict=True):
        component = component_by_seller[seller]
        mask = [False] * len(original)
        qualifying_hashes = set()
        for index, token_hash in _eligible_shingle_positions(original, shingle_length):
            own_component_count = component_seller_df[(token_hash, component)]
            external_seller_count = seller_df[token_hash] - own_component_count
            external_component_count = component_df[token_hash] - int(own_component_count > 0)
            if (
                external_seller_count < min_seller_df
                or external_component_count < min_component_df
            ):
                continue
            qualifying_hashes.add(token_hash)
            for position in range(index, min(index + shingle_length, len(mask))):
                mask[position] = True
        maximum_chars = int(math.floor(len(original) * float(cfg["maximum_mask_fraction"])))
        runs = _mask_runs(mask, minimum_run, maximum_chars)
        pieces = []
        cursor = 0
        for start, end in runs:
            pieces.append(original[cursor:start])
            pieces.append(" ")
            cursor = end
        pieces.append(original[cursor:])
        clean = re.sub(r"\s+", " ", "".join(pieces)).strip()
        masked_characters = sum(end - start for start, end in runs)
        fraction = masked_characters / max(len(original), 1)
        reliable = _content_character_count(clean) >= int(
            cfg["minimum_reliable_remaining_characters"]
        )
        if not clean:
            clean = "content unavailable after boilerplate removal"
        total_masked += masked_characters
        reliable_count += int(reliable)
        records.append(
            {
                "seller_uid": seller,
                "component_id": component,
                "normalized_source_sha256": hashlib.sha256(
                    original.encode("utf-8")
                ).hexdigest(),
                "decontaminated_text": clean,
                "decontaminated_text_sha256": hashlib.sha256(
                    clean.encode("utf-8")
                ).hexdigest(),
                "original_character_count": len(original),
                "remaining_character_count": len(clean),
                "masked_character_count": masked_characters,
                "boilerplate_fraction": round(fraction, 12),
                "masked_span_count": len(runs),
                "qualifying_shingle_hash_count": len(qualifying_hashes),
                "decontaminated_text_reliable": int(reliable),
            }
        )
    catalog_records = []
    for token_hash in sorted(seller_df):
        token_seller_df = seller_df[token_hash]
        token_component_df = component_df[token_hash]
        if token_seller_df < min_seller_df or token_component_df < min_component_df:
            continue
        catalog_records.append(
            {
                "shingle_sha256": hashlib.sha256(token_hash.encode("utf-8")).hexdigest(),
                "seller_document_frequency": token_seller_df,
                "component_document_frequency": token_component_df,
                "character_length": shingle_length,
            }
        )
    summary = {
        "seller_count": len(seller_uids),
        "component_count": len(set(component_by_seller.values())),
        "catalog_hash_count": len(catalog_records),
        "total_masked_character_count": total_masked,
        "reliable_seller_count": reliable_count,
        "reliable_seller_fraction": reliable_count / max(len(seller_uids), 1),
        "label_evidence_type_or_model_score_read": False,
        "component_cross_fitted": True,
    }
    return records, catalog_records, summary


def render_jsonl(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Step25 refuses to render an empty JSONL")
    return ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode(
        "utf-8"
    )


def write_jsonl_immutable(path: Path, rows: list[dict]) -> None:
    step24.write_bytes_immutable(path, render_jsonl(rows))


def write_csv_immutable_allow_empty(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """Persist a deterministic CSV while preserving a valid zero-row result."""
    if rows and any(list(row) != fieldnames for row in rows):
        raise ValueError(f"Step25 CSV field order differs: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    step24.write_bytes_immutable(path, payload)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def template_output_paths(output_root: Path, pool_name: str) -> tuple[Path, Path, Path]:
    return (
        output_root / "template_catalogs" / f"catalog_hashes.{pool_name}.csv",
        output_root / "decontaminated_texts" / f"decontaminated_texts.{pool_name}.jsonl",
        output_root / "decontaminated_texts" / f"decontamination_summary.{pool_name}.json",
    )


def embedding_output_paths(output_root: Path, encoder_key: str, pool_name: str) -> tuple[Path, Path]:
    stem = f"{encoder_key}.decontaminated.{pool_name}"
    return output_root / "embeddings" / f"{stem}.npy", output_root / "embeddings" / f"{stem}.json"


def pair_cosine(
    row: dict,
    index: dict[str, int],
    matrix: np.ndarray,
    reliable_by_seller: dict[str, bool] | None = None,
    insufficient_value: float = 0.0,
) -> float:
    left = row["seller_uid_left"]
    right = row["seller_uid_right"]
    if left not in index or right not in index:
        raise ValueError(f"Step25 pair seller is absent from embedding cache: {row['pair_uid']}")
    if reliable_by_seller is not None and not (
        reliable_by_seller.get(left, False) and reliable_by_seller.get(right, False)
    ):
        return float(insufficient_value)
    return float(np.dot(np.asarray(matrix[index[left]]), np.asarray(matrix[index[right]])))


def occurrence_indexes(policy: dict, step24_policy: dict, rows_by_pool: dict[str, list[dict]]) -> dict:
    indexes = {}
    for pool_name, rows in rows_by_pool.items():
        train_sellers = set(step24.train_sellers(rows))
        signal_path = resolve(step24_policy["pools"][pool_name]["item_identity_signals"])
        indexes[pool_name] = item_signal_index(signal_path, train_sellers)
    return indexes


def occurrence_for_rows(rows: list[dict], indexes: dict, frequency_threshold: int) -> list[dict]:
    output = []
    for row in rows:
        by_seller, token_df = indexes[row["step25_pool"]]
        output.append(occurrence_evidence(row, by_seller, token_df, frequency_threshold))
    return output


def item_signal_index(path: Path, train_sellers: set[str]) -> tuple[dict, Counter]:
    by_seller: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sellers_by_token: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in step24.load_csv(path):
        contact_type = str(row.get("contact_type", "")).strip().lower()
        value = str(row.get("normalized_value", "")).strip().lower()
        seller = str(row.get("seller_uid", "")).strip()
        if not contact_type or not value or not seller:
            continue
        token = (contact_type, value)
        by_seller[seller][token].append(row)
        if seller in train_sellers:
            sellers_by_token[token].add(seller)
    return by_seller, Counter(
        {token: len(sellers) for token, sellers in sellers_by_token.items()}
    )


def _direct_occurrence(row: dict) -> bool:
    return (
        row.get("direct_identity_eligible") == "1"
        and row.get("seller_facing_context") == "1"
        and row.get("product_data_risk_context") != "1"
        and row.get("support_only") != "1"
    )


def _risky_occurrence(row: dict) -> bool:
    return row.get("product_data_risk_context") == "1"


def _support_occurrence(row: dict) -> bool:
    return row.get("support_only") == "1"


def occurrence_evidence(
    row: dict,
    by_seller: dict,
    token_df: Counter,
    frequency_threshold: int,
) -> dict:
    left = by_seller.get(row["seller_uid_left"], {})
    right = by_seller.get(row["seller_uid_right"], {})
    shared = sorted(set(left) & set(right))
    counts = Counter()
    distinct_items = set()
    distinct_markets = set()
    token_types = set()
    token_hashes = []
    for token in shared:
        left_occurrences = left[token]
        right_occurrences = right[token]
        all_occurrences = left_occurrences + right_occurrences
        left_direct = any(_direct_occurrence(item) for item in left_occurrences)
        right_direct = any(_direct_occurrence(item) for item in right_occurrences)
        risky = any(_risky_occurrence(item) for item in all_occurrences)
        support = any(_support_occurrence(item) for item in all_occurrences)
        high_frequency = token_df[token] > frequency_threshold
        token_types.add(token[0])
        token_hashes.append(step24.canonical_hash(token)[:16])
        for item in all_occurrences:
            source_row = str(item.get("source_row_number", "")).strip()
            source_dataset = str(item.get("source_dataset", "")).strip()
            if source_row or source_dataset:
                distinct_items.add(f"{source_dataset}:{source_row}")
            market = str(item.get("source_market_raw", "")).strip()
            if market:
                distinct_markets.add(market)
        if left_direct and right_direct and (risky or support):
            counts["mixed_context"] += 1
        elif left_direct and right_direct and not high_frequency:
            counts["verified_direct"] += 1
        elif risky:
            counts["risky_only"] += 1
        elif support:
            counts["support_only"] += 1
        elif high_frequency:
            counts["high_frequency"] += 1
        else:
            counts["ambiguous"] += 1
    if counts["verified_direct"]:
        state = "verified_direct_both_sides"
    elif counts["mixed_context"]:
        state = "direct_with_mixed_context"
    elif counts["risky_only"]:
        state = "risky_only_shared"
    elif counts["support_only"]:
        state = "support_only_shared"
    elif counts["high_frequency"]:
        state = "high_frequency_public"
    elif shared:
        state = "ambiguous"
    else:
        state = "no_shared_identifier"
    financial_phone_types = {
        "phone",
        "crypto_wallet",
        "wallet",
        "pgp_fingerprint",
        "pgp_public_key",
        "qq",
        "wechat",
        "jabber",
    }
    return {
        "evidence_state": state,
        "verified_direct_token_count": int(counts["verified_direct"]),
        "risky_only_token_count": int(counts["risky_only"]),
        "support_only_token_count": int(counts["support_only"]),
        "mixed_context_token_count": int(counts["mixed_context"]),
        "high_frequency_token_count": int(counts["high_frequency"]),
        "ambiguous_token_count": int(counts["ambiguous"]),
        "shared_token_count": len(shared),
        "maximum_train_seller_token_frequency": max(
            (token_df[token] for token in shared), default=0
        ),
        "distinct_item_count": len(distinct_items),
        "distinct_market_count": len(distinct_markets),
        "public_url_or_domain_flag": int(
            bool(token_types & {"url", "domain", "external_url"})
        ),
        "identifier_type_telegram_flag": int("telegram" in token_types),
        "identifier_type_email_flag": int("email" in token_types),
        "identifier_type_financial_phone_flag": int(
            bool(token_types & financial_phone_types)
        ),
        "shared_token_hashes": token_hashes,
        "shared_identifier_types": sorted(token_types),
    }


def reliability_feature_matrix(evidence: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(evidence), len(feature_names)), dtype=float)
    for index, item in enumerate(evidence):
        values = {
            "verified_direct_token_count_log1p": math.log1p(item["verified_direct_token_count"]),
            "risky_only_token_count_log1p": math.log1p(item["risky_only_token_count"]),
            "support_only_token_count_log1p": math.log1p(item["support_only_token_count"]),
            "mixed_context_token_count_log1p": math.log1p(item["mixed_context_token_count"]),
            "high_frequency_token_count_log1p": math.log1p(item["high_frequency_token_count"]),
            "shared_token_count_log1p": math.log1p(item["shared_token_count"]),
            "distinct_item_count_log1p": math.log1p(item["distinct_item_count"]),
            "distinct_market_count_log1p": math.log1p(item["distinct_market_count"]),
            "public_url_or_domain_flag": item["public_url_or_domain_flag"],
            "identifier_type_telegram_flag": item["identifier_type_telegram_flag"],
            "identifier_type_email_flag": item["identifier_type_email_flag"],
            "identifier_type_financial_phone_flag": item[
                "identifier_type_financial_phone_flag"
            ],
        }
        matrix[index] = [float(values[name]) for name in feature_names]
    return matrix


def fit_offset_reliability_expert(
    matrix: np.ndarray,
    labels: np.ndarray,
    clean_probabilities: np.ndarray,
    cfg: dict,
) -> dict:
    scaled, standardization = step9.fit_standardization(matrix, True)
    offset = step9.safe_logit(np.asarray(clean_probabilities, dtype=float), 1e-6)
    labels = np.asarray(labels, dtype=float)
    params = np.zeros(scaled.shape[1] + 1, dtype=float)
    l2 = float(cfg["base_l2_penalty"])
    converged = False
    final_delta = math.inf
    for iteration in range(1, int(cfg["max_iter"]) + 1):
        logits = offset + params[0] + scaled @ params[1:]
        probabilities = step9.safe_sigmoid(logits)
        residual = probabilities - labels
        gradient = np.empty(len(params), dtype=float)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = scaled.T @ residual + l2 * params[1:]
        curvature = probabilities * (1.0 - probabilities)
        weighted = scaled * curvature[:, None]
        hessian = np.empty((len(params), len(params)), dtype=float)
        hessian[0, 0] = float(np.sum(curvature))
        hessian[0, 1:] = np.sum(weighted, axis=0)
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = scaled.T @ weighted + np.eye(scaled.shape[1]) * l2
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        params -= np.clip(delta, -5.0, 5.0)
        final_delta = float(np.linalg.norm(delta))
        if final_delta <= float(cfg["tolerance"]):
            converged = True
            break
    return {
        "model_family": "step25_occurrence_offset_logistic_l2",
        "feature_names": list(cfg["feature_names"]),
        "standardization": standardization,
        "parameter_intercept": float(params[0]),
        "parameter_coefficients": params[1:].tolist(),
        "base_l2_penalty": l2,
        "solver_iterations": iteration,
        "solver_converged": converged,
        "solver_final_delta_norm": final_delta,
    }


def reliability_corrections(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    scaled = step9.apply_standardization(matrix, artifact["standardization"])
    return float(artifact["parameter_intercept"]) + scaled @ np.asarray(
        artifact["parameter_coefficients"], dtype=float
    )


def apply_direction_constrained_reliability(
    clean_probabilities: np.ndarray,
    evidence: list[dict],
    corrections: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    clean_logits = step9.safe_logit(np.asarray(clean_probabilities, dtype=float), 1e-6)
    applied = np.zeros(len(corrections), dtype=float)
    decisions = []
    for index, (item, raw_delta) in enumerate(zip(evidence, corrections, strict=True)):
        state = item["evidence_state"]
        if state == "verified_direct_both_sides":
            delta = max(0.0, float(raw_delta))
            action = "nonnegative_uplift"
        elif state in {"risky_only_shared", "support_only_shared", "high_frequency_public"}:
            delta = min(0.0, float(raw_delta))
            action = "nonpositive_downgrade"
        else:
            delta = 0.0
            action = "no_score_change"
        applied[index] = delta
        decisions.append(
            {
                "evidence_state": state,
                "expert_action": action,
                "raw_logit_correction": float(raw_delta),
                "applied_logit_correction": delta,
            }
        )
    return step9.safe_sigmoid(clean_logits + applied), decisions
