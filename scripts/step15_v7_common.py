#!/usr/bin/env python3
"""Shared, test-independent utilities for Step9-v7 and Step15-v7."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def supervised_eligible(row: dict) -> bool:
    return (
        row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    )


def load_joined_rows(policy: dict) -> dict[str, list[dict]]:
    assignment_path = resolve(policy["representative_validation"]["split_assignment_output"])
    assignments = {row["pair_uid"]: row for row in load_csv(assignment_path)}
    result = {}
    for pool_name, cfg in policy["pools"].items():
        labels = [row for row in load_csv(resolve(cfg["frozen_labels"])) if supervised_eligible(row)]
        evidence = {row["pair_uid"]: row for row in load_csv(resolve(cfg["evidence_labels"]))}
        features = {row["pair_uid"]: row for row in load_csv(resolve(cfg["v7_pair_features"]))}
        joined = []
        for label in labels:
            pair_uid = label["pair_uid"]
            if pair_uid not in evidence or pair_uid not in features:
                raise ValueError(f"Missing v7 evidence/features for {pool_name}:{pair_uid}")
            if pool_name == "zh_target_strict":
                assignment = assignments.get(pair_uid)
                if assignment is None:
                    raise ValueError(f"Missing v7 split assignment for {pair_uid}")
                split_name = assignment["v7_split_name"]
                component_id = assignment["v7_component_id"]
            else:
                split_name = label["split_name"]
                component_id = label["split_component_id"]
            joined.append(
                {
                    **features[pair_uid],
                    **label,
                    "evidence_type": evidence[pair_uid]["evidence_type"],
                    "evidence_type_confident": evidence[pair_uid]["evidence_type_confident"],
                    "v7_split_name": split_name,
                    "v7_component_id": component_id,
                    "step15_pool": pool_name,
                    "domain": cfg["domain"],
                }
            )
        result[pool_name] = joined
    return result


def load_embedding_index(pool_cfg: dict) -> tuple[dict[str, int], np.ndarray, dict]:
    required = {"clean_e5_cache_metadata", "clean_e5_cache_matrix"}
    if not required.issubset(pool_cfg):
        raise ValueError(
            "Step15-v7 clean ranking requires the identifier-redacted E5 cache; "
            "the legacy profile_text E5 cache is forbidden"
        )
    metadata_path = resolve(pool_cfg["clean_e5_cache_metadata"])
    matrix_path = resolve(pool_cfg["clean_e5_cache_matrix"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = np.load(matrix_path, mmap_mode="r")
    if metadata.get("identifier_redacted") is not True:
        raise ValueError(f"Embedding cache is not identifier-redacted: {metadata_path}")
    seller_uids = list(metadata.get("seller_uids", []))
    if list(matrix.shape) != list(metadata.get("shape", [])):
        raise ValueError(f"Embedding cache shape mismatch: {matrix_path}")
    if len(seller_uids) != matrix.shape[0] or len(set(seller_uids)) != len(seller_uids):
        raise ValueError(f"Embedding cache seller UID contract failed: {metadata_path}")
    return {uid: index for index, uid in enumerate(seller_uids)}, matrix, metadata


def fixed_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Latent projection dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(output_dim),
        size=(input_dim, output_dim),
    ).astype(np.float32)


def projected_pair_latents(rows: list[dict], pool_cfg: dict, latent_cfg: dict) -> np.ndarray:
    seller_index, embeddings, metadata = load_embedding_index(pool_cfg)
    embedding_dim = int(embeddings.shape[1])
    output_dim = int(latent_cfg["projection_dimensions"])
    projection = fixed_projection(2 * embedding_dim, output_dim, int(latent_cfg["projection_seed"]))
    output = np.empty((len(rows), output_dim), dtype=np.float64)
    batch_size = 512
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        left_indices = []
        right_indices = []
        for row in batch:
            left_uid = str(row["seller_uid_left"])
            right_uid = str(row["seller_uid_right"])
            if left_uid not in seller_index or right_uid not in seller_index:
                raise ValueError(
                    f"Pair seller missing from {metadata['model_key']} cache: {row['pair_uid']}"
                )
            left_indices.append(seller_index[left_uid])
            right_indices.append(seller_index[right_uid])
        left = np.asarray(embeddings[left_indices], dtype=np.float32)
        right = np.asarray(embeddings[right_indices], dtype=np.float32)
        symmetric = np.concatenate([np.abs(left - right), left * right], axis=1)
        output[start : start + len(batch)] = np.asarray(symmetric @ projection, dtype=np.float64)
    return output


def strict_clean_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.empty((len(rows), len(feature_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(feature_names):
            value = str(row.get(name, "")).strip()
            matrix[row_index, feature_index] = np.nan if value == "" else float(value)
    return matrix


def fit_train_median_imputation(matrix: np.ndarray) -> dict:
    medians = np.nanmedian(np.asarray(matrix, dtype=float), axis=0)
    if np.any(~np.isfinite(medians)):
        bad = np.flatnonzero(~np.isfinite(medians)).tolist()
        raise ValueError(f"Features entirely missing on train: {bad}")
    return {"strategy": "train_median_per_feature", "values": medians.tolist()}


def apply_imputation(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    values = np.asarray(artifact["values"], dtype=float)
    if matrix.shape[1] != len(values):
        raise ValueError("Imputation dimension mismatch")
    return np.where(np.isfinite(matrix), matrix, values)


def labels_array(rows: list[dict]) -> np.ndarray:
    return np.asarray([1.0 if row["review_label"] == "positive" else 0.0 for row in rows])


def factorized_evidence_weights(rows: list[dict], cfg: dict) -> tuple[np.ndarray, dict]:
    for row in rows:
        if not str(row.get("domain", "")).strip():
            raise ValueError(f"Factorized weighting row lacks domain: {row.get('pair_uid')}")
        if not str(row.get("v7_component_id", "")).strip():
            raise ValueError(
                f"Factorized weighting row lacks seller component: {row.get('pair_uid')}"
            )
        domain = row["domain"]
        evidence_type = row.get("evidence_type")
        if domain not in cfg["domain_factor"] or evidence_type not in cfg[
            "evidence_type_factor"
        ].get(domain, {}):
            raise ValueError(
                f"Factorized weighting has no factor for {domain}:{evidence_type}"
            )
    component_counts = Counter((row["domain"], row["v7_component_id"]) for row in rows)
    raw_component = np.asarray(
        [1.0 / math.sqrt(component_counts[(row["domain"], row["v7_component_id"])]) for row in rows],
        dtype=float,
    )
    component = raw_component.copy()
    component_scales = {}
    strata = sorted({(row["domain"], row["evidence_type"]) for row in rows})
    for domain, evidence_type in strata:
        mask = np.asarray(
            [
                row["domain"] == domain and row["evidence_type"] == evidence_type
                for row in rows
            ]
        )
        mean = float(np.mean(raw_component[mask]))
        scale = 1.0 / mean if mean > 0 else 1.0
        component[mask] *= scale
        component_scales[f"{domain}:{evidence_type}"] = scale
    weights = np.empty(len(rows), dtype=float)
    factor_rows = []
    for index, row in enumerate(rows):
        domain = row["domain"]
        evidence_type = row["evidence_type"]
        domain_factor = float(cfg["domain_factor"][domain])
        evidence_factor = float(cfg["evidence_type_factor"][domain][evidence_type])
        confidence = float(row.get("training_sample_weight") or 1.0)
        confidence = float(
            np.clip(
                confidence,
                float(cfg["confidence_factor"]["minimum"]),
                float(cfg["confidence_factor"]["maximum"]),
            )
        )
        value = domain_factor * evidence_factor * confidence * component[index]
        value = float(np.clip(value, float(cfg["min_weight"]), float(cfg["max_weight"])))
        weights[index] = value
        factor_rows.append((domain, evidence_type, confidence, component[index], value))
    if np.any(weights >= 8.0):
        raise ValueError("Factorized evidence weighting accidentally reproduced a global 8x weight")
    by_domain_evidence = defaultdict(list)
    for domain, evidence_type, confidence, component_factor, final in factor_rows:
        by_domain_evidence[f"{domain}:{evidence_type}"].append(
            (confidence, component_factor, final)
        )
    diagnostics = {
        "formula": cfg["formula"],
        "component_normalization_scales": component_scales,
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
        "max": float(np.max(weights)),
        "by_domain_evidence": {
            key: {
                "count": len(values),
                "mean_confidence_factor": float(np.mean([value[0] for value in values])),
                "mean_component_factor": float(np.mean([value[1] for value in values])),
                "mean_final_weight": float(np.mean([value[2] for value in values])),
            }
            for key, values in sorted(by_domain_evidence.items())
        },
    }
    return weights, diagnostics


def deterministic_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def stratified_support_sample(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    if ratio <= 1e-12:
        return []
    if ratio >= 1.0 - 1e-12:
        return list(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["review_label"], row["evidence_type"])].append(row)
    sampled = []
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: deterministic_rank(row["pair_uid"], seed))
        count = max(1, int(round(len(ordered) * ratio)))
        sampled.extend(ordered[:count])
    sampled.sort(key=lambda row: row["pair_uid"])
    if len({row["review_label"] for row in sampled}) != 2:
        raise ValueError(f"Support sample ratio={ratio} lacks both labels")
    return sampled


def build_mixup_schedule(
    rows: list[dict],
    latents: np.ndarray,
    real_weights: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    if not cfg.get("target_domain_only"):
        raise ValueError("V7 latent mixup must remain target-domain-only")
    if not cfg.get("same_domain_only") or not cfg.get("same_evidence_type_only"):
        raise ValueError("V7 latent mixup requires same-domain, same-evidence parents")
    if cfg.get("gap_reference_scope") != "target_domain_support_only":
        raise ValueError("V7 latent mixup budget must use the target-domain support gap")
    eligible = [
        index
        for index, row in enumerate(rows)
        if row["review_label"] == "positive"
        and (not cfg.get("target_domain_only") or row["domain"] == "zh")
        and float(row.get("training_sample_weight") or 1.0)
        >= float(cfg["minimum_parent_confidence"])
    ]
    groups = defaultdict(list)
    for index in eligible:
        row = rows[index]
        groups[(row["domain"], row["evidence_type"])].append(index)
    eligible = [index for indices in groups.values() if len(indices) >= 2 for index in indices]
    target_mask = np.asarray([row["domain"] == "zh" for row in rows], dtype=bool)
    labels = labels_array(rows)
    positive_weight = float(np.sum(real_weights[target_mask & (labels == 1.0)]))
    negative_weight = float(np.sum(real_weights[target_mask & (labels == 0.0)]))
    full_weight_gap = max(0.0, negative_weight - positive_weight)
    closure_fraction = float(cfg.get("target_effective_weight_gap_closure_fraction", 1.0))
    if closure_fraction <= 0.0 or closure_fraction > 1.0:
        raise ValueError("Latent mixup gap-closure fraction must be in (0, 1]")
    required_weight = full_weight_gap * closure_fraction
    rng = np.random.default_rng(seed)
    candidates = []
    nearest_k = int(cfg["nearest_neighbor_k"])
    max_per_parent = int(cfg.get("max_synthetic_rows_per_eligible_parent", 1))
    if max_per_parent <= 0:
        raise ValueError("Latent mixup max synthetic rows per parent must be positive")
    for round_index in range(max_per_parent):
        ordered_anchors = sorted(
            eligible,
            key=lambda index: deterministic_rank(
                f"{round_index}|{rows[index]['pair_uid']}", seed
            ),
        )
        for anchor in ordered_anchors:
            row = rows[anchor]
            peers = [
                index
                for index in groups[(row["domain"], row["evidence_type"])]
                if index != anchor
            ]
            distances = np.sum((latents[peers] - latents[anchor]) ** 2, axis=1)
            nearest = [
                peers[index] for index in np.argsort(distances, kind="mergesort")[:nearest_k]
            ]
            partner = nearest[int(rng.integers(0, len(nearest)))]
            lam = float(rng.beta(float(cfg["beta_alpha"]), float(cfg["beta_alpha"])))
            weight = float(min(real_weights[anchor], real_weights[partner]))
            candidates.append(
                {
                    "anchor_index": anchor,
                    "partner_index": partner,
                    "lambda_partner": lam,
                    "synthetic_weight": weight,
                    "augmentation_round": round_index,
                }
            )
    schedule = []
    accumulated = 0.0
    for candidate in candidates:
        if accumulated >= required_weight - 1e-12:
            break
        remaining = required_weight - accumulated
        scheduled = dict(candidate)
        scheduled["synthetic_weight"] = min(
            float(candidate["synthetic_weight"]), float(remaining)
        )
        if scheduled["synthetic_weight"] <= 1e-12:
            break
        schedule.append(scheduled)
        accumulated += scheduled["synthetic_weight"]
    budget_satisfied = required_weight <= 1e-12 or abs(accumulated - required_weight) <= 1e-10
    diagnostics = {
        "eligible_parent_count": len(eligible),
        "gap_reference_scope": "target_domain_support_only",
        "target_domain_real_row_count": int(np.sum(target_mask)),
        "candidate_schedule_count": len(candidates),
        "synthetic_row_count": len(schedule),
        "real_positive_effective_weight": positive_weight,
        "real_negative_effective_weight": negative_weight,
        "full_negative_minus_positive_effective_weight_gap": full_weight_gap,
        "target_gap_closure_fraction": closure_fraction,
        "target_additional_positive_weight": required_weight,
        "actual_synthetic_effective_weight": accumulated,
        "schedule_budget_satisfied": budget_satisfied,
        "actual_full_gap_closure_fraction": 0.0
        if full_weight_gap <= 0.0
        else accumulated / full_weight_gap,
        "max_synthetic_rows_per_eligible_parent": max_per_parent,
        "cross_domain_parent_pairs": sum(
            rows[item["anchor_index"]]["domain"] != rows[item["partner_index"]]["domain"]
            for item in schedule
        ),
    }
    if diagnostics["cross_domain_parent_pairs"]:
        raise ValueError("Latent mixup schedule contains cross-domain parents")
    return schedule, diagnostics


def augment_from_schedule(
    clean: np.ndarray,
    latent: np.ndarray,
    rows: list[dict],
    schedule: list[dict],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    synthetic_clean = []
    synthetic_latent = []
    synthetic_weight = []
    manifest = []
    for index, item in enumerate(schedule):
        anchor = int(item["anchor_index"])
        partner = int(item["partner_index"])
        lam = float(item["lambda_partner"])
        synthetic_clean.append(clean[anchor].copy())
        if mode == "latent_pair_embedding_mixup":
            synthetic_latent.append((1.0 - lam) * latent[anchor] + lam * latent[partner])
        elif mode == "equal_effective_weight_duplication":
            synthetic_latent.append(latent[anchor].copy())
        else:
            raise ValueError(f"Unsupported augmentation mode: {mode}")
        synthetic_weight.append(float(item["synthetic_weight"]))
        manifest.append(
            {
                "synthetic_pair_uid": f"synthetic_train_only::{mode}::{index:05d}",
                "synthetic_train_only": "1",
                "mode": mode,
                "anchor_pair_uid": rows[anchor]["pair_uid"],
                "partner_pair_uid": rows[partner]["pair_uid"],
                "domain": rows[anchor]["domain"],
                "evidence_type": rows[anchor]["evidence_type"],
                "lambda_partner": f"{lam:.12f}",
                "augmentation_round": item.get("augmentation_round", 0),
                "training_sample_weight": f"{item['synthetic_weight']:.12f}",
            }
        )
    if not schedule:
        return (
            np.empty((0, clean.shape[1])),
            np.empty((0, latent.shape[1])),
            np.empty(0),
            [],
        )
    return (
        np.asarray(synthetic_clean),
        np.asarray(synthetic_latent),
        np.asarray(synthetic_weight),
        manifest,
    )


def item_signal_index_many(
    paths: list[Path], frequency_reference_sellers: set[str] | None = None
) -> tuple[dict, Counter[tuple[str, str]]]:
    by_seller = defaultdict(lambda: defaultdict(list))
    sellers_by_token = defaultdict(set)
    for path in paths:
        for row in load_csv(path):
            token = (row["contact_type"].strip().lower(), row["normalized_value"].strip().lower())
            if not token[0] or not token[1]:
                continue
            seller = row["seller_uid"]
            by_seller[seller][token].append(row)
            if frequency_reference_sellers is None or seller in frequency_reference_sellers:
                sellers_by_token[token].add(seller)
    return by_seller, Counter({token: len(sellers) for token, sellers in sellers_by_token.items()})


def item_signal_index(path: Path) -> tuple[dict, Counter[tuple[str, str]]]:
    return item_signal_index_many([path])


def relation_reliability(row: dict, by_seller: dict, token_df: Counter, cfg: dict) -> dict:
    left_uid = row["seller_uid_left"]
    right_uid = row["seller_uid_right"]
    left = by_seller.get(left_uid, {})
    right = by_seller.get(right_uid, {})
    shared = sorted(set(left) & set(right))
    strong_direct = []
    public_noise = []
    ambiguous = []
    for token in shared:
        left_occ = left[token]
        right_occ = right[token]

        def direct(occurrences: list[dict]) -> bool:
            return any(
                item.get("direct_identity_eligible") == "1"
                and item.get("seller_facing_context") == "1"
                and item.get("product_data_risk_context") != "1"
                and item.get("support_only") != "1"
                for item in occurrences
            )

        risky = any(
            item.get("product_data_risk_context") == "1" or item.get("support_only") == "1"
            for item in left_occ + right_occ
        )
        high_frequency = token_df[token] > int(
            cfg.get("public_identifier_seller_frequency_threshold", 3)
        )
        if direct(left_occ) and direct(right_occ) and not risky and not high_frequency:
            strong_direct.append(token)
        elif risky or high_frequency:
            public_noise.append(token)
        else:
            ambiguous.append(token)
    if strong_direct:
        decision = "verified_seller_facing_direct"
        multiplier = float(cfg["direct_identifier_score_multiplier"])
    elif public_noise:
        decision = "public_or_product_contact_veto"
        multiplier = float(cfg["public_noise_score_multiplier"])
    elif shared or str(row.get("has_shared_contact_exact", "")) == "1":
        decision = "ambiguous_identifier_no_score_change"
        multiplier = 1.0
    else:
        decision = "no_identifier_no_score_change"
        multiplier = float(cfg["no_identifier_score_multiplier"])
    return {
        "decision": decision,
        "score_multiplier": multiplier,
        "strong_direct_token_count": len(strong_direct),
        "public_noise_token_count": len(public_noise),
        "ambiguous_token_count": len(ambiguous),
        "shared_token_count": len(shared),
        "strong_direct_token_hashes": [canonical_hash(token)[:16] for token in strong_direct],
        "public_noise_token_hashes": [canonical_hash(token)[:16] for token in public_noise],
    }


def apply_reliability_veto(
    rows: list[dict], probabilities: np.ndarray, pool_cfg: dict, cfg: dict
) -> tuple[np.ndarray, list[dict], dict]:
    configured = pool_cfg.get("item_identity_signal_sources")
    paths = (
        [resolve(value) for value in configured]
        if configured
        else [resolve(pool_cfg["item_identity_signals"])]
    )
    reference_sellers = pool_cfg.get("identifier_frequency_reference_sellers")
    by_seller, token_df = item_signal_index_many(
        paths,
        None if reference_sellers is None else set(reference_sellers),
    )
    decisions = [relation_reliability(row, by_seller, token_df, cfg) for row in rows]
    multipliers = np.asarray([item["score_multiplier"] for item in decisions], dtype=float)
    adjusted = np.clip(np.asarray(probabilities, dtype=float) * multipliers, 0.0, 1.0)
    counts = Counter(item["decision"] for item in decisions)
    return adjusted, decisions, {"decision_counts": dict(sorted(counts.items()))}
