from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import step7_train_baseline_models as step7


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "schema" / "step15_evidence_type_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Step 15 evidence-type incremental hard-negative scorers. "
            "The script keeps Step 5 labels frozen and evaluates on fixed zh_valid/zh_test."
        )
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH), help="Path to Step 15 policy JSON.")
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment name from policy. Defaults to policy default_experiments.",
    )
    parser.add_argument(
        "--allow-legacy-output-overwrite",
        action="store_true",
        help=(
            "Allow explicit runs of legacy step15_e5_* experiments. Those names write the original "
            "first-pass artifact/prediction paths and can overwrite old Step 15 outputs."
        ),
    )
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="Phase id from policy curriculum_phases. Defaults to all phases.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="Training seed. Repeat for multiple seeds. Defaults to policy training.default_seeds.",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help=(
            "Validate the selected experiment/phase/seed contract and output isolation against the policy, "
            "then exit before loading data or training."
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train and write validation artifacts only. No zh_test row is loaded or scored, "
            "including validation-selected endpoints."
        ),
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def label_to_int(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return 0
    raise ValueError(f"Unsupported binary label for Step15 identity training: {label}")


def output_path(template: str, experiment_name: str, phase_id: str, seed: int) -> Path:
    return ROOT / template.format(experiment_name=experiment_name, phase_id=phase_id, seed=seed)


def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    step7.write_csv(temporary, rows, fieldnames)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    step7.write_json(temporary, payload)
    temporary.replace(path)


def load_pool(pool_name: str, pool_cfg: dict) -> list[dict]:
    frozen_rows = step7.load_csv(resolve_path(pool_cfg["frozen_labels"]))
    feature_rows = step7.load_csv(resolve_path(pool_cfg["pair_features"]))
    label_rows = step7.load_csv(resolve_path(pool_cfg["label_output"]))
    joined_rows = step7.join_frozen_with_features(frozen_rows, feature_rows)
    label_index = {row["pair_uid"]: row for row in label_rows}
    missing = [row["pair_uid"] for row in joined_rows if row["pair_uid"] not in label_index]
    if missing:
        raise ValueError(f"{pool_name} is missing {len(missing)} Step15 evidence labels; first={missing[0]}")
    merged = []
    for row in joined_rows:
        label_row = label_index[row["pair_uid"]]
        merged_row = dict(row)
        merged_row["step15_pool"] = pool_name
        for key, value in label_row.items():
            if key in {
                "identity_label",
                "evidence_type",
                "evidence_type_confident",
                "identity_training_eligible",
                "has_direct_identifier_signal",
                "has_template_clone_signal",
                "has_semantic_topic_signal",
                "has_public_contact_or_url_noise_signal",
                "evidence_type_reasons",
            }:
                merged_row[key] = value
        merged.append(merged_row)
    return merged


def validate_features(rows_by_pool: dict[str, list[dict]], feature_names: list[str]) -> None:
    for pool_name, rows in rows_by_pool.items():
        if not rows:
            raise ValueError(f"No rows loaded for {pool_name}")
        missing = [feature for feature in feature_names if feature not in rows[0]]
        if missing:
            raise ValueError(f"{pool_name} is missing Step15 features: {missing}")


def rows_to_feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    x = np.empty((len(rows), len(feature_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, feature_name in enumerate(feature_names):
            x[i, j] = as_float(row, feature_name)
    return x


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(x_train), x_train, means)
    stds = filled.std(axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)
    return means, stds


def apply_standardizer(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, means)
    return (filled - means) / stds


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def fit_standardizer_bundle(
    rows: list[dict],
    feature_names: list[str],
    preprocessing_cfg: dict | None = None,
) -> dict:
    preprocessing_cfg = preprocessing_cfg or {}
    raw = rows_to_feature_matrix(rows, feature_names)
    transformed = raw.copy()
    domain_feature_names = [
        str(name) for name in preprocessing_cfg.get("domain_standardized_features", [])
    ]
    unknown_features = [name for name in domain_feature_names if name not in feature_names]
    if unknown_features:
        raise ValueError(f"Domain-standardized features are absent from the feature set: {unknown_features}")
    allowed_domains = [str(name) for name in preprocessing_cfg.get("allowed_domains", [])]
    domain_stats: dict[str, dict] = {}
    if domain_feature_names:
        if not allowed_domains:
            raise ValueError("Domain-standardized features require explicit allowed_domains")
        observed_domains = [str(row.get("step15_pool", "")) for row in rows]
        unknown_domains = sorted(set(observed_domains) - set(allowed_domains))
        if unknown_domains:
            raise ValueError(f"Unknown domains while fitting Step15 scaler: {unknown_domains}")
        feature_indices = [feature_names.index(name) for name in domain_feature_names]
        for domain in allowed_domains:
            row_indices = [idx for idx, value in enumerate(observed_domains) if value == domain]
            if not row_indices:
                raise ValueError(f"No train rows available for configured scaler domain: {domain}")
            values = raw[np.asarray(row_indices, dtype=int)][:, feature_indices]
            means, stds = fit_standardizer(values)
            transformed[np.ix_(np.asarray(row_indices, dtype=int), np.asarray(feature_indices, dtype=int))] = apply_standardizer(
                values,
                means,
                stds,
            )
            domain_stats[domain] = {
                "means": [float(value) for value in means],
                "stds": [float(value) for value in stds],
                "row_count": len(row_indices),
            }
    global_means, global_stds = fit_standardizer(transformed)
    serializable = {
        "version": "step15-v6-train-only-domain-aware-v1",
        "feature_names": feature_names,
        "domain_standardized_features": domain_feature_names,
        "allowed_domains": allowed_domains,
        "domain_stats": domain_stats,
        "global_means": [float(value) for value in global_means],
        "global_stds": [float(value) for value in global_stds],
        "fit_row_count": len(rows),
    }
    serializable["sha256"] = canonical_json_sha256(serializable)
    return serializable


def apply_standardizer_bundle(rows: list[dict], feature_names: list[str], bundle: dict) -> np.ndarray:
    if list(bundle.get("feature_names", [])) != list(feature_names):
        raise ValueError("Step15 standardizer feature lineage mismatch")
    raw = rows_to_feature_matrix(rows, feature_names)
    transformed = raw.copy()
    domain_feature_names = list(bundle.get("domain_standardized_features", []))
    if domain_feature_names:
        feature_indices = [feature_names.index(name) for name in domain_feature_names]
        domain_stats = bundle.get("domain_stats", {})
        for idx, row in enumerate(rows):
            domain = str(row.get("step15_pool", ""))
            if domain not in domain_stats:
                raise ValueError(f"Unknown domain while applying Step15 scaler: {domain or '<empty>'}")
            stats = domain_stats[domain]
            means = np.asarray(stats["means"], dtype=float)
            stds = np.asarray(stats["stds"], dtype=float)
            values = raw[idx, feature_indices][None, :]
            transformed[idx, feature_indices] = apply_standardizer(values, means, stds)[0]
    return apply_standardizer(
        transformed,
        np.asarray(bundle["global_means"], dtype=float),
        np.asarray(bundle["global_stds"], dtype=float),
    )


def select_train_rows(
    rows_by_pool: dict[str, list[dict]],
    experiment_cfg: dict,
    phase_cfg: dict,
    policy: dict,
) -> list[dict]:
    included_evidence = set(phase_cfg["included_evidence_types"])
    train_rows: list[dict] = []
    splits = policy["splits"]
    if bool(experiment_cfg.get("include_source_train", False)):
        train_rows.extend(
            row
            for row in rows_by_pool["en_content_train_pool"]
            if row.get("split_name") == splits["source_train"]
            and row.get("identity_training_eligible") == "1"
            and row.get("evidence_type") in included_evidence
        )
    if bool(experiment_cfg.get("include_target_train", True)):
        train_rows.extend(
            row
            for row in rows_by_pool["zh_target_strict"]
            if row.get("split_name") == splits["target_train"]
            and row.get("identity_training_eligible") == "1"
            and row.get("evidence_type") in included_evidence
        )
    label_scope = str(experiment_cfg.get("training_label_scope", "gold_plus_all_silver"))
    if label_scope == "gold_plus_all_silver":
        return train_rows
    if label_scope == "gold_only":
        return [row for row in train_rows if str(row.get("silver_train_only", "")) != "1"]
    if label_scope == "gold_plus_high_confidence_silver":
        allowed_tiers = {
            str(value) for value in experiment_cfg.get("high_confidence_silver_label_tiers", [])
        }
        if not allowed_tiers:
            raise ValueError(
                "gold_plus_high_confidence_silver requires high_confidence_silver_label_tiers"
            )
        return [
            row
            for row in train_rows
            if str(row.get("silver_train_only", "")) != "1"
            or str(row.get("label_tier", "")) in allowed_tiers
        ]
    raise ValueError(f"Unsupported Step15 training_label_scope: {label_scope}")


def select_eval_rows(rows: list[dict], split_name: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("split_name") == split_name
        and row.get("review_label") in {"positive", "negative"}
        and row.get("usable_for_supervision") == "1"
        and row.get("usable_for_core_transfer") == "1"
    ]


def y_from_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([label_to_int(row["review_label"]) for row in rows], dtype=float)


def evidence_indices(rows: list[dict], evidence_types: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(evidence_types)}
    values = []
    for row in rows:
        if row.get("evidence_type_confident") != "1":
            values.append(-1)
            continue
        values.append(index.get(row.get("evidence_type", ""), -1))
    return np.asarray(values, dtype=int)


def balanced_binary_weights(y: np.ndarray) -> np.ndarray:
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    total = max(len(y), 1)
    pos_weight = total / (2.0 * positives)
    neg_weight = total / (2.0 * negatives)
    return np.where(y == 1.0, pos_weight, neg_weight).astype(float)


def legacy_domain_balanced_binary_weights(y: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, dict]:
    """Preserve the pre-v5r raw-row-count behavior for explicit legacy reruns."""
    weights = balanced_binary_weights(y)
    domain_counts = Counter(str(row.get("step15_pool", "")) for row in rows)
    present_domains = [domain for domain, count in domain_counts.items() if count > 0]
    if not present_domains:
        return weights, {
            "enabled": True,
            "method": "legacy_raw_row_count_before_quality_weights",
            "row_counts": {},
            "domain_factors": {},
        }
    total = max(len(rows), 1)
    domain_factor = {
        domain: total / (len(present_domains) * max(domain_counts[domain], 1))
        for domain in present_domains
    }
    adjusted = weights * np.asarray(
        [domain_factor.get(str(row.get("step15_pool", "")), 1.0) for row in rows],
        dtype=float,
    )
    return adjusted, {
        "enabled": True,
        "method": "legacy_raw_row_count_before_quality_weights",
        "row_counts": dict(sorted(domain_counts.items())),
        "domain_factors": {key: round(value, 6) for key, value in sorted(domain_factor.items())},
    }


def apply_effective_domain_balance(
    weights: np.ndarray,
    rows: list[dict],
    allowed_domains: list[str],
) -> tuple[np.ndarray, dict]:
    """Equalize effective domain mass after all other sample weights are applied."""
    domains = [str(value) for value in allowed_domains]
    if len(domains) < 2 or len(set(domains)) != len(domains):
        raise ValueError("Step15 effective domain balancing requires at least two unique allowed domains")
    if len(weights) != len(rows):
        raise ValueError("Step15 effective domain-balance row/weight length mismatch")

    observed = [str(row.get("step15_pool", "")) for row in rows]
    unknown_counts = Counter(domain for domain in observed if domain not in domains)
    if unknown_counts:
        raise ValueError(
            "Step15 effective domain balancing encountered non-real or unknown domains: "
            + ", ".join(f"{domain or '<empty>'}={count}" for domain, count in sorted(unknown_counts.items()))
        )

    before_mass = {
        domain: float(np.sum(weights[np.asarray([value == domain for value in observed], dtype=bool)]))
        for domain in domains
    }
    empty_domains = [domain for domain, mass in before_mass.items() if mass <= 0.0]
    if empty_domains:
        raise ValueError(f"Step15 effective domain balancing has zero weighted mass for: {empty_domains}")

    total_mass = float(sum(before_mass.values()))
    target_mass = total_mass / float(len(domains))
    factors = {domain: target_mass / before_mass[domain] for domain in domains}
    adjusted = weights.astype(float, copy=True) * np.asarray([factors[domain] for domain in observed], dtype=float)

    mean_before = float(np.mean(weights)) if len(weights) else 0.0
    mean_after_raw = float(np.mean(adjusted)) if len(adjusted) else 0.0
    if mean_before > 0.0 and mean_after_raw > 0.0:
        adjusted *= mean_before / mean_after_raw

    after_mass = {
        domain: float(np.sum(adjusted[np.asarray([value == domain for value in observed], dtype=bool)]))
        for domain in domains
    }
    return adjusted, {
        "enabled": True,
        "method": "post_quality_effective_weight_mass",
        "allowed_domains": domains,
        "row_counts": dict(sorted(Counter(observed).items())),
        "mass_before": {key: round(value, 6) for key, value in before_mass.items()},
        "domain_factors": {key: round(value, 6) for key, value in factors.items()},
        "mass_after": {key: round(value, 6) for key, value in after_mass.items()},
        "mean_before": round(mean_before, 6),
        "mean_after": round(float(np.mean(adjusted)), 6) if len(adjusted) else 0.0,
    }


def apply_identity_weight_multipliers(
    weights: np.ndarray,
    rows: list[dict],
    multipliers: dict[str, float],
    *,
    normalize_mean: bool = True,
) -> tuple[np.ndarray, dict]:
    if not multipliers:
        return weights, {
            "enabled": False,
            "multipliers": {},
            "applied_counts": {},
            "mean_before": round(float(np.mean(weights)), 6) if len(weights) else 0.0,
            "mean_after": round(float(np.mean(weights)), 6) if len(weights) else 0.0,
        }

    adjusted = weights.astype(float, copy=True)
    applied_counts: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        evidence_type = str(row.get("evidence_type", ""))
        factor = float(multipliers.get(evidence_type, 1.0))
        if factor <= 0.0:
            raise ValueError(f"Identity weight multiplier for {evidence_type} must be positive, got {factor}")
        adjusted[idx] *= factor
        if factor != 1.0:
            applied_counts[evidence_type] += 1

    mean_before = float(np.mean(weights)) if len(weights) else 0.0
    mean_after_raw = float(np.mean(adjusted)) if len(adjusted) else 0.0
    if normalize_mean and mean_after_raw > 0.0:
        adjusted *= mean_before / mean_after_raw

    return adjusted, {
        "enabled": True,
        "multipliers": {key: round(float(value), 6) for key, value in sorted(multipliers.items())},
        "applied_counts": dict(sorted(applied_counts.items())),
        "normalize_mean": bool(normalize_mean),
        "mean_before": round(mean_before, 6),
        "mean_after_raw": round(mean_after_raw, 6),
        "mean_after": round(float(np.mean(adjusted)), 6) if len(adjusted) else 0.0,
    }


def apply_row_training_sample_weights(weights: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, dict]:
    multipliers = np.asarray([step7.row_training_sample_weight(row) for row in rows], dtype=float)
    if len(multipliers) != len(weights):
        raise ValueError("Step15 row sample-weight multiplier length mismatch")
    adjusted = weights.astype(float, copy=True) * multipliers
    before_mean = float(np.mean(weights)) if len(weights) else 0.0
    after_mean = float(np.mean(adjusted)) if len(adjusted) else 0.0
    if before_mean > 0.0 and after_mean > 0.0:
        adjusted *= before_mean / after_mean
    return adjusted, {
        "enabled": bool(np.any(np.abs(multipliers - 1.0) > 1e-12)),
        "min_multiplier": round(float(np.min(multipliers)), 6) if len(multipliers) else 1.0,
        "mean_multiplier": round(float(np.mean(multipliers)), 6) if len(multipliers) else 1.0,
        "max_multiplier": round(float(np.max(multipliers)), 6) if len(multipliers) else 1.0,
    }


def apply_component_inverse_sqrt_weights(weights: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, dict]:
    """Downweight repeated train edges from the same seller component."""
    real_rows = [row for row in rows if str(row.get("synthetic_train_only", "")) != "1"]
    component_keys = [
        (str(row.get("step15_pool", "")), str(row.get("split_component_id", "")))
        for row in real_rows
    ]
    if any(not pool or not component for pool, component in component_keys):
        missing = sum(1 for pool, component in component_keys if not pool or not component)
        raise ValueError(f"Step15 component weighting found {missing} real rows without component lineage")
    counts = Counter(component_keys)
    pair_uid_to_factor = {
        (str(row.get("step15_pool", "")), str(row.get("pair_uid", ""))): 1.0 / math.sqrt(counts[key])
        for row, key in zip(real_rows, component_keys, strict=True)
    }
    factors: list[float] = []
    for row in rows:
        if str(row.get("synthetic_train_only", "")) != "1":
            factors.append(
                pair_uid_to_factor[(str(row.get("step15_pool", "")), str(row.get("pair_uid", "")))]
            )
            continue
        parent_keys = (
            (
                str(row.get("mixup_parent_left_pool", "")),
                str(row.get("mixup_parent_left_pair_uid", "")),
            ),
            (
                str(row.get("mixup_parent_right_pool", "")),
                str(row.get("mixup_parent_right_pair_uid", "")),
            ),
        )
        parent_factors = [pair_uid_to_factor[key] for key in parent_keys if key in pair_uid_to_factor]
        factors.append(min(parent_factors) if parent_factors else 1.0)
    factors_array = np.asarray(factors, dtype=float)
    if len(factors_array) != len(weights):
        raise ValueError("Step15 component-weight row/weight length mismatch")
    mean_factor = float(np.mean(factors_array)) if len(factors_array) else 1.0
    if mean_factor <= 0.0:
        raise ValueError("Step15 component weights have non-positive mean")
    factors_array /= mean_factor
    adjusted = weights.astype(float, copy=True) * factors_array
    return adjusted, {
        "enabled": True,
        "method": "inverse_sqrt_train_edge_count_normalized_mean_one",
        "real_component_count": len(counts),
        "largest_component_edge_count": max(counts.values(), default=0),
        "factor_min": round(float(np.min(factors_array)), 6) if len(factors_array) else 1.0,
        "factor_mean": round(float(np.mean(factors_array)), 6) if len(factors_array) else 1.0,
        "factor_max": round(float(np.max(factors_array)), 6) if len(factors_array) else 1.0,
    }


def apply_class_balance_multipliers(
    weights: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict]:
    base_weights = weights.astype(float, copy=True)
    positive_mask = y == 1.0
    negative_mask = y == 0.0
    positive_mass = float(base_weights[positive_mask].sum())
    negative_mass = float(base_weights[negative_mask].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError(
            "Step15 class balancing requires positive effective weight mass and negative effective weight mass"
        )
    target_mass = (positive_mass + negative_mass) / 2.0
    positive_factor = target_mass / positive_mass
    negative_factor = target_mass / negative_mass
    factors = np.where(positive_mask, positive_factor, negative_factor)
    adjusted = base_weights * factors
    before_mean = float(np.mean(weights)) if len(weights) else 0.0
    raw_mean = float(np.mean(adjusted)) if len(adjusted) else 0.0
    if before_mean > 0.0 and raw_mean > 0.0:
        adjusted *= before_mean / raw_mean
    positive_mass_after = float(adjusted[positive_mask].sum())
    negative_mass_after = float(adjusted[negative_mask].sum())
    return adjusted, {
        "enabled": True,
        "method": "equalize_effective_weight_mass_after_component_and_row_quality_weights",
        "positive_factor": round(float(positive_factor), 6),
        "negative_factor": round(float(negative_factor), 6),
        "positive_mass_before": round(positive_mass, 6),
        "negative_mass_before": round(negative_mass, 6),
        "positive_mass_after": round(positive_mass_after, 6),
        "negative_mass_after": round(negative_mass_after, 6),
        "mean_before": round(before_mean, 6),
        "mean_after": round(float(np.mean(adjusted)), 6) if len(adjusted) else 0.0,
    }


def evidence_weights(
    y_evidence: np.ndarray,
    evidence_type_count: int,
    base_effective_weights: np.ndarray | None = None,
    *,
    class_balance: bool = True,
) -> tuple[np.ndarray, dict]:
    weights = np.zeros(len(y_evidence), dtype=float)
    mask = y_evidence >= 0
    if not np.any(mask):
        return weights, {"enabled": False, "labeled_row_count": 0, "class_factors": {}}
    base = (
        np.ones(len(y_evidence), dtype=float)
        if base_effective_weights is None
        else np.asarray(base_effective_weights, dtype=float).copy()
    )
    if len(base) != len(y_evidence):
        raise ValueError("Step15 auxiliary evidence/base effective weight length mismatch")
    weights[mask] = base[mask]
    present_classes = sorted({int(value) for value in y_evidence[mask]})
    unknown_classes = [value for value in present_classes if value >= evidence_type_count]
    if unknown_classes:
        raise ValueError(f"Step15 auxiliary evidence labels exceed configured classes: {unknown_classes}")
    factors: dict[int, float] = {value: 1.0 for value in present_classes}
    mass_before = {
        value: float(weights[y_evidence == value].sum()) for value in present_classes
    }
    if class_balance:
        if any(mass <= 0.0 for mass in mass_before.values()):
            raise ValueError("Step15 auxiliary evidence class has non-positive effective mass")
        target_mass = sum(mass_before.values()) / max(len(present_classes), 1)
        factors = {value: target_mass / mass_before[value] for value in present_classes}
        for value, factor in factors.items():
            weights[y_evidence == value] *= factor
    return weights, {
        "enabled": True,
        "method": "identity_effective_weight_chain_then_evidence_class_balance",
        "class_balance": bool(class_balance),
        "labeled_row_count": int(mask.sum()),
        "present_class_count": len(present_classes),
        "class_mass_before": {str(key): round(value, 6) for key, value in mass_before.items()},
        "class_factors": {str(key): round(value, 6) for key, value in factors.items()},
        "class_mass_after": {
            str(value): round(float(weights[y_evidence == value].sum()), 6)
            for value in present_classes
        },
    }


def add_positive_mixup(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_evidence: np.ndarray,
    rows: list[dict],
    cfg: dict,
    rng: np.random.Generator,
    feature_names: list[str],
    scope_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int, dict]:
    scope = str(scope_override or cfg.get("scope", "all_positive"))
    all_pos_indices = np.where(y_train == 1.0)[0]
    rejection_counts: Counter[str] = Counter()
    pos_indices_list: list[int] = []
    minimum_source_weight = float(cfg.get("minimum_source_training_sample_weight", 0.0) or 0.0)
    require_core_transfer = bool(cfg.get("require_usable_for_core_transfer", False))
    require_core_eligible = bool(cfg.get("require_core_transfer_eligible", False))
    require_confident_evidence = bool(cfg.get("require_evidence_type_confident", False))
    for raw_idx in all_pos_indices:
        idx = int(raw_idx)
        row = rows[idx]
        if require_core_transfer and str(row.get("usable_for_core_transfer", "")).strip().lower() not in {
            "1",
            "true",
            "yes",
            "y",
        }:
            rejection_counts["not_usable_for_core_transfer"] += 1
            continue
        if require_core_eligible and str(row.get("core_transfer_eligible", "")).strip().lower() not in {
            "1",
            "true",
            "yes",
            "y",
        }:
            rejection_counts["not_core_transfer_eligible"] += 1
            continue
        if require_confident_evidence and str(row.get("evidence_type_confident", "")) != "1":
            rejection_counts["evidence_type_not_confident"] += 1
            continue
        if step7.row_training_sample_weight(row) + 1e-12 < minimum_source_weight:
            rejection_counts["below_minimum_source_weight"] += 1
            continue
        pos_indices_list.append(idx)
    pos_indices = np.asarray(pos_indices_list, dtype=int)
    if scope == "target_train_only":
        pos_indices = np.asarray(
            [idx for idx in pos_indices if rows[int(idx)].get("step15_pool") == "zh_target_strict"],
            dtype=int,
        )
    elif scope == "source_train_only":
        pos_indices = np.asarray(
            [idx for idx in pos_indices if rows[int(idx)].get("step15_pool") == "en_content_train_pool"],
            dtype=int,
        )
    elif scope not in {"all_positive", "same_evidence_type_only", "same_domain_same_evidence_type"}:
        raise ValueError(f"Unsupported Step15 positive_mixup scope: {scope}")

    diagnostics = {
        "enabled": True,
        "scope": scope,
        "all_positive_source_count": int(len(all_pos_indices)),
        "eligible_positive_source_count": int(len(pos_indices)),
        "minimum_source_training_sample_weight": round(minimum_source_weight, 6),
        "require_usable_for_core_transfer": require_core_transfer,
        "require_core_transfer_eligible": require_core_eligible,
        "require_evidence_type_confident": require_confident_evidence,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "synthetic_train_only": True,
        "synthetic_row_count": 0,
        "skipped_reason": None,
    }
    if len(pos_indices) < 2:
        diagnostics["skipped_reason"] = "fewer_than_two_eligible_positive_sources"
        return x_train, y_train, y_evidence, rows, 0, diagnostics
    multiplier = float(cfg.get("multiplier", 0.0))
    synthetic_count = int(math.ceil(len(pos_indices) * multiplier))
    if synthetic_count <= 0:
        diagnostics["skipped_reason"] = "non_positive_multiplier_or_synthetic_count"
        return x_train, y_train, y_evidence, rows, 0, diagnostics
    alpha = float(cfg.get("beta_alpha", 0.4))
    if alpha <= 0.0:
        raise ValueError("Step15 positive_mixup beta_alpha must be positive")

    grouped_indices: dict[tuple[str, ...], list[int]] = defaultdict(list)
    if scope == "same_domain_same_evidence_type":
        for idx in pos_indices:
            row = rows[int(idx)]
            grouped_indices[(str(row.get("step15_pool", "")), str(row.get("evidence_type", "")))].append(int(idx))
    elif scope == "same_evidence_type_only":
        for idx in pos_indices:
            if int(y_evidence[int(idx)]) >= 0:
                grouped_indices[(str(int(y_evidence[int(idx)])),)].append(int(idx))
    else:
        grouped_indices[(scope,)] = [int(value) for value in pos_indices]

    eligible_groups = {key: values for key, values in grouped_indices.items() if len(values) >= 2}
    eligible_anchor_indices = [idx for values in eligible_groups.values() for idx in values]
    diagnostics["eligible_group_counts"] = {
        "||".join(key): len(values) for key, values in sorted(eligible_groups.items())
    }
    diagnostics["eligible_anchor_count"] = len(eligible_anchor_indices)
    if not eligible_anchor_indices:
        diagnostics["skipped_reason"] = "no_group_with_two_eligible_positive_sources"
        return x_train, y_train, y_evidence, rows, 0, diagnostics

    synthetic_count = int(math.ceil(len(eligible_anchor_indices) * multiplier))
    if synthetic_count <= 0:
        diagnostics["skipped_reason"] = "non_positive_grouped_synthetic_count"
        return x_train, y_train, y_evidence, rows, 0, diagnostics

    nearest_neighbor_k = int(cfg.get("nearest_neighbor_k", 0) or 0)
    neighbor_map: dict[int, list[int]] = {}
    for group_indices in eligible_groups.values():
        group_array = np.asarray(group_indices, dtype=int)
        if nearest_neighbor_k <= 0:
            for idx in group_indices:
                neighbor_map[idx] = [candidate for candidate in group_indices if candidate != idx]
            continue
        group_x = x_train[group_array]
        limit = max(1, min(nearest_neighbor_k, len(group_indices) - 1))
        for local_idx, global_idx in enumerate(group_indices):
            distances = np.sum((group_x - group_x[local_idx]) ** 2, axis=1)
            ranked_local = [int(value) for value in np.argsort(distances) if int(value) != local_idx]
            neighbor_map[global_idx] = [group_indices[value] for value in ranked_local[:limit]]

    copy_anchor_feature_names = [
        str(name) for name in cfg.get("copy_anchor_feature_names", []) if str(name) in feature_names
    ]
    copy_anchor_indices = {feature_names.index(name) for name in copy_anchor_feature_names}
    interpolate_indices = [idx for idx in range(len(feature_names)) if idx not in copy_anchor_indices]
    synthetic_weight_mode = str(cfg.get("synthetic_weight_mode", "uniform"))
    if synthetic_weight_mode not in {"uniform", "minimum_parent_weight"}:
        raise ValueError(f"Unsupported Step15 positive_mixup synthetic_weight_mode: {synthetic_weight_mode}")

    synthetic_x: list[np.ndarray] = []
    synthetic_rows: list[dict] = []
    cross_domain_parent_count = 0
    cross_evidence_parent_count = 0
    weak_parent_count = 0
    for _ in range(synthetic_count):
        left = int(rng.choice(np.asarray(eligible_anchor_indices, dtype=int)))
        candidates = neighbor_map[left]
        right = int(rng.choice(np.asarray(candidates, dtype=int)))
        lam = float(rng.beta(alpha, alpha))
        synthetic_vector = x_train[left].copy()
        if interpolate_indices:
            synthetic_vector[interpolate_indices] = (
                (1.0 - lam) * x_train[left, interpolate_indices]
                + lam * x_train[right, interpolate_indices]
            )
        synthetic_x.append(synthetic_vector)
        left_row = rows[int(left)]
        right_row = rows[int(right)]
        left_pool = str(left_row.get("step15_pool", ""))
        right_pool = str(right_row.get("step15_pool", ""))
        synthetic_pool = left_pool if left_pool == right_pool else "cross_domain_mixup"
        left_evidence = str(left_row.get("evidence_type", ""))
        right_evidence = str(right_row.get("evidence_type", ""))
        left_weight = float(step7.row_training_sample_weight(left_row))
        right_weight = float(step7.row_training_sample_weight(right_row))
        synthetic_weight = min(left_weight, right_weight) if synthetic_weight_mode == "minimum_parent_weight" else 1.0
        cross_domain_parent_count += int(left_pool != right_pool)
        cross_evidence_parent_count += int(left_evidence != right_evidence)
        weak_parent_count += int(min(left_weight, right_weight) < 1.0 - 1e-12)
        synthetic_rows.append(
            {
                "pair_uid": f"synthetic_positive_mixup::{len(synthetic_rows)}",
                "review_label": "positive",
                "identity_label": "same_controller",
                "evidence_type": "synthetic_train_only",
                "evidence_type_confident": "0",
                "identity_training_eligible": "1",
                "step15_pool": synthetic_pool,
                "split_name": "train",
                "synthetic_train_only": "1",
                "usable_for_supervision": "1",
                "usable_for_core_transfer": "1",
                "core_transfer_eligible": "1",
                "training_sample_weight": round(float(synthetic_weight), 6),
                "mixup_parent_left_pair_uid": str(left_row.get("pair_uid", "")),
                "mixup_parent_right_pair_uid": str(right_row.get("pair_uid", "")),
                "mixup_parent_left_pool": left_pool,
                "mixup_parent_right_pool": right_pool,
                "mixup_parent_left_evidence_type": left_evidence,
                "mixup_parent_right_evidence_type": right_evidence,
                "mixup_parent_left_training_sample_weight": round(left_weight, 6),
                "mixup_parent_right_training_sample_weight": round(right_weight, 6),
                "mixup_lambda_right": round(lam, 8),
            }
        )
    x_aug = np.vstack([x_train, np.asarray(synthetic_x, dtype=float)])
    y_aug = np.concatenate([y_train, np.ones(synthetic_count, dtype=float)])
    evidence_aug = np.concatenate([y_evidence, np.full(synthetic_count, -1, dtype=int)])
    diagnostics.update(
        {
            "synthetic_row_count": synthetic_count,
            "nearest_neighbor_k": nearest_neighbor_k,
            "synthetic_weight_mode": synthetic_weight_mode,
            "copy_anchor_feature_names": copy_anchor_feature_names,
            "interpolated_feature_count": len(interpolate_indices),
            "cross_domain_parent_count": cross_domain_parent_count,
            "cross_evidence_type_parent_count": cross_evidence_parent_count,
            "synthetic_rows_with_any_subunit_parent_weight": weak_parent_count,
            "synthetic_weight_min": round(
                min(float(row["training_sample_weight"]) for row in synthetic_rows), 6
            ),
            "synthetic_weight_mean": round(
                float(np.mean([float(row["training_sample_weight"]) for row in synthetic_rows])), 6
            ),
            "synthetic_weight_max": round(
                max(float(row["training_sample_weight"]) for row in synthetic_rows), 6
            ),
        }
    )
    return x_aug, y_aug, evidence_aug, rows + synthetic_rows, synthetic_count, diagnostics


def add_negative_mixup(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_evidence: np.ndarray,
    rows: list[dict],
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    if not bool(cfg.get("enabled", False)):
        return x_train, y_train, y_evidence, rows, 0

    target_evidence_types = {str(value) for value in cfg.get("evidence_types", [])}
    if not target_evidence_types:
        return x_train, y_train, y_evidence, rows, 0

    scope = str(cfg.get("scope", "all_negative"))
    eligible_indices = [
        int(idx)
        for idx in np.where(y_train == 0.0)[0]
        if str(rows[int(idx)].get("evidence_type", "")) in target_evidence_types
    ]
    if scope == "target_train_only":
        eligible_indices = [
            idx for idx in eligible_indices if rows[int(idx)].get("step15_pool") == "zh_target_strict"
        ]
    elif scope == "source_train_only":
        eligible_indices = [
            idx for idx in eligible_indices if rows[int(idx)].get("step15_pool") == "en_content_train_pool"
        ]
    elif scope != "all_negative":
        raise ValueError(f"Unsupported Step15 negative_mixup scope: {scope}")

    if len(eligible_indices) < 2:
        return x_train, y_train, y_evidence, rows, 0

    multiplier = float(cfg.get("multiplier", 1.0))
    synthetic_count = int(round(len(eligible_indices) * multiplier))
    if synthetic_count <= 0:
        return x_train, y_train, y_evidence, rows, 0

    alpha = float(cfg.get("beta_alpha", 0.6))
    synthetic_x: list[np.ndarray] = []
    synthetic_rows: list[dict] = []
    for _ in range(synthetic_count):
        left, right = rng.choice(eligible_indices, size=2, replace=False)
        lam = float(rng.beta(alpha, alpha))
        synthetic_x.append((1.0 - lam) * x_train[left] + lam * x_train[right])
        left_pool = rows[int(left)].get("step15_pool", "")
        right_pool = rows[int(right)].get("step15_pool", "")
        synthetic_pool = left_pool if left_pool == right_pool else "cross_domain_negative_mixup"
        synthetic_rows.append(
            {
                "pair_uid": f"synthetic_negative_mixup::{len(synthetic_rows)}",
                "review_label": "negative",
                "identity_label": "different_controller",
                "evidence_type": "synthetic_train_only",
                "evidence_type_confident": "0",
                "step15_pool": synthetic_pool,
                "split_name": "train",
                "synthetic_train_only": "1",
            }
        )

    x_aug = np.vstack([x_train, np.asarray(synthetic_x, dtype=float)])
    y_aug = np.concatenate([y_train, np.zeros(synthetic_count, dtype=float)])
    evidence_aug = np.concatenate([y_evidence, np.full(synthetic_count, -1, dtype=int)])
    return x_aug, y_aug, evidence_aug, rows + synthetic_rows, synthetic_count


def init_params(input_dim: int, hidden_dim: int, evidence_type_count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = 0.03
    return {
        "w1": rng.normal(0.0, scale, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=float),
        "wi": rng.normal(0.0, scale, size=hidden_dim),
        "bi": np.zeros(1, dtype=float),
        "we": rng.normal(0.0, scale, size=(hidden_dim, evidence_type_count)),
        "be": np.zeros(evidence_type_count, dtype=float),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    identity_prob = sigmoid(hidden @ params["wi"] + float(params["bi"][0]))
    evidence_prob = softmax(hidden @ params["we"] + params["be"])
    return hidden, identity_prob, evidence_prob


def multitask_loss_and_grads(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y_identity: np.ndarray,
    y_evidence: np.ndarray,
    identity_weights: np.ndarray,
    evidence_sample_weights: np.ndarray,
    lambda_evidence: float,
    l2_weight: float,
) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
    hidden, p_identity, p_evidence = forward(params, x)
    p_identity_clipped = np.clip(p_identity, 1e-8, 1.0 - 1e-8)

    identity_weight_sum = max(float(identity_weights.sum()), 1e-12)
    identity_loss = -float(
        np.sum(identity_weights * (y_identity * np.log(p_identity_clipped) + (1.0 - y_identity) * np.log(1.0 - p_identity_clipped)))
        / identity_weight_sum
    )

    d_identity_logit = (p_identity - y_identity) * identity_weights / identity_weight_sum
    grads = {
        "w1": np.zeros_like(params["w1"]),
        "b1": np.zeros_like(params["b1"]),
        "wi": hidden.T @ d_identity_logit + l2_weight * params["wi"],
        "bi": np.asarray([float(np.sum(d_identity_logit))]),
        "we": np.zeros_like(params["we"]),
        "be": np.zeros_like(params["be"]),
    }
    d_hidden = d_identity_logit[:, None] * params["wi"][None, :]

    evidence_loss = 0.0
    evidence_mask = (y_evidence >= 0) & (evidence_sample_weights > 0.0) & (lambda_evidence > 0.0)
    if np.any(evidence_mask):
        selected = np.where(evidence_mask)[0]
        evidence_weight_sum = max(float(evidence_sample_weights[selected].sum()), 1e-12)
        selected_probs = np.clip(p_evidence[selected, y_evidence[selected]], 1e-8, 1.0)
        evidence_loss = -float(np.sum(evidence_sample_weights[selected] * np.log(selected_probs)) / evidence_weight_sum)
        d_evidence_logit = np.zeros_like(p_evidence)
        d_evidence_logit[selected] = p_evidence[selected]
        d_evidence_logit[selected, y_evidence[selected]] -= 1.0
        d_evidence_logit[selected] *= (lambda_evidence * evidence_sample_weights[selected] / evidence_weight_sum)[:, None]
        grads["we"] = hidden.T @ d_evidence_logit + l2_weight * params["we"]
        grads["be"] = d_evidence_logit.sum(axis=0)
        d_hidden += d_evidence_logit @ params["we"].T

    d_z1 = d_hidden * (1.0 - hidden * hidden)
    grads["w1"] = x.T @ d_z1 + l2_weight * params["w1"]
    grads["b1"] = d_z1.sum(axis=0)

    l2_loss = 0.5 * l2_weight * float(
        np.sum(params["w1"] ** 2) + np.sum(params["wi"] ** 2) + np.sum(params["we"] ** 2)
    )
    total_loss = identity_loss + lambda_evidence * evidence_loss + l2_loss
    diagnostics = {
        "identity_loss": identity_loss,
        "evidence_loss": evidence_loss,
        "l2_loss": l2_loss,
        "total_loss": total_loss,
    }
    return total_loss, grads, diagnostics


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray]],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for key in params:
        state["m"][key] = beta1 * state["m"][key] + (1.0 - beta1) * grads[key]
        state["v"][key] = beta2 * state["v"][key] + (1.0 - beta2) * (grads[key] ** 2)
        m_hat = state["m"][key] / (1.0 - beta1**step)
        v_hat = state["v"][key] / (1.0 - beta2**step)
        params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def metric_value(metric_name: str, y_true: np.ndarray, prob: np.ndarray) -> float:
    if metric_name == "average_precision":
        value = step7.average_precision_score(y_true, prob)
    elif metric_name == "roc_auc":
        value = step7.roc_auc_score(y_true, prob)
    elif metric_name == "logloss":
        value = -step7.binary_logloss(y_true, prob)
    else:
        raise ValueError(f"Unsupported Step15 early stopping metric: {metric_name}")
    if value is None:
        return float("-inf")
    return float(value)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_evidence: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    evidence_type_count: int,
    train_cfg: dict,
    lambda_evidence: float,
    seed: int,
    train_rows: list[dict],
    initial_params: dict[str, np.ndarray] | None = None,
    domain_balanced_identity_loss: bool = False,
    domain_balance_mode: str = "legacy_raw_row_count_before_quality_weights",
    domain_balance_domains: list[str] | None = None,
    identity_weight_multipliers: dict[str, float] | None = None,
    normalize_identity_weight_mean: bool = True,
    component_aware_identity_loss: bool = False,
    weight_pipeline_version: str = "legacy_v5r",
    fixed_update_budget: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    hidden_dim = int(train_cfg["hidden_dim"])
    params = copy.deepcopy(initial_params) if initial_params is not None else init_params(x_train.shape[1], hidden_dim, evidence_type_count, rng)
    state = {
        "m": {key: np.zeros_like(value) for key, value in params.items()},
        "v": {key: np.zeros_like(value) for key, value in params.items()},
    }

    component_weight_diagnostics = {"enabled": False, "method": "disabled"}
    class_weight_diagnostics = {"enabled": bool(train_cfg.get("class_balanced_identity_loss", True))}
    if weight_pipeline_version == "v6_component_row_class_evidence_domain":
        identity_weights = np.ones(len(y_train), dtype=float)
        if component_aware_identity_loss:
            identity_weights, component_weight_diagnostics = apply_component_inverse_sqrt_weights(
                identity_weights,
                train_rows,
            )
        identity_weights, row_sample_weight_diagnostics = apply_row_training_sample_weights(
            identity_weights,
            train_rows,
        )
        if train_cfg.get("class_balanced_identity_loss", True):
            identity_weights, class_weight_diagnostics = apply_class_balance_multipliers(
                identity_weights,
                y_train,
            )
        identity_weights, identity_weight_multiplier_diagnostics = apply_identity_weight_multipliers(
            identity_weights,
            train_rows,
            identity_weight_multipliers or {},
            normalize_mean=normalize_identity_weight_mean,
        )
        domain_balance_diagnostics = {
            "enabled": False,
            "method": domain_balance_mode,
        }
    elif domain_balanced_identity_loss and domain_balance_mode == "legacy_raw_row_count_before_quality_weights":
        identity_weights, domain_balance_diagnostics = legacy_domain_balanced_binary_weights(y_train, train_rows)
    elif train_cfg.get("class_balanced_identity_loss", True):
        identity_weights = balanced_binary_weights(y_train)
        domain_balance_diagnostics = {
            "enabled": False,
            "method": domain_balance_mode,
        }
    else:
        identity_weights = np.ones(len(y_train))
        domain_balance_diagnostics = {
            "enabled": False,
            "method": domain_balance_mode,
        }
    if weight_pipeline_version != "v6_component_row_class_evidence_domain":
        identity_weights, identity_weight_multiplier_diagnostics = apply_identity_weight_multipliers(
            identity_weights,
            train_rows,
            identity_weight_multipliers or {},
            normalize_mean=normalize_identity_weight_mean,
        )
        identity_weights, row_sample_weight_diagnostics = apply_row_training_sample_weights(identity_weights, train_rows)
    if domain_balanced_identity_loss and domain_balance_mode == "post_quality_effective_weight_mass":
        identity_weights, domain_balance_diagnostics = apply_effective_domain_balance(
            identity_weights,
            train_rows,
            domain_balance_domains or ["en_content_train_pool", "zh_target_strict"],
        )
    elif domain_balanced_identity_loss and domain_balance_mode != "legacy_raw_row_count_before_quality_weights":
        raise ValueError(f"Unsupported Step15 domain_balance_mode: {domain_balance_mode}")
    elif not domain_balanced_identity_loss:
        domain_balance_diagnostics = {
            "enabled": False,
            "method": domain_balance_mode,
            "allowed_domains": list(domain_balance_domains or ["en_content_train_pool", "zh_target_strict"]),
        }
    ev_weights, auxiliary_weight_diagnostics = evidence_weights(
        y_evidence,
        evidence_type_count,
        identity_weights,
        class_balance=bool(train_cfg.get("class_balanced_evidence_loss", True)),
    )

    max_epochs = int(fixed_update_budget or train_cfg["max_epochs"])
    if max_epochs <= 0:
        raise ValueError("Step15 training update budget must be positive")
    patience = int(train_cfg["patience_epochs"])
    learning_rate = float(train_cfg["learning_rate"])
    l2_weight = float(train_cfg["l2_weight"])
    metric_name = str(train_cfg["early_stopping_metric"])

    best_params = copy.deepcopy(params)
    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_loss = {}

    for epoch in range(1, max_epochs + 1):
        _, grads, last_loss = multitask_loss_and_grads(
            params,
            x_train,
            y_train,
            y_evidence,
            identity_weights,
            ev_weights,
            lambda_evidence,
            l2_weight,
        )
        adam_update(params, grads, state, epoch, learning_rate)

        _, valid_prob, _ = forward(params, x_valid)
        valid_metric = metric_value(metric_name, y_valid, valid_prob)
        if valid_metric > best_metric + 1e-8:
            best_metric = valid_metric
            best_params = copy.deepcopy(params)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if fixed_update_budget is None and epochs_without_improvement >= patience:
            break

    diagnostics = {
        "best_epoch": int(best_epoch),
        "trained_epoch_count": int(epoch),
        "training_budget_mode": "fixed_optimizer_updates_with_best_valid_checkpoint"
        if fixed_update_budget is not None
        else "validation_early_stopping",
        "fixed_update_budget": fixed_update_budget,
        "early_stopping_metric": metric_name,
        "best_valid_metric": round(float(best_metric), 6),
        "last_loss": {key: round(float(value), 6) for key, value in last_loss.items()},
        "lambda_evidence": round(float(lambda_evidence), 6),
        "initialization": "warm_start" if initial_params is not None else "random",
        "domain_balanced_identity_loss": bool(domain_balanced_identity_loss),
        "effective_domain_balance": domain_balance_diagnostics,
        "identity_weight_multipliers": identity_weight_multiplier_diagnostics,
        "row_sample_weight_multipliers": row_sample_weight_diagnostics,
        "component_weight_multipliers": component_weight_diagnostics,
        "class_weight_multipliers": class_weight_diagnostics,
        "auxiliary_evidence_weight_multipliers": auxiliary_weight_diagnostics,
        "weight_pipeline_version": weight_pipeline_version,
    }
    return best_params, diagnostics


def prediction_rows(rows: list[dict], probabilities: np.ndarray, threshold: float, experiment_token: str) -> list[dict]:
    predictions = (probabilities >= threshold).astype(int)
    output = []
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=True):
        output.append(
            {
                "experiment_name": experiment_token,
                "pair_uid": row["pair_uid"],
                "data_bucket": row["data_bucket"],
                "split_name": row["split_name"],
                "review_label": row["review_label"],
                "y_true": label_to_int(row["review_label"]),
                # Keep Python's round-trip float representation. Ranking metrics are
                # recomputed from this CSV, so six-decimal quantization can create
                # artificial score ties and change ROC-AUC/AP/PR-AUC.
                "prob_positive": float(probability),
                "threshold": float(threshold),
                "pred_positive": int(prediction),
                "split_component_id": row.get("split_component_id", ""),
                "review_stratum": row.get("review_stratum", ""),
                "evidence_type": row.get("evidence_type", ""),
                "identity_label": row.get("identity_label", ""),
                "source_seller_raw_left": row.get("source_seller_raw_left", ""),
                "source_seller_raw_right": row.get("source_seller_raw_right", ""),
            }
        )
    return output


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row.get("review_label", "") for row in rows).items())),
        "evidence_type_counts": dict(sorted(Counter(row.get("evidence_type", "") for row in rows).items())),
        "split_counts": dict(sorted(Counter(row.get("split_name", "") for row in rows).items())),
    }


def feature_importance(params: dict[str, np.ndarray], feature_names: list[str], limit: int = 25) -> list[dict]:
    scores = np.sum(np.abs(params["w1"] * params["wi"][None, :]), axis=1)
    records = [
        {"feature_name": feature, "importance": round(float(score), 8)}
        for feature, score in zip(feature_names, scores, strict=True)
    ]
    records.sort(key=lambda item: (-item["importance"], item["feature_name"]))
    return records[:limit]


def write_positive_mixup_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "pair_uid",
        "review_label",
        "identity_label",
        "evidence_type",
        "evidence_type_confident",
        "identity_training_eligible",
        "step15_pool",
        "split_name",
        "synthetic_train_only",
        "usable_for_supervision",
        "usable_for_core_transfer",
        "core_transfer_eligible",
        "training_sample_weight",
        "mixup_parent_left_pair_uid",
        "mixup_parent_right_pair_uid",
        "mixup_parent_left_pool",
        "mixup_parent_right_pool",
        "mixup_parent_left_evidence_type",
        "mixup_parent_right_evidence_type",
        "mixup_parent_left_training_sample_weight",
        "mixup_parent_right_training_sample_weight",
        "mixup_lambda_right",
    ]
    manifest_rows = [
        {field: row.get(field, "") for field in fields}
        for row in rows
    ]
    atomic_write_csv(path, manifest_rows, fields)


def artifact_payload(
    params: dict[str, np.ndarray],
    feature_names: list[str],
    standardizer_bundle: dict,
    evidence_types: list[str],
    diagnostics: dict,
) -> dict:
    return {
        "feature_names": feature_names,
        "feature_means": [round(float(value), 10) for value in standardizer_bundle["global_means"]],
        "feature_stds": [round(float(value), 10) for value in standardizer_bundle["global_stds"]],
        "standardizer_bundle": standardizer_bundle,
        "evidence_types": evidence_types,
        "training_diagnostics": diagnostics,
        "params": {
            key: np.round(value, 10).tolist()
            for key, value in params.items()
        },
    }


def run_single(
    experiment_name: str,
    experiment_cfg: dict,
    phase_cfg: dict,
    seed: int,
    policy: dict,
    rows_by_pool: dict[str, list[dict]],
    initial_params: dict[str, np.ndarray] | None = None,
    standardizer_override: dict | None = None,
    evaluate_test: bool = True,
) -> tuple[dict, dict[str, np.ndarray]]:
    feature_names = policy["feature_sets"][experiment_cfg["feature_set"]]
    validate_features(rows_by_pool, feature_names)

    train_phase_cfg = phase_cfg
    evidence_override = experiment_cfg.get("training_evidence_types_override")
    if evidence_override is not None:
        train_phase_cfg = {
            **phase_cfg,
            "included_evidence_types": [str(value) for value in evidence_override],
        }
    train_rows = select_train_rows(rows_by_pool, experiment_cfg, train_phase_cfg, policy)
    valid_rows = select_eval_rows(rows_by_pool["zh_target_strict"], policy["splits"]["target_valid"])
    test_rows = (
        select_eval_rows(rows_by_pool["zh_target_strict"], policy["splits"]["target_test"])
        if evaluate_test
        else []
    )
    if not train_rows:
        raise ValueError(f"{experiment_name}/{phase_cfg['phase_id']} has no train rows")
    if not valid_rows or (evaluate_test and not test_rows):
        raise ValueError("Step15 requires non-empty fixed zh_valid and endpoint zh_test rows")

    if standardizer_override is None:
        standardizer_bundle = fit_standardizer_bundle(
            train_rows,
            feature_names,
            policy.get("feature_preprocessing", {}).get(experiment_cfg["feature_set"], {}),
        )
        standardizer_source = "current_phase_train"
    else:
        standardizer_bundle = standardizer_override
        standardizer_source = "common_final_phase_train"
    x_train = apply_standardizer_bundle(train_rows, feature_names, standardizer_bundle)
    x_valid = apply_standardizer_bundle(valid_rows, feature_names, standardizer_bundle)
    x_test = (
        apply_standardizer_bundle(test_rows, feature_names, standardizer_bundle)
        if evaluate_test
        else None
    )
    y_train = y_from_rows(train_rows)
    y_valid = y_from_rows(valid_rows)
    y_test = y_from_rows(test_rows) if evaluate_test else None
    y_ev = evidence_indices(train_rows, policy["evidence_types"])
    train_rows_for_weights = list(train_rows)

    synthetic_count = 0
    positive_synthetic_count = 0
    negative_synthetic_count = 0
    positive_synthetic_rows: list[dict] = []
    positive_mixup_cfg = dict(policy["training"].get("positive_mixup", {}))
    positive_mixup_cfg.update(experiment_cfg.get("positive_mixup", {}))
    positive_mixup_diagnostics = {
        "enabled": False,
        "scope": str(
            experiment_cfg.get("positive_mixup_scope_override")
            or positive_mixup_cfg.get("scope", "all_positive")
        ),
        "synthetic_row_count": 0,
        "skipped_reason": "phase_does_not_enable_positive_mixup",
    }
    if bool(phase_cfg.get("use_positive_mixup", False)) and not bool(
        experiment_cfg.get("disable_positive_mixup", False)
    ):
        x_train, y_train, y_ev, train_rows_for_weights, positive_synthetic_count, positive_mixup_diagnostics = add_positive_mixup(
            x_train,
            y_train,
            y_ev,
            train_rows,
            positive_mixup_cfg,
            np.random.default_rng(seed + 7919),
            feature_names,
            experiment_cfg.get("positive_mixup_scope_override"),
        )
        positive_synthetic_rows = train_rows_for_weights[len(train_rows):]
    if bool(phase_cfg.get("use_negative_mixup", False)) and bool(experiment_cfg.get("negative_mixup", {}).get("enabled", False)):
        x_train, y_train, y_ev, train_rows_for_weights, negative_synthetic_count = add_negative_mixup(
            x_train,
            y_train,
            y_ev,
            train_rows_for_weights,
            experiment_cfg.get("negative_mixup", {}),
            np.random.default_rng(seed + 15485863),
        )
    synthetic_count = positive_synthetic_count + negative_synthetic_count

    lambda_evidence = float(experiment_cfg.get("lambda_evidence", 0.0))
    params, diagnostics = train_model(
        x_train,
        y_train,
        y_ev,
        x_valid,
        y_valid,
        len(policy["evidence_types"]),
        policy["training"],
        lambda_evidence,
        seed,
        train_rows_for_weights,
        initial_params=initial_params,
        domain_balanced_identity_loss=bool(experiment_cfg.get("domain_balanced_identity_loss", False)),
        domain_balance_mode=str(
            experiment_cfg.get("domain_balance_mode", "legacy_raw_row_count_before_quality_weights")
        ),
        domain_balance_domains=[str(value) for value in experiment_cfg.get(
            "domain_balance_domains",
            ["en_content_train_pool", "zh_target_strict"],
        )],
        identity_weight_multipliers=experiment_cfg.get("identity_evidence_type_weight_multipliers", {}),
        normalize_identity_weight_mean=bool(experiment_cfg.get("normalize_identity_weight_mean", True)),
        component_aware_identity_loss=bool(experiment_cfg.get("component_aware_identity_loss", False)),
        weight_pipeline_version=str(experiment_cfg.get("weight_pipeline_version", "legacy_v5r")),
        fixed_update_budget=(
            int(experiment_cfg["fixed_update_budget_per_phase"])
            if experiment_cfg.get("fixed_update_budget_per_phase") is not None
            else None
        ),
    )
    diagnostics["training_mode"] = str(experiment_cfg.get("training_mode", "from_scratch_each_phase"))
    diagnostics["training_evidence_types"] = list(train_phase_cfg["included_evidence_types"])
    diagnostics["positive_mixup_disabled_by_experiment"] = bool(
        experiment_cfg.get("disable_positive_mixup", False)
    )
    diagnostics["positive_mixup_scope"] = str(positive_mixup_diagnostics.get("scope", "all_positive"))
    diagnostics["positive_pair_mixup"] = positive_mixup_diagnostics
    diagnostics["standardizer_source"] = standardizer_source
    diagnostics["standardizer_sha256"] = standardizer_bundle["sha256"]

    _, valid_prob, valid_evidence_prob = forward(params, x_valid)
    test_prob = None
    if evaluate_test:
        assert x_test is not None
        _, test_prob, _ = forward(params, x_test)
    threshold = step7.choose_threshold(y_valid, valid_prob, policy["threshold_selection"]["metric"], policy)
    valid_metrics = step7.evaluate_probabilities(y_valid, valid_prob, threshold)
    test_metrics = (
        step7.evaluate_probabilities(y_test, test_prob, threshold)
        if evaluate_test and y_test is not None and test_prob is not None
        else None
    )

    experiment_token = f"{experiment_name}_{phase_cfg['phase_id']}_seed_{seed}"
    valid_predictions = prediction_rows(valid_rows, valid_prob, threshold, experiment_token)
    test_predictions = (
        prediction_rows(test_rows, test_prob, threshold, experiment_token)
        if evaluate_test and test_prob is not None
        else []
    )

    valid_path = output_path(policy["outputs"]["zh_valid_predictions_template"], experiment_name, phase_cfg["phase_id"], seed)
    test_path = (
        output_path(
            policy["outputs"]["zh_test_predictions_template"],
            experiment_name,
            phase_cfg["phase_id"],
            seed,
        )
        if evaluate_test
        else None
    )
    artifact_path = output_path(policy["outputs"]["artifact_template"], experiment_name, phase_cfg["phase_id"], seed)
    mixup_manifest_path = None
    mixup_manifest_template = policy["outputs"].get("positive_mixup_manifest_template")
    if positive_synthetic_rows and mixup_manifest_template:
        mixup_manifest_path = output_path(mixup_manifest_template, experiment_name, phase_cfg["phase_id"], seed)
        write_positive_mixup_manifest(mixup_manifest_path, positive_synthetic_rows)
    atomic_write_csv(valid_path, valid_predictions, list(valid_predictions[0].keys()))
    if test_path is not None:
        atomic_write_csv(test_path, test_predictions, list(test_predictions[0].keys()))

    artifact = artifact_payload(params, feature_names, standardizer_bundle, policy["evidence_types"], diagnostics)
    artifact["frozen_zh_valid_threshold"] = float(threshold)
    artifact["threshold_selection_scope"] = "fixed_zh_valid_only_never_zh_test"
    artifact["feature_importance"] = feature_importance(params, feature_names)
    atomic_write_json(artifact_path, artifact)

    run_record = {
        "experiment_name": experiment_name,
        "phase_id": phase_cfg["phase_id"],
        "phase_index": int(phase_cfg["phase_index"]),
        "seed": int(seed),
        "role": experiment_cfg.get("role", ""),
        "feature_set": experiment_cfg["feature_set"],
        "use_identifier_features": bool(experiment_cfg.get("use_identifier_features", False)),
        "configured_phase_evidence_types": phase_cfg["included_evidence_types"],
        "training_evidence_types": train_phase_cfg["included_evidence_types"],
        "included_evidence_types": train_phase_cfg["included_evidence_types"],
        "synthetic_train_only_mixup_count": int(synthetic_count),
        "synthetic_train_only_positive_mixup_count": int(positive_synthetic_count),
        "synthetic_train_only_negative_mixup_count": int(negative_synthetic_count),
        "train_dataset": summarize_rows(train_rows),
        "zh_valid_dataset": summarize_rows(valid_rows),
        "zh_test_dataset": summarize_rows(test_rows) if evaluate_test else None,
        "zh_test_evaluation_role": (
            "final_preregistered_endpoint_only" if evaluate_test else "not_evaluated_intermediate_phase"
        ),
        "training_diagnostics": diagnostics,
        "zh_valid_metrics": valid_metrics,
        "frozen_zh_valid_threshold": float(threshold),
        "zh_test_metrics": test_metrics,
        "output_paths": {
            "artifact": str(artifact_path.relative_to(ROOT)),
            "zh_valid_predictions": str(valid_path.relative_to(ROOT)),
            "zh_test_predictions": str(test_path.relative_to(ROOT)) if test_path is not None else None,
            "positive_mixup_manifest": (
                str(mixup_manifest_path.relative_to(ROOT)) if mixup_manifest_path is not None else None
            ),
        },
        "top_feature_importance": artifact["feature_importance"][:10],
    }
    return run_record, params


def summarize_runs(runs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[f"{run['experiment_name']}::{run['phase_id']}"].append(run)
    summary = {}
    for key, items in sorted(grouped.items()):
        def metric_values(metric_name: str) -> list[float]:
            values = []
            for item in items:
                metrics = item.get("zh_test_metrics") or {}
                value = metrics.get(metric_name)
                if value is not None:
                    values.append(float(value))
            return values

        def summarize_metric(metric_name: str) -> dict[str, float | None]:
            values = metric_values(metric_name)
            if not values:
                return {"mean": None, "min": None, "max": None}
            return {
                "mean": round(float(np.mean(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
            }

        auc_values = metric_values("roc_auc")
        ap_values = metric_values("average_precision")
        pr_auc_summary = summarize_metric("pr_auc")
        f1_summary = summarize_metric("f1")
        map_summary = summarize_metric("map")
        mrr_summary = summarize_metric("mrr")
        summary[key] = {
            "run_count": len(items),
            "seeds": [item["seed"] for item in items],
            "zh_test_roc_auc_mean": round(float(np.mean(auc_values)), 6) if auc_values else None,
            "zh_test_roc_auc_min": round(float(np.min(auc_values)), 6) if auc_values else None,
            "zh_test_roc_auc_max": round(float(np.max(auc_values)), 6) if auc_values else None,
            "zh_test_average_precision_mean": round(float(np.mean(ap_values)), 6) if ap_values else None,
            "zh_test_average_precision_min": round(float(np.min(ap_values)), 6) if ap_values else None,
            "zh_test_average_precision_max": round(float(np.max(ap_values)), 6) if ap_values else None,
            "zh_test_pr_auc_mean": pr_auc_summary["mean"],
            "zh_test_pr_auc_min": pr_auc_summary["min"],
            "zh_test_pr_auc_max": pr_auc_summary["max"],
            "zh_test_f1_mean": f1_summary["mean"],
            "zh_test_f1_min": f1_summary["min"],
            "zh_test_f1_max": f1_summary["max"],
            "zh_test_map_mean": map_summary["mean"],
            "zh_test_map_min": map_summary["min"],
            "zh_test_map_max": map_summary["max"],
            "zh_test_mrr_mean": mrr_summary["mean"],
            "zh_test_mrr_min": mrr_summary["min"],
            "zh_test_mrr_max": mrr_summary["max"],
        }
    return summary


def select_validation_only_candidates(runs: list[dict], policy: dict) -> dict:
    selections: dict[str, dict] = {}
    expected_seeds = sorted(int(seed) for seed in policy["training"]["default_seeds"])
    for group_name, cfg in sorted(policy.get("validation_only_model_selection", {}).items()):
        candidates = [str(value) for value in cfg["candidate_experiments"]]
        phase_id = str(cfg["phase_id"])
        metric_name = str(cfg.get("metric", "average_precision"))
        candidate_runs = [run for run in runs if run["experiment_name"] in candidates]
        if not candidate_runs:
            selections[group_name] = {
                "selection_scope": "fixed_zh_valid_only_never_zh_test",
                "metric": metric_name,
                "status": "not_run",
                "candidates": [],
                "selected_experiment": None,
            }
            continue
        records = []
        for experiment_name in candidates:
            matching = [
                run
                for run in runs
                if run["experiment_name"] == experiment_name and run["phase_id"] == phase_id
            ]
            matching_seeds = sorted(int(run["seed"]) for run in matching)
            if matching_seeds != expected_seeds:
                raise ValueError(
                    "Step15 validation-only selection requires the complete preregistered seed set; "
                    f"experiment={experiment_name}, phase={phase_id}, "
                    f"expected={expected_seeds}, actual={matching_seeds}"
                )
            values = [
                float(run["zh_valid_metrics"][metric_name])
                for run in matching
                if run["zh_valid_metrics"].get(metric_name) is not None
            ]
            ensemble_metric = None
            if matching:
                y_by_pair: dict[str, float] = {}
                score_maps: list[dict[str, float]] = []
                for run in matching:
                    prediction_path = resolve_path(run["output_paths"]["zh_valid_predictions"])
                    prediction_rows = step7.load_csv(prediction_path)
                    score_map = {}
                    for row in prediction_rows:
                        pair_uid = str(row["pair_uid"])
                        y_true = float(row["y_true"])
                        if pair_uid in y_by_pair and y_by_pair[pair_uid] != y_true:
                            raise ValueError(f"Step15 validation files disagree on y_true for {pair_uid}")
                        y_by_pair[pair_uid] = y_true
                        score_map[pair_uid] = float(row["prob_positive"])
                    score_maps.append(score_map)
                pair_order = sorted(y_by_pair)
                if any(set(score_map) != set(pair_order) for score_map in score_maps):
                    raise ValueError(f"Step15 validation seed files have mismatched pair coverage for {experiment_name}")
                y_valid = np.asarray([y_by_pair[pair_uid] for pair_uid in pair_order], dtype=float)
                ensemble_scores = np.asarray(
                    [[score_map[pair_uid] for pair_uid in pair_order] for score_map in score_maps],
                    dtype=float,
                ).mean(axis=0)
                if metric_name == "average_precision":
                    ensemble_metric = step7.average_precision_score(y_valid, ensemble_scores)
                elif metric_name == "roc_auc":
                    ensemble_metric = step7.roc_auc_score(y_valid, ensemble_scores)
                else:
                    raise ValueError(f"Unsupported Step15 validation-only selection metric: {metric_name}")
            records.append(
                {
                    "experiment_name": experiment_name,
                    "phase_id": phase_id,
                    "expected_seeds": expected_seeds,
                    "actual_seeds": matching_seeds,
                    "seed_count": len(values),
                    "mean_per_seed_zh_valid_metric": round(float(np.mean(values)), 8) if values else None,
                    "ensemble_zh_valid_metric": None
                    if ensemble_metric is None
                    else round(float(ensemble_metric), 8),
                }
            )
        eligible = [record for record in records if record["ensemble_zh_valid_metric"] is not None]
        tie_break_order = [str(value) for value in cfg.get("tie_break_order", candidates)]
        if set(tie_break_order) != set(candidates):
            raise ValueError(
                f"Step15 validation selection tie-break must list every candidate exactly once: {group_name}"
            )
        tie_rank = {experiment_name: index for index, experiment_name in enumerate(tie_break_order)}
        selected = min(
            eligible,
            key=lambda record: (
                -float(record["ensemble_zh_valid_metric"]),
                tie_rank[record["experiment_name"]],
            ),
        ) if eligible else None
        selections[group_name] = {
            "selection_scope": "fixed_zh_valid_only_never_zh_test",
            "metric": metric_name,
            "status": "selected" if selected else "incomplete",
            "tie_break_order": tie_break_order,
            "candidates": records,
            "selected_experiment": selected["experiment_name"] if selected else None,
        }
    return selections


def validation_selection_candidate_experiments(policy: dict) -> set[str]:
    return {
        str(experiment_name)
        for cfg in policy.get("validation_only_model_selection", {}).values()
        for experiment_name in cfg.get("candidate_experiments", [])
    }


def params_from_artifact(artifact: dict) -> dict[str, np.ndarray]:
    params = artifact.get("params") or {}
    required = {"w1", "b1", "wi", "bi", "we", "be"}
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Step15 artifact is missing frozen model parameters: {missing}")
    return {key: np.asarray(params[key], dtype=float) for key in sorted(required)}


def materialize_validation_selected_test_predictions(
    runs: list[dict],
    selections: dict[str, dict],
    policy: dict,
    rows_by_pool: dict[str, list[dict]],
) -> list[dict]:
    selected_pairs = {
        (str(selection["selected_experiment"]), str(cfg["phase_id"]))
        for group_name, cfg in policy.get("validation_only_model_selection", {}).items()
        for selection in [selections.get(group_name, {})]
        if selection.get("selected_experiment")
    }
    if not selected_pairs:
        return runs
    expected_seeds = sorted(int(seed) for seed in policy["training"]["default_seeds"])
    test_rows = select_eval_rows(rows_by_pool["zh_target_strict"], policy["splits"]["target_test"])
    if not test_rows:
        raise ValueError("Cannot materialize a validation-selected endpoint without fixed zh_test rows")
    updated = []
    for run in runs:
        key = (str(run["experiment_name"]), str(run["phase_id"]))
        if key not in selected_pairs:
            updated.append(run)
            continue
        matching_seeds = sorted(
            int(item["seed"])
            for item in runs
            if (str(item["experiment_name"]), str(item["phase_id"])) == key
        )
        if matching_seeds != expected_seeds:
            raise ValueError(
                "Validation-selected test materialization requires all preregistered seeds: "
                f"endpoint={key}, expected={expected_seeds}, actual={matching_seeds}"
            )
        artifact_path = resolve_path(run["output_paths"]["artifact"])
        valid_path = resolve_path(run["output_paths"]["zh_valid_predictions"])
        artifact = step7.load_json(artifact_path)
        valid_predictions = step7.load_csv(valid_path)
        rounded_thresholds = {float(row["threshold"]) for row in valid_predictions}
        if len(rounded_thresholds) != 1:
            raise ValueError(f"Validation predictions do not bind one frozen threshold: {valid_path}")
        threshold = float(artifact["frozen_zh_valid_threshold"])
        if round(threshold, 6) != next(iter(rounded_thresholds)):
            raise ValueError(f"Artifact and validation prediction thresholds disagree: {artifact_path}")
        feature_names = list(artifact["feature_names"])
        x_test = apply_standardizer_bundle(test_rows, feature_names, artifact["standardizer_bundle"])
        _, test_prob, _ = forward(params_from_artifact(artifact), x_test)
        y_test = y_from_rows(test_rows)
        experiment_token = f"{run['experiment_name']}_{run['phase_id']}_seed_{run['seed']}"
        test_predictions = prediction_rows(test_rows, test_prob, threshold, experiment_token)
        test_path = output_path(
            policy["outputs"]["zh_test_predictions_template"],
            run["experiment_name"],
            run["phase_id"],
            int(run["seed"]),
        )
        atomic_write_csv(test_path, test_predictions, list(test_predictions[0].keys()))
        updated_run = copy.deepcopy(run)
        updated_run["zh_test_dataset"] = summarize_rows(test_rows)
        updated_run["zh_test_evaluation_role"] = (
            "validation_selected_frozen_artifact_preregistered_endpoint_only"
        )
        updated_run["zh_test_metrics"] = step7.evaluate_probabilities(y_test, test_prob, threshold)
        updated_run["output_paths"]["zh_test_predictions"] = str(test_path.relative_to(ROOT))
        updated_run["test_materialization"] = {
            "training_reused": False,
            "artifact_sha256": file_sha256(artifact_path),
            "selection_scope": "fixed_zh_valid_only_never_zh_test",
            "threshold_source": "frozen_zh_valid_predictions",
        }
        updated.append(updated_run)
    return updated


def build_input_manifest(
    policy_path: Path,
    policy: dict,
    extra_paths: list[Path] | None = None,
) -> dict:
    paths = {
        policy_path,
        Path(__file__).resolve(),
        Path(step7.__file__).resolve(),
    }
    for pool_cfg in policy["pools"].values():
        for key in ("frozen_labels", "pair_features", "label_output"):
            paths.add(resolve_path(pool_cfg[key]))
    lineage_cfg = policy.get("inductive_feature_lineage", {})
    if bool(lineage_cfg.get("enabled", False)):
        paths.add(ROOT / "scripts" / "step15_build_v6_inductive_pair_features.py")
        paths.add(resolve_path(lineage_cfg["reference_bundle_output"]))
        paths.add(resolve_path(lineage_cfg["manifest_output"]))
    paths.update(path.resolve() for path in (extra_paths or []))
    records = []
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.exists():
            raise FileNotFoundError(f"Step15 input manifest path does not exist: {path}")
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "git_commit": current_git_commit(),
        "inputs": records,
        "manifest_sha256": canonical_json_sha256({"inputs": records}),
    }


def validate_inductive_feature_lineage(policy_path: Path, policy: dict) -> dict | None:
    cfg = policy.get("inductive_feature_lineage", {})
    if not bool(cfg.get("enabled", False)):
        return None
    manifest_path = resolve_path(cfg["manifest_output"])
    reference_path = resolve_path(cfg["reference_bundle_output"])
    manifest = step7.load_json(manifest_path)
    expected_self_hash = str(manifest.get("manifest_sha256", ""))
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not expected_self_hash or canonical_json_sha256(core) != expected_self_hash:
        raise ValueError("Step15-v6 inductive feature manifest self-hash mismatch")
    if manifest.get("reference_scope") != "frozen_train_sellers_only" or bool(
        manifest.get("transductive_valid_or_test_covariates_used_for_reference", True)
    ):
        raise ValueError("Step15-v6 pair features do not satisfy the train-only reference contract")
    if file_sha256(policy_path) != manifest.get("policy_sha256"):
        raise ValueError("Step15-v6 inductive feature manifest was built under a different policy")
    producer_path = resolve_path(str(manifest.get("producer", "")))
    if not producer_path.exists() or file_sha256(producer_path) != manifest.get("producer_sha256"):
        raise ValueError("Step15-v6 inductive feature producer hash mismatch")
    if not reference_path.exists() or file_sha256(reference_path) != manifest.get("reference_bundle_sha256"):
        raise ValueError("Step15-v6 train-only corpus reference hash mismatch")
    for domain, pool_cfg in policy["pools"].items():
        record = (manifest.get("domains") or {}).get(domain)
        feature_path = resolve_path(pool_cfg["pair_features"])
        if not record or str(feature_path.relative_to(ROOT)) != record.get("output_path"):
            raise ValueError(f"Step15-v6 inductive feature path is not manifest-bound: {domain}")
        if not feature_path.exists() or file_sha256(feature_path) != record.get("output_sha256"):
            raise ValueError(f"Step15-v6 inductive feature hash mismatch: {domain}")
    return manifest


def resolve_experiment_phase_plan(
    experiment_name: str,
    experiment_cfg: dict,
    requested_phase_ids: list[str],
    phase_by_id: dict[str, dict],
) -> list[dict]:
    configured_ids = [
        str(value) for value in experiment_cfg.get("phase_ids", list(phase_by_id.keys()))
    ]
    selected_ids = [phase_id for phase_id in configured_ids if phase_id in set(requested_phase_ids)]
    if not selected_ids:
        raise ValueError(f"{experiment_name} has no selected phases after applying phase_ids")
    training_mode = str(experiment_cfg.get("training_mode", "from_scratch_each_phase"))
    if training_mode == "warm_start_curriculum":
        final_position = max(configured_ids.index(phase_id) for phase_id in selected_ids)
        required_prefix = configured_ids[: final_position + 1]
        if selected_ids != required_prefix:
            raise ValueError(
                f"{experiment_name} warm-start phases must be a complete configured prefix; "
                f"requested_selected={selected_ids}, required_prefix={required_prefix}"
            )
    return [phase_by_id[phase_id] for phase_id in selected_ids]


def should_evaluate_test_endpoint(
    experiment_name: str,
    phase_id: str,
    selected_phase_ids: list[str],
    policy: dict,
    *,
    validation_only: bool = False,
) -> bool:
    if validation_only:
        return False
    endpoint_only = (
        str(
            (policy.get("evaluation_boundary", {}) or {}).get(
                "zh_test_evaluation_schedule", "all_phases"
            )
        )
        == "final_preregistered_endpoint_only"
    )
    if not endpoint_only:
        return True
    configured_phase_ids = [
        str(value)
        for value in policy["experiments"][experiment_name].get("phase_ids", [])
    ]
    if not configured_phase_ids or selected_phase_ids != configured_phase_ids:
        return False
    if phase_id != configured_phase_ids[-1]:
        return False
    return experiment_name not in validation_selection_candidate_experiments(policy)


def main() -> None:
    args = parse_args()
    policy_path = resolve_path(args.policy)
    policy = step7.load_json(policy_path)

    experiments = args.experiments or policy["default_experiments"]
    legacy_experiments = [experiment for experiment in experiments if experiment.startswith("step15_e5_")]
    if legacy_experiments and not args.allow_legacy_output_overwrite:
        raise SystemExit(
            "Refusing to run legacy Step 15 experiments because they write first-pass "
            "artifact/prediction filenames and may overwrite old outputs: "
            f"{', '.join(legacy_experiments)}. Use step15_v2_* experiments, or pass "
            "--allow-legacy-output-overwrite only for an intentional legacy rerun."
        )
    seeds = args.seeds or policy["training"]["default_seeds"]
    phase_by_id = {phase["phase_id"]: phase for phase in policy["curriculum_phases"]}
    requested_phase_ids = args.phases or list(phase_by_id.keys())
    unknown_experiments = [experiment for experiment in experiments if experiment not in policy["experiments"]]
    if unknown_experiments:
        raise SystemExit(
            "Unknown Step15 experiment(s) for policy "
            f"{policy_path} version={policy.get('version')}: {', '.join(unknown_experiments)}"
        )
    unknown_phases = [phase_id for phase_id in requested_phase_ids if phase_id not in phase_by_id]
    if unknown_phases:
        raise SystemExit(
            "Unknown Step15 phase(s) for policy "
            f"{policy_path} version={policy.get('version')}: {', '.join(unknown_phases)}"
        )
    phases = [phase_by_id[phase_id] for phase_id in requested_phase_ids]
    if len(experiments) != len(set(experiments)):
        raise SystemExit("Duplicate --experiment selections are not allowed")
    if len(requested_phase_ids) != len(set(requested_phase_ids)):
        raise SystemExit("Duplicate --phase selections are not allowed")
    if len(seeds) != len(set(seeds)):
        raise SystemExit("Duplicate --seed selections are not allowed")
    experiment_phase_plans = {
        experiment_name: resolve_experiment_phase_plan(
            experiment_name,
            policy["experiments"][experiment_name],
            requested_phase_ids,
            phase_by_id,
        )
        for experiment_name in experiments
    }

    if args.validate_config_only:
        v5r_experiments = [experiment for experiment in experiments if experiment.startswith("step15_v5r_")]
        if v5r_experiments:
            expected_summary = "reports/step15_v5r_weighted_mixup_summary.json"
            actual_summary = str(policy.get("outputs", {}).get("summary_json", ""))
            if actual_summary != expected_summary:
                raise SystemExit(
                    "Step15 v5r output isolation failed: expected summary_json="
                    f"{expected_summary}, got {actual_summary or '<missing>'}"
                )
            for experiment in v5r_experiments:
                experiment_cfg = policy["experiments"][experiment]
                mixup_cfg = dict(policy.get("training", {}).get("positive_mixup", {}))
                mixup_cfg.update(experiment_cfg.get("positive_mixup", {}))
                required_contract = {
                    "scope": "same_domain_same_evidence_type",
                    "synthetic_weight_mode": "minimum_parent_weight",
                    "minimum_source_training_sample_weight": 0.55,
                    "nearest_neighbor_k": 5,
                }
                mismatches = {
                    key: {"expected": value, "actual": mixup_cfg.get(key)}
                    for key, value in required_contract.items()
                    if mixup_cfg.get(key) != value
                }
                if mismatches:
                    raise SystemExit(f"Step15 v5r mixup contract mismatch for {experiment}: {mismatches}")
                if bool(experiment_cfg.get("domain_balanced_identity_loss", False)) and str(
                    experiment_cfg.get("domain_balance_mode", "")
                ) != "post_quality_effective_weight_mass":
                    raise SystemExit(
                        f"Step15 v5r domain-balance contract mismatch for {experiment}: "
                        f"domain_balance_mode={experiment_cfg.get('domain_balance_mode')!r}"
                    )
        if "step15-v6" in str(policy.get("version", "")):
            expected_seeds = list(range(20260320, 20260330))
            if list(policy["training"].get("default_seeds", [])) != expected_seeds:
                raise SystemExit("Step15 v6 requires the preregistered ten seeds 20260320..20260329")
            if int(policy["training"].get("hidden_dim", 0)) != 16:
                raise SystemExit("Step15 v6 hidden_dim must remain fixed at 16")
            if not bool(policy.get("common_standardizer", {}).get("enabled", False)):
                raise SystemExit("Step15 v6 common final-Phase3 train-only standardizer is disabled")
            lineage_cfg = policy.get("inductive_feature_lineage", {})
            if not bool(lineage_cfg.get("enabled", False)) or lineage_cfg.get("reference_scope") != "frozen_train_sellers_only":
                raise SystemExit("Step15 v6 requires frozen train-seller corpus reference features")
            for pool_name, pool_cfg in policy["pools"].items():
                if not str(pool_cfg.get("pair_features", "")).startswith("reports/step15_v6/features/"):
                    raise SystemExit(f"Step15 v6 pool does not use isolated inductive features: {pool_name}")
            if str(
                (policy.get("evaluation_boundary", {}) or {}).get(
                    "zh_test_evaluation_schedule", ""
                )
            ) != "final_preregistered_endpoint_only":
                raise SystemExit("Step15 v6 must evaluate zh_test only at preregistered endpoints")
            strict_features = set(policy["feature_sets"]["strict_clean_30d"])
            forbidden = set(policy.get("forbidden_strict_clean_features", []))
            leakage = sorted(strict_features & forbidden)
            if leakage:
                raise SystemExit(f"Step15 v6 strict-clean feature contract failed: {leakage}")
            if len(strict_features) != 30:
                raise SystemExit(f"Step15 v6 strict-clean feature count must be 30, got {len(strict_features)}")
            for selection_name, selection_cfg in policy.get("validation_only_model_selection", {}).items():
                candidates = [str(value) for value in selection_cfg.get("candidate_experiments", [])]
                tie_order = [str(value) for value in selection_cfg.get("tie_break_order", [])]
                if not candidates or tie_order != candidates:
                    raise SystemExit(
                        f"Step15 v6 validation selection lacks preregistered simplicity tie-break: {selection_name}"
                    )
            summary_path = str(policy.get("outputs", {}).get("summary_json", ""))
            if not summary_path.startswith("reports/step15_v6/"):
                raise SystemExit(f"Step15 v6 outputs are not isolated: {summary_path}")
            matched_pairs = [
                (
                    "step15_v6_m2b_matched_budget_full_data_replay",
                    "step15_v6_m3_warm_start_curriculum",
                ),
                (
                    "step15_v6_m4c_matched_continuation_no_mixup",
                    "step15_v6_m4_trusted_positive_mixup",
                ),
            ]
            for control_name, treatment_name in matched_pairs:
                control_cfg = policy["experiments"][control_name]
                treatment_cfg = policy["experiments"][treatment_name]
                if control_cfg.get("phase_ids") != treatment_cfg.get("phase_ids"):
                    raise SystemExit(
                        f"Step15 v6 matched comparison has unequal phase plans: {control_name} vs {treatment_name}"
                    )
                control_budget = control_cfg.get("fixed_update_budget_per_phase")
                treatment_budget = treatment_cfg.get("fixed_update_budget_per_phase")
                if control_budget != treatment_budget or int(control_budget or 0) <= 0:
                    raise SystemExit(
                        f"Step15 v6 matched comparison has unequal fixed update budgets: "
                        f"{control_name}={control_budget}, {treatment_name}={treatment_budget}"
                    )
            curriculum_cfg = policy["experiments"]["step15_v6_m3_warm_start_curriculum"]
            curriculum_total_budget = int(curriculum_cfg["fixed_update_budget_per_phase"]) * len(
                curriculum_cfg["phase_ids"]
            )
            for all_at_once_name in (
                "step15_v6_m0_all_at_once_binary",
                "step15_v6_m1_evidence_weighted",
                "step15_v6_m2_domain_balanced",
            ):
                all_at_once_cfg = policy["experiments"][all_at_once_name]
                all_at_once_total_budget = int(
                    all_at_once_cfg.get("fixed_update_budget_per_phase") or 0
                ) * len(all_at_once_cfg.get("phase_ids", []))
                if all_at_once_total_budget != curriculum_total_budget:
                    raise SystemExit(
                        "Step15 v6 all-at-once/curriculum total-budget mismatch: "
                        f"{all_at_once_name}={all_at_once_total_budget}, "
                        f"step15_v6_m3_warm_start_curriculum={curriculum_total_budget}"
                    )
            for experiment in experiments:
                cfg = policy["experiments"][experiment]
                if not bool(cfg.get("component_aware_identity_loss", False)):
                    raise SystemExit(f"Step15 v6 experiment lacks component-aware weighting: {experiment}")
                if cfg.get("weight_pipeline_version") != "v6_component_row_class_evidence_domain":
                    raise SystemExit(f"Step15 v6 weight pipeline mismatch: {experiment}")
                unknown = sorted(set(cfg.get("phase_ids", [])) - set(phase_by_id))
                if unknown:
                    raise SystemExit(f"Step15 v6 experiment has unknown phases: {experiment} {unknown}")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "validate_config_only",
                    "policy": str(policy_path.relative_to(ROOT)),
                    "policy_version": policy.get("version"),
                    "experiments": experiments,
                    "phases": requested_phase_ids,
                    "seeds": seeds,
                    "summary_json": policy.get("outputs", {}).get("summary_json"),
                },
                indent=2,
            )
        )
        return

    inductive_feature_manifest = validate_inductive_feature_lineage(policy_path, policy)
    input_manifest = build_input_manifest(policy_path, policy)
    if inductive_feature_manifest is not None:
        input_manifest["inductive_feature_manifest_sha256"] = inductive_feature_manifest["manifest_sha256"]
    summary_path = resolve_path(policy["outputs"]["summary_json"])
    summary_write_mode = str(policy.get("outputs", {}).get("summary_write_mode", "replace"))
    previous_summary = None
    if summary_write_mode == "merge_by_experiment_phase_seed_same_input_manifest_only" and summary_path.exists():
        previous_summary = step7.load_json(summary_path)
        if previous_summary.get("policy_version") != policy.get("version"):
            raise ValueError(
                "Refusing to start Step15 training because the existing summary uses a different policy version; use a new versioned output path"
            )
        previous_manifest_sha = (previous_summary.get("input_manifest") or {}).get("manifest_sha256")
        if previous_manifest_sha != input_manifest.get("manifest_sha256"):
            raise ValueError(
                "Refusing to start Step15 training because the existing summary uses a different code/data input manifest; use a new versioned output path"
            )
    elif summary_write_mode not in {"replace", "merge_by_experiment_phase_seed_same_input_manifest_only"}:
        raise ValueError(f"Unsupported Step15 summary_write_mode: {summary_write_mode}")

    rows_by_pool = {pool_name: load_pool(pool_name, pool_cfg) for pool_name, pool_cfg in policy["pools"].items()}
    def should_evaluate_test(experiment_name: str, phase_id: str) -> bool:
        selected_phase_ids = [
            str(phase["phase_id"])
            for phase in experiment_phase_plans[experiment_name]
        ]
        return should_evaluate_test_endpoint(
            experiment_name,
            phase_id,
            selected_phase_ids,
            policy,
            validation_only=args.validation_only,
        )

    common_standardizers: dict[str, dict] = {}
    common_standardizer_cfg = policy.get("common_standardizer", {})
    if bool(common_standardizer_cfg.get("enabled", False)):
        reference_experiment_name = str(common_standardizer_cfg["reference_experiment"])
        reference_phase_id = str(common_standardizer_cfg["reference_phase_id"])
        if reference_experiment_name not in policy["experiments"]:
            raise ValueError(f"Unknown common-standardizer reference experiment: {reference_experiment_name}")
        if reference_phase_id not in phase_by_id:
            raise ValueError(f"Unknown common-standardizer reference phase: {reference_phase_id}")
        reference_cfg = policy["experiments"][reference_experiment_name]
        for feature_set_name in sorted({policy["experiments"][name]["feature_set"] for name in experiments}):
            feature_names = policy["feature_sets"][feature_set_name]
            validate_features(rows_by_pool, feature_names)
            scaler_reference_cfg = dict(reference_cfg)
            scaler_reference_cfg["feature_set"] = feature_set_name
            scaler_reference_cfg["training_label_scope"] = str(
                common_standardizer_cfg.get("training_label_scope", "gold_plus_all_silver")
            )
            reference_rows = select_train_rows(
                rows_by_pool,
                scaler_reference_cfg,
                phase_by_id[reference_phase_id],
                policy,
            )
            if not reference_rows:
                raise ValueError(f"No rows for common Step15 standardizer feature set: {feature_set_name}")
            common_standardizers[feature_set_name] = fit_standardizer_bundle(
                reference_rows,
                feature_names,
                policy.get("feature_preprocessing", {}).get(feature_set_name, {}),
            )

    runs = []
    for experiment_name in experiments:
        experiment_cfg = policy["experiments"][experiment_name]
        experiment_phases = experiment_phase_plans[experiment_name]
        shared_standardizer = common_standardizers.get(experiment_cfg["feature_set"])
        training_mode = str(experiment_cfg.get("training_mode", "from_scratch_each_phase"))
        if training_mode == "warm_start_curriculum":
            feature_names = policy["feature_sets"][experiment_cfg["feature_set"]]
            validate_features(rows_by_pool, feature_names)
            final_phase_cfg = experiment_phases[-1]
            final_train_rows = select_train_rows(rows_by_pool, experiment_cfg, final_phase_cfg, policy)
            if not final_train_rows:
                raise ValueError(f"{experiment_name}/{final_phase_cfg['phase_id']} has no final-phase train rows")
            warm_start_standardizer = shared_standardizer or fit_standardizer_bundle(
                final_train_rows,
                feature_names,
                policy.get("feature_preprocessing", {}).get(experiment_cfg["feature_set"], {}),
            )
            for seed in seeds:
                initial_params = None
                for phase_cfg in experiment_phases:
                    run_record, initial_params = run_single(
                        experiment_name,
                        experiment_cfg,
                        phase_cfg,
                        int(seed),
                        policy,
                        rows_by_pool,
                        initial_params=initial_params,
                        standardizer_override=warm_start_standardizer,
                        evaluate_test=should_evaluate_test(
                            experiment_name, str(phase_cfg["phase_id"])
                        ),
                    )
                    runs.append(run_record)
        elif training_mode == "from_scratch_each_phase":
            for phase_cfg in experiment_phases:
                for seed in seeds:
                    run_record, _ = run_single(
                        experiment_name,
                        experiment_cfg,
                        phase_cfg,
                        int(seed),
                        policy,
                        rows_by_pool,
                        standardizer_override=shared_standardizer,
                        evaluate_test=should_evaluate_test(
                            experiment_name, str(phase_cfg["phase_id"])
                        ),
                    )
                    runs.append(run_record)
        else:
            raise ValueError(f"Unsupported Step15 training_mode for {experiment_name}: {training_mode}")

    merged_runs = list(runs)
    merged_experiments = list(experiments)
    merged_seeds = [int(seed) for seed in seeds]
    merged_phases = [phase["phase_id"] for phase in phases]
    merged_standardizers = dict(common_standardizers)
    if summary_write_mode == "merge_by_experiment_phase_seed_same_input_manifest_only" and previous_summary:
        previous = previous_summary
        keyed_runs = {
            (str(run["experiment_name"]), str(run["phase_id"]), int(run["seed"])): run
            for run in previous.get("runs", [])
        }
        for run in runs:
            keyed_runs[(str(run["experiment_name"]), str(run["phase_id"]), int(run["seed"]))] = run
        merged_runs = [keyed_runs[key] for key in sorted(keyed_runs)]
        merged_experiments = sorted(set(previous.get("experiments", [])) | set(experiments))
        merged_seeds = sorted({int(seed) for seed in previous.get("seeds", [])} | {int(seed) for seed in seeds})
        merged_phases = sorted(
            set(previous.get("phases", [])) | {phase["phase_id"] for phase in phases},
            key=lambda phase_id: int(phase_by_id[phase_id]["phase_index"]),
        )
        merged_standardizers = {
            **(previous.get("standardizer_bundles") or {}),
            **common_standardizers,
        }

    validation_selections = select_validation_only_candidates(merged_runs, policy)
    if not args.validation_only:
        merged_runs = materialize_validation_selected_test_predictions(
            merged_runs,
            validation_selections,
            policy,
            rows_by_pool,
        )

    summary = {
        "step": "step15_train_incremental_hard_negative",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy.get("version"),
        "experiments": merged_experiments,
        "seeds": merged_seeds,
        "phases": merged_phases,
        "summary_write_mode": summary_write_mode,
        "baseline_references": policy.get("baseline_references", {}),
        "hard_rule_status": {
            "step5_files_modified": False,
            "fixed_zh_valid_test": True,
            "uncertain_rows_used_for_identity_training": False,
            "synthetic_rows_train_only": True,
            "step11_cluster_decisions_used_as_same_controller_ground_truth": False,
            "zh_test_role": policy.get("evaluation_boundary", {}).get(
                "zh_test_role",
                "fixed_internal_development_test",
            ),
        },
        "input_manifest": input_manifest,
        "validation_only_model_selection": validation_selections,
        "standardizer_bundles": merged_standardizers,
        "runs": merged_runs,
        "run_summary": summarize_runs(merged_runs),
    }
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path.relative_to(ROOT)),
                "new_run_count": len(runs),
                "merged_run_count": len(merged_runs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
