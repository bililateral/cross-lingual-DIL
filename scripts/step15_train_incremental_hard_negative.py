from __future__ import annotations

import argparse
import copy
import json
import math
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
    return train_rows


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


def domain_balanced_binary_weights(y: np.ndarray, rows: list[dict]) -> np.ndarray:
    weights = balanced_binary_weights(y)
    domain_counts = Counter(str(row.get("step15_pool", "")) for row in rows)
    present_domains = [domain for domain, count in domain_counts.items() if count > 0]
    if not present_domains:
        return weights
    total = max(len(rows), 1)
    domain_factor = {
        domain: total / (len(present_domains) * max(domain_counts[domain], 1))
        for domain in present_domains
    }
    return weights * np.asarray([domain_factor.get(str(row.get("step15_pool", "")), 1.0) for row in rows], dtype=float)


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


def evidence_weights(y_evidence: np.ndarray, evidence_type_count: int) -> np.ndarray:
    weights = np.zeros(len(y_evidence), dtype=float)
    mask = y_evidence >= 0
    if not np.any(mask):
        return weights
    counts = Counter(int(value) for value in y_evidence[mask])
    total = int(np.sum(mask))
    for idx, value in enumerate(y_evidence):
        if value < 0:
            continue
        weights[idx] = total / (max(evidence_type_count, 1) * max(counts[int(value)], 1))
    return weights


def add_positive_mixup(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_evidence: np.ndarray,
    rows: list[dict],
    cfg: dict,
    rng: np.random.Generator,
    scope_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    scope = str(scope_override or cfg.get("scope", "all_positive"))
    pos_indices = np.where(y_train == 1.0)[0]
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
    elif scope not in {"all_positive", "same_evidence_type_only"}:
        raise ValueError(f"Unsupported Step15 positive_mixup scope: {scope}")
    if len(pos_indices) < 2:
        return x_train, y_train, y_evidence, rows, 0
    multiplier = float(cfg.get("multiplier", 0.0))
    synthetic_count = int(math.ceil(len(pos_indices) * multiplier))
    if synthetic_count <= 0:
        return x_train, y_train, y_evidence, rows, 0
    alpha = float(cfg.get("beta_alpha", 0.4))
    synthetic_x = []
    synthetic_rows = []
    for _ in range(synthetic_count):
        if scope == "same_evidence_type_only":
            evidence_groups: dict[int, list[int]] = defaultdict(list)
            for idx in pos_indices:
                evidence_groups[int(y_evidence[int(idx)])].append(int(idx))
            eligible_groups = [indices for evidence, indices in evidence_groups.items() if evidence >= 0 and len(indices) >= 2]
            if not eligible_groups:
                return x_train, y_train, y_evidence, rows, 0
            group_indices = eligible_groups[int(rng.integers(0, len(eligible_groups)))]
            left, right = rng.choice(np.asarray(group_indices, dtype=int), size=2, replace=False)
        else:
            left, right = rng.choice(pos_indices, size=2, replace=False)
        lam = float(rng.beta(alpha, alpha))
        synthetic_x.append((1.0 - lam) * x_train[left] + lam * x_train[right])
        left_row = rows[int(left)]
        right_row = rows[int(right)]
        left_pool = str(left_row.get("step15_pool", ""))
        right_pool = str(right_row.get("step15_pool", ""))
        synthetic_pool = left_pool if left_pool == right_pool else "cross_domain_mixup"
        synthetic_rows.append(
            {
                "pair_uid": f"synthetic_mixup::{len(synthetic_rows)}",
                "review_label": "positive",
                "evidence_type": "synthetic_train_only",
                "step15_pool": synthetic_pool,
                "synthetic_train_only": "1",
            }
        )
    x_aug = np.vstack([x_train, np.asarray(synthetic_x, dtype=float)])
    y_aug = np.concatenate([y_train, np.ones(synthetic_count, dtype=float)])
    evidence_aug = np.concatenate([y_evidence, np.full(synthetic_count, -1, dtype=int)])
    return x_aug, y_aug, evidence_aug, rows + synthetic_rows, synthetic_count


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
    identity_weight_multipliers: dict[str, float] | None = None,
    normalize_identity_weight_mean: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(seed)
    hidden_dim = int(train_cfg["hidden_dim"])
    params = copy.deepcopy(initial_params) if initial_params is not None else init_params(x_train.shape[1], hidden_dim, evidence_type_count, rng)
    state = {
        "m": {key: np.zeros_like(value) for key, value in params.items()},
        "v": {key: np.zeros_like(value) for key, value in params.items()},
    }

    if domain_balanced_identity_loss:
        identity_weights = domain_balanced_binary_weights(y_train, train_rows)
    elif train_cfg.get("class_balanced_identity_loss", True):
        identity_weights = balanced_binary_weights(y_train)
    else:
        identity_weights = np.ones(len(y_train))
    identity_weights, identity_weight_multiplier_diagnostics = apply_identity_weight_multipliers(
        identity_weights,
        train_rows,
        identity_weight_multipliers or {},
        normalize_mean=normalize_identity_weight_mean,
    )
    ev_weights = (
        evidence_weights(y_evidence, evidence_type_count)
        if train_cfg.get("class_balanced_evidence_loss", True)
        else np.where(y_evidence >= 0, 1.0, 0.0)
    )

    max_epochs = int(train_cfg["max_epochs"])
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
        if epochs_without_improvement >= patience:
            break

    diagnostics = {
        "best_epoch": int(best_epoch),
        "trained_epoch_count": int(epoch),
        "early_stopping_metric": metric_name,
        "best_valid_metric": round(float(best_metric), 6),
        "last_loss": {key: round(float(value), 6) for key, value in last_loss.items()},
        "lambda_evidence": round(float(lambda_evidence), 6),
        "initialization": "warm_start" if initial_params is not None else "random",
        "domain_balanced_identity_loss": bool(domain_balanced_identity_loss),
        "identity_weight_multipliers": identity_weight_multiplier_diagnostics,
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
                "prob_positive": round(float(probability), 6),
                "pred_positive": int(prediction),
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


def artifact_payload(
    params: dict[str, np.ndarray],
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    evidence_types: list[str],
    diagnostics: dict,
) -> dict:
    return {
        "feature_names": feature_names,
        "feature_means": [round(float(value), 10) for value in means],
        "feature_stds": [round(float(value), 10) for value in stds],
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
    standardizer_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    feature_names = policy["feature_sets"][experiment_cfg["feature_set"]]
    validate_features(rows_by_pool, feature_names)

    train_rows = select_train_rows(rows_by_pool, experiment_cfg, phase_cfg, policy)
    valid_rows = select_eval_rows(rows_by_pool["zh_target_strict"], policy["splits"]["target_valid"])
    test_rows = select_eval_rows(rows_by_pool["zh_target_strict"], policy["splits"]["target_test"])
    if not train_rows:
        raise ValueError(f"{experiment_name}/{phase_cfg['phase_id']} has no train rows")
    if not valid_rows or not test_rows:
        raise ValueError("Step15 requires non-empty fixed zh_valid and zh_test rows")

    x_train_raw = rows_to_feature_matrix(train_rows, feature_names)
    x_valid_raw = rows_to_feature_matrix(valid_rows, feature_names)
    x_test_raw = rows_to_feature_matrix(test_rows, feature_names)
    if standardizer_override is None:
        means, stds = fit_standardizer(x_train_raw)
        standardizer_source = "current_phase_train"
    else:
        means, stds = standardizer_override
        standardizer_source = "warm_start_final_phase_train"
    x_train = apply_standardizer(x_train_raw, means, stds)
    x_valid = apply_standardizer(x_valid_raw, means, stds)
    x_test = apply_standardizer(x_test_raw, means, stds)
    y_train = y_from_rows(train_rows)
    y_valid = y_from_rows(valid_rows)
    y_test = y_from_rows(test_rows)
    y_ev = evidence_indices(train_rows, policy["evidence_types"])
    train_rows_for_weights = list(train_rows)

    synthetic_count = 0
    positive_synthetic_count = 0
    negative_synthetic_count = 0
    if bool(phase_cfg.get("use_positive_mixup", False)):
        x_train, y_train, y_ev, train_rows_for_weights, positive_synthetic_count = add_positive_mixup(
            x_train,
            y_train,
            y_ev,
            train_rows,
            policy["training"].get("positive_mixup", {}),
            np.random.default_rng(seed + 7919),
            experiment_cfg.get("positive_mixup_scope_override"),
        )
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
        identity_weight_multipliers=experiment_cfg.get("identity_evidence_type_weight_multipliers", {}),
        normalize_identity_weight_mean=bool(experiment_cfg.get("normalize_identity_weight_mean", True)),
    )
    diagnostics["training_mode"] = str(experiment_cfg.get("training_mode", "from_scratch_each_phase"))
    diagnostics["positive_mixup_scope"] = str(
        experiment_cfg.get("positive_mixup_scope_override")
        or policy["training"].get("positive_mixup", {}).get("scope", "all_positive")
    )
    diagnostics["standardizer_source"] = standardizer_source

    _, valid_prob, valid_evidence_prob = forward(params, x_valid)
    _, test_prob, test_evidence_prob = forward(params, x_test)
    threshold = step7.choose_threshold(y_valid, valid_prob, policy["threshold_selection"]["metric"], policy)
    valid_metrics = step7.evaluate_probabilities(y_valid, valid_prob, threshold)
    test_metrics = step7.evaluate_probabilities(y_test, test_prob, threshold)

    experiment_token = f"{experiment_name}_{phase_cfg['phase_id']}_seed_{seed}"
    valid_predictions = prediction_rows(valid_rows, valid_prob, threshold, experiment_token)
    test_predictions = prediction_rows(test_rows, test_prob, threshold, experiment_token)

    valid_path = output_path(policy["outputs"]["zh_valid_predictions_template"], experiment_name, phase_cfg["phase_id"], seed)
    test_path = output_path(policy["outputs"]["zh_test_predictions_template"], experiment_name, phase_cfg["phase_id"], seed)
    artifact_path = output_path(policy["outputs"]["artifact_template"], experiment_name, phase_cfg["phase_id"], seed)
    step7.write_csv(valid_path, valid_predictions, list(valid_predictions[0].keys()))
    step7.write_csv(test_path, test_predictions, list(test_predictions[0].keys()))

    artifact = artifact_payload(params, feature_names, means, stds, policy["evidence_types"], diagnostics)
    artifact["feature_importance"] = feature_importance(params, feature_names)
    step7.write_json(artifact_path, artifact)

    run_record = {
        "experiment_name": experiment_name,
        "phase_id": phase_cfg["phase_id"],
        "phase_index": int(phase_cfg["phase_index"]),
        "seed": int(seed),
        "role": experiment_cfg.get("role", ""),
        "feature_set": experiment_cfg["feature_set"],
        "use_identifier_features": bool(experiment_cfg.get("use_identifier_features", False)),
        "included_evidence_types": phase_cfg["included_evidence_types"],
        "synthetic_train_only_mixup_count": int(synthetic_count),
        "synthetic_train_only_positive_mixup_count": int(positive_synthetic_count),
        "synthetic_train_only_negative_mixup_count": int(negative_synthetic_count),
        "train_dataset": summarize_rows(train_rows),
        "zh_valid_dataset": summarize_rows(valid_rows),
        "zh_test_dataset": summarize_rows(test_rows),
        "training_diagnostics": diagnostics,
        "zh_valid_metrics": valid_metrics,
        "zh_test_metrics": test_metrics,
        "output_paths": {
            "artifact": str(artifact_path.relative_to(ROOT)),
            "zh_valid_predictions": str(valid_path.relative_to(ROOT)),
            "zh_test_predictions": str(test_path.relative_to(ROOT)),
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
        auc_values = [item["zh_test_metrics"]["roc_auc"] for item in items if item["zh_test_metrics"]["roc_auc"] is not None]
        ap_values = [
            item["zh_test_metrics"]["average_precision"]
            for item in items
            if item["zh_test_metrics"]["average_precision"] is not None
        ]
        summary[key] = {
            "run_count": len(items),
            "seeds": [item["seed"] for item in items],
            "zh_test_roc_auc_mean": round(float(np.mean(auc_values)), 6) if auc_values else None,
            "zh_test_roc_auc_min": round(float(np.min(auc_values)), 6) if auc_values else None,
            "zh_test_roc_auc_max": round(float(np.max(auc_values)), 6) if auc_values else None,
            "zh_test_average_precision_mean": round(float(np.mean(ap_values)), 6) if ap_values else None,
            "zh_test_average_precision_min": round(float(np.min(ap_values)), 6) if ap_values else None,
            "zh_test_average_precision_max": round(float(np.max(ap_values)), 6) if ap_values else None,
        }
    return summary


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
    phases = [phase_by_id[phase_id] for phase_id in (args.phases or list(phase_by_id.keys()))]

    rows_by_pool = {pool_name: load_pool(pool_name, pool_cfg) for pool_name, pool_cfg in policy["pools"].items()}

    runs = []
    for experiment_name in experiments:
        if experiment_name not in policy["experiments"]:
            raise SystemExit(f"Unknown Step15 experiment: {experiment_name}")
        experiment_cfg = policy["experiments"][experiment_name]
        training_mode = str(experiment_cfg.get("training_mode", "from_scratch_each_phase"))
        if training_mode == "warm_start_curriculum":
            feature_names = policy["feature_sets"][experiment_cfg["feature_set"]]
            validate_features(rows_by_pool, feature_names)
            final_phase_cfg = phases[-1]
            final_train_rows = select_train_rows(rows_by_pool, experiment_cfg, final_phase_cfg, policy)
            if not final_train_rows:
                raise ValueError(f"{experiment_name}/{final_phase_cfg['phase_id']} has no final-phase train rows")
            final_x_train_raw = rows_to_feature_matrix(final_train_rows, feature_names)
            warm_start_standardizer = fit_standardizer(final_x_train_raw)
            for seed in seeds:
                initial_params = None
                for phase_cfg in phases:
                    run_record, initial_params = run_single(
                        experiment_name,
                        experiment_cfg,
                        phase_cfg,
                        int(seed),
                        policy,
                        rows_by_pool,
                        initial_params=initial_params,
                        standardizer_override=warm_start_standardizer,
                    )
                    runs.append(run_record)
        elif training_mode == "from_scratch_each_phase":
            for phase_cfg in phases:
                for seed in seeds:
                    run_record, _ = run_single(experiment_name, experiment_cfg, phase_cfg, int(seed), policy, rows_by_pool)
                    runs.append(run_record)
        else:
            raise ValueError(f"Unsupported Step15 training_mode for {experiment_name}: {training_mode}")

    summary = {
        "step": "step15_train_incremental_hard_negative",
        "policy": str(policy_path.relative_to(ROOT)),
        "policy_version": policy.get("version"),
        "experiments": experiments,
        "seeds": seeds,
        "phases": [phase["phase_id"] for phase in phases],
        "baseline_references": policy.get("baseline_references", {}),
        "hard_rule_status": {
            "step5_files_modified": False,
            "fixed_zh_valid_test": True,
            "uncertain_rows_used_for_identity_training": False,
            "synthetic_rows_train_only": True,
            "step11_cluster_decisions_used_as_same_controller_ground_truth": False,
        },
        "runs": runs,
        "run_summary": summarize_runs(runs),
    }
    summary_path = resolve_path(policy["outputs"]["summary_json"])
    step7.write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "run_count": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
