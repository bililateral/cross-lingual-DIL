#!/usr/bin/env python3
"""Shared contracts and numerical helpers for Step25-v3."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import step15_v7_common as v7_common
import step24_common as step24
import step25_common as step25_v1
import step25_v2_common as step25_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step25_v3_copy_aware_dual_channel_policy.json"


def resolve(value: str | Path) -> Path:
    return step24.resolve(value)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(
    path: str | Path = DEFAULT_POLICY,
) -> tuple[Path, dict, dict, dict, dict, dict]:
    policy_path = resolve(path)
    policy = load_json(policy_path)
    step15_v7_policy = load_json(resolve(policy["inputs"]["step15_v7_policy"]))
    step24_policy = load_json(resolve(policy["inputs"]["step24_policy"]))
    step25_v1_policy = load_json(resolve(policy["inputs"]["step25_v1_policy"]))
    step25_v2_policy = load_json(resolve(policy["inputs"]["step25_v2_policy"]))
    step24.validate_policy(step24_policy)
    step25_v1.validate_policy(step25_v1_policy, step24_policy)
    step25_v2.validate_policy(step25_v2_policy, step24_policy, step25_v1_policy)
    validate_policy(
        policy,
        step15_v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    )
    return (
        policy_path,
        policy,
        step15_v7_policy,
        step24_policy,
        step25_v1_policy,
        step25_v2_policy,
    )


def validate_policy(
    policy: dict,
    step15_v7_policy: dict,
    step24_policy: dict,
    step25_v1_policy: dict,
    step25_v2_policy: dict,
) -> None:
    boundary = policy["boundary"]
    if not boundary["hypothesis_informed_by_step25_v1_v2"]:
        raise ValueError("Step25-v3 must remain explicitly retrospective on D0")
    for key in (
        "valid_or_test_read_forbidden",
        "parameter_or_threshold_search_on_d0_forbidden",
        "publication_promotion_hard_false",
        "step11_or_step17_entry_forbidden",
    ):
        if boundary.get(key) is not True:
            raise ValueError(f"Step25-v3 boundary constraint was relaxed: {key}")
    roots = {
        policy["outputs_root"],
        policy["inputs"]["step24_outputs_root"],
        policy["inputs"]["step25_v1_outputs_root"],
        policy["inputs"]["step25_v2_outputs_root"],
    }
    if len(roots) != 4:
        raise ValueError("Step25-v3 output root must be isolated from every parent")
    if policy["inputs"]["step24_outputs_root"] != step24_policy["outputs_root"]:
        raise ValueError("Step25-v3 Step24 root disagrees with the frozen policy")
    if policy["inputs"]["step25_v1_outputs_root"] != step25_v1_policy["outputs_root"]:
        raise ValueError("Step25-v3 Step25-v1 root disagrees with the frozen policy")
    if policy["inputs"]["step25_v2_outputs_root"] != step25_v2_policy["outputs_root"]:
        raise ValueError("Step25-v3 Step25-v2 root disagrees with the frozen policy")

    feature_contract = policy["feature_contract"]
    for key in (
        "clean_scorer_identifier_features_forbidden",
        "candidate_rule_features_forbidden",
        "review_label_or_evidence_type_as_feature_forbidden",
        "valid_or_test_fitted_statistics_forbidden",
        "missing_clean_style_encoded_as_zero_forbidden",
    ):
        if feature_contract.get(key) is not True:
            raise ValueError(f"Step25-v3 feature isolation was relaxed: {key}")
    if feature_contract["unreliable_pair_local_clean_value"] != "raw_style_fallback":
        raise ValueError("Step25-v3 unreliable local style must fall back to raw style")
    if float(feature_contract["unreliable_pair_local_delta_value"]) != 0.0:
        raise ValueError("Step25-v3 unreliable raw-clean residual must be zero")

    groups = {
        "raw_channel": feature_contract["raw_channel"],
        "clean_channel": feature_contract["clean_channel"],
        "copy_residual_channel": feature_contract["copy_residual_channel"],
        "copy_risk_channel": feature_contract["copy_risk_channel"],
        "semantic_sensitivity_channel": feature_contract["semantic_sensitivity_channel"],
        "pair_local_reliability_only": ["pair_local_style_reliable"],
    }
    all_names = [name for values in groups.values() for name in values]
    if any("identifier" in name and name != "identifier_redacted_e5_cosine" for name in all_names):
        raise ValueError("Step25-v3 clean feature group contains an identifier feature")
    specs = policy["evaluation"]["model_specs"]
    expected_specs = {
        "C0_matched_raw_style",
        "C1_raw_plus_clean_no_copy_penalty",
        "C2_copy_aware_dual_channel_primary",
        "C3_semantic_plus_copy_aware_sensitivity",
    }
    if set(specs) != expected_specs:
        raise ValueError("Step25-v3 fixed C0-C3 matrix changed")
    if policy["evaluation"]["primary_model"] != "C2_copy_aware_dual_channel_primary":
        raise ValueError("Step25-v3 primary must remain C2")
    if policy["evaluation"]["matched_baseline_model"] != "C0_matched_raw_style":
        raise ValueError("Step25-v3 matched baseline must remain C0")
    if specs["C3_semantic_plus_copy_aware_sensitivity"]["role"] != "sensitivity_only":
        raise ValueError("Step25-v3 semantic model must remain sensitivity-only")
    for model_name, spec in specs.items():
        feature_names = expand_feature_groups(spec["feature_groups"], groups)
        directions = spec["coefficient_directions"]
        if set(directions) != set(feature_names):
            raise ValueError(f"Step25-v3 directions do not cover features: {model_name}")
        if any(value not in {"nonnegative", "nonpositive", "unconstrained"} for value in directions.values()):
            raise ValueError(f"Step25-v3 has an unknown coefficient direction: {model_name}")
    primary_directions = specs["C2_copy_aware_dual_channel_primary"][
        "coefficient_directions"
    ]
    for name in feature_contract["copy_residual_channel"]:
        if primary_directions[name] != "nonpositive":
            raise ValueError("Step25-v3 copy residual can only be nonpositive")
    for name in feature_contract["copy_risk_channel"]:
        if name.endswith("reliable"):
            continue
        if primary_directions[name] != "nonpositive":
            raise ValueError("Step25-v3 copy-risk coefficient can only be nonpositive")

    evaluation = policy["evaluation"]
    if evaluation["canonical_split"] != "train" or int(evaluation["fold_count"]) != 5:
        raise ValueError("Step25-v3 must use canonical train and five grouped folds")
    logistic = evaluation["logistic"]
    if (
        logistic["solver"] != "projected_gradient_logistic_l2_with_backtracking"
        or float(logistic["l2_penalty"]) != 10.0
        or logistic["class_weight"] != "none"
        or logistic["standardize_features"] is not True
    ):
        raise ValueError("Step25-v3 fixed constrained LR/L2 contract changed")
    if policy["missingness_closure_control"][
        "pair_local_reliability_intersection_forbidden"
    ] is not True:
        raise ValueError("Step25-v3 missingness closure cannot use local intersection")
    operational = policy["operational_identifier_control"]
    if operational["training_domain"] != "en_only" or operational[
        "chinese_labels_for_expert_training_forbidden"
    ] is not True:
        raise ValueError("Step25-v3 operational expert must remain English-only")
    if policy["d0_to_d1_replication_gates"]["all_gates_required"] is not True:
        raise ValueError("Step25-v3 cannot relax all-gates-required promotion to D1")
    if step15_v7_policy["factorized_evidence_weighting"]["forbid_global_eight_x_multiplier"] is not True:
        raise ValueError("Step25-v3 requires the corrected factorized evidence weights")
    outputs = policy["outputs"]
    if len(outputs) != len(set(outputs.values())):
        raise ValueError("Step25-v3 output map contains a collision")


def feature_groups(policy: dict) -> dict[str, list[str]]:
    contract = policy["feature_contract"]
    return {
        "raw_channel": list(contract["raw_channel"]),
        "clean_channel": list(contract["clean_channel"]),
        "copy_residual_channel": list(contract["copy_residual_channel"]),
        "copy_risk_channel": list(contract["copy_risk_channel"]),
        "semantic_sensitivity_channel": list(contract["semantic_sensitivity_channel"]),
        "pair_local_reliability_only": ["pair_local_style_reliable"],
    }


def expand_feature_groups(names: list[str], groups: dict[str, list[str]]) -> list[str]:
    result: list[str] = []
    for group_name in names:
        if group_name not in groups:
            raise ValueError(f"Unknown Step25-v3 feature group: {group_name}")
        for feature_name in groups[group_name]:
            if feature_name not in result:
                result.append(feature_name)
    return result


def model_feature_names(policy: dict, model_name: str) -> list[str]:
    spec = policy["evaluation"]["model_specs"][model_name]
    return expand_feature_groups(spec["feature_groups"], feature_groups(policy))


def load_rows(
    step25_v2_policy: dict,
    step24_policy: dict,
    step25_v1_policy: dict,
) -> dict[str, list[dict]]:
    rows = step25_v2.load_rows(step25_v2_policy, step24_policy, step25_v1_policy)
    for pool_rows in rows.values():
        if any(row.get("split_name") != "train" for row in pool_rows):
            raise ValueError("Step25-v3 encountered a non-train canonical row")
    return rows


def require_parent_manifests(policy: dict) -> dict:
    parents = {}
    for key, root_key, manifest_name in (
        ("step25_v1", "step25_v1_outputs_root", "step25_sync_manifest.json"),
        ("step25_v2", "step25_v2_outputs_root", "step25_v2_sync_manifest.json"),
    ):
        path = resolve(policy["inputs"][root_key]) / manifest_name
        if not path.is_file():
            raise FileNotFoundError(f"Step25-v3 frozen parent manifest is missing: {path}")
        payload = load_json(path)
        if payload.get("publication_promotion_eligible") is not False:
            raise ValueError(f"Step25-v3 parent manifest carries an invalid promotion: {path}")
        parents[key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": step24.sha256_file(path),
            "manifest_sha256": payload.get("manifest_sha256"),
        }
    return parents


def frozen_parent_references(policy: dict) -> dict:
    specifications = {
        "step25_v1": {
            "path_key": "step25_v1_evaluation_summary",
            "metric_paths": {
                "target_oof_raw_style_lr_l2_control": (
                    "model_metrics",
                    "target_oof_raw_style_lr_l2_control",
                ),
                "target_oof_decontaminated_style_lr_l2_primary": (
                    "model_metrics",
                    "target_oof_decontaminated_style_lr_l2_primary",
                ),
            },
        },
        "step25_v2": {
            "path_key": "step25_v2_evaluation_summary",
            "metric_paths": {
                "english_grouped_oof_P0_raw_style": (
                    "model_metrics",
                    "english_grouped_oof",
                    "P0_raw_style_matched_missingness",
                ),
                "english_grouped_oof_P2_pair_local_clean": (
                    "model_metrics",
                    "english_grouped_oof",
                    "P2_pair_local_clean_style_matched_missingness",
                ),
                "target_grouped_oof_P0_raw_style": (
                    "model_metrics",
                    "target_grouped_oof",
                    "P0_raw_style_matched_missingness",
                ),
                "target_grouped_oof_P2_pair_local_clean": (
                    "model_metrics",
                    "target_grouped_oof",
                    "P2_pair_local_clean_style_matched_missingness",
                ),
                "target_grouped_oof_P3_raw_fallback": (
                    "model_metrics",
                    "target_grouped_oof",
                    "P3_pair_local_clean_raw_fallback",
                ),
            },
        },
    }
    references = {}
    for parent_name, specification in specifications.items():
        path = resolve(policy["inputs"][specification["path_key"]])
        if not path.is_file():
            raise FileNotFoundError(
                f"Step25-v3 frozen parent evaluation summary is missing: {path}"
            )
        payload = load_json(path)
        if payload.get("status") != "pass":
            raise ValueError(f"Step25-v3 parent evaluation did not pass: {path}")
        if payload.get("publication_promotion_eligible") is not False:
            raise ValueError(
                f"Step25-v3 parent evaluation carries an invalid promotion: {path}"
            )
        metrics = {}
        for metric_name, key_path in specification["metric_paths"].items():
            value = payload
            for key in key_path:
                if key not in value:
                    raise ValueError(
                        "Step25-v3 frozen parent metric is missing: "
                        f"{parent_name}:{metric_name}"
                    )
                value = value[key]
            metrics[metric_name] = value
        references[parent_name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": step24.sha256_file(path),
            "version": payload.get("version"),
            "publication_promotion_eligible": False,
            "metrics": metrics,
        }
    return references


def matrix_for_rows(
    rows: list[dict], feature_index: dict[str, dict], feature_names: list[str]
) -> np.ndarray:
    matrix = []
    for row in rows:
        feature = feature_index.get(row["pair_uid"])
        if feature is None:
            raise ValueError(f"Step25-v3 feature missing for pair: {row['pair_uid']}")
        matrix.append([float(feature[name]) for name in feature_names])
    result = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Step25-v3 feature matrix contains non-finite values")
    return result


def direction_bounds(feature_names: list[str], directions: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(feature_names), -np.inf, dtype=np.float64)
    upper = np.full(len(feature_names), np.inf, dtype=np.float64)
    for index, name in enumerate(feature_names):
        direction = directions[name]
        if direction == "nonnegative":
            lower[index] = 0.0
        elif direction == "nonpositive":
            upper[index] = 0.0
        elif direction != "unconstrained":
            raise ValueError(f"Unknown coefficient direction: {name}:{direction}")
    return lower, upper


def _objective_gradient(
    params: np.ndarray,
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    logits = params[0] + matrix @ params[1:]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits)))
    loss += 0.5 * l2_penalty * float(params[1:] @ params[1:])
    residual = weights * (probabilities - labels)
    gradient = np.empty_like(params)
    gradient[0] = float(np.sum(residual))
    gradient[1:] = matrix.T @ residual + l2_penalty * params[1:]
    return loss, gradient


def fit_direction_constrained_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: list[str],
    directions: dict[str, str],
    cfg: dict,
) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("Step25-v3 constrained logistic shape mismatch")
    if len(feature_names) != matrix.shape[1] or set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Step25-v3 constrained logistic requires aligned two-class data")
    if np.any(sample_weights <= 0.0) or not np.all(np.isfinite(sample_weights)):
        raise ValueError("Step25-v3 constrained logistic weights must be finite and positive")
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    scaled = (matrix - means) / scales
    lower, upper = direction_bounds(feature_names, directions)
    params = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    prevalence = float(np.sum(sample_weights * labels) / np.sum(sample_weights))
    params[0] = math.log(max(prevalence, 1e-6) / max(1.0 - prevalence, 1e-6))
    step_size = 1.0
    converged = False
    final_projected_gradient = math.inf
    previous_loss = math.inf
    for iteration in range(1, int(cfg["max_iter"]) + 1):
        loss, gradient = _objective_gradient(
            params, scaled, labels, sample_weights, float(cfg["l2_penalty"])
        )
        accepted = False
        trial_step = step_size
        while trial_step >= float(cfg["minimum_step_size"]):
            candidate = params - trial_step * gradient
            candidate[1:] = np.minimum(np.maximum(candidate[1:], lower), upper)
            candidate_loss, _ = _objective_gradient(
                candidate, scaled, labels, sample_weights, float(cfg["l2_penalty"])
            )
            displacement = candidate - params
            armijo = loss + float(cfg["armijo_constant"]) * float(gradient @ displacement)
            if candidate_loss <= armijo + 1e-12:
                accepted = True
                break
            trial_step *= float(cfg["backtracking_factor"])
        if not accepted:
            raise ValueError("Step25-v3 projected logistic line search failed")
        params = candidate
        step_size = min(trial_step / max(float(cfg["backtracking_factor"]), 1e-12), 1.0)
        _updated_loss, updated_gradient = _objective_gradient(
            params, scaled, labels, sample_weights, float(cfg["l2_penalty"])
        )
        projected = params - updated_gradient
        projected[1:] = np.minimum(np.maximum(projected[1:], lower), upper)
        final_projected_gradient = float(np.max(np.abs(params - projected)))
        relative_loss = (
            abs(previous_loss - candidate_loss) / max(1.0, abs(previous_loss))
            if math.isfinite(previous_loss)
            else math.inf
        )
        if final_projected_gradient <= float(cfg["tolerance"]) or (
            iteration > 5 and relative_loss <= float(cfg["tolerance"])
        ):
            converged = True
            break
        previous_loss = candidate_loss
    raw_coefficients = params[1:] / scales
    raw_intercept = params[0] - float(means @ raw_coefficients)
    for coefficient, name in zip(raw_coefficients, feature_names, strict=True):
        direction = directions[name]
        if direction == "nonnegative" and coefficient < -1e-10:
            raise ValueError(f"Step25-v3 violated nonnegative direction: {name}")
        if direction == "nonpositive" and coefficient > 1e-10:
            raise ValueError(f"Step25-v3 violated nonpositive direction: {name}")
    return {
        "model_family": "step25_v3_direction_constrained_logistic_l2",
        "feature_names": feature_names,
        "coefficient_directions": directions,
        "standardization": {"means": means.tolist(), "scales": scales.tolist()},
        "parameter_intercept_standardized": float(params[0]),
        "parameter_coefficients_standardized": params[1:].tolist(),
        "parameter_intercept_raw": raw_intercept,
        "parameter_coefficients_raw": raw_coefficients.tolist(),
        "solver_iterations": iteration,
        "solver_converged": converged,
        "solver_final_projected_gradient": final_projected_gradient,
        "l2_penalty": float(cfg["l2_penalty"]),
    }


def apply_direction_constrained_logistic(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    means = np.asarray(artifact["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(artifact["standardization"]["scales"], dtype=np.float64)
    scaled = (matrix - means) / scales
    logits = float(artifact["parameter_intercept_standardized"]) + scaled @ np.asarray(
        artifact["parameter_coefficients_standardized"], dtype=np.float64
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def normalized_factorized_weights(rows: list[dict], weighting_cfg: dict) -> tuple[np.ndarray, dict]:
    weights, summary = v7_common.factorized_evidence_weights(rows, weighting_cfg)
    weights = np.asarray(weights, dtype=np.float64)
    weights *= len(rows) / float(np.sum(weights))
    return weights, summary
