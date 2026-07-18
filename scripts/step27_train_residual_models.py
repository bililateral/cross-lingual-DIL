#!/usr/bin/env python3
"""Train the preregistered Step27 source-offset residual controls.

The English source scorer is the frozen Step24 identifier-redacted E5 LR/L2
artifact.  The primary residual model keeps its source-logit coefficient fixed
at one.  M0 uses real Chinese train rows, M1 adds an equal-effective-weight
duplication control, and M2 replaces those duplicates with recomputed synthetic
views.  Synthetic descendants always follow their real parent component/fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import step27_common as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "schema" / "step27_english_pretrained_synthetic_adaptation_policy.json"
FIXED_SOURCE_ARTIFACT = (
    ROOT
    / "reports"
    / "step24_content_independent_authorship"
    / "v1_20260717"
    / "step24_model_artifacts.json"
)
FIXED_SOURCE_KEY = ("artifacts", "source_only", "e5_lr_l2_control")
MODEL_IDS = (
    "step27_m0_real_only",
    "step27_m1_equal_effective_weight_duplication",
    "step27_m2_synthetic",
)
EXPLORATORY_MODEL_IDS = (
    "step27_m2_learned_source_alpha",
    "step27_m2_target_only_alpha_zero",
)
SENSITIVITY_MODEL_ID = "step27_m2_silver_sensitivity"
REPORTING_MODEL_IDS = (*MODEL_IDS, *EXPLORATORY_MODEL_IDS, SENSITIVITY_MODEL_ID)
DEFAULT_SEEDS = tuple(range(20260320, 20260330))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--materialize-features-only", action="store_true")
    parser.add_argument("--score-valid", action="store_true")
    parser.add_argument("--oof-gate-summary")
    parser.add_argument("--score-internal-test", action="store_true")
    parser.add_argument("--valid-gate-summary")
    return parser.parse_args()


def resolve(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None and len(path.parts) == 1:
        return base / path
    return ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
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
    return completed.stdout.strip() or None


def render_csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Step27 refuses to render an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Incomplete Step27 temporary output exists: {temporary}")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different Step27 artifact: {path}")
        return
    with temporary.open("xb") as handle:
        handle.write(payload)
    temporary.replace(path)


def write_json_immutable(path: Path, value: dict) -> None:
    write_immutable(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
    )


def write_csv_immutable(path: Path, rows: list[dict]) -> None:
    write_immutable(path, render_csv(rows))


def first_value(mapping: dict, paths: Iterable[tuple[str, ...]], default: object = None) -> object:
    for keys in paths:
        value: object = mapping
        found = True
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                found = False
                break
            value = value[key]
        if found:
            return value
    return default


def outputs_root(policy: dict) -> Path:
    value = first_value(
        policy,
        (("outputs_root",), ("outputs", "root"), ("artifacts", "outputs_root")),
        "reports/step27_english_pretrained_synthetic/v1_20260718",
    )
    return resolve(str(value))


def policy_path(policy: dict, key: str, default_relative: str) -> Path:
    value = first_value(
        policy,
        (
            ("inputs", key),
            ("artifacts", key),
            ("outputs", key),
            ("paths", key),
        ),
    )
    if value is None:
        return outputs_root(policy) / default_relative
    return resolve(str(value), outputs_root(policy))


def bool_value(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def numeric(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step27 field {field!r} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Step27 field {field!r} is not finite: {value!r}")
    return result


def row_uid(row: dict) -> str:
    return str(row.get("pair_uid") or row.get("synthetic_pair_uid") or "")


def row_label(row: dict) -> int:
    label = str(row.get("review_label") or row.get("label") or "").strip().casefold()
    if label in {"positive", "1", "true"}:
        return 1
    if label in {"negative", "0", "false"}:
        return 0
    raise ValueError(f"Step27 row has a non-binary label: {row_uid(row)}={label!r}")


def row_component(row: dict) -> str:
    value = str(
        row.get("step27_component_id")
        or row.get("component_id")
        or row.get("recomputed_component_id")
        or row.get("v7_component_id")
        or row.get("parent_component_id")
        or ""
    )
    if not value:
        raise ValueError(f"Step27 row has no seller-component id: {row_uid(row)}")
    return value


def row_fold(row: dict) -> int:
    raw = row.get(
        "step27_fold_id",
        row.get("fold_id", row.get("parent_fold_id", row.get("fold", ""))),
    )
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step27 row has no valid fold id: {row_uid(row)}={raw!r}") from exc


def row_weight(row: dict) -> float:
    value = row.get(
        "synthetic_training_sample_weight",
        row.get("training_sample_weight", row.get("sample_weight", 1.0)),
    )
    weight = numeric(value, "training_sample_weight")
    if weight <= 0.0:
        raise ValueError(f"Step27 row weight must be positive: {row_uid(row)}={weight}")
    return weight


def feature_names(policy: dict, real_rows: list[dict]) -> list[str]:
    configured = first_value(
        policy,
        (
            ("training", "residual_feature_names"),
            ("features", "residual_feature_names"),
            ("residual_model", "feature_names"),
            ("pair_representation", "residual_feature_names"),
        ),
    )
    if configured is not None:
        names = [str(value) for value in configured]
    else:
        names = sorted(field[9:] for field in real_rows[0] if field.startswith("residual_"))
        names = [f"residual_{name}" for name in names]
    if not names or len(names) != len(set(names)):
        raise ValueError("Step27 residual feature names are empty or duplicated")
    forbidden_tokens = ("contact", "identifier", "pgp", "email", "telegram", "wallet", "uppercase")
    source_name = source_feature_name(policy)
    for name in names:
        if name == source_name:
            raise ValueError("The fixed source feature cannot also be a residual feature")
        if any(token in name.casefold() for token in forbidden_tokens):
            raise ValueError(f"Identifier/shortcut residual feature is forbidden: {name}")
    return names


def source_feature_name(policy: dict) -> str:
    return str(
        first_value(
            policy,
            (("training", "source_feature_name"), ("source_model", "feature_name")),
            "identifier_redacted_e5_cosine",
        )
    )


def validate_policy(policy: dict, policy_path_value: Path) -> dict:
    fold_count = int(
        first_value(
            policy,
            (("training", "fold_count"), ("evaluation", "fold_count"), ("cross_validation", "fold_count")),
            4,
        )
    )
    seeds = tuple(
        int(value)
        for value in first_value(
            policy,
            (("training", "seeds"), ("evaluation", "seeds"), ("replication", "seeds"), ("seeds",)),
            list(DEFAULT_SEEDS),
        )
    )
    offset_coefficient = float(
        first_value(
            policy,
            (("training", "source_offset_coefficient"), ("residual_model", "source_offset_coefficient")),
            1.0,
        )
    )
    model_ids = tuple(
        str(value)
        for value in first_value(policy, (("training", "model_ids"),), list(MODEL_IDS))
    )
    source_value = first_value(
        policy,
        (
            ("inputs", "step24_model_artifacts"),
            ("frozen_english_source_scorer", "artifact_json_path"),
            ("inputs", "step24_source_artifact"),
        ),
        str(FIXED_SOURCE_ARTIFACT.relative_to(ROOT)),
    )
    source_path = resolve(str(source_value))
    if source_path.resolve() != FIXED_SOURCE_ARTIFACT.resolve():
        raise ValueError(f"Step27 source artifact must remain fixed at {FIXED_SOURCE_ARTIFACT}")
    if fold_count != 4:
        raise ValueError("Step27 requires exactly four seller-component OOF folds")
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"Step27 preregistered seeds must be {list(DEFAULT_SEEDS)}")
    if offset_coefficient != 1.0:
        raise ValueError("Step27 primary source offset coefficient must equal one")
    if model_ids != MODEL_IDS:
        raise ValueError(f"Step27 model matrix must remain {list(MODEL_IDS)}")
    controls = policy.get("models", {}).get("exploratory_controls", {})
    expected_controls = {
        "learned_source_logit_coefficient": EXPLORATORY_MODEL_IDS[0],
        "target_only_alpha_zero": EXPLORATORY_MODEL_IDS[1],
        "silver_sensitivity_M2": SENSITIVITY_MODEL_ID,
    }
    for key, expected_model_id in expected_controls.items():
        control = controls.get(key, {})
        if not bool_value(control.get("enabled")) or control.get("model_id") != expected_model_id:
            raise ValueError(f"Step27 required diagnostic control changed or was disabled: {key}")
    run_id = str(first_value(policy, (("run_id",), ("training", "run_id")), "")).strip()
    if not run_id:
        run_id = outputs_root(policy).name
    return {
        "policy_path": policy_path_value,
        "fold_count": fold_count,
        "seeds": seeds,
        "offset_coefficient": offset_coefficient,
        "model_ids": model_ids,
        "run_id": run_id,
        "source_artifact_path": source_path,
        "l2_penalty": float(
            first_value(
                policy,
                (("training", "l2_penalty"), ("residual_model", "l2_penalty"), ("models", "common_logistic", "l2_penalty")),
                10.0,
            )
        ),
        "max_iter": int(first_value(policy, (("training", "max_iter"), ("models", "common_logistic", "max_iter")), 400)),
        "tolerance": float(
            first_value(policy, (("training", "tolerance"), ("models", "common_logistic", "tolerance")), 1e-8)
        ),
    }


def source_artifact(path: Path) -> dict:
    payload = load_json(path)
    value: object = payload
    for key in FIXED_SOURCE_KEY:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Fixed Step24 E5 source artifact key is missing: {'.'.join(FIXED_SOURCE_KEY)}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError("Fixed Step24 E5 source artifact is malformed")
    if value.get("feature_names") != ["identifier_redacted_e5_cosine"]:
        raise ValueError("Step27 fixed source scorer must be the Step24 E5-only LR/L2 control")
    logistic = value.get("logistic_artifact") or {}
    if float(logistic.get("l2_penalty", math.nan)) != 10.0:
        raise ValueError("Step24 E5 source LR/L2 penalty changed")
    if logistic.get("class_weight") != "none":
        raise ValueError("Step24 E5 source class-weight contract changed")
    return value


def validate_frozen_source_contract(policy: dict, path: Path, artifact: dict) -> None:
    expected = policy.get("frozen_english_source_scorer", {})
    expected_sha = str(policy.get("inputs", {}).get("step24_model_artifacts_sha256", ""))
    if expected_sha and sha256_file(path) != expected_sha:
        raise ValueError("Step27 frozen Step24 source artifact hash changed")
    for key in (
        "train_row_count",
        "train_positive_count",
        "train_negative_count",
        "train_pair_uid_sha256",
        "feature_names",
    ):
        if key in expected and artifact.get(key) != expected[key]:
            raise ValueError(f"Step27 frozen Step24 source contract changed: {key}")
    logistic = artifact["logistic_artifact"]
    for key in ("parameter_intercept", "l2_penalty"):
        if key in expected and not math.isclose(
            float(logistic[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Step27 frozen Step24 source parameter changed: {key}")
    for key, observed in (
        ("standardization_mean", logistic["standardization"]["mean"]),
        ("standardization_scale", logistic["standardization"]["scale"]),
        ("parameter_coefficients", logistic["parameter_coefficients"]),
    ):
        if key in expected and not np.allclose(
            np.asarray(observed, dtype=float),
            np.asarray(expected[key], dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Step27 frozen Step24 source parameter changed: {key}")


def source_logit(row: dict, artifact: dict, source_name: str) -> float:
    value = numeric(row.get(source_name), source_name)
    logistic = artifact["logistic_artifact"]
    standardization = logistic["standardization"]
    mean = float(standardization["mean"][0])
    scale = float(standardization["scale"][0])
    coefficient = float(logistic["parameter_coefficients"][0])
    if scale <= 0.0:
        raise ValueError("Step24 source standardization scale is invalid")
    return float(logistic["parameter_intercept"]) + coefficient * ((value - mean) / scale)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    values = np.asarray([[numeric(row.get(name), name) for name in names] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape != (len(rows), len(names)):
        raise ValueError("Step27 feature matrix shape is invalid")
    return values


def fit_offset_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray,
    l2_penalty: float,
    max_iter: int,
    tolerance: float,
    standardization: dict | None = None,
) -> dict:
    if len(x) == 0 or set(np.asarray(y, dtype=int)) != {0, 1}:
        raise ValueError("Step27 residual training requires a non-empty two-class dataset")
    if standardization is None:
        means = np.average(x, axis=0, weights=weights)
        variances = np.average((x - means) ** 2, axis=0, weights=weights)
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales = np.where(scales > 1e-12, scales, 1.0)
        standardization_reference = "current_training_rows"
    else:
        means = np.asarray(standardization["mean"], dtype=float)
        scales = np.asarray(standardization["scale"], dtype=float)
        if means.shape != (x.shape[1],) or scales.shape != (x.shape[1],):
            raise ValueError("Step27 shared real-row standardization dimensions disagree")
        if np.any(scales <= 0.0) or not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("Step27 shared real-row standardization is invalid")
        standardization_reference = "real_fold_train_rows_shared_by_M0_M1_M2"
    z = (x - means) / scales
    parameters = np.zeros(z.shape[1] + 1, dtype=float)
    ridge = np.diag(np.r_[0.0, np.full(z.shape[1], float(l2_penalty))])
    design = np.column_stack([np.ones(len(z)), z])
    converged = False
    final_delta = math.inf
    for iteration in range(1, max_iter + 1):
        eta = offset + design @ parameters
        probabilities = np.clip(sigmoid(eta), 1e-9, 1.0 - 1e-9)
        gradient = design.T @ (weights * (probabilities - y)) + ridge @ parameters
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + ridge
        hessian += np.eye(len(parameters)) * 1e-10
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        parameters -= delta
        final_delta = float(np.linalg.norm(delta))
        if final_delta <= tolerance:
            converged = True
            break
    if not np.isfinite(parameters).all() or not math.isfinite(final_delta):
        raise FloatingPointError("Step27 LR/L2 solver produced non-finite parameters")
    return {
        "feature_mean": means.tolist(),
        "feature_scale": scales.tolist(),
        "standardization_reference": standardization_reference,
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:].tolist(),
        "l2_penalty": float(l2_penalty),
        "source_offset_coefficient": 1.0,
        "solver_iterations": int(iteration),
        "solver_converged": bool(converged),
        "solver_final_delta_norm": final_delta,
    }


def predict_offset_logistic(artifact: dict, x: np.ndarray, offset: np.ndarray) -> np.ndarray:
    means = np.asarray(artifact["feature_mean"], dtype=float)
    scales = np.asarray(artifact["feature_scale"], dtype=float)
    coefficients = np.asarray(artifact["coefficients"], dtype=float)
    logits = offset + float(artifact["intercept"]) + ((x - means) / scales) @ coefficients
    return sigmoid(logits)


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positive = int(y.sum())
    if positive == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    sorted_y = y[order]
    tp = 0
    fp = 0
    ap = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        group = sorted_y[position:end]
        group_positive = int(group.sum())
        tp += group_positive
        fp += int(len(group) - group_positive)
        ap += (group_positive / positive) * (tp / (tp + fp))
        position = end
    return float(ap)


def pr_auc_trapezoidal(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positive = int(y.sum())
    if positive == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    sorted_y = y[order]
    tp = 0
    fp = 0
    recall = [0.0]
    precision = [1.0]
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        group = sorted_y[position:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall.append(tp / positive)
        precision.append(tp / (tp + fp))
        position = end
    precision_values = np.asarray(precision, dtype=float)
    recall_values = np.asarray(recall, dtype=float)
    return float(
        np.sum(
            (recall_values[1:] - recall_values[:-1])
            * (precision_values[1:] + precision_values[:-1])
            * 0.5
        )
    )


def select_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, np.asarray(scores, dtype=float), 1.0])
    best = (-math.inf, -math.inf, 0.5)
    for threshold in candidates:
        predicted = scores >= threshold
        positive = y == 1
        negative = ~positive
        recall = float(predicted[positive].mean()) if positive.any() else math.nan
        specificity = float((~predicted[negative]).mean()) if negative.any() else math.nan
        balanced = (recall + specificity) / 2.0
        candidate = (balanced, threshold, threshold)
        if candidate > best:
            best = candidate
    return float(best[2])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    tn = int(np.sum((~predicted) & (y == 0)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "roc_auc": roc_auc(y, scores),
        "average_precision": average_precision(y, scores),
        "pr_auc_trapezoidal": pr_auc_trapezoidal(y, scores),
        "threshold": float(threshold),
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def input_manifest(policy_path_value: Path, paths: list[Path], run_id: str) -> dict:
    records = []
    for path in sorted(set(paths), key=lambda value: str(value)):
        if not path.is_file():
            raise FileNotFoundError(f"Step27 required input is missing: {path}")
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    core = {"run_id": run_id, "inputs": records}
    return {**core, "manifest_sha256": canonical_hash(core)}


def file_bundle_sha256(paths: list[Path]) -> str:
    return canonical_hash(
        [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path)}
            for path in sorted(paths, key=lambda value: str(value))
        ]
    )


def completion_manifest(run_id: str, input_manifest_sha256: str, paths: list[Path]) -> dict:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Step27 completion output is missing: {path}")
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    core = {
        "status": "complete",
        "run_id": run_id,
        "input_manifest_sha256": input_manifest_sha256,
        "outputs": records,
    }
    return {**core, "manifest_sha256": canonical_hash(core)}


def validate_completion_manifest(path: Path, expected: dict) -> None:
    if load_json(path) != expected:
        raise ValueError(f"Step27 completion manifest no longer matches outputs: {path}")


def merge_fold_manifest(real_rows: list[dict], fold_rows: list[dict], fold_count: int) -> list[dict]:
    index: dict[str, dict] = {}
    for row in fold_rows:
        uid = row_uid(row)
        if not uid or uid in index:
            raise ValueError(f"Step27 fold manifest has an invalid/duplicate pair uid: {uid!r}")
        index[uid] = row
    merged = []
    component_fold: dict[str, int] = {}
    for row in real_rows:
        uid = row_uid(row)
        assignment = index.get(uid)
        if assignment is None:
            if str(row.get("split_name")) == "train":
                raise ValueError(f"Step27 train pair is missing from the fold manifest: {uid}")
            merged.append(dict(row))
            continue
        item = {**row, **{key: value for key, value in assignment.items() if value not in (None, "")}}
        if str(item.get("split_name")) == "train":
            fold = row_fold(item)
            if not 0 <= fold < fold_count:
                raise ValueError(f"Step27 fold id is outside [0,{fold_count}): {uid}={fold}")
            component = row_component(item)
            previous = component_fold.setdefault(component, fold)
            if previous != fold:
                raise ValueError(f"Seller component crosses Step27 OOF folds: {component}")
        merged.append(item)
    return merged


def validate_rows(
    real_rows: list[dict],
    synthetic_rows: list[dict],
    names: list[str],
    fold_count: int,
    seeds: tuple[int, ...],
    required_real_splits: tuple[str, ...] = ("train", "valid", "test"),
) -> None:
    real_uids = [row_uid(row) for row in real_rows]
    if any(not uid for uid in real_uids) or len(real_uids) != len(set(real_uids)):
        raise ValueError("Step27 real feature table has empty/duplicate pair_uid values")
    splits = Counter(str(row.get("split_name")) for row in real_rows)
    unexpected_splits = sorted(set(splits) - set(required_real_splits))
    if unexpected_splits:
        raise ValueError(
            f"Step27 real feature table exposes splits outside this invocation: {unexpected_splits}"
        )
    for split in required_real_splits:
        rows = [row for row in real_rows if row.get("split_name") == split]
        if not rows or {row_label(row) for row in rows} != {0, 1}:
            raise ValueError(f"Step27 real {split} split is empty or single-class: {splits}")
    train_index = {row_uid(row): row for row in real_rows if row.get("split_name") == "train"}
    observed_seeds = set()
    synthetic_uids = set()
    for row in synthetic_rows:
        uid = row_uid(row)
        if not uid or uid in synthetic_uids or uid in set(real_uids):
            raise ValueError(f"Step27 synthetic uid is empty/duplicate/collides with real data: {uid!r}")
        synthetic_uids.add(uid)
        if not (
            str(row.get("synthetic_split_name")) == "synthetic_train_only"
            and bool_value(row.get("synthetic_train_only"))
            and str(row.get("split_name")) == "train"
        ):
            raise ValueError(f"Synthetic row is not train-only: {uid}")
        parent_uid = str(row.get("parent_pair_uid") or row.get("source_pair_uid") or "")
        parent = train_index.get(parent_uid)
        if parent is None:
            raise ValueError(f"Synthetic row parent is absent from canonical train: {uid}->{parent_uid}")
        if row_label(row) != row_label(parent):
            raise ValueError(f"Synthetic child label differs from its real parent: {uid}")
        if row_component(row) != row_component(parent) or row_fold(row) != row_fold(parent):
            raise ValueError(f"Synthetic child escaped its parent component/fold: {uid}")
        seed = int(row.get("generation_seed", row.get("seed", -1)))
        if seed not in seeds:
            raise ValueError(f"Synthetic row has a non-preregistered generation seed: {uid}={seed}")
        observed_seeds.add(seed)
    if observed_seeds != set(seeds):
        raise ValueError(f"Synthetic table does not cover all preregistered seeds: {sorted(observed_seeds)}")
    required = set(names)
    for table_name, rows in (("real", real_rows), ("synthetic", synthetic_rows)):
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"Step27 {table_name} feature table is missing residual fields: {missing}")


def validate_delayed_evaluation_rows(
    rows: list[dict], split_name: str, names: list[str], policy: dict
) -> None:
    if split_name not in {"valid", "test"}:
        raise ValueError(f"Invalid delayed evaluation split: {split_name}")
    expected = policy["canonical_chinese_boundary"]["split_counts"][split_name]
    if len(rows) != int(expected["rows"]):
        raise ValueError(f"Step27 {split_name} row count changed: {len(rows)} != {expected['rows']}")
    labels = [row_label(row) for row in rows]
    if sum(labels) != int(expected["positive"]) or len(labels) - sum(labels) != int(
        expected["negative"]
    ):
        raise ValueError(f"Step27 {split_name} label counts changed")
    uids = [row_uid(row) for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError(f"Step27 {split_name} has empty/duplicate pair UIDs")
    missing = sorted(set(names) - set(rows[0]))
    if missing:
        raise ValueError(f"Step27 {split_name} is missing residual fields: {missing}")


def materialize_feature_tables(
    policy: dict, cfg: dict, real_split: str | None = None
) -> tuple[list[dict], list[dict], list[Path]]:
    """Load immutable pair features, optionally opening exactly one canonical split."""
    real_root = common.output_root(policy) / "pair_features" / "real"
    if real_split is not None and real_split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported Step27 real feature split: {real_split}")
    real_path = real_root / (
        f"real_pair_features.{real_split}.csv" if real_split else "real_pair_features.csv"
    )
    real_manifest = real_root / "manifest.json"
    for path in (real_path, real_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Run step27_build_pair_features.py first: {path}")
    common.assert_existing_manifest_identity(real_manifest, common.load_json(real_manifest)["identity"])
    real_features = common.load_csv(real_path)
    synthetic_all = []
    feature_paths = [real_path, real_manifest]
    for seed in cfg["seeds"]:
        feature_root = common.track_root(policy, seed, "primary") / "pair_features"
        feature_path = feature_root / "synthetic_pair_features.csv"
        feature_manifest = feature_root / "manifest.json"
        for path in (feature_path, feature_manifest):
            if not path.is_file():
                raise FileNotFoundError(f"Run step27_build_pair_features.py first: {path}")
        common.assert_existing_manifest_identity(
            feature_manifest, common.load_json(feature_manifest)["identity"]
        )
        synthetic = common.load_csv(feature_path)
        if any(int(row.get("seed", -1)) != seed for row in synthetic):
            raise ValueError(f"Step27 synthetic feature seed mismatch: {feature_path}")
        feature_paths.extend([feature_path, feature_manifest])
        synthetic_all.extend(synthetic)
    return real_features, synthetic_all, feature_paths


def load_sensitivity_feature_tables(policy: dict, cfg: dict) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    paths: list[Path] = []
    for seed in cfg["seeds"]:
        feature_root = common.track_root(policy, seed, "silver_sensitivity") / "pair_features"
        feature_path = feature_root / "synthetic_pair_features.csv"
        feature_manifest = feature_root / "manifest.json"
        for path in (feature_path, feature_manifest):
            if not path.is_file():
                raise FileNotFoundError(f"Run the Step27 silver sensitivity feature stage first: {path}")
        common.assert_existing_manifest_identity(
            feature_manifest, common.load_json(feature_manifest)["identity"]
        )
        seed_rows = common.load_csv(feature_path)
        if any(int(row.get("seed", -1)) != seed for row in seed_rows):
            raise ValueError(f"Step27 silver sensitivity feature seed mismatch: {feature_path}")
        rows.extend(seed_rows)
        paths.extend([feature_path, feature_manifest])
    return rows, paths


def model_training_rows(
    model_id: str,
    real_train: list[dict],
    synthetic_seed_rows: list[dict],
    held_fold: int | None,
) -> tuple[list[dict], list[dict]]:
    base = [row for row in real_train if held_fold is None or row_fold(row) != held_fold]
    eligible_children = [
        row for row in synthetic_seed_rows if held_fold is None or row_fold(row) != held_fold
    ]
    if held_fold is not None and any(row_fold(row) == held_fold for row in eligible_children):
        raise AssertionError("Held-out synthetic descendants entered a Step27 training fold")
    if model_id == MODEL_IDS[0]:
        return base, []
    if model_id == MODEL_IDS[1]:
        parent_index = {row_uid(row): row for row in base}
        duplicates = []
        for child in eligible_children:
            parent_uid = str(child.get("parent_pair_uid") or child.get("source_pair_uid") or "")
            parent = parent_index.get(parent_uid)
            if parent is None:
                raise ValueError(f"M1 duplication parent is absent from the training fold: {parent_uid}")
            duplicate = dict(parent)
            duplicate["pair_uid"] = f"duplication::{row_uid(child)}"
            duplicate["training_sample_weight"] = row_weight(child)
            duplicate["step27_duplicate_of"] = parent_uid
            duplicates.append(duplicate)
        return base, duplicates
    if model_id == MODEL_IDS[2]:
        return base, eligible_children
    raise ValueError(f"Unknown Step27 model id: {model_id}")


def train_one(
    rows: list[dict],
    additions: list[dict],
    names: list[str],
    source: dict,
    source_name: str,
    cfg: dict,
    source_mode: str = "fixed_unit_offset",
) -> dict:
    combined = rows + additions
    residual_x = matrix(combined, names)
    y = np.asarray([row_label(row) for row in combined], dtype=float)
    raw_weights = np.asarray([row_weight(row) for row in combined], dtype=float)
    source_logits = np.asarray(
        [source_logit(row, source, source_name) for row in combined], dtype=float
    )
    real_residual_x = matrix(rows, names)
    real_source_logits = np.asarray(
        [source_logit(row, source, source_name) for row in rows], dtype=float
    )
    if source_mode == "fixed_unit_offset":
        x = residual_x
        real_x = real_residual_x
        offsets = source_logits
        artifact_feature_names = list(names)
    elif source_mode == "target_only_alpha_zero":
        x = residual_x
        real_x = real_residual_x
        offsets = np.zeros(len(combined), dtype=float)
        artifact_feature_names = list(names)
    elif source_mode == "learned_source_alpha":
        x = np.column_stack([source_logits, residual_x])
        real_x = np.column_stack([real_source_logits, real_residual_x])
        offsets = np.zeros(len(combined), dtype=float)
        artifact_feature_names = ["frozen_english_source_logit", *names]
    else:
        raise ValueError(f"Unknown Step27 source mode: {source_mode}")
    real_weights = np.asarray([row_weight(row) for row in rows], dtype=float)
    real_effective_weight = float(real_weights.sum())
    raw_combined_effective_weight = float(raw_weights.sum())
    if real_effective_weight <= 0.0 or raw_combined_effective_weight <= 0.0:
        raise ValueError("Step27 effective training weights must be positive")
    weight_normalization_factor = real_effective_weight / raw_combined_effective_weight
    weights = raw_weights * weight_normalization_factor
    real_mean = np.average(real_x, axis=0, weights=real_weights)
    real_variance = np.average((real_x - real_mean) ** 2, axis=0, weights=real_weights)
    real_scale = np.sqrt(np.maximum(real_variance, 0.0))
    real_scale = np.where(real_scale > 1e-12, real_scale, 1.0)
    artifact = fit_offset_logistic(
        x,
        y,
        weights,
        offsets,
        cfg["l2_penalty"],
        cfg["max_iter"],
        cfg["tolerance"],
        standardization={"mean": real_mean.tolist(), "scale": real_scale.tolist()},
    )
    if not artifact["solver_converged"]:
        raise RuntimeError(
            "Step27 LR/L2 solver did not reach the preregistered tolerance; "
            f"final_delta_norm={artifact['solver_final_delta_norm']}"
        )
    artifact.update(
        {
            "artifact_feature_names": artifact_feature_names,
            "residual_feature_names": names,
            "source_mode": source_mode,
            "source_offset_coefficient": 1.0 if source_mode == "fixed_unit_offset" else 0.0,
            "real_training_row_count": len(rows),
            "added_training_row_count": len(additions),
            "effective_real_weight": float(sum(row_weight(row) for row in rows)),
            "effective_added_weight": float(sum(row_weight(row) for row in additions)),
            "raw_combined_effective_weight": raw_combined_effective_weight,
            "normalized_combined_effective_weight": float(weights.sum()),
            "target_real_effective_weight": real_effective_weight,
            "fold_train_weight_normalization_factor": weight_normalization_factor,
            "regularization_weight_scale_matched_to_real_only": True,
        }
    )
    if source_mode == "learned_source_alpha":
        source_scale = float(artifact["feature_scale"][0])
        source_mean = float(artifact["feature_mean"][0])
        standardized_coefficient = float(artifact["coefficients"][0])
        learned_alpha = standardized_coefficient / source_scale
        artifact["learned_source_alpha"] = learned_alpha
        artifact["learned_source_alpha_standardized_coefficient"] = standardized_coefficient
        artifact["learned_source_alpha_center"] = source_mean
        artifact["effective_unstandardized_intercept"] = (
            float(artifact["intercept"]) - learned_alpha * source_mean
        )
    return artifact


def predict_rows(rows: list[dict], artifact: dict, names: list[str], source: dict, source_name: str) -> np.ndarray:
    residual_x = matrix(rows, names)
    source_logits = np.asarray(
        [source_logit(row, source, source_name) for row in rows], dtype=float
    )
    source_mode = str(artifact.get("source_mode", "fixed_unit_offset"))
    if source_mode == "fixed_unit_offset":
        x = residual_x
        offsets = source_logits
    elif source_mode == "target_only_alpha_zero":
        x = residual_x
        offsets = np.zeros(len(rows), dtype=float)
    elif source_mode == "learned_source_alpha":
        x = np.column_stack([source_logits, residual_x])
        offsets = np.zeros(len(rows), dtype=float)
    else:
        raise ValueError(f"Unknown Step27 artifact source mode: {source_mode}")
    return predict_offset_logistic(
        artifact,
        x,
        offsets,
    )


def source_mode_for_model(model_id: str) -> str:
    if model_id == EXPLORATORY_MODEL_IDS[0]:
        return "learned_source_alpha"
    if model_id == EXPLORATORY_MODEL_IDS[1]:
        return "target_only_alpha_zero"
    return "fixed_unit_offset"


def training_model_id_for(model_id: str) -> str:
    if model_id in EXPLORATORY_MODEL_IDS or model_id == SENSITIVITY_MODEL_ID:
        return MODEL_IDS[2]
    return model_id


def validate_persisted_artifact_contract(
    artifact: dict, model_id: str, seed: int, names: list[str], manifest_sha256: str
) -> None:
    expected_mode = source_mode_for_model(model_id)
    expected_features = (
        ["frozen_english_source_logit", *names]
        if expected_mode == "learned_source_alpha"
        else list(names)
    )
    if (
        artifact.get("model_id") != model_id
        or int(artifact.get("seed", -1)) != seed
        or artifact.get("source_mode") != expected_mode
        or artifact.get("artifact_feature_names") != expected_features
        or artifact.get("residual_feature_names") != list(names)
        or artifact.get("input_manifest_sha256") != manifest_sha256
    ):
        raise ValueError(f"Step27 persisted artifact routing contract changed: {model_id}/seed={seed}")
    expected_offset = 1.0 if expected_mode == "fixed_unit_offset" else 0.0
    if float(artifact.get("source_offset_coefficient", math.nan)) != expected_offset:
        raise ValueError(f"Step27 persisted artifact has the wrong source offset: {model_id}")
    width = len(expected_features)
    arrays = [
        np.asarray(artifact.get("feature_mean", []), dtype=float),
        np.asarray(artifact.get("feature_scale", []), dtype=float),
        np.asarray(artifact.get("coefficients", []), dtype=float),
    ]
    if any(array.shape != (width,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Step27 persisted artifact dimensions/values are invalid: {model_id}")
    if np.any(arrays[1] <= 0.0) or not math.isfinite(float(artifact.get("intercept", math.nan))):
        raise ValueError(f"Step27 persisted artifact scaling/intercept is invalid: {model_id}")
    if not bool_value(artifact.get("solver_converged")):
        raise ValueError(f"Step27 persisted artifact is not converged: {model_id}")


def prediction_rows(
    rows: list[dict], scores: np.ndarray, model_id: str, split: str, seed: int | str, threshold: float
) -> list[dict]:
    output = []
    for row, score in zip(rows, scores, strict=True):
        output.append(
            {
                "pair_uid": row_uid(row),
                "split_name": split,
                "review_label": "positive" if row_label(row) else "negative",
                "label": row_label(row),
                "component_id": row_component(row),
                "evidence_type": row.get("evidence_type", ""),
                "model_id": model_id,
                "seed": seed,
                "prob_positive": round(float(score), 12),
                "frozen_oof_threshold": round(float(threshold), 12),
                "predicted_label": int(score >= threshold),
            }
        )
    return output


def aggregate_seed_predictions(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["split_name"], row["pair_uid"])].append(row)
    output = []
    for key, values in sorted(groups.items()):
        model_id, split, uid = key
        seed_values = [int(value["seed"]) for value in values]
        seeds = set(seed_values)
        if seeds != set(DEFAULT_SEEDS) or len(seed_values) != len(DEFAULT_SEEDS):
            raise ValueError(f"Seed aggregation is incomplete for {model_id}/{split}/{uid}")
        first = values[0]
        for value in values[1:]:
            for field in (
                "review_label",
                "label",
                "component_id",
                "evidence_type",
                "frozen_oof_threshold",
            ):
                if value[field] != first[field]:
                    raise ValueError(
                        f"Seed aggregation metadata differs for {model_id}/{split}/{uid}: {field}"
                    )
        score = float(np.mean([float(value["prob_positive"]) for value in values]))
        threshold = float(first["frozen_oof_threshold"])
        output.append(
            {
                **{key: first[key] for key in (
                    "pair_uid", "split_name", "review_label", "label", "component_id", "evidence_type", "model_id"
                )},
                "seed_aggregation": "per_real_pair_arithmetic_mean_across_ten_preregistered_seeds",
                "seed_count": len(seeds),
                "seeds_are_independent_inferential_units": "0",
                "prob_positive": round(score, 12),
                "frozen_oof_threshold": round(threshold, 12),
                "predicted_label": int(score >= threshold),
            }
        )
    return output


def score_frozen_split_after_gate(
    *,
    policy: dict,
    cfg: dict,
    real_rows: list[dict],
    names: list[str],
    source: dict,
    source_name: str,
    gate_summary_path: Path,
    split_name: str,
    current_input_manifest: dict,
) -> None:
    if split_name not in {"valid", "test"}:
        raise ValueError(f"Unsupported Step27 delayed evaluation split: {split_name}")
    root = outputs_root(policy)
    training_dir = root / "training"
    summary_path = training_dir / "step27_training_summary.json"
    artifacts_path = training_dir / "step27_model_artifacts.json"
    training_manifest_path = training_dir / "step27_training_input_manifest.json"
    for path in (summary_path, artifacts_path, training_manifest_path, gate_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Step27 internal-test gate input is missing: {path}")
    training_summary = load_json(summary_path)
    artifact_bundle = load_json(artifacts_path)
    training_manifest = load_json(training_manifest_path)
    gate = load_json(gate_summary_path)
    promotion = gate.get("promotion", {})
    if gate.get("run_id") != cfg["run_id"] or training_summary.get("run_id") != cfg["run_id"]:
        raise ValueError(f"Step27 {split_name} gate belongs to another run")
    required_flag = "eligible_for_valid" if split_name == "valid" else "eligible_for_internal_test"
    if not promotion.get(required_flag, False):
        raise ValueError(f"Step27 {split_name} remains blocked because its prerequisite gate failed")
    if promotion.get("test_metrics_used_for_gate", False):
        raise ValueError(f"Step27 {split_name} gate illegally used test metrics")
    expected_mode = "oof_gate" if split_name == "valid" else "valid_gate"
    if gate.get("analysis_contract", {}).get("audit_mode") != expected_mode:
        raise ValueError(
            f"Step27 {split_name} requires a {expected_mode} summary, not "
            f"{gate.get('analysis_contract', {}).get('audit_mode')!r}"
        )
    training_manifest_core = dict(training_manifest)
    training_manifest_sha256 = training_manifest_core.pop("manifest_sha256", None)
    if not training_manifest_sha256 or training_manifest_sha256 != canonical_hash(
        training_manifest_core
    ):
        raise ValueError("Step27 frozen training input manifest failed its self-hash check")
    for record in training_manifest.get("inputs", []):
        path = resolve(record["path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Step27 frozen training input changed before delayed scoring: {path}")
    if (
        training_summary.get("input_manifest_sha256") != training_manifest_sha256
        or artifact_bundle.get("input_manifest_sha256") != training_manifest_sha256
    ):
        raise ValueError(
            "Step27 delayed scoring code/data manifest differs from the frozen training artifacts"
        )
    thresholds = training_summary["frozen_oof_thresholds"]
    evaluation_rows = [row for row in real_rows if row.get("split_name") == split_name]
    seed_rows: list[dict] = []
    for model_id in REPORTING_MODEL_IDS:
        for seed in cfg["seeds"]:
            artifact = artifact_bundle["artifacts"][model_id][str(seed)]
            validate_persisted_artifact_contract(
                artifact, model_id, seed, names, training_manifest_sha256
            )
            scores = predict_rows(evaluation_rows, artifact, names, source, source_name)
            seed_rows.extend(
                prediction_rows(
                    evaluation_rows,
                    scores,
                    model_id,
                    split_name,
                    seed,
                    float(thresholds[model_id]),
                )
            )
    mean_rows = aggregate_seed_predictions(seed_rows)
    metrics_by_model = {}
    for model_id in REPORTING_MODEL_IDS:
        selected = [row for row in mean_rows if row["model_id"] == model_id]
        y = np.asarray([int(row["label"]) for row in selected], dtype=int)
        scores = np.asarray([float(row["prob_positive"]) for row in selected], dtype=float)
        metrics_by_model[model_id] = metrics(y, scores, float(thresholds[model_id]))
    output_dir = root / ("valid_diagnostic" if split_name == "valid" else "internal_test_diagnostic")
    gate_binding = {
        "run_id": cfg["run_id"],
        "split_name": split_name,
        "policy_sha256": sha256_file(cfg["policy_path"]),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "common_sha256": sha256_file(Path(common.__file__).resolve()),
        "frozen_source_artifact_sha256": sha256_file(cfg["source_artifact_path"]),
        "training_summary_sha256": sha256_file(summary_path),
        "model_artifacts_sha256": sha256_file(artifacts_path),
        "pair_feature_bundle_sha256": training_summary["pair_feature_bundle_sha256"],
        "training_input_manifest_sha256": training_manifest_sha256,
        "evaluation_input_manifest_sha256": current_input_manifest["manifest_sha256"],
        "prerequisite_gate_summary_path": str(gate_summary_path.relative_to(ROOT)).replace("\\", "/"),
        "prerequisite_gate_summary_sha256": sha256_file(gate_summary_path),
        "test_metrics_used_for_gate": False,
    }
    binding_path = output_dir / f"step27_{split_name}_gate_binding.json"
    seed_path = output_dir / f"step27_{split_name}_seed_predictions.csv"
    mean_path = output_dir / f"step27_{split_name}_seed_mean_predictions.csv"
    diagnostic_path = output_dir / f"step27_{split_name}_summary.json"
    write_json_immutable(binding_path, gate_binding)
    write_csv_immutable(seed_path, seed_rows)
    write_csv_immutable(mean_path, mean_rows)
    write_json_immutable(
        diagnostic_path,
        {
            "status": "complete",
            "run_id": cfg["run_id"],
            "role": (
                "single_open_retrospective_development_gate"
                if split_name == "valid"
                else "retrospective_internal_diagnostic_only"
            ),
            "split_name": split_name,
            "threshold_reselection_performed": False,
            "prerequisite_gate_summary_sha256": sha256_file(gate_summary_path),
            "training_input_manifest_sha256": training_manifest_sha256,
            "evaluation_input_manifest_sha256": current_input_manifest["manifest_sha256"],
            "model_metrics": metrics_by_model,
            "output_paths": {
                "gate_binding": str(binding_path.relative_to(ROOT)).replace("\\", "/"),
                "seed_predictions": str(seed_path.relative_to(ROOT)).replace("\\", "/"),
                "seed_mean_predictions": str(mean_path.relative_to(ROOT)).replace("\\", "/"),
            },
        },
    )
    delayed_payloads = [binding_path, seed_path, mean_path, diagnostic_path]
    delayed_completion_path = output_dir / f"step27_{split_name}_completion_manifest.json"
    delayed_completion = completion_manifest(
        cfg["run_id"], current_input_manifest["manifest_sha256"], delayed_payloads
    )
    if delayed_completion_path.is_file():
        validate_completion_manifest(delayed_completion_path, delayed_completion)
    else:
        write_json_immutable(delayed_completion_path, delayed_completion)
    print(
        json.dumps(
            {
                "status": "complete",
                "split_name": split_name,
                "role": "development" if split_name == "valid" else "internal_diagnostic_only",
                "metrics": metrics_by_model,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    policy_path_value = resolve(args.policy)
    policy = load_json(policy_path_value)
    cfg = validate_policy(policy, policy_path_value)
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "run_id": cfg["run_id"],
                    "primary_models": list(MODEL_IDS),
                    "diagnostic_models": [*EXPLORATORY_MODEL_IDS, SENSITIVITY_MODEL_ID],
                },
                indent=2,
            )
        )
        return

    root = outputs_root(policy)
    source_path = cfg["source_artifact_path"]
    if args.score_valid and args.score_internal_test:
        raise ValueError("Step27 permits only one delayed evaluation split per invocation")
    active_split = "valid" if args.score_valid else "test" if args.score_internal_test else "train"
    real_rows, synthetic_rows, feature_paths = materialize_feature_tables(
        policy, cfg, real_split=active_split
    )
    sensitivity_rows, sensitivity_feature_paths = load_sensitivity_feature_tables(policy, cfg)
    duplication_feature_paths = [
        common.track_root(policy, seed, track)
        / "pair_features"
        / "equal_weight_duplication_pair_features.csv"
        for seed in cfg["seeds"]
        for track in ("primary", "silver_sensitivity")
    ]
    all_feature_paths = feature_paths + sensitivity_feature_paths + duplication_feature_paths
    manifest_paths = [
        policy_path_value,
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        source_path,
        common.parent_root(policy) / "manifest.json",
        *(common.seed_root(policy, seed) / "generation_manifest.json" for seed in cfg["seeds"]),
        *all_feature_paths,
    ]
    manifest = input_manifest(policy_path_value, manifest_paths, cfg["run_id"])
    if not real_rows:
        raise ValueError(f"Step27 active split has no feature rows: {active_split}")
    names = feature_names(policy, real_rows)
    source_name = source_feature_name(policy)
    source = source_artifact(source_path)
    validate_frozen_source_contract(policy, source_path, source)
    for table_name, rows in (("real", real_rows), ("synthetic", synthetic_rows)):
        if source_name not in rows[0]:
            raise ValueError(f"Step27 {table_name} table is missing frozen-source feature {source_name}")
    if active_split == "train":
        validate_rows(
            real_rows,
            synthetic_rows,
            names,
            cfg["fold_count"],
            cfg["seeds"],
            required_real_splits=("train",),
        )
        validate_rows(
            real_rows,
            sensitivity_rows,
            names,
            cfg["fold_count"],
            cfg["seeds"],
            required_real_splits=("train",),
        )
    else:
        validate_delayed_evaluation_rows(real_rows, active_split, names, policy)
    if args.score_valid:
        if not args.oof_gate_summary:
            raise ValueError("--score-valid requires --oof-gate-summary")
        score_frozen_split_after_gate(
            policy=policy,
            cfg=cfg,
            real_rows=real_rows,
            names=names,
            source=source,
            source_name=source_name,
            gate_summary_path=resolve(args.oof_gate_summary),
            split_name="valid",
            current_input_manifest=manifest,
        )
        return
    if args.score_internal_test:
        if not args.valid_gate_summary:
            raise ValueError("--score-internal-test requires --valid-gate-summary")
        score_frozen_split_after_gate(
            policy=policy,
            cfg=cfg,
            real_rows=real_rows,
            names=names,
            source=source,
            source_name=source_name,
            gate_summary_path=resolve(args.valid_gate_summary),
            split_name="test",
            current_input_manifest=manifest,
        )
        return
    if args.materialize_features_only:
        print(
            json.dumps(
                {
                    "status": "features_materialized",
                    "feature_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in feature_paths],
                    "active_real_split": active_split,
                    "real_rows": len(real_rows),
                    "synthetic_rows_total": len(synthetic_rows),
                    "silver_sensitivity_rows_total": len(sensitivity_rows),
                },
                indent=2,
            )
        )
        return
    if args.validate_inputs_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "input_manifest_sha256": manifest["manifest_sha256"],
                    "real_rows": len(real_rows),
                    "synthetic_rows": len(synthetic_rows),
                    "silver_sensitivity_rows": len(sensitivity_rows),
                    "residual_features": names,
                },
                indent=2,
            )
        )
        return

    train_rows = [row for row in real_rows if row.get("split_name") == "train"]
    synthetic_by_seed = {
        seed: [row for row in synthetic_rows if int(row.get("generation_seed", row.get("seed", -1))) == seed]
        for seed in cfg["seeds"]
    }
    sensitivity_by_seed = {
        seed: [
            row
            for row in sensitivity_rows
            if int(row.get("generation_seed", row.get("seed", -1))) == seed
        ]
        for seed in cfg["seeds"]
    }
    all_model_ids = REPORTING_MODEL_IDS

    output_dir = root / "training"
    manifest_path = output_dir / "step27_training_input_manifest.json"
    payload_outputs = [
        manifest_path,
        output_dir / "step27_training_summary.json",
        output_dir / "step27_seed_predictions.csv",
        output_dir / "step27_seed_mean_predictions.csv",
        output_dir / "step27_model_artifacts.json",
    ]
    completion_path = output_dir / "step27_training_completion_manifest.json"
    expected_outputs = [*payload_outputs, completion_path]
    if output_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Step27 training output directory exists without its manifest: {output_dir}")
        old_manifest = load_json(manifest_path)
        if old_manifest.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("Refusing to overwrite Step27 training outputs across a different code/data manifest")
        if all(path.is_file() for path in expected_outputs):
            validate_completion_manifest(
                completion_path,
                completion_manifest(cfg["run_id"], manifest["manifest_sha256"], payload_outputs),
            )
            print(json.dumps({"status": "identical_replay_already_complete", "output_dir": str(output_dir)}, indent=2))
            return
        raise FileExistsError(f"Step27 training run is incomplete and immutable; use a new run-scoped output root: {output_dir}")

    seed_prediction_records: list[dict] = []
    artifacts: dict[str, dict] = {}
    oof_seed_scores: dict[str, dict[int, np.ndarray]] = {model_id: {} for model_id in all_model_ids}
    oof_y = np.asarray([row_label(row) for row in train_rows], dtype=int)
    for model_id in all_model_ids:
        for seed in cfg["seeds"]:
            seed_children = (
                sensitivity_by_seed[seed]
                if model_id == SENSITIVITY_MODEL_ID
                else synthetic_by_seed[seed]
            )
            training_model_id = training_model_id_for(model_id)
            scores = np.full(len(train_rows), np.nan, dtype=float)
            for held_fold in range(cfg["fold_count"]):
                fit_real, additions = model_training_rows(
                    training_model_id, train_rows, seed_children, held_fold
                )
                held_indices = [index for index, row in enumerate(train_rows) if row_fold(row) == held_fold]
                held_rows = [train_rows[index] for index in held_indices]
                if not held_rows:
                    raise ValueError(f"Step27 fold {held_fold} has no held-out real rows")
                artifact = train_one(
                    fit_real,
                    additions,
                    names,
                    source,
                    source_name,
                    cfg,
                    source_mode=source_mode_for_model(model_id),
                )
                held_scores = predict_rows(held_rows, artifact, names, source, source_name)
                scores[np.asarray(held_indices, dtype=int)] = held_scores
            if not np.isfinite(scores).all():
                raise ValueError(f"Step27 OOF scores are incomplete: {model_id}/seed={seed}")
            oof_seed_scores[model_id][seed] = scores

    frozen_thresholds = {}
    for model_id in all_model_ids:
        mean_oof = np.mean(np.vstack([oof_seed_scores[model_id][seed] for seed in cfg["seeds"]]), axis=0)
        threshold = select_threshold(oof_y, mean_oof)
        frozen_thresholds[model_id] = threshold
        for seed in cfg["seeds"]:
            seed_prediction_records.extend(
                prediction_rows(
                    train_rows,
                    oof_seed_scores[model_id][seed],
                    model_id,
                    "train_oof",
                    seed,
                    threshold,
                )
            )

    # Full-train artifacts are frozen now, but valid/test content remains unopened.
    for model_id in all_model_ids:
        artifacts[model_id] = {}
        for seed in cfg["seeds"]:
            seed_children = (
                sensitivity_by_seed[seed]
                if model_id == SENSITIVITY_MODEL_ID
                else synthetic_by_seed[seed]
            )
            training_model_id = training_model_id_for(model_id)
            fit_real, additions = model_training_rows(
                training_model_id, train_rows, seed_children, None
            )
            artifact = train_one(
                fit_real,
                additions,
                names,
                source,
                source_name,
                cfg,
                source_mode=source_mode_for_model(model_id),
            )
            artifact.update(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "source_artifact_path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_artifact_key": ".".join(FIXED_SOURCE_KEY),
                    "source_offset_coefficient": artifact["source_offset_coefficient"],
                    "input_manifest_sha256": manifest["manifest_sha256"],
                }
            )
            artifacts[model_id][str(seed)] = artifact

    seed_mean_records = aggregate_seed_predictions(seed_prediction_records)
    model_metrics: dict[str, dict] = {}
    for model_id in all_model_ids:
        model_metrics[model_id] = {}
        for split in ("train_oof",):
            selected = [row for row in seed_mean_records if row["model_id"] == model_id and row["split_name"] == split]
            y = np.asarray([int(row["label"]) for row in selected], dtype=int)
            scores = np.asarray([float(row["prob_positive"]) for row in selected], dtype=float)
            model_metrics[model_id][split] = metrics(y, scores, frozen_thresholds[model_id])

    summary = {
        "status": "complete",
        "run_id": cfg["run_id"],
        "policy_version": policy.get("version"),
        "git_commit": git_commit(),
        "input_manifest_sha256": manifest["manifest_sha256"],
        "pair_feature_bundle_sha256": file_bundle_sha256(all_feature_paths),
        "scientific_contract": {
            "source_model": "Step24 identifier-redacted E5 source-only LR/L2",
            "source_artifact_key": ".".join(FIXED_SOURCE_KEY),
            "source_offset_coefficient": 1.0,
            "fold_count": cfg["fold_count"],
            "fold_unit": "seller_component",
            "synthetic_children_inherit_parent_fold": True,
            "seed_count": len(cfg["seeds"]),
            "seed_aggregation": "per_real_pair_seed_mean",
            "seeds_are_independent_inferential_units": False,
            "valid_or_test_used_for_configuration_selection": False,
            "valid_scored": False,
            "valid_requires_separate_passing_oof_gate": True,
            "internal_test_scored": False,
            "internal_test_requires_separate_passing_valid_gate": True,
            "primary_comparison": "step27_m2_synthetic_vs_step27_m1_equal_effective_weight_duplication",
            "required_secondary_comparison": "step27_m2_synthetic_vs_step27_m0_real_only",
            "silver_sensitivity_model": SENSITIVITY_MODEL_ID,
            "silver_sensitivity_can_satisfy_primary_gate": False,
            "exploratory_controls": list(EXPLORATORY_MODEL_IDS),
            "exploratory_controls_can_satisfy_primary_gate": False,
        },
        "residual_feature_names": names,
        "counts": {
            "real": dict(Counter(str(row.get("split_name")) for row in real_rows)),
            "synthetic_total": len(synthetic_rows),
            "synthetic_by_seed": {str(seed): len(synthetic_by_seed[seed]) for seed in cfg["seeds"]},
            "silver_sensitivity_total": len(sensitivity_rows),
            "silver_sensitivity_by_seed": {
                str(seed): len(sensitivity_by_seed[seed]) for seed in cfg["seeds"]
            },
        },
        "frozen_oof_thresholds": frozen_thresholds,
        "model_metrics": model_metrics,
        "output_paths": {
            "input_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "seed_predictions": str((output_dir / "step27_seed_predictions.csv").relative_to(ROOT)).replace("\\", "/"),
            "seed_mean_predictions": str((output_dir / "step27_seed_mean_predictions.csv").relative_to(ROOT)).replace("\\", "/"),
            "model_artifacts": str((output_dir / "step27_model_artifacts.json").relative_to(ROOT)).replace("\\", "/"),
            "completion_manifest": str(completion_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    write_json_immutable(manifest_path, manifest)
    write_csv_immutable(output_dir / "step27_seed_predictions.csv", seed_prediction_records)
    write_csv_immutable(output_dir / "step27_seed_mean_predictions.csv", seed_mean_records)
    write_json_immutable(
        output_dir / "step27_model_artifacts.json",
        {
            "run_id": cfg["run_id"],
            "input_manifest_sha256": manifest["manifest_sha256"],
            "artifacts": artifacts,
        },
    )
    write_json_immutable(output_dir / "step27_training_summary.json", summary)
    write_json_immutable(
        completion_path,
        completion_manifest(cfg["run_id"], manifest["manifest_sha256"], payload_outputs),
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "metrics": model_metrics}, indent=2))


if __name__ == "__main__":
    main()
