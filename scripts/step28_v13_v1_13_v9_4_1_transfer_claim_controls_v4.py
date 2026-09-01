#!/usr/bin/env python3
"""Development-only controls that isolate the frozen English M0 contribution.

This module never reads Audit-A/B truth.  It leaves every frozen V3 artifact
unchanged and publishes to a new V4 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common_v2
import step28_v13_v1_13_v9_4_1_train_development_v2 as train_v2


POLICY_PATH = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_transfer_claim_controls_v4_policy.json"
)
EXPECTED_VERSION = "step28-v13-v1.13-v9.4.1-transfer-claim-controls-v4"
MODEL_IDS = ("i0_identity_only", "t1_m0_plus_identity", "m2_v3_reference")


class TransferClaimControlError(ValueError):
    """Raised when the V4 development-control contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value, dtype="<f8"), allow_pickle=False)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != POLICY_PATH.resolve():
        raise TransferClaimControlError("Only the default V4 policy is valid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != EXPECTED_VERSION:
        raise TransferClaimControlError("V4 policy identity drift")
    observed = canonical_self_hash(value)
    if value.get("canonical_self_hash") != observed:
        raise TransferClaimControlError(
            "V4 policy canonical self-hash drift: "
            f"expected={value.get('canonical_self_hash')} observed={observed}"
        )
    if (
        value.get("status")
        != "DEVELOPMENT_CONTROL_AUTHORIZED_AUDIT_TRUTH_SEALED"
        or value.get("truth_boundary", {}).get("audit_a_labels_or_qrels_allowed")
        is not False
        or value.get("truth_boundary", {}).get("audit_b_labels_or_qrels_allowed")
        is not False
        or float(value.get("fit", {}).get("l2", -1.0)) != 0.01
        or value.get("fit", {}).get("intercept_penalized") is not False
    ):
        raise TransferClaimControlError("V4 scientific boundary drift")
    expected_models = {
        value["models"]["identity_only"]["id"],
        value["models"]["m0_plus_identity"]["id"],
        value["models"]["reference"]["id"],
    }
    if expected_models != set(MODEL_IDS):
        raise TransferClaimControlError("V4 model registry drift")
    for label, spec in value["frozen_inputs"].items():
        pinned = ROOT / str(spec["path"])
        if (
            not pinned.is_file()
            or pinned.stat().st_size != int(spec["size_bytes"])
            or sha256_file(pinned) != str(spec["sha256"])
        ):
            raise TransferClaimControlError(f"Frozen input drift: {label}")
    return value


def _binary_labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8)
    if result.ndim != 1 or not set(np.unique(result).tolist()) <= {0, 1}:
        raise TransferClaimControlError("Labels are not a binary vector")
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return np.ascontiguousarray(result, dtype="<f8")


def objective_and_gradient(
    theta: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    l2: float,
    offset: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Logistic loss with an unpenalized intercept and penalized identity beta."""

    parameters = np.asarray(theta, dtype="<f8")
    features = np.asarray(phi, dtype="<f8", order="C")
    y = _binary_labels(labels).astype(np.float64)
    if (
        features.ndim != 2
        or parameters.shape != (features.shape[1] + 1,)
        or len(features) != len(y)
        or not np.isfinite(features).all()
        or not np.isfinite(parameters).all()
        or not math.isfinite(float(l2))
        or l2 <= 0.0
    ):
        raise TransferClaimControlError("Control objective inputs are invalid")
    if offset is None:
        fixed = np.zeros(len(y), dtype="<f8")
    else:
        fixed = np.asarray(offset, dtype="<f8")
        if fixed.shape != (len(y),) or not np.isfinite(fixed).all():
            raise TransferClaimControlError("Control offset is invalid")
    intercept = float(parameters[0])
    beta = parameters[1:]
    eta = fixed + intercept + features @ beta
    data_loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    probabilities = _stable_sigmoid(eta)
    residual = probabilities - y
    gradient = np.empty_like(parameters)
    gradient[0] = float(np.mean(residual))
    gradient[1:] = features.T @ residual / len(y) + float(l2) * beta
    objective = data_loss + 0.5 * float(l2) * float(beta @ beta)
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise TransferClaimControlError("Control objective became non-finite")
    return objective, np.ascontiguousarray(gradient, dtype="<f8")


def fit_control(
    phi: np.ndarray,
    labels: np.ndarray,
    l2: float,
    offset: np.ndarray | None,
) -> dict[str, Any]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise TransferClaimControlError("SciPy is required for V4 controls") from exc

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return objective_and_gradient(theta, phi, labels, l2, offset)

    result = minimize(
        evaluate,
        np.zeros(phi.shape[1] + 1, dtype="<f8"),
        jac=True,
        method="L-BFGS-B",
        bounds=None,
        options={
            "gtol": 1e-8,
            "ftol": 0.0,
            "maxiter": 10000,
            "maxfun": 200000,
            "maxls": 50,
        },
    )
    theta = np.ascontiguousarray(result.x, dtype="<f8")
    objective, gradient = evaluate(theta)
    gradient_norm = float(np.max(np.abs(gradient)))
    if (
        not bool(result.success)
        or not np.isfinite(theta).all()
        or gradient_norm > 1e-7
    ):
        raise TransferClaimControlError(
            "Control optimizer did not converge: "
            f"success={bool(result.success)} gradient_inf={gradient_norm}"
        )
    return {
        "theta": theta,
        "objective": objective,
        "gradient_infinity_norm": gradient_norm,
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
    }


def predict_control(
    artifact: Mapping[str, Any],
    phi: np.ndarray,
    offset: np.ndarray | None,
) -> np.ndarray:
    theta = np.asarray(artifact["theta"], dtype="<f8")
    if offset is None:
        fixed = np.zeros(len(phi), dtype="<f8")
    else:
        fixed = np.asarray(offset, dtype="<f8")
    eta = fixed + float(theta[0]) + np.asarray(phi, dtype="<f8") @ theta[1:]
    return _stable_sigmoid(eta)


def _verified_v2_file(relative: str, policy: Mapping[str, Any]) -> Path:
    manifest_spec = policy["frozen_inputs"]["train_development_v2_manifest"]
    manifest_path = ROOT / str(manifest_spec["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {str(entry["path"]): entry for entry in manifest["files"]}
    if relative not in entries:
        raise TransferClaimControlError(f"V2 manifest does not register {relative}")
    path = manifest_path.parent / relative
    spec = entries[relative]
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["size_bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise TransferClaimControlError(f"Frozen V2 artifact drift: {relative}")
    return path


def _load_inputs(policy: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    execution = train_v2.load_execution_policy()
    v3_policy, projection, _formal_root = train_v2._validate_public_prerequisites(
        execution
    )
    train = train_v2._load_public_split(execution, projection, "train")
    development = train_v2._load_public_split(execution, projection, "development")
    private_root = ROOT / str(execution["private_supervision_root"])
    private_paths: dict[str, Path] = {}
    for label, spec in execution["authorized_private_inputs"].items():
        path = private_root / str(spec["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(spec["size_bytes"])
            or sha256_file(path) != str(spec["sha256"])
        ):
            raise TransferClaimControlError(f"Private development input drift: {label}")
        private_paths[label] = path
    layout = execution["expected_layout"]
    train_labels = train_v2._read_labels(
        private_paths["train_labels"],
        train,
        expected_rows_per_world=int(layout["rows_per_world"]),
        expected_positive_per_world=int(layout["positive_rows_per_world"]),
    )
    development_labels = train_v2._read_labels(
        private_paths["development_labels"],
        development,
        expected_rows_per_world=int(layout["rows_per_world"]),
        expected_positive_per_world=int(layout["positive_rows_per_world"]),
    )
    relevance = train_v2._read_qrels_relevance(
        private_paths["development_qrels"], development
    )
    if not np.array_equal(relevance, development_labels):
        raise TransferClaimControlError("Development labels and qrels disagree")
    return execution, v3_policy, train, development, train_labels, development_labels, relevance


def _evaluate_model(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: Mapping[str, Any],
    relevance: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    threshold = core.select_development_threshold(
        labels, probabilities, rows["world_ordinals"]
    )
    report = core.complete_classification_report(
        labels, probabilities, rows["world_ordinals"], threshold
    )
    report["retrieval"] = core.retrieval_report(
        probabilities,
        rows["world_ordinals"],
        rows["seller_uid_left"],
        rows["seller_uid_right"],
        relevance,
    )["aggregate"]
    return threshold, report


def validate_contract() -> dict[str, Any]:
    policy = load_policy()
    execution = train_v2.load_execution_policy()
    train_v2._validate_public_prerequisites(execution)
    return {
        "status": "PASSED_V4_CONTROL_CONTRACT_NO_TRUTH_READ",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
    }


def run() -> dict[str, Any]:
    policy = load_policy()
    (
        _execution,
        v3_policy,
        train,
        development,
        train_labels,
        development_labels,
        relevance,
    ) = _load_inputs(policy)
    output = ROOT / str(policy["output_root"])
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise TransferClaimControlError("V4 output already exists and cannot be overwritten")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        scale, mu = core.common_v1.fit_identity_transform(
            train["identity33"], train["world_uids"]
        )
        train_phi, _train_active = core.common_v1.apply_identity_transform(
            train["identity33"], scale, mu
        )
        development_phi, _development_active = core.common_v1.apply_identity_transform(
            development["identity33"], scale, mu
        )
        train_offset = common_v2.raw_logit(train["m0_probability"])
        development_offset = common_v2.raw_logit(development["m0_probability"])
        l2 = float(policy["fit"]["l2"])
        artifacts = {
            "i0_identity_only": fit_control(
                train_phi, train_labels, l2, offset=None
            ),
            "t1_m0_plus_identity": fit_control(
                train_phi, train_labels, l2, offset=train_offset
            ),
        }
        predictions = {
            "i0_identity_only": predict_control(
                artifacts["i0_identity_only"], development_phi, offset=None
            ),
            "t1_m0_plus_identity": predict_control(
                artifacts["t1_m0_plus_identity"],
                development_phi,
                offset=development_offset,
            ),
            "m2_v3_reference": np.load(
                _verified_v2_file("predictions/development/m2.npy", policy),
                allow_pickle=False,
            ),
        }
        if set(predictions) != set(MODEL_IDS):
            raise TransferClaimControlError("V4 prediction registry drift")
        thresholds: dict[str, float] = {}
        reports: dict[str, Any] = {}
        for model_id in MODEL_IDS:
            values = np.ascontiguousarray(predictions[model_id], dtype="<f8")
            if values.shape != (int(policy["fit"]["development_rows"]),):
                raise TransferClaimControlError(f"Prediction shape drift: {model_id}")
            threshold, report = _evaluate_model(
                values, development_labels, development, relevance
            )
            thresholds[model_id] = float(threshold)
            reports[model_id] = report

        indices = core.build_bootstrap_indices(v3_policy, "development")
        bootstrap = {
            model_id: core.bootstrap_pooled_score_metrics(
                development_labels,
                predictions[model_id],
                development["world_ordinals"],
                indices,
            )["average_precision"]
            for model_id in ("i0_identity_only", "t1_m0_plus_identity")
        }
        delta_series = (
            bootstrap["t1_m0_plus_identity"] - bootstrap["i0_identity_only"]
        )
        point_delta = float(
            reports["t1_m0_plus_identity"]["pooled"]["average_precision"]
            - reports["i0_identity_only"]["pooled"]["average_precision"]
        )
        interval = np.quantile(delta_series, [0.025, 0.975], method="linear")
        gate = policy["primary_development_diagnostic"]
        passed = (
            point_delta > float(gate["practical_delta_strictly_greater_than"])
            and float(interval[0])
            > float(gate["interval_lower_bound_strictly_greater_than"])
        )
        comparison = {
            "estimand": gate["estimand"],
            "point_delta": point_delta,
            "percentile_95_interval": [float(interval[0]), float(interval[1])],
            "bootstrap_replicates": int(len(delta_series)),
            "bootstrap_index_sha256": hashlib.sha256(
                indices.tobytes(order="C")
            ).hexdigest(),
            "practical_delta_threshold": float(
                gate["practical_delta_strictly_greater_than"]
            ),
            "development_hypothesis_gate_passed": bool(passed),
        }
        for model_id, artifact in artifacts.items():
            model_root = building / "models" / model_id
            save_array(model_root / "theta.npy", artifact["theta"])
            save_array(model_root / "scale.npy", scale)
            save_array(model_root / "mu.npy", mu)
            write_json(
                model_root / "fit.json",
                {
                    "model_id": model_id,
                    "l2": l2,
                    "intercept": float(artifact["theta"][0]),
                    "objective": float(artifact["objective"]),
                    "gradient_infinity_norm": float(
                        artifact["gradient_infinity_norm"]
                    ),
                    "optimizer_status": int(artifact["optimizer_status"]),
                    "optimizer_message": str(artifact["optimizer_message"]),
                    "m0_offset_used": model_id == "t1_m0_plus_identity",
                },
            )
        for model_id, values in predictions.items():
            save_array(building / "predictions" / f"{model_id}.npy", values)
        write_json(building / "development_thresholds.json", thresholds)
        write_json(
            building / "development_evaluation.json",
            {
                "status": "POSTHOC_DEVELOPMENT_CONTROL_COMPLETE_AUDIT_TRUTH_SEALED",
                "models": reports,
                "primary_comparison": comparison,
                "claim_boundary": policy["claim_boundary"],
            },
        )
        write_json(
            building / "control_summary.json",
            {
                "status": "POSTHOC_DEVELOPMENT_CONTROL_COMPLETE_AUDIT_TRUTH_SEALED",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "l2": l2,
                "models": list(MODEL_IDS),
                "primary_comparison": comparison,
                "truth_read_counts": {
                    "train_labels": 1,
                    "development_labels": 1,
                    "development_qrels": 1,
                    "audit_a_labels_or_qrels": 0,
                    "audit_b_labels_or_qrels": 0,
                },
                "audit_predictions_created": False,
            },
        )
        payload_files = [
            file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        manifest = {
            "status": "POSTHOC_DEVELOPMENT_CONTROL_COMPLETE_AUDIT_TRUTH_SEALED",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "producer": {
                "path": Path(__file__).relative_to(ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__)),
            },
            "files": payload_files,
            "audit_a_truth_reads": 0,
            "audit_b_truth_reads": 0,
        }
        write_json(building / "manifest.json", manifest)
        building.replace(output)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    return {
        "status": "POSTHOC_DEVELOPMENT_CONTROL_COMPLETE_AUDIT_TRUTH_SEALED",
        "output_root": output.relative_to(ROOT).as_posix(),
        "primary_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-contract", "run"))
    args = parser.parse_args()
    result = validate_contract() if args.command == "validate-contract" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
