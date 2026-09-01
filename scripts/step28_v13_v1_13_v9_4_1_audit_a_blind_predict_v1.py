#!/usr/bin/env python3
"""Publish V9.4.1 Audit-A predictions without reading Audit-A truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import step28_v13_v1_13_v9_4_1_blind_stage_protocol_v3 as blind
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core
import step28_v13_v1_13_v9_4_1_train_development_v2 as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_audit_a_blind_prediction_policy_v1.json"
)


class AuditABlindPredictionError(ValueError):
    """Raised when the frozen blind-prediction boundary is violated."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(training._canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(policy.get("canonical_self_hash", ""))
    candidate = dict(policy)
    candidate.pop("canonical_self_hash", None)
    if len(expected_hash) != 64 or _canonical_sha256(candidate) != expected_hash:
        raise AuditABlindPredictionError("Audit-A blind policy self-hash drift")
    if policy.get("status") != "AUDIT_A_BLIND_PREDICTION_AUTHORIZED_NO_TRUTH":
        raise AuditABlindPredictionError("Audit-A blind policy status drift")
    if policy.get("split") != "audit_a":
        raise AuditABlindPredictionError("Only Audit-A blind prediction is authorized")
    if policy.get("authorization") != {
        "audit_a_blind_prediction_authorized": True,
        "audit_a_truth_authorized": False,
        "audit_b_blind_prediction_authorized": False,
        "audit_b_truth_authorized": False,
    }:
        raise AuditABlindPredictionError("Blind authorization boundary drift")
    if policy.get("private_inputs") != [] or policy.get("truth_read_budget") != {
        "audit_a_labels_or_qrels": 0,
        "audit_b_labels_or_qrels": 0,
    }:
        raise AuditABlindPredictionError("Blind policy exposes a private input")
    return policy


def _load_parent_json(
    spec: Mapping[str, Any], expected_self_hash: str, label: str
) -> dict[str, Any]:
    path = training._verify_file_record(ROOT, spec, label)
    return training._load_json_with_self_hash(path, expected_self_hash, label)


def _manifest_record(manifest: Mapping[str, Any], relative_path: str) -> Mapping[str, Any]:
    matches = [item for item in manifest["files"] if item["path"] == relative_path]
    if len(matches) != 1:
        raise AuditABlindPredictionError(
            f"Training manifest does not contain one {relative_path} record"
        )
    return matches[0]


def _training_path(
    model_root: Path,
    manifest: Mapping[str, Any],
    relative_path: str,
) -> Path:
    return training._verify_file_record(
        model_root,
        _manifest_record(manifest, relative_path),
        f"training artifact {relative_path}",
    )


def _load_thresholds(
    model_root: Path, manifest: Mapping[str, Any]
) -> dict[str, float]:
    path = _training_path(model_root, manifest, "development_thresholds.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != set(core.MODEL_IDS):
        raise AuditABlindPredictionError("Development threshold registry drift")
    thresholds = {model_id: float(raw[model_id]) for model_id in core.MODEL_IDS}
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise AuditABlindPredictionError("Development threshold is not finite")
    return thresholds


def _load_residual_models(
    model_root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for model_id in core.M1_IDS + ("m2",):
        arrays: dict[str, np.ndarray] = {}
        for name in ("beta", "scale", "mu"):
            path = _training_path(
                model_root, manifest, f"models/{model_id}/{name}.npy"
            )
            value = np.load(path, allow_pickle=False)
            if value.shape != (33,) or value.dtype.str != "<f8" or not value.flags.c_contiguous:
                raise AuditABlindPredictionError(
                    f"{model_id} {name} shape/dtype/order drift"
                )
            if not np.isfinite(value).all():
                raise AuditABlindPredictionError(f"{model_id} {name} is not finite")
            arrays[name] = value
        if np.any(arrays["scale"] <= 0.0):
            raise AuditABlindPredictionError(f"{model_id} scale is not positive")
        result[model_id] = arrays
    return result


def _load_m3_models(
    model_root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - formal runtime gate
        raise AuditABlindPredictionError("LightGBM is required for blind prediction") from exc
    result: dict[str, dict[str, Any]] = {}
    for model_id, feature_count in (("m3_base", 24), ("m3_joint", 57)):
        model_path = _training_path(
            model_root, manifest, f"models/{model_id}/model.txt"
        )
        model_bytes = model_path.read_bytes()
        if b"\r\n" in model_bytes or b"\x00" in model_bytes:
            raise AuditABlindPredictionError(f"{model_id} model serialization drift")
        medians_path = _training_path(
            model_root, manifest, f"models/{model_id}/medians.npy"
        )
        medians = np.load(medians_path, allow_pickle=False)
        if (
            medians.shape != (feature_count,)
            or medians.dtype.str != "<f8"
            or not medians.flags.c_contiguous
            or not np.isfinite(medians).all()
        ):
            raise AuditABlindPredictionError(f"{model_id} median artifact drift")
        booster = lgb.Booster(model_file=str(model_path))
        if booster.num_trees() != 200:
            raise AuditABlindPredictionError(f"{model_id} tree-count drift")
        result[model_id] = {"model": booster, "medians": medians}
    return result


def _load_models(
    model_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "residual": _load_residual_models(model_root, manifest),
        "m3": _load_m3_models(model_root, manifest),
    }


def _predict_models(
    rows: Mapping[str, Any], models: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {
        "c0": np.ascontiguousarray(rows["c0_probability"], dtype="<f8"),
        "m0": np.ascontiguousarray(rows["m0_probability"], dtype="<f8"),
    }
    for model_id, artifact in models["residual"].items():
        predictions[model_id] = np.ascontiguousarray(
            core.predict_residual_model(
                artifact, rows["m0_probability"], rows["identity33"]
            ),
            dtype="<f8",
        )
    matrices = {
        "m3_base": rows["base24"],
        "m3_joint": np.ascontiguousarray(
            np.column_stack((rows["base24"], rows["identity33"])), dtype="<f8"
        ),
    }
    for model_id in ("m3_base", "m3_joint"):
        artifact = models["m3"][model_id]
        matrix = core.common_v1.impute_with_medians(
            matrices[model_id], artifact["medians"]
        )
        predictions[model_id] = np.ascontiguousarray(
            artifact["model"].predict(matrix), dtype="<f8"
        )
    if set(predictions) != set(core.MODEL_IDS):
        raise AuditABlindPredictionError("Blind prediction model registry drift")
    row_count = len(rows["pair_uids"])
    for model_id, probability in predictions.items():
        if (
            probability.shape != (row_count,)
            or probability.dtype.str != "<f8"
            or not probability.flags.c_contiguous
            or not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
        ):
            raise AuditABlindPredictionError(f"{model_id} probability artifact drift")
    return predictions


def _validate_development_replay(
    policy: Mapping[str, Any],
    projection: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    models: Mapping[str, Any],
) -> dict[str, str]:
    rows = training._load_public_split(policy, projection, "development")
    observed = _predict_models(rows, models)
    model_root = ROOT / str(policy["training_output_root"])
    result: dict[str, str] = {}
    for model_id in core.MODEL_IDS:
        path = _training_path(
            model_root,
            training_manifest,
            f"predictions/development/{model_id}.npy",
        )
        expected = np.load(path, allow_pickle=False)
        if not np.array_equal(observed[model_id], expected):
            maximum = float(np.max(np.abs(observed[model_id] - expected)))
            raise AuditABlindPredictionError(
                f"Development disk replay drift for {model_id}: max_abs={maximum}"
            )
        result[model_id] = hashlib.sha256(
            observed[model_id].tobytes(order="C")
        ).hexdigest()
    return result


def _load_frozen_inputs(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any]]:
    for label, spec in policy["frozen_code_inputs"].items():
        training._verify_file_record(ROOT, spec, label)
    projection_spec = policy["frozen_public_inputs"]["public_projection_manifest"]
    projection = _load_parent_json(
        projection_spec,
        str(projection_spec["canonical_self_hash"]),
        "public projection manifest",
    )
    training_spec = policy["frozen_public_inputs"]["training_manifest"]
    training_manifest = _load_parent_json(
        training_spec,
        str(training_spec["canonical_self_hash"]),
        "training manifest",
    )
    if training_manifest.get("status") != (
        "TRAINING_AND_DEVELOPMENT_EVALUATION_COMPLETE_AUDIT_TRUTH_SEALED"
    ) or training_manifest.get("development_gate_status") != (
        "PASSED_DEVELOPMENT_M1_M0_EQUIVALENCE_GATE"
    ):
        raise AuditABlindPredictionError("Training parent is not Audit-A eligible")
    if training_manifest.get("audit_truth_reads") != {"audit_a": 0, "audit_b": 0}:
        raise AuditABlindPredictionError("Training parent audit-truth boundary drift")
    model_root = ROOT / str(policy["training_output_root"])
    thresholds = _load_thresholds(model_root, training_manifest)
    models = _load_models(model_root, training_manifest)
    return projection, training_manifest, thresholds, models


def validate_contract() -> dict[str, Any]:
    policy = load_policy()
    projection, training_manifest, thresholds, models = _load_frozen_inputs(policy)
    return {
        "status": "PASSED_AUDIT_A_BLIND_CONTRACT_NO_TRUTH_READ_NO_PREDICTION",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "projection_canonical_self_hash": projection["canonical_self_hash"],
        "training_manifest_canonical_self_hash": training_manifest["canonical_self_hash"],
        "threshold_model_count": len(thresholds),
        "loaded_trained_model_count": len(models["residual"]) + len(models["m3"]),
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
        "prediction_written": False,
    }


def run_blind_prediction() -> dict[str, Any]:
    policy = load_policy()
    projection, training_manifest, thresholds, models = _load_frozen_inputs(policy)
    output_root = ROOT / str(policy["formal_output_root"])
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists():
        raise AuditABlindPredictionError("Formal Audit-A blind output already exists")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        print("[1/4] 重放开发集，核对十个落盘模型与冻结预测", flush=True)
        replay = _validate_development_replay(
            policy, projection, training_manifest, models
        )
        print("[2/4] 加载审核甲公开投影，不读取任何审核真值", flush=True)
        rows = training._load_public_split(policy, projection, "audit_a")
        print("[3/4] 生成并冻结审核甲十组盲预测", flush=True)
        predictions = _predict_models(rows, models)
        row_binding = [
            item
            for item in projection["split_row_key_bindings"]
            if item["split"] == "audit_a"
        ]
        if len(row_binding) != 1:
            raise AuditABlindPredictionError("Audit-A row-key binding drift")
        numerical_payload = blind.build_blind_prediction_payload(
            split="audit_a",
            predictions=predictions,
            thresholds=thresholds,
            row_key_sha256=str(row_binding[0]["row_keys_sha256"]),
            training_parent_sha256=str(training_manifest["canonical_self_hash"]),
        )
        for model_id, probability in predictions.items():
            training._save_array(
                building / "predictions" / f"{model_id}.npy", probability
            )
        _write_json(building / "thresholds.json", thresholds)
        _write_json(building / "blind_numerical_payload.json", numerical_payload)
        summary = {
            "status": "AUDIT_A_BLIND_PREDICTIONS_FROZEN_NO_TRUTH_READ",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "public_projection_canonical_self_hash": projection["canonical_self_hash"],
            "training_manifest_canonical_self_hash": training_manifest[
                "canonical_self_hash"
            ],
            "split": "audit_a",
            "row_count": len(rows["pair_uids"]),
            "model_order": list(core.MODEL_IDS),
            "row_keys_path": (
                "reports/step28_model_experiment/"
                "v9_4_1_public_projection_v1_20260831/base_v1/audit_a/row_keys.csv"
            ),
            "row_keys_sha256": row_binding[0]["row_keys_sha256"],
            "development_disk_replay_exact": True,
            "development_probability_sha256": replay,
            "audit_a_labels_or_qrels_reads": 0,
            "audit_b_labels_or_qrels_reads": 0,
            "model_parameters_updated": False,
            "thresholds_updated": False,
            "audit_b_prediction_created": False,
        }
        _write_json(building / "blind_prediction_summary.json", summary)
        print("[4/4] 重载预测文件并发布完整性清单", flush=True)
        for model_id in core.MODEL_IDS:
            path = building / "predictions" / f"{model_id}.npy"
            reloaded = np.load(path, allow_pickle=False)
            if not np.array_equal(reloaded, predictions[model_id]):
                raise AuditABlindPredictionError(
                    f"Published probability replay drift: {model_id}"
                )
        files = [
            training._file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        manifest = {
            "status": summary["status"],
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "public_projection_canonical_self_hash": projection["canonical_self_hash"],
            "training_manifest_canonical_self_hash": training_manifest[
                "canonical_self_hash"
            ],
            "split": "audit_a",
            "row_count": len(rows["pair_uids"]),
            "model_order": list(core.MODEL_IDS),
            "truth_reads": {"audit_a": 0, "audit_b": 0},
            "files": files,
        }
        manifest["canonical_self_hash"] = _canonical_sha256(manifest)
        _write_json(building / "manifest.json", manifest)
        os.replace(building, output_root)
        return {
            "status": summary["status"],
            "output_root": output_root.relative_to(ROOT).as_posix(),
            "manifest_canonical_self_hash": manifest["canonical_self_hash"],
            "row_count": len(rows["pair_uids"]),
            "model_count": len(predictions),
            "audit_a_truth_reads": 0,
            "audit_b_truth_reads": 0,
        }
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract", "run"))
    args = parser.parse_args()
    result = validate_contract() if args.command == "validate-contract" else run_blind_prediction()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
