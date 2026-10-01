#!/usr/bin/env python3
"""Audit frozen development predictions; no fitting, qrels or audit-split reads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from itertools import zip_longest
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "reports/step28_model_experiment"
PROJECTION = EXPERIMENT / "v9_4_1_public_projection_v1_20260831"
TRAINING = EXPERIMENT / "v9_4_1_train_development_v2_20260901"
CONTROLS = EXPERIMENT / "v9_4_1_transfer_claim_controls_v4_20260901"
OUTPUT = ROOT / "reports/saved_model_audit/20260906/diagnostic.json"
PINS = {
    "schema/step28_train_development_execution_policy.json":
        "d392d33639567d79db7ca064945e36ddd5815b3540eb614f0ed5765673baa876",
    "schema/step28_model_experiment_policy.json":
        "a473624feec04efd024a4334815fa854b6ec4023b88bf21c709a6c5dccf61a9e",
    "reports/step28_model_experiment/v9_4_1_train_development_v2_20260901/manifest.json":
        "d86266e8abd405494ef409d32d2a5528e95c9cedd348594d063daf6306b033bc",
    "reports/step28_model_experiment/v9_4_1_transfer_claim_controls_v4_20260901/manifest.json":
        "4b0229e58bbec0b6c29b73fea93d18230910e713c1ab232f114dc62c77cb1c72",
}


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_file(base: Path, spec: dict[str, Any], relative: str,
                 records: list[dict[str, Any]]) -> Path:
    if spec["path"] != relative:
        raise ValueError("Unexpected input path")
    path = base / relative
    record = file_record(path)
    if record["sha256"] != spec["sha256"] or (
        "size_bytes" in spec and record["size_bytes"] != spec["size_bytes"]
    ):
        raise ValueError(f"Input hash/size mismatch: {relative}")
    records.append(record)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aligned_development_labels(row_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, worlds, seen = [], [], set()
    with row_path.open(encoding="utf-8", newline="") as rh, \
            label_path.open(encoding="utf-8", newline="") as lh:
        rows, truth = csv.DictReader(rh), csv.DictReader(lh)
        if rows.fieldnames != ["split", "world_ordinal", "world_uid", "canonical_pair_uid",
                               "seller_uid_left", "seller_uid_right"] or truth.fieldnames != [
                                   "canonical_pair_uid", "world_uid", "label"]:
            raise ValueError("Row or label header mismatch")
        for row, label in zip_longest(rows, truth):
            if row is None or label is None:
                raise ValueError("Row/label count mismatch")
            if row["split"] != "development":
                raise ValueError("Only development supervision is allowed")
            left, right = row["seller_uid_left"], row["seller_uid_right"]
            key = row["canonical_pair_uid"]
            if not left or left >= right or key != f"{left}||{right}" or key in seen:
                raise ValueError("Invalid or duplicate canonical pair")
            if key != label["canonical_pair_uid"] or row["world_uid"] != label["world_uid"]:
                raise ValueError("Row/label order mismatch")
            if label["label"] not in ("0", "1"):
                raise ValueError("Nonbinary label")
            seen.add(key)
            labels.append(int(label["label"]))
            worlds.append(row["world_uid"])
    if not labels:
        raise ValueError("Empty development split")
    return np.asarray(labels, dtype=np.int8), np.asarray(worlds)


def curve_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid metric inputs")
    if len(np.unique(y)) != 2:
        return {"ap": None, "trapezoidal_pr_auc": None, "roc_auc": None}
    precision, recall, _ = precision_recall_curve(y, p)
    return {"ap": float(average_precision_score(y, p)),
            "trapezoidal_pr_auc": float(auc(recall, precision)),
            "roc_auc": float(roc_auc_score(y, p))}


def summarize_subset(labels: np.ndarray, scores: np.ndarray, groups: list[np.ndarray],
                     mask: np.ndarray) -> dict[str, Any]:
    selected = labels[mask]
    per_world = [curve_metrics(labels[index[mask[index]]], scores[index[mask[index]]])
                 for index in groups]
    eligible = [point for point in per_world if point["ap"] is not None]
    return {"rows": int(mask.sum()), "positives": int(selected.sum()),
            "negatives": int(len(selected) - selected.sum()),
            "prevalence": float(selected.mean()) if len(selected) else None,
            "pooled": curve_metrics(selected, scores[mask]),
            "world_equal_both_classes_only": {
                key: float(np.mean([point[key] for point in eligible])) if eligible else None
                for key in ("ap", "trapezoidal_pr_auc", "roc_auc")},
            "eligible_worlds": len(eligible), "total_worlds": len(groups)}


def run() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    pinned = {path: read_json(checked_file(ROOT, {"path": path, "sha256": digest}, path, records))
              for path, digest in PINS.items()}
    execution = pinned["schema/step28_train_development_execution_policy.json"]
    features = pinned["schema/step28_model_experiment_policy.json"]["feature_contract"]
    training = pinned[(TRAINING / "manifest.json").relative_to(ROOT).as_posix()]
    controls = pinned[(CONTROLS / "manifest.json").relative_to(ROOT).as_posix()]

    def artifact(base: Path, manifest: dict[str, Any], relative: str) -> Path:
        spec = next(item for item in manifest["files"] if item["path"] == relative)
        return checked_file(base, spec, relative, records)

    projection_spec = execution["frozen_public_inputs"]["public_projection_manifest"]
    projection = read_json(checked_file(ROOT, projection_spec,
                                       (PROJECTION / "public_projection_manifest.json").relative_to(ROOT).as_posix(), records))
    base_manifest = read_json(checked_file(PROJECTION, projection["base_manifest_file"],
                                         "base_v1/base_projection_manifest.json", records))
    identity_manifest = read_json(checked_file(PROJECTION, projection["identity_manifest_file"],
                                             "identity_v1/identity_projection_manifest.json", records))
    matrices, row_files = {}, {}
    for split in ("train", "development"):
        entry = next(item for item in base_manifest["splits"] if item["split"] == split)
        base_split = read_json(checked_file(PROJECTION / "base_v1", entry["manifest_file"],
                                           f"{split}/split_manifest.json", records))
        identity_split = next(item for item in identity_manifest["splits"] if item["split"] == split)
        row_files[split] = checked_file(PROJECTION / "base_v1", base_split["row_keys_file"],
                                       f"{split}/row_keys.csv", records)
        checked_file(PROJECTION / "identity_v1", identity_split["row_keys_file"],
                     f"{split}/row_keys.csv", records)
        if base_split["row_keys_file"]["sha256"] != identity_split["row_keys_file"]["sha256"]:
            raise ValueError("Base/identity row alignment mismatch")
        pair = []
        for root, spec, name, columns in (
            ("base_v1", base_split["base24_file"], "base24.npy", 24),
            ("identity_v1", identity_split["identity33_file"], "identity33.npy", 33),
        ):
            array = np.load(checked_file(PROJECTION / root, spec, f"{split}/{name}", records), allow_pickle=False)
            if array.shape != (identity_split["row_count"], columns) or array.dtype.str != "<f8" or np.isinf(array).any():
                raise ValueError("Public matrix shape/dtype/value mismatch")
            if columns == 33 and not np.isfinite(array).all():
                raise ValueError("Nonfinite identity features")
            pair.append(array)
        matrices[split] = pair

    # This fixed relative path is the only supervision opened by this audit.
    label_path = checked_file(ROOT / execution["private_supervision_root"],
                              execution["authorized_private_inputs"]["development_labels"],
                              "development/pair_labels.csv", records)
    labels, worlds = aligned_development_labels(row_files["development"], label_path)
    summary = read_json(artifact(TRAINING, training, "training_summary.json"))
    if hashlib.sha256(labels.tobytes()).hexdigest() != summary["development_label_vector_sha256"]:
        raise ValueError("Development labels differ from frozen training evaluation")
    base, identity = matrices["development"]
    if len(labels) != len(base):
        raise ValueError("Labels/matrix length mismatch")
    groups = [np.flatnonzero(worlds == world) for world in sorted(set(worlds))]
    names = features["legacy18"] + features["labse6"] + features["identity33"]
    thresholds = read_json(artifact(TRAINING, training, "development_thresholds.json"))
    formal = read_json(artifact(TRAINING, training, "development_evaluation.json"))
    strong = (identity[:, features["identity33"].index("verified_direct_token_count_log1p")] > 0) | (
        identity[:, features["identity33"].index("strong_rotation_path_count_log1p")] > 0)
    zero = np.all(identity == 0, axis=1)
    masks = {"all": np.ones(len(labels), dtype=bool), "strong": strong,
             "no_strong": ~strong, "zero_identity": zero, "active_without_strong": ~strong & ~zero}
    models, replay = {}, {}
    for model_id in ("m0", "m2", "m3_base", "m3_joint", "i0_identity_only"):
        if model_id == "i0_identity_only":
            path = artifact(CONTROLS, controls, "predictions/i0_identity_only.npy")
        else:
            path = artifact(TRAINING, training, f"predictions/development/{model_id}.npy")
        scores = np.load(path, allow_pickle=False)
        if scores.shape != labels.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid frozen prediction array")
        models[model_id] = {name: summarize_subset(labels, scores, groups, mask)
                            for name, mask in masks.items()}
        if model_id != "i0_identity_only":
            for local, saved in (("ap", "average_precision"), ("trapezoidal_pr_auc", "trapezoidal_pr_auc"), ("roc_auc", "roc_auc")):
                if abs(models[model_id]["all"]["pooled"][local] - formal["model_points"][model_id]["pooled"][saved]) > 1e-12:
                    raise ValueError("Independent metrics do not reproduce frozen report")
        if model_id.startswith("m3_"):
            joint = model_id == "m3_joint"
            matrix = np.column_stack((base, identity)) if joint else base
            train = np.column_stack(matrices["train"]) if joint else matrices["train"][0]
            medians = np.load(artifact(TRAINING, training, f"models/{model_id}/medians.npy"), allow_pickle=False)
            if not np.array_equal(np.nanmedian(train, axis=0), medians):
                raise ValueError("Saved medians do not match training-only columns")
            imputed = np.where(np.isnan(matrix), medians, matrix)
            booster = lgb.Booster(model_file=str(artifact(TRAINING, training, f"models/{model_id}/model.txt")))
            predicted = np.asarray(booster.predict(imputed, num_threads=1), dtype="<f8")
            if not np.array_equal(predicted, scores):
                raise ValueError("Saved M3 model does not reproduce development probabilities")
            indices = np.asarray([len(labels) - 1, 0, 731, 19, 2])
            if not np.array_equal(booster.predict(imputed[indices], num_threads=1), scores[indices]):
                raise ValueError("M3 inference depends on row order or batch size")
            gain = booster.feature_importance(importance_type="gain")
            top = sorted(range(len(gain)), key=lambda i: (-gain[i], i))[:12]
            predicted_positive = scores >= thresholds[model_id]
            replay[model_id] = {
                "prediction_max_abs_error": float(np.max(np.abs(predicted - scores))),
                "prediction_value_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
                "training_only_medians_match": True, "five_reordered_rows_match": True,
                "feature_count": booster.num_feature(), "trees": booster.num_trees(),
                "training_gain_by_block": {"base24": float(gain[:24].sum()), "identity33": float(gain[24:].sum())},
                "top_training_gain_features": [{"feature": names[i], "gain": float(gain[i])} for i in top],
                "frozen_threshold": thresholds[model_id],
                "predicted_positive_equals_strong_rule": bool(np.array_equal(predicted_positive, strong)),
                "strong_min_probability": float(scores[strong].min()),
                "no_strong_max_probability": float(scores[~strong].max()),
                "no_strong_at_frozen_threshold": {
                    "tp": int(np.sum(predicted_positive & ~strong & (labels == 1))),
                    "fp": int(np.sum(predicted_positive & ~strong & (labels == 0))),
                    "fn": int(np.sum(~predicted_positive & ~strong & (labels == 1))),
                    "tn": int(np.sum(~predicted_positive & ~strong & (labels == 0)))},
            }
    import sklearn
    return {"status": "EXPLORATORY_SAVED_DEVELOPMENT_MODEL_AUDIT_COMPLETE",
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__},
            "script": file_record(Path(__file__).resolve()), "inputs": records,
            "models": models, "m3_replay": replay,
            "truth_reads": {"development_labels": 1, "train_labels": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
            "limitations": [
                "Post hoc development diagnosis, not new confirmatory performance or real Chinese darknet validation.",
                "Conditional subsets change class prevalence; their AP is not directly comparable to full-set AP.",
                "One-class worlds are excluded from subset curve means, with eligible counts reported; no confidence intervals or significance claims.",
                "The saved matrix-to-model path is replayed. Raw text generation/extraction and original fitting are not re-executed or newly certified.",
                "No fitting, new threshold, frozen-result overwrite, Linux access, audit predictions or audit truth read.",
                "Tree training gain is descriptive, not causal importance or proof of generalization."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError("This audit receipt already exists; do not overwrite research evidence")
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": OUTPUT.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
