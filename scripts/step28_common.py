from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> dict:
    with resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict]:
    with resolve(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: str | Path) -> list[dict]:
    with resolve(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_uid(namespace: str, *parts: object) -> str:
    raw = "|".join([namespace, *(str(part) for part in parts)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def output_root(policy: dict) -> Path:
    return resolve(policy["outputs_root"])


def _write_bytes_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable Step28 artifact differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable Step28 artifact differs: {path}")
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_immutable(path: Path, payload: dict | list) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_immutable(path, data)


def write_jsonl_immutable(path: Path, rows: Iterable[dict]) -> None:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _write_bytes_immutable(path, data.encode("utf-8"))


def write_csv_immutable(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    _write_bytes_immutable(path, buffer.getvalue().encode("utf-8"))


def file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_stage_manifest(path: Path, stage: str, inputs: list[Path], outputs: list[Path], metadata: dict) -> None:
    missing = [str(item) for item in [*inputs, *outputs] if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Step28 manifest cannot close; missing: {missing}")
    payload = {
        "step": 28,
        "stage": stage,
        "complete": True,
        "inputs": [file_record(item) for item in inputs],
        "outputs": [file_record(item) for item in outputs],
        "metadata": metadata,
    }
    write_json_immutable(path, payload)


def validate_stage_manifest(path: Path, expected_stage: str) -> dict:
    payload = load_json(path)
    if payload.get("step") != 28 or payload.get("stage") != expected_stage or payload.get("complete") is not True:
        raise RuntimeError(f"invalid or incomplete Step28 stage manifest: {path}")
    for record in [*payload.get("inputs", []), *payload.get("outputs", [])]:
        artifact = resolve(record["path"])
        if not artifact.is_file():
            raise FileNotFoundError(f"Step28 stage artifact is missing: {artifact}")
        if artifact.stat().st_size != int(record["size_bytes"]) or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"Step28 stage artifact drifted after completion: {artifact}")
    return payload


def validate_policy(policy: dict) -> None:
    if policy.get("step") != 28:
        raise ValueError("not a Step28 policy")
    position = policy["scientific_position"]
    if not position["old_valid_test_access_forbidden"]:
        raise ValueError("Step28 must keep old valid/test sealed")
    for key, value in policy["inputs"].items():
        if key.endswith("_sha256"):
            continue
        lower = str(value).lower()
        if "valid" in lower or "internal_test" in lower or "/test" in lower or "\\test" in lower:
            raise ValueError(f"Step28 input crosses old valid/test boundary: {key}")
    names = policy["model"]["feature_names"]
    forbidden = policy["feature_boundary"]["forbidden_feature_name_fragments"]
    for name in names:
        if any(fragment in name.lower() for fragment in forbidden):
            raise ValueError(f"forbidden Step28 model feature: {name}")


def validate_frozen_inputs(policy: dict) -> list[Path]:
    validate_policy(policy)
    paths = []
    for key, expected in policy["inputs"].items():
        if not key.endswith("_sha256"):
            continue
        base_key = key[: -len("_sha256")]
        path = resolve(policy["inputs"][base_key])
        if sha256_file(path) != expected:
            raise ValueError(f"frozen Step28 input drifted: {base_key}")
        paths.append(path)
    return paths


def frozen_source_artifact(policy: dict) -> dict:
    payload = load_json(policy["inputs"]["frozen_step24_model_artifacts"])
    value: object = payload
    for key in policy["frozen_source_scorer"]["artifact_key"]:
        value = value[key]  # type: ignore[index]
    artifact = value  # type: ignore[assignment]
    expected = policy["frozen_source_scorer"]
    logistic = artifact["logistic_artifact"]
    checks = {
        "feature_name": artifact["feature_names"][0],
        "standardization_mean": logistic["standardization"]["mean"][0],
        "standardization_scale": logistic["standardization"]["scale"][0],
        "parameter_intercept": logistic["parameter_intercept"],
        "parameter_coefficient": logistic["parameter_coefficients"][0],
    }
    for key, observed in checks.items():
        wanted = expected[key]
        if isinstance(wanted, float):
            if not math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"frozen source scorer changed: {key}")
        elif observed != wanted:
            raise ValueError(f"frozen source scorer changed: {key}")
    return artifact


def source_logit_from_cosine(cosine: float, policy: dict) -> float:
    cfg = policy["frozen_source_scorer"]
    standardized = (float(cosine) - float(cfg["standardization_mean"])) / float(cfg["standardization_scale"])
    return float(cfg["parameter_intercept"]) + float(cfg["parameter_coefficient"]) * standardized


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    clipped = np.clip(array, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(probability: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    return np.log(value / (1.0 - value))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    true_positives = 0
    false_positives = 0
    value = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group = y[order[index:end]]
        group_positives = int(group.sum())
        true_positives += group_positives
        false_positives += len(group) - group_positives
        precision = true_positives / max(true_positives + false_positives, 1)
        value += precision * (group_positives / positives)
        index = end
    return float(value)


def weighted_average_precision(
    labels: np.ndarray, scores: np.ndarray, sample_weight: np.ndarray
) -> float:
    """Tie-aware average precision with non-negative observation weights."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    if y.shape != s.shape or y.shape != weight.shape:
        raise ValueError("weighted AP inputs must have identical shapes")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weighted AP requires finite non-negative weights")
    positive_weight = float(np.sum(weight[y == 1]))
    if positive_weight <= 0.0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    cumulative_positive = 0.0
    cumulative_total = 0.0
    value = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group_indices = order[index:end]
        group_positive = float(np.sum(weight[group_indices][y[group_indices] == 1]))
        group_total = float(np.sum(weight[group_indices]))
        cumulative_positive += group_positive
        cumulative_total += group_total
        precision = cumulative_positive / max(cumulative_total, 1e-15)
        value += precision * (group_positive / positive_weight)
        index = end
    return float(value)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def weighted_roc_auc(
    labels: np.ndarray, scores: np.ndarray, sample_weight: np.ndarray
) -> float:
    """Weighted Mann-Whitney AUC; score ties receive half credit."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    weight = np.asarray(sample_weight, dtype=float)
    if y.shape != s.shape or y.shape != weight.shape:
        raise ValueError("weighted AUC inputs must have identical shapes")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weighted AUC requires finite non-negative weights")
    positive_total = float(np.sum(weight[y == 1]))
    negative_total = float(np.sum(weight[y == 0]))
    if positive_total <= 0.0 or negative_total <= 0.0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    negative_below = 0.0
    concordant = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and s[order[end]] == s[order[index]]:
            end += 1
        group_indices = order[index:end]
        group_positive = float(np.sum(weight[group_indices][y[group_indices] == 1]))
        group_negative = float(np.sum(weight[group_indices][y[group_indices] == 0]))
        concordant += group_positive * (negative_below + 0.5 * group_negative)
        negative_below += group_negative
        index = end
    return float(concordant / (positive_total * negative_total))


def feature_vector(row: dict, names: list[str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in names], dtype=float)
