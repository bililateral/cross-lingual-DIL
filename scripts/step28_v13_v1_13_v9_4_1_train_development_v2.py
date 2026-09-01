#!/usr/bin/env python3
"""Train V9.4.1 models and evaluate the development split only (V2).

This is the minimal supervised wrapper around the frozen V3 numerical core.
It may read train/development supervision named by its execution policy.  It
cannot read audit A/B truth and it never creates audit predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_confirmatory_evaluator_v3 as evaluator
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_train_development_execution_policy_v2.json"
)
POLICY_SIZE_BYTES = 4474
POLICY_SHA256 = "2e73edbe28ae9c001b1aa3a72b16dd90bd342897bcf17782c3e87720a10d249d"
POLICY_CANONICAL_SELF_HASH = (
    "52c41288b62388f1a7d4b2704a8fd6a13dd1e3c9683c8b385cc8cf314dee72a4"
)
ROW_KEY_FIELDS = (
    "split",
    "world_ordinal",
    "world_uid",
    "canonical_pair_uid",
    "seller_uid_left",
    "seller_uid_right",
)
LABEL_FIELDS = ("canonical_pair_uid", "world_uid", "label")
QREL_FIELDS = ("world_uid", "query_seller_uid", "relevant_seller_uids")


class TrainDevelopmentError(ValueError):
    """Raised when the train/development execution contract is violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_file_record(base: Path, spec: Mapping[str, Any], label: str) -> Path:
    path = base / str(spec["path"])
    if not path.is_file():
        raise TrainDevelopmentError(f"Missing {label}: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise TrainDevelopmentError(f"{label} size drift")
    if _sha256_file(path) != str(spec["sha256"]):
        raise TrainDevelopmentError(f"{label} SHA-256 drift")
    return path


def _load_json_with_self_hash(
    path: Path, expected_self_hash: str, label: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if recorded != expected_self_hash or _canonical_sha256(body) != recorded:
        raise TrainDevelopmentError(f"{label} canonical self-hash drift")
    return value


def load_execution_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY.resolve():
        raise TrainDevelopmentError("Only the frozen execution policy path is valid")
    raw = path.read_bytes()
    if len(raw) != POLICY_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise TrainDevelopmentError("Execution policy bytes drift")
    policy = json.loads(raw.decode("utf-8"))
    recorded = policy.get("canonical_self_hash")
    body = dict(policy)
    body.pop("canonical_self_hash", None)
    if recorded != POLICY_CANONICAL_SELF_HASH or _canonical_sha256(body) != recorded:
        raise TrainDevelopmentError("Execution policy canonical self-hash drift")
    if policy.get("status") != "TRAIN_DEVELOPMENT_AUTHORIZED_AUDIT_TRUTH_SEALED":
        raise TrainDevelopmentError("Execution policy status drift")
    if policy.get("version") != "step28-v13-v1.13-v9.4.1-train-development-execution-v2":
        raise TrainDevelopmentError("Execution policy version drift")
    authorization = policy.get("authorization", {})
    if authorization != {
        "train_development_truth_authorized": True,
        "model_training_authorized": True,
        "audit_a_blind_prediction_authorized": False,
        "audit_a_truth_authorized": False,
        "audit_b_blind_prediction_authorized": False,
        "audit_b_truth_authorized": False,
    }:
        raise TrainDevelopmentError("Execution authorization boundary drift")
    if set(policy["authorized_private_inputs"]) != {
        "train_labels",
        "development_labels",
        "development_qrels",
    }:
        raise TrainDevelopmentError("Private input allow-list drift")
    if policy["truth_read_budget"] != {
        "train_labels": 1,
        "development_labels": 1,
        "development_qrels": 1,
        "train_qrels": 0,
        "audit_a_labels_or_qrels": 0,
        "audit_b_labels_or_qrels": 0,
    }:
        raise TrainDevelopmentError("Truth-read budget drift")
    return policy


def _validate_public_prerequisites(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for label, spec in policy["frozen_code_inputs"].items():
        _verify_file_record(ROOT, spec, label)
    v3_policy = core.load_policy()
    frozen = policy["frozen_public_inputs"]
    if v3_policy["canonical_self_hash"] != frozen["v3_training_policy"][
        "canonical_self_hash"
    ]:
        raise TrainDevelopmentError("Frozen V3 policy binding drift")
    loaded: dict[str, dict[str, Any]] = {}
    for label in ("public_projection_manifest", "formal_root_manifest", "formal_quality_result"):
        spec = frozen[label]
        path = _verify_file_record(ROOT, spec, label)
        loaded[label] = _load_json_with_self_hash(
            path, str(spec["canonical_self_hash"]), label
        )
    projection = loaded["public_projection_manifest"]
    if projection.get("status") != (
        "FROZEN_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION_TRAINING_INPUT_READY"
    ):
        raise TrainDevelopmentError("Public projection is not training-input ready")
    quality = loaded["formal_quality_result"]
    if (
        quality.get("status")
        != "PASSED_FORMAL_500X4_ROOT_QUALITY_TRAINING_QUALIFIED"
        or quality.get("training_qualified") is not True
        or quality.get("truth_access", {}).get("audit_a_truth_semantic_reads") != 0
        or quality.get("truth_access", {}).get("audit_b_truth_semantic_reads") != 0
    ):
        raise TrainDevelopmentError("Formal quality result is not eligible for training")
    return v3_policy, projection, loaded["formal_root_manifest"]


def _manifest_entry(manifest: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    matches = [entry for entry in manifest["splits"] if entry["split"] == split]
    if len(matches) != 1:
        raise TrainDevelopmentError(f"Manifest does not contain one {split} entry")
    return matches[0]


def _read_row_keys(
    path: Path,
    split: str,
    *,
    expected_rows: int,
    expected_worlds: int,
    expected_rows_per_world: int,
) -> dict[str, Any]:
    world_ordinals: list[int] = []
    world_uids: list[str] = []
    pair_uids: list[str] = []
    seller_left: list[str] = []
    seller_right: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_KEY_FIELDS:
            raise TrainDevelopmentError(f"{split} row-key header drift")
        for row in reader:
            if row["split"] != split:
                raise TrainDevelopmentError(f"{split} row-key split drift")
            try:
                ordinal = int(row["world_ordinal"])
            except ValueError as exc:
                raise TrainDevelopmentError(f"{split} world ordinal is not an integer") from exc
            left = row["seller_uid_left"]
            right = row["seller_uid_right"]
            if not left or not right or left >= right or row["canonical_pair_uid"] != f"{left}||{right}":
                raise TrainDevelopmentError(f"{split} canonical pair drift")
            world_ordinals.append(ordinal)
            world_uids.append(row["world_uid"])
            pair_uids.append(row["canonical_pair_uid"])
            seller_left.append(left)
            seller_right.append(right)
    if len(pair_uids) != expected_rows or len(set(pair_uids)) != expected_rows:
        raise TrainDevelopmentError(f"{split} row-key count or uniqueness drift")
    counts = Counter(zip(world_ordinals, world_uids))
    if len(counts) != expected_worlds or set(world_ordinals) != set(range(expected_worlds)):
        raise TrainDevelopmentError(f"{split} world registry drift")
    if set(counts.values()) != {expected_rows_per_world}:
        raise TrainDevelopmentError(f"{split} rows-per-world drift")
    ordinal_to_uid: dict[int, str] = {}
    for ordinal, uid in counts:
        if ordinal in ordinal_to_uid and ordinal_to_uid[ordinal] != uid:
            raise TrainDevelopmentError(f"{split} ordinal maps to multiple worlds")
        ordinal_to_uid[ordinal] = uid
    if len(ordinal_to_uid) != expected_worlds:
        raise TrainDevelopmentError(f"{split} world ordinal mapping drift")
    return {
        "world_ordinals": np.asarray(world_ordinals, dtype="<i8"),
        "world_uids": world_uids,
        "pair_uids": pair_uids,
        "seller_uid_left": seller_left,
        "seller_uid_right": seller_right,
    }


def _load_array(
    path: Path, shape: tuple[int, ...], dtype: str, label: str
) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape != shape or value.dtype.str != dtype or not value.flags.c_contiguous:
        raise TrainDevelopmentError(f"{label} shape/dtype/order drift")
    if np.issubdtype(value.dtype, np.floating) and np.isinf(value).any():
        raise TrainDevelopmentError(f"{label} contains infinity")
    return value


def _load_public_split(
    policy: Mapping[str, Any],
    projection: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    public_root = ROOT / str(policy["public_projection_root"])
    base_root = public_root / "base_v1"
    identity_root = public_root / "identity_v1"
    base_top_spec = projection["base_manifest_file"]
    identity_top_spec = projection["identity_manifest_file"]
    base_manifest_path = _verify_file_record(public_root, base_top_spec, "base manifest")
    identity_manifest_path = _verify_file_record(
        public_root, identity_top_spec, "identity manifest"
    )
    base_manifest = _load_json_with_self_hash(
        base_manifest_path, projection["base_manifest_canonical_self_hash"], "base manifest"
    )
    identity_manifest = _load_json_with_self_hash(
        identity_manifest_path,
        projection["identity_manifest_canonical_self_hash"],
        "identity manifest",
    )
    base_entry = _manifest_entry(base_manifest, split)
    identity_entry = _manifest_entry(identity_manifest, split)
    base_split_path = _verify_file_record(
        base_root, base_entry["manifest_file"], f"{split} base split manifest"
    )
    base_split = _load_json_with_self_hash(
        base_split_path,
        base_entry["manifest_canonical_self_hash"],
        f"{split} base split manifest",
    )
    layout = policy["expected_layout"]
    row_count = int(layout["rows_per_split"])
    row_key_path = _verify_file_record(
        base_root, base_split["row_keys_file"], f"{split} row keys"
    )
    identity_row_path = _verify_file_record(
        identity_root, identity_entry["row_keys_file"], f"{split} identity row keys"
    )
    if (
        base_split["row_keys_file"]["sha256"]
        != identity_entry["row_keys_file"]["sha256"]
        or _sha256_file(row_key_path) != _sha256_file(identity_row_path)
    ):
        raise TrainDevelopmentError(f"{split} base/identity row-key binding drift")
    rows = _read_row_keys(
        row_key_path,
        split,
        expected_rows=row_count,
        expected_worlds=int(layout["worlds_per_split"]),
        expected_rows_per_world=int(layout["rows_per_world"]),
    )
    base24_path = _verify_file_record(
        base_root, base_split["base24_file"], f"{split} base24"
    )
    m0_path = _verify_file_record(
        base_root, base_split["m0_probability_file"], f"{split} M0 probability"
    )
    c0_path = _verify_file_record(
        base_root, base_split["c0_probability_file"], f"{split} C0 probability"
    )
    identity_path = _verify_file_record(
        identity_root, identity_entry["identity33_file"], f"{split} identity33"
    )
    rows.update(
        {
            "base24": _load_array(
                base24_path, (row_count, int(layout["base_feature_count"])), "<f8", "base24"
            ),
            "m0_probability": _load_array(m0_path, (row_count,), "<f8", "M0 probability"),
            "c0_probability": _load_array(c0_path, (row_count,), "<f8", "C0 probability"),
            "identity33": _load_array(
                identity_path,
                (row_count, int(layout["identity_feature_count"])),
                "<f8",
                "identity33",
            ),
        }
    )
    if split == "train":
        maps = {}
        entries = {entry["repeat_id"]: entry for entry in identity_entry["m1_source_index_files"]}
        if set(entries) != set(layout["m1_repeat_ids"]):
            raise TrainDevelopmentError("Training M1 mapping registry drift")
        for repeat_id in layout["m1_repeat_ids"]:
            path = _verify_file_record(
                identity_root,
                entries[repeat_id]["file"],
                f"training M1 mapping {repeat_id}",
            )
            maps[f"m1_{repeat_id}"] = _load_array(
                path, (row_count,), "<i8", f"M1 mapping {repeat_id}"
            )
        rows["m1_source_indices"] = maps
    return rows


def _read_labels(
    path: Path,
    rows: Mapping[str, Any],
    *,
    expected_rows_per_world: int,
    expected_positive_per_world: int,
) -> np.ndarray:
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_FIELDS:
            raise TrainDevelopmentError("Label header drift")
        for index, row in enumerate(reader):
            if index >= len(rows["pair_uids"]):
                raise TrainDevelopmentError("Label row count exceeds public row keys")
            if (
                row["canonical_pair_uid"] != rows["pair_uids"][index]
                or row["world_uid"] != rows["world_uids"][index]
                or row["label"] not in ("0", "1")
            ):
                raise TrainDevelopmentError(f"Label/public alignment drift at row {index}")
            labels.append(int(row["label"]))
    if len(labels) != len(rows["pair_uids"]):
        raise TrainDevelopmentError("Label/public row count drift")
    result = np.asarray(labels, dtype=np.int8)
    worlds = np.asarray(rows["world_ordinals"], dtype=np.int64)
    counts = np.bincount(worlds, minlength=int(np.max(worlds)) + 1)
    positives = np.bincount(worlds, weights=result, minlength=len(counts))
    if not np.all(counts == expected_rows_per_world) or not np.all(
        positives == expected_positive_per_world
    ):
        raise TrainDevelopmentError("Label class balance per world drift")
    return result


def _read_qrels_relevance(path: Path, rows: Mapping[str, Any]) -> np.ndarray:
    relevance_by_query: dict[tuple[str, str], frozenset[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainDevelopmentError(f"Invalid qrels JSON at line {line_number}") from exc
            if tuple(row) != QREL_FIELDS:
                raise TrainDevelopmentError(f"Qrels schema/order drift at line {line_number}")
            world_uid = row["world_uid"]
            query = row["query_seller_uid"]
            relevant_raw = row["relevant_seller_uids"]
            if (
                not isinstance(world_uid, str)
                or not isinstance(query, str)
                or not isinstance(relevant_raw, list)
                or len(relevant_raw) not in (1, 2)
                or not all(isinstance(value, str) for value in relevant_raw)
                or query in relevant_raw
                or relevant_raw != sorted(set(relevant_raw), key=lambda value: value.encode("utf-8"))
            ):
                raise TrainDevelopmentError(f"Invalid qrels row at line {line_number}")
            key = (world_uid, query)
            if key in relevance_by_query:
                raise TrainDevelopmentError("Duplicate qrels query")
            relevance_by_query[key] = frozenset(relevant_raw)
    sellers_by_world: dict[str, set[str]] = {}
    for world_uid, left, right in zip(
        rows["world_uids"], rows["seller_uid_left"], rows["seller_uid_right"]
    ):
        sellers_by_world.setdefault(world_uid, set()).update((left, right))
    expected_queries = {
        (world_uid, seller)
        for world_uid, sellers in sellers_by_world.items()
        for seller in sellers
    }
    if set(relevance_by_query) != expected_queries:
        raise TrainDevelopmentError("Qrels query universe does not match public sellers")
    for (world_uid, query), relevant in relevance_by_query.items():
        if not set(relevant) <= sellers_by_world[world_uid]:
            raise TrainDevelopmentError("Qrels contains an unknown seller")
        for candidate in relevant:
            if query not in relevance_by_query[(world_uid, candidate)]:
                raise TrainDevelopmentError("Qrels relevance is not symmetric")
    result = np.fromiter(
        (
            int(right in relevance_by_query[(world_uid, left)])
            for world_uid, left, right in zip(
                rows["world_uids"], rows["seller_uid_left"], rows["seller_uid_right"]
            )
        ),
        dtype=np.int8,
        count=len(rows["pair_uids"]),
    )
    return result


def _predict_m3(artifact: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    imputed = core.common_v1.impute_with_medians(matrix, artifact["medians"])
    return np.asarray(artifact["model"].predict_proba(imputed)[:, 1], dtype="<f8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_array(path: Path, value: np.ndarray, dtype: str = "<f8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(value, dtype=dtype), allow_pickle=False)


def _write_m3_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.booster_.model_to_string().encode("utf-8")
    if b"\r\n" in payload or b"\x00" in payload:
        raise TrainDevelopmentError("LightGBM model string is not canonical UTF-8 text")
    path.write_bytes(payload)


def _save_training_artifacts(
    root: Path,
    residual: Mapping[str, Any],
    m3: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    for model_id, artifact in residual["artifacts"].items():
        model_root = root / "models" / model_id
        _save_array(model_root / "beta.npy", artifact["beta"])
        _save_array(model_root / "scale.npy", artifact["scale"])
        _save_array(model_root / "mu.npy", artifact["mu"])
        _write_json(
            model_root / "fit.json",
            {
                "model_id": model_id,
                "l2": float(residual["selected_l2"]),
                "objective": float(artifact["objective"]),
                "gradient_infinity_norm": float(artifact["gradient_infinity_norm"]),
                "optimizer_status": int(artifact["optimizer_status"]),
                "optimizer_message": str(artifact["optimizer_message"]),
            },
        )
    for model_id, artifact in m3.items():
        model_root = root / "models" / model_id
        _write_m3_model(model_root / "model.txt", artifact["model"])
        _save_array(model_root / "medians.npy", artifact["medians"])
        _write_json(
            model_root / "fit.json",
            {
                "model_id": model_id,
                "selected_grid_index": int(artifact["selected_grid_index"]),
                "selected_grid": [float(value) for value in artifact["selected_grid"]],
                "oof_average_precision_by_grid": [
                    float(value) for value in artifact["oof_average_precision_by_grid"]
                ],
            },
        )
    for model_id, values in predictions.items():
        _save_array(root / "predictions" / "development" / f"{model_id}.npy", values)
    for model_id, values in residual["selected_oof_probabilities"].items():
        _save_array(root / "predictions" / "train_oof" / f"{model_id}.npy", values)
    for model_id, artifact in m3.items():
        _save_array(
            root / "predictions" / "train_oof" / f"{model_id}.npy",
            artifact["selected_oof_probabilities"],
        )
    return [
        _file_record(path, root)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]


def _validate_reloaded_m3_models(
    root: Path,
    m3: Mapping[str, Any],
    development_matrices: Mapping[str, np.ndarray],
    expected_predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - formal runtime gate
        raise TrainDevelopmentError("LightGBM is required to replay saved M3 models") from exc
    result: dict[str, Any] = {}
    for model_id in ("m3_base", "m3_joint"):
        model_root = root / "models" / model_id
        model_bytes = (model_root / "model.txt").read_bytes()
        if b"\r\n" in model_bytes or b"\x00" in model_bytes:
            raise TrainDevelopmentError(f"{model_id} serialized model bytes were rewritten")
        booster = lgb.Booster(model_file=str(model_root / "model.txt"))
        medians = np.load(model_root / "medians.npy", allow_pickle=False)
        if not np.array_equal(medians, np.asarray(m3[model_id]["medians"], dtype="<f8")):
            raise TrainDevelopmentError(f"{model_id} serialized medians drift")
        matrix = core.common_v1.impute_with_medians(
            development_matrices[model_id], medians
        )
        replay = np.ascontiguousarray(booster.predict(matrix), dtype="<f8")
        expected = np.ascontiguousarray(expected_predictions[model_id], dtype="<f8")
        if not np.array_equal(replay, expected):
            maximum = float(np.max(np.abs(replay - expected)))
            raise TrainDevelopmentError(
                f"{model_id} disk replay probability bytes drift: max_abs={maximum}"
            )
        result[model_id] = {
            "tree_count": int(booster.num_trees()),
            "probability_byte_match": True,
            "probability_sha256": hashlib.sha256(replay.tobytes(order="C")).hexdigest(),
        }
    return result


def validate_contract() -> dict[str, Any]:
    policy = load_execution_policy()
    v3_policy, projection, _ = _validate_public_prerequisites(policy)
    private_root = ROOT / str(policy["private_supervision_root"])
    for label, spec in policy["authorized_private_inputs"].items():
        path = private_root / str(spec["path"])
        if not path.is_file() or path.stat().st_size != int(spec["size_bytes"]):
            raise TrainDevelopmentError(f"Authorized private input metadata drift: {label}")
    return {
        "status": "PASSED_TRAIN_DEVELOPMENT_CONTRACT_VALIDATION_NO_TRUTH_READ",
        "execution_policy_canonical_self_hash": policy["canonical_self_hash"],
        "v3_policy_canonical_self_hash": v3_policy["canonical_self_hash"],
        "public_projection_canonical_self_hash": projection["canonical_self_hash"],
        "supervision_or_audit_truth_read": False,
        "model_training_performed": False,
    }


def run_training() -> dict[str, Any]:
    policy = load_execution_policy()
    v3_policy, projection, _ = _validate_public_prerequisites(policy)
    output_root = ROOT / str(policy["formal_output_root"])
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists():
        raise TrainDevelopmentError("Formal train/development output already exists")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    truth_reads = {key: 0 for key in policy["truth_read_budget"]}
    try:
        print("[1/6] 校验并加载训练、开发公开特征", flush=True)
        train = _load_public_split(policy, projection, "train")
        development = _load_public_split(policy, projection, "development")
        private_root = ROOT / str(policy["private_supervision_root"])
        private_paths = {
            label: _verify_file_record(private_root, spec, label)
            for label, spec in policy["authorized_private_inputs"].items()
        }
        layout = policy["expected_layout"]
        print("[2/6] 一次性读取并逐行对齐训练、开发标签与开发检索真值", flush=True)
        train_labels = _read_labels(
            private_paths["train_labels"],
            train,
            expected_rows_per_world=int(layout["rows_per_world"]),
            expected_positive_per_world=int(layout["positive_rows_per_world"]),
        )
        truth_reads["train_labels"] = 1
        development_labels = _read_labels(
            private_paths["development_labels"],
            development,
            expected_rows_per_world=int(layout["rows_per_world"]),
            expected_positive_per_world=int(layout["positive_rows_per_world"]),
        )
        truth_reads["development_labels"] = 1
        development_relevance = _read_qrels_relevance(
            private_paths["development_qrels"], development
        )
        truth_reads["development_qrels"] = 1
        if not np.array_equal(development_labels, development_relevance):
            raise TrainDevelopmentError("Development labels and qrels disagree")
        print("[3/6] 训练五个 M1 对照与 M2 身份迁移模块", flush=True)
        residual = core.fit_m1_m2_family(
            train["m0_probability"],
            train["identity33"],
            train_labels,
            train["world_uids"],
            train["m1_source_indices"],
            train["seller_uid_left"],
            train["seller_uid_right"],
        )
        print("[4/6] 训练 M3-base 与 M3-joint", flush=True)
        m3 = core.fit_m3_family(
            train["base24"], train["identity33"], train_labels, train["world_uids"]
        )
        predictions: dict[str, np.ndarray] = {
            "c0": development["c0_probability"],
            "m0": development["m0_probability"],
        }
        for model_id, artifact in residual["artifacts"].items():
            predictions[model_id] = core.predict_residual_model(
                artifact, development["m0_probability"], development["identity33"]
            )
        predictions["m3_base"] = _predict_m3(m3["m3_base"], development["base24"])
        predictions["m3_joint"] = _predict_m3(
            m3["m3_joint"],
            np.ascontiguousarray(
                np.column_stack((development["base24"], development["identity33"])),
                dtype="<f8",
            ),
        )
        if set(predictions) != set(core.MODEL_IDS):
            raise TrainDevelopmentError("Development prediction registry drift")
        print("[5/6] 冻结开发阈值并计算完整分类、检索与自助法指标", flush=True)
        thresholds = {
            "c0": float(policy["frozen_thresholds"]["c0"]),
            "m0": float(policy["frozen_thresholds"]["m0"]),
        }
        for model_id in core.M1_IDS + ("m2", "m3_base", "m3_joint"):
            threshold = core.select_development_threshold(
                development_labels,
                predictions[model_id],
                development["world_ordinals"],
            )
            if not math.isfinite(threshold):
                raise TrainDevelopmentError(f"Non-finite development threshold: {model_id}")
            thresholds[model_id] = threshold
        bootstrap_indices = core.build_bootstrap_indices(v3_policy, "development")
        development_evaluation = evaluator.evaluate_split_from_raw_inputs(
            policy=v3_policy,
            split="development",
            predictions=predictions,
            thresholds=thresholds,
            world_ordinals=development["world_ordinals"],
            seller_uid_left=development["seller_uid_left"],
            seller_uid_right=development["seller_uid_right"],
            labels=development_labels,
            retrieval_relevance=development_relevance,
            actual_bootstrap_indices=bootstrap_indices,
        )
        if truth_reads != policy["truth_read_budget"]:
            raise TrainDevelopmentError("Truth-read budget was not followed exactly")
        print("[6/6] 保存模型、开发预测、完整指标与可复核清单", flush=True)
        _save_training_artifacts(building, residual, m3, predictions)
        m3_replay = _validate_reloaded_m3_models(
            building,
            m3,
            {
                "m3_base": development["base24"],
                "m3_joint": np.ascontiguousarray(
                    np.column_stack(
                        (development["base24"], development["identity33"])
                    ),
                    dtype="<f8",
                ),
            },
            predictions,
        )
        _write_json(building / "development_thresholds.json", thresholds)
        _write_json(building / "development_evaluation.json", development_evaluation)
        runtime = core.validate_supervised_runtime()
        summary = {
            "status": "TRAINING_AND_DEVELOPMENT_EVALUATION_COMPLETE_AUDIT_TRUTH_SEALED",
            "execution_policy_canonical_self_hash": policy["canonical_self_hash"],
            "v3_policy_canonical_self_hash": v3_policy["canonical_self_hash"],
            "public_projection_canonical_self_hash": projection["canonical_self_hash"],
            "train_rows": len(train_labels),
            "development_rows": len(development_labels),
            "train_label_vector_sha256": hashlib.sha256(train_labels.tobytes()).hexdigest(),
            "development_label_vector_sha256": hashlib.sha256(
                development_labels.tobytes()
            ).hexdigest(),
            "selected_shared_l2": float(residual["selected_l2"]),
            "shared_log_loss_by_l2": {
                str(key): float(value) for key, value in residual["shared_loss_by_l2"].items()
            },
            "m3_selected_grid": {
                model_id: {
                    "index": int(artifact["selected_grid_index"]),
                    "value": [float(value) for value in artifact["selected_grid"]],
                }
                for model_id, artifact in m3.items()
            },
            "m3_disk_replay": m3_replay,
            "development_gate": development_evaluation["gate"],
            "truth_read_counts": truth_reads,
            "audit_a_truth_reads": 0,
            "audit_b_truth_reads": 0,
            "audit_predictions_created": False,
            "runtime": runtime | {
                "platform": platform.platform(),
                "python_executable": sys.executable,
            },
        }
        _write_json(building / "training_summary.json", summary)
        output_files = [
            _file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        manifest = {
            "status": summary["status"],
            "execution_policy_canonical_self_hash": policy["canonical_self_hash"],
            "development_gate_status": development_evaluation["gate"]["status"],
            "truth_read_counts": truth_reads,
            "audit_truth_reads": {"audit_a": 0, "audit_b": 0},
            "audit_predictions_created": False,
            "files": output_files,
        }
        manifest["canonical_self_hash"] = _canonical_sha256(manifest)
        _write_json(building / "manifest.json", manifest)
        os.replace(building, output_root)
        return {
            "status": summary["status"],
            "output_root": output_root.relative_to(ROOT).as_posix(),
            "development_gate_status": development_evaluation["gate"]["status"],
            "manifest_canonical_self_hash": manifest["canonical_self_hash"],
            "audit_truth_reads": {"audit_a": 0, "audit_b": 0},
        }
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract", "run"))
    args = parser.parse_args()
    result = validate_contract() if args.command == "validate-contract" else run_training()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
