#!/usr/bin/env python3
"""Freeze base24 plus frozen M0/C0 probabilities from an opaque GPU return.

The command line is validation-only until a later exact-commit authorization
wrapper owns the one-time formal publication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_base24_shared_v2 as shared_base24
import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as gpu_encoder
import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as experiment_common
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as training_common
import step28_v13_v1_13_v9_4_1_prepare_base_projection_v1 as preparer
import step28_v13_v1_13_v9_4_1_prepare_public_projection_v1 as predecessor
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as common
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common


def load_json(path: Path) -> dict[str, Any]:
    return common.load_json(path)


def verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("canonical_self_hash")
    body = dict(value)
    body.pop("canonical_self_hash", None)
    if not isinstance(claimed, str) or common.canonical_sha256(body) != claimed:
        raise common.PublicProjectionContractError(f"{label} self-hash drift")


def verify_relative_file(
    root: Path, spec: Mapping[str, Any], expected_relative: str, *, label: str
) -> Path:
    if spec.get("path") != expected_relative:
        raise common.PublicProjectionContractError(f"{label} path drift")
    path = (root / expected_relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise common.PublicProjectionContractError(f"{label} escapes its root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != int(spec.get("size_bytes", -1))
        or common.sha256_file(path) != spec.get("sha256")
    ):
        raise common.PublicProjectionContractError(f"{label} file pin drift")
    return path


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def matrix_value_sha256(value: np.ndarray) -> str:
    return preparer.matrix_value_sha256(value)


def render_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_row_keys(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != preparer.ROW_KEY_FIELDS:
            raise common.PublicProjectionContractError("Base row-key schema drift")
        return list(reader)


def validate_cpu_stage(
    policy: Mapping[str, Any], cpu_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(cpu_root / "cpu_stage_manifest.json")
    verify_self_hash(manifest, label="base CPU stage manifest")
    if (
        manifest.get("status") != "PREPARED_UNPUBLISHED_LABEL_FREE_CPU_STAGE"
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("total_pair_count") != 756000
        or manifest.get("labels_controllers_membership_qrels_or_audit_truth_read")
        is not False
        or manifest.get("identity33_read") is not False
        or manifest.get("model_training_or_scoring_performed") is not False
    ):
        raise common.PublicProjectionContractError("Base CPU stage lineage drift")
    splits = manifest.get("splits")
    if not isinstance(splits, list) or [row.get("split") for row in splits] != list(
        common.SPLITS
    ) or [row.get("part_id") for row in splits] != [
        f"part_{index:03d}" for index in range(4)
    ]:
        raise common.PublicProjectionContractError("Base CPU split registry drift")
    expected_shape = tuple(policy["formal_outputs"]["base24_shape_per_split"][:1] + [18])
    expected_paths = {"cpu_stage_manifest.json"}
    validated = []
    for row in splits:
        split = str(row["split"])
        row_relative = f"{split}/row_keys.csv"
        legacy_relative = f"{split}/legacy18.npy"
        row_path = verify_relative_file(
            cpu_root, row["row_keys_file"], row_relative, label=f"{split} CPU row keys"
        )
        legacy_path = verify_relative_file(
            cpu_root,
            row["legacy18_file"],
            legacy_relative,
            label=f"{split} legacy18",
        )
        expected_paths.update((row_relative, legacy_relative))
        row_keys = read_row_keys(row_path)
        legacy18 = np.load(legacy_path, allow_pickle=False)
        if (
            len(row_keys) != expected_shape[0]
            or legacy18.shape != expected_shape
            or legacy18.dtype.str != "<f8"
            or not legacy18.flags.c_contiguous
            or not np.isfinite(legacy18).all()
            or row.get("legacy18_shape") != list(legacy18.shape)
            or row.get("legacy18_dtype") != legacy18.dtype.str
            or row.get("legacy18_value_sha256") != matrix_value_sha256(legacy18)
        ):
            raise common.PublicProjectionContractError("Base CPU matrix drift")
        validated.append(
            {
                "split": split,
                "part_id": str(row["part_id"]),
                "row_keys": row_keys,
                "row_path": row_path,
                "legacy18": np.ascontiguousarray(legacy18, dtype="<f8"),
            }
        )
    actual_paths = {
        path.relative_to(cpu_root).as_posix()
        for path in cpu_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.PublicProjectionContractError("Base CPU file universe drift")
    return manifest, validated


def _stringify_row_key(row: Mapping[str, Any]) -> dict[str, str]:
    return {name: str(row[name]) for name in preparer.ROW_KEY_FIELDS}


def rebind_unpublished_inputs_to_formal_sources(
    policy: Mapping[str, Any],
    cpu_parts: Sequence[Mapping[str, Any]],
    transfer_parts: Sequence[Mapping[str, Any]],
) -> None:
    experiment_policy = experiment_common.load_policy()
    successor = training_common.load_policy()
    reference, _seller_records, _pairs = shared_base24.reconstruct_frozen_english_public(
        successor
    )
    transfer_by_id = {str(row["part_id"]): row for row in transfer_parts}
    for cpu in cpu_parts:
        split = str(cpu["split"])
        part_id = str(cpu["part_id"])
        paths = predecessor.verify_split_public_inputs(
            experiment_policy, split, predecessor.BASE_PUBLIC_ROLES
        )
        worlds = list(preparer.iter_jsonl(paths["worlds.jsonl"]))
        sellers = list(preparer.iter_jsonl(paths["sellers.jsonl"]))
        pairs = preparer.read_csv(paths["complete_model_pair_endpoints.csv"])
        expected_row_keys, seller_uids, seller_worlds = preparer.validate_public_row_order(
            split,
            worlds,
            sellers,
            pairs,
            policy["formal_dataset"]["counts_per_split"],
        )
        if cpu["row_keys"] != [_stringify_row_key(row) for row in expected_row_keys]:
            raise common.PublicProjectionContractError("CPU row keys are not source-bound")
        profiles = {
            str(row["seller_uid"]): shared_base24.project_model_profile(row)
            for row in preparer.iter_jsonl(paths["model_seller_profiles.jsonl"])
        }
        if set(profiles) != set(seller_uids):
            raise common.PublicProjectionContractError("Rebound profile universe drift")
        expected_legacy18 = shared_base24.legacy18_matrix(
            pairs, profiles, reference, policy["feature_contract"]["legacy18"]
        )
        if not np.array_equal(cpu["legacy18"], expected_legacy18):
            raise common.PublicProjectionContractError("legacy18 is not source-bound")
        unique_rows, seller_rows, opaque_sellers = preparer.build_opaque_text_workload(
            preparer.iter_jsonl(paths["redacted_items.jsonl"]),
            valid_worlds={str(row["world_uid"]) for row in worlds},
            seller_uids=seller_uids,
            seller_worlds=seller_worlds,
            expected_item_count=int(policy["formal_dataset"]["counts_per_split"]["items"]),
        )
        expected_pairs = preparer.opaque_pair_rows(pairs, opaque_sellers)
        transfer = transfer_by_id[part_id]
        if (
            transfer["text_rows"] != unique_rows
            or transfer["seller_rows"] != seller_rows
            or transfer["pair_rows"] != expected_pairs
        ):
            raise common.PublicProjectionContractError("GPU transfer is not source-bound")


def impute(raw: np.ndarray, medians: np.ndarray) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or medians.shape != (matrix.shape[1],) or np.isinf(matrix).any():
        raise common.PublicProjectionContractError("Frozen-model input contract drift")
    missing = np.isnan(matrix)
    output = np.where(missing, medians[None, :], matrix) if missing.any() else matrix
    output = np.ascontiguousarray(output, dtype="<f8")
    if not np.isfinite(output).all():
        raise common.PublicProjectionContractError("Frozen-model imputation failed")
    return output


def predict_positive(model: Any, matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if raw.shape != (len(matrix), 2):
        raise common.PublicProjectionContractError("Frozen-model probability shape drift")
    return training_common.validate_p0(
        np.ascontiguousarray(raw[:, 1], dtype="<f8")
    )


def finalize_to_temporary(
    policy: Mapping[str, Any],
    cpu_root: Path,
    transfer_root: Path,
    gpu_return_root: Path,
    base_root: Path,
) -> dict[str, Any]:
    """Build an unpublished base projection for a future authorized wrapper."""

    if base_root.exists():
        raise common.PublicProjectionContractError("Base output root already exists")
    cpu_manifest, cpu_parts = validate_cpu_stage(policy, cpu_root)
    gpu_policy = gpu_common.load_policy()
    transfer_manifest, transfer_parts = gpu_encoder.validate_transfer(
        gpu_policy, transfer_root
    )
    if transfer_manifest.get("cpu_stage_canonical_self_hash") != cpu_manifest[
        "canonical_self_hash"
    ] or transfer_manifest.get("public_policy_canonical_self_hash") != policy[
        "canonical_self_hash"
    ]:
        raise common.PublicProjectionContractError("CPU/transfer lineage drift")
    gpu_manifest, gpu_parts = gpu_encoder.validate_gpu_return(
        gpu_policy, transfer_manifest, transfer_parts, gpu_return_root
    )
    rebind_unpublished_inputs_to_formal_sources(policy, cpu_parts, transfer_parts)
    experiment_policy = experiment_common.load_policy()
    runtime = experiment_common.validate_supervised_cpu_runtime(experiment_policy)
    models = experiment_common.validate_frozen_model_payloads(experiment_policy)
    gpu_by_id = {str(row["part_id"]): row for row in gpu_parts}
    base_root.mkdir(parents=True)
    split_records = []
    try:
        for cpu in cpu_parts:
            split = str(cpu["split"])
            labse6 = gpu_by_id[str(cpu["part_id"])]["values"]
            base24 = shared_base24.combine_base24(cpu["legacy18"], labse6)
            m0_matrix = impute(base24, models["m0"]["imputation_medians"])
            c0_matrix = impute(base24[:, :18], models["c0"]["imputation_medians"])
            m0_probability = predict_positive(models["m0"]["model"], m0_matrix)
            c0_probability = predict_positive(models["c0"]["model"], c0_matrix)
            split_root = base_root / split
            split_root.mkdir()
            row_path = split_root / "row_keys.csv"
            base_path = split_root / "base24.npy"
            m0_path = split_root / "m0_probability.npy"
            c0_path = split_root / "c0_probability.npy"
            shutil.copyfile(cpu["row_path"], row_path)
            np.save(base_path, base24, allow_pickle=False)
            np.save(m0_path, m0_probability, allow_pickle=False)
            np.save(c0_path, c0_probability, allow_pickle=False)
            split_manifest = {
                "step": "step28_v13_v1_13_v9_4_1_base_projection_split_v1",
                "status": "FROZEN_LABEL_FREE_BASE24_M0_C0_NO_MODEL_TRAINING",
                "split": split,
                "part_id": cpu["part_id"],
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "cpu_stage_canonical_self_hash": cpu_manifest["canonical_self_hash"],
                "transfer_manifest_canonical_self_hash": transfer_manifest[
                    "canonical_self_hash"
                ],
                "gpu_return_manifest_canonical_self_hash": gpu_manifest[
                    "canonical_self_hash"
                ],
                "row_keys_file": file_record(row_path, base_root),
                "base24_file": file_record(base_path, base_root),
                "base24_shape": list(base24.shape),
                "base24_dtype": base24.dtype.str,
                "base24_value_sha256": matrix_value_sha256(base24),
                "m0_probability_file": file_record(m0_path, base_root),
                "m0_probability_value_sha256": matrix_value_sha256(m0_probability),
                "c0_probability_file": file_record(c0_path, base_root),
                "c0_probability_value_sha256": matrix_value_sha256(c0_probability),
                "m0_model_sha256": experiment_policy["frozen_models"]["m0"]["sha256"],
                "c0_model_sha256": experiment_policy["frozen_models"]["c0"]["sha256"],
                "supervision_or_audit_truth_read": False,
                "identity33_read": False,
                "model_parameters_updated": False,
                "threshold_selected": False,
            }
            split_manifest["canonical_self_hash"] = common.canonical_sha256(
                split_manifest
            )
            manifest_path = split_root / "split_manifest.json"
            render_json(manifest_path, split_manifest)
            split_records.append(
                {
                    "split": split,
                    "manifest_file": file_record(manifest_path, base_root),
                    "manifest_canonical_self_hash": split_manifest[
                        "canonical_self_hash"
                    ],
                }
            )
        root_manifest = {
            "step": "step28_v13_v1_13_v9_4_1_base_projection_v1",
            "status": "FROZEN_LABEL_FREE_FOUR_SPLIT_BASE_PROJECTION_NO_MODEL_TRAINING",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "cpu_stage_canonical_self_hash": cpu_manifest["canonical_self_hash"],
            "transfer_manifest_canonical_self_hash": transfer_manifest[
                "canonical_self_hash"
            ],
            "gpu_return_manifest_canonical_self_hash": gpu_manifest[
                "canonical_self_hash"
            ],
            "supervised_cpu_runtime_used_for_frozen_scoring": runtime,
            "splits": split_records,
            "total_world_count": 2000,
            "total_pair_count": 756000,
            "supervision_or_audit_truth_read": False,
            "identity33_read": False,
            "model_parameters_updated": False,
            "threshold_selected": False,
            "training_authorized": False,
        }
        root_manifest["canonical_self_hash"] = common.canonical_sha256(root_manifest)
        render_json(base_root / "base_projection_manifest.json", root_manifest)
    except BaseException:
        shutil.rmtree(base_root, ignore_errors=True)
        raise
    validate_base_output(policy, base_root)
    return root_manifest


def validate_base_output(policy: Mapping[str, Any], base_root: Path) -> dict[str, Any]:
    manifest = load_json(base_root / "base_projection_manifest.json")
    verify_self_hash(manifest, label="base projection manifest")
    if (
        manifest.get("status")
        != "FROZEN_LABEL_FREE_FOUR_SPLIT_BASE_PROJECTION_NO_MODEL_TRAINING"
        or manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("total_world_count") != 2000
        or manifest.get("total_pair_count") != 756000
        or manifest.get("supervision_or_audit_truth_read") is not False
        or manifest.get("identity33_read") is not False
        or manifest.get("model_parameters_updated") is not False
        or manifest.get("threshold_selected") is not False
        or manifest.get("training_authorized") is not False
    ):
        raise common.PublicProjectionContractError("Base projection root lineage drift")
    splits = manifest.get("splits")
    if not isinstance(splits, list) or [row.get("split") for row in splits] != list(
        common.SPLITS
    ):
        raise common.PublicProjectionContractError("Base projection split order drift")
    expected_paths = {"base_projection_manifest.json"}
    expected_shape = tuple(policy["formal_outputs"]["base24_shape_per_split"])
    probability_shape = tuple(policy["formal_outputs"]["probability_shape_per_split"])
    for record in splits:
        split = str(record["split"])
        manifest_relative = f"{split}/split_manifest.json"
        split_path = verify_relative_file(
            base_root,
            record["manifest_file"],
            manifest_relative,
            label=f"{split} base manifest",
        )
        expected_paths.add(manifest_relative)
        split_manifest = load_json(split_path)
        verify_self_hash(split_manifest, label=f"{split} base manifest")
        if (
            split_manifest.get("canonical_self_hash")
            != record.get("manifest_canonical_self_hash")
            or split_manifest.get("status")
            != "FROZEN_LABEL_FREE_BASE24_M0_C0_NO_MODEL_TRAINING"
            or split_manifest.get("split") != split
            or split_manifest.get("policy_canonical_self_hash")
            != policy["canonical_self_hash"]
            or split_manifest.get("supervision_or_audit_truth_read") is not False
            or split_manifest.get("identity33_read") is not False
            or split_manifest.get("model_parameters_updated") is not False
            or split_manifest.get("threshold_selected") is not False
        ):
            raise common.PublicProjectionContractError("Base split lineage drift")
        row_relative = f"{split}/row_keys.csv"
        base_relative = f"{split}/base24.npy"
        m0_relative = f"{split}/m0_probability.npy"
        c0_relative = f"{split}/c0_probability.npy"
        row_path = verify_relative_file(
            base_root, split_manifest["row_keys_file"], row_relative, label="base row keys"
        )
        base_path = verify_relative_file(
            base_root, split_manifest["base24_file"], base_relative, label="base24"
        )
        m0_path = verify_relative_file(
            base_root,
            split_manifest["m0_probability_file"],
            m0_relative,
            label="M0 probability",
        )
        c0_path = verify_relative_file(
            base_root,
            split_manifest["c0_probability_file"],
            c0_relative,
            label="C0 probability",
        )
        expected_paths.update((row_relative, base_relative, m0_relative, c0_relative))
        row_keys = read_row_keys(row_path)
        base24 = np.load(base_path, allow_pickle=False)
        m0 = np.load(m0_path, allow_pickle=False)
        c0 = np.load(c0_path, allow_pickle=False)
        if (
            len(row_keys) != expected_shape[0]
            or base24.shape != expected_shape
            or base24.dtype.str != "<f8"
            or not base24.flags.c_contiguous
            or np.isinf(base24).any()
            or split_manifest.get("base24_shape") != list(base24.shape)
            or split_manifest.get("base24_dtype") != base24.dtype.str
            or split_manifest.get("base24_value_sha256")
            != matrix_value_sha256(base24)
        ):
            raise common.PublicProjectionContractError("Published base24 drift")
        for probability, name in ((m0, "m0"), (c0, "c0")):
            probability = training_common.validate_p0(probability)
            if probability.shape != probability_shape or split_manifest.get(
                f"{name}_probability_value_sha256"
            ) != matrix_value_sha256(probability):
                raise common.PublicProjectionContractError(
                    f"Published {name.upper()} probability drift"
                )
    actual_paths = {
        path.relative_to(base_root).as_posix()
        for path in base_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise common.PublicProjectionContractError("Base output file universe drift")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract",))
    parser.parse_args()
    policy = common.load_policy()
    print(
        json.dumps(
            {
                "status": "PASSED_BASE_FINALIZER_CONTRACT_NO_FORMAL_EXECUTION",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "formal_projection_executed": False,
                "supervision_or_audit_truth_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
