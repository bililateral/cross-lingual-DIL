#!/usr/bin/env python3
"""Freeze the four-split label-free identity33 projection and five M1 maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common
import step28_v13_v1_13_v9_4_1_prepare_public_projection_v1 as projection


OUTPUT_SUBDIRECTORY = "identity_v1"
ROW_KEY_FIELDS = (
    "split",
    "world_ordinal",
    "world_uid",
    "canonical_pair_uid",
    "seller_uid_left",
    "seller_uid_right",
)
IMPLEMENTATION_FILES = {
    "common": "scripts/step28_v13_v1_13_v9_4_1_model_experiment_common_v1.py",
    "public_projection": "scripts/step28_v13_v1_13_v9_4_1_prepare_public_projection_v1.py",
    "identity_projection": "scripts/step28_v13_v1_13_v9_4_1_freeze_identity_projection_v1.py",
}


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def implementation_file_records() -> dict[str, dict[str, Any]]:
    records = {}
    for role, relative in IMPLEMENTATION_FILES.items():
        path = common.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing identity implementation file: {path}")
        records[role] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
    return records


def render_manifest(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_pair_uid(left: str, right: str) -> str:
    first, second = common.canonical_pair_endpoints(left, right)
    return f"{first}||{second}"


def validate_world_k28(
    pair_endpoints: Sequence[tuple[str, str]],
) -> list[str]:
    if len(pair_endpoints) != 378 or len(set(pair_endpoints)) != 378:
        raise common.ModelExperimentContractError("World pair universe drift")
    sellers = sorted(
        {endpoint for pair in pair_endpoints for endpoint in pair},
        key=lambda value: value.encode("utf-8"),
    )
    if len(sellers) != 28:
        raise common.ModelExperimentContractError("World seller universe drift")
    expected = {
        common.canonical_pair_endpoints(left, right)
        for index, left in enumerate(sellers)
        for right in sellers[index + 1 :]
    }
    if set(pair_endpoints) != expected:
        raise common.ModelExperimentContractError("World is not complete K28")
    return sellers


def m1_source_indices_for_world(
    world_uid: str,
    pair_endpoints: Sequence[tuple[str, str]],
    repeat_id: str,
) -> np.ndarray:
    sellers = validate_world_k28(pair_endpoints)
    row_by_edge = {edge: index for index, edge in enumerate(pair_endpoints)}
    mapping = common.build_m1_mapping(world_uid, sellers, repeat_id)
    source = np.asarray(
        [row_by_edge[mapping[edge]] for edge in pair_endpoints], dtype="<i8"
    )
    if (
        source.shape != (378,)
        or len(set(int(value) for value in source)) != 378
        or np.any(source == np.arange(378, dtype=np.int64))
    ):
        raise common.ModelExperimentContractError("M1 local source-index drift")
    for destination_index, source_index in enumerate(source):
        if set(pair_endpoints[destination_index]) & set(pair_endpoints[int(source_index)]):
            raise common.ModelExperimentContractError("M1 mapped endpoints overlap")
    return source


def validate_m1_global_indices(
    values: np.ndarray,
    pair_endpoints: Sequence[tuple[str, str]],
) -> str:
    source = np.asarray(values)
    if (
        source.ndim != 1
        or source.dtype.str != "<i8"
        or len(source) != len(pair_endpoints)
        or len(source) == 0
        or len(source) % 378 != 0
    ):
        raise common.ModelExperimentContractError("Published M1 map shape/dtype drift")
    for start in range(0, len(source), 378):
        block = np.asarray(source[start : start + 378], dtype=np.int64)
        local = block - start
        if (
            int(local.min()) < 0
            or int(local.max()) >= 378
            or len(set(int(value) for value in local)) != 378
            or np.any(local == np.arange(378, dtype=np.int64))
        ):
            raise common.ModelExperimentContractError(
                "Published M1 map is not a within-world derangement"
            )
        endpoints = pair_endpoints[start : start + 378]
        validate_world_k28(endpoints)
        for destination, source_index in enumerate(local):
            if set(endpoints[destination]) & set(endpoints[int(source_index)]):
                raise common.ModelExperimentContractError(
                    "Published M1 mapped endpoints overlap"
                )
    return hashlib.sha256(
        np.asarray(source, dtype="<i8").tobytes(order="C")
    ).hexdigest()


def validate_exact_m1_repeat(
    values: np.ndarray,
    pair_endpoints: Sequence[tuple[str, str]],
    world_uids: Sequence[str],
    repeat_id: str,
) -> str:
    value_hash = validate_m1_global_indices(values, pair_endpoints)
    if len(world_uids) * 378 != len(pair_endpoints):
        raise common.ModelExperimentContractError("Published M1 world-key count drift")
    source = np.asarray(values)
    for world_ordinal, world_uid in enumerate(world_uids):
        start = world_ordinal * 378
        expected = m1_source_indices_for_world(
            world_uid,
            pair_endpoints[start : start + 378],
            repeat_id,
        )
        observed = np.asarray(source[start : start + 378], dtype=np.int64) - start
        if not np.array_equal(observed, expected):
            raise common.ModelExperimentContractError(
                f"Published M1 map does not match frozen {repeat_id} construction"
            )
    return value_hash


def _open_csv_reader(path: Path) -> tuple[Any, csv.DictReader]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    return handle, csv.DictReader(handle)


def build_split(
    policy: Mapping[str, Any], split: str, temporary_root: Path
) -> dict[str, Any]:
    paths = projection.verify_split_public_inputs(
        policy, split, projection.IDENTITY_PUBLIC_ROLES
    )
    worlds = projection.load_worlds(paths["worlds.jsonl"], split)
    split_root = temporary_root / split
    split_root.mkdir(parents=True, exist_ok=False)
    row_keys_path = split_root / "row_keys.csv"
    matrix_path = split_root / "identity33.npy"
    identity_names = list(policy["feature_contract"]["identity33"])
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype="<f8", shape=(189000, 33)
    )
    m1_paths: dict[str, Path] = {}
    m1_arrays: dict[str, np.memmap] = {}
    if split == "train":
        for repeat_id in policy["m1"]["repeat_ids"]:
            path = split_root / f"m1_source_row_index_{repeat_id}.npy"
            m1_paths[repeat_id] = path
            m1_arrays[repeat_id] = np.lib.format.open_memmap(
                path, mode="w+", dtype="<i8", shape=(189000,)
            )

    endpoint_handle, endpoints = _open_csv_reader(
        paths["complete_model_pair_endpoints.csv"]
    )
    identity_handle, identities = _open_csv_reader(paths["identity33_all_pairs.csv"])
    active_count = 0
    row_key_digest = hashlib.sha256()
    world_pair_endpoints: list[tuple[str, str]] = []
    try:
        if endpoints.fieldnames != [
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        ]:
            raise common.ModelExperimentContractError("Endpoint CSV header drift")
        if identities.fieldnames != [
            "canonical_pair_uid",
            "world_uid",
            *identity_names,
        ]:
            raise common.ModelExperimentContractError("Identity CSV header drift")
        common.validate_identity33_column_names(policy, identities.fieldnames[2:])
        with row_keys_path.open("w", encoding="utf-8", newline="") as row_handle:
            writer = csv.DictWriter(
                row_handle, fieldnames=list(ROW_KEY_FIELDS), lineterminator="\n"
            )
            writer.writeheader()
            for row_index in range(189000):
                endpoint = next(endpoints, None)
                identity = next(identities, None)
                if endpoint is None or identity is None:
                    raise common.ModelExperimentContractError(
                        "Identity projection ended before 189,000 rows"
                    )
                world_ordinal = row_index // 378
                expected_world_uid = str(worlds[world_ordinal]["world_uid"])
                left = str(endpoint["seller_uid_left"])
                right = str(endpoint["seller_uid_right"])
                pair_uid = _canonical_pair_uid(left, right)
                if (
                    endpoint["world_uid"] != expected_world_uid
                    or endpoint["canonical_pair_uid"] != pair_uid
                    or identity["world_uid"] != expected_world_uid
                    or identity["canonical_pair_uid"] != pair_uid
                ):
                    raise common.ModelExperimentContractError(
                        f"Identity row alignment drift at {split}:{row_index}"
                    )
                row_key = {
                    "split": split,
                    "world_ordinal": world_ordinal,
                    "world_uid": expected_world_uid,
                    "canonical_pair_uid": pair_uid,
                    "seller_uid_left": left,
                    "seller_uid_right": right,
                }
                writer.writerow(row_key)
                row_key_digest.update(common.canonical_json_bytes(row_key) + b"\n")
                values = np.asarray(
                    [float(identity[name]) for name in identity_names], dtype="<f8"
                )
                if not np.isfinite(values).all():
                    raise common.ModelExperimentContractError(
                        f"Identity value is non-finite at {split}:{row_index}"
                    )
                matrix[row_index] = values
                active_count += int(np.any(values != 0.0))
                world_pair_endpoints.append(
                    common.canonical_pair_endpoints(left, right)
                )
                if len(world_pair_endpoints) == 378:
                    validate_world_k28(world_pair_endpoints)
                    if split == "train":
                        start = row_index + 1 - 378
                        for repeat_id, target in m1_arrays.items():
                            local = m1_source_indices_for_world(
                                expected_world_uid, world_pair_endpoints, repeat_id
                            )
                            target[start : start + 378] = start + local
                    world_pair_endpoints = []
            if next(endpoints, None) is not None or next(identities, None) is not None:
                raise common.ModelExperimentContractError(
                    "Identity projection has rows after 189,000"
                )
    finally:
        endpoint_handle.close()
        identity_handle.close()
        matrix.flush()
        del matrix
        for repeat_id in list(m1_arrays):
            value = m1_arrays.pop(repeat_id)
            value.flush()
            del value
    if world_pair_endpoints:
        raise common.ModelExperimentContractError("Incomplete final identity world")

    matrix_values = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix_values.shape != (189000, 33) or matrix_values.dtype.str != "<f8":
        raise common.ModelExperimentContractError("Published identity matrix drift")
    matrix_value_sha256 = common.matrix_value_sha256(matrix_values)
    del matrix_values
    m1_records = []
    m1_value_hashes = []
    for repeat_id, path in m1_paths.items():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (189000,) or values.dtype.str != "<i8":
            raise common.ModelExperimentContractError("Published M1 map shape drift")
        value_hash = hashlib.sha256(
            np.asarray(values, dtype="<i8").tobytes(order="C")
        ).hexdigest()
        m1_value_hashes.append(value_hash)
        m1_records.append(
            {
                "repeat_id": repeat_id,
                "value_sha256": value_hash,
                "file": file_record(path, temporary_root),
            }
        )
        del values
    if m1_records and len(set(m1_value_hashes)) != 5:
        raise common.ModelExperimentContractError("Five formal M1 maps are not distinct")
    return {
        "split": split,
        "world_count": 500,
        "row_count": 189000,
        "active_identity_row_count": active_count,
        "row_key_stream_sha256": row_key_digest.hexdigest(),
        "identity33_column_names_sha256": common.canonical_sha256(identity_names),
        "identity33_value_sha256": matrix_value_sha256,
        "row_keys_file": file_record(row_keys_path, temporary_root),
        "identity33_file": file_record(matrix_path, temporary_root),
        "m1_source_index_files": m1_records,
        "labels_qrels_membership_or_controller_read": False,
        "text_or_seller_profile_files_read": False,
        "audit_truth_read": False,
    }


def _validate_row_keys(
    path: Path, split: str
) -> tuple[str, list[tuple[str, str]], list[str]]:
    digest = hashlib.sha256()
    endpoints: list[tuple[str, str]] = []
    world_uids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(ROW_KEY_FIELDS):
            raise common.ModelExperimentContractError("Published row-key header drift")
        for row_index, row in enumerate(reader):
            if row_index >= 189000:
                raise common.ModelExperimentContractError(
                    "Published row-key count exceeds 189,000"
                )
            world_ordinal = row_index // 378
            left = str(row["seller_uid_left"])
            right = str(row["seller_uid_right"])
            if (
                row["split"] != split
                or row["world_ordinal"] != str(world_ordinal)
                or row["canonical_pair_uid"] != _canonical_pair_uid(left, right)
                or not row["world_uid"]
            ):
                raise common.ModelExperimentContractError(
                    f"Published row-key alignment drift at {split}:{row_index}"
                )
            if row_index % 378 == 0:
                world_uids.append(str(row["world_uid"]))
            elif row["world_uid"] != world_uids[-1]:
                raise common.ModelExperimentContractError(
                    f"Published row-key world drift at {split}:{row_index}"
                )
            canonical = common.canonical_pair_endpoints(left, right)
            endpoints.append(canonical)
            canonical_row = {
                "split": split,
                "world_ordinal": world_ordinal,
                "world_uid": str(row["world_uid"]),
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
            digest.update(common.canonical_json_bytes(canonical_row) + b"\n")
            if len(endpoints) % 378 == 0:
                validate_world_k28(endpoints[-378:])
    if len(endpoints) != 189000 or len(world_uids) != 500 or len(set(world_uids)) != 500:
        raise common.ModelExperimentContractError("Published row-key count/world drift")
    return digest.hexdigest(), endpoints, world_uids


def _validate_formal_source_binding(
    policy: Mapping[str, Any],
    split: str,
    row_keys_path: Path,
    matrix: np.ndarray,
    *,
    expected_row_count: int = 189000,
    pairs_per_world: int = 378,
) -> None:
    """Rebind the published rows and values to the pinned formal public inputs."""

    paths = projection.verify_split_public_inputs(
        policy, split, projection.IDENTITY_PUBLIC_ROLES
    )
    worlds = projection.load_worlds(paths["worlds.jsonl"], split)
    if len(worlds) * pairs_per_world != expected_row_count:
        raise common.ModelExperimentContractError(
            "Formal identity source world/row count drift"
        )
    identity_names = list(policy["feature_contract"]["identity33"])
    endpoint_handle, endpoints = _open_csv_reader(
        paths["complete_model_pair_endpoints.csv"]
    )
    identity_handle, identities = _open_csv_reader(
        paths["identity33_all_pairs.csv"]
    )
    row_handle, row_keys = _open_csv_reader(row_keys_path)
    try:
        if endpoints.fieldnames != [
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        ]:
            raise common.ModelExperimentContractError(
                "Formal endpoint CSV header drift during identity rebind"
            )
        if identities.fieldnames != [
            "canonical_pair_uid",
            "world_uid",
            *identity_names,
        ]:
            raise common.ModelExperimentContractError(
                "Formal identity CSV header drift during identity rebind"
            )
        common.validate_identity33_column_names(policy, identities.fieldnames[2:])
        if row_keys.fieldnames != list(ROW_KEY_FIELDS):
            raise common.ModelExperimentContractError(
                "Published row-key header drift during formal source rebind"
            )
        for row_index in range(expected_row_count):
            endpoint = next(endpoints, None)
            identity = next(identities, None)
            row_key = next(row_keys, None)
            if endpoint is None or identity is None or row_key is None:
                raise common.ModelExperimentContractError(
                    "Formal identity source rebind ended before the expected row count"
                )
            world_ordinal = row_index // pairs_per_world
            expected_world_uid = str(worlds[world_ordinal]["world_uid"])
            left = str(endpoint["seller_uid_left"])
            right = str(endpoint["seller_uid_right"])
            pair_uid = _canonical_pair_uid(left, right)
            expected_key = {
                "split": split,
                "world_ordinal": str(world_ordinal),
                "world_uid": expected_world_uid,
                "canonical_pair_uid": pair_uid,
                "seller_uid_left": left,
                "seller_uid_right": right,
            }
            if (
                endpoint["world_uid"] != expected_world_uid
                or endpoint["canonical_pair_uid"] != pair_uid
                or identity["world_uid"] != expected_world_uid
                or identity["canonical_pair_uid"] != pair_uid
                or dict(row_key) != expected_key
            ):
                raise common.ModelExperimentContractError(
                    f"Published identity row is not bound to the formal source at "
                    f"{split}:{row_index}"
                )
            try:
                source_values = np.asarray(
                    [float(identity[name]) for name in identity_names], dtype="<f8"
                )
            except (TypeError, ValueError) as exc:
                raise common.ModelExperimentContractError(
                    f"Formal identity source value is invalid at {split}:{row_index}"
                ) from exc
            observed_values = np.ascontiguousarray(matrix[row_index], dtype="<f8")
            if (
                not np.isfinite(source_values).all()
                or source_values.tobytes(order="C")
                != observed_values.tobytes(order="C")
            ):
                raise common.ModelExperimentContractError(
                    f"Published identity values are not bound to the formal source at "
                    f"{split}:{row_index}"
                )
        if (
            next(endpoints, None) is not None
            or next(identities, None) is not None
            or next(row_keys, None) is not None
        ):
            raise common.ModelExperimentContractError(
                "Formal identity source rebind has rows after the expected row count"
            )
    finally:
        endpoint_handle.close()
        identity_handle.close()
        row_handle.close()


def _validate_split_payload(
    policy: Mapping[str, Any], output_root: Path, record: Mapping[str, Any]
) -> str:
    split = str(record["split"])
    if (
        record.get("world_count") != 500
        or record.get("row_count") != 189000
        or record.get("labels_qrels_membership_or_controller_read") is not False
        or record.get("text_or_seller_profile_files_read") is not False
        or record.get("audit_truth_read") is not False
        or record.get("identity33_column_names_sha256")
        != common.canonical_sha256(policy["feature_contract"]["identity33"])
    ):
        raise common.ModelExperimentContractError("Identity split manifest drift")
    row_record = record["row_keys_file"]
    identity_record = record["identity33_file"]
    row_path = common.verify_file_pin(
        {**row_record, "path": str(output_root / row_record["path"])},
        label=f"identity projection {row_record['path']}",
    )
    matrix_path = common.verify_file_pin(
        {**identity_record, "path": str(output_root / identity_record["path"])},
        label=f"identity projection {identity_record['path']}",
    )
    row_digest, endpoints, world_uids = _validate_row_keys(row_path, split)
    if row_digest != record.get("row_key_stream_sha256"):
        raise common.ModelExperimentContractError("Identity row-key digest drift")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if (
        matrix.shape != (189000, 33)
        or matrix.dtype.str != "<f8"
        or not np.isfinite(matrix).all()
        or common.matrix_value_sha256(matrix) != record.get("identity33_value_sha256")
        or int(common.active_mask(matrix).sum())
        != record.get("active_identity_row_count")
    ):
        raise common.ModelExperimentContractError("Identity matrix semantic drift")
    _validate_formal_source_binding(policy, split, row_path, matrix)
    del matrix
    m1_records = record.get("m1_source_index_files")
    expected_repeat_ids = policy["m1"]["repeat_ids"] if split == "train" else []
    if (
        not isinstance(m1_records, list)
        or [value.get("repeat_id") for value in m1_records] != expected_repeat_ids
    ):
        raise common.ModelExperimentContractError("Identity M1 file universe drift")
    value_hashes = []
    for value in m1_records:
        file_spec = value["file"]
        path = common.verify_file_pin(
            {**file_spec, "path": str(output_root / file_spec["path"])},
            label=f"identity projection {file_spec['path']}",
        )
        indices = np.load(path, mmap_mode="r", allow_pickle=False)
        value_hash = validate_exact_m1_repeat(
            indices, endpoints, world_uids, str(value["repeat_id"])
        )
        del indices
        if value_hash != value.get("value_sha256"):
            raise common.ModelExperimentContractError("Identity M1 value digest drift")
        value_hashes.append(value_hash)
    if value_hashes and len(set(value_hashes)) != 5:
        raise common.ModelExperimentContractError("Five published M1 maps are not distinct")
    return split


def validate_published(policy: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "identity_projection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Identity projection manifest is missing: {manifest_path}")
    manifest = common.load_json(manifest_path)
    common.verify_self_hash(manifest, label="formal identity projection manifest")
    if (
        manifest.get("policy_canonical_self_hash") != policy["canonical_self_hash"]
        or manifest.get("status")
        != "FROZEN_LABEL_FREE_IDENTITY33_AND_TRAIN_M1_MAPS_NO_MODEL_TRAINING"
        or manifest.get("labels_qrels_membership_or_controller_read") is not False
        or manifest.get("text_or_seller_profile_files_read") is not False
        or manifest.get("audit_truth_read") is not False
        or manifest.get("model_training_or_scoring_performed") is not False
        or manifest.get("formal_root_manifest_canonical_self_hash")
        != policy["dataset_qualification"]["root_manifest"]["canonical_self_hash"]
        or manifest.get("implementation_files") != implementation_file_records()
        or manifest.get("total_world_count") != 2000
        or manifest.get("total_row_count") != 756000
    ):
        raise common.ModelExperimentContractError("Identity projection lineage drift")
    splits = manifest.get("splits", [])
    if [value.get("split") for value in splits] != list(projection.SPLITS):
        raise common.ModelExperimentContractError("Identity projection split order drift")
    validated = [_validate_split_payload(policy, output_root, value) for value in splits]
    if validated != list(projection.SPLITS):
        raise common.ModelExperimentContractError("Identity split validation order drift")
    expected_files = {"identity_projection_manifest.json"}
    for split in projection.SPLITS:
        expected_files.update(
            {f"{split}/row_keys.csv", f"{split}/identity33.npy"}
        )
    expected_files.update(
        f"train/m1_source_row_index_{repeat_id}.npy"
        for repeat_id in policy["m1"]["repeat_ids"]
    )
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise common.ModelExperimentContractError("Identity output file universe drift")
    return manifest


def publish(policy: Mapping[str, Any]) -> dict[str, Any]:
    parent = common.resolve(policy["outputs"]["public_projection"])
    output_root = parent / OUTPUT_SUBDIRECTORY
    if output_root.exists():
        return validate_published(policy, output_root)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".identity_v1.", dir=parent))
    try:
        splits = [build_split(policy, split, temporary) for split in projection.SPLITS]
        manifest = {
            "step": "step28_v13_v1_13_v9_4_1_identity_projection_v1",
            "status": "FROZEN_LABEL_FREE_IDENTITY33_AND_TRAIN_M1_MAPS_NO_MODEL_TRAINING",
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "formal_root_manifest_canonical_self_hash": policy[
                "dataset_qualification"
            ]["root_manifest"]["canonical_self_hash"],
            "implementation_files": implementation_file_records(),
            "splits": splits,
            "total_world_count": 2000,
            "total_row_count": 756000,
            "labels_qrels_membership_or_controller_read": False,
            "text_or_seller_profile_files_read": False,
            "audit_truth_read": False,
            "model_training_or_scoring_performed": False,
        }
        manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
        render_manifest(temporary / "identity_projection_manifest.json", manifest)
        validated = validate_published(policy, temporary)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    policy = common.load_policy()
    output_root = (
        common.resolve(policy["outputs"]["public_projection"])
        / OUTPUT_SUBDIRECTORY
    )
    manifest = publish(policy) if args.command == "build" else validate_published(
        policy, output_root
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
