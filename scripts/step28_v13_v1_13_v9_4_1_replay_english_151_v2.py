#!/usr/bin/env python3
"""Replay the frozen English 151-pair M0/C0 inference chain exactly.

The command is label-free and performs no fit, threshold selection, or metric
calculation.  It verifies legacy18, LaBSE6, imputation, column order, and the
two frozen LightGBM payloads before publishing a small immutable receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as predecessor_common
import step28_v13_v1_13_v9_4_1_base24_shared_v2 as shared_base24
import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common


OUTPUT_ROOT = (
    common.ROOT
    / "reports"
    / "step28_model_experiment"
    / "v9_4_1_training_v2_20260830"
    / "english_151_replay_attempt2"
)
MANIFEST_NAME = "english_151_replay_manifest.json"
PREDICTIONS_NAME = "english_151_probabilities.no_labels.csv"
STATUS = "PASSED_ENGLISH_151_M0_C0_EXACT_REPLAY"
IMPLEMENTATION_PATHS = {
    "successor_policy": common.DEFAULT_POLICY,
    "successor_common": Path(common.__file__).resolve(),
    "shared_base24_builder": Path(shared_base24.__file__).resolve(),
    "replay_script": Path(__file__).resolve(),
    "predecessor_common": Path(predecessor_common.__file__).resolve(),
    "successor_policy_tests": common.ROOT
    / "tests"
    / "test_step28_v13_v1_13_v9_4_1_model_training_policy_v2_contracts.py",
    "replay_tests": common.ROOT
    / "tests"
    / "test_step28_v13_v1_13_v9_4_1_replay_english_151_v2_contracts.py",
    "step7_v4_policy": common.ROOT
    / "schema"
    / "step7_v4_raw_item_authorship_selection_policy.json",
    "step7_v4_common": common.ROOT / "scripts" / "step7_v4_common.py",
    "step7_v4_prepare_source_data": common.ROOT
    / "scripts"
    / "step7_v4_prepare_source_data.py",
    "step7_v3_1_source_data": common.ROOT / "scripts" / "step7_v3_1_source_data.py",
    "step7_v3_1_prepare_source_data": common.ROOT
    / "scripts"
    / "step7_v3_1_prepare_source_data.py",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise common.ModelTrainingContractError(f"CSV has no header: {path}")
        rows = list(reader)
    return rows


def _matrix_hash(matrix: np.ndarray) -> str:
    if matrix.dtype.str != "<f8" or not matrix.flags.c_contiguous:
        raise common.ModelTrainingContractError("Replay matrix is not contiguous <f8")
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _vector_hash(vector: np.ndarray) -> str:
    if vector.dtype.str != "<f8" or not vector.flags.c_contiguous:
        raise common.ModelTrainingContractError("Replay vector is not contiguous <f8")
    return hashlib.sha256(vector.tobytes(order="C")).hexdigest()


def _load_model_payload(
    spec: Mapping[str, Any],
    expected_names: Sequence[str],
    expected_threshold: float,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    path = common._verify_pin(spec, label=f"frozen model {Path(spec['path']).name}")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise common.ModelTrainingContractError("Frozen model payload is not a mapping")
    candidate = payload.get("candidate")
    classifier = payload.get("classifier_artifact")
    if not isinstance(candidate, dict) or not isinstance(classifier, dict):
        raise common.ModelTrainingContractError("Frozen model payload structure drift")
    if list(candidate.get("feature_names", [])) != list(expected_names):
        raise common.ModelTrainingContractError("Frozen model feature order drift")
    if int(candidate.get("feature_count", -1)) != len(expected_names):
        raise common.ModelTrainingContractError("Frozen model feature count drift")
    if float(candidate.get("selected_threshold")) != float(expected_threshold):
        raise common.ModelTrainingContractError("Frozen model threshold drift")
    if payload.get("valid_label_values_read_for_fit_or_scoring") is not False:
        raise common.ModelTrainingContractError("Frozen model valid-label boundary drift")
    if payload.get("historical_test_label_values_read") is not False:
        raise common.ModelTrainingContractError("Frozen model historical-label boundary drift")
    medians = np.asarray(candidate.get("imputation_medians"), dtype="<f8")
    if medians.shape != (len(expected_names),) or not np.isfinite(medians).all():
        raise common.ModelTrainingContractError("Frozen model imputation medians drift")
    model = classifier.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise common.ModelTrainingContractError("Frozen classifier has no predict_proba")
    if int(getattr(model, "n_features_in_", -1)) != len(expected_names):
        raise common.ModelTrainingContractError("Frozen classifier feature count drift")
    return payload, model, np.ascontiguousarray(medians, dtype="<f8")


def _build_feature_matrix(
    rows: Sequence[Mapping[str, str]],
    indices: Sequence[int],
    feature_names: Sequence[str],
    medians: np.ndarray,
) -> np.ndarray:
    matrix = np.empty((len(indices), len(feature_names)), dtype="<f8")
    for output_index, source_index in enumerate(indices):
        row = rows[source_index]
        for column_index, name in enumerate(feature_names):
            if name not in row:
                raise common.ModelTrainingContractError(f"Missing replay feature: {name}")
            raw = row[name]
            matrix[output_index, column_index] = np.nan if raw == "" else float(raw)
    if np.isinf(matrix).any():
        raise common.ModelTrainingContractError("Replay feature matrix contains infinity")
    missing = np.isnan(matrix)
    if np.any(missing):
        matrix = np.where(missing, medians[None, :], matrix)
    matrix = np.ascontiguousarray(matrix, dtype="<f8")
    if not np.isfinite(matrix).all():
        raise common.ModelTrainingContractError("Imputed replay matrix is non-finite")
    return matrix


def _predict(model: Any, matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if probabilities.shape != (matrix.shape[0], 2):
        raise common.ModelTrainingContractError("Frozen classifier probability shape drift")
    result = np.ascontiguousarray(probabilities[:, 1], dtype="<f8")
    common.validate_p0(result)
    return result


def _input_record(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(spec["path"]),
        "size_bytes": int(spec["size_bytes"]),
        "sha256": str(spec["sha256"]),
    }


def _implementation_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, path in IMPLEMENTATION_PATHS.items():
        if not path.is_file():
            raise common.ModelTrainingContractError(
                f"English replay implementation file is missing: {path}"
            )
        records[name] = {
            "path": path.relative_to(common.ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
    return records


def run_replay(policy: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predecessor = predecessor_common.load_policy()
    runtime = predecessor_common.validate_supervised_cpu_runtime(predecessor)
    replay = policy["english_151_replay"]
    inputs = replay["inputs"]
    for name, spec in inputs.items():
        common._verify_pin(spec, label=f"English replay input {name}")

    pair_rows = _read_csv(common.resolve(inputs["pair_manifest"]["path"]))
    opaque_rows = _read_csv(common.resolve(inputs["opaque_pair_manifest"]["path"]))
    legacy_rows = _read_csv(common.resolve(inputs["legacy18_features"]["path"]))
    labse_rows = _read_csv(common.resolve(inputs["labse6_scores"]["path"]))
    row_count = len(pair_rows)
    if row_count != 733 or any(
        len(rows) != row_count for rows in (opaque_rows, legacy_rows, labse_rows)
    ):
        raise common.ModelTrainingContractError("English source row-count alignment drift")
    for index in range(row_count):
        if pair_rows[index].get("pair_uid") != legacy_rows[index].get("pair_uid"):
            raise common.ModelTrainingContractError(
                f"English original pair alignment drift at row {index}"
            )
        if opaque_rows[index].get("pair_uid") != labse_rows[index].get("pair_uid"):
            raise common.ModelTrainingContractError(
                f"English opaque pair alignment drift at row {index}"
            )
    valid_indices = [
        index for index, row in enumerate(pair_rows) if row.get("split_name") == "valid"
    ]
    if len(valid_indices) != int(replay["pair_count"]):
        raise common.ModelTrainingContractError("English valid split is not exactly 151 pairs")
    if len({opaque_rows[index]["pair_uid"] for index in valid_indices}) != 151:
        raise common.ModelTrainingContractError("English valid opaque pair IDs are not unique")
    crosswalk_spec = replay["canonical_to_opaque_crosswalk"]
    full_crosswalk = [
        [pair_rows[index]["pair_uid"], opaque_rows[index]["pair_uid"]]
        for index in range(row_count)
    ]
    valid_crosswalk = [full_crosswalk[index] for index in valid_indices]
    observed_crosswalk = {
        "full_733_canonical_json_sha256": common.canonical_sha256(full_crosswalk),
        "valid_151_canonical_json_sha256": common.canonical_sha256(valid_crosswalk),
    }
    if any(observed_crosswalk[name] != crosswalk_spec[name] for name in observed_crosswalk):
        raise common.ModelTrainingContractError("English canonical/opaque crosswalk drift")

    features = predecessor["feature_contract"]
    m0_names = [*features["legacy18"], *features["labse6"]]
    c0_names = list(features["legacy18"])
    _, m0_model, m0_medians = _load_model_payload(
        inputs["m0_model"],
        m0_names,
        predecessor["frozen_models"]["m0"]["threshold"],
    )
    _, c0_model, c0_medians = _load_model_payload(
        inputs["c0_model"],
        c0_names,
        predecessor["frozen_models"]["c0"]["threshold"],
    )
    reference, seller_records, reconstructed_pairs = (
        shared_base24.reconstruct_frozen_english_public(policy)
    )
    for index, (frozen_pair, reconstructed_pair) in enumerate(
        zip(pair_rows, reconstructed_pairs, strict=True)
    ):
        for field in (
            "pair_uid",
            "split_name",
            "seller_uid_left",
            "seller_uid_right",
        ):
            if frozen_pair[field] != reconstructed_pair[field]:
                raise common.ModelTrainingContractError(
                    f"Shared-builder English pair reconstruction drift at row {index}:{field}"
                )
    selected_pairs = [reconstructed_pairs[index] for index in valid_indices]
    generated_legacy18 = shared_base24.legacy18_matrix(
        selected_pairs,
        seller_records,
        reference,
        c0_names,
    )
    frozen_legacy18 = _build_feature_matrix(
        legacy_rows, valid_indices, c0_names, c0_medians
    )
    if generated_legacy18.tobytes(order="C") != frozen_legacy18.tobytes(order="C"):
        raise common.ModelTrainingContractError(
            "Shared legacy18 builder does not reproduce the frozen English table"
        )
    labse6_matrix = _build_feature_matrix(
        labse_rows,
        valid_indices,
        features["labse6"],
        m0_medians[18:],
    )
    m0_matrix = shared_base24.combine_base24(generated_legacy18, labse6_matrix)
    c0_matrix = generated_legacy18
    m0_probabilities = _predict(m0_model, m0_matrix)
    c0_probabilities = _predict(c0_model, c0_matrix)

    observed = {
        "m0_matrix_sha256": _matrix_hash(m0_matrix),
        "m0_probability_sha256": _vector_hash(m0_probabilities),
        "c0_matrix_sha256": _matrix_hash(c0_matrix),
        "c0_probability_sha256": _vector_hash(c0_probabilities),
    }
    expected = {name: str(replay[name]) for name in observed}
    if observed != expected:
        differences = [name for name in observed if observed[name] != expected[name]]
        raise common.ModelTrainingContractError(
            "English 151-pair exact replay drift: " + ",".join(differences)
        )

    predictions = [
        {
            "opaque_pair_uid": opaque_rows[index]["pair_uid"],
            "m0_probability": format(float(m0_probabilities[position]), ".17g"),
            "c0_probability": format(float(c0_probabilities[position]), ".17g"),
        }
        for position, index in enumerate(valid_indices)
    ]
    manifest = {
        "step": "step28_v13_v1_13_v9_4_1_replay_english_151_v2",
        "status": STATUS,
        "successor_policy_canonical_self_hash": policy["canonical_self_hash"],
        "predecessor_policy_canonical_self_hash": predecessor["canonical_self_hash"],
        "input_records": {name: _input_record(spec) for name, spec in inputs.items()},
        "implementation_records": _implementation_records(),
        "runtime": runtime,
        "source_row_count": row_count,
        "valid_pair_count": len(valid_indices),
        "canonical_to_opaque_crosswalk": observed_crosswalk,
        "feature_shapes": {
            "m0": list(m0_matrix.shape),
            "c0": list(c0_matrix.shape),
        },
        "shared_builder_legacy18_sha256": _matrix_hash(generated_legacy18),
        "shared_builder_matches_frozen_legacy18_table": True,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "all_four_exact_matches": observed == expected,
        "labels_or_identity_evidence_read": 0,
        "controller_or_membership_read": 0,
        "qrels_or_retrieval_truth_read": 0,
        "audit_truth_read": 0,
        "model_parameters_updated": False,
        "model_training_or_threshold_selection_performed": False,
        "m0_m1_m2_m3_training_authorized": False,
    }
    return manifest, predictions


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["opaque_pair_uid", "m0_probability", "c0_probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def publish(policy: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise common.ModelTrainingContractError(
            f"Publication root already exists; validate it instead: {OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="english_151_replay.", dir=OUTPUT_ROOT.parent))
    try:
        manifest, predictions = run_replay(policy)
        prediction_path = temporary / PREDICTIONS_NAME
        _write_predictions(prediction_path, predictions)
        manifest["prediction_file"] = {
            "path": PREDICTIONS_NAME,
            "size_bytes": prediction_path.stat().st_size,
            "sha256": common.sha256_file(prediction_path),
            "row_count": len(predictions),
        }
        unsigned = dict(manifest)
        manifest["canonical_self_hash"] = common.canonical_sha256(unsigned)
        _write_json(temporary / MANIFEST_NAME, manifest)
        os.replace(temporary, OUTPUT_ROOT)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_output(policy: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = OUTPUT_ROOT / MANIFEST_NAME
    if not manifest_path.is_file():
        raise common.ModelTrainingContractError("English replay manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    claimed = unsigned.pop("canonical_self_hash", None)
    if not isinstance(claimed, str) or common.canonical_sha256(unsigned) != claimed:
        raise common.ModelTrainingContractError("English replay manifest self-hash mismatch")
    if manifest.get("status") != STATUS:
        raise common.ModelTrainingContractError("English replay status drift")
    if manifest.get("successor_policy_canonical_self_hash") != policy["canonical_self_hash"]:
        raise common.ModelTrainingContractError("English replay successor policy drift")
    if manifest.get("input_records") != {
        name: _input_record(spec)
        for name, spec in policy["english_151_replay"]["inputs"].items()
    }:
        raise common.ModelTrainingContractError("English replay input-record drift")
    if manifest.get("implementation_records") != _implementation_records():
        raise common.ModelTrainingContractError("English replay implementation-record drift")
    if manifest.get("all_four_exact_matches") is not True:
        raise common.ModelTrainingContractError("English replay did not match all four hashes")
    replay = policy["english_151_replay"]
    expected = {
        name: replay[name]
        for name in (
            "m0_matrix_sha256",
            "m0_probability_sha256",
            "c0_matrix_sha256",
            "c0_probability_sha256",
        )
    }
    if manifest.get("observed_sha256") != expected:
        raise common.ModelTrainingContractError("English replay observed hashes drift")
    if (
        manifest.get("shared_builder_legacy18_sha256")
        != policy["english_151_replay"]["shared_builder_legacy18_matrix_sha256"]
        or manifest.get("shared_builder_matches_frozen_legacy18_table") is not True
    ):
        raise common.ModelTrainingContractError("English shared legacy18 evidence drift")
    for field in (
        "labels_or_identity_evidence_read",
        "controller_or_membership_read",
        "qrels_or_retrieval_truth_read",
        "audit_truth_read",
    ):
        if manifest.get(field) != 0:
            raise common.ModelTrainingContractError(f"English replay boundary drift: {field}")
    if manifest.get("model_parameters_updated") is not False:
        raise common.ModelTrainingContractError("English replay updated model parameters")
    if manifest.get("model_training_or_threshold_selection_performed") is not False:
        raise common.ModelTrainingContractError("English replay performed training")
    predecessor = predecessor_common.load_policy()
    if manifest.get("predecessor_policy_canonical_self_hash") != predecessor[
        "canonical_self_hash"
    ]:
        raise common.ModelTrainingContractError("English replay predecessor policy drift")
    expected_runtime = predecessor_common.validate_supervised_cpu_runtime(predecessor)
    if manifest.get("runtime") != expected_runtime:
        raise common.ModelTrainingContractError("English replay runtime drift")
    if manifest.get("source_row_count") != 733 or manifest.get("valid_pair_count") != 151:
        raise common.ModelTrainingContractError("English replay source-count drift")
    if manifest.get("feature_shapes") != {"m0": [151, 24], "c0": [151, 18]}:
        raise common.ModelTrainingContractError("English replay feature-shape drift")
    prediction = manifest.get("prediction_file", {})
    prediction_path = OUTPUT_ROOT / str(prediction.get("path", ""))
    if not prediction_path.is_file():
        raise common.ModelTrainingContractError("English replay prediction file is missing")
    if prediction_path.stat().st_size != int(prediction.get("size_bytes", -1)):
        raise common.ModelTrainingContractError("English replay prediction size drift")
    if common.sha256_file(prediction_path) != prediction.get("sha256"):
        raise common.ModelTrainingContractError("English replay prediction SHA-256 drift")
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "opaque_pair_uid",
            "m0_probability",
            "c0_probability",
        ]:
            raise common.ModelTrainingContractError(
                "English replay prediction header drift"
            )
        rows = list(reader)
    if len(rows) != 151 or len({row["opaque_pair_uid"] for row in rows}) != 151:
        raise common.ModelTrainingContractError("English replay prediction row drift")
    inputs = policy["english_151_replay"]["inputs"]
    pair_rows = _read_csv(common.resolve(inputs["pair_manifest"]["path"]))
    opaque_rows = _read_csv(common.resolve(inputs["opaque_pair_manifest"]["path"]))
    if len(pair_rows) != 733 or len(opaque_rows) != 733:
        raise common.ModelTrainingContractError("English replay alignment source drift")
    valid_indices = [
        index for index, row in enumerate(pair_rows) if row.get("split_name") == "valid"
    ]
    expected_opaque_ids = [opaque_rows[index]["pair_uid"] for index in valid_indices]
    if [row["opaque_pair_uid"] for row in rows] != expected_opaque_ids:
        raise common.ModelTrainingContractError("English replay opaque pair order drift")
    full_crosswalk = [
        [pair_rows[index]["pair_uid"], opaque_rows[index]["pair_uid"]]
        for index in range(733)
    ]
    valid_crosswalk = [full_crosswalk[index] for index in valid_indices]
    observed_crosswalk = {
        "full_733_canonical_json_sha256": common.canonical_sha256(full_crosswalk),
        "valid_151_canonical_json_sha256": common.canonical_sha256(valid_crosswalk),
    }
    crosswalk_spec = policy["english_151_replay"]["canonical_to_opaque_crosswalk"]
    if any(observed_crosswalk[name] != crosswalk_spec[name] for name in observed_crosswalk):
        raise common.ModelTrainingContractError("English canonical/opaque crosswalk drift")
    if manifest.get("canonical_to_opaque_crosswalk") != observed_crosswalk:
        raise common.ModelTrainingContractError("English replay manifest crosswalk drift")
    try:
        m0_probabilities = common.validate_p0(
            np.asarray([float(row["m0_probability"]) for row in rows], dtype="<f8")
        )
        c0_probabilities = common.validate_p0(
            np.asarray([float(row["c0_probability"]) for row in rows], dtype="<f8")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise common.ModelTrainingContractError(
            "English replay probability text is invalid"
        ) from exc
    if _vector_hash(m0_probabilities) != expected["m0_probability_sha256"]:
        raise common.ModelTrainingContractError(
            "English replay M0 CSV does not round-trip to frozen float64 bytes"
        )
    if _vector_hash(c0_probabilities) != expected["c0_probability_sha256"]:
        raise common.ModelTrainingContractError(
            "English replay C0 CSV does not round-trip to frozen float64 bytes"
        )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="publish the immutable replay")
    action.add_argument(
        "--validate-output",
        action="store_true",
        help="validate an existing immutable replay without scoring",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    policy = common.load_policy()
    if args.run:
        result = publish(policy)
    else:
        result = validate_output(policy)
    print(json.dumps({
        "status": result["status"],
        "canonical_self_hash": result["canonical_self_hash"],
        "valid_pair_count": result["valid_pair_count"],
        "all_four_exact_matches": result["all_four_exact_matches"],
        "model_training_or_threshold_selection_performed": result[
            "model_training_or_threshold_selection_performed"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
