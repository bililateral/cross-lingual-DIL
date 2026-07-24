#!/usr/bin/env python3
"""Audit style-free Step7-v4 feature sets across three classifier families."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import scipy
import sklearn
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v3_1_selection_core as solver  # noqa: E402
import step7_v3_1_source_data as source  # noqa: E402
import step7_v4_common as parent_common  # noqa: E402
import step7_v4_selection_core as corrected_logistic_solver  # noqa: E402
import step7_v4_select_source_model as parent_selector  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
POLICY_PATH = ROOT / "schema" / "step7_v4_1_style_free_classifier_policy.json"
EXPECTED_VERSION = "2026-07-24-step7-v4.1-style-free-classifier-audit-v1"
CLASSIFIER_IDS = ("l2_logistic", "rbf_svm", "lightgbm")
FEATURE_SET_IDS = (
    "legacy18",
    "e5",
    "labse",
    "e5_labse",
    "legacy18_e5",
    "legacy18_labse",
    "legacy18_e5_labse",
)
NULL_CANDIDATE_ID = "null__component_weighted_prevalence"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def verify_file_record(record: dict, role: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Step7-v4.1 missing {role}: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Step7-v4.1 {role} byte-size drift")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Step7-v4.1 {role} SHA-256 drift")
    return path


def feature_blocks(parent_policy: dict) -> dict[str, list[str]]:
    return {
        "legacy18": list(source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES),
        "e5_6": parent_common.encoder_feature_names(
            parent_policy["embedding_models"]["multilingual_e5_large"]
        ),
        "labse6": parent_common.encoder_feature_names(
            parent_policy["embedding_models"]["labse"]
        ),
    }


def feature_set_specs(policy: dict, parent_policy: dict) -> list[dict]:
    blocks = feature_blocks(parent_policy)
    specs = []
    for item in policy["feature_sets"]:
        names = [
            name for block in item["blocks"] for name in blocks[block]
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Step7-v4.1 duplicate feature in set: {item['id']}"
            )
        specs.append({**item, "feature_names": names})
    return specs


def candidate_specs(policy: dict, parent_policy: dict) -> list[dict]:
    output = [
        {
            "id": NULL_CANDIDATE_ID,
            "classifier_id": "weighted_prevalence",
            "feature_set_id": "none",
            "blocks": [],
            "feature_names": [],
            "feature_count": 0,
            "role": "null_control",
            "transfer_eligible": False,
        }
    ]
    for feature_set in feature_set_specs(policy, parent_policy):
        for classifier_id in CLASSIFIER_IDS:
            output.append(
                {
                    "id": f"{classifier_id}__{feature_set['id']}",
                    "classifier_id": classifier_id,
                    "feature_set_id": feature_set["id"],
                    "blocks": list(feature_set["blocks"]),
                    "feature_names": list(feature_set["feature_names"]),
                    "feature_count": len(feature_set["feature_names"]),
                    "role": feature_set["role"],
                    "transfer_eligible": bool(
                        feature_set["transfer_eligible"]
                    ),
                }
            )
    return output


def validate_policy(policy: dict, *, require_frozen: bool = True) -> None:
    if policy.get("version") != EXPECTED_VERSION:
        raise ValueError("Step7-v4.1 policy version drift")
    parent_record = policy["parent_contract"]
    parent_path = resolve(parent_record["policy_path"])
    if (
        not parent_path.is_file()
        or sha256_file(parent_path) != parent_record["policy_sha256"]
    ):
        raise ValueError("Step7-v4.1 parent policy drift")
    parent_policy = parent_common.load_json(parent_path)
    if parent_policy.get("version") != parent_record["parent_version"]:
        raise ValueError("Step7-v4.1 parent version drift")

    implementation = policy.get("implementation", {})
    expected_implementation = {
        "selector",
        "parent_selector",
        "parent_common",
        "selection_solver",
        "corrected_float64_logistic_solver",
        "source_feature_module",
    }
    if set(implementation) != expected_implementation:
        raise ValueError("Step7-v4.1 implementation universe drift")
    for role, record in implementation.items():
        path = resolve(record["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Step7-v4.1 implementation missing: {role}"
            )
        expected = str(record["sha256"])
        if (
            role == "selector"
            and not require_frozen
            and expected == "TO_BE_FROZEN_AFTER_IMPLEMENTATION"
        ):
            continue
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(
                f"Step7-v4.1 implementation SHA-256 drift: {role}"
            )

    expected_dependencies = {
        "python_major_minor": ".".join(
            platform.python_version_tuple()[:2]
        ),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
    }
    if policy.get("dependencies") != expected_dependencies:
        raise ValueError(
            "Step7-v4.1 dependency versions do not match the frozen policy"
        )

    if set(policy["pinned_inputs"]) != {
        "pair_manifest",
        "legacy18_pair_features",
        "e5_pair_aggregates",
        "labse_pair_aggregates",
        "train_labels",
        "valid_labels",
        "development_labels_manifest",
        "source_preparation_manifest",
        "gpu_output_manifest",
        "gpu_sync_manifest",
    }:
        raise ValueError("Step7-v4.1 pinned-input universe drift")
    for role, record in policy["pinned_inputs"].items():
        if (
            set(record) != {"path", "size_bytes", "sha256"}
            or int(record["size_bytes"]) <= 0
            or len(str(record["sha256"])) != 64
        ):
            raise ValueError(
                f"Step7-v4.1 malformed pinned input: {role}"
            )

    blocks = feature_blocks(parent_policy)
    if set(policy["feature_blocks"]) != set(blocks):
        raise ValueError("Step7-v4.1 feature-block universe drift")
    feature_sets = policy["feature_sets"]
    if [item["id"] for item in feature_sets] != list(FEATURE_SET_IDS):
        raise ValueError("Step7-v4.1 feature-set order/universe drift")
    for item in feature_sets:
        if set(item) != {
            "id",
            "blocks",
            "role",
            "transfer_eligible",
        }:
            raise ValueError("Step7-v4.1 feature-set schema drift")
        if not item["blocks"] or any(
            block not in blocks for block in item["blocks"]
        ):
            raise ValueError(
                f"Step7-v4.1 unknown/empty block: {item['id']}"
            )
    forbidden = tuple(policy["forbidden_feature_name_prefixes"])
    if forbidden != ("style_", "raw_pcm_", "raw_mstyle_"):
        raise ValueError("Step7-v4.1 forbidden-prefix contract drift")
    for spec in feature_set_specs(policy, parent_policy):
        if any(name.startswith(forbidden) for name in spec["feature_names"]):
            raise ValueError(
                f"Step7-v4.1 style feature entered {spec['id']}"
            )
    specs = candidate_specs(policy, parent_policy)
    if len(specs) != 1 + len(FEATURE_SET_IDS) * len(CLASSIFIER_IDS):
        raise AssertionError("Step7-v4.1 candidate count drift")

    classifiers = policy["classifiers"]
    if tuple(classifiers) != CLASSIFIER_IDS:
        raise ValueError("Step7-v4.1 classifier order/universe drift")
    logistic = classifiers["l2_logistic"]
    l2_grid = [float(value) for value in logistic["initial_l2_grid"]]
    if (
        l2_grid != sorted(set(l2_grid))
        or any(value <= 0.0 for value in l2_grid)
        or logistic["boundary_optimum_is_failure"]
        or not float(logistic["minimum_l2"]) < l2_grid[0]
        or not float(logistic["maximum_l2"]) > l2_grid[-1]
    ):
        raise ValueError("Step7-v4.1 L2 grid contract drift")
    svm = classifiers["rbf_svm"]
    if (
        svm["calibration_fold_count"] != 3
        or not math.isclose(
            float(svm["calibration_l2_penalty"]),
            0.01,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or [float(value) for value in svm["c_grid"]]
        != [0.1, 1.0, 10.0]
        or [
            float(value)
            for value in svm["gamma_multiplier_over_feature_count_grid"]
        ]
        != [0.1, 1.0, 10.0]
    ):
        raise ValueError("Step7-v4.1 RBF-SVM grid contract drift")
    boosting = classifiers["lightgbm"]
    if (
        len(boosting["grid"]) != 8
        or boosting["fixed_parameters"].get("deterministic") is not True
        or boosting["fixed_parameters"].get("force_col_wise") is not True
        or int(boosting["fixed_parameters"].get("num_threads", 0)) != 1
    ):
        raise ValueError("Step7-v4.1 LightGBM grid contract drift")

    training = policy["training"]
    if (
        int(training["outer_fold_count"]) != 5
        or len(training["outer_seeds"]) != 5
        or len(set(training["outer_seeds"])) != 5
        or int(training["inner_fold_count"]) != 4
        or int(training["final_hyperparameter_fold_count"]) != 5
        or training["sample_weight"]
        != "component_equal_normalized_to_row_count"
        or training["class_weight"] != "none"
        or training["evidence_type_training_weight"] != "forbidden"
        or training["l2_parameterization"]
        != (
            "weighted_mean_logloss_plus_half_l2_squared_"
            "coefficient_norm"
        )
    ):
        raise ValueError("Step7-v4.1 training discipline drift")
    bootstrap = policy["evaluation"]["bootstrap"]
    if (
        bootstrap["group"] != "component_id"
        or int(bootstrap["resamples"]) != 5000
        or not 0.0 < float(bootstrap["confidence"]) < 1.0
    ):
        raise ValueError("Step7-v4.1 bootstrap contract drift")
    if (
        policy["selection_rule"]["post_hoc_design_can_formally_certify_m0"]
        is not False
        or policy["selection_rule"]["valid_metrics_may_change_selection"]
        is not False
        or policy["selection_rule"]["new_real_english_confirmation_required"]
        is not True
    ):
        raise ValueError("Step7-v4.1 claim boundary drift")

    outputs = policy["outputs"]
    expected_outputs = {
        "root",
        "train_selection_lock",
        "blind_scoring_lock",
        "model_artifacts",
        "selected_model_template",
        "train_oof_predictions",
        "no_clone_oof_predictions",
        "blind_valid_predictions",
        "valid_predictions",
        "selection_summary",
    }
    if set(outputs) != expected_outputs:
        raise ValueError("Step7-v4.1 output universe drift")
    root = outputs["root"].rstrip("/")
    if root != (
        "reports/step7_v4_1_style_free_classifier_selection/"
        "v1_20260724"
    ):
        raise ValueError("Step7-v4.1 output root drift")
    for role, value in outputs.items():
        if role != "root" and not str(value).startswith(root + "/"):
            raise ValueError(
                f"Step7-v4.1 output escapes version root: {role}"
            )


def load_policy(*, require_frozen: bool = True) -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    validate_policy(policy, require_frozen=require_frozen)
    return policy


def verify_pinned_inputs(policy: dict) -> dict[str, dict]:
    output = {}
    for role, record in policy["pinned_inputs"].items():
        path = verify_file_record(record, role)
        output[role] = {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def load_style_free_fixed_features(
    policy: dict,
    parent_policy: dict,
    preparation_manifest: dict,
    preparation_bundle: dict,
) -> tuple[dict[str, dict[str, float | None]], dict]:
    outputs = parent_policy["outputs"]
    bundle_path = resolve(policy["pinned_inputs"]["gpu_output_manifest"]["path"])
    bundle = parent_common.load_json(bundle_path)
    parent_common.verify_canonical_self_hash(
        bundle, "bundle_content_sha256", "Step7-v4 GPU output bundle"
    )
    if (
        bundle.get("step") != "step7_v4_label_free_gpu_output_bundle"
        or bundle.get("version") != parent_policy["version"]
        or bundle.get("label_or_raw_source_files_present_in_gpu_workspace")
        is not False
        or bundle.get("embedding_matrices_published") is not False
        or bundle.get("policy_sha256")
        != policy["parent_contract"]["policy_sha256"]
        or bundle.get("gpu_sync_manifest_sha256")
        != policy["pinned_inputs"]["gpu_sync_manifest"]["sha256"]
        or int(bundle.get("file_count", -1)) != len(bundle.get("files", []))
        or int(bundle.get("total_file_bytes", -1))
        != sum(int(record["size_bytes"]) for record in bundle.get("files", []))
    ):
        raise ValueError("Step7-v4.1 parent GPU bundle boundary drift")
    bundle_records = {
        record["path"]: record for record in bundle["files"]
    }
    if len(bundle_records) != len(bundle["files"]):
        raise ValueError("Step7-v4.1 parent GPU bundle repeats a path")

    pair_rows = preparation_bundle["pair_rows"]
    opaque_pair_rows = preparation_bundle["gpu_pair_rows"]
    public_pair_uids = [row["pair_uid"] for row in pair_rows]
    opaque_pair_uids = [row["pair_uid"] for row in opaque_pair_rows]
    if len(public_pair_uids) != len(opaque_pair_uids):
        raise ValueError("Step7-v4.1 public/opaque pair count drift")
    features: dict[str, dict[str, float | None]] = {
        pair_uid: {} for pair_uid in public_pair_uids
    }

    legacy_names = list(source.MODEL_ELIGIBLE_TRANSFER_FEATURE_NAMES)
    legacy_rows = preparation_bundle["legacy_rows"]
    if [row["pair_uid"] for row in legacy_rows] != public_pair_uids:
        raise ValueError("Step7-v4.1 legacy18 row order drift")
    for public_pair_uid, row in zip(
        public_pair_uids, legacy_rows, strict=True
    ):
        for name in legacy_names:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(
                    f"Step7-v4.1 non-finite legacy18 value: {name}"
                )
            features[public_pair_uid][name] = value

    selected_models = {
        "multilingual_e5_large": "e5_pair_aggregates",
        "labse": "labse_pair_aggregates",
    }
    model_audits = {}
    for model_key, input_role in selected_models.items():
        model_cfg = parent_policy["embedding_models"][model_key]
        score_path = resolve(policy["pinned_inputs"][input_role]["path"])
        score_record = parent_common.file_record(score_path)
        if bundle_records.get(relative(score_path)) != score_record:
            raise ValueError(
                f"Step7-v4.1 GPU bundle does not pin {model_key} scores"
            )
        score_rows = parent_common.load_csv(score_path)
        names = parent_common.encoder_feature_names(model_cfg)
        audit_names = parent_common.frequency_audit_feature_names(model_cfg)
        expected_schema = ["pair_uid", *names, *audit_names]
        if (
            not score_rows
            or any(list(row) != expected_schema for row in score_rows)
            or [row["pair_uid"] for row in score_rows] != opaque_pair_uids
        ):
            raise ValueError(
                f"Step7-v4.1 pair-score schema/order drift: {model_key}"
            )
        missing_counts = Counter()
        for public_pair_uid, row in zip(
            public_pair_uids, score_rows, strict=True
        ):
            for name, audit_name in zip(names, audit_names, strict=True):
                primary = str(row[name]).strip()
                audit = str(row[audit_name]).strip()
                if bool(primary) != bool(audit):
                    raise ValueError(
                        "Step7-v4.1 primary/multiplicity missingness drift: "
                        f"{model_key}/{name}"
                    )
                if not primary:
                    features[public_pair_uid][name] = None
                    missing_counts[name] += 1
                    continue
                value = float(primary)
                audit_value = float(audit)
                if (
                    not math.isfinite(value)
                    or not math.isfinite(audit_value)
                    or value < -1.000001
                    or value > 1.000001
                    or audit_value < -1.000001
                    or audit_value > 1.000001
                ):
                    raise ValueError(
                        f"Step7-v4.1 invalid aggregate: {model_key}/{name}"
                    )
                features[public_pair_uid][name] = value

        runtime_path = resolve(
            outputs["model_runtime_manifest_template"].format(
                model_key=model_key
            )
        )
        runtime_record = parent_common.file_record(runtime_path)
        if bundle_records.get(relative(runtime_path)) != runtime_record:
            raise ValueError(
                f"Step7-v4.1 GPU bundle does not pin {model_key} runtime"
            )
        runtime = parent_common.load_json(runtime_path)
        parent_common.verify_canonical_self_hash(
            runtime,
            "runtime_content_sha256",
            f"Step7-v4 {model_key} runtime",
        )
        if (
            runtime.get("step")
            != "step7_v4_encode_complete_item_shared_chunks"
            or runtime.get("version") != parent_policy["version"]
            or runtime.get("model_key") != model_key
            or runtime.get("feature_generation_reads_label_values")
            is not False
            or runtime.get("label_or_raw_source_files_present_in_gpu_workspace")
            is not False
            or runtime.get("encoder_parameters_updated") is not False
            or runtime.get("embedding_matrix_published") is not False
            or runtime.get("embedding_matrix_ephemeral") is not True
            or runtime.get("pair_count") != len(pair_rows)
            or runtime.get("aggregate_feature_names") != names
            or runtime.get("multiplicity_audit_feature_names") != audit_names
            or runtime.get("pair_scores") != score_record
            or runtime.get("policy_sha256")
            != policy["parent_contract"]["policy_sha256"]
            or runtime.get("source_preparation_manifest_file_sha256")
            != policy["pinned_inputs"]["source_preparation_manifest"][
                "sha256"
            ]
            or runtime.get("source_preparation_manifest_content_sha256")
            != preparation_manifest["manifest_content_sha256"]
            or runtime.get("gpu_sync_manifest_sha256")
            != policy["pinned_inputs"]["gpu_sync_manifest"]["sha256"]
            or runtime.get("device") != "cuda"
        ):
            raise ValueError(
                f"Step7-v4.1 selected encoder runtime drift: {model_key}"
            )
        model_audits[model_key] = {
            "score_file": score_record,
            "runtime_file": runtime_record,
            "feature_names": names,
            "missing_count_by_feature": dict(sorted(missing_counts.items())),
            "encoder_parameters_updated": False,
            "feature_generation_reads_label_values": False,
        }

    expected_names = {
        name
        for names in feature_blocks(parent_policy).values()
        for name in names
    }
    forbidden = tuple(policy["forbidden_feature_name_prefixes"])
    for pair_uid, values in features.items():
        if set(values) != expected_names:
            raise ValueError(
                f"Step7-v4.1 style-free feature universe drift: {pair_uid}"
            )
        if any(name.startswith(forbidden) for name in values):
            raise AssertionError(
                "Step7-v4.1 author-style feature entered fixed features"
            )
    return features, {
        "gpu_bundle": parent_common.file_record(bundle_path),
        "selected_encoder_models_opened": list(selected_models),
        "author_style_encoder_score_or_runtime_files_opened": False,
        "selected_model_audits": model_audits,
        "retained_feature_names": sorted(expected_names),
        "retained_feature_count": len(expected_names),
        "selectable_style_feature_count": 0,
    }


def load_style_free_parent_data(
    policy: dict,
) -> tuple[
    dict,
    dict,
    dict,
    dict[str, dict[str, float | None]],
    dict[str, dict],
    dict[str, str],
    dict,
]:
    parent_policy_path = resolve(policy["parent_contract"]["policy_path"])
    parent_policy = parent_common.load_policy(parent_policy_path)
    parent_common.verify_implementation_files(parent_policy)
    preparation_manifest, preparation_bundle = (
        parent_common.validate_preparation_artifacts(parent_policy)
    )
    fixed_features, style_free_gpu_audit = load_style_free_fixed_features(
        policy,
        parent_policy,
        preparation_manifest,
        preparation_bundle,
    )
    pair_rows = preparation_bundle["pair_rows"]
    seller_records, seller_markets, legacy_replay = (
        parent_selector.replay_legacy_context(
            parent_policy, pair_rows, fixed_features
        )
    )
    factory = parent_selector.FeatureFactory(
        parent_policy, pair_rows, fixed_features, seller_records
    )
    return (
        parent_policy,
        preparation_manifest,
        preparation_bundle,
        fixed_features,
        seller_records,
        seller_markets,
        {
            **style_free_gpu_audit,
            "legacy18_replay": legacy_replay,
            "factory": factory,
        },
    )


def labels_array(rows: list[dict]) -> np.ndarray:
    return parent_selector.labels_array(rows)


def component_weights(rows: list[dict], policy: dict) -> np.ndarray:
    return solver.component_weights(
        rows, policy["training"]["sample_weight"]
    )


def weighted_standardizer(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 2
        or sample_weights.shape != (len(values),)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights <= 0.0)
    ):
        raise ValueError("Step7-v4.1 standardizer input drift")
    total = float(np.sum(sample_weights))
    mean = np.sum(values * sample_weights[:, None], axis=0) / total
    variance = (
        np.sum(
            ((values - mean) ** 2) * sample_weights[:, None], axis=0
        )
        / total
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    standardized = (values - mean) / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Step7-v4.1 standardization produced non-finite data")
    return standardized, mean, scale


def hyperparameter_grid(
    policy: dict, classifier_id: str
) -> list[dict[str, float | int]]:
    cfg = policy["classifiers"][classifier_id]
    if classifier_id == "l2_logistic":
        return [
            {"l2_penalty": float(value)}
            for value in cfg["initial_l2_grid"]
        ]
    if classifier_id == "rbf_svm":
        return [
            {
                "c": float(c_value),
                "gamma_multiplier": float(gamma_multiplier),
            }
            for c_value in cfg["c_grid"]
            for gamma_multiplier in cfg[
                "gamma_multiplier_over_feature_count_grid"
            ]
        ]
    if classifier_id == "lightgbm":
        return [
            {
                "num_leaves": int(item["num_leaves"]),
                "min_child_samples": int(item["min_child_samples"]),
                "learning_rate": float(item["learning_rate"]),
                "n_estimators": int(item["n_estimators"]),
            }
            for item in cfg["grid"]
        ]
    raise ValueError(f"Unknown Step7-v4.1 classifier: {classifier_id}")


def hyperparameter_complexity_key(
    classifier_id: str, parameters: dict
) -> tuple:
    if classifier_id == "l2_logistic":
        return (float(parameters["l2_penalty"]),)
    if classifier_id == "rbf_svm":
        return (
            -float(parameters["c"]),
            -float(parameters["gamma_multiplier"]),
        )
    if classifier_id == "lightgbm":
        return (
            -int(parameters["num_leaves"]),
            int(parameters["min_child_samples"]),
            -int(parameters["n_estimators"]),
            -float(parameters["learning_rate"]),
        )
    raise ValueError(f"Unknown Step7-v4.1 classifier: {classifier_id}")


def calibration_splits(
    rows: list[dict], fold_count: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    assignments = parent_selector.balanced_component_folds(
        rows, int(fold_count), int(seed)
    )
    splits = []
    audits = []
    for fold in range(int(fold_count)):
        train_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignments[row["component_id"]] != fold
            ],
            dtype=int,
        )
        hold_indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if assignments[row["component_id"]] == fold
            ],
            dtype=int,
        )
        train_components = {
            rows[int(index)]["component_id"] for index in train_indices
        }
        hold_components = {
            rows[int(index)]["component_id"] for index in hold_indices
        }
        if train_components & hold_components:
            raise ValueError(
                "Step7-v4.1 SVM calibration leaks a component"
            )
        if (
            set(labels_array([rows[int(index)] for index in train_indices]))
            != {0, 1}
            or set(
                labels_array([rows[int(index)] for index in hold_indices])
            )
            != {0, 1}
        ):
            raise ValueError(
                "Step7-v4.1 SVM calibration fold lacks a class"
            )
        splits.append((train_indices, hold_indices))
        audits.append(
            {
                "fold": fold,
                "train_row_count": len(train_indices),
                "holdout_row_count": len(hold_indices),
                "train_component_count": len(train_components),
                "holdout_component_count": len(hold_components),
                "component_overlap_count": 0,
            }
        )
    return splits, {
        "fold_count": int(fold_count),
        "assignment_sha256": canonical_hash(assignments),
        "folds": audits,
    }


def fit_corrected_logistic(
    policy: dict,
    matrix: np.ndarray,
    rows: list[dict],
    l2_penalty: float,
) -> dict:
    cfg = policy["training"]
    weights = component_weights(rows, policy)
    weight_total = float(np.sum(weights))
    if float(l2_penalty) < 0.0:
        raise ValueError("Step7-v4.1 L2 penalty cannot be negative")
    solver_sum_loss_l2_penalty = float(l2_penalty) * weight_total
    artifact = corrected_logistic_solver.fit_logistic(
        np.asarray(matrix, dtype=np.float64),
        labels_array(rows),
        weights,
        solver_sum_loss_l2_penalty,
        int(cfg["max_iter"]),
        float(cfg["tolerance"]),
        float(cfg["armijo_c1"]),
        float(cfg["minimum_line_search_step"]),
    )
    if artifact.get("solver_converged") is not True:
        raise ValueError(
            "Step7-v4.1 corrected logistic solver did not converge"
        )
    if not math.isclose(
        float(artifact["sample_weight_total"]),
        weight_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Step7-v4.1 logistic sample-weight drift")
    artifact["solver_sum_loss_l2_penalty"] = float(
        artifact.pop("l2_penalty")
    )
    artifact["l2_penalty"] = float(l2_penalty)
    artifact["l2_parameterization"] = cfg["l2_parameterization"]
    artifact["solver_implementation"] = relative(
        Path(corrected_logistic_solver.__file__)
    )
    return artifact


def fit_svc_base(
    policy: dict,
    parameters: dict,
    matrix: np.ndarray,
    rows: list[dict],
    *,
    seed: int,
) -> dict:
    cfg = policy["classifiers"]["rbf_svm"]
    x = np.asarray(matrix, dtype=np.float64)
    weights = component_weights(rows, policy)
    standardized, mean, scale = weighted_standardizer(x, weights)
    gamma = float(parameters["gamma_multiplier"]) / x.shape[1]
    model = SVC(
        C=float(parameters["c"]),
        kernel="rbf",
        gamma=gamma,
        shrinking=True,
        probability=False,
        tol=float(cfg["tolerance"]),
        cache_size=512,
        class_weight=None,
        verbose=False,
        max_iter=int(cfg["maximum_iterations"]),
        decision_function_shape="ovr",
        break_ties=False,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(
            standardized,
            labels_array(rows),
            sample_weight=weights,
        )
    if int(getattr(model, "fit_status_", 1)) != 0:
        raise ValueError("Step7-v4.1 RBF-SVM did not converge")
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "gamma": gamma,
        "fit_audit": {
            "fit_completed": True,
            "solver_converged": True,
            "support_vector_count": int(len(model.support_)),
            "solver_iterations": [
                int(value)
                for value in np.asarray(model.n_iter_).reshape(-1)
            ],
        },
    }


def svc_decision(matrix: np.ndarray, base_artifact: dict) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    standardized = (
        x - np.asarray(base_artifact["mean"], dtype=np.float64)
    ) / np.asarray(base_artifact["scale"], dtype=np.float64)
    values = np.asarray(
        base_artifact["model"].decision_function(standardized),
        dtype=np.float64,
    ).reshape(-1)
    if values.shape != (len(x),) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Step7-v4.1 RBF-SVM produced invalid decision margins"
        )
    return values


def fit_classifier(
    policy: dict,
    classifier_id: str,
    parameters: dict,
    matrix: np.ndarray,
    rows: list[dict],
    *,
    seed: int,
    factory: parent_selector.FeatureFactory | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    x = np.asarray(matrix, dtype=np.float64)
    y = labels_array(rows)
    weights = component_weights(rows, policy)
    if x.ndim != 2 or x.shape[0] != len(rows) or x.shape[1] == 0:
        raise ValueError("Step7-v4.1 classifier matrix shape drift")
    if not np.all(np.isfinite(x)):
        raise ValueError("Step7-v4.1 classifier matrix is non-finite")

    if classifier_id == "l2_logistic":
        artifact = fit_corrected_logistic(
            policy, x, rows, float(parameters["l2_penalty"])
        )
        return {
            "classifier_id": classifier_id,
            "parameters": dict(parameters),
            "model": artifact,
            "fit_audit": {
                "fit_completed": True,
                "solver_converged": artifact["solver_converged"],
                "solver_iterations": artifact["solver_iterations"],
                "solver_final_normalized_gradient_inf_norm": artifact[
                    "solver_final_normalized_gradient_inf_norm"
                ],
                "internal_estimator_fit_count": 1,
            },
        }

    if classifier_id == "rbf_svm":
        cfg = policy["classifiers"]["rbf_svm"]
        if factory is None or feature_names is None:
            raise ValueError(
                "Step7-v4.1 RBF-SVM requires fold-local feature factory"
            )
        split_seed = derived_seed(seed, "svm_probability_calibration")
        cv, calibration_audit = calibration_splits(
            rows, int(cfg["calibration_fold_count"]), split_seed
        )
        calibration_oof_margins = np.full(
            len(rows), np.nan, dtype=np.float64
        )
        calibration_fit_audits = []
        for fold, (train_indices, hold_indices) in enumerate(cv):
            calibration_train_rows = [
                rows[int(index)] for index in train_indices
            ]
            calibration_hold_rows = [
                rows[int(index)] for index in hold_indices
            ]
            calibration_train, calibration_hold, medians, reference_audit = (
                factory.design(
                    calibration_train_rows,
                    calibration_hold_rows,
                    feature_names,
                )
            )
            base = fit_svc_base(
                policy,
                parameters,
                calibration_train,
                calibration_train_rows,
                seed=derived_seed(seed, "svm_calibration_base", fold),
            )
            calibration_oof_margins[hold_indices] = svc_decision(
                calibration_hold, base
            )
            calibration_fit_audits.append(
                {
                    "fold": fold,
                    "imputation_median_count": len(medians),
                    "imputation_medians_sha256": canonical_hash(medians),
                    "feature_reference_audit": reference_audit,
                    "base_fit": base["fit_audit"],
                }
            )
        if not np.all(np.isfinite(calibration_oof_margins)):
            raise ValueError(
                "Step7-v4.1 SVM calibration OOF margins are incomplete"
            )
        calibration_model = fit_corrected_logistic(
            policy,
            calibration_oof_margins[:, None],
            rows,
            float(cfg["calibration_l2_penalty"]),
        )
        final_base = fit_svc_base(
            policy,
            parameters,
            x,
            rows,
            seed=derived_seed(seed, "svm_final_base"),
        )
        return {
            "classifier_id": classifier_id,
            "parameters": {
                **dict(parameters),
                "gamma": final_base["gamma"],
            },
            "model": final_base["model"],
            "mean": final_base["mean"],
            "scale": final_base["scale"],
            "calibration_logistic": calibration_model,
            "fit_audit": {
                "fit_completed": True,
                "solver_converged": True,
                "calibration_split": calibration_audit,
                "calibration_fits": calibration_fit_audits,
                "calibration_logistic": {
                    key: value
                    for key, value in calibration_model.items()
                    if key
                    not in {"mean", "scale", "intercept", "coefficients"}
                },
                "final_base_fit": final_base["fit_audit"],
                "internal_estimator_fit_count": (
                    int(cfg["calibration_fold_count"]) + 2
                ),
            },
        }

    if classifier_id == "lightgbm":
        fixed = dict(
            policy["classifiers"]["lightgbm"]["fixed_parameters"]
        )
        model = LGBMClassifier(
            **fixed,
            **parameters,
            random_state=int(seed),
            bagging_seed=int(seed),
            feature_fraction_seed=int(seed),
            data_random_seed=int(seed),
        )
        model.fit(x, y, sample_weight=weights)
        if not hasattr(model, "booster_"):
            raise ValueError(
                "Step7-v4.1 LightGBM fit did not produce a booster"
            )
        tree_count = int(model.booster_.num_trees())
        requested_tree_count = int(parameters["n_estimators"])
        if tree_count <= 0 or tree_count > requested_tree_count:
            raise ValueError(
                "Step7-v4.1 LightGBM tree accounting drift"
            )
        return {
            "classifier_id": classifier_id,
            "parameters": dict(parameters),
            "model": model,
            "fit_audit": {
                "fit_completed": True,
                "solver_converged": True,
                "requested_tree_count": requested_tree_count,
                "tree_count": tree_count,
                "full_requested_tree_budget_materialized": bool(
                    tree_count == requested_tree_count
                ),
                "natural_no_further_split_termination_allowed": True,
                "best_iteration": (
                    None
                    if model.best_iteration_ in {None, 0}
                    else int(model.best_iteration_)
                ),
                "internal_estimator_fit_count": 1,
            },
        }
    raise ValueError(f"Unknown Step7-v4.1 classifier: {classifier_id}")


def apply_classifier(matrix: np.ndarray, artifact: dict) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    classifier_id = artifact["classifier_id"]
    if classifier_id == "l2_logistic":
        probabilities = solver.apply_logistic(x, artifact["model"])
    elif classifier_id == "rbf_svm":
        margins = svc_decision(x, artifact)
        probabilities = solver.apply_logistic(
            margins[:, None], artifact["calibration_logistic"]
        )
    elif classifier_id == "lightgbm":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but "
                    "LGBMClassifier was fitted with feature names"
                ),
                category=UserWarning,
            )
            probabilities = artifact["model"].predict_proba(
                x, validate_features=False
            )[:, 1]
    else:
        raise ValueError(
            f"Unknown Step7-v4.1 classifier: {classifier_id}"
        )
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.shape != (len(x),)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            "Step7-v4.1 classifier produced invalid probabilities"
        )
    return values


def fit_weighted_prevalence(
    rows: list[dict], policy: dict
) -> dict:
    labels = labels_array(rows)
    weights = component_weights(rows, policy)
    prevalence = float(np.average(labels, weights=weights))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Step7-v4.1 null prevalence is degenerate")
    return {
        "classifier_id": "weighted_prevalence",
        "probability": prevalence,
        "fit_audit": {
            "fit_completed": True,
            "solver_converged": True,
            "internal_estimator_fit_count": 1,
        },
    }


def apply_weighted_prevalence(row_count: int, artifact: dict) -> np.ndarray:
    return np.full(
        int(row_count), float(artifact["probability"]), dtype=np.float64
    )


def metric_key(result: dict) -> tuple[float, float, float, float]:
    return (
        float(result["component_equal_average_precision"]),
        float(result["row_average_precision"]),
        float(result["component_equal_roc_auc"]),
        float(result["row_roc_auc"]),
    )


def compact_fit_audit(audit: dict) -> dict:
    output = {
        "fit_completed": bool(audit["fit_completed"]),
        "solver_converged": bool(audit["solver_converged"]),
        "internal_estimator_fit_count": int(
            audit["internal_estimator_fit_count"]
        ),
    }
    if "solver_iterations" in audit:
        output["solver_iterations"] = json_ready(
            audit["solver_iterations"]
        )
    if "solver_final_normalized_gradient_inf_norm" in audit:
        output["solver_final_normalized_gradient_inf_norm"] = float(
            audit["solver_final_normalized_gradient_inf_norm"]
        )
    if "tree_count" in audit:
        output["requested_tree_count"] = int(
            audit["requested_tree_count"]
        )
        output["tree_count"] = int(audit["tree_count"])
        output["full_requested_tree_budget_materialized"] = bool(
            audit["full_requested_tree_budget_materialized"]
        )
        output["natural_no_further_split_termination_allowed"] = (
            True
        )
        output["best_iteration"] = audit["best_iteration"]
    if "calibration_split" in audit:
        calibration_fits = audit["calibration_fits"]
        output["calibration_fold_count"] = len(calibration_fits)
        output["calibration_assignment_sha256"] = audit[
            "calibration_split"
        ]["assignment_sha256"]
        output["calibration_component_overlap_count"] = sum(
            int(item["component_overlap_count"])
            for item in audit["calibration_split"]["folds"]
        )
        output["calibration_fold_local_preprocessing_audit_sha256"] = (
            canonical_hash(
                [
                    {
                        "fold": item["fold"],
                        "imputation_median_count": item[
                            "imputation_median_count"
                        ],
                        "imputation_medians_sha256": item[
                            "imputation_medians_sha256"
                        ],
                        "feature_reference_audit": item[
                            "feature_reference_audit"
                        ],
                    }
                    for item in calibration_fits
                ]
            )
        )
        output["calibration_support_vector_counts"] = [
            int(item["base_fit"]["support_vector_count"])
            for item in calibration_fits
        ]
        output["final_support_vector_count"] = int(
            audit["final_base_fit"]["support_vector_count"]
        )
        output["calibration_logistic"] = audit[
            "calibration_logistic"
        ]
    return output


def compact_tuning_audit(tuned: dict) -> dict:
    grid = tuned["grid"]
    return {
        "classifier_id": tuned["classifier_id"],
        "selected_parameters": tuned["selected_parameters"],
        "selected_at_grid_boundary": tuned[
            "selected_at_grid_boundary"
        ],
        "search_stop_reason": tuned["search_stop_reason"],
        "adaptive_extension_count": tuned[
            "adaptive_extension_count"
        ],
        "selected_threshold": tuned["selected_threshold"],
        "threshold_selection": tuned["threshold_selection"],
        "fold_assignment_sha256": tuned[
            "fold_assignment_sha256"
        ],
        "fold_diagnostics": tuned["fold_diagnostics"],
        "formal_model_fit_count": tuned[
            "formal_model_fit_count"
        ],
        "grid_metrics": [
            {
                key: item[key]
                for key in (
                    "parameters",
                    "component_equal_average_precision",
                    "row_average_precision",
                    "component_equal_roc_auc",
                    "row_roc_auc",
                    "formal_model_fit_count",
                )
            }
            for item in grid
        ],
        "full_grid_fold_audit_sha256": canonical_hash(
            [item["folds"] for item in grid]
        ),
    }


def tuning_selection_key(
    classifier_id: str, result: dict
) -> tuple:
    return (
        *metric_key(result),
        *hyperparameter_complexity_key(
            classifier_id, result["parameters"]
        ),
    )


def tune_classifier(
    policy: dict,
    factory: parent_selector.FeatureFactory,
    rows: list[dict],
    feature_names: list[str],
    classifier_id: str,
    *,
    fold_count: int,
    fold_seed: int,
) -> dict:
    assignments = parent_selector.balanced_component_folds(
        rows, int(fold_count), int(fold_seed)
    )
    designs = []
    for fold in range(int(fold_count)):
        fit_rows = [
            row
            for row in rows
            if assignments[row["component_id"]] != fold
        ]
        hold_rows = [
            row
            for row in rows
            if assignments[row["component_id"]] == fold
        ]
        fit_matrix, hold_matrix, medians, reference_audit = (
            factory.design(fit_rows, hold_rows, feature_names)
        )
        designs.append(
            {
                "fold": fold,
                "fit_rows": fit_rows,
                "hold_rows": hold_rows,
                "fit_matrix": fit_matrix,
                "hold_matrix": hold_matrix,
                "imputation_median_count": len(medians),
                "imputation_medians_sha256": canonical_hash(medians),
                "reference_audit": reference_audit,
            }
        )
    row_index = {
        row["pair_uid"]: index for index, row in enumerate(rows)
    }
    labels = labels_array(rows)
    weights = component_weights(rows, policy)

    def evaluate(parameters: dict) -> dict:
        oof = np.full(len(rows), np.nan, dtype=np.float64)
        fold_records = []
        formal_model_fit_count = 0
        for design in designs:
            model_seed = derived_seed(
                fold_seed,
                classifier_id,
                canonical_hash(parameters),
                design["fold"],
            )
            artifact = fit_classifier(
                policy,
                classifier_id,
                parameters,
                design["fit_matrix"],
                design["fit_rows"],
                seed=model_seed,
                factory=factory,
                feature_names=feature_names,
            )
            probabilities = apply_classifier(
                design["hold_matrix"], artifact
            )
            indices = np.asarray(
                [
                    row_index[row["pair_uid"]]
                    for row in design["hold_rows"]
                ],
                dtype=int,
            )
            oof[indices] = probabilities
            fit_count = int(
                artifact["fit_audit"]["internal_estimator_fit_count"]
            )
            formal_model_fit_count += fit_count
            fold_records.append(
                {
                    "fold": design["fold"],
                    "fit_row_count": len(design["fit_rows"]),
                    "holdout_row_count": len(design["hold_rows"]),
                    "imputation_median_count": design[
                        "imputation_median_count"
                    ],
                    "imputation_medians_sha256": design[
                        "imputation_medians_sha256"
                    ],
                    "reference_audit": design["reference_audit"],
                    "model_seed": model_seed,
                    "fit_audit": compact_fit_audit(
                        artifact["fit_audit"]
                    ),
                }
            )
        if not np.all(np.isfinite(oof)):
            raise ValueError(
                "Step7-v4.1 inner OOF probabilities are incomplete"
            )
        return {
            "parameters": dict(parameters),
            "component_equal_average_precision": (
                solver.weighted_average_precision(labels, oof, weights)
            ),
            "row_average_precision": solver.average_precision(labels, oof),
            "component_equal_roc_auc": parent_selector.weighted_roc_auc(
                labels, oof, weights
            ),
            "row_roc_auc": solver.roc_auc(labels, oof),
            "folds": fold_records,
            "formal_model_fit_count": formal_model_fit_count,
            "oof_scores": oof,
        }

    results = [
        evaluate(parameters)
        for parameters in hyperparameter_grid(policy, classifier_id)
    ]
    stop_reason = "fixed_preregistered_grid"
    extension_count = 0
    if classifier_id == "l2_logistic":
        cfg = policy["classifiers"]["l2_logistic"]
        stop_reason = "selected_interior"
        while True:
            results.sort(
                key=lambda item: float(
                    item["parameters"]["l2_penalty"]
                )
            )
            selected = max(
                results,
                key=lambda item: tuning_selection_key(
                    classifier_id, item
                ),
            )
            position = results.index(selected)
            if position not in {0, len(results) - 1}:
                stop_reason = "selected_interior"
                break
            alternatives = [
                item for item in results if item is not selected
            ]
            strict_improvement = (
                not alternatives
                or metric_key(selected)
                > max(metric_key(item) for item in alternatives)
            )
            if not strict_improvement:
                stop_reason = (
                    "boundary_without_strict_metric_improvement"
                )
                break
            current = float(
                selected["parameters"]["l2_penalty"]
            )
            factor = float(cfg["boundary_extension_factor"])
            if position == 0:
                proposed = current / factor
                if proposed < float(cfg["minimum_l2"]) * (
                    1.0 - 1e-12
                ):
                    stop_reason = "lower_search_limit_reached"
                    break
            else:
                proposed = current * factor
                if proposed > float(cfg["maximum_l2"]) * (
                    1.0 + 1e-12
                ):
                    stop_reason = "upper_search_limit_reached"
                    break
            if any(
                math.isclose(
                    proposed,
                    float(item["parameters"]["l2_penalty"]),
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
                for item in results
            ):
                raise ValueError(
                    "Step7-v4.1 adaptive L2 search repeated a value"
                )
            results.append(evaluate({"l2_penalty": proposed}))
            extension_count += 1

    if classifier_id == "l2_logistic":
        results.sort(
            key=lambda item: float(item["parameters"]["l2_penalty"])
        )
    selected = max(
        results,
        key=lambda item: tuning_selection_key(classifier_id, item),
    )
    threshold, threshold_audit = solver.choose_threshold(
        labels, selected["oof_scores"], weights
    )
    return {
        "classifier_id": classifier_id,
        "selected_parameters": dict(selected["parameters"]),
        "selected_at_grid_boundary": (
            bool(selected is results[0] or selected is results[-1])
            if classifier_id == "l2_logistic"
            else None
        ),
        "search_stop_reason": stop_reason,
        "adaptive_extension_count": extension_count,
        "grid": [
            {
                key: json_ready(value)
                for key, value in item.items()
                if key != "oof_scores"
            }
            for item in results
        ],
        "oof_scores": selected["oof_scores"],
        "selected_threshold": float(threshold),
        "threshold_selection": threshold_audit,
        "fold_assignment_sha256": canonical_hash(assignments),
        "fold_diagnostics": solver.component_fold_diagnostics(
            rows, assignments, int(fold_count)
        ),
        "formal_model_fit_count": sum(
            int(item["formal_model_fit_count"]) for item in results
        ),
    }


def candidate_rank(candidate_results: dict[str, dict]) -> list[str]:
    return sorted(
        candidate_results,
        key=lambda candidate_id: (
            -float(
                candidate_results[candidate_id]["metrics"][
                    "component_equal"
                ]["average_precision"]
            ),
            -float(
                candidate_results[candidate_id]["metrics"]["row"][
                    "average_precision"
                ]
            ),
            -float(
                candidate_results[candidate_id]["metrics"][
                    "component_equal"
                ]["roc_auc"]
            ),
            -float(
                candidate_results[candidate_id]["metrics"]["row"][
                    "roc_auc"
                ]
            ),
            int(candidate_results[candidate_id]["feature_count"]),
            candidate_id,
        ),
    )


def run_nested_selection(
    policy: dict,
    parent_policy: dict,
    factory: parent_selector.FeatureFactory,
    train_rows: list[dict],
    *,
    progress_label: str,
) -> tuple[dict[str, dict], list[str], dict]:
    specs = candidate_specs(policy, parent_policy)
    cfg = policy["training"]
    row_index = {
        row["pair_uid"]: index
        for index, row in enumerate(train_rows)
    }
    state = {
        spec["id"]: {
            **{
                key: value
                for key, value in spec.items()
                if key != "id"
            },
            "candidate_id": spec["id"],
            "outer_selected_hyperparameters": [],
            "outer_fold_records": [],
            "seed_scores": [],
            "seed_metrics": [],
            "formal_model_fit_count": 0,
        }
        for spec in specs
    }
    outer_audit = []
    for outer_seed in cfg["outer_seeds"]:
        print(
            f"[Step7-v4.1/{progress_label}] outer repeat "
            f"seed={outer_seed}",
            flush=True,
        )
        assignments = parent_selector.balanced_component_folds(
            train_rows,
            int(cfg["outer_fold_count"]),
            int(outer_seed),
        )
        seed_predictions = {
            spec["id"]: np.full(
                len(train_rows), np.nan, dtype=np.float64
            )
            for spec in specs
        }
        for outer_fold in range(int(cfg["outer_fold_count"])):
            outer_fit = [
                row
                for row in train_rows
                if assignments[row["component_id"]] != outer_fold
            ]
            outer_hold = [
                row
                for row in train_rows
                if assignments[row["component_id"]] == outer_fold
            ]
            fit_sellers = {
                row[endpoint]
                for row in outer_fit
                for endpoint in ("seller_uid_left", "seller_uid_right")
            }
            hold_sellers = {
                row[endpoint]
                for row in outer_hold
                for endpoint in ("seller_uid_left", "seller_uid_right")
            }
            if fit_sellers & hold_sellers:
                raise ValueError(
                    "Step7-v4.1 outer component fold leaks a seller"
                )
            print(
                f"[Step7-v4.1/{progress_label}] seed={outer_seed} "
                f"fold={outer_fold + 1}/"
                f"{int(cfg['outer_fold_count'])}",
                flush=True,
            )
            inner_seed = derived_seed(
                outer_seed, outer_fold, "inner_component_folds"
            )
            for candidate_index, spec in enumerate(specs, start=1):
                candidate_id = spec["id"]
                print(
                    f"[Step7-v4.1/{progress_label}] fitting "
                    f"{candidate_id} ({candidate_index}/{len(specs)})",
                    flush=True,
                )
                if candidate_id == NULL_CANDIDATE_ID:
                    artifact = fit_weighted_prevalence(
                        outer_fit, policy
                    )
                    probabilities = apply_weighted_prevalence(
                        len(outer_hold), artifact
                    )
                    tuned = None
                    medians: list[float] = []
                    reference_audit = {
                        "status": "not_applicable_no_features"
                    }
                    selected_parameters = {
                        "weighted_prevalence": artifact["probability"]
                    }
                    fit_count = 1
                else:
                    tuned = tune_classifier(
                        policy,
                        factory,
                        outer_fit,
                        spec["feature_names"],
                        spec["classifier_id"],
                        fold_count=int(cfg["inner_fold_count"]),
                        fold_seed=inner_seed,
                    )
                    fit_matrix, hold_matrix, medians, reference_audit = (
                        factory.design(
                            outer_fit,
                            outer_hold,
                            spec["feature_names"],
                        )
                    )
                    artifact = fit_classifier(
                        policy,
                        spec["classifier_id"],
                        tuned["selected_parameters"],
                        fit_matrix,
                        outer_fit,
                        seed=derived_seed(
                            outer_seed,
                            outer_fold,
                            candidate_id,
                            "outer_final",
                        ),
                        factory=factory,
                        feature_names=spec["feature_names"],
                    )
                    probabilities = apply_classifier(
                        hold_matrix, artifact
                    )
                    selected_parameters = dict(
                        tuned["selected_parameters"]
                    )
                    fit_count = int(
                        tuned["formal_model_fit_count"]
                    ) + int(
                        artifact["fit_audit"][
                            "internal_estimator_fit_count"
                        ]
                    )
                indices = np.asarray(
                    [
                        row_index[row["pair_uid"]]
                        for row in outer_hold
                    ],
                    dtype=int,
                )
                seed_predictions[candidate_id][indices] = probabilities
                state[candidate_id][
                    "outer_selected_hyperparameters"
                ].append(selected_parameters)
                state[candidate_id]["formal_model_fit_count"] += fit_count
                state[candidate_id]["outer_fold_records"].append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": outer_fold,
                        "fit_row_count": len(outer_fit),
                        "holdout_row_count": len(outer_hold),
                        "fit_holdout_seller_overlap_count": 0,
                        "inner_seed": inner_seed,
                        "inner_tuning": (
                            None
                            if tuned is None
                            else compact_tuning_audit(tuned)
                        ),
                        "outer_imputation_median_count": len(medians),
                        "outer_imputation_medians_sha256": canonical_hash(
                            medians
                        ),
                        "outer_reference_audit": reference_audit,
                        "outer_fit_audit": compact_fit_audit(
                            artifact["fit_audit"]
                        ),
                    }
                )
        seed_result_view = {}
        for spec in specs:
            candidate_id = spec["id"]
            scores = seed_predictions[candidate_id]
            if not np.all(np.isfinite(scores)):
                raise ValueError(
                    "Step7-v4.1 outer OOF probabilities are incomplete"
                )
            weights = component_weights(train_rows, policy)
            threshold, _threshold_audit = solver.choose_threshold(
                labels_array(train_rows), scores, weights
            )
            metrics = parent_selector.full_metrics(
                train_rows, scores, threshold
            )
            state[candidate_id]["seed_scores"].append(scores)
            state[candidate_id]["seed_metrics"].append(
                {
                    "outer_seed": int(outer_seed),
                    "selected_threshold_diagnostic": float(threshold),
                    "metrics": metrics,
                }
            )
            seed_result_view[candidate_id] = {
                "feature_count": spec["feature_count"],
                "metrics": metrics,
            }
        outer_audit.append(
            {
                "outer_seed": int(outer_seed),
                "component_fold_assignment_sha256": canonical_hash(
                    assignments
                ),
                "fold_diagnostics": solver.component_fold_diagnostics(
                    train_rows,
                    assignments,
                    int(cfg["outer_fold_count"]),
                ),
                "seed_winner": candidate_rank(seed_result_view)[0],
            }
        )

    results = {}
    labels = labels_array(train_rows)
    weights = component_weights(train_rows, policy)
    for spec in specs:
        candidate_id = spec["id"]
        item = state[candidate_id]
        scores_by_seed = np.vstack(item.pop("seed_scores"))
        expected_shape = (
            len(cfg["outer_seeds"]),
            len(train_rows),
        )
        if scores_by_seed.shape != expected_shape:
            raise AssertionError(
                "Step7-v4.1 repeated outer OOF shape drift"
            )
        mean_scores = np.mean(scores_by_seed, axis=0)
        threshold, threshold_audit = solver.choose_threshold(
            labels, mean_scores, weights
        )
        results[candidate_id] = {
            **item,
            "selected_threshold": float(threshold),
            "threshold_selection": threshold_audit,
            "mean_repeated_nested_oof_scores": mean_scores,
            "outer_seed_oof_scores": scores_by_seed,
            "metrics": parent_selector.full_metrics(
                train_rows, mean_scores, threshold
            ),
            "all_formal_fits_converged": True,
        }
    ranking = candidate_rank(results)
    return results, ranking, {
        "outer_fold_count": int(cfg["outer_fold_count"]),
        "outer_seeds": list(cfg["outer_seeds"]),
        "inner_fold_count": int(cfg["inner_fold_count"]),
        "outer_seed_audit": outer_audit,
        "all_candidate_formal_model_fit_count": sum(
            int(result["formal_model_fit_count"])
            for result in results.values()
        ),
        "all_formal_fits_converged": True,
    }


def preflight_nested_support(
    policy: dict, rows: list[dict], *, role: str
) -> dict:
    cfg = policy["training"]
    calibration_folds = int(
        policy["classifiers"]["rbf_svm"]["calibration_fold_count"]
    )
    checked_fit_subsets = {}

    def check_calibration_support(
        fit_rows: list[dict], seed: int, subset_role: str
    ) -> None:
        assignments = parent_selector.balanced_component_folds(
            fit_rows, calibration_folds, seed
        )
        key = canonical_hash(
            sorted(row["pair_uid"] for row in fit_rows)
        )
        checked_fit_subsets.setdefault(
            key,
            {
                "role_example": subset_role,
                "row_count": len(fit_rows),
                "component_count": len(
                    {row["component_id"] for row in fit_rows}
                ),
                "positive_count": sum(
                    row["review_label"] == "positive"
                    for row in fit_rows
                ),
                "negative_count": sum(
                    row["review_label"] == "negative"
                    for row in fit_rows
                ),
                "calibration_assignment_sha256": canonical_hash(
                    assignments
                ),
            },
        )

    for outer_seed in cfg["outer_seeds"]:
        outer_assignments = parent_selector.balanced_component_folds(
            rows,
            int(cfg["outer_fold_count"]),
            int(outer_seed),
        )
        for outer_fold in range(int(cfg["outer_fold_count"])):
            outer_fit = [
                row
                for row in rows
                if outer_assignments[row["component_id"]] != outer_fold
            ]
            check_calibration_support(
                outer_fit,
                derived_seed(
                    outer_seed,
                    outer_fold,
                    "preflight_outer_calibration",
                ),
                "outer_fit",
            )
            inner_seed = derived_seed(
                outer_seed, outer_fold, "inner_component_folds"
            )
            inner_assignments = parent_selector.balanced_component_folds(
                outer_fit,
                int(cfg["inner_fold_count"]),
                inner_seed,
            )
            for inner_fold in range(int(cfg["inner_fold_count"])):
                inner_fit = [
                    row
                    for row in outer_fit
                    if inner_assignments[row["component_id"]]
                    != inner_fold
                ]
                check_calibration_support(
                    inner_fit,
                    derived_seed(
                        outer_seed,
                        outer_fold,
                        inner_fold,
                        "preflight_inner_calibration",
                    ),
                    "inner_fit",
                )
    final_seed = int(cfg["final_hyperparameter_seed"])
    final_assignments = parent_selector.balanced_component_folds(
        rows,
        int(cfg["final_hyperparameter_fold_count"]),
        final_seed,
    )
    for fold in range(int(cfg["final_hyperparameter_fold_count"])):
        final_inner_fit = [
            row
            for row in rows
            if final_assignments[row["component_id"]] != fold
        ]
        check_calibration_support(
            final_inner_fit,
            derived_seed(final_seed, fold, "preflight_final_tuning"),
            "final_hyperparameter_fit",
        )
    check_calibration_support(
        rows,
        derived_seed(final_seed, "preflight_full_fit"),
        "full_fit",
    )
    return {
        "status": "pass",
        "role": role,
        "row_count": len(rows),
        "positive_count": sum(
            row["review_label"] == "positive" for row in rows
        ),
        "negative_count": sum(
            row["review_label"] == "negative" for row in rows
        ),
        "component_count": len(
            {row["component_id"] for row in rows}
        ),
        "unique_classifier_fit_subsets_checked": len(
            checked_fit_subsets
        ),
        "minimum_fit_subset_positive_count": min(
            item["positive_count"]
            for item in checked_fit_subsets.values()
        ),
        "minimum_fit_subset_negative_count": min(
            item["negative_count"]
            for item in checked_fit_subsets.values()
        ),
        "minimum_fit_subset_component_count": min(
            item["component_count"]
            for item in checked_fit_subsets.values()
        ),
        "fit_subset_audit_sha256": canonical_hash(
            checked_fit_subsets
        ),
        "calibration_fold_count": calibration_folds,
    }


def result_scores(results: dict[str, dict]) -> dict[str, np.ndarray]:
    return {
        candidate_id: np.asarray(
            result["mean_repeated_nested_oof_scores"],
            dtype=np.float64,
        )
        for candidate_id, result in results.items()
    }


def assess_selection(
    policy: dict,
    train_rows: list[dict],
    results: dict[str, dict],
    ranking: list[str],
    nested_audit: dict,
    no_clone_rows: list[dict],
    no_clone_results: dict[str, dict],
    no_clone_ranking: list[str],
    no_clone_nested_audit: dict,
) -> dict:
    bootstrap = policy["evaluation"]["bootstrap"]
    resamples = int(bootstrap["resamples"])
    confidence = float(bootstrap["confidence"])
    seed = int(bootstrap["seed"])
    winner, runner_up = ranking[:2]
    winner_delta = parent_selector.grouped_bootstrap_delta(
        train_rows,
        results[winner]["mean_repeated_nested_oof_scores"],
        results[runner_up]["mean_repeated_nested_oof_scores"],
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )
    simultaneous = (
        parent_selector.grouped_bootstrap_winner_above_all(
            train_rows,
            result_scores(results),
            winner,
            resamples=resamples,
            seed=derived_seed(seed, "overall_winner_above_all"),
            confidence=confidence,
        )
    )
    no_clone_simultaneous = (
        parent_selector.grouped_bootstrap_winner_above_all(
            no_clone_rows,
            result_scores(no_clone_results),
            winner,
            resamples=resamples,
            seed=derived_seed(
                seed, "overall_winner_above_all_no_clone"
            ),
            confidence=confidence,
        )
    )
    seed_winners = [
        item["seed_winner"]
        for item in nested_audit["outer_seed_audit"]
    ]
    no_clone_seed_winners = [
        item["seed_winner"]
        for item in no_clone_nested_audit["outer_seed_audit"]
    ]
    winner_rate = seed_winners.count(winner) / len(seed_winners)
    no_clone_winner_rate = (
        no_clone_seed_winners.count(winner)
        / len(no_clone_seed_winners)
    )
    overall_rule = policy["selection_rule"][
        "overall_unique_current_best_requires"
    ]
    overall_unique = bool(
        winner_rate
        >= float(
            overall_rule[
                "winner_rate_across_outer_repeats_at_least"
            ]
        )
        and winner_delta["probability_delta_above_zero"]
        >= float(
            overall_rule[
                "component_bootstrap_probability_delta_above_runner_up_at_least"
            ]
        )
        and simultaneous[
            "probability_winner_strictly_above_all_candidates"
        ]
        >= float(
            overall_rule[
                "simultaneous_component_bootstrap_probability_winner_above_all_candidates_at_least"
            ]
        )
        and no_clone_winner_rate
        >= float(
            overall_rule[
                "no_exact_clone_nested_winner_rate_across_outer_repeats_at_least"
            ]
        )
        and no_clone_simultaneous[
            "probability_winner_strictly_above_all_candidates"
        ]
        >= float(
            overall_rule[
                "no_exact_clone_component_bootstrap_probability_winner_above_all_candidates_at_least"
            ]
        )
        and nested_audit["all_formal_fits_converged"] is True
        and no_clone_nested_audit["all_formal_fits_converged"] is True
    )

    c0_ids = [
        candidate_id
        for candidate_id in ranking
        if results[candidate_id]["feature_set_id"] == "legacy18"
    ]
    transfer_ids = [
        candidate_id
        for candidate_id in ranking
        if results[candidate_id]["transfer_eligible"]
    ]
    if len(c0_ids) != len(CLASSIFIER_IDS) or not transfer_ids:
        raise AssertionError(
            "Step7-v4.1 C0/transfer candidate universe drift"
        )
    best_c0 = c0_ids[0]
    best_transfer = transfer_ids[0]
    best_transfer_vs_c0 = (
        parent_selector.grouped_bootstrap_delta(
            train_rows,
            results[best_transfer][
                "mean_repeated_nested_oof_scores"
            ],
            results[best_c0]["mean_repeated_nested_oof_scores"],
            resamples=resamples,
            seed=derived_seed(seed, "best_transfer_vs_best_c0"),
            confidence=confidence,
        )
    )
    no_clone_c0_ids = [
        candidate_id
        for candidate_id in no_clone_ranking
        if no_clone_results[candidate_id]["feature_set_id"]
        == "legacy18"
    ]
    best_no_clone_c0 = no_clone_c0_ids[0]
    no_clone_transfer_vs_c0 = (
        parent_selector.grouped_bootstrap_delta(
            no_clone_rows,
            no_clone_results[best_transfer][
                "mean_repeated_nested_oof_scores"
            ],
            no_clone_results[best_no_clone_c0][
                "mean_repeated_nested_oof_scores"
            ],
            resamples=resamples,
            seed=derived_seed(
                seed, "best_transfer_vs_best_no_clone_c0"
            ),
            confidence=confidence,
        )
    )
    transfer_rule = policy["selection_rule"][
        "transfer_capable_m0_additionally_requires"
    ]
    transfer_internal_gate = bool(
        overall_unique
        and winner == best_transfer
        and results[winner]["transfer_eligible"] is True
        and best_transfer_vs_c0["ci_lower"]
        > float(
            transfer_rule[
                "component_bootstrap_ci_lower_above_best_legacy18_c0"
            ]
        )
        and best_transfer_vs_c0["probability_delta_above_zero"]
        >= float(
            transfer_rule[
                "component_bootstrap_probability_above_best_legacy18_c0_at_least"
            ]
        )
        and no_clone_transfer_vs_c0["probability_delta_above_zero"]
        >= float(
            transfer_rule[
                "no_exact_clone_component_bootstrap_probability_above_best_legacy18_c0_at_least"
            ]
        )
    )

    family_comparisons = {}
    for feature_set_id in FEATURE_SET_IDS:
        ids = [
            candidate_id
            for candidate_id in ranking
            if results[candidate_id]["feature_set_id"]
            == feature_set_id
        ]
        if len(ids) != len(CLASSIFIER_IDS):
            raise AssertionError(
                "Step7-v4.1 classifier-family comparison drift"
            )
        family_winner = ids[0]
        stability = (
            parent_selector.grouped_bootstrap_winner_above_all(
                train_rows,
                {
                    candidate_id: results[candidate_id][
                        "mean_repeated_nested_oof_scores"
                    ]
                    for candidate_id in ids
                },
                family_winner,
                resamples=resamples,
                seed=derived_seed(
                    seed, "classifier_family", feature_set_id
                ),
                confidence=confidence,
            )
        )
        family_comparisons[feature_set_id] = {
            "candidate_ranking": ids,
            "winner": family_winner,
            "winner_classifier_id": results[family_winner][
                "classifier_id"
            ],
            "winner_vs_other_classifiers_component_bootstrap": (
                stability
            ),
        }

    return {
        "best_current_pipeline": winner,
        "runner_up": runner_up,
        "seed_winners": seed_winners,
        "winner_rate_across_outer_seeds": winner_rate,
        "winner_vs_runner_up_component_bootstrap": winner_delta,
        "winner_vs_all_candidates_simultaneous_component_bootstrap": (
            simultaneous
        ),
        "overall_unique_current_best_gate_passed": overall_unique,
        "overall_selection_status": (
            "stable_current_best_style_free_pipeline_posthoc_requires_new_confirmation"
            if overall_unique
            else "no_stable_unique_current_best_style_free_pipeline"
        ),
        "no_exact_clone_robustness": {
            "training_contract": (
                "complete_nested_retraining_after_removing_every_pair_"
                "with_exact_clean_title_or_description_overlap"
            ),
            "row_count": len(no_clone_rows),
            "positive_count": sum(
                row["review_label"] == "positive"
                for row in no_clone_rows
            ),
            "negative_count": sum(
                row["review_label"] == "negative"
                for row in no_clone_rows
            ),
            "candidate_ranking": no_clone_ranking,
            "outer_seed_winners": no_clone_seed_winners,
            "original_winner_rate": no_clone_winner_rate,
            "original_winner_vs_all_candidates_bootstrap": (
                no_clone_simultaneous
            ),
        },
        "best_legacy18_c0": best_c0,
        "best_transfer_candidate": best_transfer,
        "best_transfer_vs_best_c0_component_bootstrap": (
            best_transfer_vs_c0
        ),
        "best_no_clone_legacy18_c0": best_no_clone_c0,
        "best_transfer_vs_best_no_clone_c0_component_bootstrap": (
            no_clone_transfer_vs_c0
        ),
        "transfer_capable_internal_gate_passed": transfer_internal_gate,
        "transfer_capable_m0_status": (
            "posthoc_transfer_candidate_passed_internal_gates_requires_new_real_english_confirmation"
            if transfer_internal_gate
            else "no_transfer_capable_m0"
        ),
        "formal_m0_certified": False,
        "formal_m0_certification_blocker": (
            "post_hoc_design_and_no_new_real_english_confirmation_set"
        ),
        "classifier_family_comparisons": family_comparisons,
        "validation_metrics_used_for_selection": False,
        "historical_test_labels_read": False,
        "new_real_english_confirmation_required": True,
    }


def tune_final_hyperparameters(
    policy: dict,
    parent_policy: dict,
    factory: parent_selector.FeatureFactory,
    train_rows: list[dict],
    nested_results: dict[str, dict],
) -> dict[str, dict]:
    cfg = policy["training"]
    output = {}
    for candidate_index, spec in enumerate(
        candidate_specs(policy, parent_policy), start=1
    ):
        candidate_id = spec["id"]
        print(
            f"[Step7-v4.1/final-tuning] {candidate_id} "
            f"({candidate_index}/"
            f"{len(candidate_specs(policy, parent_policy))})",
            flush=True,
        )
        if candidate_id == NULL_CANDIDATE_ID:
            prevalence = fit_weighted_prevalence(
                train_rows, policy
            )["probability"]
            output[candidate_id] = {
                "classifier_id": "weighted_prevalence",
                "selected_parameters": {
                    "weighted_prevalence": prevalence
                },
                "training_contract": "all_train_weighted_prevalence",
            }
            continue
        tuned = tune_classifier(
            policy,
            factory,
            train_rows,
            spec["feature_names"],
            spec["classifier_id"],
            fold_count=int(cfg["final_hyperparameter_fold_count"]),
            fold_seed=int(cfg["final_hyperparameter_seed"]),
        )
        output[candidate_id] = compact_tuning_audit(tuned)
        if (
            nested_results[candidate_id]["all_formal_fits_converged"]
            is not True
        ):
            raise ValueError(
                "Step7-v4.1 final tuning followed a failed nested fit"
            )
    return output


def selected_model_ids(decision: dict) -> list[str]:
    ordered = [
        decision["best_current_pipeline"],
        decision["best_transfer_candidate"],
        decision["best_legacy18_c0"],
    ]
    output = []
    for candidate_id in ordered:
        if candidate_id not in output:
            output.append(candidate_id)
    return output


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def classifier_model_audit(artifact: dict) -> dict:
    classifier_id = artifact["classifier_id"]
    output = {
        "classifier_id": classifier_id,
        "parameters": json_ready(artifact.get("parameters", {})),
        "fit_audit": compact_fit_audit(artifact["fit_audit"]),
    }
    if classifier_id == "l2_logistic":
        output["replayable_logistic_artifact"] = json_ready(
            artifact["model"]
        )
    elif classifier_id == "rbf_svm":
        model = artifact["model"]
        output.update(
            {
                "support_vector_count": int(len(model.support_)),
                "support_vectors_sha256": array_sha256(
                    model.support_vectors_
                ),
                "dual_coefficients_sha256": array_sha256(
                    model.dual_coef_
                ),
                "intercept_sha256": array_sha256(model.intercept_),
                "standardization_mean_sha256": array_sha256(
                    artifact["mean"]
                ),
                "standardization_scale_sha256": array_sha256(
                    artifact["scale"]
                ),
                "calibration_logistic": json_ready(
                    artifact["calibration_logistic"]
                ),
            }
        )
    elif classifier_id == "lightgbm":
        model_text = artifact["model"].booster_.model_to_string()
        output.update(
            {
                "booster_model_text_sha256": hashlib.sha256(
                    model_text.encode("utf-8")
                ).hexdigest(),
                "booster_model_text_character_count": len(model_text),
                "tree_count": int(
                    artifact["model"].booster_.num_trees()
                ),
            }
        )
    elif classifier_id == "weighted_prevalence":
        output["probability"] = float(artifact["probability"])
    else:
        raise ValueError(
            f"Step7-v4.1 unknown final classifier: {classifier_id}"
        )
    return output


def serialize_joblib(value: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3, protocol=5)
    return buffer.getvalue()


def fit_final_candidates_and_score_unlabelled_valid(
    policy: dict,
    parent_policy: dict,
    factory: parent_selector.FeatureFactory,
    train_rows: list[dict],
    valid_pair_rows: list[dict],
    nested_results: dict[str, dict],
    final_tuning: dict[str, dict],
    decision: dict,
) -> tuple[dict, dict[str, np.ndarray], dict[str, bytes]]:
    valid_scores = {}
    artifact_metadata = {}
    selected_ids = set(selected_model_ids(decision))
    selected_bundles: dict[str, bytes] = {}
    for candidate_index, spec in enumerate(
        candidate_specs(policy, parent_policy), start=1
    ):
        candidate_id = spec["id"]
        print(
            f"[Step7-v4.1/final-fit] {candidate_id} "
            f"({candidate_index}/"
            f"{len(candidate_specs(policy, parent_policy))})",
            flush=True,
        )
        if candidate_id == NULL_CANDIDATE_ID:
            artifact = fit_weighted_prevalence(train_rows, policy)
            scores = apply_weighted_prevalence(
                len(valid_pair_rows), artifact
            )
            valid_matrix = None
            medians = []
            reference_audit = {
                "status": "not_applicable_no_features"
            }
        else:
            fit_matrix, valid_matrix, medians, reference_audit = (
                factory.design(
                    train_rows,
                    valid_pair_rows,
                    spec["feature_names"],
                )
            )
            artifact = fit_classifier(
                policy,
                spec["classifier_id"],
                final_tuning[candidate_id]["selected_parameters"],
                fit_matrix,
                train_rows,
                seed=derived_seed(
                    policy["training"]["final_hyperparameter_seed"],
                    candidate_id,
                    "final_all_train_fit",
                ),
                factory=factory,
                feature_names=spec["feature_names"],
            )
            scores = apply_classifier(valid_matrix, artifact)
        valid_scores[candidate_id] = scores
        metadata = {
            "candidate_id": candidate_id,
            "classifier_id": spec["classifier_id"],
            "feature_set_id": spec["feature_set_id"],
            "blocks": spec["blocks"],
            "feature_names": spec["feature_names"],
            "feature_count": spec["feature_count"],
            "transfer_eligible": spec["transfer_eligible"],
            "selected_threshold": float(
                nested_results[candidate_id]["selected_threshold"]
            ),
            "final_hyperparameter_selection": final_tuning[
                candidate_id
            ],
            "imputation_medians": medians,
            "feature_reference_audit": reference_audit,
            "model_audit": classifier_model_audit(artifact),
            "fit_row_count": len(train_rows),
            "unlabelled_valid_score_count": len(valid_pair_rows),
            "valid_label_values_read_for_fit_or_scoring": False,
            "historical_test_label_values_read": False,
        }
        artifact_metadata[candidate_id] = metadata
        if candidate_id in selected_ids:
            bundle = {
                "step": "step7_v4_1_selected_style_free_pipeline",
                "version": policy["version"],
                "candidate": {
                    key: value
                    for key, value in metadata.items()
                    if key != "model_audit"
                },
                "classifier_artifact": artifact,
                "policy_sha256": sha256_file(POLICY_PATH),
                "producer_sha256": sha256_file(SCRIPT_PATH),
                "valid_label_values_read_for_fit_or_scoring": False,
                "historical_test_label_values_read": False,
            }
            payload = serialize_joblib(bundle)
            replayed = joblib.load(io.BytesIO(payload))
            if candidate_id == NULL_CANDIDATE_ID:
                replayed_scores = apply_weighted_prevalence(
                    len(valid_pair_rows),
                    replayed["classifier_artifact"],
                )
            else:
                if valid_matrix is None:
                    raise AssertionError(
                        "Step7-v4.1 selected model lost its valid matrix"
                    )
                replayed_scores = apply_classifier(
                    valid_matrix,
                    replayed["classifier_artifact"],
                )
            if not np.array_equal(
                np.asarray(scores, dtype=np.float64),
                np.asarray(replayed_scores, dtype=np.float64),
            ):
                raise ValueError(
                    "Step7-v4.1 selected joblib model replay drift"
                )
            selected_bundles[candidate_id] = payload
    return artifact_metadata, valid_scores, selected_bundles


def serialize_probability(value: float) -> str:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(
            "Step7-v4.1 cannot serialize an invalid probability"
        )
    return repr(number)


def train_prediction_rows(
    policy: dict,
    rows: list[dict],
    ranking: list[str],
    results: dict[str, dict],
) -> list[dict]:
    output = []
    seeds = list(policy["training"]["outer_seeds"])
    for candidate_id in ranking:
        seed_scores = np.asarray(
            results[candidate_id]["outer_seed_oof_scores"],
            dtype=np.float64,
        )
        mean_scores = np.asarray(
            results[candidate_id][
                "mean_repeated_nested_oof_scores"
            ],
            dtype=np.float64,
        )
        for row_index, row in enumerate(rows):
            record: dict[str, object] = {
                "pair_uid": row["pair_uid"],
                "component_id": row["component_id"],
                "review_label": row["review_label"],
                "candidate_id": candidate_id,
                "mean_repeated_nested_oof_probability": (
                    serialize_probability(mean_scores[row_index])
                ),
            }
            for seed_index, seed in enumerate(seeds):
                record[f"outer_seed_{seed}_oof_probability"] = (
                    serialize_probability(
                        seed_scores[seed_index, row_index]
                    )
                )
            output.append(record)
    return output


def blind_valid_prediction_rows(
    valid_pair_rows: list[dict],
    ranking: list[str],
    scores: dict[str, np.ndarray],
) -> list[dict]:
    output = []
    for candidate_id in ranking:
        values = np.asarray(scores[candidate_id], dtype=np.float64)
        if (
            values.shape != (len(valid_pair_rows),)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Step7-v4.1 blind valid score shape/value drift"
            )
        for row, probability in zip(
            valid_pair_rows, values, strict=True
        ):
            output.append(
                {
                    "pair_uid": row["pair_uid"],
                    "candidate_id": candidate_id,
                    "probability": serialize_probability(probability),
                }
            )
    return output


def replay_blind_valid_scores(
    valid_pair_rows: list[dict],
    ranking: list[str],
    rows: list[dict],
) -> dict[str, np.ndarray]:
    expected_count = len(valid_pair_rows) * len(ranking)
    if len(rows) != expected_count:
        raise ValueError(
            "Step7-v4.1 blind valid prediction count drift"
        )
    output = {}
    position = 0
    seen = set()
    for candidate_id in ranking:
        values = []
        for pair in valid_pair_rows:
            row = rows[position]
            position += 1
            expected_key = (pair["pair_uid"], candidate_id)
            observed_key = (row["pair_uid"], row["candidate_id"])
            if observed_key != expected_key or observed_key in seen:
                raise ValueError(
                    "Step7-v4.1 blind valid prediction order/key drift"
                )
            seen.add(observed_key)
            number = float(row["probability"])
            if serialize_probability(number) != row["probability"]:
                raise ValueError(
                    "Step7-v4.1 blind probability round-trip drift"
                )
            values.append(number)
        output[candidate_id] = np.asarray(
            values, dtype=np.float64
        )
    if position != expected_count or len(seen) != expected_count:
        raise AssertionError(
            "Step7-v4.1 blind valid replay accounting drift"
        )
    return output


def labelled_valid_prediction_rows(
    valid_rows: list[dict],
    ranking: list[str],
    scores: dict[str, np.ndarray],
) -> list[dict]:
    output = []
    for candidate_id in ranking:
        for row, probability in zip(
            valid_rows, scores[candidate_id], strict=True
        ):
            output.append(
                {
                    "pair_uid": row["pair_uid"],
                    "component_id": row["component_id"],
                    "review_label": row["review_label"],
                    "candidate_id": candidate_id,
                    "probability": serialize_probability(probability),
                }
            )
    return output


def compact_candidate_result(result: dict) -> dict:
    return {
        key: json_ready(value)
        for key, value in result.items()
        if key
        not in {
            "mean_repeated_nested_oof_scores",
            "outer_seed_oof_scores",
        }
    }


def run_selection(policy: dict) -> dict:
    pinned_input_audit = verify_pinned_inputs(policy)
    (
        parent_policy,
        preparation_manifest,
        preparation_bundle,
        fixed_features,
        _seller_records,
        seller_markets,
        data_audit,
    ) = load_style_free_parent_data(policy)
    factory = data_audit.pop("factory")
    pair_rows = preparation_bundle["pair_rows"]

    # Only train label values are opened before both locks. The valid file may
    # be byte-hashed for integrity, but its rows/labels are not parsed here.
    train_rows = parent_selector.load_label_split(
        parent_policy, pair_rows, "train"
    )
    overlap_by_pair = parent_common.exact_overlap_audit_by_pair(
        pair_rows, preparation_bundle["seller_text_rows"]
    )
    no_clone_rows = [
        row
        for row in train_rows
        if not overlap_by_pair[row["pair_uid"]][
            "any_exact_clean_text_overlap"
        ]
    ]
    if (
        not no_clone_rows
        or {row["review_label"] for row in no_clone_rows}
        != {"positive", "negative"}
    ):
        raise ValueError(
            "Step7-v4.1 no-clone retraining lacks both classes"
        )
    preflight = {
        "train": preflight_nested_support(
            policy, train_rows, role="train"
        ),
        "no_exact_clone": preflight_nested_support(
            policy, no_clone_rows, role="no_exact_clone"
        ),
    }
    print(
        "[Step7-v4.1] structural preflight passed; starting main nested CV",
        flush=True,
    )
    nested_results, ranking, nested_audit = run_nested_selection(
        policy,
        parent_policy,
        factory,
        train_rows,
        progress_label="main",
    )
    print(
        "[Step7-v4.1] starting complete no-exact-clone nested retraining",
        flush=True,
    )
    (
        no_clone_results,
        no_clone_ranking,
        no_clone_nested_audit,
    ) = run_nested_selection(
        policy,
        parent_policy,
        factory,
        no_clone_rows,
        progress_label="no-clone",
    )
    decision = assess_selection(
        policy,
        train_rows,
        nested_results,
        ranking,
        nested_audit,
        no_clone_rows,
        no_clone_results,
        no_clone_ranking,
        no_clone_nested_audit,
    )
    final_tuning = tune_final_hyperparameters(
        policy,
        parent_policy,
        factory,
        train_rows,
        nested_results,
    )

    lock_payload = {
        "step": "step7_v4_1_train_only_style_free_selection_lock",
        "version": policy["version"],
        "design_status": policy["result_scope"]["design_status"],
        "candidate_count": len(ranking),
        "candidate_ranking": ranking,
        "selection_decision": decision,
        "candidate_thresholds": {
            candidate_id: float(
                nested_results[candidate_id]["selected_threshold"]
            )
            for candidate_id in ranking
        },
        "final_hyperparameter_selection": final_tuning,
        "style_feature_count": 0,
        "author_style_encoder_score_or_runtime_files_opened": False,
        "valid_label_values_read": False,
        "historical_test_label_values_read": False,
        "policy_sha256": sha256_file(POLICY_PATH),
        "producer_sha256": sha256_file(SCRIPT_PATH),
    }
    lock_payload = json_ready(lock_payload)
    lock_payload["lock_content_sha256"] = canonical_hash(lock_payload)
    lock_path = resolve(policy["outputs"]["train_selection_lock"])
    parent_common.write_json_immutable(lock_path, lock_payload)
    observed_lock = parent_common.load_json(lock_path)
    parent_common.verify_canonical_self_hash(
        observed_lock,
        "lock_content_sha256",
        "Step7-v4.1 train-only selection lock",
    )
    if observed_lock != lock_payload:
        raise ValueError(
            "Step7-v4.1 train-only selection lock replay drift"
        )
    print(
        "[Step7-v4.1] train-only selection and final hyperparameters locked",
        flush=True,
    )

    valid_pair_rows = [
        row for row in pair_rows if row["split_name"] == "valid"
    ]
    (
        final_artifact_metadata,
        unlabelled_valid_scores,
        selected_model_payloads,
    ) = fit_final_candidates_and_score_unlabelled_valid(
        policy,
        parent_policy,
        factory,
        train_rows,
        valid_pair_rows,
        nested_results,
        final_tuning,
        decision,
    )
    selected_model_records = {}
    for candidate_id in selected_model_ids(decision):
        model_path = resolve(
            policy["outputs"]["selected_model_template"].format(
                candidate_id=candidate_id
            )
        )
        parent_common.write_bytes_immutable(
            model_path, selected_model_payloads[candidate_id]
        )
        selected_model_records[candidate_id] = (
            parent_common.file_record(model_path)
        )

    artifact_payload = {
        "step": "step7_v4_1_final_train_style_free_model_artifacts",
        "version": policy["version"],
        "train_selection_lock": parent_common.file_record(lock_path),
        "train_selection_lock_content_sha256": lock_payload[
            "lock_content_sha256"
        ],
        "best_current_pipeline": decision["best_current_pipeline"],
        "best_transfer_candidate": decision[
            "best_transfer_candidate"
        ],
        "best_legacy18_c0": decision["best_legacy18_c0"],
        "selected_replayable_joblib_models": selected_model_records,
        "joblib_model_roles": [
            "best_current_pipeline",
            "best_transfer_candidate",
            "best_legacy18_c0",
        ],
        "candidates": final_artifact_metadata,
        "valid_label_values_read_for_fit_or_scoring": False,
        "historical_test_label_values_read": False,
        "policy_sha256": sha256_file(POLICY_PATH),
        "producer_sha256": sha256_file(SCRIPT_PATH),
    }
    artifact_payload = json_ready(artifact_payload)
    artifact_payload["artifact_content_sha256"] = canonical_hash(
        artifact_payload
    )
    artifact_path = resolve(policy["outputs"]["model_artifacts"])
    parent_common.write_json_immutable(
        artifact_path, artifact_payload
    )

    train_prediction_path = resolve(
        policy["outputs"]["train_oof_predictions"]
    )
    no_clone_prediction_path = resolve(
        policy["outputs"]["no_clone_oof_predictions"]
    )
    parent_common.write_csv_immutable(
        train_prediction_path,
        train_prediction_rows(
            policy, train_rows, ranking, nested_results
        ),
    )
    parent_common.write_csv_immutable(
        no_clone_prediction_path,
        train_prediction_rows(
            policy,
            no_clone_rows,
            no_clone_ranking,
            no_clone_results,
        ),
    )

    blind_rows = blind_valid_prediction_rows(
        valid_pair_rows, ranking, unlabelled_valid_scores
    )
    blind_path = resolve(
        policy["outputs"]["blind_valid_predictions"]
    )
    parent_common.write_csv_immutable(blind_path, blind_rows)
    observed_blind_rows = parent_common.load_csv(blind_path)
    if observed_blind_rows != blind_rows:
        raise ValueError(
            "Step7-v4.1 blind valid prediction byte replay drift"
        )
    locked_valid_scores = replay_blind_valid_scores(
        valid_pair_rows, ranking, observed_blind_rows
    )
    for candidate_id in ranking:
        if not np.array_equal(
            locked_valid_scores[candidate_id],
            np.asarray(
                unlabelled_valid_scores[candidate_id],
                dtype=np.float64,
            ),
        ):
            raise ValueError(
                "Step7-v4.1 blind valid score numeric replay drift"
            )
    blind_lock_payload = {
        "step": "step7_v4_1_blind_valid_scoring_lock",
        "version": policy["version"],
        "train_selection_lock": parent_common.file_record(lock_path),
        "model_artifacts": parent_common.file_record(artifact_path),
        "selected_joblib_models": selected_model_records,
        "blind_valid_predictions": parent_common.file_record(blind_path),
        "candidate_ranking": ranking,
        "candidate_count": len(ranking),
        "valid_pair_count": len(valid_pair_rows),
        "blind_prediction_row_count": len(blind_rows),
        "probability_serialization": (
            "python_float_repr_exact_round_trip"
        ),
        "valid_label_values_read": False,
        "historical_test_label_values_read": False,
        "policy_sha256": sha256_file(POLICY_PATH),
        "producer_sha256": sha256_file(SCRIPT_PATH),
    }
    blind_lock_payload = json_ready(blind_lock_payload)
    blind_lock_payload["lock_content_sha256"] = canonical_hash(
        blind_lock_payload
    )
    blind_lock_path = resolve(
        policy["outputs"]["blind_scoring_lock"]
    )
    parent_common.write_json_immutable(
        blind_lock_path, blind_lock_payload
    )
    observed_blind_lock = parent_common.load_json(blind_lock_path)
    parent_common.verify_canonical_self_hash(
        observed_blind_lock,
        "lock_content_sha256",
        "Step7-v4.1 blind valid lock",
    )
    if observed_blind_lock != blind_lock_payload:
        raise ValueError(
            "Step7-v4.1 blind valid lock replay drift"
        )
    del unlabelled_valid_scores
    print(
        "[Step7-v4.1] models and blind valid scores locked; "
        "opening already-viewed valid labels for diagnostics",
        flush=True,
    )

    valid_rows = parent_selector.load_label_split(
        parent_policy, pair_rows, "valid"
    )
    if [row["pair_uid"] for row in valid_rows] != [
        row["pair_uid"] for row in valid_pair_rows
    ]:
        raise ValueError(
            "Step7-v4.1 valid label/prescore order drift"
        )
    valid_metrics = {
        candidate_id: parent_selector.full_metrics(
            valid_rows,
            locked_valid_scores[candidate_id],
            nested_results[candidate_id]["selected_threshold"],
        )
        for candidate_id in ranking
    }
    valid_prediction_rows = labelled_valid_prediction_rows(
        valid_rows, ranking, locked_valid_scores
    )
    valid_prediction_path = resolve(
        policy["outputs"]["valid_predictions"]
    )
    parent_common.write_csv_immutable(
        valid_prediction_path, valid_prediction_rows
    )

    summary = {
        "step": "step7_v4_1_style_free_classifier_audit",
        "version": policy["version"],
        "objective": policy["objective"],
        "result_scope": policy["result_scope"],
        "train_selection_lock": lock_payload,
        "train_selection_lock_file": parent_common.file_record(
            lock_path
        ),
        "blind_scoring_lock": blind_lock_payload,
        "blind_scoring_lock_file": parent_common.file_record(
            blind_lock_path
        ),
        "preflight": preflight,
        "train_only_candidate_ranking": ranking,
        "selection_decision": decision,
        "nested_training_audit": nested_audit,
        "candidate_train_results": {
            candidate_id: compact_candidate_result(
                nested_results[candidate_id]
            )
            for candidate_id in ranking
        },
        "no_exact_clone_candidate_ranking": no_clone_ranking,
        "no_exact_clone_nested_training_audit": (
            no_clone_nested_audit
        ),
        "no_exact_clone_candidate_train_results": {
            candidate_id: compact_candidate_result(
                no_clone_results[candidate_id]
            )
            for candidate_id in no_clone_ranking
        },
        "valid_loaded_after_train_selection_lock": True,
        "valid_loaded_after_blind_scoring_lock": True,
        "valid_metrics_may_change_selection": False,
        "valid_development_status": (
            "already_repeatedly_viewed_descriptive_diagnostic_only"
        ),
        "candidate_valid_development_metrics": valid_metrics,
        "historical_test_labels_read": False,
        "new_real_english_confirmation_required": True,
        "formal_m0_certified": False,
        "style_free_data_audit": data_audit,
        "pinned_input_audit": pinned_input_audit,
        "parent_source_preparation": {
            "manifest_content_sha256": preparation_manifest[
                "manifest_content_sha256"
            ],
            "pair_count": len(pair_rows),
            "fixed_style_free_feature_row_count": len(
                fixed_features
            ),
        },
        "selection_execution_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "gpu_used_for_v4_1_selection": False,
        },
        "outputs": {
            "train_selection_lock": parent_common.file_record(
                lock_path
            ),
            "model_artifacts": parent_common.file_record(
                artifact_path
            ),
            "selected_models": selected_model_records,
            "train_oof_predictions": parent_common.file_record(
                train_prediction_path
            ),
            "no_clone_oof_predictions": parent_common.file_record(
                no_clone_prediction_path
            ),
            "blind_valid_predictions": parent_common.file_record(
                blind_path
            ),
            "blind_scoring_lock": parent_common.file_record(
                blind_lock_path
            ),
            "valid_predictions": parent_common.file_record(
                valid_prediction_path
            ),
        },
        "policy_sha256": sha256_file(POLICY_PATH),
        "producer_sha256": sha256_file(SCRIPT_PATH),
    }
    summary = json_ready(summary)
    summary["summary_content_sha256"] = canonical_hash(summary)
    summary_path = resolve(policy["outputs"]["selection_summary"])
    parent_common.write_json_immutable(summary_path, summary)
    return summary


def smoke_test(policy: dict) -> dict:
    (
        parent_policy,
        _preparation_manifest,
        preparation_bundle,
        _fixed_features,
        _seller_records,
        _seller_markets,
        data_audit,
    ) = load_style_free_parent_data(policy)
    factory = data_audit["factory"]
    train_rows = parent_selector.load_label_split(
        parent_policy, preparation_bundle["pair_rows"], "train"
    )
    assignments = parent_selector.balanced_component_folds(
        train_rows, 5, 2026072491
    )
    fit_rows = [
        row for row in train_rows if assignments[row["component_id"]] != 0
    ]
    hold_rows = [
        row for row in train_rows if assignments[row["component_id"]] == 0
    ]
    e5_names = feature_blocks(parent_policy)["e5_6"]
    fit_matrix, hold_matrix, _medians, _audit = factory.design(
        fit_rows, hold_rows, e5_names
    )
    outputs = {}
    for classifier_id in CLASSIFIER_IDS:
        parameters = hyperparameter_grid(policy, classifier_id)[0]
        artifact = fit_classifier(
            policy,
            classifier_id,
            parameters,
            fit_matrix,
            fit_rows,
            seed=derived_seed("smoke", classifier_id),
            factory=factory,
            feature_names=e5_names,
        )
        first = apply_classifier(hold_matrix, artifact)
        payload = serialize_joblib(
            {"classifier_artifact": artifact}
        )
        replayed = joblib.load(io.BytesIO(payload))
        second = apply_classifier(
            hold_matrix, replayed["classifier_artifact"]
        )
        if not np.array_equal(first, second):
            raise ValueError(
                f"Step7-v4.1 {classifier_id} smoke replay drift"
            )
        outputs[classifier_id] = {
            "fit_audit": compact_fit_audit(
                artifact["fit_audit"]
            ),
            "holdout_probability_count": len(first),
            "minimum_probability": float(np.min(first)),
            "maximum_probability": float(np.max(first)),
            "joblib_size_bytes": len(payload),
            "joblib_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "status": "pass",
        "fit_row_count": len(fit_rows),
        "holdout_row_count": len(hold_rows),
        "feature_count": len(e5_names),
        "classifiers": outputs,
        "formal_outputs_written": False,
        "valid_label_values_read": False,
        "historical_test_label_values_read": False,
    }


def hardest_no_clone_subset_smoke_test(policy: dict) -> dict:
    (
        parent_policy,
        _preparation_manifest,
        preparation_bundle,
        _fixed_features,
        _seller_records,
        _seller_markets,
        data_audit,
    ) = load_style_free_parent_data(policy)
    factory = data_audit["factory"]
    train_rows = parent_selector.load_label_split(
        parent_policy, preparation_bundle["pair_rows"], "train"
    )
    overlap = parent_common.exact_overlap_audit_by_pair(
        preparation_bundle["pair_rows"],
        preparation_bundle["seller_text_rows"],
    )
    rows = [
        row
        for row in train_rows
        if not overlap[row["pair_uid"]][
            "any_exact_clean_text_overlap"
        ]
    ]
    cfg = policy["training"]
    fit_subsets = []
    for outer_seed in cfg["outer_seeds"]:
        outer_assignments = parent_selector.balanced_component_folds(
            rows,
            int(cfg["outer_fold_count"]),
            int(outer_seed),
        )
        for outer_fold in range(int(cfg["outer_fold_count"])):
            outer_fit = [
                row
                for row in rows
                if outer_assignments[row["component_id"]]
                != outer_fold
            ]
            inner_seed = derived_seed(
                outer_seed, outer_fold, "inner_component_folds"
            )
            inner_assignments = (
                parent_selector.balanced_component_folds(
                    outer_fit,
                    int(cfg["inner_fold_count"]),
                    inner_seed,
                )
            )
            for inner_fold in range(int(cfg["inner_fold_count"])):
                inner_fit = [
                    row
                    for row in outer_fit
                    if inner_assignments[row["component_id"]]
                    != inner_fold
                ]
                fit_subsets.append(
                    (
                        sum(
                            row["review_label"] == "positive"
                            for row in inner_fit
                        ),
                        len(inner_fit),
                        int(outer_seed),
                        outer_fold,
                        inner_fold,
                        inner_fit,
                    )
                )
    (
        positive_count,
        row_count,
        outer_seed,
        outer_fold,
        inner_fold,
        hardest_rows,
    ) = min(
        fit_subsets,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
            item[4],
        ),
    )
    largest_spec = next(
        spec
        for spec in feature_set_specs(policy, parent_policy)
        if spec["id"] == "legacy18_e5_labse"
    )
    fit_matrix, _empty, medians, reference_audit = factory.design(
        hardest_rows, [], largest_spec["feature_names"]
    )
    classifier_audits = {}
    for classifier_id in CLASSIFIER_IDS:
        records = []
        for parameter_index, parameters in enumerate(
            hyperparameter_grid(policy, classifier_id)
        ):
            artifact = fit_classifier(
                policy,
                classifier_id,
                parameters,
                fit_matrix,
                hardest_rows,
                seed=derived_seed(
                    "hardest_no_clone",
                    classifier_id,
                    parameter_index,
                ),
                factory=factory,
                feature_names=largest_spec["feature_names"],
            )
            probabilities = apply_classifier(fit_matrix, artifact)
            records.append(
                {
                    "parameters": parameters,
                    "fit_audit": compact_fit_audit(
                        artifact["fit_audit"]
                    ),
                    "probability_minimum": float(
                        np.min(probabilities)
                    ),
                    "probability_maximum": float(
                        np.max(probabilities)
                    ),
                }
            )
        classifier_audits[classifier_id] = records
    return {
        "status": "pass",
        "scope": (
            "all_preregistered_hyperparameters_on_the_actual_"
            "no_clone_inner_fit_with_fewest_positives"
        ),
        "row_count": row_count,
        "positive_count": positive_count,
        "negative_count": row_count - positive_count,
        "component_count": len(
            {row["component_id"] for row in hardest_rows}
        ),
        "outer_seed": outer_seed,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "feature_set_id": largest_spec["id"],
        "feature_count": len(largest_spec["feature_names"]),
        "imputation_median_count": len(medians),
        "imputation_medians_sha256": canonical_hash(medians),
        "reference_audit": reference_audit,
        "classifier_audits": classifier_audits,
        "formal_outputs_written": False,
        "valid_label_values_read": False,
        "historical_test_label_values_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate frozen policy, dependency, implementation, and input pins.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fit one fold for each classifier without writing formal outputs.",
    )
    parser.add_argument(
        "--hardest-no-clone-smoke-test",
        action="store_true",
        help=(
            "Fit every fixed hyperparameter on the actual no-clone inner "
            "training subset with the fewest positives, without outputs."
        ),
    )
    args = parser.parse_args()
    policy = load_policy()
    if args.validate_config_only:
        pinned = verify_pinned_inputs(policy)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "candidate_count": len(
                        candidate_specs(
                            policy,
                            parent_common.load_json(
                                resolve(
                                    policy["parent_contract"][
                                        "policy_path"
                                    ]
                                )
                            ),
                        )
                    ),
                    "classifier_ids": list(CLASSIFIER_IDS),
                    "feature_set_ids": list(FEATURE_SET_IDS),
                    "style_feature_count": 0,
                    "pinned_input_count": len(pinned),
                    "numerical_execution_performed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.smoke_test:
        print(
            json.dumps(
                smoke_test(policy), ensure_ascii=False, indent=2
            )
        )
        return
    if args.hardest_no_clone_smoke_test:
        print(
            json.dumps(
                hardest_no_clone_subset_smoke_test(policy),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = run_selection(policy)
    decision = summary["selection_decision"]
    print(
        json.dumps(
            {
                "status": decision["overall_selection_status"],
                "best_current_pipeline": decision[
                    "best_current_pipeline"
                ],
                "best_transfer_candidate": decision[
                    "best_transfer_candidate"
                ],
                "best_legacy18_c0": decision[
                    "best_legacy18_c0"
                ],
                "transfer_capable_m0_status": decision[
                    "transfer_capable_m0_status"
                ],
                "formal_m0_certified": False,
                "historical_test_labels_read": False,
                "summary": relative(
                    resolve(policy["outputs"]["selection_summary"])
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
