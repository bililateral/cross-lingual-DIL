#!/usr/bin/env python3
"""Load and enforce the frozen V9.4 model-visible shortcut policy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy
import sklearn
import threadpoolctl

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_joint_noise_signatures_v9_4 as signatures_v94
import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94
import step28_v13_v1_13_quality_probe_core_v9_4 as core_v94
import step28_v13_v1_13_quality_probe_labels_v9_4 as labels_v94
import step28_v13_v1_13_quality_probe_preparer_v9_4 as preparer_v94
import step28_v13_v1_13_model_visible_source_guard_v9_4 as source_guard_v94


VERSION = "2026-08-27-step28-v13-v1-13-quality-probe-policy-v9-4"
ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_v9_4_model_visible_shortcut_policy.json"
)
POLICY_SHA256 = "ecaf745973ef0d6cc3417c299a0e01e2b1305f61290c25cba6dd193a93ca8911"
FORMAL_VIEW = "model_visible_14"
FORMAL_AP_BASELINE = 20 / 378
FORMAL_SPLIT_WORLDS = 500
FORMAL_PAIRS_PER_WORLD = 378
FORMAL_POSITIVES_PER_WORLD = 20
FORMAL_BOOTSTRAP = {
    "generator": "numpy.random.Generator(PCG64)",
    "replicates": 9999,
    "development_world_count": 500,
    "seed": 281320260810,
    "draws_dtype": "little-endian int64",
    "draws_shape": [9999, 500],
    "draws_raw_i8_c_sha256": (
        "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661"
    ),
    "streaming_batch_size": 16,
    "quantile": "0.950000000000",
    "quantile_method": "linear",
    "refit_within_replicate": False,
    "family_maximum_within_replicate": True,
}
FORMAL_GATES = {
    "maximum_single_feature_symmetric_auc": "0.520000000000",
    "maximum_family_symmetric_auc": "0.530000000000",
    "maximum_family_average_precision_uplift": "0.010000000000",
    "bootstrap_95_upper_symmetric_auc": "0.530000000000",
    "bootstrap_95_upper_average_precision_uplift": "0.015000000000",
}
PASS_CLAIM = "未检出14项已登记模型可见无关变量超过冻结阈值的捷径"


class QualityProbePolicyV94Error(core_v94.QualityProbeCoreV94Error):
    """Raised when the formal V9.4 policy or gate result drifts."""


def _exact(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return tuple(observed) == tuple(expected) and all(
            _exact(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def validate_policy_payload(policy: Mapping[str, Any]) -> None:
    if type(policy) is not dict:
        raise QualityProbePolicyV94Error("Policy root type drift")
    expected_top = (
        "version",
        "status",
        "authorization",
        "runtime_contract",
        "dataset_contract",
        "upstream_contract",
        "matrix_contract",
        "average_precision_baseline",
        "probe_models",
        "bootstrap",
        "gates",
        "pass_claim",
    )
    if tuple(policy) != expected_top:
        raise QualityProbePolicyV94Error("Policy top-level schema/order drift")
    if policy["version"] != (
        "2026-08-27-step28-v13-v1-13-v9-4-model-visible-shortcut-policy"
    ) or policy["status"] != "FROZEN_IMPLEMENTATION_POLICY_NO_RUN_NO_DATA_NO_TRAINING":
        raise QualityProbePolicyV94Error("Policy version/status drift")
    expected_authorization = {
        "prebuild_shortcut_gate": False,
        "method_root_build": False,
        "truth_unsealing": False,
        "m0_m1_m2_m3": False,
    }
    expected_runtime = {
        "python_implementation": "CPython",
        "python_version": "3.10.11",
        "numpy_version": "2.2.6",
        "scipy_version": "1.15.3",
        "scikit_learn_version": "1.7.2",
        "threadpoolctl_version": "3.6.0",
        "thread_limit": 1,
    }
    expected_dataset = {
        "train_world_count": 500,
        "development_world_count": 500,
        "audit_a_world_count": 2,
        "audit_b_world_count": 2,
        "seller_count_per_world": 28,
        "pair_count_per_world": 378,
        "positive_pair_count_per_world": 20,
        "negative_pair_count_per_world": 358,
        "train_registered_residual": [20, 20, 20],
        "development_registered_residual": [22, 22, 22],
    }
    expected_upstream = {
        "balanced_schedule_version": schedule_v94.VERSION,
        "train_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS["train"],
        "development_public_design_seed": schedule_v94.PUBLIC_DESIGN_SEEDS[
            "development"
        ],
        "balanced_schedule_maximum_iterations": schedule_v94.MAX_ITERATIONS,
        "direct_r2_plan_read": False,
        "noise_signature_version": signatures_v94.VERSION,
        "noise_signature_source_role": (
            "label_free_real_chinese_training_side"
        ),
        "noise_signature_rows_sha256": (
            signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
        ),
        "noise_signature_set_commitment_sha256": (
            signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
        ),
        "time_key_commitment_sha256": None,
    }
    expected_matrix = {
        "view": FORMAL_VIEW,
        "seller_feature_count": len(matrix_v94.SELLER_FEATURES),
        "pair_feature_count": len(matrix_v94.PAIR_FEATURES),
        "dtype": "little-endian float64",
        "row_order": "world_uid_utf8_then_canonical_pair_uid_utf8",
        "label_join": "exact_world_uid_and_canonical_pair_uid_after_matrix_freeze",
        "prebuild_source": (
            "truth_free_noise_signature_and_public_world_schedule_only"
        ),
        "time_bucket_derivation": (
            "HMAC-SHA256-first-u64-big-endian-modulo-4"
        ),
        "final_replay": (
            "actual_public_title_description_and_time_index_exact_bytes"
        ),
    }
    expected_baseline = {"numerator": 20, "denominator": 378}
    expected_models = {
        "logistic_l2": {
            "preprocessing": core_v94.LOGISTIC_PREPROCESSING,
            "standard_scaler": core_v94.STANDARD_SCALER_CONFIG,
            **core_v94.LOGISTIC_CONFIG,
        },
        "hist_gradient_boosting_depth2": {
            "class": core_v94.TREE_CLASS,
            "preprocessing": core_v94.TREE_PREPROCESSING,
            **core_v94.TREE_CONFIG,
        },
    }
    checks = (
        ("authorization", policy["authorization"], expected_authorization),
        ("runtime", policy["runtime_contract"], expected_runtime),
        ("dataset", policy["dataset_contract"], expected_dataset),
        ("upstream", policy["upstream_contract"], expected_upstream),
        ("matrix", policy["matrix_contract"], expected_matrix),
        ("baseline", policy["average_precision_baseline"], expected_baseline),
        ("models", policy["probe_models"], expected_models),
        ("bootstrap", policy["bootstrap"], FORMAL_BOOTSTRAP),
        ("gates", policy["gates"], FORMAL_GATES),
        ("pass claim", policy["pass_claim"], PASS_CLAIM),
    )
    for label, observed, expected in checks:
        if not _exact(observed, expected):
            raise QualityProbePolicyV94Error(f"Policy {label} drift")
    observed_runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "threadpoolctl_version": threadpoolctl.__version__,
        "thread_limit": 1,
    }
    if not _exact(observed_runtime, expected_runtime) or sys.byteorder != "little":
        raise QualityProbePolicyV94Error("Policy runtime drift")


def load_formal_policy(path: Path = POLICY_PATH) -> Mapping[str, Any]:
    resolved = path.resolve()
    if resolved != POLICY_PATH.resolve():
        raise QualityProbePolicyV94Error("Formal policy path drift")
    raw = resolved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise QualityProbePolicyV94Error("Formal policy byte hash drift")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityProbePolicyV94Error("Formal policy JSON drift") from error
    validate_policy_payload(payload)
    return _deep_freeze(payload)


def validate_formal_split(
    *,
    row_keys: Sequence[tuple[str, str]],
    labels: np.ndarray,
    split: str,
) -> None:
    if split not in {"train", "development"}:
        raise QualityProbePolicyV94Error("Formal split name drift")
    expected_rows = FORMAL_SPLIT_WORLDS * FORMAL_PAIRS_PER_WORLD
    if len(row_keys) != expected_rows:
        raise QualityProbePolicyV94Error(f"{split} formal row count drift")
    checked_labels = core_v94._validate_labels(labels, expected_rows, label=split)
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(value, str) or not value for value in key)
        for key in row_keys
    ):
        raise QualityProbePolicyV94Error(f"{split} formal row-key type drift")
    if tuple(row_keys) != tuple(
        sorted(
            row_keys,
            key=lambda key: (key[0].encode("utf-8"), key[1].encode("utf-8")),
        )
    ) or len(row_keys) != len(set(row_keys)):
        raise QualityProbePolicyV94Error(f"{split} formal row-key order drift")
    world_counts = Counter(key[0] for key in row_keys)
    positive_counts: Counter[str] = Counter()
    for key, value in zip(row_keys, checked_labels, strict=True):
        positive_counts[key[0]] += int(value)
    if (
        len(world_counts) != FORMAL_SPLIT_WORLDS
        or set(world_counts.values()) != {FORMAL_PAIRS_PER_WORLD}
        or set(positive_counts) != set(world_counts)
        or set(positive_counts.values()) != {FORMAL_POSITIVES_PER_WORLD}
    ):
        raise QualityProbePolicyV94Error(f"{split} formal per-world closure drift")


def _validate_upstream_material_commitments(
    *,
    noise_signature_set: signatures_v94.NoiseSignatureSet,
    time_key_hex: str,
    expected_noise_signature_rows_sha256: str,
    expected_noise_signature_set_commitment_sha256: str,
    expected_time_key_commitment_sha256: str,
) -> None:
    expected_commitments = (
        expected_noise_signature_rows_sha256,
        expected_noise_signature_set_commitment_sha256,
        expected_time_key_commitment_sha256,
    )
    if not all(_is_sha256(value) for value in expected_commitments):
        raise QualityProbePolicyV94Error("Formal upstream capability drift")
    try:
        time_key = bytes.fromhex(time_key_hex)
    except (TypeError, ValueError) as error:
        raise QualityProbePolicyV94Error(
            "Formal upstream capability drift"
        ) from error
    if (
        type(time_key_hex) is not str
        or len(time_key_hex) != 64
        or time_key_hex != time_key_hex.lower()
        or any(character not in "0123456789abcdef" for character in time_key_hex)
        or len(time_key) != 32
        or hashlib.sha256(time_key).hexdigest()
        != expected_time_key_commitment_sha256
    ):
        raise QualityProbePolicyV94Error("Formal upstream capability drift")
    signatures_v94.verify_noise_signatures(noise_signature_set)
    if (
        noise_signature_set.commitment["signature_rows_sha256"]
        != expected_noise_signature_rows_sha256
        or noise_signature_set.commitment[
            "signature_set_commitment_sha256"
        ]
        != expected_noise_signature_set_commitment_sha256
    ):
        raise QualityProbePolicyV94Error("Formal upstream capability drift")


def _validate_formal_inputs(
    *,
    train_prepared: preparer_v94.PreparedSplit,
    development_prepared: preparer_v94.PreparedSplit,
    train_labels: labels_v94.FrozenLabelSplit,
    development_labels: labels_v94.FrozenLabelSplit,
    train_schedule: schedule_v94.SplitSchedule,
    development_schedule: schedule_v94.SplitSchedule,
    schedule_pair_receipt: Mapping[str, Any],
    noise_signature_set: signatures_v94.NoiseSignatureSet,
    expected_noise_signature_rows_sha256: str,
    expected_noise_signature_set_commitment_sha256: str,
    expected_time_key_commitment_sha256: str,
) -> None:
    signatures_v94.verify_noise_signatures(noise_signature_set)
    expected_pair_receipt = schedule_v94.validate_train_development_pair(
        train_schedule,
        development_schedule,
    )
    if (
        type(schedule_pair_receipt) is not MappingProxyType
        or tuple(schedule_pair_receipt) != tuple(expected_pair_receipt)
        or dict(schedule_pair_receipt) != dict(expected_pair_receipt)
        or not all(_is_sha256(value) for value in (
            expected_noise_signature_rows_sha256,
            expected_noise_signature_set_commitment_sha256,
            expected_time_key_commitment_sha256,
        ))
        or noise_signature_set.commitment["signature_rows_sha256"]
        != expected_noise_signature_rows_sha256
        or noise_signature_set.commitment[
            "signature_set_commitment_sha256"
        ]
        != expected_noise_signature_set_commitment_sha256
    ):
        raise QualityProbePolicyV94Error("Formal upstream capability drift")
    pair_audit_commitment = schedule_pair_receipt[
        "pair_audit_commitment_sha256"
    ]
    for split, prepared, frozen_labels, schedule in (
        ("train", train_prepared, train_labels, train_schedule),
        (
            "development",
            development_prepared,
            development_labels,
            development_schedule,
        ),
    ):
        preparer_v94.verify_prepared_split(prepared)
        labels_v94.verify_frozen_labels(frozen_labels, prepared=prepared)
        schedule_v94.verify_split_schedule(schedule)
        matrix = prepared.matrix
        core_v94.verify_frozen_matrix(matrix)
        if (
            prepared.split != split
            or schedule.split != split
            or matrix.view != FORMAL_VIEW
            or matrix.column_names != matrix_v94.PAIR_FEATURES
            or prepared.commitment["world_source_sha256"]
            != schedule.commitment["public_worlds_sha256"]
            or prepared.commitment[
                "split_schedule_commitment_sha256"
            ]
            != schedule.commitment["split_schedule_commitment_sha256"]
            or prepared.commitment[
                "schedule_pair_audit_commitment_sha256"
            ]
            != pair_audit_commitment
            or prepared.commitment["noise_signatures_sha256"]
            != expected_noise_signature_rows_sha256
            or prepared.commitment[
                "noise_signature_set_commitment_sha256"
            ]
            != expected_noise_signature_set_commitment_sha256
            or prepared.commitment["time_key_commitment_sha256"]
            != expected_time_key_commitment_sha256
            or frozen_labels.commitment["truth_source_version"]
            != schedule_v94.VERSION
            or frozen_labels.commitment["truth_formula"]
            != labels_v94.TRUTH_FORMULA
            or frozen_labels.commitment["truth_read_count"] != 1
            or frozen_labels.commitment["audit_truth_read_count"] != 0
            or frozen_labels.commitment[
                "private_controller_truth_sha256"
            ]
            != schedule.commitment["private_controller_truth_sha256"]
            or matrix.values.shape
            != (FORMAL_SPLIT_WORLDS * FORMAL_PAIRS_PER_WORLD, len(matrix_v94.PAIR_FEATURES))
            or frozen_labels.row_keys != matrix.row_keys
        ):
            raise QualityProbePolicyV94Error(f"{split} formal matrix/join drift")
        validate_formal_split(
            row_keys=matrix.row_keys,
            labels=frozen_labels.values,
            split=split,
        )
    train_worlds = {key[0] for key in train_prepared.matrix.row_keys}
    development_worlds = {
        key[0] for key in development_prepared.matrix.row_keys
    }
    if train_worlds & development_worlds:
        raise QualityProbePolicyV94Error("Formal train/development world overlap")
    if (
        train_prepared.commitment["split_schedule_commitment_sha256"]
        == development_prepared.commitment[
            "split_schedule_commitment_sha256"
        ]
        or train_labels.commitment["private_controller_truth_sha256"]
        == development_labels.commitment["private_controller_truth_sha256"]
    ):
        raise QualityProbePolicyV94Error(
            "Formal train/development upstream identity drift"
        )


def _assemble_formal_inputs_after_authorization(
    *,
    train_schedule: schedule_v94.SplitSchedule,
    development_schedule: schedule_v94.SplitSchedule,
    noise_signature_set: signatures_v94.NoiseSignatureSet,
    time_key_hex: str,
    expected_noise_signature_rows_sha256: str,
    expected_noise_signature_set_commitment_sha256: str,
    expected_time_key_commitment_sha256: str,
) -> dict[str, Any]:
    """Build and validate the complete input chain without fitting a model."""

    _validate_upstream_material_commitments(
        noise_signature_set=noise_signature_set,
        time_key_hex=time_key_hex,
        expected_noise_signature_rows_sha256=(
            expected_noise_signature_rows_sha256
        ),
        expected_noise_signature_set_commitment_sha256=(
            expected_noise_signature_set_commitment_sha256
        ),
        expected_time_key_commitment_sha256=(
            expected_time_key_commitment_sha256
        ),
    )
    schedule_pair_receipt = schedule_v94.validate_train_development_pair(
        train_schedule,
        development_schedule,
    )
    noise_rows = signatures_v94.signature_dicts(noise_signature_set)
    pair_audit_commitment = schedule_pair_receipt[
        "pair_audit_commitment_sha256"
    ]
    prepared: dict[str, preparer_v94.PreparedSplit] = {}
    for split, schedule in (
        ("train", train_schedule),
        ("development", development_schedule),
    ):
        prepared[split] = preparer_v94._prepare_split(
            worlds=schedule_v94.public_world_dicts(schedule),
            noise_signatures=noise_rows,
            time_key_hex=time_key_hex,
            expected_world_count=FORMAL_SPLIT_WORLDS,
            split_schedule_commitment_sha256=schedule.commitment[
                "split_schedule_commitment_sha256"
            ],
            schedule_pair_audit_commitment_sha256=pair_audit_commitment,
            noise_signature_set_commitment_sha256=noise_signature_set.commitment[
                "signature_set_commitment_sha256"
            ],
        )
    train_labels = labels_v94._open_controller_truth_after_preparation(
        prepared=prepared["train"],
        schedule=train_schedule,
    )
    development_labels = labels_v94._open_controller_truth_after_preparation(
        prepared=prepared["development"],
        schedule=development_schedule,
    )
    _validate_formal_inputs(
        train_prepared=prepared["train"],
        development_prepared=prepared["development"],
        train_labels=train_labels,
        development_labels=development_labels,
        train_schedule=train_schedule,
        development_schedule=development_schedule,
        schedule_pair_receipt=schedule_pair_receipt,
        noise_signature_set=noise_signature_set,
        expected_noise_signature_rows_sha256=(
            expected_noise_signature_rows_sha256
        ),
        expected_noise_signature_set_commitment_sha256=(
            expected_noise_signature_set_commitment_sha256
        ),
        expected_time_key_commitment_sha256=(
            expected_time_key_commitment_sha256
        ),
    )
    return {
        "train_prepared": prepared["train"],
        "development_prepared": prepared["development"],
        "train_labels": train_labels,
        "development_labels": development_labels,
        "schedule_pair_receipt": schedule_pair_receipt,
    }


def run_authorized_formal_gate(
    *,
    train_schedule: schedule_v94.SplitSchedule,
    development_schedule: schedule_v94.SplitSchedule,
    noise_signature_set: signatures_v94.NoiseSignatureSet,
    time_key_hex: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only after a future policy revision explicitly authorizes the gate."""

    validate_policy_payload(_deep_thaw(policy))
    if policy["authorization"]["prebuild_shortcut_gate"] is not True:
        raise QualityProbePolicyV94Error("Formal prebuild shortcut gate is unauthorized")
    source_closure = source_guard_v94.audit_registered_sources()
    expected_time_key_commitment = policy["upstream_contract"][
        "time_key_commitment_sha256"
    ]
    if not _is_sha256(expected_time_key_commitment):
        raise QualityProbePolicyV94Error("Formal time key is not frozen")
    inputs = _assemble_formal_inputs_after_authorization(
        train_schedule=train_schedule,
        development_schedule=development_schedule,
        noise_signature_set=noise_signature_set,
        time_key_hex=time_key_hex,
        expected_noise_signature_rows_sha256=policy["upstream_contract"][
            "noise_signature_rows_sha256"
        ],
        expected_noise_signature_set_commitment_sha256=policy[
            "upstream_contract"
        ]["noise_signature_set_commitment_sha256"],
        expected_time_key_commitment_sha256=expected_time_key_commitment,
    )
    train_prepared = inputs["train_prepared"]
    development_prepared = inputs["development_prepared"]
    train_labels = inputs["train_labels"]
    development_labels = inputs["development_labels"]
    mutable_policy = _deep_thaw(policy)
    result = core_v94._evaluate_family(
        train={FORMAL_VIEW: train_prepared.matrix},
        development={FORMAL_VIEW: development_prepared.matrix},
        train_labels=train_labels.values,
        development_labels=development_labels.values,
        train_label_row_keys=train_labels.row_keys,
        development_label_row_keys=development_labels.row_keys,
        policy=mutable_policy,
        average_precision_baseline=FORMAL_AP_BASELINE,
        bootstrap=True,
    )
    comparison = _compare_formal_gates(result, policy)
    passed = comparison["all_gates_passed"]
    return {
        "source_closure": source_closure,
        "upstream_schedule_pair_receipt": inputs[
            "schedule_pair_receipt"
        ],
        "metrics": result,
        "decision": {
            "version": VERSION,
            "status": "PASSED" if passed else "FAILED",
            "comparisons": comparison["comparisons"],
            "claim": PASS_CLAIM if passed else None,
        },
    }


def _compare_formal_gates(
    result: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare frozen values only; never emit a scientific status or claim."""

    validate_policy_payload(_deep_thaw(policy))
    try:
        single_by_view = result[
            "single_feature_maximum_symmetric_roc_auc_by_view"
        ]
        model_results = result["model_results"]
        bootstrap = result["bootstrap"]
        observed = {
            "maximum_single_feature_symmetric_auc": single_by_view[FORMAL_VIEW],
            "maximum_family_symmetric_auc": result["maximum_symmetric_roc_auc"],
            "maximum_family_average_precision_uplift": result[
                "maximum_average_precision_uplift"
            ],
            "bootstrap_95_upper_symmetric_auc": bootstrap[
                "symmetric_auc_95_upper"
            ],
            "bootstrap_95_upper_average_precision_uplift": bootstrap[
                "average_precision_uplift_95_upper"
            ],
        }
    except (KeyError, TypeError) as error:
        raise QualityProbePolicyV94Error("Formal result schema drift") from error
    expected_models = (
        f"{FORMAL_VIEW}::logistic_l2",
        f"{FORMAL_VIEW}::hist_gradient_boosting_depth2",
    )
    bootstrap_fields = (
        "replicates",
        "world_count",
        "score_family_size",
        "draws_raw_i8_c_sha256",
        "family_max_symmetric_auc_vector_sha256",
        "family_max_average_precision_uplift_vector_sha256",
        "symmetric_auc_95_upper",
        "average_precision_uplift_95_upper",
    )
    if (
        tuple(single_by_view) != (FORMAL_VIEW,)
        or tuple(model_results) != expected_models
        or bootstrap is None
        or tuple(bootstrap) != bootstrap_fields
        or type(bootstrap["replicates"]) is not int
        or type(bootstrap["world_count"]) is not int
        or type(bootstrap["score_family_size"]) is not int
        or bootstrap["replicates"] != FORMAL_BOOTSTRAP["replicates"]
        or bootstrap["world_count"] != FORMAL_BOOTSTRAP["development_world_count"]
        or bootstrap["score_family_size"] != len(expected_models)
        or bootstrap["draws_raw_i8_c_sha256"]
        != FORMAL_BOOTSTRAP["draws_raw_i8_c_sha256"]
    ):
        raise QualityProbePolicyV94Error("Formal result family drift")
    model_metric_values: list[tuple[float, float]] = []
    for model in expected_models:
        record = model_results[model]
        if (
            type(record) is not dict
            or tuple(record)
            != ("symmetric_roc_auc", "average_precision", "score_vector_sha256")
            or type(record["symmetric_roc_auc"]) is not float
            or type(record["average_precision"]) is not float
            or not math.isfinite(record["symmetric_roc_auc"])
            or not math.isfinite(record["average_precision"])
            or not 0.5 <= record["symmetric_roc_auc"] <= 1.0
            or not 0.0 <= record["average_precision"] <= 1.0
            or not _is_sha256(record["score_vector_sha256"])
        ):
            raise QualityProbePolicyV94Error("Formal model result drift")
        model_metric_values.append(
            (record["symmetric_roc_auc"], record["average_precision"])
        )
    recomputed_auc = max(value[0] for value in model_metric_values)
    recomputed_ap_uplift = max(
        value[1] - FORMAL_AP_BASELINE for value in model_metric_values
    )
    if (
        result["maximum_symmetric_roc_auc"] != recomputed_auc
        or result["maximum_average_precision_uplift"] != recomputed_ap_uplift
        or not _is_sha256(bootstrap["family_max_symmetric_auc_vector_sha256"])
        or not _is_sha256(
            bootstrap["family_max_average_precision_uplift_vector_sha256"]
        )
    ):
        raise QualityProbePolicyV94Error("Formal result aggregate drift")
    thresholds = {name: float(policy["gates"][name]) for name in FORMAL_GATES}
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in observed.values()
    ):
        raise QualityProbePolicyV94Error("Formal result value drift")
    comparisons = {
        name: {
            "observed": observed[name],
            "maximum_allowed": thresholds[name],
            "passed": observed[name] <= thresholds[name],
        }
        for name in FORMAL_GATES
    }
    passed = all(value["passed"] for value in comparisons.values())
    return {
        "comparisons": comparisons,
        "all_gates_passed": passed,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
