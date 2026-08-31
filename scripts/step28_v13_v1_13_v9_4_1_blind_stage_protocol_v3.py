#!/usr/bin/env python3
"""Fail-closed stage protocol for V9.4.1 blind Audit-A/Audit-B predictions."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from typing import Any, Mapping, Sequence

import numpy as np

import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


STAGE_ORDER = (
    "V3_IMPLEMENTATION_FROZEN_NO_FORMAL_EXECUTION",
    "CURRENT_REGRESSION_COMMIT_PUSH_INDEPENDENT_REVIEW_FROZEN",
    "PUBLIC_PROJECTION_FROZEN_NO_TRUTH",
    "TRAIN_DEVELOPMENT_MODELS_AND_THRESHOLDS_FROZEN",
    "AUDIT_A_BLIND_PREDICTIONS_FROZEN_NO_TRUTH",
    "AUDIT_A_EVALUATION_FROZEN_PASSED",
    "AUDIT_B_BLIND_PREDICTIONS_FROZEN_NO_TRUTH",
    "AUDIT_B_EVALUATION_FROZEN",
)


def validate_next_stage(current: str, requested: str) -> None:
    if current not in STAGE_ORDER or requested not in STAGE_ORDER:
        raise core.ModelTrainingV3Error("Unknown V3 stage status")
    if STAGE_ORDER.index(requested) != STAGE_ORDER.index(current) + 1:
        raise core.ModelTrainingV3Error(
            f"Non-adjacent or backward V3 stage transition: {current} -> {requested}"
        )


def build_blind_prediction_payload(
    *,
    split: str,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
    row_key_sha256: str,
    training_parent_sha256: str,
) -> dict[str, Any]:
    """Build an in-memory blind payload without accepting any truth argument.

    This function does not authorize or write a formal artifact.  The later
    exact-commit wrapper must first validate and consume the split-specific
    blind-prediction capability.
    """

    if split not in ("audit_a", "audit_b"):
        raise core.ModelTrainingV3Error("Blind prediction split drift")
    if set(predictions) != set(core.MODEL_IDS) or set(thresholds) != set(core.MODEL_IDS):
        raise core.ModelTrainingV3Error("Blind model registry drift")
    if len(row_key_sha256) != 64 or len(training_parent_sha256) != 64:
        raise core.ModelTrainingV3Error("Blind parent digest format drift")
    models = {}
    row_count = None
    for model_id in core.MODEL_IDS:
        probability = core._float64_vector(predictions[model_id], label=model_id)
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise core.ModelTrainingV3Error(f"Blind probability range drift: {model_id}")
        if row_count is None:
            row_count = len(probability)
        elif len(probability) != row_count:
            raise core.ModelTrainingV3Error("Blind prediction row-count drift")
        threshold = float(thresholds[model_id])
        if np.isnan(threshold):
            raise core.ModelTrainingV3Error(f"Blind threshold is NaN: {model_id}")
        predicted = np.ascontiguousarray(probability >= threshold, dtype=np.uint8)
        models[model_id] = {
            "row_count": len(probability),
            "threshold": threshold,
            "probability_value_sha256": hashlib.sha256(
                probability.tobytes(order="C")
            ).hexdigest(),
            "predicted_class_value_sha256": hashlib.sha256(
                predicted.tobytes(order="C")
            ).hexdigest(),
        }
    payload = {
        "status": "BLIND_NUMERICAL_PAYLOAD_READY_NO_FORMAL_WRITE_AUTHORIZATION",
        "split": split,
        "model_order": list(core.MODEL_IDS),
        "row_count": row_count,
        "row_key_sha256": row_key_sha256,
        "training_parent_sha256": training_parent_sha256,
        "models": models,
        "labels_qrels_membership_or_controllers_read": False,
        "audit_truth_read": False,
        "formal_artifact_written": False,
    }
    payload["canonical_self_hash"] = core.canonical_sha256(payload)
    return payload


def validate_truth_free_signature() -> None:
    forbidden = {"labels", "truth", "qrels", "relevance", "controller", "membership"}
    parameters = set(inspect.signature(build_blind_prediction_payload).parameters)
    if parameters & forbidden:
        raise core.ModelTrainingV3Error("Blind builder accepts a truth-bearing argument")


def require_formal_blind_prediction_authorization() -> None:
    raise core.ModelTrainingV3Error(
        "The V3 blind numerical protocol has no formal write capability.  A later "
        "exact-commit, split-specific one-time authorization wrapper is required."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-contract", "formal-blind-predict"))
    args = parser.parse_args()
    policy = core.load_policy()
    validate_truth_free_signature()
    if args.command == "formal-blind-predict":
        require_formal_blind_prediction_authorization()
    print(
        json.dumps(
            {
                "status": "PASSED_BLIND_STAGE_PROTOCOL_NO_FORMAL_EXECUTION",
                "policy_canonical_self_hash": policy["canonical_self_hash"],
                "audit_truth_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
