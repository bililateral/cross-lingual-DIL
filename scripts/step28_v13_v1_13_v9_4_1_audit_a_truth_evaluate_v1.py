#!/usr/bin/env python3
"""Evaluate frozen V9.4.1 Audit-A predictions against Audit-A truth once."""

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

import step28_v13_v1_13_v9_4_1_confirmatory_evaluator_v3 as evaluator
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core
import step28_v13_v1_13_v9_4_1_train_development_v2 as training


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT
    / "schema"
    / "step28_v13_v1_13_v9_4_1_audit_a_truth_evaluation_policy_v1.json"
)


class AuditATruthEvaluationError(ValueError):
    """Raised when the Audit-A truth-evaluation boundary is violated."""


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
    if path.resolve() != DEFAULT_POLICY.resolve():
        raise AuditATruthEvaluationError("Only the Audit-A V1 policy path is valid")
    policy = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(policy.get("canonical_self_hash", ""))
    unsigned = dict(policy)
    unsigned.pop("canonical_self_hash", None)
    if len(claimed) != 64 or _canonical_sha256(unsigned) != claimed:
        raise AuditATruthEvaluationError("Audit-A evaluation policy self-hash drift")
    if (
        policy.get("version")
        != "step28-v13-v1.13-v9.4.1-audit-a-truth-evaluation-v1"
        or policy.get("status")
        != "AUDIT_A_TRUTH_EVALUATION_AUTHORIZED_AUDIT_B_SEALED"
        or policy.get("split") != "audit_a"
    ):
        raise AuditATruthEvaluationError("Audit-A evaluation policy identity drift")
    if set(policy.get("authorized_private_inputs", {})) != {
        "audit_a_labels",
        "audit_a_qrels",
    }:
        raise AuditATruthEvaluationError("Audit-A private input allow-list drift")
    if policy.get("truth_read_budget") != {
        "audit_a_labels_semantic_reads": 1,
        "audit_a_qrels_semantic_reads": 1,
        "audit_b_labels_or_qrels_semantic_reads": 0,
    }:
        raise AuditATruthEvaluationError("Audit truth-read budget drift")
    if policy.get("authorization") != {
        "audit_a_truth_evaluation_authorized": True,
        "model_or_threshold_update_authorized": False,
        "audit_b_blind_prediction_authorized": False,
        "audit_b_truth_authorized": False,
    }:
        raise AuditATruthEvaluationError("Audit authorization boundary drift")
    private_paths = [
        str(item["path"])
        for item in policy["authorized_private_inputs"].values()
    ]
    if any(not item.startswith("audit_a/") for item in private_paths):
        raise AuditATruthEvaluationError("Audit-B or non-Audit-A private input exposed")
    if policy.get("expected_layout") != {
        "worlds": 500,
        "rows": 189000,
        "rows_per_world": 378,
        "positive_rows_per_world": 20,
        "model_order": list(core.MODEL_IDS),
    }:
        raise AuditATruthEvaluationError("Audit-A formal layout drift")
    if policy.get("bootstrap") != {
        "replicates": 9999,
        "world_count": 500,
        "index_bytes_sha256": (
            "617be9200ad55b45eda8b1800989d7e0b50579bb53ecee675713f8ba2cd4c3e4"
        ),
    }:
        raise AuditATruthEvaluationError("Audit-A bootstrap contract drift")
    if (
        policy.get("blind_output_root")
        != "reports/step28_model_experiment/"
        "v9_4_1_audit_a_blind_predictions_v1_20260901"
        or policy.get("training_output_root")
        != "reports/step28_model_experiment/"
        "v9_4_1_train_development_v2_20260901"
        or policy.get("formal_output_root")
        != "reports/step28_model_experiment/"
        "v9_4_1_audit_a_truth_evaluation_v1_20260901"
        or policy.get("overwrite_existing_output_allowed") is not False
    ):
        raise AuditATruthEvaluationError("Audit-A output lineage drift")
    return policy


def _manifest_record(
    manifest: Mapping[str, Any], relative_path: str
) -> Mapping[str, Any]:
    matches = [item for item in manifest["files"] if item["path"] == relative_path]
    if len(matches) != 1:
        raise AuditATruthEvaluationError(
            f"Blind manifest does not contain one {relative_path} record"
        )
    return matches[0]


def _load_json_with_manifest(
    blind_root: Path,
    manifest: Mapping[str, Any],
    relative_path: str,
) -> dict[str, Any]:
    path = training._verify_file_record(
        blind_root,
        _manifest_record(manifest, relative_path),
        f"blind artifact {relative_path}",
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_blind_files(
    policy: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    blind_root = ROOT / str(policy["blind_output_root"])
    expected = {
        "blind_numerical_payload.json",
        "blind_prediction_summary.json",
        "thresholds.json",
        *(f"predictions/{model_id}.npy" for model_id in core.MODEL_IDS),
    }
    recorded = {str(item["path"]) for item in manifest["files"]}
    actual = {
        path.relative_to(blind_root).as_posix()
        for path in blind_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if recorded != expected or actual != expected:
        raise AuditATruthEvaluationError("Blind output file registry drift")
    for item in manifest["files"]:
        training._verify_file_record(blind_root, item, f"blind artifact {item['path']}")


def _load_frozen_public_inputs(
    policy: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, float],
]:
    for label, spec in policy["frozen_code_inputs"].items():
        training._verify_file_record(ROOT, spec, label)
    v3_spec = policy["frozen_public_inputs"]["v3_training_policy"]
    training._verify_file_record(ROOT, v3_spec, "V3 training policy")
    v3_policy = core.load_policy()
    if v3_policy.get("canonical_self_hash") != v3_spec["canonical_self_hash"]:
        raise AuditATruthEvaluationError("V3 training policy binding drift")
    blind_spec = policy["frozen_public_inputs"]["audit_a_blind_manifest"]
    blind_path = training._verify_file_record(ROOT, blind_spec, "Audit-A blind manifest")
    blind_manifest = training._load_json_with_self_hash(
        blind_path,
        str(blind_spec["canonical_self_hash"]),
        "Audit-A blind manifest",
    )
    training_spec = policy["frozen_public_inputs"]["training_manifest"]
    training_path = training._verify_file_record(
        ROOT, training_spec, "training manifest"
    )
    training_manifest = training._load_json_with_self_hash(
        training_path,
        str(training_spec["canonical_self_hash"]),
        "training manifest",
    )
    layout = policy["expected_layout"]
    if (
        blind_manifest.get("status")
        != "AUDIT_A_BLIND_PREDICTIONS_FROZEN_NO_TRUTH_READ"
        or blind_manifest.get("split") != "audit_a"
        or blind_manifest.get("row_count") != int(layout["rows"])
        or blind_manifest.get("model_order") != list(layout["model_order"])
        or blind_manifest.get("truth_reads") != {"audit_a": 0, "audit_b": 0}
        or blind_manifest.get("training_manifest_canonical_self_hash")
        != training_manifest.get("canonical_self_hash")
    ):
        raise AuditATruthEvaluationError("Blind manifest is not Audit-A truth eligible")
    if (
        training_manifest.get("status")
        != "TRAINING_AND_DEVELOPMENT_EVALUATION_COMPLETE_AUDIT_TRUTH_SEALED"
        or training_manifest.get("development_gate_status")
        != "PASSED_DEVELOPMENT_M1_M0_EQUIVALENCE_GATE"
        or training_manifest.get("audit_truth_reads") != {"audit_a": 0, "audit_b": 0}
    ):
        raise AuditATruthEvaluationError("Training parent is not Audit-A eligible")
    _validate_blind_files(policy, blind_manifest)
    blind_root = ROOT / str(policy["blind_output_root"])
    summary = _load_json_with_manifest(
        blind_root, blind_manifest, "blind_prediction_summary.json"
    )
    numerical = _load_json_with_manifest(
        blind_root, blind_manifest, "blind_numerical_payload.json"
    )
    thresholds_raw = _load_json_with_manifest(
        blind_root, blind_manifest, "thresholds.json"
    )
    if set(thresholds_raw) != set(core.MODEL_IDS):
        raise AuditATruthEvaluationError("Frozen threshold registry drift")
    thresholds = {model_id: float(thresholds_raw[model_id]) for model_id in core.MODEL_IDS}
    if any(math.isnan(value) for value in thresholds.values()):
        raise AuditATruthEvaluationError("Frozen threshold is NaN")
    training_root = ROOT / str(policy["training_output_root"])
    development_threshold_path = training._verify_file_record(
        training_root,
        _manifest_record(training_manifest, "development_thresholds.json"),
        "frozen development thresholds",
    )
    development_thresholds_raw = json.loads(
        development_threshold_path.read_text(encoding="utf-8")
    )
    if set(development_thresholds_raw) != set(core.MODEL_IDS):
        raise AuditATruthEvaluationError("Development threshold registry drift")
    development_thresholds = {
        model_id: float(development_thresholds_raw[model_id])
        for model_id in core.MODEL_IDS
    }
    if thresholds != development_thresholds:
        raise AuditATruthEvaluationError(
            "Blind thresholds differ from frozen development thresholds"
        )
    row_spec = policy["frozen_public_inputs"]["audit_a_row_keys"]
    row_path = training._verify_file_record(ROOT, row_spec, "Audit-A row keys")
    if (
        summary.get("status") != "AUDIT_A_BLIND_PREDICTIONS_FROZEN_NO_TRUTH_READ"
        or summary.get("row_keys_sha256") != row_spec["sha256"]
        or summary.get("row_keys_path") != row_spec["path"]
        or summary.get("audit_a_labels_or_qrels_reads") != 0
        or summary.get("audit_b_labels_or_qrels_reads") != 0
        or summary.get("model_parameters_updated") is not False
        or summary.get("thresholds_updated") is not False
        or summary.get("audit_b_prediction_created") is not False
    ):
        raise AuditATruthEvaluationError("Blind summary boundary drift")
    if (
        numerical.get("split") != "audit_a"
        or numerical.get("row_count") != int(layout["rows"])
        or numerical.get("row_key_sha256") != row_spec["sha256"]
        or numerical.get("model_order") != list(core.MODEL_IDS)
        or numerical.get("audit_truth_read") is not False
        or numerical.get("labels_qrels_membership_or_controllers_read") is not False
    ):
        raise AuditATruthEvaluationError("Blind numerical payload boundary drift")
    numerical_unsigned = dict(numerical)
    numerical_claimed = numerical_unsigned.pop("canonical_self_hash", None)
    if numerical_claimed != core.canonical_sha256(numerical_unsigned):
        raise AuditATruthEvaluationError("Blind numerical payload self-hash drift")
    rows = training._read_row_keys(
        row_path,
        "audit_a",
        expected_rows=int(layout["rows"]),
        expected_worlds=int(layout["worlds"]),
        expected_rows_per_world=int(layout["rows_per_world"]),
    )
    predictions: dict[str, np.ndarray] = {}
    for model_id in core.MODEL_IDS:
        path = training._verify_file_record(
            blind_root,
            _manifest_record(blind_manifest, f"predictions/{model_id}.npy"),
            f"Audit-A prediction {model_id}",
        )
        probability = np.load(path, allow_pickle=False)
        if (
            probability.shape != (int(layout["rows"]),)
            or probability.dtype.str != "<f8"
            or not probability.flags.c_contiguous
            or not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
        ):
            raise AuditATruthEvaluationError(f"Audit-A probability drift: {model_id}")
        model_payload = numerical["models"].get(model_id, {})
        value_hash = hashlib.sha256(probability.tobytes(order="C")).hexdigest()
        predicted = np.ascontiguousarray(
            probability >= thresholds[model_id], dtype=np.uint8
        )
        class_hash = hashlib.sha256(predicted.tobytes(order="C")).hexdigest()
        if (
            model_payload.get("row_count") != len(probability)
            or float(model_payload.get("threshold", math.nan)) != thresholds[model_id]
            or model_payload.get("probability_value_sha256") != value_hash
            or model_payload.get("predicted_class_value_sha256") != class_hash
        ):
            raise AuditATruthEvaluationError(
                f"Blind numerical binding drift: {model_id}"
            )
        predictions[model_id] = probability
    return v3_policy, blind_manifest, rows, predictions, thresholds


def validate_contract() -> dict[str, Any]:
    policy = load_policy()
    v3_policy, blind_manifest, rows, predictions, thresholds = (
        _load_frozen_public_inputs(policy)
    )
    private_root = ROOT / str(policy["private_supervision_root"])
    for label, spec in policy["authorized_private_inputs"].items():
        path = private_root / str(spec["path"])
        if not path.is_file() or path.stat().st_size != int(spec["size_bytes"]):
            raise AuditATruthEvaluationError(
                f"Authorized private input metadata drift: {label}"
            )
    bootstrap = core.build_bootstrap_indices(v3_policy, "audit_a")
    if bootstrap.shape != (
        int(policy["bootstrap"]["replicates"]),
        int(policy["bootstrap"]["world_count"]),
    ):
        raise AuditATruthEvaluationError("Audit-A bootstrap shape drift")
    bootstrap_hash = hashlib.sha256(bootstrap.tobytes(order="C")).hexdigest()
    if bootstrap_hash != policy["bootstrap"]["index_bytes_sha256"]:
        raise AuditATruthEvaluationError("Audit-A bootstrap index drift")
    return {
        "status": "PASSED_AUDIT_A_TRUTH_EVALUATION_CONTRACT_NO_TRUTH_READ",
        "policy_canonical_self_hash": policy["canonical_self_hash"],
        "blind_manifest_canonical_self_hash": blind_manifest["canonical_self_hash"],
        "v3_policy_canonical_self_hash": v3_policy["canonical_self_hash"],
        "row_count": len(rows["pair_uids"]),
        "model_count": len(predictions),
        "threshold_count": len(thresholds),
        "bootstrap_index_sha256": bootstrap_hash,
        "audit_a_truth_reads": 0,
        "audit_b_truth_reads": 0,
        "formal_evaluation_performed": False,
    }


def run_evaluation() -> dict[str, Any]:
    policy = load_policy()
    v3_policy, blind_manifest, rows, predictions, thresholds = (
        _load_frozen_public_inputs(policy)
    )
    output_root = ROOT / str(policy["formal_output_root"])
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists():
        raise AuditATruthEvaluationError("Formal Audit-A evaluation output already exists")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    truth_reads = {key: 0 for key in policy["truth_read_budget"]}
    try:
        print("[1/4] 已核验冻结预测、阈值、行键和重抽样索引", flush=True)
        private_root = ROOT / str(policy["private_supervision_root"])
        private_paths = {
            label: training._verify_file_record(private_root, spec, label)
            for label, spec in policy["authorized_private_inputs"].items()
        }
        layout = policy["expected_layout"]
        print("[2/4] 一次性读取并逐行对齐审核甲标签和检索真值", flush=True)
        labels = training._read_labels(
            private_paths["audit_a_labels"],
            rows,
            expected_rows_per_world=int(layout["rows_per_world"]),
            expected_positive_per_world=int(layout["positive_rows_per_world"]),
        )
        truth_reads["audit_a_labels_semantic_reads"] = 1
        relevance = training._read_qrels_relevance(
            private_paths["audit_a_qrels"], rows
        )
        truth_reads["audit_a_qrels_semantic_reads"] = 1
        if not np.array_equal(labels, relevance):
            raise AuditATruthEvaluationError("Audit-A labels and qrels disagree")
        bootstrap = core.build_bootstrap_indices(v3_policy, "audit_a")
        if bootstrap.shape != (
            int(policy["bootstrap"]["replicates"]),
            int(policy["bootstrap"]["world_count"]),
        ):
            raise AuditATruthEvaluationError("Audit-A bootstrap shape drift")
        bootstrap_hash = hashlib.sha256(bootstrap.tobytes(order="C")).hexdigest()
        if bootstrap_hash != policy["bootstrap"]["index_bytes_sha256"]:
            raise AuditATruthEvaluationError("Audit-A bootstrap index drift")
        print("[3/4] 计算完整分类、检索、配对比较和 9,999 次世界重抽样", flush=True)
        evaluation = evaluator.evaluate_split_from_raw_inputs(
            policy=v3_policy,
            split="audit_a",
            predictions=predictions,
            thresholds=thresholds,
            frozen_development_thresholds=thresholds,
            world_ordinals=rows["world_ordinals"],
            seller_uid_left=rows["seller_uid_left"],
            seller_uid_right=rows["seller_uid_right"],
            labels=labels,
            retrieval_relevance=relevance,
            actual_bootstrap_indices=bootstrap,
        )
        if truth_reads != policy["truth_read_budget"]:
            raise AuditATruthEvaluationError("Truth-read budget was not followed exactly")
        passed = bool(evaluation["gate"].get("numerical_gate_passed"))
        status = (
            "AUDIT_A_TRUTH_EVALUATION_COMPLETE_NUMERICAL_GATE_PASSED_AUDIT_B_SEALED"
            if passed
            else "AUDIT_A_TRUTH_EVALUATION_COMPLETE_NUMERICAL_GATE_FAILED_AUDIT_B_CLOSED"
        )
        print("[4/4] 保存正式审核甲结果和完整性清单", flush=True)
        _write_json(building / "audit_a_evaluation.json", evaluation)
        summary = {
            "status": status,
            "runner_sha256": training._sha256_file(Path(__file__)),
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "blind_manifest_canonical_self_hash": blind_manifest[
                "canonical_self_hash"
            ],
            "v3_policy_canonical_self_hash": v3_policy["canonical_self_hash"],
            "split": "audit_a",
            "row_count": len(labels),
            "model_order": list(core.MODEL_IDS),
            "label_value_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "qrels_relevance_value_sha256": hashlib.sha256(
                relevance.tobytes()
            ).hexdigest(),
            "bootstrap_index_sha256": bootstrap_hash,
            "truth_read_counts": truth_reads,
            "model_parameters_updated": False,
            "thresholds_updated": False,
            "audit_a_numerical_gate_passed": passed,
            "future_audit_b_blind_prediction_may_be_requested": passed,
            "audit_b_blind_prediction_authorized_by_this_result": False,
            "audit_b_truth_authorized_by_this_result": False,
            "audit_b_predictions_created": False,
            "audit_b_truth_reads": 0,
        }
        _write_json(building / "audit_a_evaluation_summary.json", summary)
        files = [
            training._file_record(path, building)
            for path in sorted(building.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
        manifest = {
            "status": status,
            "runner_sha256": training._sha256_file(Path(__file__)),
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "blind_manifest_canonical_self_hash": blind_manifest[
                "canonical_self_hash"
            ],
            "split": "audit_a",
            "row_count": len(labels),
            "model_order": list(core.MODEL_IDS),
            "truth_read_counts": truth_reads,
            "audit_b_truth_reads": 0,
            "audit_b_predictions_created": False,
            "numerical_gate_passed": passed,
            "files": files,
        }
        manifest["canonical_self_hash"] = _canonical_sha256(manifest)
        _write_json(building / "manifest.json", manifest)
        os.replace(building, output_root)
        return {
            "status": status,
            "output_root": output_root.relative_to(ROOT).as_posix(),
            "manifest_canonical_self_hash": manifest["canonical_self_hash"],
            "numerical_gate_passed": passed,
            "audit_a_truth_reads": {
                "labels": truth_reads["audit_a_labels_semantic_reads"],
                "qrels": truth_reads["audit_a_qrels_semantic_reads"],
            },
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
    result = validate_contract() if args.command == "validate-contract" else run_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
