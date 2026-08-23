#!/usr/bin/env python3
"""Receipt-authorized execution adapter for the frozen V9.1 quality design.

The frozen scientific policy deliberately keeps every production capability
closed and does not contain a post-build root pin.  This adapter keeps that
policy byte-for-byte unchanged.  It receives the exact root and the consumed
one-shot capability as a separate immutable execution context, while reusing
the frozen runner's loading/structure path and the frozen validator's numeric
helpers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import step28_v13_v1_13_quality_audit_runner_v9 as frozen_runner
import step28_v13_v1_13_quality_channel_policy_v9 as channel_policy
import step28_v13_v1_13_quality_probe_validator_v9 as frozen_validator


ROOT = Path(__file__).resolve().parents[1]
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
ROOT_MANIFEST_PIN_PATH = "root_manifest.json"
PENDING_RECEIPT_RELATIVE_PATH = (
    "private_custody/"
    "step28_v13_v1_13_v9_1_quality_audit_authorization.json"
)
EXPECTED_ATTEMPT_INDEX = 1
EXPECTED_RECEIPT_STATUS = "ALLOW_ONE_V9_1_DESIGN_QUALITY_AUDIT"
EXPECTED_CLAIM_BOUNDARY = (
    "ONE_V9_1_DESIGN_QUALITY_AUDIT_ONLY_NO_FORMAL_DATA_"
    "NO_AUDIT_AB_TRUTH_NO_TRAINING_NO_MODEL_METRICS"
)
EXPECTED_REVIEW_FINAL_LINE = "允许运行一次V9.1冻结质量审计"
EXPECTED_ROOT_MANIFEST_CANONICAL_SELF_HASH = (
    "f10086faa5f68b08a4d25a6e49943fb18ede0858ca50bad711d7bb2f4d94200f"
)
EXPECTED_EQUIVALENCE_CANONICAL_SELF_HASH = (
    "b5a19ae5c2f5c3694368ad1eacb01e559a6819727a48b2f64ca27910ae30c92b"
)
EXPECTED_EQUIVALENCE_PATHS = (
    "/full_profile_sha256",
    "/masked_profile_sha256",
    "/neutral_profile_sha256",
    "/neutral_receipt/neutral_profile_sha256",
)
EXPECTED_CAPABILITIES = {
    "quality_audit_run": True,
    "quality_metric_generation": True,
    "formal_seed": False,
    "formal_500_by_4": False,
    "audit_a_b_truth_open": False,
    "model_training": False,
    "model_metric_generation": False,
}

preparer = frozen_validator.preparer
truth_capability = frozen_validator.truth_capability
np = frozen_validator.np
average_precision_score = frozen_validator.average_precision_score
threadpool_limits = frozen_validator.threadpool_limits


class QualityAuditExecutionAdapterError(RuntimeError):
    """The consumed execution context is not the exact authorized context."""


@dataclass(frozen=True)
class ConsumedQualityAuditExecution:
    """Non-secret proof that one exact quality-audit receipt was consumed."""

    receipt_id: str
    overlay_policy_canonical_self_hash: str
    base_policy_canonical_self_hash: str
    capabilities_canonical_sha256: str
    pending_receipt_path: Path
    consumed_receipt_path: Path
    consumed_receipt_size_bytes: int
    consumed_receipt_sha256: str
    result_path: Path
    dataset_root: Path
    root_manifest_repository_path: str
    root_manifest_size_bytes: int
    root_manifest_sha256: str
    root_manifest_canonical_self_hash: str

    def root_pin(self) -> truth_capability.RootManifestPin:
        return truth_capability.RootManifestPin(
            path=ROOT_MANIFEST_PIN_PATH,
            size_bytes=self.root_manifest_size_bytes,
            sha256=self.root_manifest_sha256,
            canonical_self_hash=self.root_manifest_canonical_self_hash,
        )

    def root_binding(self) -> dict[str, Any]:
        return {
            "path": self.root_manifest_repository_path,
            "size_bytes": self.root_manifest_size_bytes,
            "sha256": self.root_manifest_sha256,
            "canonical_self_hash": self.root_manifest_canonical_self_hash,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def capabilities_canonical_sha256(capabilities: Mapping[str, Any]) -> str:
    if (
        not isinstance(capabilities, dict)
        or set(capabilities) != set(EXPECTED_CAPABILITIES)
        or any(type(value) is not bool for value in capabilities.values())
        or dict(capabilities) != EXPECTED_CAPABILITIES
    ):
        raise QualityAuditExecutionAdapterError(
            "Quality-audit execution capability widened"
        )
    return hashlib.sha256(_canonical_json_bytes(dict(capabilities))).hexdigest()


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt root is not an object"
        )
    return value


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise QualityAuditExecutionAdapterError(
            "Quality-audit source escaped the repository"
        ) from exc
    if not resolved.is_file():
        raise QualityAuditExecutionAdapterError(
            "Quality-audit source is unavailable"
        )
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": frozen_runner._sha256_file(resolved),
    }


def _git_identity() -> tuple[str, str]:
    """Recheck the reviewed commit and tree on every adapter entry."""

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if status.stdout:
            raise QualityAuditExecutionAdapterError(
                "Git worktree is not clean for reviewed quality audit"
            )
        values: list[str] = []
        for revision in ("HEAD", "HEAD^{tree}"):
            result = subprocess.run(
                ["git", "rev-parse", revision],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            values.append(result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualityAuditExecutionAdapterError(
            "Reviewed Git identity cannot be verified by execution adapter"
        ) from exc
    if any(GIT_OBJECT_RE.fullmatch(value) is None for value in values):
        raise QualityAuditExecutionAdapterError(
            "Reviewed Git identity is malformed for execution adapter"
        )
    return values[0], values[1]


def _validate_complete_receipt_binding(
    *,
    payload: Mapping[str, Any],
    execution: ConsumedQualityAuditExecution,
) -> None:
    """Reuse the public entry's exact schema and all-file binding validator.

    The import is deliberately local: the public entry imports this adapter,
    while the adapter needs the entry's single source of truth only after both
    modules have finished loading.  This closes direct-call drift without
    duplicating the receipt schema in two production modules.
    """

    try:
        import step28_v13_v1_13_run_quality_audit_once_v9_1 as authorization_entry

        overlay_policy = authorization_entry._load_overlay_policy()
        static = authorization_entry._validate_static_inputs(overlay_policy)
        git_commit, git_tree = _git_identity()
        receipt_id = authorization_entry._validate_receipt_payload(
            payload=payload,
            policy=overlay_policy,
            static=static,
            git_commit=git_commit,
            git_tree=git_tree,
        )
    except Exception as exc:
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt complete binding drift"
        ) from exc
    if (
        receipt_id != execution.receipt_id
        or overlay_policy.get("canonical_self_hash")
        != execution.overlay_policy_canonical_self_hash
        or static["quality_policy"].get("canonical_self_hash")
        != execution.base_policy_canonical_self_hash
    ):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt complete binding drift"
        )


def _validate_consumed_receipt_evidence(
    execution: ConsumedQualityAuditExecution,
    *,
    base_policy: Mapping[str, Any],
) -> None:
    expected_pending = (ROOT / PENDING_RECEIPT_RELATIVE_PATH).resolve()
    expected_consumed = expected_pending.with_name(
        f"{expected_pending.stem}.consumed."
        f"{execution.consumed_receipt_sha256}.json"
    )
    if (
        execution.pending_receipt_path != expected_pending
        or execution.consumed_receipt_path != expected_consumed
        or execution.pending_receipt_path.exists()
        or not execution.consumed_receipt_path.is_file()
        or execution.consumed_receipt_path.stat().st_size
        != execution.consumed_receipt_size_bytes
        or frozen_runner._sha256_file(execution.consumed_receipt_path)
        != execution.consumed_receipt_sha256
    ):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt file binding drift"
        )
    try:
        raw = execution.consumed_receipt_path.read_bytes()
    except OSError as exc:
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt cannot be read"
        ) from exc
    payload = _strict_json_object(raw)
    _validate_complete_receipt_binding(payload=payload, execution=execution)
    unsigned = dict(payload)
    receipt_id = unsigned.pop("canonical_self_hash", None)
    if (
        receipt_id != execution.receipt_id
        or hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        != execution.receipt_id
        or payload.get("status") != EXPECTED_RECEIPT_STATUS
        or payload.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
        or payload.get("review_final_line") != EXPECTED_REVIEW_FINAL_LINE
        or payload.get("attempt_index") != EXPECTED_ATTEMPT_INDEX
        or payload.get("capabilities") != EXPECTED_CAPABILITIES
        or payload.get("input_design_root")
        != execution.dataset_root.relative_to(ROOT).as_posix()
        or payload.get("result_path")
        != execution.result_path.relative_to(ROOT).as_posix()
        or payload.get("root_manifest") != execution.root_binding()
        or not isinstance(payload.get("overlay_policy"), Mapping)
        or payload["overlay_policy"].get("canonical_self_hash")
        != execution.overlay_policy_canonical_self_hash
        or not isinstance(payload.get("frozen_quality_policy"), Mapping)
        or payload["frozen_quality_policy"].get("canonical_self_hash")
        != base_policy["canonical_self_hash"]
        or payload.get("quality_audit_execution_adapter")
        != _file_binding(Path(__file__))
    ):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit receipt payload binding drift"
        )


def build_consumed_execution(
    *,
    receipt_id: str,
    overlay_policy_canonical_self_hash: str,
    base_policy: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    pending_receipt_path: Path,
    consumed_receipt_binding: Mapping[str, Any],
    result_path: Path,
    dataset_root: Path,
    root_manifest_binding: Mapping[str, Any],
) -> ConsumedQualityAuditExecution:
    """Build the explicit context after the caller consumed its receipt."""

    channel_policy.validate_policy(base_policy)
    if (
        not HEX_SHA256_RE.fullmatch(receipt_id)
        or not HEX_SHA256_RE.fullmatch(overlay_policy_canonical_self_hash)
        or set(root_manifest_binding)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or not HEX_SHA256_RE.fullmatch(str(root_manifest_binding["sha256"]))
        or not HEX_SHA256_RE.fullmatch(
            str(root_manifest_binding["canonical_self_hash"])
        )
        or not isinstance(root_manifest_binding["size_bytes"], int)
        or isinstance(root_manifest_binding["size_bytes"], bool)
        or root_manifest_binding["size_bytes"] <= 0
        or set(consumed_receipt_binding) != {"path", "size_bytes", "sha256"}
        or not isinstance(consumed_receipt_binding["size_bytes"], int)
        or isinstance(consumed_receipt_binding["size_bytes"], bool)
        or consumed_receipt_binding["size_bytes"] <= 0
        or not HEX_SHA256_RE.fullmatch(
            str(consumed_receipt_binding["sha256"])
        )
    ):
        raise QualityAuditExecutionAdapterError(
            "Quality-audit execution context schema drift"
        )
    resolved_root = dataset_root.resolve()
    manifest_path = (resolved_root / "root_manifest.json").resolve()
    try:
        expected_path = manifest_path.relative_to(ROOT).as_posix()
        resolved_root.relative_to(ROOT)
    except ValueError as exc:
        raise QualityAuditExecutionAdapterError(
            "Quality-audit dataset root escaped the repository"
        ) from exc
    if str(root_manifest_binding["path"]) != expected_path:
        raise QualityAuditExecutionAdapterError(
            "Quality-audit root manifest path drift"
        )
    consumed_path = (ROOT / str(consumed_receipt_binding["path"])).resolve()
    pending_path = pending_receipt_path.resolve()
    resolved_result_path = result_path.resolve()
    execution = ConsumedQualityAuditExecution(
        receipt_id=receipt_id,
        overlay_policy_canonical_self_hash=overlay_policy_canonical_self_hash,
        base_policy_canonical_self_hash=str(base_policy["canonical_self_hash"]),
        capabilities_canonical_sha256=capabilities_canonical_sha256(capabilities),
        pending_receipt_path=pending_path,
        consumed_receipt_path=consumed_path,
        consumed_receipt_size_bytes=int(
            consumed_receipt_binding["size_bytes"]
        ),
        consumed_receipt_sha256=str(consumed_receipt_binding["sha256"]),
        result_path=resolved_result_path,
        dataset_root=resolved_root,
        root_manifest_repository_path=expected_path,
        root_manifest_size_bytes=int(root_manifest_binding["size_bytes"]),
        root_manifest_sha256=str(root_manifest_binding["sha256"]),
        root_manifest_canonical_self_hash=str(
            root_manifest_binding["canonical_self_hash"]
        ),
    )
    validate_consumed_execution(execution, base_policy=base_policy)
    return execution


def validate_consumed_execution(
    execution: ConsumedQualityAuditExecution,
    *,
    base_policy: Mapping[str, Any],
) -> None:
    channel_policy.validate_policy(base_policy)
    if (
        type(execution) is not ConsumedQualityAuditExecution
        or execution.base_policy_canonical_self_hash
        != base_policy["canonical_self_hash"]
        or not HEX_SHA256_RE.fullmatch(execution.receipt_id)
        or not HEX_SHA256_RE.fullmatch(
            execution.overlay_policy_canonical_self_hash
        )
        or execution.capabilities_canonical_sha256
        != capabilities_canonical_sha256(EXPECTED_CAPABILITIES)
        or not HEX_SHA256_RE.fullmatch(execution.consumed_receipt_sha256)
        or execution.pending_receipt_path
        != execution.pending_receipt_path.resolve()
        or execution.consumed_receipt_path
        != execution.consumed_receipt_path.resolve()
        or execution.result_path != execution.result_path.resolve()
        or execution.result_path.exists()
        or execution.dataset_root != execution.dataset_root.resolve()
        or not execution.dataset_root.is_dir()
    ):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit execution binding drift"
        )
    manifest_path = (execution.dataset_root / "root_manifest.json").resolve()
    try:
        expected_path = manifest_path.relative_to(ROOT).as_posix()
        execution.dataset_root.relative_to(ROOT)
        execution.pending_receipt_path.relative_to(ROOT)
        execution.consumed_receipt_path.relative_to(ROOT)
        execution.result_path.relative_to(ROOT)
    except ValueError as exc:
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit path escaped the repository"
        ) from exc
    pin = execution.root_pin()
    if (
        pin.path != ROOT_MANIFEST_PIN_PATH
        or expected_path != execution.root_manifest_repository_path
        or not manifest_path.is_file()
        or manifest_path.stat().st_size != pin.size_bytes
        or frozen_runner._sha256_file(manifest_path) != pin.sha256
    ):
        raise QualityAuditExecutionAdapterError(
            "Consumed quality-audit root pin drift"
        )
    _validate_consumed_receipt_evidence(execution, base_policy=base_policy)


def _build_bound_truth_capability(
    *,
    execution: ConsumedQualityAuditExecution,
    policy: Mapping[str, Any],
) -> truth_capability.FormalTrainDevelopmentTruthCapability:
    """Compose the repository lineage binding with the root-local truth pin."""

    validate_consumed_execution(execution, base_policy=policy)
    value = (
        truth_capability.FormalTrainDevelopmentTruthCapability
        .from_pinned_design_root(
            dataset_root=execution.dataset_root,
            root_manifest_pin=execution.root_pin(),
        )
    )
    if value.root_binding() != execution.root_binding():
        raise frozen_validator.QualityProbeValidationError(
            "Formal truth capability does not match consumed execution root pin"
        )
    return value


def aggregate_authorized_formal_structure(
    *,
    public_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    structure_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    policy: Mapping[str, Any],
    execution: ConsumedQualityAuditExecution,
) -> dict[str, Any]:
    """Run the frozen formal structure core under the consumed capability.

    The frozen policy deliberately keeps ``quality_audit_run`` false, so its
    original public wrapper cannot represent a post-build one-shot grant.  The
    consumed execution context supplies only that grant; every scientific
    parameter still comes from the byte-pinned frozen policy.
    """

    validate_consumed_execution(execution, base_policy=policy)
    execution_snapshot = copy.deepcopy(execution)
    policy_bytes = _canonical_json_bytes(policy)
    receipt = frozen_runner.structure_aggregator._aggregate(
        public_rows_by_split=public_rows_by_split,
        structure_rows_by_split=structure_rows_by_split,
        expected_world_counts=policy["design_scale"]["world_counts"],
        expected_sellers_per_world=policy["design_scale"][
            "seller_count_per_world"
        ],
        maximum_position_deviation=policy["quality_gates"][
            "code_character_position_maximum_absolute_deviation_from_one_sixteenth"
        ],
        enforce_position_margin=True,
        claim_boundary="V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
    )
    validate_consumed_execution(execution, base_policy=policy)
    if execution != execution_snapshot:
        raise QualityAuditExecutionAdapterError(
            "Consumed execution context changed during structure aggregation"
        )
    if _canonical_json_bytes(policy) != policy_bytes:
        raise QualityAuditExecutionAdapterError(
            "Frozen quality policy changed during structure aggregation"
        )
    return receipt


def _evaluate_authorized_formal_family(
    *,
    train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    preloaded_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    design: frozen_validator.ProbeFamilyDesign,
    policy: Mapping[str, Any],
    train_eligibility: preparer.FrozenTextEligibility | None,
    development_eligibility: preparer.FrozenTextEligibility | None,
    execution: ConsumedQualityAuditExecution,
) -> dict[str, Any]:
    """Run the frozen formal numeric design under separate execution authority."""

    frozen_validator._validate_runtime()
    validate_consumed_execution(execution, base_policy=policy)
    caller_design = design
    caller_design_snapshot = replace(caller_design)
    design = replace(caller_design_snapshot)
    policy_snapshot = _canonical_json_bytes(policy)
    private_policy = json.loads(policy_snapshot.decode("utf-8"))
    channel_policy.validate_policy(private_policy)
    execution_snapshot = copy.deepcopy(execution)
    if (
        not isinstance(preloaded_truth, Mapping)
        or set(preloaded_truth) != {"train", "development"}
        or design.claim_boundary
        != "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING"
    ):
        raise frozen_validator.QualityProbeValidationError(
            "Authorized formal calculation input boundary drift"
        )
    train, development = frozen_validator._validate_matrix_sets(
        train_matrices, development_matrices, design
    )
    private_matrix_commitments = tuple(
        preparer.current_feature_matrix_commitment_json(frozen)
        for frozen in (*train, *development)
    )
    is_text = design.excluded_pairs_per_world > 0
    if (
        is_text
        and (train_eligibility is None or development_eligibility is None)
    ) or (
        not is_text
        and (train_eligibility is not None or development_eligibility is not None)
    ):
        raise frozen_validator.QualityProbeValidationError(
            "Eligibility capability/family drift"
        )
    train_mask_source = (
        frozen_validator._validate_eligibility(
            train_eligibility,
            row_keys=train[0].row_keys,
            excluded_pairs_per_world=design.excluded_pairs_per_world,
        )
        if is_text
        else None
    )
    development_mask_source = (
        frozen_validator._validate_eligibility(
            development_eligibility,
            row_keys=development[0].row_keys,
            excluded_pairs_per_world=design.excluded_pairs_per_world,
        )
        if is_text
        else None
    )
    train_mask = (
        None
        if train_mask_source is None
        else np.array(train_mask_source, dtype=bool, order="C", copy=True)
    )
    development_mask = (
        None
        if development_mask_source is None
        else np.array(development_mask_source, dtype=bool, order="C", copy=True)
    )
    for mask in (train_mask, development_mask):
        if mask is not None:
            mask.setflags(write=False)
    private_eligibility_commitments = tuple(
        preparer.current_text_eligibility_commitment_json(frozen)
        for frozen in (train_eligibility, development_eligibility)
        if frozen is not None
    )
    loader_calls: Counter[str] = Counter()

    def counted_truth_loader(split: str) -> Sequence[Mapping[str, Any]]:
        if split not in {"train", "development"}:
            raise frozen_validator.QualityProbeValidationError(
                "Audit truth loader call attempted"
            )
        loader_calls[split] += 1
        if loader_calls[split] != 1:
            raise frozen_validator.QualityProbeValidationError(
                "Truth loader called more than once"
            )
        return preloaded_truth[split]

    train_labels_full = frozen_validator._load_and_validate_truth(
        split="train",
        truth_loader=counted_truth_loader,
        row_keys=train[0].row_keys,
        design=design,
        eligibility=train_mask,
    )
    development_labels_full = frozen_validator._load_and_validate_truth(
        split="development",
        truth_loader=counted_truth_loader,
        row_keys=development[0].row_keys,
        design=design,
        eligibility=development_mask,
    )
    for index, frozen in enumerate((*train, *development)):
        preparer.verify_frozen_feature_matrix(frozen)
        if (
            preparer.current_feature_matrix_commitment_json(frozen)
            != private_matrix_commitments[index]
        ):
            raise frozen_validator.QualityProbeValidationError(
                "Feature matrix changed after truth open"
            )
    if train_eligibility is not None:
        preparer.verify_frozen_text_eligibility(train_eligibility)
    if development_eligibility is not None:
        preparer.verify_frozen_text_eligibility(development_eligibility)
    for index, frozen in enumerate(
        value
        for value in (train_eligibility, development_eligibility)
        if value is not None
    ):
        if (
            preparer.current_text_eligibility_commitment_json(frozen)
            != private_eligibility_commitments[index]
        ):
            raise frozen_validator.QualityProbeValidationError(
                "Text eligibility changed after truth open"
            )
    validate_consumed_execution(execution, base_policy=policy)
    if execution != execution_snapshot:
        raise frozen_validator.QualityProbeValidationError(
            "Consumed execution context changed after truth open"
        )
    if _canonical_json_bytes(policy) != policy_snapshot:
        raise frozen_validator.QualityProbeValidationError(
            "Quality policy changed after truth open"
        )
    if caller_design != caller_design_snapshot:
        raise frozen_validator.QualityProbeValidationError(
            "Probe design changed after truth open"
        )
    train_labels = (
        train_labels_full if train_mask is None else train_labels_full[train_mask]
    )
    development_labels = (
        development_labels_full
        if development_mask is None
        else development_labels_full[development_mask]
    )
    single = frozen_validator._single_feature_maximum(
        development, development_labels, development_mask
    )
    model_scores: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    for train_value, development_value in zip(train, development):
        train_x = (
            train_value.values
            if train_mask is None
            else train_value.values[train_mask]
        )
        development_x = (
            development_value.values
            if development_mask is None
            else development_value.values[development_mask]
        )
        scores = frozen_validator._fit_probe_models(
            train_x=train_x,
            train_y=train_labels,
            development_x=development_x,
            policy=private_policy,
        )
        if tuple(scores) != ("logistic_l2", "hist_gradient_boosting_depth2"):
            raise frozen_validator.QualityProbeValidationError(
                "Probe model family cardinality drift"
            )
        for model_kind, score in scores.items():
            name = f"{development_value.view}::{model_kind}"
            model_scores[name] = score
            model_metrics[name] = {
                "symmetric_auc": frozen_validator.symmetric_auc(
                    development_labels, score
                ),
                "average_precision": float(
                    average_precision_score(development_labels, score)
                ),
                "prediction_vector_sha256": frozen_validator._vector_sha256(score),
            }
    if len(model_metrics) != design.expected_views * 2:
        raise frozen_validator.QualityProbeValidationError(
            "Probe model family cardinality drift"
        )
    model_auc_maximum = max(
        value["symmetric_auc"] for value in model_metrics.values()
    )
    model_ap_uplift_maximum = max(
        value["average_precision"] - design.average_precision_baseline
        for value in model_metrics.values()
    )
    auc_winners = sorted(
        (
            name
            for name, value in model_metrics.items()
            if value["symmetric_auc"] == model_auc_maximum
        ),
        key=lambda value: value.encode("utf-8"),
    )
    ap_winners = sorted(
        (
            name
            for name, value in model_metrics.items()
            if value["average_precision"] - design.average_precision_baseline
            == model_ap_uplift_maximum
        ),
        key=lambda value: value.encode("utf-8"),
    )
    development_world_uids_full = tuple(
        world_uid for world_uid, _pair_uid in development[0].row_keys
    )
    development_world_uids = (
        development_world_uids_full
        if development_mask is None
        else tuple(
            value
            for value, keep in zip(development_world_uids_full, development_mask)
            if keep
        )
    )
    ordered_development_worlds = frozen_validator._ordered_worlds(
        development[0].row_keys
    )
    draws = frozen_validator.generate_bootstrap_draws(
        replicates=design.bootstrap_replicates,
        world_count=design.expected_worlds,
        seed=design.bootstrap_seed,
    )
    draws_hash = frozen_validator._sha256_bytes(draws.tobytes(order="C"))
    if (
        design.require_formal_bootstrap_binding
        and draws_hash != frozen_validator.FORMAL_BOOTSTRAP_SHA256
    ):
        raise frozen_validator.QualityProbeValidationError(
            "Formal bootstrap matrix hash drift"
        )
    with threadpool_limits(limits=1):
        bootstrap = frozen_validator._bootstrap_family_upper(
            labels=development_labels,
            row_world_uids=development_world_uids,
            ordered_world_uids=ordered_development_worlds,
            score_family=model_scores,
            baseline=design.average_precision_baseline,
            draws=draws,
        )
    gates = private_policy["quality_gates"]
    failures: list[str] = []
    comparisons = (
        (
            "maximum_single_feature_symmetric_auc",
            single["maximum_symmetric_auc"],
            gates["maximum_single_feature_symmetric_auc"],
        ),
        (
            "maximum_family_symmetric_auc",
            model_auc_maximum,
            gates["maximum_family_symmetric_auc"],
        ),
        (
            "maximum_family_average_precision_uplift",
            model_ap_uplift_maximum,
            gates["maximum_family_average_precision_uplift"],
        ),
        (
            "bootstrap_95_upper_symmetric_auc",
            bootstrap["symmetric_auc_95_upper"],
            gates["bootstrap_95_upper_symmetric_auc"],
        ),
        (
            "bootstrap_95_upper_average_precision_uplift",
            bootstrap["average_precision_uplift_95_upper"],
            gates["bootstrap_95_upper_average_precision_uplift"],
        ),
    )
    for name, observed, threshold in comparisons:
        if not math.isfinite(float(observed)) or float(observed) > float(threshold):
            failures.append(name)
    gate_checks = {
        name: {
            "observed": float(observed),
            "maximum_allowed": float(threshold),
            "passed": math.isfinite(float(observed))
            and float(observed) <= float(threshold),
        }
        for name, observed, threshold in comparisons
    }
    receipt: dict[str, Any] = {
        "version": frozen_validator.VERSION,
        "status": (
            "INTERNAL_PROBE_PASS_NO_STANDALONE_CLAIM"
            if not failures
            else "INTERNAL_PROBE_GATE_TRIGGERED_NO_STANDALONE_CLAIM"
        ),
        "claim_boundary": "INTERNAL_FORMAL_PROBE_CALCULATION_NO_STANDALONE_CLAIM",
        "design_claim_boundary": design.claim_boundary,
        "family": design.family,
        "train_world_count": design.expected_worlds,
        "development_world_count": design.expected_worlds,
        "full_pair_count_per_world": design.pairs_per_world,
        "eligible_pair_count_per_world": (
            design.pairs_per_world - design.excluded_pairs_per_world
        ),
        "positive_pair_count_per_world": design.positives_per_world,
        "average_precision_baseline": design.average_precision_baseline,
        "quality_policy_canonical_self_hash": private_policy[
            "canonical_self_hash"
        ],
        "execution_context": {
            "receipt_id": execution.receipt_id,
            "overlay_policy_canonical_self_hash": (
                execution.overlay_policy_canonical_self_hash
            ),
            "capabilities_canonical_sha256": (
                execution.capabilities_canonical_sha256
            ),
            "root_manifest": execution.root_binding(),
        },
        "input_commitments": {
            "train": [
                {"view": value.view, "sha256": value.commitment_sha256}
                for value in train
            ],
            "development": [
                {"view": value.view, "sha256": value.commitment_sha256}
                for value in development
            ],
            "train_text_eligibility_sha256": (
                None
                if train_eligibility is None
                else train_eligibility.commitment_sha256
            ),
            "development_text_eligibility_sha256": (
                None
                if development_eligibility is None
                else development_eligibility.commitment_sha256
            ),
        },
        "single_feature": single,
        "model_family": {
            "model_count": len(model_metrics),
            "maximum_symmetric_auc": model_auc_maximum,
            "maximum_symmetric_auc_winner": auc_winners[0],
            "maximum_symmetric_auc_tie_count": len(auc_winners),
            "maximum_average_precision_uplift": model_ap_uplift_maximum,
            "maximum_average_precision_uplift_winner": ap_winners[0],
            "maximum_average_precision_uplift_tie_count": len(ap_winners),
            "models": model_metrics,
        },
        "bootstrap": bootstrap,
        "gate_checks": gate_checks,
        "gate_failures": failures,
        "truth_loader_call_counts": {
            "train": loader_calls["train"],
            "development": loader_calls["development"],
            "audit_a": loader_calls["audit_a"],
            "audit_b": loader_calls["audit_b"],
        },
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = frozen_validator._sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    return receipt


def evaluate_authorized_formal_probe_families(
    *,
    text_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    text_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_train_matrices: Sequence[preparer.FrozenFeatureMatrix],
    code_development_matrices: Sequence[preparer.FrozenFeatureMatrix],
    policy: Mapping[str, Any],
    train_text_eligibility: preparer.FrozenTextEligibility,
    development_text_eligibility: preparer.FrozenTextEligibility,
    execution: ConsumedQualityAuditExecution,
) -> dict[str, Any]:
    """Open train/development truth once after all label-free state freezes."""

    validate_consumed_execution(execution, base_policy=policy)
    execution_snapshot = copy.deepcopy(execution)
    dataset_root = execution.dataset_root
    root_manifest_pin = execution.root_pin()
    expected_root_binding = execution.root_binding()
    manifest_path = (dataset_root / "root_manifest.json").resolve()
    try:
        root_path = manifest_path.relative_to(truth_capability.ROOT).as_posix()
    except ValueError:
        root_path = manifest_path.as_posix()
    supplied_root_binding = {
        "path": root_path,
        "size_bytes": root_manifest_pin.size_bytes,
        "sha256": root_manifest_pin.sha256,
        "canonical_self_hash": root_manifest_pin.canonical_self_hash,
    }
    if supplied_root_binding != expected_root_binding:
        raise frozen_validator.QualityProbeValidationError(
            "Formal dataset root does not match consumed execution root pin"
        )
    text_design = frozen_validator.formal_design_for_family("text", policy)
    code_design = frozen_validator.formal_design_for_family(
        "code_and_slot", policy
    )
    text_train, text_development = frozen_validator._validate_matrix_sets(
        text_train_matrices, text_development_matrices, text_design
    )
    code_train, code_development = frozen_validator._validate_matrix_sets(
        code_train_matrices, code_development_matrices, code_design
    )
    frozen_validator._validate_eligibility(
        train_text_eligibility,
        row_keys=text_train[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    frozen_validator._validate_eligibility(
        development_text_eligibility,
        row_keys=text_development[0].row_keys,
        excluded_pairs_per_world=text_design.excluded_pairs_per_world,
    )
    all_matrices = (
        *text_train,
        *text_development,
        *code_train,
        *code_development,
    )
    pretruth_matrix_bytes = tuple(
        preparer.current_feature_matrix_commitment_json(value)
        for value in all_matrices
    )
    pretruth_eligibility_bytes = (
        preparer.current_text_eligibility_commitment_json(
            train_text_eligibility
        ),
        preparer.current_text_eligibility_commitment_json(
            development_text_eligibility
        ),
    )
    policy_bytes = _canonical_json_bytes(policy)
    truth = _build_bound_truth_capability(
        execution=execution,
        policy=policy,
    )
    truth_pins = truth._begin_bound_transaction(
        expected_root_binding=expected_root_binding
    )
    preloaded_truth: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in truth_capability.SUPERVISED_SPLITS:
        rows, split_receipt = truth_capability._read_pinned_truth_csv(
            truth_pins[split]
        )
        truth._record_split_receipt(split=split, receipt=split_receipt)
        preloaded_truth[split] = rows
    frozen_validator._verify_feature_bundle_unchanged(
        all_matrices,
        pretruth_matrix_bytes,
        error_message="Formal matrix bundle changed after truth open",
    )
    frozen_validator._verify_eligibility_bundle_unchanged(
        (train_text_eligibility, development_text_eligibility),
        pretruth_eligibility_bytes,
        error_message="Formal eligibility bundle changed after truth open",
    )
    validate_consumed_execution(execution, base_policy=policy)
    if execution != execution_snapshot:
        raise frozen_validator.QualityProbeValidationError(
            "Consumed execution context changed after truth open"
        )
    if _canonical_json_bytes(policy) != policy_bytes:
        raise frozen_validator.QualityProbeValidationError(
            "Quality policy changed after truth open"
        )
    try:
        text_receipt = _evaluate_authorized_formal_family(
            train_matrices=text_train,
            development_matrices=text_development,
            preloaded_truth=preloaded_truth,
            design=text_design,
            policy=policy,
            train_eligibility=train_text_eligibility,
            development_eligibility=development_text_eligibility,
            execution=execution,
        )
        frozen_validator._verify_feature_bundle_unchanged(
            all_matrices,
            pretruth_matrix_bytes,
            error_message="Formal matrix bundle changed between probe families",
        )
        frozen_validator._verify_eligibility_bundle_unchanged(
            (train_text_eligibility, development_text_eligibility),
            pretruth_eligibility_bytes,
            error_message=(
                "Formal eligibility bundle changed between probe families"
            ),
        )
        validate_consumed_execution(execution, base_policy=policy)
        if execution != execution_snapshot:
            raise frozen_validator.QualityProbeValidationError(
                "Consumed execution context changed between probe families"
            )
        if _canonical_json_bytes(policy) != policy_bytes:
            raise frozen_validator.QualityProbeValidationError(
                "Quality policy changed between probe families"
            )
        code_receipt = _evaluate_authorized_formal_family(
            train_matrices=code_train,
            development_matrices=code_development,
            preloaded_truth=preloaded_truth,
            design=code_design,
            policy=policy,
            train_eligibility=None,
            development_eligibility=None,
            execution=execution,
        )
    finally:
        preloaded_truth.clear()
    frozen_validator._verify_feature_bundle_unchanged(
        all_matrices,
        pretruth_matrix_bytes,
        error_message="Formal matrix bundle changed after probe families",
    )
    frozen_validator._verify_eligibility_bundle_unchanged(
        (train_text_eligibility, development_text_eligibility),
        pretruth_eligibility_bytes,
        error_message="Formal eligibility bundle changed after probe families",
    )
    validate_consumed_execution(execution, base_policy=policy)
    if execution != execution_snapshot:
        raise frozen_validator.QualityProbeValidationError(
            "Consumed execution context changed after probe families"
        )
    if _canonical_json_bytes(policy) != policy_bytes:
        raise frozen_validator.QualityProbeValidationError(
            "Quality policy changed after probe families"
        )
    truth_receipt = truth.aggregate_receipt()
    if any(
        truth_receipt[split][field] != 0
        for split in ("audit_a", "audit_b")
        for field in (
            "file_open_count",
            "byte_read_count",
            "materialized_row_count",
        )
    ):
        raise frozen_validator.QualityProbeValidationError(
            "Audit truth access count drift"
        )
    status = (
        "PASS"
        if not text_receipt["gate_failures"]
        and not code_receipt["gate_failures"]
        else "DATASET_INVALIDATED"
    )
    receipt: dict[str, Any] = {
        "version": frozen_validator.VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "family_receipts": {
            "text": text_receipt,
            "code_and_slot": code_receipt,
        },
        "truth_file_access": truth_receipt,
        "audit_a_b_truth_remained_sealed": True,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
    }
    receipt["canonical_self_hash"] = frozen_validator._sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    return receipt


def _input_file_verification_scope(
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    loaded: Mapping[str, Mapping[str, Any]],
    supervised_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Report byte-verified inputs separately from deliberately unopened files."""

    declared: dict[str, dict[str, Any]] = {}
    for split in frozen_runner.SPLITS:
        records = frozen_runner._manifest_records(manifests[split])
        for relative, record in records.items():
            path = f"{split}/{relative}"
            declared[path] = {
                "path": path,
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
    label_free_paths = {
        source.path
        for split in frozen_runner.SPLITS
        for source in loaded[split]["sources"].values()
    }
    actual_paths = set(label_free_paths)
    truth_paths: set[str] = set()
    if supervised_receipt is not None:
        access = supervised_receipt.get("truth_file_access")
        if not isinstance(access, Mapping):
            raise frozen_validator.QualityProbeValidationError(
                "Truth file verification receipt is absent"
            )
        for split in truth_capability.SUPERVISED_SPLITS:
            split_receipt = access.get(split)
            path = f"{split}/{truth_capability.TRUTH_RELATIVE_PATH}"
            if (
                not isinstance(split_receipt, Mapping)
                or split_receipt.get("file_open_count") != 1
                or split_receipt.get("sha256") != declared[path]["sha256"]
            ):
                raise frozen_validator.QualityProbeValidationError(
                    "Supervised truth byte-verification receipt drift"
                )
            truth_paths.add(path)
        for split in ("audit_a", "audit_b"):
            split_receipt = access.get(split)
            if (
                not isinstance(split_receipt, Mapping)
                or any(
                    split_receipt.get(field) != 0
                    for field in (
                        "file_open_count",
                        "byte_read_count",
                        "materialized_row_count",
                    )
                )
            ):
                raise frozen_validator.QualityProbeValidationError(
                    "Audit truth seal receipt drift"
                )
        actual_paths.update(truth_paths)
    if not actual_paths <= set(declared):
        raise frozen_runner.QualityAuditRunnerError(
            "Byte-verified file registry escaped the declared manifest universe"
        )
    manifest_pin_only_paths = set(declared) - actual_paths
    audit_truth_paths = {
        f"{split}/{truth_capability.TRUTH_RELATIVE_PATH}"
        for split in ("audit_a", "audit_b")
    }
    if not audit_truth_paths <= manifest_pin_only_paths:
        raise frozen_validator.QualityProbeValidationError(
            "Audit truth entered the byte-verified input registry"
        )

    def commitment(paths: set[str]) -> str:
        bindings = [declared[path] for path in sorted(paths, key=str.encode)]
        return hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()

    actual_ordered = sorted(actual_paths, key=str.encode)
    manifest_only_ordered = sorted(manifest_pin_only_paths, key=str.encode)
    return {
        "declared_data_file_count": len(declared),
        "label_free_actual_byte_verified_count": len(label_free_paths),
        "supervised_truth_actual_byte_verified_count": len(truth_paths),
        "actual_byte_verified_count": len(actual_paths),
        "actual_byte_verified_paths": actual_ordered,
        "actual_byte_verified_binding_sha256": commitment(actual_paths),
        "manifest_pin_only_count": len(manifest_pin_only_paths),
        "manifest_pin_only_paths": manifest_only_ordered,
        "manifest_pin_only_binding_sha256": commitment(
            manifest_pin_only_paths
        ),
        "audit_a_b_truth_manifest_pin_only_paths": sorted(
            audit_truth_paths, key=str.encode
        ),
        "audit_a_b_truth_actual_byte_read_count": 0,
        "declared_unclassified_count": len(declared)
        - len(actual_paths)
        - len(manifest_pin_only_paths),
        "scope_claim": (
            "ACTUAL_BYTES_VERIFIED_ONLY_FOR_LISTED_PATHS_"
            "OTHER_PATHS_MANIFEST_PIN_ONLY"
        ),
    }


def run_authorized_formal_quality_audit(
    *,
    policy: Mapping[str, Any],
    execution: ConsumedQualityAuditExecution,
    state: dict[str, str],
) -> dict[str, Any]:
    """Run the frozen scientific audit with an explicit consumed context."""

    state["stage"] = "consumed_execution_and_root_binding"
    validate_consumed_execution(execution, base_policy=policy)
    dataset_root = execution.dataset_root
    root_pin = execution.root_pin()
    state["stage"] = "root_manifest_and_physical_universe"
    root_manifest, manifests = frozen_runner._load_root_manifests(
        dataset_root=dataset_root, root_pin=root_pin
    )
    equivalence = root_manifest.get("v9_1_equivalence_replay")
    if (
        root_manifest.get("status")
        != "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED"
        or root_manifest.get("canonical_self_hash")
        != EXPECTED_ROOT_MANIFEST_CANONICAL_SELF_HASH
        or root_manifest.get("execution_mode") != "design_preflight"
        or root_manifest.get("scientific_use_forbidden") is not True
        or root_manifest.get("formal_seed_created") is not False
        or root_manifest.get("formal_rows_created") != 0
        or root_manifest.get("training_started") is not False
        or not isinstance(equivalence, Mapping)
        or equivalence.get("status")
        != "PASS_EXACT_MECHANICAL_PROFILE_COMMITMENT_REPAIR"
        or equivalence.get("canonical_self_hash")
        != EXPECTED_EQUIVALENCE_CANONICAL_SELF_HASH
        or frozen_runner._canonical_self_hash(equivalence)
        != EXPECTED_EQUIVALENCE_CANONICAL_SELF_HASH
        or equivalence.get("same_random_authority") is not True
        or equivalence.get("unchanged_file_count") != 68
        or equivalence.get("changed_structure_file_count") != 4
        or tuple(equivalence.get("allowed_changed_json_paths", ()))
        != EXPECTED_EQUIVALENCE_PATHS
    ):
        raise frozen_runner.QualityAuditRunnerError(
            "V9.1 design-only root or equivalence binding drift"
        )
    state["stage"] = "builder_policy_binding"
    builder_policy = frozen_runner._validate_builder_policy_binding(
        root_manifest
    )
    state["stage"] = "label_free_split_loading"
    loaded = {
        split: frozen_runner._load_split_label_free(
            dataset_root=dataset_root,
            split=split,
            manifest=manifests[split],
        )
        for split in frozen_runner.SPLITS
    }
    state["stage"] = "four_split_uid_endpoint_and_view_closure"
    frozen_runner._validate_public_uid_registries(
        root_manifest=root_manifest,
        manifests=manifests,
        loaded=loaded,
    )
    state["stage"] = "builder_authority_replay"
    context = frozen_runner.scientific.build_execution_context(
        builder_policy, execution_mode="design_preflight"
    )
    if context.output_root.resolve() != dataset_root:
        raise frozen_runner.QualityAuditRunnerError(
            "Builder authority/output root binding drift"
        )
    records_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in frozen_runner.SPLITS
    }
    for record in context.world_records:
        records_by_split[str(record["split"])].append(dict(record))
    for split in frozen_runner.SPLITS:
        records_by_split[split].sort(
            key=lambda row: int(row["split_ordinal"])
        )
        expected_world_projection = [
            {
                "world_uid": str(row["world_uid"]),
                "split_ordinal": int(row["split_ordinal"]),
            }
            for row in records_by_split[split]
        ]
        if list(loaded[split]["worlds"]) != expected_world_projection:
            raise frozen_runner.QualityAuditRunnerError(
                "Persisted world authority replay drift"
            )
    builder_policy_source = frozen_runner._repo_source(
        frozen_runner.scientific.DEFAULT_POLICY_PATH
    )
    code_key = frozen_runner.document_capacity.derive_code_key(
        context.document_variation_key
    )
    id_key = str(
        context.effective_policy["randomness"][context.base_mode]["id_key_hex"]
    )
    frozen_runner._validate_builder_seller_authority(
        loaded=loaded,
        records_by_split=records_by_split,
        id_key=id_key,
        expected_sellers_per_world=policy["design_scale"][
            "seller_count_per_world"
        ],
    )
    state["stage"] = "label_free_structure_schema_and_zero_gates"
    structure_receipt = aggregate_authorized_formal_structure(
        public_rows_by_split={
            split: loaded[split]["public_code"]
            for split in frozen_runner.SPLITS
        },
        structure_rows_by_split={
            split: loaded[split]["structure_audit"]
            for split in frozen_runner.SPLITS
        },
        policy=policy,
        execution=execution,
    )
    if structure_receipt["status"] != "PASS":
        receipt: dict[str, Any] = {
            "version": frozen_runner.VERSION,
            "status": "DATASET_INVALIDATED",
            "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
            "structure": structure_receipt,
            "input_file_verification_scope": _input_file_verification_scope(
                manifests=manifests,
                loaded=loaded,
                supervised_receipt=None,
            ),
            "supervised_truth_opened": False,
            "audit_a_b_truth_open_count": 0,
            "formal_500_by_4_generated": False,
            "training_started": False,
        }
        receipt["canonical_self_hash"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        return receipt
    state["stage"] = "loaded_model_view_structure_binding"
    frozen_runner._validate_loaded_structure_bindings(
        loaded=loaded,
        expected_clone_count_per_world=builder_policy[
            "exact_title_clone_endpoint_qualification"
        ]["expected_exact_title_clone_count_per_world"],
    )
    state["stage"] = "label_free_feature_freeze"
    text_matrices: dict[
        str, tuple[preparer.FrozenFeatureMatrix, ...]
    ] = {}
    code_matrices: dict[
        str, tuple[preparer.FrozenFeatureMatrix, ...]
    ] = {}
    eligibilities: dict[str, preparer.FrozenTextEligibility] = {}
    for split in ("train", "development"):
        endpoints = loaded[split]["endpoints"]
        ordered_world_uids = tuple(
            str(row["world_uid"]) for row in loaded[split]["worlds"]
        )
        source_map = loaded[split]["sources"]
        surface_values: list[preparer.FrozenFeatureMatrix] = []
        for surface in policy["model_views"]["order"]:
            item_path, profile_path = frozen_runner.SURFACE_FILES[surface]
            items, profiles = loaded[split]["surface_rows"][surface]
            surface_values.extend(
                preparer.prepare_text_surface_matrices(
                    surface=surface,
                    items=items,
                    profiles=profiles,
                    endpoints=endpoints,
                    ordered_world_uids=ordered_world_uids,
                    sources=frozen_runner._source_tuple(
                        source_map[frozen_runner.ENDPOINT_PATH],
                        source_map[item_path],
                        source_map[profile_path],
                    ),
                )
            )
        text_matrices[split] = tuple(surface_values)
        eligibilities[split] = preparer.freeze_text_eligibility(
            eligibility_rows=loaded[split]["eligibility"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            sources=frozen_runner._source_tuple(
                source_map[frozen_runner.ELIGIBILITY_PATH],
                source_map[frozen_runner.ENDPOINT_PATH],
            ),
        )
        public_matrix = preparer.prepare_public_code_matrix(
            public_rows=loaded[split]["public_code"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            sources=frozen_runner._source_tuple(
                source_map[frozen_runner.PUBLIC_CODE_PATH],
                source_map[frozen_runner.ENDPOINT_PATH],
            ),
        )
        expected_ordinals = {
            str(row["world_uid"]): int(row["mode_global_ordinal"])
            for row in records_by_split[split]
        }
        expected_seller_slots = {
            (
                world_uid,
                frozen_runner.structure.base_uid(
                    key_hex=id_key,
                    entity_kind="seller",
                    parent_uid_or_mode=world_uid,
                    ordinal=slot,
                ),
            ): slot
            for world_uid in ordered_world_uids
            for slot in range(policy["design_scale"]["seller_count_per_world"])
        }
        decoded_matrix = preparer.prepare_decoded_slot_matrix(
            public_rows=loaded[split]["public_code"],
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            expected_mode_global_ordinal_by_world=expected_ordinals,
            expected_seller_slot_by_world_and_seller=expected_seller_slots,
            decode_coordinate=lambda _world_uid, code: (
                frozen_runner.document_capacity.decode_code(
                    code_key=code_key, code=code
                )
            ),
            sources=frozen_runner._source_tuple(
                builder_policy_source,
                source_map[frozen_runner.PUBLIC_CODE_PATH],
                source_map[frozen_runner.ENDPOINT_PATH],
                source_map[frozen_runner.WORLDS_PATH],
            ),
        )
        code_matrices[split] = (public_matrix, decoded_matrix)
    state["stage"] = "train_development_truth_and_supervised_gates"
    supervised_receipt = evaluate_authorized_formal_probe_families(
        text_train_matrices=text_matrices["train"],
        text_development_matrices=text_matrices["development"],
        code_train_matrices=code_matrices["train"],
        code_development_matrices=code_matrices["development"],
        policy=policy,
        train_text_eligibility=eligibilities["train"],
        development_text_eligibility=eligibilities["development"],
        execution=execution,
    )
    status = (
        "PASS"
        if structure_receipt["status"] == "PASS"
        and supervised_receipt["status"] == "PASS"
        else "DATASET_INVALIDATED"
    )
    state["stage"] = "aggregate_success_receipt"
    receipt = {
        "version": frozen_runner.VERSION,
        "status": status,
        "claim_boundary": "V9_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "root_manifest": execution.root_binding(),
        "structure": structure_receipt,
        "supervised": supervised_receipt,
        "input_file_verification_scope": _input_file_verification_scope(
            manifests=manifests,
            loaded=loaded,
            supervised_receipt=supervised_receipt,
        ),
        "audit_a_b_truth_remained_sealed": True,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    receipt["canonical_self_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt
