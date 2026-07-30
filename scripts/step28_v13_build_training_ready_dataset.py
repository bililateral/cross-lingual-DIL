#!/usr/bin/env python3
"""Build the non-smoke Step28-v13 Chinese training/evaluation dataset.

The implementation creates an explicit ``training_ready`` execution mode from
the frozen formal public streams plus exactly one split-private structure key.
No smoke data, smoke UID, smoke identity value, English label, Chinese label,
M0 score or adapter score is read while constructing a split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import stat
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import step28_v13_common as common
import step28_v13_feature_derangement as feature_derangement
import step28_v13_generate_dataset as legacy_generator
import step28_v13_history_features as history_features
import step28_v13_independent_dgp_comparator as independent_comparator
import step28_v13_independent_private_dgp_replay as independent_replay
import step28_v13_integrity_receipts as integrity_receipts
import step28_v13_identity_values as identity_values
import step28_v13_mechanism_stratified_c40 as mechanism_c40
import step28_v13_metadata_shortcut_common as shortcut_common
import step28_v13_producer_dgp_projection as producer_projection
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_mod
import step28_v13_project_null_nuisance as nuisance_projector
import step28_v13_run_metadata_shortcut_audit as shortcut_audit
import step28_v13_seal_classification_labels as label_sealer
import step28_v13_smoke_private_regeneration as regeneration
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder


MODE = "training_ready"
SPLITS = ("train", "development", "audit_a", "audit_b")
DESIGN_ONLY_STRUCTURE_KEY_HEX = (
    "9d732cddc0b5aa7e168f7c67b3c8c4e97f773b2f6dc71de5be18d70db54f6f91"
)
PUBLIC_RANDOM_FIELDS = (
    "id_namespace_key_hex",
    "id_key_hex",
    "identity_value_key_hex",
    "text_key_hex",
    "candidate_key_hex",
    "query_key_hex",
    "rewire_key_hexes",
)
DEFAULT_OVERLAY = (
    common.ROOT / "schema" / "step28_v13_training_ready_dataset_policy.json"
)
MANIFEST_VERSION = (
    "2026-07-30-step28-v13-training-ready-split-manifest-v3"
)
IMPLEMENTATION_CONTRACT_VERSION = (
    "2026-07-30-step28-v13-training-ready-implementation-contract-v3"
)
BUILDER_SOURCE_CLOSURE_ROLE = "training_ready_dataset_builder"
_EXACT_PREFLIGHT_FULL_REPLAY_CACHE: set[str] = set()


def _canonical_self_hash(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload["self_sha256"] = None
    return common.canonical_sha256(payload)


def _implementation_contract_payload(
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the scientific implementation subset stable across release phases."""

    custody = copy.deepcopy(overlay["private_structure_key_custody"])
    custody["commitments"] = {split: None for split in SPLITS}
    custody["ceremony_receipt"] = None
    return {
        "version": IMPLEMENTATION_CONTRACT_VERSION,
        "run_id": overlay["run_id"],
        "output_root": overlay["output_root"],
        "target_release_claim_level": overlay[
            "target_release_claim_level"
        ],
        "current_readiness_semantics": overlay[
            "current_readiness_semantics"
        ],
        "base_policy": copy.deepcopy(overlay["base_policy"]),
        "scientific_contract": copy.deepcopy(
            overlay["scientific_contract"]
        ),
        "world_counts": copy.deepcopy(overlay["world_counts"]),
        "sellers_per_world": overlay["sellers_per_world"],
        "complete_pairs_per_world": overlay["complete_pairs_per_world"],
        "classification_pairs_per_world": overlay[
            "classification_pairs_per_world"
        ],
        "classification_positive_count_per_world": copy.deepcopy(
            overlay["classification_positive_count_per_world"]
        ),
        "positive_mechanism_count_per_world": overlay[
            "positive_mechanism_count_per_world"
        ],
        "negative_flag_count_per_world": overlay[
            "negative_flag_count_per_world"
        ],
        "candidate_design": copy.deepcopy(overlay["candidate_design"]),
        "dataset_builder": copy.deepcopy(overlay["dataset_builder"]),
        "release_tools": copy.deepcopy(overlay["release_tools"]),
        "handle_encoding": overlay["handle_encoding"],
        "identity_value_salts": copy.deepcopy(
            overlay["identity_value_salts"]
        ),
        "base_execution_mode": overlay["base_execution_mode"],
        "base_execution_mode_semantics": overlay[
            "base_execution_mode_semantics"
        ],
        "public_randomness_source": overlay["public_randomness_source"],
        "private_structure_key_custody": custody,
        "retrieval": copy.deepcopy(overlay["retrieval"]),
        "supervision": copy.deepcopy(overlay["supervision"]),
        "shortcut_gate": copy.deepcopy(overlay["shortcut_gate"]),
        "m1": copy.deepcopy(overlay["m1"]),
        "release_status_required": overlay["release_status_required"],
    }


def implementation_contract_sha256(
    overlay: Mapping[str, Any],
) -> str:
    return common.canonical_sha256(
        _implementation_contract_payload(overlay)
    )


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    path_text = str(spec.get("path", ""))
    expected = spec.get("sha256")
    if not path_text or not isinstance(expected, str) or len(expected) != 64:
        raise common.ContractError(f"Unfrozen training-ready pin: {label}")
    path = common.repo_path(path_text)
    if not path.is_file() or common.sha256_file(path) != expected.lower():
        raise common.ContractError(f"Training-ready pin drift: {label}")
    return path


def _validate_dataset_builder_closure(
    overlay: Mapping[str, Any],
) -> None:
    builder_spec = overlay.get("dataset_builder")
    if not isinstance(builder_spec, Mapping) or set(builder_spec) != {
        "implementation",
        "sha256",
        "implementation_closure",
    }:
        raise common.ContractError("Dataset-builder pin schema drift")
    _verify_pin(
        {
            "path": builder_spec["implementation"],
            "sha256": builder_spec["sha256"],
        },
        label="training-ready dataset builder",
    )
    closure = builder_spec.get("implementation_closure")
    if not isinstance(closure, Mapping) or set(closure) != {
        "role",
        "member_count",
        "members",
        "canonical_sha256",
    }:
        raise common.ContractError("Dataset-builder source closure drift")
    observed_members = list(
        integrity_receipts._source_closure_members(
            BUILDER_SOURCE_CLOSURE_ROLE
        )
    )
    if (
        closure.get("role") != BUILDER_SOURCE_CLOSURE_ROLE
        or type(closure.get("member_count")) is not int
        or closure["member_count"] != len(observed_members)
        or closure.get("members") != observed_members
        or closure.get("canonical_sha256")
        != integrity_receipts._source_closure_sha256(
            BUILDER_SOURCE_CLOSURE_ROLE
        )
        or builder_spec["implementation"] not in observed_members
    ):
        raise common.ContractError(
            "Dataset-builder recursive source closure drift"
        )


def _validate_implementation_contract(
    overlay: Mapping[str, Any],
) -> None:
    registered = overlay.get("implementation_contract")
    if not isinstance(registered, Mapping) or set(registered) != {
        "version",
        "sha256",
    }:
        raise common.ContractError("Implementation-contract schema drift")
    observed = implementation_contract_sha256(overlay)
    if (
        registered.get("version") != IMPLEMENTATION_CONTRACT_VERSION
        or registered.get("sha256") != observed
    ):
        raise common.ContractError("Implementation-contract hash drift")


def _validate_scientific_contract(
    overlay: Mapping[str, Any],
) -> None:
    specification = overlay.get("scientific_contract")
    if (
        not isinstance(specification, Mapping)
        or set(specification) != {"path", "sha256"}
    ):
        raise common.ContractError(
            "Training-ready scientific-contract pin schema drift"
        )
    document = common.load_json(
        _verify_pin(
            specification,
            label="training-ready scientific contract",
        )
    )
    _validate_canonical_self_hash(
        document,
        label="Training-ready scientific contract",
    )
    sample_size = document.get("sample_size_design", {})
    artifact = sample_size.get("artifact", {})
    producer = sample_size.get("producer", {})
    if (
        document.get("version")
        != (
            "2026-07-30-step28-v13-training-ready-"
            "scientific-contract-v1"
        )
        or document.get("status")
        != "FROZEN_PRE_KEY_SCIENTIFIC_RULES"
        or document.get("target_release_claim_level")
        != overlay["target_release_claim_level"]
        or document.get("split_design", {}).get("worlds")
        != overlay["world_counts"]
        or document.get("split_design", {}).get(
            "sellers_per_world"
        )
        != overlay["sellers_per_world"]
        or document.get("split_design", {}).get(
            "complete_pairs_per_world"
        )
        != overlay["complete_pairs_per_world"]
        or document.get("split_design", {}).get(
            "classification_pairs_per_world"
        )
        != overlay["classification_pairs_per_world"]
        or document.get("truth_and_sampling", {}).get(
            "c40_label_blind"
        )
        is not False
        or sample_size.get("confirmatory_power_certified") is not False
        or sample_size.get(
            "binary_power_based_success_claim_forbidden"
        )
        is not True
        or document.get("retrieval_metrics", {}).get(
            "primary_average_precision_metric"
        )
        != "MAP@10"
        or document.get("inference", {}).get(
            "confirmatory_success_or_failure_language_forbidden"
        )
        is not True
    ):
        raise common.ContractError(
            "Training-ready scientific-contract content drift"
        )
    artifact_path = _verify_pin(
        artifact,
        label="fixed-sample sensitivity artifact",
    )
    _verify_pin(
        producer,
        label="fixed-sample sensitivity producer",
    )
    sensitivity = common.load_json(artifact_path)
    _validate_canonical_self_hash(
        sensitivity,
        label="Fixed-sample sensitivity artifact",
    )
    if (
        sensitivity.get("status")
        != (
            "PASS_FIXED_SAMPLE_ESTIMATION_DESIGN_"
            "NOT_CONFIRMATORY_POWER_CERTIFIED"
        )
        or sensitivity.get("world_counts") != overlay["world_counts"]
        or sensitivity.get(
            "formal_private_structure_key_created_or_read"
        )
        is not False
        or sensitivity.get("formal_world_materialized") is not False
        or sensitivity.get("claim_boundary", {}).get(
            "confirmatory_power_certified"
        )
        is not False
        or sensitivity.get("claim_boundary", {}).get(
            "binary_success_or_failure_from_power_is_forbidden"
        )
        is not True
    ):
        raise common.ContractError(
            "Fixed-sample sensitivity artifact content drift"
        )


def _validate_canonical_self_hash(
    document: Mapping[str, Any],
    *,
    label: str,
) -> None:
    payload = dict(document)
    observed = payload.pop("canonical_self_hash", None)
    if (
        not isinstance(observed, str)
        or len(observed) != 64
        or observed != common.canonical_sha256(payload)
    ):
        raise common.ContractError(f"{label} self hash drift")


def _close_float(
    left: Any,
    right: Any,
    *,
    absolute_tolerance: float = 1e-15,
) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
    )


def _validate_exact_preflight_checkpoint_evidence(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    split: str,
    overlay: Mapping[str, Any],
) -> None:
    manifest = report.get("checkpoint_manifest")
    expected_roles = [
        "started",
        "shortcut_arrays",
        "shortcut",
        "identity33",
        "aggregate",
    ]
    if split in {"audit_a", "audit_b"}:
        expected_roles.append("retrieval")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "version",
            "complete",
            "split",
            "artifact_count",
            "artifacts",
        }
        or manifest.get("version")
        != (
            "2026-07-30-step28-v13-exact-preflight-"
            "checkpoint-manifest-v1"
        )
        or manifest.get("complete") is not True
        or manifest.get("split") != split
        or int(manifest.get("artifact_count", -1))
        != len(expected_roles)
        or not isinstance(manifest.get("artifacts"), list)
        or [
            item.get("role")
            for item in manifest["artifacts"]
            if isinstance(item, Mapping)
        ]
        != expected_roles
    ):
        raise common.ContractError(
            f"Exact-preflight checkpoint manifest drift: {split}"
        )
    prefix = report_path.stem + ".checkpoint."
    artifact_paths: dict[str, Path] = {}
    for item in manifest["artifacts"]:
        role = str(item.get("role", ""))
        expected_format = (
            "npz" if role == "shortcut_arrays" else "json"
        )
        if (
            not isinstance(item, Mapping)
            or set(item) != {"role", "path", "sha256", "format"}
            or item.get("format") != expected_format
            or not str(item.get("path", "")).startswith(
                report_path.parent.relative_to(common.ROOT).as_posix()
                + "/"
                + prefix
            )
        ):
            raise common.ContractError(
                f"Exact-preflight checkpoint pin drift: {split}/{role}"
            )
        path = _verify_pin(
            {
                "path": item["path"],
                "sha256": item["sha256"],
            },
            label=f"exact preflight checkpoint {split}/{role}",
        )
        if path.is_symlink() or path.parent.resolve() != report_path.parent:
            raise common.ContractError(
                f"Exact-preflight checkpoint path drift: {split}/{role}"
            )
        artifact_paths[role] = path

    checkpoints: dict[str, Mapping[str, Any]] = {}
    expected_checkpoint_keys = {
        "started": {
            "version",
            "stage",
            "split",
            "world_count",
            "bootstrap_replicates",
            "design_only",
            "formal_private_structure_key_created_or_read",
            "overlay_sha256",
            "implementation_contract_sha256",
            "builder_source_closure_sha256",
            "builder_implementation_sha256",
            "canonical_self_hash",
        },
        "shortcut": {
            "version",
            "stage",
            "stage_completed",
            "split",
            "design_only",
            "formal_private_structure_key_created_or_read",
            "elapsed_seconds",
            "projection_row_count",
            "label_row_count",
            "oof_row_count",
            "oof_rows_sha256",
            "projection_rows_sha256",
            "label_rows_sha256",
            "array_checkpoint",
            "array_schema",
            "metadata_shortcut_audit",
            "overlay_sha256",
            "canonical_self_hash",
        },
        "identity33": {
            "version",
            "stage",
            "stage_completed",
            "split",
            "elapsed_seconds",
            "audit",
            "canonical_self_hash",
        },
        "aggregate": {
            "version",
            "stage",
            "stage_completed",
            "split",
            "elapsed_seconds",
            "audit",
            "canonical_self_hash",
        },
        "retrieval": {
            "version",
            "stage",
            "stage_completed",
            "split",
            "elapsed_seconds",
            "query_count",
            "relation_count",
            "qrel_count",
            "canonical_self_hash",
        },
    }
    for role in expected_roles:
        if role == "shortcut_arrays":
            continue
        document = common.load_json(artifact_paths[role])
        _validate_canonical_self_hash(
            document,
            label=f"Exact-preflight checkpoint {split}/{role}",
        )
        if (
            set(document) != expected_checkpoint_keys[role]
            or
            document.get("version")
            != (
                "2026-07-30-step28-v13-exact-"
                "preflight-checkpoint-v2"
            )
            or document.get("split") != split
        ):
            raise common.ContractError(
                f"Exact-preflight checkpoint content drift: {split}/{role}"
            )
        checkpoints[role] = document
    started = checkpoints["started"]
    if (
        started.get("stage") != "preflight_started"
        or int(started.get("world_count", -1))
        != int(overlay["world_counts"][split])
        or int(started.get("bootstrap_replicates", -1))
        != int(overlay["shortcut_gate"]["bootstrap_replicates"])
        or started.get("design_only") is not True
        or started.get(
            "formal_private_structure_key_created_or_read"
        )
        is not False
        or started.get("implementation_contract_sha256")
        != report.get("implementation_contract_sha256")
        or started.get("builder_source_closure_sha256")
        != report.get("builder_source_closure_sha256")
        or started.get("builder_implementation_sha256")
        != report.get("builder_implementation_sha256")
        or started.get("overlay_sha256")
        != report.get("overlay_sha256")
    ):
        raise common.ContractError(
            f"Exact-preflight started checkpoint drift: {split}"
        )
    shortcut_checkpoint = checkpoints["shortcut"]
    identity_checkpoint = checkpoints["identity33"]
    aggregate_checkpoint = checkpoints["aggregate"]
    if (
        shortcut_checkpoint.get("stage")
        != "metadata_shortcut_audit"
        or shortcut_checkpoint.get("stage_completed") is not True
        or shortcut_checkpoint.get("design_only") is not True
        or shortcut_checkpoint.get(
            "formal_private_structure_key_created_or_read"
        )
        is not False
        or shortcut_checkpoint.get("metadata_shortcut_audit")
        != report.get("metadata_shortcut_audit")
        or shortcut_checkpoint.get("overlay_sha256")
        != report.get("overlay_sha256")
        or identity_checkpoint.get("stage")
        != "identity33_matrix_audit"
        or identity_checkpoint.get("stage_completed") is not True
        or identity_checkpoint.get("audit")
        != report.get("identity33_matrix_audit")
        or aggregate_checkpoint.get("stage")
        != "aggregate_payload_validation"
        or aggregate_checkpoint.get("stage_completed") is not True
        or aggregate_checkpoint.get("audit", {}).get(
            "all_keysets_and_foreign_keys_exact"
        )
        is not True
        or aggregate_checkpoint.get("audit", {}).get(
            "all_source_dataset_names_training_ready"
        )
        is not True
        or aggregate_checkpoint.get("audit", {}).get(
            "identity_values_replayed_exactly"
        )
        is not True
    ):
        raise common.ContractError(
            f"Exact-preflight checkpoint stage drift: {split}"
        )
    if split in {"audit_a", "audit_b"}:
        retrieval = checkpoints["retrieval"]
        world_count = int(overlay["world_counts"][split])
        if (
            retrieval.get("stage") != "retrieval_build"
            or retrieval.get("stage_completed") is not True
            or int(retrieval.get("query_count", -1))
            != world_count * 4
            or int(retrieval.get("relation_count", -1))
            != world_count * 4 * 27
            or int(retrieval.get("qrel_count", -1))
            != world_count * 4 * 27
        ):
            raise common.ContractError(
                f"Exact-preflight retrieval checkpoint drift: {split}"
            )

    expected_array_keys = {
        "feature_names",
        "x",
        "y",
        "pair_uids",
        "world_uids",
        "folds",
        "score_logistic_l2",
        "score_gradient_tree",
        "score_rbf_svm",
        "bootstrap_statistics",
        "projection_rows_canonical_json_utf8",
        "label_rows_canonical_json_utf8",
        "oof_rows_canonical_json_utf8",
    }
    with np.load(
        artifact_paths["shortcut_arrays"],
        allow_pickle=False,
    ) as archive:
        if set(archive.files) != expected_array_keys:
            raise common.ContractError(
                f"Exact-preflight array schema drift: {split}"
            )
        arrays = {
            name: np.asarray(archive[name])
            for name in expected_array_keys
        }
    observed_array_schema = {
        name: {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
        }
        for name, values in sorted(arrays.items())
    }
    array_record = shortcut_checkpoint.get("array_checkpoint")
    expected_array_path = artifact_paths[
        "shortcut_arrays"
    ].relative_to(common.ROOT).as_posix()
    if (
        shortcut_checkpoint.get("array_schema")
        != observed_array_schema
        or not isinstance(array_record, Mapping)
        or set(array_record) != {"path", "sha256"}
        or array_record.get("path") != expected_array_path
        or array_record.get("sha256")
        != common.sha256_file(artifact_paths["shortcut_arrays"])
    ):
        raise common.ContractError(
            f"Exact-preflight array pin drift: {split}"
        )
    row_count = int(report["candidate_count"])
    bootstrap_replicates = int(
        overlay["shortcut_gate"]["bootstrap_replicates"]
    )
    features = list(shortcut_common.PAIR_FEATURES)
    if (
        arrays["x"].dtype != np.float64
        or arrays["x"].shape != (row_count, len(features))
        or arrays["y"].shape != (row_count,)
        or arrays["y"].dtype != np.int8
        or arrays["folds"].shape != (row_count,)
        or arrays["folds"].dtype != np.int64
        or arrays["bootstrap_statistics"].shape
        != (bootstrap_replicates,)
        or arrays["bootstrap_statistics"].dtype != np.float64
        or any(
            arrays[name].ndim != 1
            or arrays[name].dtype != np.uint8
            or arrays[name].size == 0
            for name in (
                "projection_rows_canonical_json_utf8",
                "label_rows_canonical_json_utf8",
                "oof_rows_canonical_json_utf8",
            )
        )
        or arrays["feature_names"].tolist() != features
        or arrays["pair_uids"].shape != (row_count,)
        or arrays["world_uids"].shape != (row_count,)
        or len(set(arrays["pair_uids"].tolist())) != row_count
        or not np.all(np.isfinite(arrays["x"]))
        or not np.all(np.isfinite(arrays["bootstrap_statistics"]))
        or set(arrays["y"].tolist()) != {0, 1}
        or int(np.sum(arrays["y"])) != int(report["positive_count"])
        or int(shortcut_checkpoint.get("projection_row_count", -1))
        != row_count
        or int(shortcut_checkpoint.get("label_row_count", -1))
        != row_count
        or int(shortcut_checkpoint.get("oof_row_count", -1))
        != row_count
    ):
        raise common.ContractError(
            f"Exact-preflight array content drift: {split}"
        )
    world_uids = [str(value) for value in arrays["world_uids"].tolist()]
    ordered_worlds = sorted(set(world_uids), key=lambda value: value.encode("utf-8"))
    world_count = int(overlay["world_counts"][split])
    world_counts = Counter(world_uids)
    if (
        len(ordered_worlds) != world_count
        or set(world_counts.values()) != {40}
        or int(
            aggregate_checkpoint.get("audit", {}).get(
                "world_uid_count",
                -1,
            )
        )
        != world_count
        or int(
            aggregate_checkpoint.get("audit", {}).get(
                "seller_uid_count",
                -1,
            )
        )
        != world_count * int(overlay["sellers_per_world"])
        or int(
            aggregate_checkpoint.get("audit", {}).get(
                "pair_uid_count",
                -1,
            )
        )
        != world_count * int(overlay["complete_pairs_per_world"])
    ):
        raise common.ContractError(
            f"Exact-preflight array world partition drift: {split}"
        )
    expected_folds = shortcut_audit.assign_world_folds(
        world_uids,
        seed=int(overlay["shortcut_gate"]["fold_seed"]),
        fold_count=int(overlay["shortcut_gate"]["fold_count"]),
    )
    fold_vector = np.asarray(
        [expected_folds[value] for value in world_uids],
        dtype=np.int64,
    )
    if not np.array_equal(arrays["folds"], fold_vector):
        raise common.ContractError(
            f"Exact-preflight fold evidence drift: {split}"
        )
    canonical_payloads = {
        name: arrays[name].tobytes(order="C")
        for name in (
            "projection_rows_canonical_json_utf8",
            "label_rows_canonical_json_utf8",
            "oof_rows_canonical_json_utf8",
        )
    }
    try:
        projection_rows = json.loads(
            canonical_payloads[
                "projection_rows_canonical_json_utf8"
            ].decode("utf-8")
        )
        label_rows = json.loads(
            canonical_payloads[
                "label_rows_canonical_json_utf8"
            ].decode("utf-8")
        )
        oof_rows = json.loads(
            canonical_payloads[
                "oof_rows_canonical_json_utf8"
            ].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise common.ContractError(
            f"Exact-preflight canonical row evidence is invalid: {split}"
        ) from error
    if (
        common.canonical_json_bytes(projection_rows)
        != canonical_payloads["projection_rows_canonical_json_utf8"]
        or common.canonical_json_bytes(label_rows)
        != canonical_payloads["label_rows_canonical_json_utf8"]
        or common.canonical_json_bytes(oof_rows)
        != canonical_payloads["oof_rows_canonical_json_utf8"]
        or not isinstance(projection_rows, list)
        or not isinstance(label_rows, list)
        or not isinstance(oof_rows, list)
        or len(projection_rows) != row_count
        or len(label_rows) != row_count
        or len(oof_rows) != row_count
        or hashlib.sha256(
            canonical_payloads[
                "projection_rows_canonical_json_utf8"
            ]
        ).hexdigest()
        != shortcut_checkpoint.get("projection_rows_sha256")
        or hashlib.sha256(
            canonical_payloads["label_rows_canonical_json_utf8"]
        ).hexdigest()
        != shortcut_checkpoint.get("label_rows_sha256")
        or hashlib.sha256(
            canonical_payloads["oof_rows_canonical_json_utf8"]
        ).hexdigest()
        != shortcut_checkpoint.get("oof_rows_sha256")
    ):
        raise common.ContractError(
            f"Exact-preflight canonical row evidence drift: {split}"
        )
    projection_by_pair = {
        str(row.get("canonical_pair_uid", "")): row
        for row in projection_rows
        if isinstance(row, Mapping)
    }
    label_by_pair = {
        str(row.get("canonical_pair_uid", "")): row
        for row in label_rows
        if isinstance(row, Mapping)
    }
    pair_values = [
        str(value) for value in arrays["pair_uids"].tolist()
    ]
    expected_projection_keys = {
        "world_uid",
        "canonical_pair_uid",
        *features,
    }
    if (
        len(projection_by_pair) != row_count
        or len(label_by_pair) != row_count
        or set(projection_by_pair) != set(pair_values)
        or set(label_by_pair) != set(pair_values)
        or any(
            set(projection_by_pair[pair_uid])
            != expected_projection_keys
            or set(label_by_pair[pair_uid])
            != {"canonical_pair_uid", "label"}
            or str(
                projection_by_pair[pair_uid]["world_uid"]
            )
            != world_uids[index]
            or type(label_by_pair[pair_uid]["label"]) is not str
            or label_by_pair[pair_uid]["label"]
            != str(int(arrays["y"][index]))
            or any(
                not _close_float(
                    projection_by_pair[pair_uid][name],
                    arrays["x"][index, feature_index],
                )
                for feature_index, name in enumerate(features)
            )
            for index, pair_uid in enumerate(pair_values)
        )
    ):
        raise common.ContractError(
            f"Exact-preflight input row evidence drift: {split}"
        )
    shortcut = report["metadata_shortcut_audit"]
    scores_by_model = {
        model: arrays[f"score_{model}"]
        for model in shortcut_audit.MODEL_ORDER
    }
    point_statistic = 0.5
    for model, scores in scores_by_model.items():
        if (
            scores.shape != (row_count,)
            or scores.dtype != np.float64
            or not np.all(np.isfinite(scores))
        ):
            raise common.ContractError(
                f"Exact-preflight OOF evidence drift: {split}/{model}"
            )
        auc, symmetric = shortcut_audit.symmetric_auc(
            arrays["y"].astype(np.int64, copy=False),
            scores,
        )
        metrics = shortcut["model_metrics"][model]
        if (
            not _close_float(metrics["roc_auc"], auc)
            or not _close_float(
                metrics["roc_auc_symmetric"],
                symmetric,
            )
        ):
            raise common.ContractError(
                f"Exact-preflight OOF metric drift: {split}/{model}"
            )
        point_statistic = max(point_statistic, symmetric)
    expected_oof_keys = {
        "canonical_pair_uid",
        "world_uid",
        "label",
        "fold",
        "score_logistic_l2",
        "score_gradient_tree",
        "score_rbf_svm",
    }
    if any(
        not isinstance(row, Mapping)
        or set(row) != expected_oof_keys
        or row["canonical_pair_uid"] != pair_values[index]
        or row["world_uid"] != world_uids[index]
        or row["label"] != str(int(arrays["y"][index]))
        or row["fold"] != str(int(arrays["folds"][index]))
        or row["score_logistic_l2"]
        != format(arrays["score_logistic_l2"][index], ".17g")
        or row["score_gradient_tree"]
        != format(arrays["score_gradient_tree"][index], ".17g")
        or row["score_rbf_svm"]
        != format(arrays["score_rbf_svm"][index], ".17g")
        for index, row in enumerate(oof_rows)
    ):
        raise common.ContractError(
            f"Exact-preflight OOF row evidence drift: {split}"
        )

    replay_cache_key = common.canonical_sha256(
        {
            "version": (
                "2026-07-30-step28-v13-exact-preflight-"
                "full-replay-cache-v1"
            ),
            "split": split,
            "implementation_contract_sha256": (
                implementation_contract_sha256(overlay)
            ),
            "report": report,
            "checkpoint_artifacts": manifest["artifacts"],
        }
    )
    if replay_cache_key in _EXACT_PREFLIGHT_FULL_REPLAY_CACHE:
        return

    labels = arrays["y"].astype(np.int64, copy=False)
    (
        replayed_scores_by_model,
        replayed_folds,
        replayed_fold_audit,
    ) = shortcut_audit.compute_oof_scores(
        x=arrays["x"],
        y=labels,
        world_uids=world_uids,
        fold_by_world=expected_folds,
    )
    if (
        not np.array_equal(replayed_folds, arrays["folds"])
        or replayed_fold_audit != shortcut.get("fold_audit")
    ):
        raise common.ContractError(
            f"Exact-preflight frozen OOF fold replay drift: {split}"
        )
    for model in shortcut_audit.MODEL_ORDER:
        if not np.array_equal(
            replayed_scores_by_model[model],
            scores_by_model[model],
        ):
            raise common.ContractError(
                f"Exact-preflight frozen OOF score replay drift: "
                f"{split}/{model}"
            )

    bootstrap_statistics = arrays["bootstrap_statistics"]
    bootstrap_upper = float(
        np.quantile(bootstrap_statistics, 0.95, method="higher")
    )
    point_maximum = float(
        overlay["shortcut_gate"]["maximum_symmetric_auc"]
    )
    upper_maximum = float(
        overlay["shortcut_gate"][
            "maximum_world_bootstrap_95_upper"
        ]
    )
    if (
        not _close_float(
            shortcut["point_statistic_max_auc_symmetric"],
            point_statistic,
        )
        or not _close_float(
            shortcut["bootstrap_95_upper"],
            bootstrap_upper,
        )
        or point_statistic > point_maximum
        or bootstrap_upper > upper_maximum
        or shortcut.get("point_gate_pass") is not True
        or shortcut.get("bootstrap_upper_gate_pass") is not True
        or shortcut.get("status") != "PASS_METADATA_SHORTCUT_ONLY"
    ):
        raise common.ContractError(
            f"Exact-preflight shortcut result drift: {split}"
        )
    replayed_bootstrap_statistics: np.ndarray | None = None

    def capture_replayed_bootstrap(values: np.ndarray) -> None:
        nonlocal replayed_bootstrap_statistics
        if replayed_bootstrap_statistics is not None:
            raise common.ContractError(
                "Exact-preflight bootstrap replay emitted twice"
            )
        replayed_bootstrap_statistics = np.asarray(values).copy()

    replayed_upper, replayed_draw_sha256 = (
        shortcut_audit.world_bootstrap_upper(
            y=labels,
            world_uids=world_uids,
            score_by_model=replayed_scores_by_model,
            split=split,
            replicates=bootstrap_replicates,
            base_seed=int(
                overlay["shortcut_gate"]["bootstrap_base_seed"]
            ),
            statistics_sink=capture_replayed_bootstrap,
        )
    )
    if (
        replayed_bootstrap_statistics is None
        or not np.array_equal(
            replayed_bootstrap_statistics,
            bootstrap_statistics,
        )
        or not _close_float(replayed_upper, bootstrap_upper)
        or replayed_draw_sha256
        != shortcut.get("bootstrap_draw_matrix_sha256")
    ):
        raise common.ContractError(
            f"Exact-preflight full bootstrap replay drift: {split}"
        )
    _EXACT_PREFLIGHT_FULL_REPLAY_CACHE.add(replay_cache_key)


def _validate_exact_preflight_registry(
    overlay: Mapping[str, Any],
    *,
    require_complete: bool,
) -> None:
    registry = overlay.get("exact_implementation_preflights")
    if not isinstance(registry, Mapping):
        raise common.ContractError("Exact-preflight registry is absent")
    if not registry:
        if require_complete:
            raise common.ContractError(
                "Exact-preflight registry is incomplete"
            )
        return
    if set(registry) != set(SPLITS):
        raise common.ContractError("Exact-preflight split set drift")
    implementation_hash = implementation_contract_sha256(overlay)
    expected_builder_hash = overlay["dataset_builder"]["sha256"]
    expected_closure_hash = overlay["dataset_builder"][
        "implementation_closure"
    ]["canonical_sha256"]
    expected_candidate_hash = overlay["candidate_design"]["sha256"]
    expected_base_hash = overlay["base_policy"]["sha256"]
    expected_preflight_hash = overlay["release_tools"][
        "exact_preflight"
    ]["sha256"]
    for split in SPLITS:
        spec = registry[split]
        if not isinstance(spec, Mapping) or set(spec) != {
            "path",
            "sha256",
        }:
            raise common.ContractError(
                f"Exact-preflight pin schema drift: {split}"
            )
        report = common.load_json(
            _verify_pin(spec, label=f"exact preflight {split}")
        )
        report_path = common.repo_path(str(spec["path"]))
        _validate_canonical_self_hash(
            report,
            label=f"Exact-preflight report {split}",
        )
        shortcut = report.get("metadata_shortcut_audit", {})
        identity33 = report.get("identity33_matrix_audit", {})
        world_count = int(overlay["world_counts"][split])
        positive_count = world_count * int(
            overlay["classification_positive_count_per_world"][split]
        )
        if (
            report.get("status")
            != "PASS_EXACT_IMPLEMENTATION_DESIGN_PREFLIGHT"
            or report.get("mode") != MODE
            or report.get("split") != split
            or int(report.get("world_count", -1)) != world_count
            or int(report.get("candidate_count", -1))
            != world_count * 40
            or int(report.get("positive_count", -1)) != positive_count
            or int(report.get("negative_count", -1))
            != world_count * 40 - positive_count
            or report.get("formal_public_randomness_consumed") is not True
            or report.get("formal_private_structure_key_created_or_read")
            is not False
            or report.get("design_only_structure_key_used") is not True
            or report.get("final_release_status_granted") is not False
            or report.get("all_worlds_independent_dgp_replay_exact")
            is not True
            or report.get("all_worlds_mechanism_coverage_exact")
            is not True
            or report.get("label_formula_rowwise_exact") is not True
            or report.get("aggregate_lineage_exact") is not True
            or report.get("builder_implementation_sha256")
            != expected_builder_hash
            or report.get("builder_source_closure_sha256")
            != expected_closure_hash
            or report.get("candidate_implementation_sha256")
            != expected_candidate_hash
            or report.get("base_policy_sha256") != expected_base_hash
            or report.get("exact_preflight_implementation_sha256")
            != expected_preflight_hash
            or report.get("implementation_contract_sha256")
            != implementation_hash
            or report.get("checkpointing_enabled") is not True
            or report.get("checkpoint_manifest") is None
            or report.get("failure_diagnostics_deferred") is not False
            or report.get("metadata_shortcut_failure_diagnostics")
            is not None
            or shortcut.get("status") != "PASS_METADATA_SHORTCUT_ONLY"
            or shortcut.get("point_gate_pass") is not True
            or shortcut.get("bootstrap_upper_gate_pass") is not True
            or int(shortcut.get("bootstrap_replicates", -1)) != 9999
            or int(identity33.get("row_count", -1))
            != world_count * 378
            or int(identity33.get("feature_count", -1)) != 33
            or (
                split == "train"
                and (
                    identity33.get("no_all_zero_columns_required")
                    is not True
                    or identity33.get("no_all_zero_columns_gate_pass")
                    is not True
                )
            )
            or (
                split != "train"
                and identity33.get("no_all_zero_columns_required")
                is not False
            )
        ):
            raise common.ContractError(
                f"Exact-preflight report content drift: {split}"
            )
        _validate_exact_preflight_checkpoint_evidence(
            report=report,
            report_path=report_path,
            split=split,
            overlay=overlay,
        )


def load_overlay(
    path: Path,
    *,
    require_generation_frozen: bool,
) -> dict[str, Any]:
    overlay = common.load_json(path)
    expected_world_counts = {
        "train": 500,
        "development": 500,
        "audit_a": 500,
        "audit_b": 500,
    }
    expected_positive_counts = {
        "train": 16,
        "development": 10,
        "audit_a": 10,
        "audit_b": 10,
    }
    if (
        overlay.get("run_id") != "v13_training_ready_v1_20260729"
        or overlay.get("output_root")
        != (
            "reports/step28_synthetic_chinese_dataset/"
            "v13_training_ready_v1_20260729"
        )
        or tuple(overlay.get("world_counts", {})) != SPLITS
        or overlay.get("world_counts") != expected_world_counts
        or int(overlay.get("sellers_per_world", -1)) != 28
        or int(overlay.get("complete_pairs_per_world", -1)) != 378
        or int(overlay.get("classification_pairs_per_world", -1)) != 40
        or overlay.get("handle_encoding") != "parser_safe_hex_v2"
        or overlay.get("base_execution_mode") != MODE
        or overlay.get("claim_level")
        is not None
        or overlay.get("target_release_claim_level")
        != (
            "TRAINING_AND_FIXED_HOLDOUT_BYTES_READY_"
            "NOT_BLIND_CUSTODY_ATTESTED"
        )
        or overlay.get("current_readiness_semantics")
        != (
            "status and generation_enabled describe current readiness; "
            "target_release_claim_level applies only to the final "
            "manifest after all split releases validate"
        )
    ):
        raise common.ContractError("Training-ready overlay shape drift")
    if (
        overlay.get("status")
        not in {
            "IMPLEMENTATION_LOCK_IN_PROGRESS",
            "READY_FOR_KEY_CEREMONY",
            "FROZEN_READY_FOR_GENERATION",
        }
        or (
            overlay.get("generation_enabled") is True
        )
        != (
            overlay.get("status") == "FROZEN_READY_FOR_GENERATION"
        )
    ):
        raise common.ContractError(
            "Training-ready overlay lifecycle status drift"
        )
    for split in SPLITS:
        if int(
            overlay["classification_positive_count_per_world"][split]
        ) != expected_positive_counts[split] or int(
            overlay["classification_positive_count_per_world"][split]
        ) != int(mechanism_c40.POSITIVE_COUNT_BY_SPLIT[split]):
            raise common.ContractError("C40 class budget drift")
    if (
        int(overlay.get("positive_mechanism_count_per_world", -1)) != 8
        or int(overlay.get("negative_flag_count_per_world", -1)) != 7
        or overlay.get("public_randomness_source")
        != "base_policy.randomness.formal"
        or overlay.get("release_status_required")
        != "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
        or overlay.get("retrieval")
        != {
            "enabled_splits": ["audit_a", "audit_b"],
            "queries_per_world": 4,
            "gallery_per_query": 27,
            "query_selection": (
                "full HMAC-SHA256 under the registered query key, "
                "then seller UID UTF-8"
            ),
            "relevance_formula": (
                "int(controller(query)==controller(gallery))"
            ),
        }
        or overlay.get("supervision")
        != {
            "classification_formula": (
                "int(controller(left)==controller(right))"
            ),
            "open_splits": ["train", "development"],
            "sealed_splits": ["audit_a", "audit_b"],
        }
        or overlay.get("m1")
        != {
            "replicates": 5,
            "source": "base_policy.randomness.formal.rewire_key_hexes",
            "endpoint_disjoint": True,
            "whole_33_vector_derangement": True,
            "stratification": ["world_uid", "C40_or_complement"],
        }
    ):
        raise common.ContractError(
            "Training-ready scientific release semantics drift"
        )
    _verify_pin(overlay["base_policy"], label="base policy")
    _verify_pin(overlay["release_contract"], label="release contract")
    for name, spec in overlay["design_diagnostics"].items():
        _verify_pin(spec, label=f"design diagnostic {name}")
        required_status = spec.get("status")
        if required_status is not None:
            document = common.load_json(common.repo_path(spec["path"]))
            if document.get("status") != required_status:
                raise common.ContractError(
                    f"Design diagnostic status drift: {name}"
                )
    _verify_pin(
        overlay["identity_value_salts"]["artifact"],
        label="training-ready identity salts",
    )
    _verify_pin(
        overlay["identity_value_salts"]["builder"],
        label="training-ready identity salt builder",
    )
    salt_artifact = common.load_json(
        common.repo_path(
            overlay["identity_value_salts"]["artifact"]["path"]
        )
    )
    expected_candidate_count = (
        sum(int(value) for value in overlay["world_counts"].values()) * 96
    )
    if (
        salt_artifact.get("status") != "PASS_BOUNDARY_ONLY"
        or int(salt_artifact.get("candidate_count_per_type", -1))
        != expected_candidate_count
        or salt_artifact.get("salt_counters")
        != overlay["identity_value_salts"]["per_type_salt_counters"]
        or int(salt_artifact.get("deny_intersection_count", -1)) != 0
        or int(
            salt_artifact.get(
                "same_mode_cross_type_intersection_count", -1
            )
        )
        != 0
    ):
        raise common.ContractError("Training-ready identity salt gate failed")
    implementation = overlay["candidate_design"]
    _verify_pin(
        {
            "path": implementation["implementation"],
            "sha256": implementation["sha256"],
        },
        label="mechanism C40 implementation",
    )
    _validate_dataset_builder_closure(overlay)
    shortcut = overlay.get("shortcut_gate")
    if (
        not isinstance(shortcut, Mapping)
        or int(shortcut.get("features", -1)) != 14
        or float(shortcut.get("maximum_symmetric_auc", -1.0)) != 0.52
        or float(
            shortcut.get("maximum_world_bootstrap_95_upper", -1.0)
        )
        != 0.53
        or int(shortcut.get("bootstrap_replicates", -1)) != 9999
        or int(shortcut.get("fold_count", -1)) != 5
        or int(shortcut.get("fold_seed", -1)) != 2026072707
        or int(shortcut.get("bootstrap_base_seed", -1)) != 2026072711
        or shortcut.get("model_order")
        != ["logistic_l2", "gradient_tree", "rbf_svm"]
    ):
        raise common.ContractError("Training-ready shortcut gate drift")
    closure = shortcut.get("implementation_closure")
    if not isinstance(closure, Mapping) or set(closure) != {
        "nuisance_projector",
        "audit_runner",
        "audit_common",
        "label_sealer",
    }:
        raise common.ContractError(
            "Training-ready shortcut implementation closure drift"
        )
    for name, spec in closure.items():
        _verify_pin(spec, label=f"shortcut implementation {name}")
    release_tools = overlay.get("release_tools")
    if not isinstance(release_tools, Mapping) or set(release_tools) != {
        "exact_preflight",
        "key_initializer",
        "finalizer",
        "model_input_validator",
    }:
        raise common.ContractError(
            "Training-ready release-tool closure drift"
        )
    for name, spec in release_tools.items():
        _verify_pin(spec, label=f"release tool {name}")
    _validate_scientific_contract(overlay)
    _validate_implementation_contract(overlay)
    _validate_exact_preflight_registry(
        overlay,
        require_complete=(
            overlay.get("status") != "IMPLEMENTATION_LOCK_IN_PROGRESS"
        ),
    )
    _require_shortcut_environment(overlay)
    if require_generation_frozen:
        if (
            overlay.get("status") != "FROZEN_READY_FOR_GENERATION"
            or overlay.get("generation_enabled") is not True
            or overlay.get("self_sha256") != _canonical_self_hash(overlay)
        ):
            raise common.ContractError(
                "Training-ready generation is not frozen/enabled"
            )
        custody = overlay["private_structure_key_custody"]
        commitments = custody["commitments"]
        if (
            any(
                not isinstance(commitments.get(split), str)
                or len(commitments[split]) != 64
                for split in SPLITS
            )
            or len(set(commitments.values())) != 4
            or not isinstance(custody.get("ceremony_receipt"), Mapping)
        ):
            raise common.ContractError("Structure-key ceremony is incomplete")
        _verify_pin(
            custody["ceremony_receipt"],
            label="structure-key ceremony receipt",
        )
        ceremony = common.load_json(
            common.repo_path(custody["ceremony_receipt"]["path"])
        )
        without_self = dict(ceremony)
        ceremony_self = without_self.pop("canonical_self_hash", None)
        if (
            ceremony.get("status")
            != "PASS_SPLIT_PRIVATE_KEY_CEREMONY"
            or ceremony.get("run_id") != overlay["run_id"]
            or ceremony.get("commitments") != commitments
            or ceremony.get("commitments_unique") is not True
            or int(
                ceremony.get(
                    "forbidden_commitment_intersection_count", -1
                )
            )
            != 0
            or ceremony.get("one_split_key_per_file") is not True
            or ceremony.get("raw_structure_keys_serialized") is not False
            or ceremony.get("os_custody_attested") is not False
            or ceremony_self != common.canonical_sha256(without_self)
        ):
            raise common.ContractError(
                "Structure-key ceremony receipt content drift"
            )
    return overlay


def _require_shortcut_environment(
    overlay: Mapping[str, Any],
) -> None:
    environment = overlay["shortcut_gate"]["environment"]
    if (
        environment
        != {
            "numpy_version": "2.2.6",
            "scikit_learn_version": "1.7.2",
            "bit_generator": "PCG64DXSM",
        }
        or np.__version__ != environment["numpy_version"]
        or shortcut_audit.sklearn.__version__
        != environment["scikit_learn_version"]
        or np.random.PCG64DXSM.__name__
        != environment["bit_generator"]
    ):
        raise common.ContractError(
            "Training-ready shortcut-audit environment drift"
        )


def _load_pinned_base(
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    path = _verify_pin(overlay["base_policy"], label="base policy")
    return common.load_policy(path, mode="development_smoke")


def _execution_policy(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    *,
    structure_key_hex: str,
) -> dict[str, Any]:
    """Alias audited workers while making every data stream formal and fresh."""

    policy = copy.deepcopy(base)
    formal = copy.deepcopy(policy["randomness"]["formal"])
    policy["randomness"][MODE] = {
        field: copy.deepcopy(formal[field])
        for field in PUBLIC_RANDOM_FIELDS
    }
    policy["randomness"][MODE]["structure_key_hex"] = structure_key_hex
    counts = {
        split: int(overlay["world_counts"][split])
        for split in SPLITS
    }
    policy["modes"][MODE] = copy.deepcopy(policy["modes"]["formal"])
    policy["modes"][MODE]["run_id"] = str(overlay["run_id"])
    policy["modes"][MODE]["output_root"] = str(overlay["output_root"])
    policy["modes"][MODE]["world_counts"] = counts
    policy["modes"][MODE][
        "source_dataset_prefix"
    ] = "step28_v13_training_ready"
    value_generation = policy["identity_design"][
        "identity_value_generation"
    ]
    value_generation["handle_encoding_by_mode"][MODE] = str(
        overlay["handle_encoding"]
    )
    value_generation["salt_selection"][
        "training_ready_per_type_salt_counters"
    ] = dict(
        overlay["identity_value_salts"]["per_type_salt_counters"]
    )
    common.validate_policy(policy, mode=MODE)
    return policy


def _load_split_key(
    overlay: Mapping[str, Any],
    *,
    split: str,
) -> str:
    custody = overlay["private_structure_key_custody"]
    directory = common.repo_path(str(custody["key_directory"]))
    filename = str(custody["key_filename_pattern"]).format(split=split)
    path = (directory / filename).resolve()
    if (
        path.parent != directory.resolve()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise common.ContractError(
            f"Missing split-private structure key: {split}"
        )
    document = common.load_json(path)
    if (
        set(document)
        != {"version", "run_id", "split", "key_hex", "sha256_commitment"}
        or document["run_id"] != overlay["run_id"]
        or document["split"] != split
    ):
        raise common.ContractError("Split-private key file schema drift")
    key_hex = str(document["key_hex"])
    try:
        raw = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise common.ContractError("Split-private key is not hex") from exc
    commitment = common.sha256_bytes(raw)
    if (
        len(raw) != 32
        or key_hex != key_hex.lower()
        or commitment != document["sha256_commitment"]
        or commitment
        != overlay["private_structure_key_custody"]["commitments"][split]
    ):
        raise common.ContractError("Split-private key commitment mismatch")
    return key_hex


def _extend_world(
    world_uid: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        if "world_uid" in source:
            raise common.ContractError("Cannot duplicate world_uid")
        output.append({"world_uid": world_uid, **dict(source)})
    return output


def _empty_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        name: []
        for name in (
            "worlds",
            "sellers",
            "raw_items",
            "complete_model_pair_endpoints",
            "candidate_pairs",
            "candidate_sampling_audit",
            "seller_profiles",
            "redacted_items",
            "history_safe_occurrences",
            "history_item_index",
            "history_projection_attestations",
            "identity33_all_pairs",
            "controller_membership",
            "controller_style_groups",
            "mechanism_assignments",
            "identity_assets",
            "positive_targets",
            "negative_flags",
            "parsed_identity_occurrences",
            "identity_slots_audit",
            "identity_slots_edit",
            "noise_slots_audit",
            "render_asts",
            "override_audit",
            "redaction_diagnostics",
            "world_generation_audit",
            "classification_labels",
            "null_nuisance_projection",
            "shortcut_oof",
            "retrieval_queries",
            "retrieval_relations",
            "retrieval_qrels",
        )
    }


def _history_item_index(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": int(row["time_bucket"]),
        }
        for row in rows
    ]
    output.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
        )
    )
    return output


def _one_world(
    *,
    policy: dict[str, Any],
    split: str,
    record: Mapping[str, Any],
    structure_key_hex: str,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    style_profile: Mapping[str, Any],
) -> dict[str, Any]:
    world_uid = str(record["world_uid"])
    world = world_builder.build_world(
        policy=policy,
        template=dict(template),
        fixture=dict(fixture),
        style_profile=dict(style_profile),
        mode=MODE,
        world_record=record,
        structure_key_hex=structure_key_hex,
    )
    public = world["public"]
    private = world["private"]

    uid_pools = independent_comparator.build_observed_uid_pools(
        world_uid=world_uid,
        sellers=public["sellers"],
        items=public["items"],
    )
    replay = independent_replay.replay_typed_dgp(
        policy,
        mode=MODE,
        split=split,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        **uid_pools,
    )
    projection = producer_projection.project_world(
        world=world,
        mode=MODE,
        split=split,
    )
    replay_audit = independent_comparator.compare_typed_dgp(
        expected_replay=replay,
        producer_projection=projection,
    )
    regeneration_audit = regeneration.validate_producer_regeneration_match(
        policy,
        mode=MODE,
        split=split,
        template=template,
        fixture=fixture,
        style_profile=style_profile,
        world=world,
    )
    processed = production.process_world(
        policy,
        mode=MODE,
        split=split,
        template=template,
        world=world,
    )
    item_index = _history_item_index(public["items"])
    profiles, profile_audit = profiles_mod.build_world_profiles(
        policy,
        mode=MODE,
        split=split,
        sellers=public["sellers"],
        items=processed["public"]["profile_safe_items"],
    )
    attestation = production.build_history_projection_attestation(
        policy,
        mode=MODE,
        split=split,
        world_uid=world_uid,
        sellers=public["sellers"],
        items=public["items"],
        history_safe_occurrences=processed["public"][
            "history_safe_occurrences"
        ],
        history_item_index=item_index,
        parsed_rows=processed["private"]["parsed_identity_occurrences"],
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    identity33, identity33_audit = (
        history_features.build_identity33_all_pairs(
            policy,
            mode=MODE,
            split=split,
            history_safe_occurrences=processed["public"][
                "history_safe_occurrences"
            ],
            history_item_index=item_index,
            projection_attestations=[attestation],
            complete_model_pair_endpoints=public[
                "complete_model_pair_endpoints"
            ],
        )
    )
    candidates, candidate_audit, candidate_summary = (
        mechanism_c40.build_world_c40(
            split=split,
            candidate_key_hex=str(
                policy["randomness"][MODE]["candidate_key_hex"]
            ),
            world_uid=world_uid,
            complete_pair_endpoints=public[
                "complete_model_pair_endpoints"
            ],
            controller_membership=private["controller_membership"],
            positive_targets=private["positive_targets"],
            negative_flags=private["negative_flags"],
        )
    )
    return {
        "world": world,
        "processed": processed,
        "history_item_index": item_index,
        "profiles": profiles,
        "history_attestation": attestation,
        "identity33": identity33,
        "candidate_pairs": candidates,
        "candidate_audit": candidate_audit,
        "world_audit": {
            "world_uid": world_uid,
            "split": split,
            "mode_global_ordinal": int(record["mode_global_ordinal"]),
            "split_ordinal": int(record["split_ordinal"]),
            "profile_audit": profile_audit,
            "identity33_audit": identity33_audit,
            "candidate_summary": candidate_summary,
            "parser_audit": processed["private"][
                "parser_structural_audit"
            ],
            "redaction_audit": processed["private"][
                "redaction_structural_audit"
            ],
            "independent_typed_dgp_replay_audit": replay_audit,
            "producer_regeneration_audit": regeneration_audit,
            "world_public_sha256": common.canonical_sha256(public),
            "world_private_sha256": common.canonical_sha256(private),
        },
    }


def _append_world(
    payload: dict[str, list[dict[str, Any]]],
    built: Mapping[str, Any],
) -> None:
    world = built["world"]
    public = world["public"]
    private = world["private"]
    processed = built["processed"]
    world_uid = str(public["world"]["world_uid"])
    payload["worlds"].append(dict(public["world"]))
    payload["sellers"].extend(dict(row) for row in public["sellers"])
    payload["raw_items"].extend(dict(row) for row in public["items"])
    payload["complete_model_pair_endpoints"].extend(
        dict(row) for row in public["complete_model_pair_endpoints"]
    )
    payload["candidate_pairs"].extend(
        dict(row) for row in built["candidate_pairs"]
    )
    payload["candidate_sampling_audit"].extend(
        dict(row) for row in built["candidate_audit"]
    )
    payload["seller_profiles"].extend(
        {"world_uid": world_uid, **dict(row)}
        for row in built["profiles"]
    )
    payload["redacted_items"].extend(
        dict(row) for row in processed["public"]["redacted_items"]
    )
    payload["history_safe_occurrences"].extend(
        dict(row)
        for row in processed["public"]["history_safe_occurrences"]
    )
    payload["history_item_index"].extend(
        dict(row) for row in built["history_item_index"]
    )
    payload["history_projection_attestations"].append(
        dict(built["history_attestation"])
    )
    payload["identity33_all_pairs"].extend(
        dict(row) for row in built["identity33"]
    )
    for name in (
        "controller_membership",
        "controller_style_groups",
        "mechanism_assignments",
        "identity_slots_audit",
        "identity_slots_edit",
        "noise_slots_audit",
        "render_asts",
        "override_audit",
    ):
        payload[name].extend(dict(row) for row in private[name])
    payload["identity_assets"].extend(
        _extend_world(world_uid, private["identity_assets"])
    )
    payload["positive_targets"].extend(
        _extend_world(world_uid, private["positive_targets"])
    )
    payload["negative_flags"].extend(
        {
            "world_uid": world_uid,
            "canonical_pair_uid": str(row["canonical_pair_uid"]),
            "flag": str(row["flag"]),
            "asset_index": int(row["asset_index"]),
        }
        for row in private["negative_flags"]
    )
    payload["parsed_identity_occurrences"].extend(
        dict(row)
        for row in processed["private"]["parsed_identity_occurrences"]
    )
    payload["redaction_diagnostics"].extend(
        dict(row)
        for row in processed["private"]["redaction_diagnostics"]
    )
    payload["world_generation_audit"].append(dict(built["world_audit"]))


def _validate_formula_independently(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(
        tuple(row) != mechanism_c40.SAFE_FIELDS
        for row in candidate_rows
    ):
        raise common.ContractError(
            "Independent label validator candidate schema drift"
        )
    if any(
        set(row) != mechanism_c40.MEMBERSHIP_FIELDS
        for row in membership_rows
    ):
        raise common.ContractError(
            "Independent label validator membership schema drift"
        )
    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    controller_by_seller: dict[tuple[str, str], str] = {}
    for row in membership_rows:
        world_uid = str(row["world_uid"])
        controller_uid = str(row["controller_uid"])
        seller_uid = str(row["seller_uid"])
        seller_key = (world_uid, seller_uid)
        if seller_key in controller_by_seller:
            raise common.ContractError(
                "Independent label validator membership duplicate"
            )
        controller_by_seller[seller_key] = controller_uid
        members[(world_uid, controller_uid)].append(seller_uid)
    positive_universe = {
        common.canonical_pair_uid(left, right)
        for seller_uids in members.values()
        for left, right in itertools.combinations(
            common.utf8_sort(seller_uids), 2
        )
    }
    candidate_uids: set[str] = set()
    expected_by_pair: dict[str, str] = {}
    for row in candidate_rows:
        pair_uid = str(row["canonical_pair_uid"])
        world_uid = str(row["world_uid"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        if (
            pair_uid in candidate_uids
            or (world_uid, left) not in controller_by_seller
            or (world_uid, right) not in controller_by_seller
            or pair_uid != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError(
                "Independent label validator candidate lineage drift"
            )
        candidate_uids.add(pair_uid)
        expected_by_pair[pair_uid] = str(
            int(
                controller_by_seller[(world_uid, left)]
                == controller_by_seller[(world_uid, right)]
            )
        )
    independently_positive = positive_universe & candidate_uids
    observed_by_pair: dict[str, str] = {}
    for row in labels:
        if (
            tuple(row) != ("canonical_pair_uid", "label")
            or type(row["canonical_pair_uid"]) is not str
            or type(row["label"]) is not str
            or row["label"] not in {"0", "1"}
            or row["canonical_pair_uid"] in observed_by_pair
        ):
            raise common.ContractError(
                "Independent label validator output schema/domain drift"
            )
        observed_by_pair[row["canonical_pair_uid"]] = row["label"]
    labeled_positive = {
        pair_uid
        for pair_uid, label in observed_by_pair.items()
        if label == "1"
    }
    if (
        independently_positive != labeled_positive
        or observed_by_pair != expected_by_pair
        or len(candidate_uids) != len(candidate_rows)
    ):
        raise common.ContractError("Independent label formula replay failed")
    return {
        "formula": "int(controller(left)==controller(right))",
        "alternative_derivation": (
            "enumerate all within-controller combinations then intersect C40"
        ),
        "candidate_count": len(candidate_uids),
        "positive_count": len(independently_positive),
        "exact_rowwise_equal": True,
        "candidate_keyset_sha256": common.canonical_sha256(
            common.utf8_sort(candidate_uids)
        ),
        "positive_keyset_sha256": common.canonical_sha256(
            common.utf8_sort(independently_positive)
        ),
    }


def _identity33_matrix_audit(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    require_no_zero_columns: bool,
) -> dict[str, Any]:
    names = list(policy["history_features"]["feature_names"])
    if not rows:
        raise common.ContractError("Identity33 aggregate matrix is empty")
    matrix = np.asarray(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if (
        matrix.shape != (len(rows), 33)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
    ):
        raise common.ContractError("Identity33 aggregate matrix is invalid")
    minima = np.min(matrix, axis=0)
    maxima = np.max(matrix, axis=0)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    zero_columns = [
        names[index]
        for index in range(33)
        if maxima[index] == 0.0
    ]
    duplicate_groups: list[list[str]] = []
    grouped: dict[bytes, list[str]] = defaultdict(list)
    for index, name in enumerate(names):
        grouped[matrix[:, index].tobytes()].append(name)
    for group in grouped.values():
        if len(group) > 1:
            duplicate_groups.append(group)
    nonconstant = stds > 0.0
    if np.any(nonconstant):
        standardized = (
            (matrix[:, nonconstant] - means[nonconstant])
            / stds[nonconstant]
        )
        singular = np.linalg.svd(
            standardized,
            full_matrices=False,
            compute_uv=False,
        )
        tolerance = (
            np.max(standardized.shape)
            * np.finfo(np.float64).eps
            * singular[0]
        )
        rank = int(np.sum(singular > tolerance))
        condition = (
            float(singular[0] / singular[-1])
            if singular[-1] > 0.0
            else None
        )
    else:
        rank = 0
        condition = None
    if zero_columns and require_no_zero_columns:
        raise common.ContractError(
            f"Identity33 has all-zero columns: {zero_columns}"
        )
    return {
        "row_count": int(matrix.shape[0]),
        "feature_count": 33,
        "all_zero_columns": zero_columns,
        "no_all_zero_columns_required": require_no_zero_columns,
        "no_all_zero_columns_gate_pass": (
            not zero_columns or not require_no_zero_columns
        ),
        "duplicate_column_groups": duplicate_groups,
        "nonconstant_standardized_rank": rank,
        "nonconstant_feature_count": int(np.sum(nonconstant)),
        "standardized_condition_number": condition,
        "all_zero_row_count": int(np.sum(np.all(matrix == 0.0, axis=1))),
        "features": {
            name: {
                "minimum": float(minima[index]),
                "maximum": float(maxima[index]),
                "mean": float(means[index]),
                "standard_deviation": float(stds[index]),
            }
            for index, name in enumerate(names)
        },
    }


def _expected_graph_counts(
    policy: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, int]:
    """Derive graph-specific asset and negative-flag counts from policy."""

    graph_name = str(
        policy["identity_design"]["mechanism_by_split"][split]
    )
    mechanism_counts = policy["identity_design"][
        "mechanism_assignments"
    ][graph_name]
    mechanism_generators = policy["identity_design"][
        "mechanism_generators"
    ]
    if sum(int(value) for value in mechanism_counts.values()) != 12:
        raise common.ContractError(
            "Graph mechanism assignments do not cover 12 controllers"
        )
    positive_assets = 0
    for mechanism, count in mechanism_counts.items():
        generator = mechanism_generators[mechanism]
        declared = [
            int(generator[name])
            for name in ("identity_count", "mechanism_identity_count")
            if name in generator
        ]
        if len(declared) != 1 or declared[0] < 0:
            raise common.ContractError(
                "Mechanism identity-asset count contract is invalid"
            )
        positive_assets += int(count) * declared[0]

    dgp = policy["identity_design"]["hard_negative_dgp"][graph_name]
    support_degrees = [int(value) for value in dgp["support_hub_degrees"]]
    direct_degrees = [
        int(value)
        for value in dgp["high_frequency_direct_hub_degrees"]
    ]
    risky_count = int(dgp["risky_shared_token_count"])
    risky_degree = int(dgp["risky_shared_token_degree"])
    private_count = int(dgp["private_collision_edges"])
    false_rotation_count = int(dgp["false_rotation_paths"])
    exact_clone_count = int(
        dgp["cross_controller_exact_title_clone_pairs"]
    )
    semantic_count = int(
        dgp["cross_controller_high_semantic_similarity_pairs"]
    )
    if (
        not support_degrees
        or not direct_degrees
        or any(value < 2 for value in support_degrees + direct_degrees)
        or min(
            risky_count,
            risky_degree,
            private_count,
            false_rotation_count,
            exact_clone_count,
            semantic_count,
        )
        < 1
    ):
        raise common.ContractError("Hard-negative graph count contract is invalid")

    negative_assets = (
        len(support_degrees)
        + len(direct_degrees)
        + risky_count
        + private_count
        + 2 * false_rotation_count
    )
    pair_count = lambda degree: degree * (degree - 1) // 2
    negative_flags = (
        sum(pair_count(degree) for degree in support_degrees)
        + sum(pair_count(degree) for degree in direct_degrees)
        + risky_count * pair_count(risky_degree)
        + private_count
        + false_rotation_count
        + exact_clone_count
        + semantic_count
    )
    background_assets = int(
        policy["identity_design"]["background_private_scaffold"][
            "edge_count_per_world"
        ]
    )
    output = {
        "identity_assets": (
            background_assets + positive_assets + negative_assets
        ),
        "negative_flags": negative_flags,
    }
    expected_by_graph = {
        "G_A": {"identity_assets": 84, "negative_flags": 42},
        "G_B": {"identity_assets": 89, "negative_flags": 100},
    }
    if output != expected_by_graph.get(graph_name):
        raise common.ContractError(
            "Derived graph counts differ from the registered G_A/G_B design"
        )
    return output


def _validate_payload(
    policy: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    split: str,
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Close aggregate row counts, keys, lineage and identity-value replay."""

    world_count = int(overlay["world_counts"][split])
    graph_counts = _expected_graph_counts(policy, split=split)
    expected_counts = {
        "worlds": world_count,
        "sellers": world_count * 28,
        "complete_model_pair_endpoints": world_count * 378,
        "candidate_pairs": world_count * 40,
        "candidate_sampling_audit": world_count * 40,
        "seller_profiles": world_count * 28,
        "history_projection_attestations": world_count,
        "identity33_all_pairs": world_count * 378,
        "controller_membership": world_count * 28,
        "controller_style_groups": world_count * 12,
        "mechanism_assignments": world_count * 12,
        "identity_assets": world_count * graph_counts["identity_assets"],
        "positive_targets": world_count * 12,
        "negative_flags": world_count * graph_counts["negative_flags"],
        "override_audit": world_count * 6,
        "world_generation_audit": world_count,
        "classification_labels": world_count * 40,
        "null_nuisance_projection": world_count * 40,
        "shortcut_oof": world_count * 40,
    }
    for name, expected in expected_counts.items():
        if len(payload[name]) != expected:
            raise common.ContractError(
                f"Training-ready aggregate count drift: {name}"
            )
    if (
        len(payload["raw_items"]) != len(payload["redacted_items"])
        or len(payload["raw_items"]) != len(payload["history_item_index"])
        or not payload["identity_assets"]
        or not payload["history_safe_occurrences"]
        or len(payload["parsed_identity_occurrences"])
        != len(payload["history_safe_occurrences"])
    ):
        raise common.ContractError("Training-ready item/history count drift")

    world_uids = {str(row["world_uid"]) for row in payload["worlds"]}
    if len(world_uids) != world_count:
        raise common.ContractError("World UID uniqueness failed")
    seller_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]))
        for row in payload["sellers"]
    }
    seller_uids = {seller_uid for _world_uid, seller_uid in seller_keys}
    if (
        len(seller_keys) != world_count * 28
        or len(seller_uids) != len(seller_keys)
        or {world_uid for world_uid, _seller_uid in seller_keys}
        != world_uids
    ):
        raise common.ContractError("Seller UID/key closure failed")

    item_keys: set[tuple[str, str, str]] = set()
    item_uids: set[str] = set()
    item_count_by_seller: Counter[tuple[str, str]] = Counter()
    for row in payload["raw_items"]:
        key = (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        if (
            key in item_keys
            or key[2] in item_uids
            or key[:2] not in seller_keys
        ):
            raise common.ContractError("Raw item key/foreign-key drift")
        item_keys.add(key)
        item_uids.add(key[2])
        item_count_by_seller[key[:2]] += 1
    if (
        set(item_count_by_seller) != seller_keys
        or any(
            not 2 <= count <= 8
            for count in item_count_by_seller.values()
        )
    ):
        raise common.ContractError("Seller item cardinality drift")
    redacted_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in payload["redacted_items"]
    }
    history_item_keys = {
        (
            str(row["world_uid"]),
            str(row["seller_uid"]),
            str(row["item_uid"]),
        )
        for row in payload["history_item_index"]
    }
    if (
        redacted_keys != item_keys
        or len(redacted_keys) != len(payload["redacted_items"])
        or history_item_keys != item_keys
        or len(history_item_keys) != len(payload["history_item_index"])
    ):
        raise common.ContractError("Raw/redacted/history item keysets differ")

    pair_keys: set[tuple[str, str]] = set()
    pair_uids: set[str] = set()
    per_world_pairs: Counter[str] = Counter()
    for row in payload["complete_model_pair_endpoints"]:
        world_uid = str(row["world_uid"])
        pair_uid = str(row["canonical_pair_uid"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        key = (world_uid, pair_uid)
        if (
            key in pair_keys
            or pair_uid in pair_uids
            or (world_uid, left) not in seller_keys
            or (world_uid, right) not in seller_keys
            or pair_uid != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError("Complete-pair lineage drift")
        pair_keys.add(key)
        pair_uids.add(pair_uid)
        per_world_pairs[world_uid] += 1
    if (
        set(per_world_pairs) != world_uids
        or set(per_world_pairs.values()) != {378}
    ):
        raise common.ContractError("Complete-pair world cardinality drift")

    candidate_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in payload["candidate_pairs"]
    }
    audit_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in payload["candidate_sampling_audit"]
    }
    if (
        len(candidate_keys) != len(payload["candidate_pairs"])
        or candidate_keys != audit_keys
        or not candidate_keys <= pair_keys
    ):
        raise common.ContractError("C40 keyset closure failed")
    candidates_per_world = Counter(
        world_uid for world_uid, _pair_uid in candidate_keys
    )
    positive_mechanisms: dict[str, set[str]] = defaultdict(set)
    negative_flags: dict[str, set[str]] = defaultdict(set)
    ranks: dict[str, list[int]] = defaultdict(list)
    for row in payload["candidate_sampling_audit"]:
        world_uid = str(row["world_uid"])
        positive_mechanisms[world_uid].update(
            value
            for value in str(
                row["covered_positive_mechanisms"]
            ).split("|")
            if value
        )
        negative_flags[world_uid].update(
            value
            for value in str(row["covered_negative_flags"]).split("|")
            if value
        )
        ranks[world_uid].append(int(row["selected_rank"]))
    if any(
        candidates_per_world[world_uid] != 40
        or len(positive_mechanisms[world_uid]) != 8
        or len(negative_flags[world_uid]) != 7
        or sorted(ranks[world_uid]) != list(range(1, 41))
        for world_uid in world_uids
    ):
        raise common.ContractError("C40 per-world coverage/rank drift")

    label_by_pair = {
        str(row["canonical_pair_uid"]): int(row["label"])
        for row in payload["classification_labels"]
    }
    positive_per_world: Counter[str] = Counter()
    for world_uid, pair_uid in candidate_keys:
        if pair_uid not in label_by_pair:
            raise common.ContractError("C40 label keyset drift")
        positive_per_world[world_uid] += label_by_pair[pair_uid]
    expected_positive = int(
        overlay["classification_positive_count_per_world"][split]
    )
    if (
        len(label_by_pair) != len(payload["classification_labels"])
        or set(label_by_pair) != {pair_uid for _world_uid, pair_uid in candidate_keys}
        or any(
            positive_per_world[world_uid] != expected_positive
            for world_uid in world_uids
        )
    ):
        raise common.ContractError("C40 label balance/keyset drift")

    identity33_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in payload["identity33_all_pairs"]
    }
    if (
        identity33_keys != pair_keys
        or len(identity33_keys) != len(payload["identity33_all_pairs"])
    ):
        raise common.ContractError("Identity33 all-pair keyset drift")
    profile_keys = {
        (str(row["world_uid"]), str(row["seller_uid"]))
        for row in payload["seller_profiles"]
    }
    if profile_keys != seller_keys or len(profile_keys) != len(
        payload["seller_profiles"]
    ):
        raise common.ContractError("Seller-profile keyset drift")

    membership: dict[tuple[str, str], str] = {}
    controller_sizes: Counter[tuple[str, str]] = Counter()
    for row in payload["controller_membership"]:
        key = (str(row["world_uid"]), str(row["seller_uid"]))
        controller_uid = str(row["controller_uid"])
        if key in membership or key not in seller_keys:
            raise common.ContractError("Controller membership key drift")
        membership[key] = controller_uid
        controller_sizes[(key[0], controller_uid)] += 1
    sizes_by_world: dict[str, list[int]] = defaultdict(list)
    for (world_uid, _controller_uid), count in controller_sizes.items():
        sizes_by_world[world_uid].append(count)
    if (
        set(membership) != seller_keys
        or any(
            sorted(sizes_by_world[world_uid]) != [2] * 8 + [3] * 4
            for world_uid in world_uids
        )
    ):
        raise common.ContractError("Controller topology aggregate drift")

    expected_source_dataset = common.source_dataset_name(
        policy,
        mode=MODE,
        split=split,
    )
    for name in (
        "parsed_identity_occurrences",
        "history_safe_occurrences",
    ):
        if any(
            str(row["source_dataset"]) != expected_source_dataset
            or "development_smoke" in str(row["source_dataset"])
            for row in payload[name]
        ):
            raise common.ContractError(
                f"Training-ready source-dataset alias leaked: {name}"
            )

    key_hex = str(policy["randomness"][MODE]["identity_value_key_hex"])
    salt_map = policy["identity_design"]["identity_value_generation"][
        "salt_selection"
    ]["training_ready_per_type_salt_counters"]
    handle_encoding = str(
        policy["identity_design"]["identity_value_generation"][
            "handle_encoding_by_mode"
        ][MODE]
    )
    identity_uids: set[str] = set()
    identity_values_seen: set[str] = set()
    for row in payload["identity_assets"]:
        identity_uid = str(row["identity_uid"])
        identity_value = str(row["identity_value"])
        identity_type = str(row["identity_type"])
        global_index = int(row["global_asset_index"])
        expected_value = identity_values.identity_value(
            key_hex=key_hex,
            identity_type=identity_type,
            salt=int(salt_map[identity_type]),
            global_asset_index=global_index,
            handle_encoding=handle_encoding,
        )
        if (
            identity_uid in identity_uids
            or identity_value in identity_values_seen
            or identity_value != expected_value
            or identity_uid
            != "id_"
            + common.canonical_sha256(
                {
                    "contact_type": identity_type.strip().lower(),
                    "normalized_value": identity_value.strip().lower(),
                }
            )
        ):
            raise common.ContractError("Identity value/UID replay drift")
        identity_uids.add(identity_uid)
        identity_values_seen.add(identity_value)

    if split in {"audit_a", "audit_b"}:
        queries = payload["retrieval_queries"]
        relations = payload["retrieval_relations"]
        qrels = payload["retrieval_qrels"]
        expected_queries = world_count * int(
            overlay["retrieval"]["queries_per_world"]
        )
        query_by_uid: dict[str, tuple[str, str]] = {}
        queries_by_world: dict[str, set[str]] = defaultdict(set)
        for row in queries:
            if (
                tuple(row)
                != ("query_uid", "world_uid", "query_seller_uid")
                or any(type(value) is not str for value in row.values())
            ):
                raise common.ContractError("Retrieval query schema drift")
            query_uid = row["query_uid"]
            world_uid = row["world_uid"]
            seller_uid = row["query_seller_uid"]
            if (
                query_uid in query_by_uid
                or (world_uid, seller_uid) not in seller_keys
                or query_uid != common.query_uid(world_uid, seller_uid)
            ):
                raise common.ContractError("Retrieval query lineage drift")
            query_by_uid[query_uid] = (world_uid, seller_uid)
            queries_by_world[world_uid].add(seller_uid)
        sellers_by_world: dict[str, set[str]] = defaultdict(set)
        for world_uid, seller_uid in seller_keys:
            sellers_by_world[world_uid].add(seller_uid)
        query_key_hex = str(
            policy["randomness"][MODE]["query_key_hex"]
        )
        queries_per_world = int(
            overlay["retrieval"]["queries_per_world"]
        )
        for world_uid, seller_set in sellers_by_world.items():
            expected_query_sellers = set(
                sorted(
                    seller_set,
                    key=lambda seller_uid: (
                        common.hmac_digest(
                            query_key_hex,
                            world_uid,
                            "training_ready_retrieval_query",
                            seller_uid,
                        ),
                        seller_uid.encode("utf-8"),
                    ),
                )[:queries_per_world]
            )
            if queries_by_world[world_uid] != expected_query_sellers:
                raise common.ContractError(
                    "Retrieval query HMAC selection drift"
                )
        relation_by_uid: dict[
            str, tuple[str, str, str, str]
        ] = {}
        galleries_by_query: dict[str, set[str]] = defaultdict(set)
        for row in relations:
            if (
                tuple(row)
                != (
                    "relation_uid",
                    "query_uid",
                    "world_uid",
                    "query_seller_uid",
                    "gallery_seller_uid",
                )
                or any(type(value) is not str for value in row.values())
            ):
                raise common.ContractError("Retrieval relation schema drift")
            relation_uid = row["relation_uid"]
            query_uid = row["query_uid"]
            world_uid = row["world_uid"]
            query_seller_uid = row["query_seller_uid"]
            gallery_seller_uid = row["gallery_seller_uid"]
            if (
                relation_uid in relation_by_uid
                or query_uid not in query_by_uid
                or query_by_uid[query_uid]
                != (world_uid, query_seller_uid)
                or gallery_seller_uid == query_seller_uid
                or (world_uid, gallery_seller_uid) not in seller_keys
                or relation_uid
                != common.relation_uid(query_uid, gallery_seller_uid)
            ):
                raise common.ContractError(
                    "Retrieval relation lineage drift"
                )
            relation_by_uid[relation_uid] = (
                query_uid,
                world_uid,
                query_seller_uid,
                gallery_seller_uid,
            )
            if gallery_seller_uid in galleries_by_query[query_uid]:
                raise common.ContractError(
                    "Retrieval gallery duplicate within query"
                )
            galleries_by_query[query_uid].add(gallery_seller_uid)
        for query_uid, (world_uid, query_seller_uid) in query_by_uid.items():
            if galleries_by_query[query_uid] != (
                sellers_by_world[world_uid] - {query_seller_uid}
            ):
                raise common.ContractError(
                    "Retrieval gallery is not the other 27 sellers"
                )
        qrel_by_relation: dict[str, str] = {}
        for row in qrels:
            if (
                tuple(row) != ("relation_uid", "query_uid", "relevance")
                or any(type(value) is not str for value in row.values())
                or row["relevance"] not in {"0", "1"}
                or row["relation_uid"] in qrel_by_relation
                or row["relation_uid"] not in relation_by_uid
            ):
                raise common.ContractError("Retrieval qrel schema/key drift")
            (
                expected_query_uid,
                world_uid,
                query_seller_uid,
                gallery_seller_uid,
            ) = relation_by_uid[row["relation_uid"]]
            expected_relevance = str(
                int(
                    membership[(world_uid, query_seller_uid)]
                    == membership[(world_uid, gallery_seller_uid)]
                )
            )
            if (
                row["query_uid"] != expected_query_uid
                or row["relevance"] != expected_relevance
            ):
                raise common.ContractError(
                    "Retrieval qrel formula/FK drift"
                )
            qrel_by_relation[row["relation_uid"]] = row["relevance"]
        if (
            len(queries) != expected_queries
            or len(relations) != expected_queries * 27
            or len(qrels) != len(relations)
            or set(qrel_by_relation) != set(relation_by_uid)
        ):
            raise common.ContractError("Retrieval split count drift")
    elif any(
        payload[name]
        for name in (
            "retrieval_queries",
            "retrieval_relations",
            "retrieval_qrels",
        )
    ):
        raise common.ContractError("Retrieval rows leaked into non-audit split")
    return {
        "world_uid_count": len(world_uids),
        "seller_uid_count": len(seller_uids),
        "item_uid_count": len(item_uids),
        "pair_uid_count": len(pair_uids),
        "identity_uid_count": len(identity_uids),
        "identity_value_count": len(identity_values_seen),
        "all_keysets_and_foreign_keys_exact": True,
        "all_source_dataset_names_training_ready": True,
        "identity_values_replayed_exactly": True,
    }


def _build_retrieval(
    policy: Mapping[str, Any],
    *,
    sellers: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    queries_per_world: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    seller_by_world: dict[str, list[str]] = defaultdict(list)
    controller: dict[tuple[str, str], str] = {}
    for row in sellers:
        seller_by_world[str(row["world_uid"])].append(str(row["seller_uid"]))
    for row in memberships:
        controller[(str(row["world_uid"]), str(row["seller_uid"]))] = str(
            row["controller_uid"]
        )
    key_hex = str(policy["randomness"][MODE]["query_key_hex"])
    queries: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    qrels: list[dict[str, str]] = []
    for world_uid in common.utf8_sort(seller_by_world):
        seller_uids = common.utf8_sort(seller_by_world[world_uid])
        ranked = sorted(
            seller_uids,
            key=lambda seller_uid: (
                common.hmac_digest(
                    key_hex,
                    world_uid,
                    "training_ready_retrieval_query",
                    seller_uid,
                ),
                seller_uid.encode("utf-8"),
            ),
        )
        for query_seller_uid in ranked[:queries_per_world]:
            query_uid = common.query_uid(world_uid, query_seller_uid)
            queries.append(
                {
                    "query_uid": query_uid,
                    "world_uid": world_uid,
                    "query_seller_uid": query_seller_uid,
                }
            )
            relevant = 0
            for gallery_seller_uid in seller_uids:
                if gallery_seller_uid == query_seller_uid:
                    continue
                relation_uid = common.relation_uid(
                    query_uid, gallery_seller_uid
                )
                relations.append(
                    {
                        "relation_uid": relation_uid,
                        "query_uid": query_uid,
                        "world_uid": world_uid,
                        "query_seller_uid": query_seller_uid,
                        "gallery_seller_uid": gallery_seller_uid,
                    }
                )
                relevance = int(
                    controller[(world_uid, query_seller_uid)]
                    == controller[(world_uid, gallery_seller_uid)]
                )
                relevant += relevance
                qrels.append(
                    {
                        "relation_uid": relation_uid,
                        "query_uid": query_uid,
                        "relevance": str(relevance),
                    }
                )
            if relevant not in {1, 2}:
                raise common.ContractError(
                    "Retrieval query relevance cardinality drift"
                )
    expected_worlds = len(seller_by_world)
    if (
        len(queries) != expected_worlds * queries_per_world
        or len(relations) != len(queries) * 27
        or len(qrels) != len(relations)
        or len({row["query_uid"] for row in queries}) != len(queries)
        or len({row["relation_uid"] for row in relations})
        != len(relations)
    ):
        raise common.ContractError("Retrieval aggregate count gate failed")
    return queries, relations, qrels


def build_split_in_memory(
    policy: dict[str, Any],
    overlay: Mapping[str, Any],
    *,
    split: str,
    structure_key_hex: str,
    progress_every: int,
    allow_shortcut_failure_for_design_preflight: bool = False,
    mini_exercise_allow_incomplete_train_support: bool = False,
) -> dict[str, Any]:
    if type(progress_every) is not int or progress_every < 1:
        raise common.ContractError(
            "Training-ready progress interval must be a positive integer"
        )
    expected_worlds = int(overlay["world_counts"][split])
    if mini_exercise_allow_incomplete_train_support and (
        split != "train"
        or overlay.get("generation_enabled") is not False
        or expected_worlds >= 500
        or structure_key_hex != DESIGN_ONLY_STRUCTURE_KEY_HEX
    ):
        raise common.ContractError(
            "Incomplete train identity support is allowed only for an "
            "explicit sub-500-world design-key mini exercise"
        )
    template, fixture, style_profile = legacy_generator._load_release_inputs(
        policy,
        mode=MODE,
    )
    records = [
        row
        for row in structure.build_mode_world_pool(policy, mode=MODE)
        if row["split"] == split
    ]
    if len(records) != expected_worlds:
        raise common.ContractError("Training-ready split world-count drift")
    payload = _empty_payload()
    started = time.perf_counter()
    for ordinal, record in enumerate(records, start=1):
        built = _one_world(
            policy=policy,
            split=split,
            record=record,
            structure_key_hex=structure_key_hex,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
        )
        _append_world(payload, built)
        if ordinal % progress_every == 0 or ordinal == expected_worlds:
            print(
                json.dumps(
                    {
                        "event": "training_ready_generation_progress",
                        "split": split,
                        "worlds_complete": ordinal,
                        "worlds_total": expected_worlds,
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    labels = label_sealer.build_labels(
        candidate_rows=payload["candidate_pairs"],
        membership_rows=payload["controller_membership"],
        expected_world_count=expected_worlds,
    )
    payload["classification_labels"] = labels
    formula_audit = _validate_formula_independently(
        candidate_rows=payload["candidate_pairs"],
        membership_rows=payload["controller_membership"],
        labels=labels,
    )
    expected_positive = (
        expected_worlds
        * int(
            overlay["classification_positive_count_per_world"][split]
        )
    )
    if formula_audit["positive_count"] != expected_positive:
        raise common.ContractError("Split positive-label count drift")
    nuisance = nuisance_projector.build_projection(
        candidate_rows=payload["candidate_pairs"],
        redacted_items=payload["redacted_items"],
        history_item_rows=payload["history_item_index"],
        expected_world_count=expected_worlds,
    )
    payload["null_nuisance_projection"] = nuisance
    shortcut_report, shortcut_oof = shortcut_audit.run_audit(
        projection_rows=nuisance,
        label_rows=labels,
        split=split,
        expected_world_count=expected_worlds,
        bootstrap_replicates=int(
            overlay["shortcut_gate"]["bootstrap_replicates"]
        ),
        point_maximum=float(
            overlay["shortcut_gate"]["maximum_symmetric_auc"]
        ),
        upper_maximum=float(
            overlay["shortcut_gate"][
                "maximum_world_bootstrap_95_upper"
            ]
        ),
    )
    payload["shortcut_oof"] = shortcut_oof
    if (
        shortcut_report["status"] != "PASS_METADATA_SHORTCUT_ONLY"
        and not allow_shortcut_failure_for_design_preflight
    ):
        raise common.ContractError(
            "Formal split failed the frozen metadata-shortcut gate: "
            f"point={shortcut_report['point_statistic_max_auc_symmetric']}"
            f"/{shortcut_report['point_maximum']} "
            f"upper={shortcut_report['bootstrap_95_upper']}"
            f"/{shortcut_report['bootstrap_95_upper_maximum']}"
        )
    identity33_audit = _identity33_matrix_audit(
        policy,
        payload["identity33_all_pairs"],
        require_no_zero_columns=(
            split == "train"
            and not mini_exercise_allow_incomplete_train_support
        ),
    )
    if split in {"audit_a", "audit_b"}:
        queries, relations, qrels = _build_retrieval(
            policy,
            sellers=payload["sellers"],
            memberships=payload["controller_membership"],
            queries_per_world=int(
                overlay["retrieval"]["queries_per_world"]
            ),
        )
        payload["retrieval_queries"] = queries
        payload["retrieval_relations"] = relations
        payload["retrieval_qrels"] = qrels
    aggregate_audit = _validate_payload(
        policy,
        overlay,
        split=split,
        payload=payload,
    )
    return {
        "payload": payload,
        "formula_audit": formula_audit,
        "shortcut_report": shortcut_report,
        "identity33_audit": identity33_audit,
        "aggregate_audit": aggregate_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_csv(
    stage: Path,
    relative: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Path:
    path = stage / relative
    common.write_csv(path, [dict(row) for row in rows], list(fields))
    return path


def _write_jsonl(
    stage: Path,
    relative: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = stage / relative
    common.write_jsonl(path, [dict(row) for row in rows])
    return path


def _read_csv_exact(
    path: Path,
    *,
    fields: Sequence[str],
) -> list[dict[str, str]]:
    with open(
        common.filesystem_path(path),
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise common.ContractError(
                f"Persisted CSV header drift: {path.name}"
            )
        rows = [dict(row) for row in reader]
    if any(tuple(row) != tuple(fields) or None in row.values() for row in rows):
        raise common.ContractError(
            f"Persisted CSV row schema drift: {path.name}"
        )
    return rows


def _validate_persisted_m1(
    policy: Mapping[str, Any],
    *,
    seed_id: str,
    matrix_path: Path,
    mapping_path: Path,
    m2_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    feature_names = list(policy["history_features"]["feature_names"])
    matrix_fields = [
        "canonical_pair_uid",
        "world_uid",
        *feature_names,
    ]
    mapping_fields = [
        "rewire_seed_id",
        "world_uid",
        "universe",
        "destination_pair_uid",
        "source_pair_uid",
        "endpoint_disjoint_bool",
        "feature_vector_sha256",
    ]
    persisted = _read_csv_exact(matrix_path, fields=matrix_fields)
    mappings = _read_csv_exact(mapping_path, fields=mapping_fields)
    base = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): {
            name: str(row[name]) for name in feature_names
        }
        for row in m2_rows
    }
    output = {
        (row["world_uid"], row["canonical_pair_uid"]): {
            name: row[name] for name in feature_names
        }
        for row in persisted
    }
    endpoints = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): {
            str(row["seller_uid_left"]),
            str(row["seller_uid_right"]),
        }
        for row in endpoint_rows
    }
    c40 = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_rows
    }
    if (
        len(base) != len(m2_rows)
        or len(output) != len(persisted)
        or set(output) != set(base)
        or set(endpoints) != set(base)
    ):
        raise common.ContractError(
            "Persisted M1 matrix keyset or source keyset drift"
        )
    destination_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    if len(mappings) != len(base):
        raise common.ContractError("Persisted M1 mapping count drift")
    for row in mappings:
        world_uid = row["world_uid"]
        destination_uid = row["destination_pair_uid"]
        source_uid = row["source_pair_uid"]
        destination = (world_uid, destination_uid)
        source = (world_uid, source_uid)
        expected_universe = (
            "primary_c40"
            if destination in c40
            else "secondary_complement"
        )
        if (
            row["rewire_seed_id"] != seed_id
            or row["universe"] != expected_universe
            or row["endpoint_disjoint_bool"] != "True"
            or destination not in base
            or source not in base
            or (source in c40) != (destination in c40)
            or destination == source
            or endpoints[destination] & endpoints[source]
            or output[destination] != base[source]
        ):
            raise common.ContractError(
                "Persisted M1 mapping/vector/endpoint replay failed"
            )
        vector = [base[source][name] for name in feature_names]
        if row["feature_vector_sha256"] != common.canonical_sha256(vector):
            raise common.ContractError(
                "Persisted M1 feature-vector hash mismatch"
            )
        group = (world_uid, expected_universe)
        if (
            destination_uid in destination_sets[group]
            or source_uid in source_sets[group]
        ):
            raise common.ContractError(
                "Persisted M1 mapping is not a bijection"
            )
        destination_sets[group].add(destination_uid)
        source_sets[group].add(source_uid)
    expected_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for world_uid, pair_uid in base:
        universe = (
            "primary_c40"
            if (world_uid, pair_uid) in c40
            else "secondary_complement"
        )
        expected_groups[(world_uid, universe)].add(pair_uid)
    if (
        set(destination_sets) != set(expected_groups)
        or any(
            destination_sets[group] != expected
            or source_sets[group] != expected
            for group, expected in expected_groups.items()
        )
    ):
        raise common.ContractError(
            "Persisted M1 world/universe multiset changed"
        )
    return {
        "persisted_matrix_reread_exact": True,
        "persisted_mapping_reread_exact": True,
        "persisted_whole_vector_replay_exact": True,
        "persisted_endpoint_disjoint_bijection_exact": True,
    }


def _has_reparse_attribute(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _file_records(stage: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    root = common.filesystem_path(stage)
    root_path = Path(root)
    if (
        root_path.is_symlink()
        or not root_path.is_dir()
        or _has_reparse_attribute(root_path)
    ):
        raise common.ContractError(
            "Split release root must be a plain non-reparse directory"
        )
    discovered: list[tuple[str, Path]] = []
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        if (
            directory_path.is_symlink()
            or _has_reparse_attribute(directory_path)
        ):
            raise common.ContractError(
                "Reparse directory is forbidden in a split release"
            )
        directory_names.sort(key=lambda value: value.encode("utf-8"))
        file_names.sort(key=lambda value: value.encode("utf-8"))
        for name in directory_names:
            child = Path(directory) / name
            if (
                child.is_symlink()
                or not child.is_dir()
                or _has_reparse_attribute(child)
            ):
                raise common.ContractError(
                    "Reparse directory is forbidden in a split release"
                )
        for name in file_names:
            full = Path(directory) / name
            if (
                full.is_symlink()
                or not full.is_file()
                or _has_reparse_attribute(full)
            ):
                raise common.ContractError(
                    "Non-regular or reparse file is forbidden in a "
                    "split release"
                )
            relative = os.path.relpath(
                os.fspath(full),
                root,
            ).replace(os.sep, "/")
            if (
                relative.startswith("../")
                or relative == ".."
                or "\\" in relative
            ):
                raise common.ContractError(
                    "Split release member escaped its root"
                )
            discovered.append((relative, full))
    for relative, path in sorted(
        discovered,
        key=lambda row: row[0].encode("utf-8"),
    ):
        output.append(
            {
                "path": relative,
                "size_bytes": os.stat(os.fspath(path)).st_size,
                "sha256": common.sha256_file(path),
                "model_mount_allowed": relative
                in {
                    "observed/complete_model_pair_endpoints.csv",
                    "observed/redacted_items.jsonl",
                    "observed/seller_profiles.jsonl",
                },
            }
        )
    return output


def _validate_split_tree(
    root: Path,
    *,
    expected_manifest: Mapping[str, Any],
) -> None:
    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise common.ContractError("Split manifest is missing after write")
    observed_manifest = common.load_json(manifest_path)
    if (
        common.canonical_json_bytes(observed_manifest)
        != common.canonical_json_bytes(expected_manifest)
    ):
        raise common.ContractError("Split manifest bytes changed after write")
    without_self = dict(observed_manifest)
    observed_self = without_self.pop("canonical_self_hash", None)
    if (
        not isinstance(observed_self, str)
        or observed_self != common.canonical_sha256(without_self)
    ):
        raise common.ContractError("Split manifest self-hash mismatch")
    records = _file_records(root)
    manifest_records = [
        row for row in records if row["path"] != "split_manifest.json"
    ]
    if (
        len(records) != len(manifest_records) + 1
        or common.canonical_json_bytes(manifest_records)
        != common.canonical_json_bytes(observed_manifest["files"])
    ):
        raise common.ContractError(
            "Split manifest member set or file hashes are incomplete"
        )


def model_mount_contract(split: str) -> dict[str, Any]:
    if split not in SPLITS:
        raise common.ContractError("Unknown split for model-input contract")
    return {
        "version": "2026-07-29-step28-v13-model-input-contract-v1",
        "split": split,
        "scope": (
            "dataset_file_allowlists_only; frozen M0 p0 is an external "
            "lineage-bound input"
        ),
        "m0_exact_allowlist": [
            "observed/complete_model_pair_endpoints.csv",
            "observed/redacted_items.jsonl",
            "observed/seller_profiles.jsonl",
        ],
        "m0_output_contract": (
            "one frozen probability p0 per canonical_pair_uid"
        ),
        "adapter_required_external_input": {
            "role": "frozen_m0_p0",
            "schema": ["canonical_pair_uid", "p0"],
            "pair_keyset": "exact C40 candidate keyset for this split",
            "producer": (
                "operational M0 run with this split's exact M0 allowlist"
            ),
            "must_be_hash_pinned_before_adapter_use": True,
            "validated_by_dataset_mount_checker": False,
        },
        "m2_training_exact_allowlist": (
            [
                "observed/candidate_pairs.csv",
                "observed/identity33_all_pairs.csv",
                "supervision/classification_labels.csv",
            ]
            if split == "train"
            else []
        ),
        "m1_training_exact_allowlist_by_replicate": (
            {
                f"r{index:02d}": [
                    "observed/candidate_pairs.csv",
                    f"m1/r{index:02d}/identity33.csv",
                    "supervision/classification_labels.csv",
                ]
                for index in range(1, 6)
            }
            if split == "train"
            else {}
        ),
        "adapter_evaluation_exact_allowlist": (
            [
                "observed/candidate_pairs.csv",
                "observed/identity33_all_pairs.csv",
            ]
            if split != "train"
            else []
        ),
        "evaluation_supervision_path": (
            "supervision/classification_labels.csv"
            if split == "development"
            else (
                "sealed_supervision/classification_labels.csv"
                if split in {"audit_a", "audit_b"}
                else None
            )
        ),
        "evaluation_supervision_must_not_be_model_mounted": (
            split in {"audit_a", "audit_b"}
        ),
        "mount_observed_directory_as_a_whole": False,
        "adapter_forbidden_fields": [
            "market",
            "controller_uid",
            "mechanism",
            "label_stratum",
            "hmac_digest_hex",
            "selected_rank",
        ],
        "adapter_non_feature_join_keys": [
            "canonical_pair_uid",
            "world_uid",
        ],
        "join_keys_must_not_enter_feature_matrix": True,
    }


def _write_payload(
    policy: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    split: str,
    stage: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = result["payload"]
    schema = policy["relational_integrity"]["observed_core_schemas"]
    pair_schema = policy["relational_integrity"][
        "pair_projection_contract"
    ]["complete_model_pair_endpoints_schema"]
    profile_schema = common.load_json(
        common.verify_file_pin(
            policy["frozen_inputs"]["step3_profile_schema"],
            label="Step3 seller profile schema",
        )
    )
    profile_fields = ["world_uid", *profile_schema["profile_fields"]]
    identity_fields = [
        "canonical_pair_uid",
        "world_uid",
        *policy["history_features"]["feature_names"],
    ]

    _write_csv(stage, "observed/worlds.csv", payload["worlds"], schema["worlds.csv"])
    _write_csv(stage, "observed/sellers.csv", payload["sellers"], schema["sellers.csv"])
    _write_csv(
        stage,
        "observed/complete_model_pair_endpoints.csv",
        payload["complete_model_pair_endpoints"],
        pair_schema,
    )
    _write_csv(
        stage,
        "observed/candidate_pairs.csv",
        payload["candidate_pairs"],
        mechanism_c40.SAFE_FIELDS,
    )
    _write_jsonl(
        stage,
        "observed/seller_profiles.jsonl",
        payload["seller_profiles"],
    )
    if any(list(row) != profile_fields for row in payload["seller_profiles"]):
        raise common.ContractError("Seller-profile output schema drift")
    _write_jsonl(
        stage,
        "observed/redacted_items.jsonl",
        payload["redacted_items"],
    )
    _write_csv(
        stage,
        "observed/history_safe_occurrences.csv",
        payload["history_safe_occurrences"],
        schema["history_safe_occurrences.csv"],
    )
    _write_csv(
        stage,
        "observed/history_item_index.csv",
        payload["history_item_index"],
        schema["history_item_index.csv"],
    )
    _write_jsonl(
        stage,
        "observed/history_projection_attestations.jsonl",
        payload["history_projection_attestations"],
    )
    _write_csv(
        stage,
        "observed/identity33_all_pairs.csv",
        payload["identity33_all_pairs"],
        identity_fields,
    )
    _write_csv(
        stage,
        "audit/null_nuisance_pairs.csv",
        payload["null_nuisance_projection"],
        (
            "canonical_pair_uid",
            "world_uid",
            *shortcut_common.PAIR_FEATURES,
        ),
    )
    if split in {"train", "development"}:
        common.write_json(
            stage / "audit/metadata_shortcut_audit.json",
            dict(result["shortcut_report"]),
        )
        _write_csv(
            stage,
            "private_audit/metadata_shortcut_oof.csv",
            payload["shortcut_oof"],
            shortcut_common.OOF_FIELDS,
        )
    else:
        sealed_report_path = (
            stage
            / "sealed_supervision"
            / "metadata_shortcut_audit.private.json"
        )
        common.write_json(
            sealed_report_path,
            dict(result["shortcut_report"]),
        )
        _write_csv(
            stage,
            "sealed_supervision/metadata_shortcut_oof.private.csv",
            payload["shortcut_oof"],
            shortcut_common.OOF_FIELDS,
        )
        common.write_json(
            stage / "audit/metadata_shortcut_gate.json",
            {
                "status": result["shortcut_report"]["status"],
                "split": split,
                "full_report_sha256": common.sha256_file(
                    sealed_report_path
                ),
                "point_threshold": overlay["shortcut_gate"][
                    "maximum_symmetric_auc"
                ],
                "bootstrap_upper_threshold": overlay["shortcut_gate"][
                    "maximum_world_bootstrap_95_upper"
                ],
                "point_value_withheld_until_unseal": True,
                "bootstrap_upper_value_withheld_until_unseal": True,
            },
        )
    common.write_json(
        stage / "audit/identity33_matrix_audit.json",
        dict(result["identity33_audit"]),
    )
    common.write_json(
        stage / "audit/label_formula_validation.json",
        dict(result["formula_audit"]),
    )
    supervision_root = (
        "supervision"
        if split in {"train", "development"}
        else "sealed_supervision"
    )
    _write_csv(
        stage,
        f"{supervision_root}/classification_labels.csv",
        payload["classification_labels"],
        ("canonical_pair_uid", "label"),
    )
    _write_jsonl(
        stage,
        "private_oracle/raw_identity_bearing_items.jsonl",
        payload["raw_items"],
    )
    _write_csv(
        stage,
        "private_oracle/controller_membership.csv",
        payload["controller_membership"],
        ("world_uid", "controller_uid", "seller_uid"),
    )
    _write_csv(
        stage,
        "private_oracle/controller_style_groups.csv",
        payload["controller_style_groups"],
        ("world_uid", "controller_uid", "style_id"),
    )
    _write_csv(
        stage,
        "private_oracle/mechanism_assignments.csv",
        payload["mechanism_assignments"],
        ("world_uid", "controller_uid", "mechanism", "mechanism_slot_uid"),
    )
    _write_jsonl(
        stage,
        "private_oracle/identity_assets.jsonl",
        payload["identity_assets"],
    )
    _write_csv(
        stage,
        "private_oracle/positive_targets.csv",
        payload["positive_targets"],
        (
            "world_uid",
            "controller_uid",
            "mechanism",
            "mechanism_slot_uid",
            "seller_uid_left",
            "seller_uid_right",
            "canonical_pair_uid",
        ),
    )
    _write_csv(
        stage,
        "private_oracle/negative_flags.csv",
        payload["negative_flags"],
        ("world_uid", "canonical_pair_uid", "flag", "asset_index"),
    )
    _write_csv(
        stage,
        "private_audit/candidate_sampling_audit.csv",
        payload["candidate_sampling_audit"],
        mechanism_c40.AUDIT_FIELDS,
    )
    _write_csv(
        stage,
        "private_audit/parsed_identity_occurrences.csv",
        payload["parsed_identity_occurrences"],
        schema["parsed_identity_occurrences.structural_audit_private.csv"],
    )
    _write_jsonl(
        stage,
        "private_audit/renderer_identity_slots.audit.jsonl",
        payload["identity_slots_audit"],
    )
    _write_jsonl(
        stage,
        "private_audit/renderer_identity_slots.edit.jsonl",
        payload["identity_slots_edit"],
    )
    _write_jsonl(
        stage,
        "private_audit/renderer_noise_slots.audit.jsonl",
        payload["noise_slots_audit"],
    )
    _write_jsonl(
        stage,
        "private_audit/render_asts.jsonl",
        payload["render_asts"],
    )
    _write_jsonl(
        stage,
        "private_audit/registered_overrides.jsonl",
        payload["override_audit"],
    )
    _write_jsonl(
        stage,
        "private_audit/redaction_diagnostics.jsonl",
        payload["redaction_diagnostics"],
    )
    _write_jsonl(
        stage,
        "private_audit/world_generation_audit.jsonl",
        payload["world_generation_audit"],
    )

    if split in {"audit_a", "audit_b"}:
        _write_csv(
            stage,
            "retrieval/queries.csv",
            payload["retrieval_queries"],
            ("query_uid", "world_uid", "query_seller_uid"),
        )
        _write_csv(
            stage,
            "retrieval/relations.csv",
            payload["retrieval_relations"],
            (
                "relation_uid",
                "query_uid",
                "world_uid",
                "query_seller_uid",
                "gallery_seller_uid",
            ),
        )
        _write_csv(
            stage,
            "sealed_supervision/retrieval_qrels.csv",
            payload["retrieval_qrels"],
            ("relation_uid", "query_uid", "relevance"),
        )

    m1_receipts: list[dict[str, Any]] = []
    if split == "train":
        for replicate_index, seed_hex in enumerate(
            policy["randomness"][MODE]["rewire_key_hexes"],
            start=1,
        ):
            print(
                json.dumps(
                    {
                        "event": "m1_derangement_start",
                        "seed_commitment": common.sha256_bytes(
                            bytes.fromhex(seed_hex)
                        ),
                    }
                ),
                flush=True,
            )
            placebo = feature_derangement.build_one_feature_derangement(
                policy,
                mode=MODE,
                split=split,
                seed_hex=str(seed_hex),
                m2_identity33_all_pairs=payload[
                    "identity33_all_pairs"
                ],
                candidate_pairs=payload["candidate_pairs"],
                complete_pair_endpoints=payload[
                    "complete_model_pair_endpoints"
                ],
            )
            seed_id = str(placebo["rewire_seed_id"])
            short_directory = f"r{replicate_index:02d}"
            matrix_path = _write_csv(
                stage,
                f"m1/{short_directory}/identity33.csv",
                placebo["identity33_all_pairs"],
                identity_fields,
            )
            mapping_rows = placebo["feature_derangement_mapping"]
            mapping_path = _write_csv(
                stage,
                f"private_audit/m1/{short_directory}/mapping.csv",
                mapping_rows,
                (
                    "rewire_seed_id",
                    "world_uid",
                    "universe",
                    "destination_pair_uid",
                    "source_pair_uid",
                    "endpoint_disjoint_bool",
                    "feature_vector_sha256",
                ),
            )
            persisted_audit = _validate_persisted_m1(
                policy,
                seed_id=seed_id,
                matrix_path=matrix_path,
                mapping_path=mapping_path,
                m2_rows=payload["identity33_all_pairs"],
                candidate_rows=payload["candidate_pairs"],
                endpoint_rows=payload[
                    "complete_model_pair_endpoints"
                ],
            )
            m1_receipts.append(
                {
                    "replicate_index": replicate_index,
                    "rewire_seed_id": seed_id,
                    "matrix_path": (
                        f"m1/{short_directory}/identity33.csv"
                    ),
                    "mapping_path": (
                        f"private_audit/m1/{short_directory}/mapping.csv"
                    ),
                    "matrix_row_count": len(
                        placebo["identity33_all_pairs"]
                    ),
                    "mapping_row_count": len(mapping_rows),
                    "matrix_sha256": common.sha256_file(matrix_path),
                    "mapping_sha256": common.sha256_file(mapping_path),
                    "joint_vector_multiset_exact_by_world_and_universe": (
                        placebo[
                            "joint_vector_multiset_exact_by_world_and_universe"
                        ]
                    ),
                    "endpoint_disjoint_bijection_exact": placebo[
                        "endpoint_disjoint_bijection_exact"
                    ],
                    "labels_or_controller_inputs_read": placebo[
                        "labels_or_controller_inputs_read"
                    ],
                    **persisted_audit,
                }
            )
            if (
                m1_receipts[-1][
                    "joint_vector_multiset_exact_by_world_and_universe"
                ]
                is not True
                or m1_receipts[-1][
                    "endpoint_disjoint_bijection_exact"
                ]
                is not True
                or m1_receipts[-1][
                    "labels_or_controller_inputs_read"
                ]
                is not False
                or not all(
                    m1_receipts[-1][name] is True
                    for name in (
                        "persisted_matrix_reread_exact",
                        "persisted_mapping_reread_exact",
                        "persisted_whole_vector_replay_exact",
                        "persisted_endpoint_disjoint_bijection_exact",
                    )
                )
            ):
                raise common.ContractError(
                    "M1 derangement receipt failed closed"
                )
            del placebo
        if (
            len(m1_receipts) != int(overlay["m1"]["replicates"])
            or len(
                {row["rewire_seed_id"] for row in m1_receipts}
            )
            != int(overlay["m1"]["replicates"])
            or len({row["matrix_sha256"] for row in m1_receipts})
            != int(overlay["m1"]["replicates"])
            or len({row["mapping_sha256"] for row in m1_receipts})
            != int(overlay["m1"]["replicates"])
        ):
            raise common.ContractError("M1 replicate set drift")
    common.write_json(stage / "audit/m1_derangement_receipts.json", m1_receipts)
    common.write_json(
        stage / "model_mounts.json",
        model_mount_contract(split),
    )
    return _file_records(stage)


def write_split_release(
    policy: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    split: str,
    result: Mapping[str, Any],
) -> Path:
    release_root = common.repo_path(str(overlay["output_root"]))
    release_root.mkdir(parents=True, exist_ok=True)
    target = release_root / split
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite training-ready split: {target}"
        )
    stage = release_root / f".s-{split[0]}-{uuid.uuid4().hex[:10]}"
    if stage.exists():
        raise common.ContractError("Unexpected split staging collision")
    stage.mkdir(parents=False)
    published = False
    try:
        files = _write_payload(
            policy,
            overlay,
            split=split,
            stage=stage,
            result=result,
        )
        payload = result["payload"]
        manifest: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "status": "PASS_SPLIT_DATASET_READY",
            "run_id": overlay["run_id"],
            "split": split,
            "claim_level": overlay["target_release_claim_level"],
            "overlay_canonical_sha256": common.canonical_sha256(
                overlay
            ),
            "implementation_contract": copy.deepcopy(
                overlay["implementation_contract"]
            ),
            "implementation_contract_sha256": (
                implementation_contract_sha256(overlay)
            ),
            "scientific_contract": copy.deepcopy(
                overlay["scientific_contract"]
            ),
            "base_policy": copy.deepcopy(overlay["base_policy"]),
            "dataset_builder": copy.deepcopy(
                overlay["dataset_builder"]
            ),
            "structure_key_sha256_commitment": overlay[
                "private_structure_key_custody"
            ]["commitments"][split],
            "world_count": len(payload["worlds"]),
            "seller_count": len(payload["sellers"]),
            "item_count": len(payload["raw_items"]),
            "complete_pair_count": len(
                payload["complete_model_pair_endpoints"]
            ),
            "candidate_pair_count": len(payload["candidate_pairs"]),
            "positive_count": sum(
                int(row["label"])
                for row in payload["classification_labels"]
            ),
            "identity33_row_count": len(
                payload["identity33_all_pairs"]
            ),
            "retrieval_query_count": len(payload["retrieval_queries"]),
            "retrieval_relation_count": len(
                payload["retrieval_relations"]
            ),
            "metadata_shortcut_status": result["shortcut_report"][
                "status"
            ],
            "metadata_shortcut_max_symmetric_auc": (
                result["shortcut_report"][
                    "point_statistic_max_auc_symmetric"
                ]
                if split in {"train", "development"}
                else None
            ),
            "metadata_shortcut_bootstrap_95_upper": (
                result["shortcut_report"]["bootstrap_95_upper"]
                if split in {"train", "development"}
                else None
            ),
            "metadata_shortcut_values_withheld": split
            in {"audit_a", "audit_b"},
            "label_formula_exact": result["formula_audit"][
                "exact_rowwise_equal"
            ],
            "identity33_no_all_zero_columns": not result[
                "identity33_audit"
            ]["all_zero_columns"],
            "identity33_no_all_zero_columns_required": result[
                "identity33_audit"
            ]["no_all_zero_columns_required"],
            "aggregate_integrity": result["aggregate_audit"],
            "files": files,
            "elapsed_seconds": result["elapsed_seconds"],
        }
        manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
        common.write_json(stage / "split_manifest.json", manifest)
        _validate_split_tree(stage, expected_manifest=manifest)
        common.atomic_rename_no_replace(stage, target)
        published = True
        _validate_split_tree(target, expected_manifest=manifest)
    except Exception:
        failure_root = target if published else stage
        failure_name = (
            "INVALID_AFTER_ATOMIC_PUBLISH.json"
            if published
            else "FAILURE.json"
        )
        failure = failure_root / failure_name
        if failure_root.exists() and not failure.exists():
            common.write_json(
                failure,
                {
                    "status": "INVALID_SPLIT_STAGE_DO_NOT_USE",
                    "split": split,
                    "atomic_publish_completed": published,
                },
            )
        raise
    return target


def exercise_one_world(
    overlay: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    base = _load_pinned_base(overlay)
    design_key = DESIGN_ONLY_STRUCTURE_KEY_HEX
    policy = _execution_policy(
        base,
        overlay,
        structure_key_hex=design_key,
    )
    template, fixture, style_profile = legacy_generator._load_release_inputs(
        policy,
        mode=MODE,
    )
    record = next(
        row
        for row in structure.build_mode_world_pool(policy, mode=MODE)
        if row["split"] == split
    )
    built = _one_world(
        policy=policy,
        split=split,
        record=record,
        structure_key_hex=design_key,
        template=template,
        fixture=fixture,
        style_profile=style_profile,
    )
    public = built["world"]["public"]
    private = built["world"]["private"]
    labels = label_sealer.build_labels(
        candidate_rows=built["candidate_pairs"],
        membership_rows=private["controller_membership"],
        expected_world_count=1,
    )
    return {
        "status": "PASS_ONE_WORLD_FULL_CHAIN_EXERCISE",
        "formal_structure_key_created_or_read": False,
        "final_release_status_granted": False,
        "split": split,
        "world_uid": public["world"]["world_uid"],
        "seller_count": len(public["sellers"]),
        "item_count": len(public["items"]),
        "complete_pair_count": len(
            public["complete_model_pair_endpoints"]
        ),
        "candidate_pair_count": len(built["candidate_pairs"]),
        "positive_count": sum(int(row["label"]) for row in labels),
        "identity33_row_count": len(built["identity33"]),
        "candidate_summary": built["world_audit"]["candidate_summary"],
        "parser_exact": True,
        "redaction_exact": True,
        "independent_dgp_replay_exact": True,
        "same_producer_regeneration_exact": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--exercise-one-world", choices=SPLITS)
    parser.add_argument("--exercise-mini-split", choices=SPLITS)
    parser.add_argument("--mini-world-count", type=int, default=5)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = sum(
        (
            bool(args.validate_config_only),
            args.exercise_one_world is not None,
            args.exercise_mini_split is not None,
            args.split is not None,
        )
    )
    if actions != 1:
        raise common.ContractError(
            "Choose exactly one of --validate-config-only, "
            "--exercise-one-world, --exercise-mini-split or --split"
        )
    overlay_path = args.overlay.resolve()
    if args.validate_config_only:
        overlay = load_overlay(
            overlay_path,
            require_generation_frozen=False,
        )
        print(
            json.dumps(
                {
                    "status": "PASS_CONFIG_VALIDATION",
                    "generation_enabled": overlay["generation_enabled"],
                    "world_counts": overlay["world_counts"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.exercise_one_world is not None:
        overlay = load_overlay(
            overlay_path,
            require_generation_frozen=False,
        )
        print(
            json.dumps(
                exercise_one_world(
                    overlay,
                    split=args.exercise_one_world,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return
    if args.exercise_mini_split is not None:
        if args.mini_world_count < 5:
            raise common.ContractError(
                "Mini split requires at least five worlds"
            )
        overlay = load_overlay(
            overlay_path,
            require_generation_frozen=False,
        )
        mini_overlay = copy.deepcopy(overlay)
        split = str(args.exercise_mini_split)
        mini_overlay["world_counts"][split] = args.mini_world_count
        mini_overlay["shortcut_gate"]["bootstrap_replicates"] = 99
        mini_overlay["shortcut_gate"]["maximum_symmetric_auc"] = 1.0
        mini_overlay["shortcut_gate"][
            "maximum_world_bootstrap_95_upper"
        ] = 1.0
        base = _load_pinned_base(overlay)
        design_key = DESIGN_ONLY_STRUCTURE_KEY_HEX
        policy = _execution_policy(
            base,
            mini_overlay,
            structure_key_hex=design_key,
        )
        result = build_split_in_memory(
            policy,
            mini_overlay,
            split=split,
            structure_key_hex=design_key,
            progress_every=args.progress_every,
            mini_exercise_allow_incomplete_train_support=(
                split == "train"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "PASS_MINI_SPLIT_FULL_CHAIN_EXERCISE",
                    "formal_structure_key_created_or_read": False,
                    "final_release_status_granted": False,
                    "split": split,
                    "world_count": len(result["payload"]["worlds"]),
                    "aggregate_audit": result["aggregate_audit"],
                    "identity33_audit": {
                        "feature_count": result["identity33_audit"][
                            "feature_count"
                        ],
                        "all_zero_columns": result[
                            "identity33_audit"
                        ]["all_zero_columns"],
                        "rank": result["identity33_audit"][
                            "nonconstant_standardized_rank"
                        ],
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return

    split = str(args.split)
    overlay = load_overlay(
        overlay_path,
        require_generation_frozen=True,
    )
    structure_key_hex = _load_split_key(overlay, split=split)
    base = _load_pinned_base(overlay)
    policy = _execution_policy(
        base,
        overlay,
        structure_key_hex=structure_key_hex,
    )
    result = build_split_in_memory(
        policy,
        overlay,
        split=split,
        structure_key_hex=structure_key_hex,
        progress_every=args.progress_every,
    )
    target = write_split_release(
        policy,
        overlay,
        split=split,
        result=result,
    )
    print(f"Published training-ready split: {target}")


if __name__ == "__main__":
    main()
