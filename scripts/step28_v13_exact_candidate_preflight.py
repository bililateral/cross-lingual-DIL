#!/usr/bin/env python3
"""Preflight the exact training-ready split builder on a design-only key.

The preflight consumes the registered formal *public* random streams but never
creates or reads a split-private release key.  It calls the same in-memory
builder used for the final bytes, including independent DGP replay,
parser/redactor checks, mechanism-stratified C40 selection, label replay,
identity33 construction and the frozen 9,999-replicate shortcut audit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import step28_v13_build_training_ready_dataset as builder
import step28_v13_common as common
import step28_v13_metadata_shortcut_common as shortcut_common
import step28_v13_run_metadata_shortcut_audit as shortcut_audit


MODE = builder.MODE
SPLITS = builder.SPLITS
DEFAULT_OVERLAY = builder.DEFAULT_OVERLAY
CHECKPOINT_VERSION = (
    "2026-07-30-step28-v13-exact-preflight-checkpoint-v2"
)
CHECKPOINT_MANIFEST_VERSION = (
    "2026-07-30-step28-v13-exact-preflight-checkpoint-manifest-v1"
)
REPORT_VERSION = (
    "2026-07-31-step28-v13-training-ready-"
    "order-repair-exact-builder-preflight-v5"
)


def _checkpoint_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}.{suffix}"


def _checkpoint_targets(prefix: Path) -> dict[str, Path]:
    return {
        "started": _checkpoint_path(prefix, "started.json"),
        "shortcut_arrays": _checkpoint_path(
            prefix,
            "shortcut_inputs.npz",
        ),
        "shortcut": _checkpoint_path(prefix, "shortcut.json"),
        "identity33": _checkpoint_path(prefix, "identity33.json"),
        "retrieval": _checkpoint_path(prefix, "retrieval.json"),
        "aggregate": _checkpoint_path(prefix, "aggregate.json"),
    }


def _write_checkpoint_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    document = dict(payload)
    if "canonical_self_hash" in document:
        raise common.ContractError(
            "Checkpoint payload supplied its own self hash"
        )
    document["canonical_self_hash"] = common.canonical_sha256(document)
    common.write_json(path, document)


def _checkpoint_manifest(
    *,
    targets: dict[str, Path],
    split: str,
) -> dict[str, Any]:
    roles = [
        "started",
        "shortcut_arrays",
        "shortcut",
        "identity33",
        "aggregate",
    ]
    if split in {"audit_a", "audit_b"}:
        roles.append("retrieval")
    elif targets["retrieval"].exists():
        raise common.ContractError(
            "Non-audit exact preflight emitted retrieval checkpoint"
        )
    artifacts = []
    for role in roles:
        path = targets[role]
        if not path.is_file() or path.is_symlink():
            raise common.ContractError(
                f"Exact-preflight checkpoint missing or unsafe: {role}"
            )
        artifacts.append(
            {
                "role": role,
                "path": path.resolve()
                .relative_to(common.ROOT.resolve())
                .as_posix(),
                "sha256": common.sha256_file(path),
                "format": "npz" if role == "shortcut_arrays" else "json",
            }
        )
    return {
        "version": CHECKPOINT_MANIFEST_VERSION,
        "complete": True,
        "split": split,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _assert_fresh_checkpoint_targets(prefix: Path) -> dict[str, Path]:
    targets = _checkpoint_targets(prefix)
    collisions = [str(path) for path in targets.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite exact-preflight checkpoints: "
            + ", ".join(collisions)
        )
    return targets


def _write_npz_no_replace(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NumPy checkpoint atomically without overwrite."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite exact-preflight array checkpoint: {path}"
        )
    os.makedirs(common.filesystem_path(path.parent), exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".npz",
        dir=common.filesystem_path(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        common.atomic_rename_no_replace(Path(temporary_name), path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _canonical_sorted_rows_bytes(
    rows: list[dict[str, Any]],
    *,
    order_fields: tuple[str, ...],
) -> bytes:
    materialized = [dict(row) for row in rows]
    materialized.sort(
        key=lambda row: tuple(
            (
                str(row[field]).encode("utf-8")
                if not isinstance(row[field], (int, float, bool))
                else common.canonical_json_bytes(row[field])
            )
            for field in order_fields
        )
    )
    return common.canonical_json_bytes(materialized)


def _stage_event(stage: str, state: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "event": "exact_preflight_stage",
                "stage": stage,
                "state": state,
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _install_checkpoint_wrappers(
    *,
    targets: dict[str, Path],
    split: str,
    overlay_path: Path,
) -> Callable[[], None]:
    """Instrument expensive builder phases without changing their results."""

    original_shortcut = shortcut_audit.run_audit
    original_identity33 = builder._identity33_matrix_audit
    original_retrieval = builder._build_retrieval
    original_validate = builder._validate_payload

    def checkpoint_shortcut(**kwargs: Any) -> Any:
        stage = "metadata_shortcut_audit"
        _stage_event(stage, "started")
        started = time.perf_counter()
        if "evidence_sink" in kwargs:
            raise common.ContractError(
                "Exact preflight owns the shortcut evidence sink"
            )
        captured: dict[str, np.ndarray] = {}

        def capture_evidence(
            evidence: Any,
        ) -> None:
            nonlocal captured
            if captured or not isinstance(evidence, dict):
                raise common.ContractError(
                    "Shortcut evidence was emitted more than once or "
                    "with the wrong type"
                )
            captured = {
                str(name): np.asarray(values).copy()
                for name, values in evidence.items()
            }

        report, oof_rows = original_shortcut(
            **kwargs,
            evidence_sink=capture_evidence,
        )
        expected_evidence = {
            "folds",
            "score_logistic_l2",
            "score_gradient_tree",
            "score_rbf_svm",
            "bootstrap_statistics",
        }
        if set(captured) != expected_evidence:
            raise common.ContractError(
                "Shortcut evidence capture schema drift"
            )
        projection_rows = list(kwargs["projection_rows"])
        label_rows = list(kwargs["label_rows"])
        features = list(shortcut_common.PAIR_FEATURES)
        labels = {
            str(row["canonical_pair_uid"]): int(row["label"])
            for row in label_rows
        }
        matrix = np.asarray(
            [
                [float(row[name]) for name in features]
                for row in projection_rows
            ],
            dtype=np.float64,
        )
        pair_uids = np.asarray(
            [str(row["canonical_pair_uid"]) for row in projection_rows]
        )
        world_uids = np.asarray(
            [str(row["world_uid"]) for row in projection_rows]
        )
        label_vector = np.asarray(
            [labels[str(value)] for value in pair_uids],
            dtype=np.int8,
        )
        projection_payload = _canonical_sorted_rows_bytes(
            projection_rows,
            order_fields=("world_uid", "canonical_pair_uid"),
        )
        label_payload = _canonical_sorted_rows_bytes(
            label_rows,
            order_fields=("canonical_pair_uid",),
        )
        oof_payload = common.canonical_json_bytes(oof_rows)
        _write_npz_no_replace(
            targets["shortcut_arrays"],
            feature_names=np.asarray(features),
            x=matrix,
            y=label_vector,
            pair_uids=pair_uids,
            world_uids=world_uids,
            folds=captured["folds"],
            score_logistic_l2=captured["score_logistic_l2"],
            score_gradient_tree=captured["score_gradient_tree"],
            score_rbf_svm=captured["score_rbf_svm"],
            bootstrap_statistics=captured[
                "bootstrap_statistics"
            ],
            projection_rows_canonical_json_utf8=np.frombuffer(
                projection_payload,
                dtype=np.uint8,
            ).copy(),
            label_rows_canonical_json_utf8=np.frombuffer(
                label_payload,
                dtype=np.uint8,
            ).copy(),
            oof_rows_canonical_json_utf8=np.frombuffer(
                oof_payload,
                dtype=np.uint8,
            ).copy(),
        )
        elapsed = time.perf_counter() - started
        arrays = {
            "feature_names": np.asarray(features),
            "x": matrix,
            "y": label_vector,
            "pair_uids": pair_uids,
            "world_uids": world_uids,
            **captured,
            "projection_rows_canonical_json_utf8": np.frombuffer(
                projection_payload,
                dtype=np.uint8,
            ).copy(),
            "label_rows_canonical_json_utf8": np.frombuffer(
                label_payload,
                dtype=np.uint8,
            ).copy(),
            "oof_rows_canonical_json_utf8": np.frombuffer(
                oof_payload,
                dtype=np.uint8,
            ).copy(),
        }
        _write_checkpoint_json(
            targets["shortcut"],
            {
                "version": CHECKPOINT_VERSION,
                "stage": stage,
                "stage_completed": True,
                "split": split,
                "design_only": True,
                "formal_private_structure_key_created_or_read": False,
                "elapsed_seconds": elapsed,
                "projection_row_count": len(projection_rows),
                "label_row_count": len(label_rows),
                "oof_row_count": len(oof_rows),
                "oof_rows_sha256": common.canonical_sha256(
                    oof_rows
                ),
                "projection_rows_sha256": common.canonical_rows_sha256(
                    projection_rows,
                    order_fields=("world_uid", "canonical_pair_uid"),
                ),
                "label_rows_sha256": common.canonical_rows_sha256(
                    label_rows,
                    order_fields=("canonical_pair_uid",),
                ),
                "array_checkpoint": {
                    "path": targets["shortcut_arrays"]
                    .resolve()
                    .relative_to(common.ROOT.resolve())
                    .as_posix(),
                    "sha256": common.sha256_file(
                        targets["shortcut_arrays"]
                    ),
                },
                "array_schema": {
                    name: {
                        "shape": list(values.shape),
                        "dtype": values.dtype.str,
                    }
                    for name, values in sorted(arrays.items())
                },
                "metadata_shortcut_audit": report,
                "overlay_sha256": common.sha256_file(overlay_path),
            },
        )
        _stage_event(
            stage,
            "completed",
            elapsed_seconds=round(elapsed, 3),
            status=report["status"],
        )
        return report, oof_rows

    def checkpoint_identity33(
        policy: Any,
        rows: Any,
        *,
        require_no_zero_columns: bool,
    ) -> Any:
        stage = "identity33_matrix_audit"
        _stage_event(stage, "started")
        started = time.perf_counter()
        report = original_identity33(
            policy,
            rows,
            require_no_zero_columns=require_no_zero_columns,
        )
        elapsed = time.perf_counter() - started
        _write_checkpoint_json(
            targets["identity33"],
            {
                "version": CHECKPOINT_VERSION,
                "stage": stage,
                "stage_completed": True,
                "split": split,
                "elapsed_seconds": elapsed,
                "audit": report,
            },
        )
        _stage_event(
            stage,
            "completed",
            elapsed_seconds=round(elapsed, 3),
        )
        return report

    def checkpoint_retrieval(
        policy: Any,
        *,
        sellers: Any,
        memberships: Any,
        queries_per_world: int,
    ) -> Any:
        stage = "retrieval_build"
        _stage_event(stage, "started")
        started = time.perf_counter()
        result = original_retrieval(
            policy,
            sellers=sellers,
            memberships=memberships,
            queries_per_world=queries_per_world,
        )
        elapsed = time.perf_counter() - started
        queries, relations, qrels = result
        _write_checkpoint_json(
            targets["retrieval"],
            {
                "version": CHECKPOINT_VERSION,
                "stage": stage,
                "stage_completed": True,
                "split": split,
                "elapsed_seconds": elapsed,
                "query_count": len(queries),
                "relation_count": len(relations),
                "qrel_count": len(qrels),
            },
        )
        _stage_event(
            stage,
            "completed",
            elapsed_seconds=round(elapsed, 3),
        )
        return result

    def checkpoint_validate(
        policy: Any,
        overlay: Any,
        *,
        split: str,
        payload: Any,
    ) -> Any:
        stage = "aggregate_payload_validation"
        _stage_event(stage, "started")
        started = time.perf_counter()
        report = original_validate(
            policy,
            overlay,
            split=split,
            payload=payload,
        )
        elapsed = time.perf_counter() - started
        _write_checkpoint_json(
            targets["aggregate"],
            {
                "version": CHECKPOINT_VERSION,
                "stage": stage,
                "stage_completed": True,
                "split": split,
                "elapsed_seconds": elapsed,
                "audit": report,
            },
        )
        _stage_event(
            stage,
            "completed",
            elapsed_seconds=round(elapsed, 3),
        )
        return report

    shortcut_audit.run_audit = checkpoint_shortcut
    builder._identity33_matrix_audit = checkpoint_identity33
    builder._build_retrieval = checkpoint_retrieval
    builder._validate_payload = checkpoint_validate

    def restore() -> None:
        shortcut_audit.run_audit = original_shortcut
        builder._identity33_matrix_audit = original_identity33
        builder._build_retrieval = original_retrieval
        builder._validate_payload = original_validate

    return restore


def _shortcut_diagnostics(
    payload: dict[str, Any],
) -> dict[str, Any]:
    projections = payload["null_nuisance_projection"]
    labels = {
        str(row["canonical_pair_uid"]): int(row["label"])
        for row in payload["classification_labels"]
    }
    features = list(shortcut_common.PAIR_FEATURES)
    x = np.asarray(
        [
            [float(row[name]) for name in features]
            for row in projections
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [labels[str(row["canonical_pair_uid"])] for row in projections],
        dtype=np.int64,
    )
    world_uids = [str(row["world_uid"]) for row in projections]
    fold_by_world = shortcut_audit.assign_world_folds(
        world_uids,
        seed=2026072707,
        fold_count=5,
    )
    folds = np.asarray(
        [fold_by_world[world_uid] for world_uid in world_uids],
        dtype=np.int64,
    )

    def tree_symmetric_auc(kept_indices: list[int]) -> float:
        scores = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            test = folds == fold
            train = ~test
            scores[test] = shortcut_audit._fit_and_score_fold(
                "gradient_tree",
                x_train=x[train][:, kept_indices],
                y_train=y[train],
                x_test=x[test][:, kept_indices],
            )
        _auc, symmetric = shortcut_audit.symmetric_auc(y, scores)
        return symmetric

    univariate: dict[str, Any] = {}
    feature_ablation: dict[str, Any] = {}
    all_indices = list(range(len(features)))
    baseline_tree = tree_symmetric_auc(all_indices)
    for index, name in enumerate(features):
        values = x[:, index]
        if len(np.unique(values)) < 2:
            raw_auc = 0.5
            symmetric_auc = 0.5
        else:
            raw_auc, symmetric_auc = shortcut_audit.symmetric_auc(
                y,
                values,
            )
        no_feature = tree_symmetric_auc(
            [value for value in all_indices if value != index]
        )
        univariate[name] = {
            "roc_auc": raw_auc,
            "roc_auc_symmetric": symmetric_auc,
            "positive_mean": float(np.mean(values[y == 1])),
            "negative_mean": float(np.mean(values[y == 0])),
            "positive_standard_deviation": float(
                np.std(values[y == 1])
            ),
            "negative_standard_deviation": float(
                np.std(values[y == 0])
            ),
            "unique_value_count": int(len(np.unique(values))),
        }
        feature_ablation[name] = {
            "tree_auc_symmetric_without_feature": no_feature,
            "baseline_minus_without": baseline_tree - no_feature,
        }
    groups = {
        "item_count": [
            index
            for index, name in enumerate(features)
            if "item_count" in name
        ],
        "title_missing_rate": [
            index
            for index, name in enumerate(features)
            if "title_missing_rate" in name
        ],
        "description_missing_rate": [
            index
            for index, name in enumerate(features)
            if "description_missing_rate" in name
        ],
        "time_bucket_probabilities": [
            index
            for index, name in enumerate(features)
            if "time_bucket_probability" in name
        ],
    }
    group_ablation = {}
    for name, removed in groups.items():
        without = tree_symmetric_auc(
            [index for index in all_indices if index not in set(removed)]
        )
        group_ablation[name] = {
            "removed_features": [features[index] for index in removed],
            "tree_auc_symmetric_without_group": without,
            "baseline_minus_without": baseline_tree - without,
        }
    return {
        "purpose": (
            "diagnostic only; never used to choose or change a formal key"
        ),
        "tree_baseline_auc_symmetric_recomputed": baseline_tree,
        "univariate": univariate,
        "leave_one_feature_out_tree": feature_ablation,
        "leave_one_group_out_tree": group_ablation,
    }


def run(
    *,
    split: str,
    world_count: int,
    progress_every: int,
    bootstrap_replicates: int,
    overlay_path: Path,
    checkpoint_prefix: Path | None = None,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise common.ContractError("Invalid exact-preflight split")
    if type(progress_every) is not int or progress_every < 1:
        raise common.ContractError(
            "Exact-preflight progress interval must be positive"
        )
    overlay = builder.load_overlay(
        overlay_path,
        require_generation_frozen=False,
    )
    implementation_hash = builder.implementation_contract_sha256(
        overlay
    )
    builder_closure_hash = overlay["dataset_builder"][
        "implementation_closure"
    ]["canonical_sha256"]
    expected_world_count = int(overlay["world_counts"][split])
    expected_bootstrap = int(
        overlay["shortcut_gate"]["bootstrap_replicates"]
    )
    if (
        world_count != expected_world_count
        or bootstrap_replicates != expected_bootstrap
    ):
        raise common.ContractError(
            "Exact preflight must use the registered final split size and "
            "bootstrap count"
        )
    base = builder._load_pinned_base(overlay)
    policy = builder._execution_policy(
        base,
        overlay,
        structure_key_hex=builder.DESIGN_ONLY_STRUCTURE_KEY_HEX,
    )
    restore_wrappers: Callable[[], None] | None = None
    checkpoint_targets: dict[str, Path] | None = None
    if checkpoint_prefix is not None:
        checkpoint_prefix = checkpoint_prefix.resolve()
        try:
            checkpoint_prefix.relative_to(common.ROOT.resolve())
        except ValueError as error:
            raise common.ContractError(
                "Exact-preflight checkpoints must stay inside the repository"
            ) from error
        checkpoint_targets = _assert_fresh_checkpoint_targets(
            checkpoint_prefix
        )
        _write_checkpoint_json(
            checkpoint_targets["started"],
            {
                "version": CHECKPOINT_VERSION,
                "stage": "preflight_started",
                "split": split,
                "world_count": world_count,
                "bootstrap_replicates": bootstrap_replicates,
                "design_only": True,
                "formal_private_structure_key_created_or_read": False,
                "overlay_sha256": common.sha256_file(overlay_path),
                "implementation_contract_sha256": implementation_hash,
                "builder_source_closure_sha256": builder_closure_hash,
                "builder_implementation_sha256": common.sha256_file(
                    common.ROOT
                    / "scripts"
                    / "step28_v13_build_training_ready_dataset.py"
                ),
            },
        )
        restore_wrappers = _install_checkpoint_wrappers(
            targets=checkpoint_targets,
            split=split,
            overlay_path=overlay_path,
        )
    try:
        result = builder.build_split_in_memory(
            policy,
            overlay,
            split=split,
            structure_key_hex=builder.DESIGN_ONLY_STRUCTURE_KEY_HEX,
            progress_every=progress_every,
            allow_shortcut_failure_for_design_preflight=True,
        )
    finally:
        if restore_wrappers is not None:
            restore_wrappers()
    payload = result["payload"]
    report = result["shortcut_report"]
    positive_count = sum(
        int(row["label"]) for row in payload["classification_labels"]
    )
    world_audits = payload["world_generation_audit"]
    if (
        len(payload["worlds"]) != world_count
        or len(payload["candidate_pairs"]) != world_count * 40
        or positive_count
        != world_count
        * int(
            overlay["classification_positive_count_per_world"][split]
        )
        or len(world_audits) != world_count
        or not all(
            row["candidate_summary"][
                "all_positive_mechanisms_covered"
            ]
            and row["candidate_summary"]["all_negative_flags_covered"]
            and not row["candidate_summary"][
                "model_visible_sampling_fields"
            ]
            and row["independent_typed_dgp_replay_audit"][
                "full_typed_projection_exact"
            ]
            for row in world_audits
        )
        or result["formula_audit"]["exact_rowwise_equal"] is not True
        or result["candidate_output_order_audit"]
        != {
            "world_count": world_count,
            "candidate_pair_count": world_count * 40,
            "world_blocks_contiguous_and_exact": True,
            "independent_selected_global_rank_exact": True,
            "labels_or_controller_membership_read": False,
        }
        or result["aggregate_audit"][
            "all_keysets_and_foreign_keys_exact"
        ]
        is not True
    ):
        raise common.ContractError("Exact-preflight aggregate contract failed")
    output = {
        "version": REPORT_VERSION,
        "status": (
            "PASS_EXACT_IMPLEMENTATION_DESIGN_PREFLIGHT"
            if report["status"] == "PASS_METADATA_SHORTCUT_ONLY"
            else "FAIL_EXACT_IMPLEMENTATION_DESIGN_PREFLIGHT"
        ),
        "mode": MODE,
        "split": split,
        "world_count": world_count,
        "candidate_count": len(payload["candidate_pairs"]),
        "positive_count": positive_count,
        "negative_count": len(payload["candidate_pairs"])
        - positive_count,
        "formal_public_randomness_consumed": True,
        "formal_private_structure_key_created_or_read": False,
        "design_only_structure_key_used": True,
        "final_release_status_granted": False,
        "all_worlds_independent_dgp_replay_exact": True,
        "all_worlds_mechanism_coverage_exact": True,
        "label_formula_rowwise_exact": True,
        "candidate_output_order_audit": result[
            "candidate_output_order_audit"
        ],
        "aggregate_lineage_exact": True,
        "builder_implementation": (
            "scripts/step28_v13_build_training_ready_dataset.py"
        ),
        "builder_implementation_sha256": common.sha256_file(
            common.ROOT
            / "scripts"
            / "step28_v13_build_training_ready_dataset.py"
        ),
        "builder_source_closure_sha256": builder_closure_hash,
        "candidate_implementation": (
            "scripts/step28_v13_mechanism_stratified_c40.py"
        ),
        "candidate_implementation_sha256": common.sha256_file(
            common.ROOT
            / "scripts"
            / "step28_v13_mechanism_stratified_c40.py"
        ),
        "exact_preflight_implementation_sha256": common.sha256_file(
            common.ROOT
            / "scripts"
            / "step28_v13_exact_candidate_preflight.py"
        ),
        "implementation_contract_sha256": implementation_hash,
        "overlay_sha256": common.sha256_file(overlay_path),
        "base_policy_sha256": common.sha256_file(
            builder._verify_pin(
                overlay["base_policy"],
                label="base policy",
            )
        ),
        "metadata_shortcut_audit": report,
        "metadata_shortcut_failure_diagnostics": None,
        "failure_diagnostics_deferred": (
            report["status"] != "PASS_METADATA_SHORTCUT_ONLY"
        ),
        "identity33_matrix_audit": result["identity33_audit"],
        "elapsed_seconds": result["elapsed_seconds"],
        "checkpointing_enabled": checkpoint_targets is not None,
        "checkpoint_manifest": (
            _checkpoint_manifest(
                targets=checkpoint_targets,
                split=split,
            )
            if checkpoint_targets is not None
            else None
        ),
    }
    output["canonical_self_hash"] = common.canonical_sha256(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--world-count", type=int, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=9999)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-prefix",
        type=Path,
        help=(
            "Optional no-overwrite prefix for per-stage JSON and compressed "
            "shortcut-input checkpoints"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite exact preflight: {output}"
        )
    result = run(
        split=args.split,
        world_count=args.world_count,
        progress_every=args.progress_every,
        bootstrap_replicates=args.bootstrap_replicates,
        overlay_path=args.overlay.resolve(),
        checkpoint_prefix=(
            args.checkpoint_prefix.resolve()
            if args.checkpoint_prefix is not None
            else None
        ),
    )
    common.write_json(output, result)
    print(f"Wrote exact training-ready preflight: {output}", flush=True)


if __name__ == "__main__":
    main()
